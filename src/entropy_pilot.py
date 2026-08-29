"""entropyProfiled_* chi-squared pilot (CLAUDE.md section 7's open item, spec 2
section 2 #3): the 9 entropyProfiled_* features were dropped from the main
pipeline on a time argument alone (~496ms/channel/window documented estimate,
~68 days for the full dataset) - never checked for whether they would have
been *useful*. This pilot answers that on a stratified sample, without
re-running the full 92-feature extraction.

Reuses the 89 features already sitting in output/features/*.parquet instead
of recomputing them - only the 9 entropyProfiled_* columns are missing, so
only those need extracting here, via sample_entropy_profile() /
entropy_profile_stats() from the frozen features_reference.py (the exact
DIHC-equivalent implementation, not a synthetic stand-in - CLAUDE.md section
3's "no synthetic data" applies to a diagnostic pilot too, not just the main
run).

TIMING, benchmarked on real data 2026-08-18 before running anything (CLAUDE.md
section 3):
  - Isolated, single process, numba using its default thread pool (20 logical
    cores on this machine): ~1.48s/channel/window steady-state, measured over
    18 repeated calls after JIT warm-up. About 3x the ~496ms/channel/window
    documented in features_reference.py/metadata.json - worth flagging as its
    own finding, not just silently using the corrected number.
  - FIRST parallelization attempt (rejected): process-level parallelism across
    files (ProcessPoolExecutor, 12 workers matching run_pipeline.py's
    established default) with each worker pinned to 1 numba thread, to avoid
    12 processes x 20 numba threads oversubscribing a 20-thread machine.
    Measured *worse* than no parallelism at all: 6 concurrent single-thread
    workers -> 4.68s/channel/window effective (a real end-to-end
    ProcessPoolExecutor run, not an estimate) - slower than running everything
    in one process. Root cause: entropy_profile_stats()'s
    _cum_hist_searchsorted is itself @njit(parallel=True) (numba prange) and
    gets a real ~5x speedup from its own multithreading (1.48s at 20 threads
    vs 7.36s pinned to 1 thread, measured directly) - pinning it to 1 thread
    to make room for OS-level process parallelism destroyed more throughput
    than the extra processes bought back.
  - CONCLUSION: don't fight numba's own parallelism. No ProcessPoolExecutor,
    no thread pinning - one process, sequential over files, each window's
    entropyProfiled call gets the full thread pool to itself. At the
    ~1.48s/channel/window number this is ~17.3h for the full 2000-window x
    21-channel pilot design (2000 x 21 x 1.48 / 3600) - reported to the
    project owner before running (CLAUDE.md section 3's 12h threshold), see
    the report this run produces / the conversation this was approved in.

File-reuse sampling: 2000 windows spread evenly across ~660 files would mean
opening/filtering ~660 files just to pull 1-3 windows out of each, when
filtering a whole file (rule 3) costs more than pulling extra windows from
one already open. Instead: pick a fixed, diverse pool of files (seizure files
across a spread of subjects, including the known hard cases - CLAUDE.md's
chb12/14/06/13/20 - and a few easy ones for contrast), draw BOTH label=1 and
label=0 windows from that same pool so no extra files are opened for the
negative class. random_state=42 throughout. Measured: 79 files for 2000
windows (~25 windows/file average).
"""

import time

import numpy as np
import pandas as pd

from audit import OUTPUT_DIR
from preprocess import (
    REQUIRED_CHANNELS, WINDOW_SAMPLES, FS,
    edf_path_for, open_recording, filtered_channel,
)
from features_reference import sample_entropy_profile, entropy_profile_stats

FEATURES_DIR = OUTPUT_DIR / "features"
OUT_DIR = OUTPUT_DIR / "00f_entropy_pilot_parts"
OUT_PATH = OUTPUT_DIR / "00f_entropy_pilot_features.parquet"

RNG_SEED = 42
N_PER_CLASS = 1000  # 2000 total, matches the pilot design in Consolidated-Context.md 5.6

# Hard cases (recurring across every analysis so far - Consolidated-Context.md
# B2) plus a spread of easy/average subjects, so the pilot isn't answering
# "does entropyProfiled help on one patient" by accident.
CANDIDATE_SUBJECTS = [
    "chb12", "chb14", "chb06", "chb13", "chb20",  # hard cases
    "chb01+21", "chb03", "chb05", "chb08", "chb09", "chb10", "chb19", "chb22",  # spread
]

ENTROPY_PROFILE_STATS = ("total", "average", "maximum", "minimum", "median",
                          "standardDeviation", "variance", "kurtosis", "skewness")


def pick_sample_files_and_windows(rng):
    """Return a DataFrame of (subject_id, filename, window_start_sec, label)
    for exactly N_PER_CLASS label=1 and N_PER_CLASS label=0 windows, drawn
    from as few distinct files as possible (see module docstring)."""
    label1_parts, label0_parts, touched_files = [], [], set()

    for subject_id in CANDIDATE_SUBJECTS:
        path = FEATURES_DIR / f"{subject_id}.parquet"
        if not path.exists():
            continue
        idx = pd.read_parquet(path, columns=["subject_id", "filename", "window_start_sec", "label"])
        seizure_files = idx.loc[idx.label == 1, "filename"].unique()
        for fn in seizure_files:
            touched_files.add((subject_id, fn))
        pool = idx[idx.filename.isin(seizure_files)]
        label1_parts.append(pool[pool.label == 1])
        label0_parts.append(pool[pool.label == 0])

    label1 = pd.concat(label1_parts, ignore_index=True)
    label0 = pd.concat(label0_parts, ignore_index=True)

    n1 = min(N_PER_CLASS, len(label1))
    n0 = min(N_PER_CLASS, len(label0))
    sample1 = label1.sample(n=n1, random_state=rng)
    sample0 = label0.sample(n=n0, random_state=rng)

    sample = pd.concat([sample1, sample0], ignore_index=True)
    return sample, sorted(touched_files)


def extract_one_file(subject_id, filename, raw_subject, window_starts_sec):
    """entropyProfiled for every sampled window in one file, all 21 channels.
    Sequential, single process - see module docstring for why parallelizing
    this made it slower, not faster."""
    path = edf_path_for(raw_subject, filename)
    raw, resolved = open_recording(path)
    sigs = {}
    for ch in REQUIRED_CHANNELS:
        edf_name = resolved.get(ch)
        sigs[ch] = filtered_channel(raw, edf_name) if edf_name else None

    rows = []
    for ws_sec in window_starts_sec:
        ws_sample = int(round(ws_sec * FS))
        row = {"subject_id": subject_id, "filename": filename, "window_start_sec": ws_sec}
        for ch in REQUIRED_CHANNELS:
            sig = sigs[ch]
            if sig is None or ws_sample + WINDOW_SAMPLES > len(sig):
                for stat in ENTROPY_PROFILE_STATS:
                    row[f"{ch}__entropyProfiled_{stat}_sampleEntropy"] = np.nan
                continue
            w = sig[ws_sample:ws_sample + WINDOW_SAMPLES]
            profile = sample_entropy_profile(w)
            feats = entropy_profile_stats(profile)
            for k, v in feats.items():
                row[f"{ch}__{k}"] = v
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(RNG_SEED)

    sample, touched_files = pick_sample_files_and_windows(rng)
    print(f"Sampled {len(sample)} windows ({(sample.label == 1).sum()} label=1, "
          f"{(sample.label == 0).sum()} label=0) from {len(touched_files)} files")

    t0 = time.time()
    already_done = sum(1 for sid, fn in touched_files if (OUT_DIR / f"{sid}__{fn}.parquet").exists())
    remaining = len(touched_files) - already_done
    completed = 0
    for subject_id, filename in touched_files:
        part_path = OUT_DIR / f"{subject_id}__{filename}.parquet"
        if part_path.exists():
            continue  # checkpoint: skip files already done on a resumed run
        raw_subject = filename.split("_")[0]
        starts = sample.loc[
            (sample.subject_id == subject_id) & (sample.filename == filename), "window_start_sec"
        ].tolist()
        if not starts:
            continue

        try:
            df = extract_one_file(subject_id, filename, raw_subject, starts)
        except Exception as e:
            print(f"FAILED {subject_id}/{filename}: {e} - skipping, rest of run continues")
            continue

        tmp = part_path.with_suffix(".parquet.tmp")
        df.to_parquet(tmp, index=False)
        tmp.replace(part_path)
        completed += 1
        elapsed = time.time() - t0
        eta = elapsed / completed * (remaining - completed) if completed else float("nan")
        print(f"{completed}/{remaining} files done ({len(starts)} windows), "
              f"{elapsed:.1f}s elapsed, ETA {eta:.1f}s")

    parts = [pd.read_parquet(OUT_DIR / f"{sid}__{fn}.parquet") for sid, fn in touched_files
             if (OUT_DIR / f"{sid}__{fn}.parquet").exists()]
    entropy_df = pd.concat(parts, ignore_index=True)
    tmp = OUT_PATH.with_suffix(".parquet.tmp")
    entropy_df.to_parquet(tmp, index=False)
    tmp.replace(OUT_PATH)
    print(f"Wrote {OUT_PATH} ({len(entropy_df)} rows, {entropy_df.shape[1]} cols) "
          f"in {time.time() - t0:.1f}s total")


if __name__ == "__main__":
    main()
