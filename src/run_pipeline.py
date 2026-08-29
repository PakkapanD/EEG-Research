"""Runner: preprocess + extract features for every admitted file, checkpointed
per file so a long run can be interrupted and resumed.

Each (raw_subject, filename) is processed independently (embarrassingly
parallel) into output/features/_parts/{raw_subject}__{filename}.parquet.
Run again to resume - files whose part already exists are skipped. After all
parts for a subject exist, consolidate() concatenates them into the final
output/features/{subject_id}.parquet (chb01 and chb21 merge into
chb01+21.parquet).

Memory, which killed the previous run
-------------------------------------
That version built a Python list of one 1869-key dict per window and held every
channel of the recording in float64 at once: ~4 GB for a 4-hour file, times 19
workers, on a 15.7 GB machine. Every worker died with MemoryError and the parent
then failed to spawn replacements.

This version allocates the output block up front - n_windows x 1869 float32, 216
MB for the longest file in the study - and fills it channel by channel, reading
and filtering one channel at a time (preprocess.filtered_channel). Peak per
worker is a few hundred MB, so the default worker count is 12: measured
aggregate throughput on this 6P+8E machine is 6.5x at 8 workers, 7.4x at 12 and
8.5x at 16, and 12 leaves headroom for the longest files.

Usage:
    python run_pipeline.py                      # process everything, then consolidate
    python run_pipeline.py --workers 8
    python run_pipeline.py --only-file chb01_03.edf --workers 1   # timing probe
    python run_pipeline.py --consolidate-only
"""

import argparse
import multiprocessing as mp
import os
import time
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from audit import REQUIRED_CHANNELS
from features import FEATURE_NAMES, N_FEATURES, channel_feature_matrix, warm_jit
from preprocess import (
    edf_path_for,
    filtered_channel,
    merged_subject_id,
    open_recording,
    parse_seizure_intervals,
    summary_path_for,
    tier_a_files,
    window_metadata,
    window_schedule,
    FS,
    WINDOW_SAMPLES,
)

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
FEATURES_DIR = OUTPUT_DIR / "features"
PARTS_DIR = FEATURES_DIR / "_parts"
LOG_PATH = OUTPUT_DIR / "_pipeline_log.txt"

FEATURE_COLS_DTYPE = "float32"
META_COLUMNS = [
    "subject_id", "filename", "window_start_sec",
    "on_uniform_grid", "label", "seizure_event_id",
]
FEATURE_COLUMNS = [f"{channel}__{feature}"
                   for channel in REQUIRED_CHANNELS
                   for feature in FEATURE_NAMES]

DEFAULT_WORKERS = 12


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def part_path(raw_subject, filename):
    return PARTS_DIR / f"{raw_subject}__{filename}.parquet"


def build_work_list():
    """One entry per admitted file, with its exact window count.

    The window count is what the ETA is based on: files in this study range from
    ~10 minutes to 4 hours, so counting files finished would make the estimate
    swing wildly. Summaries are parsed once per subject rather than once per file.
    """
    files = tier_a_files()
    intervals_by_subject = {}
    items = []
    for row in files.itertuples(index=False):
        if row.subject not in intervals_by_subject:
            summary = summary_path_for(row.subject)
            intervals_by_subject[row.subject] = {
                filename: parse_seizure_intervals(summary, filename)
                for filename in files.loc[files.subject == row.subject, "filename"]
            }
        intervals = intervals_by_subject[row.subject][row.filename]
        start_samples, _ = window_schedule(int(row.duration_sec * FS), intervals)
        items.append({
            "raw_subject": row.subject,
            "filename": row.filename,
            "n_windows": len(start_samples),
        })
    # Longest first: a 4-hour file started last would leave 11 workers idle for
    # hours at the tail of the run.
    items.sort(key=lambda item: -item["n_windows"])
    return items


def process_one_file(item):
    raw_subject = item["raw_subject"]
    filename = item["filename"]
    out_path = part_path(raw_subject, filename)
    if out_path.exists():
        return {"status": "skipped", **item}

    started = time.time()
    try:
        edf_path = edf_path_for(raw_subject, filename)
        intervals = parse_seizure_intervals(summary_path_for(raw_subject), filename)

        raw, resolved = open_recording(edf_path)
        start_samples, meta = window_metadata(
            raw_subject, filename, raw.n_times, intervals
        )

        values = np.empty((len(start_samples), len(FEATURE_COLUMNS)), dtype=FEATURE_COLS_DTYPE)
        n_missing_channels = 0
        for channel_index, channel in enumerate(REQUIRED_CHANNELS):
            block = slice(channel_index * N_FEATURES, (channel_index + 1) * N_FEATURES)
            if resolved[channel] is None:
                # Channel absent from this recording (only ever the Glass-7
                # dependencies). NaN, never 0 - CLAUDE.md section 9.
                values[:, block] = np.nan
                n_missing_channels += 1
                continue
            signal = filtered_channel(raw, resolved[channel])
            values[:, block] = channel_feature_matrix(signal, start_samples, WINDOW_SAMPLES)

        df = pd.concat(
            [pd.DataFrame(meta), pd.DataFrame(values, columns=FEATURE_COLUMNS)],
            axis=1,
        )

        tmp_path = out_path.with_suffix(".parquet.tmp")
        df.to_parquet(tmp_path, index=False)
        os.replace(tmp_path, out_path)

        return {
            "status": "ok",
            **item,
            "n_missing_channels": n_missing_channels,
            "n_nan_features": int(np.isnan(values).sum()),
            "elapsed_sec": time.time() - started,
        }
    except Exception as e:
        return {
            "status": "error",
            **item,
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
        }


def run(n_workers, only_file=None, limit=None):
    PARTS_DIR.mkdir(parents=True, exist_ok=True)
    work_items = build_work_list()
    if only_file:
        work_items = [w for w in work_items if w["filename"] == only_file]
        if not work_items:
            raise SystemExit(f"No admitted file named {only_file}")
    if limit:
        work_items = work_items[:limit]

    remaining = [w for w in work_items
                 if not part_path(w["raw_subject"], w["filename"]).exists()]
    total_windows = sum(w["n_windows"] for w in work_items)
    remaining_windows = sum(w["n_windows"] for w in remaining)
    log(f"Files: {len(work_items)} total, {len(remaining)} remaining "
        f"({remaining_windows:,} of {total_windows:,} windows). Workers: {n_workers}.")

    if not remaining:
        log("Nothing to do.")
        return

    log("Warming numba JIT in the parent so workers load the cache instead of racing to write it.")
    warm_jit()

    n_ok = 0
    n_err = 0
    windows_done = 0
    started = time.time()

    with mp.Pool(processes=n_workers) as pool:
        for result in pool.imap_unordered(process_one_file, remaining):
            windows_done += result["n_windows"]
            elapsed = time.time() - started
            rate = windows_done / elapsed if elapsed > 0 else 0
            eta_hours = (remaining_windows - windows_done) / rate / 3600 if rate > 0 else float("nan")
            progress = f"{windows_done:,}/{remaining_windows:,} windows, ETA {eta_hours:.1f} h"

            if result["status"] == "ok":
                n_ok += 1
                missing = (f", {result['n_missing_channels']} channels absent"
                           if result["n_missing_channels"] else "")
                log(f"OK {result['raw_subject']}/{result['filename']} "
                    f"({result['n_windows']:,} windows in {result['elapsed_sec'] / 60:.1f} min"
                    f"{missing}) | {progress}")
            elif result["status"] == "error":
                n_err += 1
                log(f"ERROR {result['raw_subject']}/{result['filename']}: {result['error']} | {progress}")
                log(result["traceback"])
            else:
                log(f"SKIPPED {result['raw_subject']}/{result['filename']} | {progress}")

    log(f"Run finished. ok={n_ok} err={n_err}. "
        f"Elapsed {(time.time() - started) / 3600:.2f} h")


def consolidate():
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    files = tier_a_files()

    by_subject = defaultdict(list)
    for row in files.itertuples(index=False):
        by_subject[merged_subject_id(row.subject)].append(
            part_path(row.subject, row.filename)
        )

    for subject_id, parts in sorted(by_subject.items()):
        missing = [p for p in parts if not p.exists()]
        if missing:
            log(f"consolidate: {subject_id} incomplete, "
                f"missing {len(missing)}/{len(parts)} parts. Skipping.")
            continue
        df = pd.concat([pd.read_parquet(p) for p in sorted(parts)], ignore_index=True)
        df = df.sort_values(["filename", "window_start_sec"]).reset_index(drop=True)
        out_path = FEATURES_DIR / f"{subject_id}.parquet"
        tmp_path = out_path.with_suffix(".parquet.tmp")
        df.to_parquet(tmp_path, index=False)
        os.replace(tmp_path, out_path)
        log(f"consolidate: wrote {out_path} ({len(df):,} rows, {len(parts)} files)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--only-file", default=None,
                        help="process a single EDF by name (timing probe)")
    parser.add_argument("--limit", type=int, default=None,
                        help="process only the first N files of the work list")
    parser.add_argument("--consolidate-only", action="store_true")
    args = parser.parse_args()

    if args.consolidate_only:
        consolidate()
    else:
        run(args.workers, only_file=args.only_file, limit=args.limit)
        consolidate()
