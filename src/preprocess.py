"""Preprocessing: file selection, load, filter, window schedule, labelling.

Filters the whole recording before windowing (never per-window - see CLAUDE.md
non-negotiable rule 3), then slices into windows and labels them against the
subject's seizure summary file.

Channel selection reduces every file to the 21 unique required derivations
(REQUIRED_CHANNELS from audit.py), dropping the polarity-flipped P7-T7 and the
duplicate second T8-P8 occurrence at load time - before any feature extraction
or channel ranking, per CLAUDE.md section 6.3.

Two decisions here deviate from earlier versions of this file and from the
locked parameter table; both were put to the project owner and approved on
2026-08-14, and both are recorded in metadata.json (see D1/D2 below).
"""

import warnings
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from scipy.signal import butter, iirnotch, sosfiltfilt, filtfilt

from audit import REQUIRED_CHANNELS, GLASS7_DEPENDENCY, DATA_DIR, OUTPUT_DIR

mne.set_log_level("ERROR")

FS = 256
WINDOW_SEC = 4.0
WINDOW_SAMPLES = int(WINDOW_SEC * FS)   # 1024
LABEL_OVERLAP_FRACTION = 0.5            # window overlaps seizure by >=50% -> label 1

# D2 (approved 2026-08-14, deviates from the locked 0.5 s step in CLAUDE.md
# section 7). A uniform 4 s grid covers 100% of every recording with no gaps and
# is what all evaluation runs on, so recorded hours stay the denominator for
# false alarms per day (rule 5.2). The 0.5 s dense schedule is kept only inside
# a margin around each seizure, purely to give the training folds enough
# positive windows; those rows carry on_uniform_grid=False and are excluded from
# evaluation. At the locked 0.5 s step everywhere this study needs ~132 hours of
# compute and 62.6 GB, against ceilings of 12 h and 20 GB in CLAUDE.md section 11.
GRID_STEP_SEC = 4.0
DENSE_STEP_SEC = 0.5
DENSE_MARGIN_SEC = 60.0

GRID_STEP_SAMPLES = int(GRID_STEP_SEC * FS)      # 1024
DENSE_STEP_SAMPLES = int(DENSE_STEP_SEC * FS)    # 128
DENSE_MARGIN_SAMPLES = int(DENSE_MARGIN_SEC * FS)

BANDPASS = (0.5, 40.0)
BUTTER_ORDER = 4
NOTCH_FREQ = 60.0
NOTCH_Q = 30

_SOS_BP = butter(BUTTER_ORDER, BANDPASS, btype="band", fs=FS, output="sos")
_B_NOTCH, _A_NOTCH = iirnotch(w0=NOTCH_FREQ, Q=NOTCH_Q, fs=FS)

# chb21 is the same patient as chb01, recorded ~1.5 years later (Ali et al. 2024).
# Merging prevents patient leakage across LOSO folds. See CLAUDE.md section 6.1.
SUBJECT_MERGE_MAP = {"chb21": "chb01+21", "chb01": "chb01+21"}

# The 18 standard double-banana derivations: everything in REQUIRED_CHANNELS
# except the three Glass-7 dependencies, which only 640 of 668 files carry.
DOUBLE_BANANA_CHANNELS = [c for c in REQUIRED_CHANNELS if c not in GLASS7_DEPENDENCY]

AUDIT_CSV = OUTPUT_DIR / "00_channel_audit.csv"


def merged_subject_id(raw_subject):
    return SUBJECT_MERGE_MAP.get(raw_subject, raw_subject)


# ---------------------------------------------------------------------------
# File selection (D1)
# ---------------------------------------------------------------------------

def tier_a_files(audit_csv=AUDIT_CSV):
    """Every file admitted to the study, decided per file from the header audit.

    D1 (approved 2026-08-14). Tier A = all 18 double-banana derivations present
    and the file is not one of chb12's monopolar recordings (CLAUDE.md section
    6.2). That admits 668 files / 945.5 h / 181 seizures across all 23 subjects
    (after the chb01+21 merge) rather than the 470 files / 100 seizures / 16
    subjects that survive when the Glass-7 channel requirement is imposed on
    every configuration - an exclusion of 7 subjects, well past the 3-subject
    limit in CLAUDE.md section 11.

    Files missing T7-FT9 / FT9-FT10 / FT10-T8 are still processed; those three
    channels come out as NaN columns and Glass-7 is evaluated on the Tier B
    subset (has_glass7 below, 640 files / 177 seizures). Nothing is hardcoded
    here - the audit CSV is the single source of truth.

    Returns a DataFrame with subject, filename, duration_sec, n_seizures,
    has_glass7.
    """
    audit = pd.read_csv(audit_csv)
    has_double_banana = audit[[f"has_{c}" for c in DOUBLE_BANANA_CHANNELS]].all(axis=1)
    tier_a = audit[has_double_banana & ~audit["is_monopolar"]].copy()
    tier_a["has_glass7"] = tier_a[[f"has_{c}" for c in GLASS7_DEPENDENCY]].all(axis=1)
    tier_a["subject_id"] = tier_a["subject"].map(merged_subject_id)
    return tier_a[
        ["subject", "subject_id", "filename", "duration_sec", "n_seizures", "has_glass7"]
    ].sort_values(["subject", "filename"]).reset_index(drop=True)


def edf_path_for(raw_subject, filename):
    return DATA_DIR / raw_subject / filename


def summary_path_for(raw_subject):
    return DATA_DIR / raw_subject / f"{raw_subject}-summary.txt"


# ---------------------------------------------------------------------------
# Seizure annotations and labelling
# ---------------------------------------------------------------------------

def parse_seizure_intervals(summary_path, filename):
    """Return [(start_sec, end_sec), ...] for one file from a subject's summary.txt.

    Handles both 'Seizure Start Time' (single-seizure files) and
    'Seizure N Start Time' (multi-seizure files) formats found in CHB-MIT.
    """
    intervals = []
    current_file = None
    pending_start = None
    with open(summary_path, "r", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line.startswith("File Name"):
                current_file = line.split(":", 1)[-1].strip()
                pending_start = None
            elif current_file == filename and "Seizure" in line and "Start Time" in line:
                pending_start = int(line.split(":")[-1].split()[0])
            elif current_file == filename and "Seizure" in line and "End Time" in line:
                end = int(line.split(":")[-1].split()[0])
                if pending_start is not None:
                    intervals.append((pending_start, end))
                    pending_start = None
    return intervals


def label_window(window_start_sec, window_end_sec, seizure_intervals, filename):
    """Return (label, seizure_event_id) for one window."""
    window_dur = window_end_sec - window_start_sec
    for idx, (sz_start, sz_end) in enumerate(seizure_intervals):
        overlap = min(window_end_sec, sz_end) - max(window_start_sec, sz_start)
        if overlap >= LABEL_OVERLAP_FRACTION * window_dur:
            return 1, f"{filename}_sz{idx}"
    return 0, None


# ---------------------------------------------------------------------------
# Window schedule (D2)
# ---------------------------------------------------------------------------

def window_schedule(n_samples, seizure_intervals):
    """Window start offsets for one recording, in samples.

    Returns (start_samples, on_uniform_grid) sorted and deduplicated. Everything
    is computed in integer samples so that 0.5 s offsets stay exact and a dense
    window that happens to land on the grid is recognised as the same window
    rather than a near-duplicate.
    """
    last_start = n_samples - WINDOW_SAMPLES
    if last_start < 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=bool)

    grid = np.arange(0, last_start + 1, GRID_STEP_SAMPLES, dtype=np.int64)

    dense_blocks = []
    for start_sec, end_sec in seizure_intervals:
        low = max(0, int(start_sec * FS) - DENSE_MARGIN_SAMPLES)
        high = min(last_start, int(end_sec * FS) + DENSE_MARGIN_SAMPLES)
        if high < low:
            continue
        first = -(-low // DENSE_STEP_SAMPLES) * DENSE_STEP_SAMPLES  # ceil to step
        dense_blocks.append(np.arange(first, high + 1, DENSE_STEP_SAMPLES, dtype=np.int64))

    if dense_blocks:
        start_samples = np.unique(np.concatenate([grid] + dense_blocks))
    else:
        start_samples = grid

    return start_samples, start_samples % GRID_STEP_SAMPLES == 0


def window_metadata(raw_subject, filename, n_samples, seizure_intervals):
    """Per-window metadata rows for one file, aligned with window_schedule()."""
    start_samples, on_grid = window_schedule(n_samples, seizure_intervals)
    subject_id = merged_subject_id(raw_subject)

    start_secs = start_samples / FS
    labels = np.empty(len(start_samples), dtype=np.int8)
    event_ids = []
    for i, start_sec in enumerate(start_secs):
        label, event_id = label_window(
            start_sec, start_sec + WINDOW_SEC, seizure_intervals, filename
        )
        labels[i] = label
        event_ids.append(event_id)

    return start_samples, {
        "subject_id": subject_id,
        "filename": filename,
        "window_start_sec": start_secs,
        "on_uniform_grid": on_grid,
        "label": labels,
        "seizure_event_id": event_ids,
    }


# ---------------------------------------------------------------------------
# Signal loading and filtering
# ---------------------------------------------------------------------------

def resolve_channels(raw):
    """Map each of the 21 required derivations to its label in this file, or None.

    Handles mne's automatic '-0'/'-1' suffixing of repeated channel labels (the
    two 'T8-P8' channels): the FIRST occurrence is kept and the second dropped
    (CLAUDE.md section 6.3). Missing channels are not an error - files without
    the Glass-7 dependencies are still in the study and get NaN columns (D1).
    """
    resolved = {}
    for required in REQUIRED_CHANNELS:
        if required in raw.ch_names:
            resolved[required] = required
        elif f"{required}-0" in raw.ch_names:
            resolved[required] = f"{required}-0"
        else:
            resolved[required] = None
    return resolved


def open_recording(edf_path):
    """Open an EDF without reading signal data. Returns (raw, resolved_channels).

    preload=False is what keeps memory bounded: the caller pulls one channel at
    a time (see filtered_channel), so a 4-hour recording costs ~30 MB instead of
    the 621 MB float64 block that exhausted memory in the previous run.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = mne.io.read_raw_edf(edf_path, preload=False, verbose="ERROR")

    if raw.info["sfreq"] != FS:
        raise ValueError(f"{edf_path}: sampling rate {raw.info['sfreq']} != {FS}")

    missing = [c for c in DOUBLE_BANANA_CHANNELS if c not in raw.ch_names
               and f"{c}-0" not in raw.ch_names]
    if missing:
        raise ValueError(f"{edf_path}: missing double-banana channels {missing}")

    return raw, resolve_channels(raw)


def filtered_channel(raw, edf_channel_name):
    """One channel of the recording in uV, bandpass then notch, zero-phase.

    The whole recording is filtered before any windowing (rule 3): filtfilt's
    edge transient at 0.5 Hz spans several seconds and would destroy a 4 s
    window outright. sosfiltfilt for the bandpass because at 0.5 Hz / 256 Hz the
    normalized frequency is 0.0039, where the (b, a) form is numerically
    unstable and can silently return NaN (rule 4). The 60 Hz notch stays in
    (b, a) form: iirnotch only produces that form and 60 Hz is nowhere near the
    unstable region.
    """
    data = raw.get_data(picks=[edf_channel_name], units="uV")[0]
    data = sosfiltfilt(_SOS_BP, data)
    return filtfilt(_B_NOTCH, _A_NOTCH, data)


if __name__ == "__main__":
    files = tier_a_files()
    n_glass7 = int(files["has_glass7"].sum())
    print(f"Tier A files: {len(files)} "
          f"({files['duration_sec'].sum() / 3600:.1f} h, "
          f"{int(files['n_seizures'].sum())} seizures, "
          f"{files['subject_id'].nunique()} subjects after merge)")
    print(f"  of which Tier B (Glass-7 capable): {n_glass7} "
          f"({int(files.loc[files.has_glass7, 'n_seizures'].sum())} seizures)")

    row = files.iloc[0]
    intervals = parse_seizure_intervals(summary_path_for(row.subject), row.filename)
    starts, on_grid = window_schedule(int(row.duration_sec * FS), intervals)
    print(f"  {row.filename}: {len(starts)} windows "
          f"({int(on_grid.sum())} on the uniform grid, {int((~on_grid).sum())} dense)")
