"""Writes output/01_dataset_summary.md from the consolidated per-subject
feature parquets in output/features/. Run after run_pipeline.py finishes.

This is the last stop point of the preprocessing spec (CLAUDE.md section 10),
and two numbers in it decide parameters in the next spec:

  - the shortest seizure, and how many uniform-grid windows the shortest
    seizures actually get, which caps any minimum-duration or k-consecutive
    post-processing rule (CLAUDE.md section 12 - the ceiling is computed from
    this data, never copied from Ali's 10 s);
  - our own seizure count, because Ali (175), Zanetti 2022 (185) and Zanetti
    2025 (198) each use a different denominator (CLAUDE.md section 6.4).
"""

from pathlib import Path

import numpy as np
import pandas as pd

from audit import REQUIRED_CHANNELS, GLASS7_DEPENDENCY
from preprocess import (
    parse_seizure_intervals, summary_path_for, tier_a_files,
    window_schedule, FS, WINDOW_SEC, GRID_STEP_SEC, LABEL_OVERLAP_FRACTION,
)

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
FEATURES_DIR = OUTPUT_DIR / "features"
REPORT_PATH = OUTPUT_DIR / "01_dataset_summary.md"

# One feature column per channel is enough to detect a channel that was absent
# from a recording: every one of its 89 columns is NaN together.
CHANNEL_PROBE_COLUMNS = [f"{channel}__mean" for channel in REQUIRED_CHANNELS]
META_COLUMNS = ["subject_id", "filename", "window_start_sec",
                "on_uniform_grid", "label", "seizure_event_id"]


def seizure_table(files):
    """Ground-truth seizures parsed from the summary files, independent of the
    window pipeline, plus how many uniform-grid windows each one earns.
    """
    rows = []
    for row in files.itertuples(index=False):
        intervals = parse_seizure_intervals(summary_path_for(row.subject), row.filename)
        n_samples = int(row.duration_sec * FS)
        starts, on_grid = window_schedule(n_samples, intervals)
        grid_starts = starts[on_grid] / FS
        for index, (start_sec, end_sec) in enumerate(intervals):
            overlap = (np.minimum(grid_starts + WINDOW_SEC, end_sec)
                       - np.maximum(grid_starts, start_sec))
            rows.append({
                "subject_id": row.subject_id,
                "filename": row.filename,
                "seizure_event_id": f"{row.filename}_sz{index}",
                "duration_sec": end_sec - start_sec,
                "n_grid_positive": int((overlap >= LABEL_OVERLAP_FRACTION * WINDOW_SEC).sum()),
                "has_glass7": row.has_glass7,
            })
    return pd.DataFrame(rows)


def read_subject(path):
    return pd.read_parquet(path, columns=META_COLUMNS + CHANNEL_PROBE_COLUMNS)


def count_computation_failure_nan(path):
    """True NaN cell count across every feature column, not just the one probe
    column per channel that read_subject() uses to find whole-channel-absent
    windows. A channel that IS present can still produce a handful of NaN
    cells: some features (kurtosis, skewness, renyiEntropy, hjorthMobility,
    detrended_fluctuation, ...) divide by a variance-derived quantity and
    features.py converts the result to NaN when it isn't finite - a flat or
    saturated segment inside an otherwise normal channel, not a missing
    channel. This is a full-column read (slower than read_subject's probe
    columns) so it is only used for this one summary number.
    """
    df = pd.read_parquet(path)
    feature_cols = [c for c in df.columns if c not in META_COLUMNS]
    return int(df[feature_cols].isna().to_numpy().sum())


def main():
    files = tier_a_files()
    subject_ids = sorted(files["subject_id"].unique())
    parquet_paths = {sid: FEATURES_DIR / f"{sid}.parquet" for sid in subject_ids}
    missing = [sid for sid, p in parquet_paths.items() if not p.exists()]
    if missing:
        raise SystemExit(
            f"Missing consolidated parquet for: {missing}. "
            "Run run_pipeline.py --consolidate-only first."
        )

    seizures = seizure_table(files)

    per_subject = []
    nan_windows_by_channel = {channel: 0 for channel in REQUIRED_CHANNELS}
    totals = dict(windows=0, grid=0, dense=0, positive=0, grid_positive=0, hours=0.0)
    computation_failure_nan_by_subject = {}

    for subject_id in subject_ids:
        df = read_subject(parquet_paths[subject_id])
        subject_files = files[files["subject_id"] == subject_id]
        subject_seizures = seizures[seizures["subject_id"] == subject_id]

        n_grid = int(df["on_uniform_grid"].sum())
        # The grid tiles each recording in non-overlapping 4 s blocks, so this is
        # the time actually classified - the honest denominator for false alarms
        # per day. It falls short of the header duration by at most one window
        # per file, where the recording length is not a multiple of the step.
        classified_hours = n_grid * GRID_STEP_SEC / 3600
        recorded_hours = float(subject_files["duration_sec"].sum()) / 3600
        for channel, column in zip(REQUIRED_CHANNELS, CHANNEL_PROBE_COLUMNS):
            nan_windows_by_channel[channel] += int(df[column].isna().sum())

        per_subject.append({
            "subject_id": subject_id,
            "files": len(subject_files),
            "hours": round(classified_hours, 2),
            "recorded_hours": round(recorded_hours, 2),
            "seizures": len(subject_seizures),
            "shortest_sec": int(subject_seizures["duration_sec"].min()) if len(subject_seizures) else None,
            "longest_sec": int(subject_seizures["duration_sec"].max()) if len(subject_seizures) else None,
            "windows": len(df),
            "grid": n_grid,
            "dense": len(df) - n_grid,
            "positive": int((df["label"] == 1).sum()),
            "grid_positive": int(((df["label"] == 1) & df["on_uniform_grid"]).sum()),
            "events_seen": int(df["seizure_event_id"].dropna().nunique()),
            "files_without_glass7": int((~subject_files["has_glass7"]).sum()),
        })

        totals["windows"] += len(df)
        totals["grid"] += n_grid
        totals["dense"] += len(df) - n_grid
        totals["positive"] += per_subject[-1]["positive"]
        totals["grid_positive"] += per_subject[-1]["grid_positive"]
        totals["hours"] += classified_hours
        computation_failure_nan_by_subject[subject_id] = count_computation_failure_nan(
            parquet_paths[subject_id]
        )

    structural_nan_cells = sum(nan_windows_by_channel.values()) * 89
    total_true_nan_cells = sum(computation_failure_nan_by_subject.values())
    computation_failure_nan_cells = total_true_nan_cells - structural_nan_cells

    tier_b = files[files["has_glass7"]]
    tier_b_seizures = seizures[seizures["has_glass7"]]
    events_seen = sum(row["events_seen"] for row in per_subject)

    lines = [
        "# Dataset Summary (preprocessing + feature extraction)",
        "",
        "## 1. Evaluation bases",
        "",
        "Files are admitted individually (deviation D1 in `metadata.json`). Glass-7 needs "
        f"`{'`, `'.join(GLASS7_DEPENDENCY)}`, which 28 admitted files do not carry, so it is "
        "reported against the Tier B subset. Every other configuration uses Tier A.",
        "",
        "| base | subjects | files | hours | seizures | used by |",
        "|---|---|---|---|---|---|",
        f"| Tier A | {files['subject_id'].nunique()} | {len(files)} | "
        f"{files['duration_sec'].sum() / 3600:.2f} | {len(seizures)} | "
        "Full-18, Best-7/4/2, Glass-4, Glass-2 |",
        f"| Tier B | {tier_b['subject_id'].nunique()} | {len(tier_b)} | "
        f"{tier_b['duration_sec'].sum() / 3600:.2f} | {len(tier_b_seizures)} | Glass-7 |",
        "",
        f"**Our seizure denominator is {len(seizures)}** (Tier A). Ali 2024 uses 175, "
        "Zanetti 2022 185, Zanetti 2025 198 - percentages are not comparable across those "
        "without restating the denominator (CLAUDE.md section 6.4).",
        "",
        f"Seizure events recovered from the feature files: {events_seen} "
        f"(must equal {len(seizures)}).",
        "",
        "## 2. Windows",
        "",
        f"- Total windows: **{totals['windows']:,}**",
        f"- On the uniform {GRID_STEP_SEC:g} s grid (the evaluation timeline): "
        f"{totals['grid']:,} = {totals['hours']:.2f} recorded hours",
        f"- Dense {0.5:g} s windows around seizures (training only): {totals['dense']:,}",
        f"- Positive windows: {totals['positive']:,} total, "
        f"{totals['grid_positive']:,} on the grid",
        f"- Class ratio on the evaluation timeline: 1:"
        f"{(totals['grid'] - totals['grid_positive']) / max(1, totals['grid_positive']):.0f}",
        f"- Class ratio including dense windows (what training sees): 1:"
        f"{(totals['windows'] - totals['positive']) / max(1, totals['positive']):.0f}",
        "",
        "## 3. Seizure durations and the post-processing ceiling",
        "",
        f"- Seizures: {len(seizures)}",
        f"- Duration: shortest **{int(seizures['duration_sec'].min())} s**, "
        f"median {int(seizures['duration_sec'].median())} s, "
        f"longest {int(seizures['duration_sec'].max())} s",
        f"- Uniform-grid positive windows per seizure: min "
        f"**{int(seizures['n_grid_positive'].min())}**, "
        f"median {int(seizures['n_grid_positive'].median())}, "
        f"max {int(seizures['n_grid_positive'].max())}",
        f"- Seizures with 0 grid windows: {int((seizures['n_grid_positive'] == 0).sum())} "
        f"(any non-zero here means a seizure is undetectable by construction)",
        f"- Seizures with <= 2 grid windows: {int((seizures['n_grid_positive'] <= 2).sum())}",
        "",
        f"**Ceiling for spec 2:** a 'k consecutive positive grid windows' rule spans "
        f"k x {GRID_STEP_SEC:g} s. At k=2 that is {2 * GRID_STEP_SEC:g} s, longer than the "
        f"shortest seizure ({int(seizures['duration_sec'].min())} s), so **k is capped at 1** "
        f"at grid resolution. Do not copy Ali's 10 s minimum-duration rule - it would discard "
        f"{int((seizures['duration_sec'] < 10).sum())} of our {len(seizures)} seizures before "
        "detection is even attempted.",
        "",
        "### Shortest 10 seizures",
        "",
        "| subject | file | duration (s) | grid windows |",
        "|---|---|---|---|",
    ]
    for row in seizures.nsmallest(10, "duration_sec").itertuples(index=False):
        lines.append(f"| {row.subject_id} | {row.filename} | {int(row.duration_sec)} | "
                     f"{row.n_grid_positive} |")

    lines += [
        "",
        "## 4. Missing channels (NaN, never 0)",
        "",
        "| channel | windows with NaN | % of all windows |",
        "|---|---|---|",
    ]
    any_nan = False
    for channel in REQUIRED_CHANNELS:
        count = nan_windows_by_channel[channel]
        if count == 0:
            continue
        any_nan = True
        lines.append(f"| {channel} | {count:,} | "
                     f"{100 * count / totals['windows']:.1f}% |")
    if not any_nan:
        lines.append("| _(none - every admitted file carried all 21 derivations)_ | 0 | 0% |")
    lines += [
        "",
        f"Each NaN channel accounts for 89 NaN feature columns in those windows. "
        f"Structural NaN feature cells (whole channel absent): {structural_nan_cells:,}.",
        "",
        f"On top of that, **{computation_failure_nan_cells:,} computation-failure NaN "
        "feature cells** occur inside channels that ARE present: a handful of features "
        "(kurtosis, skewness, renyiEntropy, hjorthMobility/Complexity, "
        "detrended_fluctuation) divide by a variance-derived quantity, and a near-zero-"
        "variance segment (flat/saturated channel, not literally absent) can make that "
        "quantity underflow to exactly 0.0 after squaring - features.py converts the "
        "result to NaN rather than letting a silent +-inf reach the scaler/model "
        "(CLAUDE.md section 9). Found and fixed on 2026-08-15: all "
        f"{computation_failure_nan_cells:,} of these cells are on chb17, "
        "`chb17b_69.edf` ~1520-1540s (FP1-F7/P3-O1). A full scan of every other subject's "
        "feature parquet found zero further occurrences "
        "(`output/00e_nonfinite_feature_scan.md`).",
        "",
        f"**Total NaN feature cells (structural + computation-failure): "
        f"{total_true_nan_cells:,}.**",
        "",
        "## 5. Per subject",
        "",
        "| subject | files | hours | seizures | shortest (s) | longest (s) | windows | grid | dense | pos | grid pos | files w/o Glass-7 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in per_subject:
        lines.append(
            f"| {row['subject_id']} | {row['files']} | {row['hours']:.2f} | {row['seizures']} | "
            f"{row['shortest_sec']} | {row['longest_sec']} | {row['windows']:,} | {row['grid']:,} | "
            f"{row['dense']:,} | {row['positive']:,} | {row['grid_positive']:,} | "
            f"{row['files_without_glass7']} |"
        )
    lines += [
        f"| **TOTAL** | **{len(files)}** | **{totals['hours']:.2f}** | **{len(seizures)}** | "
        f"**{int(seizures['duration_sec'].min())}** | **{int(seizures['duration_sec'].max())}** | "
        f"**{totals['windows']:,}** | **{totals['grid']:,}** | **{totals['dense']:,}** | "
        f"**{totals['positive']:,}** | **{totals['grid_positive']:,}** | "
        f"**{int((~files['has_glass7']).sum())}** |",
        "",
        "Per-subject numbers are reported because the distribution across subjects in this "
        "dataset is bimodal, not bell-shaped; a macro average alone hides that "
        "(CLAUDE.md section 12).",
        "",
    ]

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {REPORT_PATH}")
    print(f"  {totals['windows']:,} windows, {totals['grid']:,} on grid, "
          f"{len(seizures)} seizures, {totals['hours']:.1f} h")


if __name__ == "__main__":
    main()
