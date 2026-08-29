"""One-off diagnostic: scan every subject's existing feature parquet for
non-finite (+-inf) feature cells, the symptom of the zero-variance overflow
bug fixed in features.py (see module docstring there). This does NOT
re-extract anything - it only reports what the *already-extracted* (pre-fix)
data currently contains, so the project owner can decide which files need
re-extraction before deciding to re-run the full pipeline (CLAUDE.md section
3: report before running anything costly).

Run: python scan_nonfinite_features.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from audit import OUTPUT_DIR

FEATURES_DIR = OUTPUT_DIR / "features"
REPORT_PATH = OUTPUT_DIR / "00e_nonfinite_feature_scan.md"


def scan_subject(path):
    """Return a DataFrame of (filename, window_start_sec, column, value) for
    every non-finite feature cell in one subject's parquet, read in one pass
    with feature columns kept at float32 to bound memory."""
    schema = pq.read_schema(path)
    all_cols = schema.names
    meta_cols = ["subject_id", "filename", "window_start_sec", "on_uniform_grid", "label"]
    feat_cols = [c for c in all_cols if c not in meta_cols and c != "seizure_event_id"]

    table = pq.read_table(path, columns=meta_cols + feat_cols)
    df = table.to_pandas()
    feat_arr = df[feat_cols].to_numpy(dtype=np.float64, copy=False)
    bad_mask = np.isinf(feat_arr)
    n_bad = int(bad_mask.sum())
    if n_bad == 0:
        return n_bad, feat_arr.size, []

    rows, cols = np.nonzero(bad_mask)
    details = []
    for r, c in zip(rows, cols):
        details.append({
            "filename": df.iloc[r]["filename"],
            "window_start_sec": float(df.iloc[r]["window_start_sec"]),
            "column": feat_cols[c],
        })
    return n_bad, feat_arr.size, details


def main():
    subject_files = sorted(FEATURES_DIR.glob("chb*.parquet"))
    results = {}
    total_bad = 0
    total_cells = 0

    for path in subject_files:
        subject_id = path.stem
        n_bad, n_cells, details = scan_subject(path)
        results[subject_id] = (n_bad, n_cells, details)
        total_bad += n_bad
        total_cells += n_cells
        flag = f"{n_bad} non-finite cells" if n_bad else "clean"
        print(f"{subject_id}: {flag} (of {n_cells:,} feature cells)")

    lines = [
        "# Non-finite (+-inf) feature cell scan (pre-fix data)",
        "",
        "Scans the existing `output/features/*.parquet` (extracted before the",
        "zero-variance overflow fix in `features.py`) for +-inf cells - the",
        "symptom of dividing by a variance that underflows to exactly 0.0 after",
        "squaring on a near-zero-variance (flat/saturated) channel segment.",
        "This is a report only; no files are re-extracted here.",
        "",
        f"Total non-finite cells: **{total_bad}** of {total_cells:,} feature cells scanned "
        f"({100 * total_bad / total_cells:.6f}%)",
        "",
        "| subject | non-finite cells | affected windows | affected files/channels |",
        "|---|---|---|---|",
    ]
    for subject_id, (n_bad, n_cells, details) in results.items():
        if n_bad == 0:
            lines.append(f"| {subject_id} | 0 | - | - |")
            continue
        n_windows = len({(d["filename"], d["window_start_sec"]) for d in details})
        by_file_channel = {}
        for d in details:
            channel = d["column"].split("__")[0]
            key = (d["filename"], channel)
            by_file_channel[key] = by_file_channel.get(key, 0) + 1
        summary = "; ".join(f"{f}/{c} ({n})" for (f, c), n in sorted(by_file_channel.items()))
        lines.append(f"| {subject_id} | {n_bad} | {n_windows} | {summary} |")

    lines.append("")
    lines.append("## Detail (first 200 rows across all affected subjects)")
    lines.append("")
    lines.append("| subject | filename | window_start_sec | column |")
    lines.append("|---|---|---|---|")
    n_shown = 0
    for subject_id, (n_bad, n_cells, details) in results.items():
        for d in details:
            if n_shown >= 200:
                break
            lines.append(f"| {subject_id} | {d['filename']} | {d['window_start_sec']:.1f} | {d['column']} |")
            n_shown += 1
        if n_shown >= 200:
            break

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {REPORT_PATH}")
    print(f"Total: {total_bad} non-finite cells across {len(subject_files)} subjects")


if __name__ == "__main__":
    main()
