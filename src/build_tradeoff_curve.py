"""Sens-vs-FA/day tradeoff curve data for spec 2 section 10 figure 3, one
curve per config, pooled across all 5 test folds (RF, scheme=5fold) so each
of the 23 subjects contributes its own held-out prediction exactly once -
NOT the tuning-split sweep in output/05a_threshold_sweep/ (that data exists
to pick operating-point thresholds per fold and pools tuning subjects, who
overlap across folds; a figure showing "what can this config actually do"
needs true held-out performance, pooled without overlap).

Uses its own coarser threshold grid (COARSE_THRESHOLD_GRID below), not
model_configs.THRESHOLD_GRID (279 points, locked for actual operating-point
selection in 06_results). Piloted the full 279-point grid on the pooled
test set for one config first: 615s/config, ~72min for all 7 - over
CLAUDE.md's 10-minute stop-and-report line. Project owner chose (2026-08-17)
a coarser grid instead, since this is a visualization-only curve (never used
to pick a threshold anywhere), not a change to any locked analysis
parameter. 45 points, log-dense in the low-probability region where the two
sufficiency criteria's operating points actually sit (same rationale as the
locked grid's own weighting, just fewer of them) - piloted at ~90s/config,
~10-12min for all 7.
"""

import time

import numpy as np
import pandas as pd

from audit import OUTPUT_DIR
from evaluate import ground_truth_events, file_durations
from evaluate_sweep import sweep_threshold_curve

FIG_DIR = OUTPUT_DIR / "07_figures"
PRED_DIR = OUTPUT_DIR / "04_predictions"

ALL_CONFIGS = ["Full-18", "Best-7", "Glass-7", "Best-4", "Glass-4", "Best-2", "Glass-2"]
LOCKED_MERGE_GAP_SEC = 8.0
LOCKED_MIN_EVENT_DURATION_SEC = 1.0

COARSE_THRESHOLD_GRID = sorted(set(round(t, 5) for t in
    list(np.geomspace(0.0005, 0.02, 10)) +
    list(np.linspace(0.02, 0.3, 10))[1:] +
    list(np.linspace(0.3, 0.8, 8))[1:] +
    # dense near 1.0 - output/06_operating_points_percentile.csv shows the
    # test thresholds that actually reach FA<=5/day sit at 0.90-1.00, not
    # 0.3-0.9 (a first version of this grid stopped at 0.9 and missed the
    # entire achievable low-FA region - the curve never dropped below ~9
    # FA/day, discovered by comparing against the operating-points table)
    list(1 - np.geomspace(0.001, 0.2, 18))
))


def pooled_test_predictions(config_name):
    parts = []
    for fold in range(5):
        path = PRED_DIR / f"{config_name}_rf_5fold{fold}.parquet"
        d = pd.read_parquet(path, columns=["subject_id", "filename", "window_start_sec",
                                            "prob", "split", "on_uniform_grid"])
        parts.append(d[(d.split == "test") & (d.on_uniform_grid)])
    return pd.concat(parts, ignore_index=True)


def main():
    ref_by_file = ground_truth_events()
    durations = file_durations()

    rows = []
    t0 = time.time()
    for config_name in ALL_CONFIGS:
        pooled = pooled_test_predictions(config_name)
        curve = sweep_threshold_curve(
            pooled, ref_by_file, durations,
            LOCKED_MERGE_GAP_SEC, LOCKED_MIN_EVENT_DURATION_SEC, COARSE_THRESHOLD_GRID,
        )
        curve["config"] = config_name
        curve["n_subjects"] = pooled.subject_id.nunique()
        rows.append(curve)
        print(f"{config_name} done, {time.time() - t0:.1f}s elapsed")

    out = pd.concat(rows, ignore_index=True)
    tmp_path = (FIG_DIR / "data_tradeoff_curve.csv.tmp")
    out.to_csv(tmp_path, index=False)
    tmp_path.replace(FIG_DIR / "data_tradeoff_curve.csv")
    print(f"Wrote {FIG_DIR / 'data_tradeoff_curve.csv'} ({len(out)} rows) in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
