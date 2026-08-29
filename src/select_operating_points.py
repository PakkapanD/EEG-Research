"""Cheap step decoupled from the ~3h sweep (evaluate_sweep.py): pick, per
fold and per sufficiency criterion (CLAUDE.md section 14), the threshold on
the TUNING curve that maximizes sensitivity subject to FA/day <= the
criterion's ceiling, then apply that exact threshold to the held-out TEST
split and score for real. Writes one row per (config, model, scheme, fold,
criterion, rule) - the material 06_results_szcore.csv / 06_results_ali.csv
get built from.

Rule for "which threshold satisfies a criterion": among tuning-curve rows
with fa_per_day_<rule> <= criterion's fa_max, take the one with the highest
sens_<rule> (ties broken toward the lower threshold, i.e. list order, since
THRESHOLD_GRID is ascending and pandas idxmax keeps the first occurrence).
If no threshold on the tuning curve satisfies the FA ceiling at all, the
fold is marked unmet rather than silently picking the least-bad point.
"""

import json
import time
from pathlib import Path

import pandas as pd

from audit import OUTPUT_DIR
from evaluate import ground_truth_events, file_durations, score_fold
from model_configs import CONFIGS, LOO_CONFIGS, MODELS, SUFFICIENCY_CRITERIA
from postprocess import make_candidate_events
from evaluate_sweep import MERGE_GAP_DEFAULT_SEC, MIN_EVENT_DURATION_DEFAULT_SEC, load_cv_folds

PRED_DIR = OUTPUT_DIR / "04_predictions"
SWEEP_DIR = OUTPUT_DIR / "05a_threshold_sweep"
OUT_PATH = OUTPUT_DIR / "06_operating_points.csv"

RULES = ["szcore", "ali"]


def pick_threshold(sweep_df, rule, fa_max, fallback_to_best_achievable=False):
    """Among tuning-curve rows with FA/day <= fa_max, pick the one maximizing
    sensitivity. If none satisfy the ceiling, the default (False) returns
    None - preserved for select_operating_points.py's own absolute-threshold
    output (output/06_operating_points.csv), which the project owner asked
    to keep documented as-is (2026-08-16 decision) rather than change
    retroactively.

    select_operating_points_percentile.py passes fallback_to_best_achievable=
    True instead: when no tuning threshold meets the ceiling, fall back to
    whichever threshold gets FA closest to (but still above) the ceiling -
    ties broken toward higher sensitivity - rather than returning None. Bug
    found 2026-08-17 (project owner): returning None here caused
    build_final_results.py's per_subject_rows() to silently drop every
    subject in that fold from every downstream table (06a_per_subject_
    results_raw.csv, 06_results_*.csv, and the figure data built from them),
    even subjects who would have scored fine individually - e.g. chb07 under
    Full-18/realtime_alert/SzCORE, dropped because fold 2's TUNING pool
    (chb05/chb10/chb13/chb14/chb22) couldn't reach FA<=2/day at any
    threshold in the grid (best achievable: 2.057, at threshold=0.99), even
    though chb07 itself was never the problem. Falling back to the best-
    achievable tuning threshold keeps that subject in the data with an
    honestly-computed test-side sens/FA and met=False (or, rarely, met=True
    if the fallback threshold happens to transfer well) instead of erasing
    it.
    """
    candidates = sweep_df[sweep_df[f"fa_per_day_{rule}"] <= fa_max]
    if len(candidates) > 0:
        idx = candidates[f"sens_{rule}"].idxmax()
        return sweep_df.loc[idx]
    if not fallback_to_best_achievable or len(sweep_df) == 0:
        return None
    min_fa = sweep_df[f"fa_per_day_{rule}"].min()
    near_min = sweep_df[sweep_df[f"fa_per_day_{rule}"] == min_fa]
    idx = near_min[f"sens_{rule}"].idxmax()
    return sweep_df.loc[idx]


def main():
    cv = load_cv_folds()
    ref_by_file = ground_truth_events()
    durations = file_durations()

    stems = []
    for config_name in CONFIGS:
        for model_name in MODELS:
            for f in range(cv["five_fold"]["n_folds"]):
                stems.append((config_name, model_name, "5fold", f))
            if config_name in LOO_CONFIGS:
                for held_out in cv["loo"]["folds"]:
                    stems.append((config_name, model_name, "loo", held_out))

    t0 = time.time()
    rows = []
    for i, (config_name, model_name, scheme, fold) in enumerate(stems):
        fold_label = f"5fold{fold}" if scheme == "5fold" else f"loo_{fold}"
        stem = f"{config_name}_{model_name}_{fold_label}"
        sweep_path = SWEEP_DIR / f"{stem}.parquet"
        pred_path = PRED_DIR / f"{stem}.parquet"
        if not sweep_path.exists() or not pred_path.exists():
            continue

        sweep_df = pd.read_parquet(sweep_path)
        pred_df = pd.read_parquet(pred_path)
        test = pred_df[(pred_df.split == "test") & (pred_df.on_uniform_grid)][
            ["subject_id", "filename", "window_start_sec", "prob"]
        ]
        test_recorded_hours = len(test) * 4.0 / 3600

        for criterion_name, crit in SUFFICIENCY_CRITERIA.items():
            for rule in RULES:
                picked = pick_threshold(sweep_df, rule, crit["fa_per_day_max"])
                row = {
                    "config": config_name, "model": model_name, "scheme": scheme,
                    "fold": str(fold), "criterion": criterion_name, "rule": rule,
                    "test_recorded_hours": test_recorded_hours,
                }
                if picked is None:
                    row.update(threshold=None, tuning_sens=None, tuning_fa=None,
                               criterion_met_on_tuning=False,
                               test_sens=None, test_fa=None, test_tp=None, test_fp=None,
                               test_ref_true=None, criterion_met_on_test=False)
                    rows.append(row)
                    continue

                threshold = float(picked["threshold"])
                tuning_sens = float(picked[f"sens_{rule}"])
                tuning_fa = float(picked[f"fa_per_day_{rule}"])
                met_on_tuning = tuning_sens >= crit["sensitivity_min"]

                events_by_file = make_candidate_events(
                    test, threshold, MERGE_GAP_DEFAULT_SEC, MIN_EVENT_DURATION_DEFAULT_SEC
                )
                totals = score_fold(events_by_file, ref_by_file, durations)
                t = totals[rule]
                test_sens = t["tp"] / t["ref_true"] if t["ref_true"] else float("nan")
                test_fa = t["fp"] / (test_recorded_hours / 24) if test_recorded_hours > 0 else float("nan")
                met_on_test = (test_sens >= crit["sensitivity_min"]) and (test_fa <= crit["fa_per_day_max"])

                row.update(threshold=threshold, tuning_sens=tuning_sens, tuning_fa=tuning_fa,
                           criterion_met_on_tuning=met_on_tuning,
                           test_sens=test_sens, test_fa=test_fa,
                           test_tp=t["tp"], test_fp=t["fp"], test_ref_true=t["ref_true"],
                           criterion_met_on_test=met_on_test)
                rows.append(row)

        if (i + 1) % 50 == 0:
            print(f"{i + 1}/{len(stems)} folds, {time.time() - t0:.1f}s elapsed")

    out_df = pd.DataFrame(rows)
    tmp_path = OUT_PATH.with_suffix(".csv.tmp")
    out_df.to_csv(tmp_path, index=False)
    tmp_path.replace(OUT_PATH)
    print(f"Wrote {OUT_PATH} ({len(out_df)} rows) in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
