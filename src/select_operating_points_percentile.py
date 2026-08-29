"""Percentile-calibrated variant of select_operating_points.py (option 1 from
the threshold-transfer discussion).

The absolute-threshold approach (select_operating_points.py) picks a probability
cutoff t* from the tuning-fold sweep and applies that same number to the test
fold. Spot-checking found this fails badly: 43/292 found-threshold cases (15%)
produce zero detections on test, and among the rest FA/day swings up to 25x
relative to what tuning predicted (output/06_operating_points.csv, documented
finding - see report). Root cause: RF's probability outputs aren't calibrated
consistently across different patients, so the same absolute number means
different things on different held-out subjects.

Fix tried here: reparameterize threshold selection in percentile space. Sweeping
over absolute thresholds and scoring on tuning (evaluate_sweep.py, already run)
is mathematically identical to sweeping over percentiles of tuning's own score
distribution - so step 1 (finding the operating point) is unchanged, just
reread from the existing sweep data rather than recomputed. The actual fix is
in how the operating point transfers to test: instead of reapplying tuning's
absolute threshold t* to test, we compute what FRACTION of tuning windows t*
flags (X* = P(tuning_prob >= t*)), then apply that same fraction to test's OWN
score distribution (threshold_test = the (1-X*) quantile of test_prob). This
self-calibrates per fold instead of assuming probability values transfer.

No new sweep needed - this reuses evaluate_sweep.py's output and only adds one
cheap quantile + one scoring pass per (fold, criterion, rule), same cost as
select_operating_points.py.
"""

import time

import numpy as np
import pandas as pd

from audit import OUTPUT_DIR
from evaluate import ground_truth_events, file_durations, score_fold
from model_configs import CONFIGS, LOO_CONFIGS, MODELS, SUFFICIENCY_CRITERIA
from postprocess import make_candidate_events, percentile_to_threshold
from evaluate_sweep import MERGE_GAP_DEFAULT_SEC, MIN_EVENT_DURATION_DEFAULT_SEC, load_cv_folds
from select_operating_points import pick_threshold, RULES

PRED_DIR = OUTPUT_DIR / "04_predictions"
SWEEP_DIR = OUTPUT_DIR / "05a_threshold_sweep"
OUT_PATH = OUTPUT_DIR / "06_operating_points_percentile.csv"


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
        tuning = pred_df[(pred_df.split == "tuning") & (pred_df.on_uniform_grid)]
        test = pred_df[(pred_df.split == "test") & (pred_df.on_uniform_grid)][
            ["subject_id", "filename", "window_start_sec", "prob"]
        ]
        tuning_probs = tuning["prob"].to_numpy()
        test_probs = test["prob"].to_numpy()
        test_recorded_hours = len(test) * 4.0 / 3600

        for criterion_name, crit in SUFFICIENCY_CRITERIA.items():
            for rule in RULES:
                # fallback_to_best_achievable=True: see pick_threshold's
                # docstring (select_operating_points.py) - if no tuning
                # threshold meets the FA ceiling, fall back to the best
                # achievable one instead of returning None, so a fold's
                # subjects still get scored on test rather than silently
                # vanishing from every downstream table. Bug found
                # 2026-08-17 via chb07/Full-18/realtime_alert.
                picked = pick_threshold(sweep_df, rule, crit["fa_per_day_max"],
                                         fallback_to_best_achievable=True)
                row = {
                    "config": config_name, "model": model_name, "scheme": scheme,
                    "fold": str(fold), "criterion": criterion_name, "rule": rule,
                    "test_recorded_hours": test_recorded_hours,
                }
                if picked is None:
                    # only reachable if the sweep itself is empty (no rows at
                    # all for this fold) - a real gap, not a ceiling miss.
                    row.update(t_star=None, percentile_flagged=None, threshold_test=None,
                               tuning_sens=None, tuning_fa=None, ceiling_met_on_tuning=None,
                               test_sens=None, test_fa=None, test_tp=None, test_fp=None,
                               test_ref_true=None, criterion_met_on_test=False)
                    rows.append(row)
                    continue

                t_star = float(picked["threshold"])
                tuning_sens = float(picked[f"sens_{rule}"])
                tuning_fa = float(picked[f"fa_per_day_{rule}"])
                ceiling_met_on_tuning = tuning_fa <= crit["fa_per_day_max"]
                percentile_flagged = float(np.mean(tuning_probs >= t_star))  # X*
                threshold_test = percentile_to_threshold(test_probs, percentile_flagged)

                events_by_file = make_candidate_events(
                    test, threshold_test, MERGE_GAP_DEFAULT_SEC, MIN_EVENT_DURATION_DEFAULT_SEC
                )
                totals = score_fold(events_by_file, ref_by_file, durations)
                t = totals[rule]
                test_sens = t["tp"] / t["ref_true"] if t["ref_true"] else float("nan")
                test_fa = t["fp"] / (test_recorded_hours / 24) if test_recorded_hours > 0 else float("nan")
                met_on_test = (test_sens >= crit["sensitivity_min"]) and (test_fa <= crit["fa_per_day_max"])

                row.update(t_star=t_star, percentile_flagged=percentile_flagged,
                           threshold_test=threshold_test,
                           tuning_sens=tuning_sens, tuning_fa=tuning_fa,
                           ceiling_met_on_tuning=ceiling_met_on_tuning,
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
