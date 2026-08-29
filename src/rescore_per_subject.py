"""Re-aggregate 5-fold's already-computed operating points per SUBJECT instead
of per fold. No new threshold selection or sweep - reuses threshold_test from
06_operating_points_percentile.csv exactly as chosen, only changes how pass/
fail is aggregated: a 5-fold test group is 3-5 subjects scored together as one
pooled sens/FA today, which means one high-FA subject (a chb12-like case) can
sink the whole fold's pass/fail regardless of what the other subjects in that
group would have scored individually - a literal 0% floor with no channel-
count information in it. LOO's test set is already exactly one subject, so its
rows are carried through unchanged for a like-for-like comparison.
"""

import pandas as pd

from audit import OUTPUT_DIR
from evaluate import ground_truth_events, file_durations, score_fold
from evaluate_sweep import MERGE_GAP_DEFAULT_SEC, MIN_EVENT_DURATION_DEFAULT_SEC
from model_configs import SUFFICIENCY_CRITERIA
from postprocess import make_candidate_events

PRED_DIR = OUTPUT_DIR / "04_predictions"
IN_PATH = OUTPUT_DIR / "06_operating_points_percentile.csv"
OUT_PATH = OUTPUT_DIR / "06_operating_points_percentile_per_subject.csv"


def main():
    # float_precision='round_trip': pandas' default CSV float parser is not
    # bit-exact - writing 0.9966666666666667 and reading it back can silently
    # produce 0.9966666666666668 (one ULP off), which flips `prob >= threshold`
    # for whichever window sits exactly at that boundary. Caught for real via
    # chb22/Full-18/realtime_alert. round_trip trades a bit of read speed for
    # an exact reparse.
    pct_df = pd.read_csv(IN_PATH, float_precision="round_trip")
    ref_by_file = ground_truth_events()
    durations = file_durations()

    rows = []

    loo = pct_df[(pct_df.scheme == "loo") & pct_df.threshold_test.notna()]
    for _, r in loo.iterrows():
        rows.append(dict(config=r.config, model=r.model, scheme=r.scheme, fold=r.fold,
                          criterion=r.criterion, rule=r.rule, subject=r.fold,
                          sens=r.test_sens, fa=r.test_fa, met=bool(r.criterion_met_on_test)))

    fivefold = pct_df[(pct_df.scheme == "5fold") & pct_df.threshold_test.notna()]
    n_groups = fivefold.groupby(["config", "model", "fold"]).ngroups
    done = 0
    for (config, model, fold), group in fivefold.groupby(["config", "model", "fold"]):
        fold_label = f"5fold{int(fold)}"
        stem = f"{config}_{model}_{fold_label}"
        pred = pd.read_parquet(PRED_DIR / f"{stem}.parquet")
        test = pred[(pred.split == "test") & (pred.on_uniform_grid)][
            ["subject_id", "filename", "window_start_sec", "prob"]
        ]
        by_subject = {s: g for s, g in test.groupby("subject_id")}

        for _, r in group.iterrows():
            crit = SUFFICIENCY_CRITERIA[r.criterion]
            for subj, g in by_subject.items():
                events = make_candidate_events(
                    g, r.threshold_test, MERGE_GAP_DEFAULT_SEC, MIN_EVENT_DURATION_DEFAULT_SEC
                )
                totals = score_fold(events, ref_by_file, durations)
                t = totals[r.rule]
                hours = len(g) * 4.0 / 3600
                sens = t["tp"] / t["ref_true"] if t["ref_true"] else float("nan")
                fa = t["fp"] / (hours / 24) if hours > 0 else float("nan")
                met = (sens >= crit["sensitivity_min"]) and (fa <= crit["fa_per_day_max"])
                rows.append(dict(config=config, model=model, scheme="5fold", fold=str(fold),
                                  criterion=r.criterion, rule=r.rule, subject=subj,
                                  sens=sens, fa=fa, met=met))
        done += 1
        if done % 20 == 0:
            print(f"{done}/{n_groups} (config,model,fold) groups re-scored per subject")

    out_df = pd.DataFrame(rows)
    tmp_path = OUT_PATH.with_suffix(".csv.tmp")
    out_df.to_csv(tmp_path, index=False)
    tmp_path.replace(OUT_PATH)
    print(f"Wrote {OUT_PATH} ({len(out_df)} rows)")


if __name__ == "__main__":
    main()
