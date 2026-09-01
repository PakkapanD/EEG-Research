"""Post-processing sensitivity analysis, spec 2 section 6 (open item 4).

Scoped to a representative subset, not all 312 folds, per the spec's own
framing ("sensitivity analysis 2-3 parameters", not an exhaustive grid):
the 3 configs that span the channel ladder's extremes and have LOO coverage
(Full-18, Glass-7, Glass-2), one 5-fold fold and one LOO fold each, RF only
(the primary model).

Subset selection is deliberately picked by *distance to the pass/fail
boundary* on the locked default (merge_gap=8s, min_duration=1s), not by
subject identity or "hardness": a subject that clears or misses a criterion
by a wide margin can't have its verdict flipped by any merge_gap/
min_duration choice in this grid, so including only such subjects would make
the sweep uninformative regardless of whether it happened to read as 0% or
100%. An earlier version of this subset used loo="chb12" throughout (the
reference hard case from the threshold-calibration analysis) - that produced
a literal 0.0% pass rate at every one of the 21 combos, which turned out to
be real (verified against output/06a_per_subject_results_raw.csv, the
already-validated per-subject data) rather than a bug, but was an
uninformative floor effect: chb12 misses every criterion by so wide a margin
under all 3 configs that no post-processing choice could have flipped it.
fold 0 (5-fold) already ranked first among the 5 folds by mean
sens/FA-margin-to-boundary, so it's kept; loo=chb12 is swapped for
loo="chb02", which sits closest to the boundary when averaged across all 3
configs and both criteria (SzCORE, RF, LOO, from 06a_per_subject_results_raw.csv)
and shows a genuine pass/fail mix rather than a uniform miss.

Sweeps merge_gap_sec x min_event_duration_sec (7 x 3 = 21 combos) with
smoothing fixed at k<=1 (the only value the data allows - CLAUDE.md section
12) and scores using the LOCKED percentile-calibrated threshold selection
(same method as select_operating_points_percentile.py): pick the tuning-curve
point maximizing sensitivity subject to FA/day <= the criterion's ceiling,
convert to the percentile of tuning windows it flags, then apply that same
percentile to the test split's own score distribution before scoring
per-subject - consistent with the locked primary metric (event-level
per-subject pass rate, SzCORE).
"""

import time

import numpy as np
import pandas as pd

from audit import OUTPUT_DIR
from evaluate import ground_truth_events, file_durations, score_fold
from evaluate_sweep import sweep_threshold_curve
from model_configs import SUFFICIENCY_CRITERIA
from postprocess import make_candidate_events, percentile_to_threshold
from select_operating_points import pick_threshold, RULES

PRED_DIR = OUTPUT_DIR / "04_predictions"
OUT_PATH = OUTPUT_DIR / "05_postprocessing_sensitivity.md"
OUT_CSV_PATH = OUTPUT_DIR / "05b_postprocessing_sensitivity_raw.csv"

SUBSET = [
    ("Full-18", "5fold", 0), ("Full-18", "loo", "chb02"),
    ("Glass-7", "5fold", 0), ("Glass-7", "loo", "chb02"),
    ("Glass-2", "5fold", 0), ("Glass-2", "loo", "chb02"),
]
MODEL = "rf"

MERGE_GAP_GRID = [0, 4, 8, 12, 16, 20, 28]  # was 30; 28 = 7 grid steps, identical behaviour at 4s spacing. output/05b_...csv still carries the old "30" label (scores unchanged) unless re-run.
MIN_EVENT_DURATION_GRID = [0, 1, 2]
LOCKED_DEFAULT = (8.0, 1.0)  # what the main 06_results run used - see report


def score_test_per_subject(test_df, ref_by_file, durations, threshold, merge_gap, min_duration, rule, crit):
    rows = []
    for subj, g in test_df.groupby("subject_id"):
        events = make_candidate_events(g, threshold, merge_gap, min_duration)
        totals = score_fold(events, ref_by_file, durations)
        t = totals[rule]
        hours = len(g) * 4.0 / 3600
        sens = t["tp"] / t["ref_true"] if t["ref_true"] else float("nan")
        fa = t["fp"] / (hours / 24) if hours > 0 else float("nan")
        met = (sens >= crit["sensitivity_min"]) and (fa <= crit["fa_per_day_max"])
        rows.append({"subject": subj, "sens": sens, "fa": fa, "met": met})
    return rows


def main():
    ref_by_file = ground_truth_events()
    durations = file_durations()

    rows = []
    t0 = time.time()
    for config_name, scheme, fold in SUBSET:
        fold_label = f"5fold{fold}" if scheme == "5fold" else f"loo_{fold}"
        stem = f"{config_name}_{MODEL}_{fold_label}"
        pred = pd.read_parquet(PRED_DIR / f"{stem}.parquet")
        tuning = pred[(pred.split == "tuning") & (pred.on_uniform_grid)][
            ["subject_id", "filename", "window_start_sec", "prob"]
        ]
        test = pred[(pred.split == "test") & (pred.on_uniform_grid)][
            ["subject_id", "filename", "window_start_sec", "prob"]
        ]
        tuning_probs = tuning["prob"].to_numpy()
        test_probs = test["prob"].to_numpy()

        for merge_gap in MERGE_GAP_GRID:
            for min_duration in MIN_EVENT_DURATION_GRID:
                tuning_curve = sweep_threshold_curve(
                    tuning, ref_by_file, durations, merge_gap, min_duration
                )
                for criterion_name, crit in SUFFICIENCY_CRITERIA.items():
                    for rule in RULES:
                        # fallback_to_best_achievable=True: same fix as A12
                        # (select_operating_points_percentile.py, 2026-08-17).
                        # Without it, this call silently zeroed out entire
                        # (config, scheme, fold, merge_gap, min_duration,
                        # criterion, rule) combos whenever the tuning pool
                        # couldn't hit the FA ceiling on ANY grid threshold -
                        # found live here 2026-08-18: Glass-2/loo(chb02)/
                        # realtime_alert/szcore had n_subjects=0 across all 21
                        # combos, silently excluded from the pooled table
                        # instead of being scored and reported as met=False.
                        picked = pick_threshold(tuning_curve, rule, crit["fa_per_day_max"],
                                                 fallback_to_best_achievable=True)
                        if picked is None:
                            # only reachable if tuning_curve itself is empty -
                            # a real gap, not a ceiling miss.
                            rows.append(dict(
                                config=config_name, scheme=scheme, fold=str(fold),
                                merge_gap=merge_gap, min_duration=min_duration,
                                criterion=criterion_name, rule=rule,
                                n_subjects=0, n_met=0, pct_met=float("nan"),
                            ))
                            continue
                        t_star = float(picked["threshold"])
                        pct_flagged = float(np.mean(tuning_probs >= t_star))
                        threshold_test = percentile_to_threshold(test_probs, pct_flagged)

                        subj_rows = score_test_per_subject(
                            test, ref_by_file, durations, threshold_test,
                            merge_gap, min_duration, rule, crit,
                        )
                        n_met = sum(r["met"] for r in subj_rows)
                        rows.append(dict(
                            config=config_name, scheme=scheme, fold=str(fold),
                            merge_gap=merge_gap, min_duration=min_duration,
                            criterion=criterion_name, rule=rule,
                            n_subjects=len(subj_rows), n_met=n_met,
                            pct_met=100 * n_met / len(subj_rows) if subj_rows else float("nan"),
                        ))
        print(f"{config_name} {fold_label} done, {time.time() - t0:.1f}s elapsed")

    raw_df = pd.DataFrame(rows)
    tmp_csv = OUT_CSV_PATH.with_suffix(".csv.tmp")
    raw_df.to_csv(tmp_csv, index=False)
    tmp_csv.replace(OUT_CSV_PATH)
    print(f"Wrote {OUT_CSV_PATH} ({len(raw_df)} rows) in {time.time() - t0:.1f}s")

    write_report(raw_df)


def write_report(raw_df):
    lines = []
    lines.append("# Post-processing sensitivity analysis (spec 2 section 6)\n")
    lines.append(f"Representative subset: {[(c, s, f) for c, s, f in SUBSET]}, model={MODEL}. "
                 f"Smoothing fixed at k<=1 (the only value the shortest-seizure ceiling allows - "
                 f"CLAUDE.md section 12). Sweeps merge_gap x min_event_duration "
                 f"({len(MERGE_GAP_GRID)} x {len(MIN_EVENT_DURATION_GRID)} = "
                 f"{len(MERGE_GAP_GRID) * len(MIN_EVENT_DURATION_GRID)} combos), scored with the locked "
                 f"percentile-calibrated threshold method, event-level per-subject pass rate (primary metric, "
                 f"spec 2 open item raised 2026-08-16).\n")

    lines.append("## Per-subject pass rate, pooled across the subset (SzCORE), by (merge_gap, min_event_duration)\n")
    sub = raw_df[raw_df.rule == "szcore"]
    pooled = sub.groupby(["merge_gap", "min_duration"]).apply(
        lambda g: pd.Series({"n_subjects": g.n_subjects.sum(), "n_met": g.n_met.sum()})
    ).reset_index()
    pooled["pct_met"] = 100 * pooled.n_met / pooled.n_subjects
    lines.append("| merge_gap (s) | min_event_duration (s) | n_subject-criterion instances | n_met | pct_met |")
    lines.append("|---|---|---|---|---|")
    for _, r in pooled.sort_values(["merge_gap", "min_duration"]).iterrows():
        marker = " **<- locked default**" if (r.merge_gap, r.min_duration) == LOCKED_DEFAULT else ""
        lines.append(f"| {r.merge_gap:g} | {r.min_duration:g} | {int(r.n_subjects)} | {int(r.n_met)} | "
                     f"{r.pct_met:.1f}%{marker} |")
    lines.append("")

    best = pooled.loc[pooled.pct_met.idxmax()]
    default_row = pooled[(pooled.merge_gap == LOCKED_DEFAULT[0]) & (pooled.min_duration == LOCKED_DEFAULT[1])]
    default_pct = default_row.pct_met.iloc[0] if len(default_row) else float("nan")
    lines.append(f"Best combo in this subset: merge_gap={best.merge_gap:g}s, "
                 f"min_event_duration={best.min_duration:g}s -> {best.pct_met:.1f}% pass rate. "
                 f"Locked default (merge_gap={LOCKED_DEFAULT[0]:g}s, min_event_duration={LOCKED_DEFAULT[1]:g}s, "
                 f"used for the main 06_results run) -> {default_pct:.1f}%.\n")

    lines.append("## By config\n")
    for config_name in ["Full-18", "Glass-7", "Glass-2"]:
        lines.append(f"### {config_name}\n")
        csub = sub[sub.config == config_name]
        pooled_c = csub.groupby(["merge_gap", "min_duration"]).apply(
            lambda g: pd.Series({"n_subjects": g.n_subjects.sum(), "n_met": g.n_met.sum()})
        ).reset_index()
        pooled_c["pct_met"] = 100 * pooled_c.n_met / pooled_c.n_subjects
        pivot = pooled_c.pivot(index="min_duration", columns="merge_gap", values="pct_met")
        lines.append("min_event_duration (rows) x merge_gap (cols), pct pass rate:\n")
        header = "| min_dur \\ merge_gap | " + " | ".join(f"{c:g}s" for c in pivot.columns) + " |"
        sep = "|---" * (len(pivot.columns) + 1) + "|"
        lines.append(header)
        lines.append(sep)
        for idx, row in pivot.iterrows():
            cells = " | ".join(f"{v:.1f}%" if pd.notna(v) else "-" for v in row)
            lines.append(f"| {idx:g}s | {cells} |")
        lines.append("")

    REPORT_PATH_TEXT = "\n".join(lines)
    OUT_PATH.write_text(REPORT_PATH_TEXT, encoding="utf-8")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
