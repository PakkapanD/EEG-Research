"""Final results tables, spec 2 sections 7-8: output/06_results_szcore.csv and
06_results_ali.csv.

Reuses the locked operating points (threshold_test, percentile-calibrated,
output/06_operating_points_percentile.csv) - no new threshold selection here.
For each (config, model, scheme, criterion), aggregates per-subject across
every fold of that scheme (all 23 subjects, once each) into:

  - macro sensitivity (mean of per-subject sensitivity) and micro sensitivity
    (pooled tp / pooled ref_true) - both required together (CLAUDE.md
    section 12: the distribution is bimodal, macro alone hides it).
  - micro FA/day (pooled false alarms / pooled recorded hours).
  - % of subjects meeting the criterion (event-level per-subject pass rate -
    the locked primary metric, spec 2 section 12 decision 2026-08-16).
  - window-level specificity, precision, F1 (explicitly labelled window-level
    - CLAUDE.md section 12: conflating window-level precision with event-level
    sensitivity without saying so is what made the old report's numbers
    uninterpretable).
  - baseline rows ("always seizure" / "never seizure") computed the same way,
    every table (spec 2 section 5).

Known gaps, deferred rather than guessed at: detection latency (needs
per-event ref<->hyp matching that timescoring's EventScoring doesn't expose
directly) and model size / inference time (models weren't persisted from
train.py, only their predictions - CLAUDE.md rule 7 - so this needs a small
separate refit-and-measure pass, not reused from here).
"""

import numpy as np
import pandas as pd

from audit import OUTPUT_DIR
from evaluate import ground_truth_events, file_durations, score_fold
from evaluate_sweep import MERGE_GAP_DEFAULT_SEC, MIN_EVENT_DURATION_DEFAULT_SEC
from model_configs import CONFIGS, LOO_CONFIGS, MODELS, SUFFICIENCY_CRITERIA
from postprocess import make_candidate_events

PRED_DIR = OUTPUT_DIR / "04_predictions"
OP_POINTS_PATH = OUTPUT_DIR / "06_operating_points_percentile.csv"
RULES = ["szcore", "ali"]


def window_confusion(sub_df, threshold):
    pred = (sub_df["prob"].to_numpy() >= threshold)
    label = sub_df["label"].to_numpy().astype(bool)
    tp = int(np.sum(pred & label))
    fp = int(np.sum(pred & ~label))
    tn = int(np.sum(~pred & ~label))
    fn = int(np.sum(~pred & label))
    return tp, fp, tn, fn


def per_subject_rows(op_points, ref_by_file, durations):
    """One row per (config, model, scheme, fold, criterion, rule, subject)
    with both event-level (SzCORE/Ali tp/fp/ref_true) and window-level
    (tp/fp/tn/fn at the same threshold_test) counts.
    """
    rows = []
    cache = {}  # (config, model, fold_label) -> test split dataframe

    for _, r in op_points.iterrows():
        if pd.isna(r.threshold_test):
            continue
        fold_label = f"5fold{int(r.fold)}" if r.scheme == "5fold" else f"loo_{r.fold}"
        key = (r.config, r.model, fold_label)
        if key not in cache:
            pred = pd.read_parquet(PRED_DIR / f"{r.config}_{r.model}_{fold_label}.parquet")
            cache[key] = pred[(pred.split == "test") & (pred.on_uniform_grid)]
        test = cache[key]

        crit = SUFFICIENCY_CRITERIA[r.criterion]
        for subj, g in test.groupby("subject_id"):
            events = make_candidate_events(
                g, r.threshold_test, MERGE_GAP_DEFAULT_SEC, MIN_EVENT_DURATION_DEFAULT_SEC
            )
            totals = score_fold(events, ref_by_file, durations)
            t = totals[r.rule]
            hours = len(g) * 4.0 / 3600
            sens = t["tp"] / t["ref_true"] if t["ref_true"] else float("nan")
            fa = t["fp"] / (hours / 24) if hours > 0 else float("nan")
            met = (sens >= crit["sensitivity_min"]) and (fa <= crit["fa_per_day_max"])

            w_tp, w_fp, w_tn, w_fn = window_confusion(g, r.threshold_test)

            rows.append(dict(
                config=r.config, model=r.model, scheme=r.scheme, fold=str(r.fold),
                criterion=r.criterion, rule=r.rule, subject=subj,
                hours=hours, event_tp=t["tp"], event_fp=t["fp"], event_ref_true=t["ref_true"],
                sens=sens, fa=fa, met=met,
                window_tp=w_tp, window_fp=w_fp, window_tn=w_tn, window_fn=w_fn,
            ))
    return pd.DataFrame(rows)


def aggregate(per_subject_df):
    out_rows = []
    group_cols = ["config", "model", "scheme", "criterion", "rule"]
    for keys, g in per_subject_df.groupby(group_cols):
        config, model, scheme, criterion, rule = keys
        n_subjects = g["subject"].nunique()
        macro_sens = g["sens"].mean()
        micro_sens = g["event_tp"].sum() / g["event_ref_true"].sum() if g["event_ref_true"].sum() else float("nan")
        micro_fa = g["event_fp"].sum() / (g["hours"].sum() / 24) if g["hours"].sum() > 0 else float("nan")
        pct_met = 100 * g["met"].sum() / n_subjects

        w_tp, w_fp, w_tn, w_fn = g["window_tp"].sum(), g["window_fp"].sum(), g["window_tn"].sum(), g["window_fn"].sum()
        window_precision = w_tp / (w_tp + w_fp) if (w_tp + w_fp) else float("nan")
        window_recall = w_tp / (w_tp + w_fn) if (w_tp + w_fn) else float("nan")
        window_f1 = (2 * window_precision * window_recall / (window_precision + window_recall)
                     if (window_precision + window_recall) else float("nan"))
        window_specificity = w_tn / (w_tn + w_fp) if (w_tn + w_fp) else float("nan")

        out_rows.append(dict(
            config=config, model=model, scheme=scheme, criterion=criterion, rule=rule,
            n_subjects=n_subjects, n_hours=round(g["hours"].sum(), 2),
            n_events_ref=int(g["event_ref_true"].sum()),
            macro_sensitivity=macro_sens, micro_sensitivity=micro_sens, micro_fa_per_day=micro_fa,
            pct_subjects_meeting_criterion=pct_met,
            window_precision=window_precision, window_recall=window_recall,
            window_f1=window_f1, window_specificity=window_specificity,
            window_tp=int(w_tp), window_fp=int(w_fp), window_tn=int(w_tn), window_fn=int(w_fn),
        ))
    return pd.DataFrame(out_rows)


def baseline_rows(ref_by_file, durations):
    """'Always seizure' and 'never seizure' baselines, per (config, scheme,
    criterion) - independent of model, since they don't use features. Uses
    the SAME per-subject test partitioning as the real results (Full-18's
    channel set / Tier A vs Glass-7's Tier B) so n_hours/n_events match.
    """
    rows = []
    for config_name, cfg in CONFIGS.items():
        # any model's predictions carry the same meta/test-subject structure
        # for a given config+scheme; reuse rf's files just to get subject/file/label.
        for scheme in (["5fold", "loo"] if config_name in LOO_CONFIGS else ["5fold"]):
            fold_labels = ([f"5fold{i}" for i in range(5)] if scheme == "5fold"
                            else None)
            if scheme == "loo":
                import json
                cv = json.loads((OUTPUT_DIR / "02_cv_folds.json").read_text(encoding="utf-8"))
                fold_labels = [f"loo_{h}" for h in cv["loo"]["folds"]]

            per_subject = []
            for fold_label in fold_labels:
                path = PRED_DIR / f"{config_name}_rf_{fold_label}.parquet"
                if not path.exists():
                    continue
                pred = pd.read_parquet(path, columns=["subject_id", "filename", "window_start_sec",
                                                        "on_uniform_grid", "label", "split", "prob"])
                test = pred[(pred.split == "test") & (pred.on_uniform_grid)]
                for subj, g in test.groupby("subject_id"):
                    hours = len(g) * 4.0 / 3600
                    label_pos = int((g.label == 1).sum())
                    label_neg = len(g) - label_pos
                    for strategy, threshold in [("always_seizure", -np.inf), ("never_seizure", np.inf)]:
                        events = make_candidate_events(
                            g, threshold, MERGE_GAP_DEFAULT_SEC, MIN_EVENT_DURATION_DEFAULT_SEC
                        )
                        if strategy == "always_seizure":
                            w_tp, w_fp, w_tn, w_fn = label_pos, label_neg, 0, 0
                        else:
                            w_tp, w_fp, w_tn, w_fn = 0, 0, label_neg, label_pos
                        for rule in RULES:
                            totals = score_fold(events, ref_by_file, durations)
                            t = totals[rule]
                            per_subject.append(dict(
                                config=config_name, scheme=scheme, strategy=strategy, rule=rule,
                                subject=subj, hours=hours, event_tp=t["tp"], event_fp=t["fp"],
                                event_ref_true=t["ref_true"],
                                window_tp=w_tp, window_fp=w_fp, window_tn=w_tn, window_fn=w_fn,
                            ))
            if not per_subject:
                continue
            psdf = pd.DataFrame(per_subject)
            for (strategy, rule), g in psdf.groupby(["strategy", "rule"]):
                for criterion_name, crit in SUFFICIENCY_CRITERIA.items():
                    n_subjects = g["subject"].nunique()
                    micro_sens = g["event_tp"].sum() / g["event_ref_true"].sum() if g["event_ref_true"].sum() else float("nan")
                    micro_fa = g["event_fp"].sum() / (g["hours"].sum() / 24) if g["hours"].sum() > 0 else float("nan")
                    per_subj_sens = g.groupby("subject").apply(
                        lambda x: x["event_tp"].sum() / x["event_ref_true"].sum() if x["event_ref_true"].sum() else float("nan")
                    )
                    macro_sens = per_subj_sens.mean()
                    per_subj_fa = g.groupby("subject").apply(
                        lambda x: x["event_fp"].sum() / (x["hours"].sum() / 24) if x["hours"].sum() > 0 else float("nan")
                    )
                    met = ((per_subj_sens >= crit["sensitivity_min"]) & (per_subj_fa <= crit["fa_per_day_max"])).sum()
                    w_tp, w_fp, w_tn, w_fn = g["window_tp"].sum(), g["window_fp"].sum(), g["window_tn"].sum(), g["window_fn"].sum()
                    precision = w_tp / (w_tp + w_fp) if (w_tp + w_fp) else float("nan")
                    recall = w_tp / (w_tp + w_fn) if (w_tp + w_fn) else float("nan")
                    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else float("nan")
                    specificity = w_tn / (w_tn + w_fp) if (w_tn + w_fp) else float("nan")
                    rows.append(dict(
                        config=config_name, model=f"baseline_{strategy}", scheme=scheme,
                        criterion=criterion_name, rule=rule,
                        n_subjects=n_subjects, n_hours=round(g["hours"].sum(), 2),
                        n_events_ref=int(g["event_ref_true"].sum()),
                        macro_sensitivity=macro_sens, micro_sensitivity=micro_sens,
                        micro_fa_per_day=micro_fa, pct_subjects_meeting_criterion=100 * met / n_subjects,
                        window_precision=precision, window_recall=recall, window_f1=f1,
                        window_specificity=specificity,
                        window_tp=int(w_tp), window_fp=int(w_fp), window_tn=int(w_tn), window_fn=int(w_fn),
                    ))
    return pd.DataFrame(rows)


def main():
    ref_by_file = ground_truth_events()
    durations = file_durations()
    # float_precision='round_trip': see rescore_per_subject.py - pandas' default
    # CSV float parser can lose the last ULP on reparse, which flips
    # `prob >= threshold_test` for a boundary window. Caught for real via
    # chb22/Full-18/realtime_alert.
    op_points = pd.read_csv(OP_POINTS_PATH, float_precision="round_trip")

    print("Computing per-subject window+event metrics...")
    per_subject_df = per_subject_rows(op_points, ref_by_file, durations)
    per_subject_df.to_csv(OUTPUT_DIR / "06a_per_subject_results_raw.csv", index=False)
    print(f"  {len(per_subject_df)} per-subject rows")

    print("Aggregating...")
    agg_df = aggregate(per_subject_df)

    print("Computing baselines...")
    base_df = baseline_rows(ref_by_file, durations)

    full_df = pd.concat([agg_df, base_df], axis=0, ignore_index=True)

    for rule in RULES:
        out_path = OUTPUT_DIR / f"06_results_{rule}.csv"
        rule_df = full_df[full_df.rule == rule].drop(columns=["rule"])
        tmp = out_path.with_suffix(".csv.tmp")
        rule_df.sort_values(["config", "criterion", "model"]).to_csv(tmp, index=False)
        tmp.replace(out_path)
        print(f"Wrote {out_path} ({len(rule_df)} rows)")


if __name__ == "__main__":
    main()
