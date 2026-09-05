"""Quantify the Tier A -> Tier B evaluation-base bias that flatters Glass-7.

Glass-7 is scored on Tier B (640 files / 917.48 h / 177 seizures) because 28
files lack its T7-FT9 / FT9-FT10 / FT10-T8 dependencies (CLAUDE.md section 8,
`01_dataset_summary.md` section 1). Every other config is scored on Tier A
(668 files / 945.49 h / 181 seizures). Any head-to-head with Glass-7 (e.g.
Glass-7 vs Best-7 in the report) therefore compares metrics computed on
different denominators.

This isolates the size of that effect: take each Tier A config's locked
percentile-calibrated operating points (output/06_operating_points_percentile.csv,
identical to build_final_results.py) and re-score its held-out predictions twice
- once on the full Tier A test windows (reproduces 06_results_szcore.csv) and
once restricted to the Tier B file subset - then report the metric shift. Only
6 subjects carry any non-Glass-7 files (chb13/15/16/17/18/19), so only their
rows move.

Model: rf only (primary). Output: output/01b_tier_effect_glass7.md +
output/01b_tier_effect_glass7_raw.csv. No training, no threshold selection -
pure re-scoring of existing predictions.
"""

import numpy as np
import pandas as pd

from audit import OUTPUT_DIR
from evaluate import ground_truth_events, file_durations, score_fold
from evaluate_sweep import MERGE_GAP_DEFAULT_SEC, MIN_EVENT_DURATION_DEFAULT_SEC
from model_configs import CONFIGS, LOO_CONFIGS, SUFFICIENCY_CRITERIA
from postprocess import make_candidate_events
from preprocess import tier_a_files

PRED_DIR = OUTPUT_DIR / "04_predictions"
OP_POINTS_PATH = OUTPUT_DIR / "06_operating_points_percentile.csv"
OUT_MD = OUTPUT_DIR / "01b_tier_effect_glass7.md"
OUT_CSV = OUTPUT_DIR / "01b_tier_effect_glass7_raw.csv"

MODEL = "rf"
RULES = ["szcore", "ali"]
# Tier A configs only - Glass-7 is already Tier B, nothing to compare.
CONFIGS_TO_PROBE = [c for c in CONFIGS if c != "Glass-7"]


def tier_b_files_by_subject():
    """{subject_id: set(filenames that carry all 3 Glass-7 dependency channels)}."""
    files = tier_a_files()
    out = {}
    for row in files.itertuples(index=False):
        if row.has_glass7:
            out.setdefault(row.subject_id, set()).add(row.filename)
    return out


def per_subject_rows(op_points, ref_by_file, durations, tier_b_by_subject):
    """One row per (config, scheme, fold, criterion, rule, subject, base) where
    base is 'tier_a' (all test files) or 'tier_b' (Glass-7-capable files only).
    """
    rows = []
    cache = {}

    for _, r in op_points.iterrows():
        if r.model != MODEL or r.config not in CONFIGS_TO_PROBE:
            continue
        if pd.isna(r.threshold_test):
            continue
        fold_label = f"5fold{int(r.fold)}" if r.scheme == "5fold" else f"loo_{r.fold}"
        key = (r.config, fold_label)
        if key not in cache:
            pred = pd.read_parquet(PRED_DIR / f"{r.config}_{MODEL}_{fold_label}.parquet")
            cache[key] = pred[(pred.split == "test") & (pred.on_uniform_grid)]
        test = cache[key]
        crit = SUFFICIENCY_CRITERIA[r.criterion]

        for subj, g in test.groupby("subject_id"):
            tb_files = tier_b_by_subject.get(subj, set())
            g_tb = g[g["filename"].isin(tb_files)]
            for base, gg in [("tier_a", g), ("tier_b", g_tb)]:
                if len(gg) == 0:
                    continue
                events = make_candidate_events(
                    gg, r.threshold_test, MERGE_GAP_DEFAULT_SEC, MIN_EVENT_DURATION_DEFAULT_SEC
                )
                totals = score_fold(events, ref_by_file, durations)
                t = totals[r.rule]
                hours = len(gg) * 4.0 / 3600
                sens = t["tp"] / t["ref_true"] if t["ref_true"] else float("nan")
                fa = t["fp"] / (hours / 24) if hours > 0 else float("nan")
                met = (sens >= crit["sensitivity_min"]) and (fa <= crit["fa_per_day_max"])
                rows.append(dict(
                    config=r.config, scheme=r.scheme, fold=str(r.fold),
                    criterion=r.criterion, rule=r.rule, subject=subj, base=base,
                    hours=hours, event_tp=t["tp"], event_fp=t["fp"], event_ref_true=t["ref_true"],
                    sens=sens, fa=fa, met=met,
                ))
    return pd.DataFrame(rows)


def aggregate(per_subject_df):
    out = []
    gcols = ["config", "scheme", "criterion", "rule", "base"]
    for keys, g in per_subject_df.groupby(gcols):
        config, scheme, criterion, rule, base = keys
        n_subjects = g["subject"].nunique()
        ref_sum = g["event_ref_true"].sum()
        hours_sum = g["hours"].sum()
        out.append(dict(
            config=config, scheme=scheme, criterion=criterion, rule=rule, base=base,
            n_subjects=n_subjects, n_hours=round(hours_sum, 2),
            n_events_ref=int(ref_sum),
            macro_sensitivity=g["sens"].mean(),
            micro_sensitivity=g["event_tp"].sum() / ref_sum if ref_sum else float("nan"),
            micro_fa_per_day=g["event_fp"].sum() / (hours_sum / 24) if hours_sum > 0 else float("nan"),
            pct_subjects_meeting_criterion=100 * g["met"].sum() / n_subjects,
        ))
    return pd.DataFrame(out)


def deltas(agg_df):
    """tier_b minus tier_a for each (config, scheme, criterion, rule)."""
    wide = agg_df.pivot_table(
        index=["config", "scheme", "criterion", "rule"], columns="base",
        values=["micro_sensitivity", "micro_fa_per_day", "macro_sensitivity",
                "pct_subjects_meeting_criterion", "n_hours", "n_events_ref"],
    )
    rows = []
    for idx, r in wide.iterrows():
        config, scheme, criterion, rule = idx
        rows.append(dict(
            config=config, scheme=scheme, criterion=criterion, rule=rule,
            hours_tier_a=r[("n_hours", "tier_a")], hours_tier_b=r[("n_hours", "tier_b")],
            events_tier_a=r[("n_events_ref", "tier_a")], events_tier_b=r[("n_events_ref", "tier_b")],
            micro_sens_tier_a=r[("micro_sensitivity", "tier_a")],
            micro_sens_tier_b=r[("micro_sensitivity", "tier_b")],
            micro_sens_delta=r[("micro_sensitivity", "tier_b")] - r[("micro_sensitivity", "tier_a")],
            micro_fa_tier_a=r[("micro_fa_per_day", "tier_a")],
            micro_fa_tier_b=r[("micro_fa_per_day", "tier_b")],
            micro_fa_delta=r[("micro_fa_per_day", "tier_b")] - r[("micro_fa_per_day", "tier_a")],
            micro_fa_pct_change=100 * (r[("micro_fa_per_day", "tier_b")] - r[("micro_fa_per_day", "tier_a")])
            / r[("micro_fa_per_day", "tier_a")] if r[("micro_fa_per_day", "tier_a")] else float("nan"),
            pct_met_tier_a=r[("pct_subjects_meeting_criterion", "tier_a")],
            pct_met_tier_b=r[("pct_subjects_meeting_criterion", "tier_b")],
            pct_met_delta=r[("pct_subjects_meeting_criterion", "tier_b")]
            - r[("pct_subjects_meeting_criterion", "tier_a")],
        ))
    return pd.DataFrame(rows)


def write_report(agg_df, delta_df, per_subject_df):
    lines = []
    lines.append("# Tier A -> Tier B evaluation-base effect (the Glass-7 comparison caveat)\n")
    lines.append(
        "Glass-7 is scored on Tier B (Glass-7-capable files: 640 / 917.48 h / 177 seizures); "
        "every other config on Tier A (668 / 945.49 h / 181 seizures). This table re-scores each "
        "Tier A config's locked operating points on the Tier B file subset and reports the shift, "
        "so a Glass-7 vs (e.g.) Best-7 comparison can be read with the denominator change accounted "
        "for. Model = rf. Post-processing = locked default (merge_gap=8 s, min_event_duration=1 s). "
        "Only chb13/15/16/17/18/19 carry any non-Glass-7 files, so only their per-subject rows move.\n")

    for rule in RULES:
        lines.append(f"\n## {rule.upper()}\n")
        d = delta_df[delta_df.rule == rule].sort_values(["scheme", "config", "criterion"])
        lines.append("| config | scheme | criterion | micro-sens A->B | Δ | micro-FA/day A->B | Δ% | pass-rate A->B | Δ pp |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for _, r in d.iterrows():
            lines.append(
                f"| {r.config} | {r.scheme} | {r.criterion} | "
                f"{r.micro_sens_tier_a:.3f}->{r.micro_sens_tier_b:.3f} | {r.micro_sens_delta:+.3f} | "
                f"{r.micro_fa_tier_a:.3f}->{r.micro_fa_tier_b:.3f} | {r.micro_fa_pct_change:+.1f}% | "
                f"{r.pct_met_tier_a:.1f}%->{r.pct_met_tier_b:.1f}% | {r.pct_met_delta:+.1f} |")
        lines.append("")
        sub = d.dropna(subset=["micro_sens_delta"])
        if len(sub):
            lines.append(
                f"**{rule.upper()} summary:** micro-sensitivity shift range "
                f"{sub.micro_sens_delta.min():+.3f} to {sub.micro_sens_delta.max():+.3f} "
                f"(mean {sub.micro_sens_delta.mean():+.3f}); micro-FA/day shift range "
                f"{sub.micro_fa_pct_change.min():+.1f}% to {sub.micro_fa_pct_change.max():+.1f}% "
                f"(mean {sub.micro_fa_pct_change.mean():+.1f}%); pass-rate shift range "
                f"{sub.pct_met_delta.min():+.1f} to {sub.pct_met_delta.max():+.1f} pp.\n")

    lines.append("\n## Per-subject detail (the 6 subjects with non-Glass-7 files)\n")
    movers = per_subject_df[per_subject_df.subject.isin(
        ["chb13", "chb15", "chb16", "chb17", "chb18", "chb19"])]
    piv = movers[movers.rule == "szcore"].pivot_table(
        index=["subject", "config", "scheme", "criterion"], columns="base",
        values=["hours", "event_ref_true", "sens", "fa"])
    lines.append("SzCORE, per (subject, config, scheme, criterion): hours / ref events / sens / FA per base.\n")
    lines.append("| subject | config | scheme | criterion | hours A->B | ref A->B | sens A->B | FA A->B |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for idx, r in piv.iterrows():
        subject, config, scheme, criterion = idx
        def cell(field, fmt):
            a, b = r[(field, "tier_a")], r[(field, "tier_b")]
            if pd.isna(b):
                return f"{a:{fmt}}->(none)"
            return f"{a:{fmt}}->{b:{fmt}}"
        lines.append(
            f"| {subject} | {config} | {scheme} | {criterion} | "
            f"{cell('hours', '.1f')} | {cell('event_ref_true', '.0f')} | "
            f"{cell('sens', '.3f')} | {cell('fa', '.3f')} |")
    lines.append("")

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT_MD}")


def main():
    ref_by_file = ground_truth_events()
    durations = file_durations()
    tier_b_by_subject = tier_b_files_by_subject()
    op_points = pd.read_csv(OP_POINTS_PATH, float_precision="round_trip")

    print("Re-scoring Tier A configs on Tier A vs Tier B file bases...")
    psdf = per_subject_rows(op_points, ref_by_file, durations, tier_b_by_subject)
    psdf.to_csv(OUT_CSV, index=False)
    print(f"  {len(psdf)} per-subject rows -> {OUT_CSV}")

    agg_df = aggregate(psdf)
    delta_df = deltas(agg_df)
    write_report(agg_df, delta_df, psdf)

    # console summary
    for rule in RULES:
        d = delta_df[delta_df.rule == rule].dropna(subset=["micro_sens_delta"])
        if len(d):
            print(f"\n{rule.upper()}: micro-sens delta {d.micro_sens_delta.min():+.3f}..{d.micro_sens_delta.max():+.3f}, "
                  f"micro-FA delta {d.micro_fa_pct_change.min():+.1f}%..{d.micro_fa_pct_change.max():+.1f}%, "
                  f"pass-rate delta {d.pct_met_delta.min():+.1f}..{d.pct_met_delta.max():+.1f} pp")


if __name__ == "__main__":
    main()
