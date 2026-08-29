"""Data prep for spec 2 section 10's four figures (docs/06-spec2-cv-training-
postprocessing-evaluation-th.md). Kept separate from make_figures.py so the
(sometimes expensive) data computation can be re-run independently of
plotting tweaks - re-rendering a figure should never require re-scoring.

Design decisions made here, superseding the original spec text where the
project owner has since locked something more specific:

  - Channel-ladder y-axis is event-level per-subject pass rate (%), not
    macro/micro sensitivity as spec 2 section 10 originally proposed - this
    was locked later in the session as the primary metric (noisy at current
    n, but the clinically-grounded one; AUC-ROC is the secondary panel that
    shows raw discrimination survives channel reduction even where pass rate
    doesn't). Pulled directly from 06_results_szcore.csv rather than
    recomputed, since that table already handles a subtlety this script
    would otherwise have to re-derive: n_subjects varies by criterion within
    the same (config, scheme), because a fold can fail to find ANY threshold
    meeting a strict FA ceiling on its tuning curve (pick_threshold returns
    None), which drops that fold's subjects from that criterion's count
    only - not a bug, see select_operating_points_percentile.py.
  - Electrode count (x-axis) uses the MEAN across the 5 folds for Best-n
    configs, since channel selection is data-driven and genuinely differs
    fold to fold (verified: Best-2 uses 3 distinct channel sets across 5
    folds, Best-4 and Best-7 use 5 distinct sets each - CLAUDE.md section 8's
    "must report actual electrode count every fold" is satisfied by also
    keeping min/max alongside the mean, not collapsing to one number).
  - AUC-ROC/PR-AUC pooled from 5-fold TEST predictions (RF, on_uniform_grid),
    window-level - this is a fresh computation, not previously in any
    06_results table. Glass-7 pools 825,185 windows vs 850,387+ for the
    other six configs (Tier B vs Tier A, CLAUDE.md section 8's dependency-
    channel caveat) - noted explicitly because PR-AUC (unlike ROC-AUC) is
    sensitive to class prevalence, so a prevalence difference between tiers
    would show up there specifically.
  - Per-patient heatmap and SzCORE-vs-Ali comparison both read
    06a_per_subject_results_raw.csv / 06_results_*.csv directly - no new
    scoring, just reshaping already-validated numbers.
"""

import glob
import json

import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from audit import OUTPUT_DIR
from model_configs import CONFIGS

FIG_DIR = OUTPUT_DIR / "07_figures"
PRED_DIR = OUTPUT_DIR / "04_predictions"

ALL_CONFIGS = ["Full-18", "Best-7", "Glass-7", "Best-4", "Glass-4", "Best-2", "Glass-2"]
LOO_CONFIGS = ["Full-18", "Glass-7", "Glass-2"]

# Subjects flagged as recurring hard cases across this project's analyses
# (low AUC / consistent criterion misses regardless of config or post-
# processing choice - see output/05_postprocessing_sensitivity.md and the
# threshold-transfer investigation) - project owner's list, 2026-08-17.
HARD_CASE_SUBJECTS = ["chb12", "chb14", "chb06", "chb13", "chb20"]


def electrode_counts_by_config():
    rows = []
    for f in glob.glob(str(PRED_DIR / "*_rf_5fold*.meta.json")):
        d = json.load(open(f))
        rows.append({"config": d["config"], "fold": d["fold"], "n_electrodes": d["n_electrodes"]})
    df = pd.DataFrame(rows)
    summary = df.groupby("config")["n_electrodes"].agg(
        electrodes_mean="mean", electrodes_min="min", electrodes_max="max"
    ).reset_index()
    return summary


def pooled_auc_by_config():
    rows = []
    for config_name in ALL_CONFIGS:
        parts = []
        for fold in range(5):
            path = PRED_DIR / f"{config_name}_rf_5fold{fold}.parquet"
            d = pd.read_parquet(path, columns=["split", "on_uniform_grid", "label", "prob"])
            parts.append(d[(d.split == "test") & (d.on_uniform_grid)])
        pooled = pd.concat(parts, ignore_index=True)
        rows.append({
            "config": config_name,
            "n_windows_auc": len(pooled),
            "n_positive_auc": int(pooled.label.sum()),
            "auc_roc": roc_auc_score(pooled.label, pooled.prob),
            "auc_pr": average_precision_score(pooled.label, pooled.prob),
        })
    return pd.DataFrame(rows)


def build_channel_ladder():
    results = pd.read_csv(OUTPUT_DIR / "06_results_szcore.csv")
    rf = results[results.model == "rf"]

    electrodes = electrode_counts_by_config()
    auc = pooled_auc_by_config()

    main = rf[rf.scheme == "5fold"][
        ["config", "criterion", "n_subjects", "pct_subjects_meeting_criterion"]
    ].rename(columns={"n_subjects": "n_subjects_5fold", "pct_subjects_meeting_criterion": "pct_met_5fold"})

    loo = rf[(rf.scheme == "loo") & (rf.config.isin(LOO_CONFIGS))][
        ["config", "criterion", "n_subjects", "pct_subjects_meeting_criterion"]
    ].rename(columns={"n_subjects": "n_subjects_loo", "pct_subjects_meeting_criterion": "pct_met_loo"})

    out = main.merge(loo, on=["config", "criterion"], how="left")
    out = out.merge(electrodes, on="config", how="left")
    out = out.merge(auc, on="config", how="left")

    out["n_channels"] = out.config.map(lambda c: CONFIGS[c]["n_channels"])
    out["family"] = out.config.map(
        lambda c: "Full-18" if c == "Full-18" else ("Glass-n" if c.startswith("Glass") else "Best-n")
    )
    out["tier"] = out.config.map(lambda c: CONFIGS[c]["tier"])

    cols = ["config", "family", "tier", "n_channels", "electrodes_mean", "electrodes_min", "electrodes_max",
            "criterion", "n_subjects_5fold", "pct_met_5fold", "n_subjects_loo", "pct_met_loo",
            "n_windows_auc", "n_positive_auc", "auc_roc", "auc_pr"]
    out = out[cols].sort_values(["criterion", "n_channels"])
    return out


def build_per_patient_heatmap():
    raw = pd.read_csv(OUTPUT_DIR / "06a_per_subject_results_raw.csv", float_precision="round_trip")
    sub = raw[(raw.model == "rf") & (raw.rule == "szcore") & (raw.scheme == "5fold")].copy()
    sub["is_hard_case"] = sub.subject.isin(HARD_CASE_SUBJECTS)
    cols = ["config", "subject", "criterion", "sens", "fa", "met", "hours",
            "event_tp", "event_fp", "event_ref_true", "is_hard_case"]
    return sub[cols].sort_values(["criterion", "config", "subject"])


def build_szcore_vs_ali():
    sz = pd.read_csv(OUTPUT_DIR / "06_results_szcore.csv")
    ali = pd.read_csv(OUTPUT_DIR / "06_results_ali.csv")
    sz = sz[(sz.model == "rf")].copy()
    ali = ali[(ali.model == "rf")].copy()
    sz["rule"] = "szcore"
    ali["rule"] = "ali"
    cols = ["config", "model", "scheme", "criterion", "rule", "n_subjects", "n_hours", "n_events_ref",
            "macro_sensitivity", "micro_sensitivity", "micro_fa_per_day", "pct_subjects_meeting_criterion"]
    out = pd.concat([sz[cols], ali[cols]], ignore_index=True)
    return out.sort_values(["config", "scheme", "criterion", "rule"])


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    ladder = build_channel_ladder()
    ladder.to_csv(FIG_DIR / "data_channel_ladder.csv", index=False)
    print(f"Wrote data_channel_ladder.csv ({len(ladder)} rows)")

    heatmap = build_per_patient_heatmap()
    heatmap.to_csv(FIG_DIR / "data_per_patient_heatmap.csv", index=False)
    print(f"Wrote data_per_patient_heatmap.csv ({len(heatmap)} rows)")

    comparison = build_szcore_vs_ali()
    comparison.to_csv(FIG_DIR / "data_szcore_vs_ali.csv", index=False)
    print(f"Wrote data_szcore_vs_ali.csv ({len(comparison)} rows)")

    print("\nNote: sens-vs-FA tradeoff curve data (figure 3) is NOT built by this "
          "script - piloted at 615s/config on the full 279-point threshold grid "
          "(~72min for all 7 configs), over the 10-minute CLAUDE.md threshold, "
          "so it's held for explicit go-ahead. See build_tradeoff_curve.py.")


if __name__ == "__main__":
    main()
