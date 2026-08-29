"""Lightweight impact test for entropyProfiled_total_sampleEntropy (follow-up
to entropy_pilot.py/entropy_pilot_analyze.py - see Consolidated-Context.md
A17). The chi-squared pilot found this feature ranks top-30 in every channel,
which means excluding it isn't cost-neutral - but re-extracting it for the
full dataset costs ~259 extra days (see the conversation this was estimated
in), completely infeasible. This script answers a narrower, cheap question
instead: on the windows we already have it for, does adding it to the
candidate pool actually change RF discrimination, or does chi2 just prefer it
without it mattering downstream?

NOT a full-pipeline-faithful measurement - explicitly scoped down, and the
result is supporting evidence for Limitations/Discussion, not a headline
number:
  - Full-18 config only (18 double-banana channels - avoids the Glass-7 NaN
    complication for a quick check).
  - GroupKFold(5) by subject over the pilot's 13 subjects, not the locked
    23-subject 5-fold/LOO scheme - 13 subjects is what the pilot sampled.
  - No undersampling: the pilot sample is already ~1:1 (stratified 1000/1000
    by design, not natural ~1-2% prevalence), so undersampling would do
    nothing here and natural-prevalence FA/day isn't computable from this
    sample at all (recorded hours isn't a meaningful denominator over a
    stratified window sample) - window-level AUC-ROC only, no sens/FA.
  - chi2 selection still fit inside each fold (CLAUDE.md rule 1), comparing
    the SAME training fold with two candidate pools: 89 existing features
    (current pipeline) vs 90 (+ entropyProfiled_total_sampleEntropy).
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import chi2
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler

from audit import OUTPUT_DIR
from entropy_pilot import OUT_PATH as ENTROPY_PATH
from entropy_pilot_analyze import load_existing_features
from features import FEATURE_NAMES
from model_configs import MODEL_HYPERPARAMS, N_FEATURES_PER_CHANNEL_SELECTED, RANDOM_STATE
from preprocess import DOUBLE_BANANA_CHANNELS  # Full-18's 18 channels

REPORT_PATH = OUTPUT_DIR / "00f_entropy_pilot_impact_test.md"
N_SPLITS = 5
ENTROPY_TOTAL_COL_SUFFIX = "entropyProfiled_total_sampleEntropy"


def select_top_per_channel(X, y, n_features_per_channel, k=N_FEATURES_PER_CHANNEL_SELECTED):
    n_channels = X.shape[1] // n_features_per_channel
    selected = []
    for ch_idx in range(n_channels):
        block = X[:, ch_idx * n_features_per_channel:(ch_idx + 1) * n_features_per_channel]
        scores, _ = chi2(block, y)
        scores = np.nan_to_num(scores, nan=-1.0)
        top = np.argsort(scores)[::-1][:k]
        selected.extend(ch_idx * n_features_per_channel + top)
    return np.array(sorted(selected))


def run_pool(merged, train_idx, test_idx, cols, n_features_per_channel):
    X_train_raw = merged.loc[train_idx, cols].values
    X_test_raw = merged.loc[test_idx, cols].values
    y_train = merged.loc[train_idx, "label"].values
    y_test = merged.loc[test_idx, "label"].values

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    sel = select_top_per_channel(X_train, y_train, n_features_per_channel)
    rf = RandomForestClassifier(**MODEL_HYPERPARAMS["rf"])
    rf.fit(X_train[:, sel], y_train)
    proba = rf.predict_proba(X_test[:, sel])[:, 1]
    return roc_auc_score(y_test, proba), proba, y_test


def main():
    if not ENTROPY_PATH.exists():
        print(f"{ENTROPY_PATH} missing - run entropy_pilot.py first.")
        return

    entropy_df = pd.read_parquet(ENTROPY_PATH)
    keys = entropy_df[["subject_id", "filename", "window_start_sec"]]
    existing = load_existing_features(sorted(entropy_df.subject_id.unique()), keys)
    merged = existing.merge(entropy_df, on=["subject_id", "filename", "window_start_sec"], how="inner")

    baseline_cols = [f"{ch}__{feat}" for ch in DOUBLE_BANANA_CHANNELS for feat in FEATURE_NAMES]
    entropy_cols = [f"{ch}__{ENTROPY_TOTAL_COL_SUFFIX}" for ch in DOUBLE_BANANA_CHANNELS]
    # 90-wide blocks per channel: this channel's 89 existing features, then its
    # entropyProfiled_total column - order matters, select_top_per_channel
    # slices by fixed-width blocks.
    with_entropy_cols = []
    for ch in DOUBLE_BANANA_CHANNELS:
        with_entropy_cols.extend(f"{ch}__{feat}" for feat in FEATURE_NAMES)
        with_entropy_cols.append(f"{ch}__{ENTROPY_TOTAL_COL_SUFFIX}")

    mask = merged[baseline_cols + entropy_cols].notna().all(axis=1)
    merged = merged.loc[mask].reset_index(drop=True)
    print(f"{len(merged)} windows usable across {merged.subject_id.nunique()} subjects "
          f"(Full-18 channels, both pools finite)")

    gkf = GroupKFold(n_splits=N_SPLITS)
    fold_results = []
    all_baseline_proba, all_entropy_proba, all_y = [], [], []
    for fold, (train_idx, test_idx) in enumerate(gkf.split(merged, groups=merged.subject_id)):
        auc_base, proba_base, y_test = run_pool(merged, train_idx, test_idx, baseline_cols, len(FEATURE_NAMES))
        auc_ent, proba_ent, _ = run_pool(merged, train_idx, test_idx, with_entropy_cols, len(FEATURE_NAMES) + 1)
        fold_results.append(dict(fold=fold, n_test=len(test_idx),
                                  test_subjects=sorted(merged.loc[test_idx, "subject_id"].unique()),
                                  auc_baseline_89=auc_base, auc_with_entropy_90=auc_ent,
                                  delta=auc_ent - auc_base))
        all_baseline_proba.append(proba_base)
        all_entropy_proba.append(proba_ent)
        all_y.append(y_test)
        print(f"fold {fold}: baseline={auc_base:.4f} with_entropy={auc_ent:.4f} delta={auc_ent - auc_base:+.4f}")

    results_df = pd.DataFrame(fold_results)
    pooled_y = np.concatenate(all_y)
    pooled_auc_base = roc_auc_score(pooled_y, np.concatenate(all_baseline_proba))
    pooled_auc_ent = roc_auc_score(pooled_y, np.concatenate(all_entropy_proba))

    lines = [
        "# entropyProfiled_total_sampleEntropy impact test (lightweight, see module docstring)\n",
        f"Full-18 (18 double-banana channels), GroupKFold(5) by subject over the pilot's "
        f"{merged.subject_id.nunique()} subjects, {len(merged)} windows "
        f"({(merged.label == 1).sum()} label=1, {(merged.label == 0).sum()} label=0 - stratified "
        f"pilot sample, NOT natural prevalence). Window-level AUC-ROC only - no sens/FA/day "
        f"claim is possible from this sample. NOT a substitute for a real CV run.\n",
        "## Per-fold AUC-ROC\n",
        "| fold | n_test | test subjects | baseline (89 feat/ch) | +entropyProfiled_total (90 feat/ch) | delta |",
        "|---|---|---|---|---|---|",
    ]
    for _, r in results_df.iterrows():
        lines.append(f"| {r.fold} | {r.n_test} | {', '.join(r.test_subjects)} | "
                     f"{r.auc_baseline_89:.4f} | {r.auc_with_entropy_90:.4f} | {r.delta:+.4f} |")
    lines.append("")
    lines.append(f"**Pooled (all folds' held-out predictions concatenated): "
                 f"baseline={pooled_auc_base:.4f}, with entropyProfiled_total={pooled_auc_ent:.4f}, "
                 f"delta={pooled_auc_ent - pooled_auc_base:+.4f}**\n")
    mean_delta = results_df["delta"].mean()
    lines.append(f"Mean per-fold delta: {mean_delta:+.4f} ({(results_df['delta'] > 0).sum()}/{N_SPLITS} "
                 f"folds improved).\n")
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {REPORT_PATH}")
    print(f"Pooled AUC: baseline={pooled_auc_base:.4f} with_entropy={pooled_auc_ent:.4f} "
          f"delta={pooled_auc_ent - pooled_auc_base:+.4f}")


if __name__ == "__main__":
    main()
