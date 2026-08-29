"""Analysis half of the entropyProfiled_* pilot (see entropy_pilot.py for the
extraction half and the timing story). Answers the actual question the pilot
exists for: would any entropyProfiled_* feature have placed in a channel's
chi-squared top-30 (N_FEATURES_PER_CHANNEL_SELECTED, model_configs.py) if it
hadn't been dropped for cost alone?

Reuses the existing 89-per-channel features from output/features/*.parquet
(no recomputation) and merges in the 9 entropyProfiled_* columns
entropy_pilot.py extracted for the same sampled windows - 89 + 9 = 98,
matching CLAUDE.md section 7's originally locked feature count exactly.

Ranked PER CHANNEL, independently, same as select_top_features_per_channel()
in train.py: MinMaxScaler (chi2 needs non-negative input) then chi2(X, y) per
channel block, ranked descending, top 30. Run per-channel rather than
globally because the three Glass-7-dependency channels (T7-FT9, FT9-FT10,
FT10-T8) are NaN in every file that lacks them (CLAUDE.md section 8) -
dropping any row where ANY of the 21 channels is NaN would throw away every
window from a non-Glass-7-capable file, which the real pipeline never does
(each config only ever touches its own channel subset). Each channel's chi2
uses only the rows where that channel itself is finite.
"""

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler

from audit import OUTPUT_DIR
from entropy_pilot import CANDIDATE_SUBJECTS, ENTROPY_PROFILE_STATS, OUT_PATH as ENTROPY_PATH
from features import FEATURE_NAMES
from model_configs import N_FEATURES_PER_CHANNEL_SELECTED
from preprocess import REQUIRED_CHANNELS

FEATURES_DIR = OUTPUT_DIR / "features"
REPORT_PATH = OUTPUT_DIR / "00f_entropy_pilot.md"

ENTROPY_COLS_PER_CHANNEL = [f"entropyProfiled_{stat}_sampleEntropy" for stat in ENTROPY_PROFILE_STATS]
N_EXISTING = len(FEATURE_NAMES)  # 89
N_TOTAL = N_EXISTING + len(ENTROPY_COLS_PER_CHANNEL)  # 98, matches CLAUDE.md section 7's original lock


def load_existing_features(subject_ids, keys):
    """keys: DataFrame with subject_id, filename, window_start_sec to keep."""
    parts = []
    for subject_id in subject_ids:
        path = FEATURES_DIR / f"{subject_id}.parquet"
        if not path.exists():
            continue
        cols = ["subject_id", "filename", "window_start_sec", "label"] + \
               [f"{ch}__{feat}" for ch in REQUIRED_CHANNELS for feat in FEATURE_NAMES]
        df = pq.read_table(path, columns=cols).to_pandas()
        sub_keys = keys[keys.subject_id == subject_id][["filename", "window_start_sec"]]
        df = df.merge(sub_keys, on=["filename", "window_start_sec"], how="inner")
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def per_channel_top30(merged):
    """Returns a DataFrame: one row per channel, whether any entropyProfiled_*
    feature made the chi2 top-30, its best rank (1 = highest score) and name."""
    y_all = merged["label"].values
    rows = []
    for ch in REQUIRED_CHANNELS:
        existing_cols = [f"{ch}__{feat}" for feat in FEATURE_NAMES]
        entropy_cols = [f"{ch}__{c}" for c in ENTROPY_COLS_PER_CHANNEL]
        block_cols = existing_cols + entropy_cols
        block = merged[block_cols]
        mask = block.notna().all(axis=1)
        n_rows = int(mask.sum())
        if n_rows < 10:
            rows.append(dict(channel=ch, n_rows=n_rows, entropy_in_top30=False,
                              best_entropy_rank=None, best_entropy_feature=None,
                              note="too few finite rows (channel likely absent from every sampled file)"))
            continue

        X = MinMaxScaler().fit_transform(block.loc[mask].values)
        y = y_all[mask.values]
        scores, _ = chi2(X, y)
        scores = np.nan_to_num(scores, nan=-1.0)
        order = np.argsort(scores)[::-1]  # descending score, rank 0 = best
        rank_of = {col_idx: rank for rank, col_idx in enumerate(order)}

        entropy_idx = list(range(N_EXISTING, N_TOTAL))
        entropy_ranks = {block_cols[i]: rank_of[i] for i in entropy_idx}
        best_col, best_rank = min(entropy_ranks.items(), key=lambda kv: kv[1])

        rows.append(dict(
            channel=ch, n_rows=n_rows,
            entropy_in_top30=best_rank < N_FEATURES_PER_CHANNEL_SELECTED,
            best_entropy_rank=best_rank + 1,  # 1-indexed for the report
            best_entropy_feature=best_col.split("__", 1)[1],
        ))
    return pd.DataFrame(rows)


def write_report(summary, merged):
    n_channels_with_hit = int(summary.entropy_in_top30.sum())
    n_channels_scored = int((summary.n_rows >= 10).sum())
    lines = [
        "# entropyProfiled_* chi-squared pilot (CLAUDE.md section 7 / spec 2 section 2 #3)\n",
        f"Stratified sample: {len(merged)} windows "
        f"({(merged.label == 1).sum()} label=1, {(merged.label == 0).sum()} label=0) "
        f"across {merged.subject_id.nunique()} subjects, {merged.filename.nunique()} files. "
        f"89 existing features (output/features/) + 9 entropyProfiled_* features (freshly extracted, "
        f"entropy_pilot.py) = 98/channel, ranked per channel with chi2 (MinMaxScaler first, "
        f"same method as train.py's select_top_features_per_channel), "
        f"top-{N_FEATURES_PER_CHANNEL_SELECTED} per channel.\n",
        f"**Headline: entropyProfiled_* placed in the chi-squared top-{N_FEATURES_PER_CHANNEL_SELECTED} "
        f"in {n_channels_with_hit} of {n_channels_scored} scoreable channels.**\n",
        "## Per-channel result\n",
        "| channel | n_rows | entropy in top-30? | best entropy rank (of 98) | best feature |",
        "|---|---|---|---|---|",
    ]
    for _, r in summary.iterrows():
        rank = r.best_entropy_rank if pd.notna(r.best_entropy_rank) else "-"
        feat = r.best_entropy_feature if r.best_entropy_feature else r.get("note", "-")
        lines.append(f"| {r.channel} | {r.n_rows} | {'YES' if r.entropy_in_top30 else 'no'} | {rank} | {feat} |")

    lines.append("")
    if n_channels_with_hit == 0:
        lines.append(
            "**Conclusion: dropping entropyProfiled_* for cost (2026-08-12) cost nothing in this "
            "sample.** Never reached the chi-squared top-30 in any scoreable channel - the tier-3 "
            "deviation is validated post-hoc as harmless, not just cost-justified. Still worth one "
            "line in Methods/Limitations (the pilot ran on a 2000-window stratified sample, not the "
            "full dataset or a real CV fold - see entropy_pilot.py docstring for the sampling design)."
        )
    else:
        lines.append(
            f"**Conclusion: entropyProfiled_* would have mattered in {n_channels_with_hit} channel(s).** "
            "The cost-only deviation (2026-08-12) is NOT validated as harmless - this needs to be "
            "written up as a real limitation in Methods, not folded into the cost justification. "
            "Consider whether affected channels overlap with any locked channel-ladder config "
            "(CLAUDE.md section 8) before deciding how much this matters for the headline results."
        )

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {REPORT_PATH}")


def main():
    if not ENTROPY_PATH.exists():
        print(f"{ENTROPY_PATH} does not exist yet - entropy_pilot.py extraction hasn't finished. "
              "Nothing to analyze.")
        return

    entropy_df = pd.read_parquet(ENTROPY_PATH)
    keys = entropy_df[["subject_id", "filename", "window_start_sec"]]
    existing = load_existing_features(sorted(entropy_df.subject_id.unique()), keys)

    merged = existing.merge(entropy_df, on=["subject_id", "filename", "window_start_sec"], how="inner")
    print(f"Merged {len(merged)} windows ({len(existing)} existing, {len(entropy_df)} entropy - "
          f"should match if every sampled window round-tripped)")

    summary = per_channel_top30(merged)
    print(summary.to_string(index=False))
    write_report(summary, merged)


if __name__ == "__main__":
    main()
