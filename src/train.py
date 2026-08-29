"""Per-fold training: prep -> fit -> predict -> save raw probabilities.

No threshold or post-processing decision is made here (spec 2 section 11):
this only produces the raw material for postprocess.py/evaluate.py, so the
whole post-processing/operating-point sweep can be redone later without
retraining (CLAUDE.md rule 7). Every config/fold's real electrode count is
recorded in the sidecar .meta.json, not just its channel count (CLAUDE.md
section 8 - Best-n can pick channels that share electrodes).

POOLING - why this isn't "load per (config, model, fold)"
-----------------------------------------------------------
A naive per-run load (read this fold's fit+tuning+test subjects' feature
columns from disk, every time) measured 140s for one (Full-18, rf, 5fold0)
run alone, almost all of it I/O: for every config, one fold's fit union
tuning union test already covers all 23 subjects (5-fold) or 22 of 23 (LOO),
so "per fold" already means "read (most of) the whole config's column slice
from disk" - and there are 28 folds per config that uses both CV schemes.
Naively repeating that 312 times (once per (config, model, fold), not even
sharing across the 3 models of the same fold) projects to ~12 hours, right at
the CLAUDE.md ceiling, on I/O alone.

Two facts remove almost all of that redundancy:
  1. Six of the seven configs (Full-18, Best-7/4/2, Glass-4, Glass-2) only
     ever need columns from the 18 double-banana channels - Best-n's ranking
     pool *is* Full-18's channel set, and Glass-4/Glass-2's channels are a
     subset of it. Only Glass-7 needs the three Glass-7-dependency columns,
     which aren't in that pool and restrict rows to Tier B.
  2. Every fold of every config re-reads the same underlying per-subject
     data; the fold only changes which subjects are fit/tuning/test.

So this file reads each of the two pools (double-banana, Glass-7) from disk
exactly once per subject into an in-memory dict, and every fold of every
config sharing a pool just slices and concatenates from what's already
resident - no further disk I/O. Within one fold, the undersample/scale/chi2-
select step (identical across the 3 models) also runs once and is reused for
all three model fits.

Usage:
    python train.py --run-all                      # every (config, model, fold), pooled
    python train.py --run-all --dry-run             # list the work, no fit
    python train.py --config Full-18 --model rf --scheme 5fold --fold 0
    python train.py --config Glass-7 --model mlp --scheme loo --fold chb12
"""

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler

from audit import OUTPUT_DIR
from features import FEATURE_NAMES
from model_configs import (
    CONFIGS, LOO_CONFIGS, MODEL_HYPERPARAMS, MODELS,
    N_FEATURES_PER_CHANNEL_SELECTED, RANDOM_STATE, UNDERSAMPLE_RATIO,
)
from preprocess import DOUBLE_BANANA_CHANNELS

FEATURES_DIR = OUTPUT_DIR / "features"
PRED_DIR = OUTPUT_DIR / "04_predictions"
CV_FOLDS_PATH = OUTPUT_DIR / "02_cv_folds.json"
LOG_PATH = OUTPUT_DIR / "_train_log.txt"

META_COLS = ["subject_id", "filename", "window_start_sec", "on_uniform_grid", "label", "seizure_event_id"]
N_FEATURES = len(FEATURE_NAMES)

MODEL_BUILDERS = {
    "rf": lambda: RandomForestClassifier(**MODEL_HYPERPARAMS["rf"]),
    "lr": lambda: LogisticRegression(**MODEL_HYPERPARAMS["lr"]),
    "mlp": lambda: MLPClassifier(hidden_layer_sizes=tuple(MODEL_HYPERPARAMS["mlp"]["hidden_layer_sizes"]),
                                  activation=MODEL_HYPERPARAMS["mlp"]["activation"],
                                  alpha=MODEL_HYPERPARAMS["mlp"]["alpha"],
                                  max_iter=MODEL_HYPERPARAMS["mlp"]["max_iter"],
                                  early_stopping=MODEL_HYPERPARAMS["mlp"]["early_stopping"],
                                  n_iter_no_change=MODEL_HYPERPARAMS["mlp"]["n_iter_no_change"],
                                  random_state=MODEL_HYPERPARAMS["mlp"]["random_state"]),
}

# Pool groups: which configs share a disk read, and the channel superset that
# read must cover. Glass-7 is its own pool (Tier B, 3 dependency channels not
# in the double-banana set).
POOL_DOUBLE_BANANA = "double_banana"
POOL_GLASS7 = "glass7"
CONFIG_POOL = {
    "Full-18": POOL_DOUBLE_BANANA, "Best-7": POOL_DOUBLE_BANANA,
    "Best-4": POOL_DOUBLE_BANANA, "Best-2": POOL_DOUBLE_BANANA,
    "Glass-4": POOL_DOUBLE_BANANA, "Glass-2": POOL_DOUBLE_BANANA,
    "Glass-7": POOL_GLASS7,
}
POOL_CHANNELS = {
    POOL_DOUBLE_BANANA: DOUBLE_BANANA_CHANNELS,
    POOL_GLASS7: CONFIGS["Glass-7"]["channels"],
}


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_cv_folds():
    return json.loads(CV_FOLDS_PATH.read_text(encoding="utf-8"))


def resolve_fold_subjects(cv, scheme, fold):
    if scheme == "5fold":
        fold = str(fold)
        folds = cv["five_fold"]["folds"]
        test_subjects = folds[fold]
        tuning_fold = str((int(fold) + 1) % cv["five_fold"]["n_folds"])
        tuning_subjects = folds[tuning_fold]
        fit_subjects = sorted(
            s for f, subs in folds.items() if f not in (fold, tuning_fold) for s in subs
        )
        label = f"5fold{fold}"
    elif scheme == "loo":
        entry = cv["loo"]["folds"][fold]
        test_subjects = [entry["held_out"]]
        tuning_subjects = entry["tuning_pool"]
        fit_subjects = entry["fit_subjects"]
        label = f"loo_{fold}"
    else:
        raise ValueError(scheme)
    return fit_subjects, tuning_subjects, test_subjects, label


def feature_columns(channels):
    return [f"{ch}__{feat}" for ch in channels for feat in FEATURE_NAMES]


def load_pool(pool_name):
    """subject_id -> DataFrame(meta + this pool's feature columns), one parquet
    read per subject, cached for the caller's lifetime."""
    channels = POOL_CHANNELS[pool_name]
    cols = META_COLS + feature_columns(channels)
    pool = {}
    t0 = time.time()
    for path in sorted(FEATURES_DIR.glob("chb*.parquet")):
        subject_id = path.stem
        table = pq.read_table(path, columns=cols)
        df = table.to_pandas()
        feat_cols = [c for c in cols if c not in META_COLS]
        df[feat_cols] = df[feat_cols].astype(np.float32)
        pool[subject_id] = df
    log(f"Loaded pool '{pool_name}' ({len(channels)} channels, {len(pool)} subjects) "
        f"in {time.time() - t0:.1f}s")
    return pool


def slice_pool(pool, subjects, columns):
    if not subjects:
        return pd.DataFrame(columns=META_COLS + columns)
    frames = [pool[s][META_COLS + columns] for s in subjects]
    return pd.concat(frames, axis=0, ignore_index=True)


def drop_nonfinite_rows(df, feat_cols):
    if len(df) == 0:
        return df, 0
    mask = df[feat_cols].isna().any(axis=1)
    n_dropped = int(mask.sum())
    return df.loc[~mask].reset_index(drop=True), n_dropped


def undersample(df, ratio=UNDERSAMPLE_RATIO, random_state=RANDOM_STATE):
    pos = df[df.label == 1]
    neg = df[df.label == 0]
    n_neg_keep = min(len(neg), len(pos) * ratio)
    neg_sample = neg.sample(n=n_neg_keep, random_state=random_state)
    return pd.concat([pos, neg_sample], axis=0).sample(frac=1, random_state=random_state).reset_index(drop=True)


def select_top_features_per_channel(X, y, channels, k=N_FEATURES_PER_CHANNEL_SELECTED):
    selected = []
    for ch_idx in range(len(channels)):
        block = X[:, ch_idx * N_FEATURES:(ch_idx + 1) * N_FEATURES]
        scores, _ = chi2(block, y)
        scores = np.nan_to_num(scores, nan=-1.0)
        top = np.argsort(scores)[::-1][:k]
        selected.extend(ch_idx * N_FEATURES + top)
    return np.array(sorted(selected))


def rank_channels_by_rf_importance(pool, fit_subjects, pool_channels, random_state=RANDOM_STATE):
    """Preliminary RF over the full ranking pool, fit on undersampled
    fit-subject data only (CLAUDE.md rule 1). Returns pool_channels sorted by
    summed per-channel feature_importances_, descending.
    """
    cols = feature_columns(pool_channels)
    fit_df = slice_pool(pool, fit_subjects, cols)
    fit_df, _ = drop_nonfinite_rows(fit_df, cols)
    df_us = undersample(fit_df)
    scaler = MinMaxScaler()
    X = scaler.fit_transform(df_us[cols].values)
    y = df_us["label"].values
    rf = RandomForestClassifier(**MODEL_HYPERPARAMS["rf"])
    rf.fit(X, y)
    importances = rf.feature_importances_.reshape(len(pool_channels), N_FEATURES).sum(axis=1)
    order = np.argsort(importances)[::-1]
    return [pool_channels[i] for i in order]


def electrode_count(channels):
    electrodes = set()
    for ch in channels:
        a, b = ch.split("-")
        electrodes.add(a)
        electrodes.add(b)
    return len(electrodes)


def output_paths(config_name, model_name, fold_label):
    stem = f"{config_name}_{model_name}_{fold_label}"
    return PRED_DIR / f"{stem}.parquet", PRED_DIR / f"{stem}.meta.json"


def fold_channels(config_name, cfg, pool, fit_subjects):
    if cfg["data_driven"]:
        return rank_channels_by_rf_importance(pool, fit_subjects, cfg["ranking_pool"])[:cfg["n_channels"]]
    return cfg["channels"]


def prep_fold(pool, channels, fit_subjects, tuning_subjects, test_subjects):
    """Everything shared by the 3 models for one (config, fold): slice from the
    resident pool, drop non-finite rows, undersample fit, scale, chi2-select.
    No disk I/O here - pool is already in memory.
    """
    cols = feature_columns(channels)
    fit_df, n_dropped_fit = drop_nonfinite_rows(slice_pool(pool, fit_subjects, cols), cols)
    tuning_df, n_dropped_tuning = drop_nonfinite_rows(slice_pool(pool, tuning_subjects, cols), cols)
    test_df, n_dropped_test = drop_nonfinite_rows(slice_pool(pool, test_subjects, cols), cols)

    fit_us = undersample(fit_df)
    scaler = MinMaxScaler()
    X_fit = scaler.fit_transform(fit_us[cols].values)
    y_fit = fit_us["label"].values
    sel_idx = select_top_features_per_channel(X_fit, y_fit, channels)

    return {
        "cols": cols, "scaler": scaler, "sel_idx": sel_idx,
        "X_fit": X_fit[:, sel_idx], "y_fit": y_fit,
        "tuning_df": tuning_df, "test_df": test_df,
        "n_fit_rows_raw": len(fit_df), "n_fit_rows_undersampled": len(fit_us),
        "n_dropped": {"fit": n_dropped_fit, "tuning": n_dropped_tuning, "test": n_dropped_test},
    }


def fit_and_save_one_model(model_name, prepped, channels, config_name, scheme, fold, fold_label,
                            fit_subjects, tuning_subjects, test_subjects, n_dropped_rank):
    pred_path, meta_path = output_paths(config_name, model_name, fold_label)
    if pred_path.exists() and meta_path.exists():
        return {"status": "skipped", "run": pred_path.stem}

    try:
        return _fit_and_save_one_model_inner(
            model_name, prepped, channels, config_name, scheme, fold, fold_label,
            fit_subjects, tuning_subjects, test_subjects, n_dropped_rank,
            pred_path, meta_path,
        )
    except Exception as e:
        # One model/fold failing must not take down a 12+ hour run (CLAUDE.md
        # section 3: a single failure is skipped, not fatal to the whole run).
        # No partial file is left behind - the .tmp write only replaces the
        # real path on success, so a retry sees this as still-not-done.
        log(f"ERROR {pred_path.stem}: {type(e).__name__}: {e}")
        return {"status": "error", "run": pred_path.stem, "error": f"{type(e).__name__}: {e}"}


def _fit_and_save_one_model_inner(model_name, prepped, channels, config_name, scheme, fold, fold_label,
                                   fit_subjects, tuning_subjects, test_subjects, n_dropped_rank,
                                   pred_path, meta_path):
    t0 = time.time()
    model = MODEL_BUILDERS[model_name]()
    model.fit(prepped["X_fit"], prepped["y_fit"])

    def predict(split_df, split_name):
        if len(split_df) == 0:
            return split_df.assign(prob=pd.Series(dtype=float), split=split_name)
        X = prepped["scaler"].transform(split_df[prepped["cols"]].values)[:, prepped["sel_idx"]]
        prob = model.predict_proba(X)[:, 1]
        out = split_df[META_COLS].copy()
        out["prob"] = prob
        out["split"] = split_name
        return out

    result_df = pd.concat(
        [predict(prepped["tuning_df"], "tuning"), predict(prepped["test_df"], "test")],
        axis=0, ignore_index=True,
    )

    PRED_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = pred_path.with_suffix(".parquet.tmp")
    result_df.to_parquet(tmp_path, index=False)
    tmp_path.replace(pred_path)

    elapsed = time.time() - t0
    meta = {
        "config": config_name, "model": model_name, "scheme": scheme, "fold": fold,
        "fold_label": fold_label, "channels": channels, "n_channels": len(channels),
        "n_electrodes": electrode_count(channels),
        "fit_subjects": fit_subjects, "tuning_subjects": tuning_subjects, "test_subjects": test_subjects,
        "n_fit_rows_raw": prepped["n_fit_rows_raw"],
        "n_fit_rows_undersampled": prepped["n_fit_rows_undersampled"],
        "n_tuning_rows": len(prepped["tuning_df"]), "n_test_rows": len(prepped["test_df"]),
        "n_features_selected": len(prepped["sel_idx"]),
        "n_dropped_nonfinite_rows": {**prepped["n_dropped"], "ranking": n_dropped_rank},
        "elapsed_sec": elapsed,
    }
    tmp_meta_path = meta_path.with_suffix(".json.tmp")
    tmp_meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp_meta_path.replace(meta_path)

    return {"status": "ok", "run": pred_path.stem, "elapsed_sec": elapsed,
            "n_rows_written": len(result_df), "output_bytes": pred_path.stat().st_size}


def config_folds(cv, config_name):
    """(scheme, fold) pairs this config is evaluated on."""
    pairs = [("5fold", f) for f in range(cv["five_fold"]["n_folds"])]
    if config_name in LOO_CONFIGS:
        pairs += [("loo", h) for h in cv["loo"]["folds"]]
    return pairs


def run_fold(config_name, scheme, fold, pool, cv, dry_run=False):
    cfg = CONFIGS[config_name]
    fit_subjects, tuning_subjects, test_subjects, fold_label = resolve_fold_subjects(cv, scheme, fold)

    pred_path_by_model = {m: output_paths(config_name, m, fold_label) for m in MODELS}
    if all(p.exists() and mp.exists() for p, mp in pred_path_by_model.values()):
        return [{"status": "skipped", "run": f"{config_name}_{m}_{fold_label}"} for m in MODELS]

    if dry_run:
        return [{"status": "dry_run", "run": f"{config_name}_{m}_{fold_label}",
                  "fit_subjects": fit_subjects, "tuning_subjects": tuning_subjects,
                  "test_subjects": test_subjects} for m in MODELS]

    fold_stem = f"{config_name}_{{model}}_{fold_label}"
    try:
        n_dropped_rank = 0
        if cfg["data_driven"]:
            channels = rank_channels_by_rf_importance(pool, fit_subjects, cfg["ranking_pool"])[:cfg["n_channels"]]
        else:
            channels = cfg["channels"]

        prepped = prep_fold(pool, channels, fit_subjects, tuning_subjects, test_subjects)
    except Exception as e:
        # Ranking/prep is shared by all 3 models of this fold - if it fails,
        # none of the 3 can proceed, but the rest of the run must (CLAUDE.md
        # section 3: one failure is skipped, never fatal to the whole run).
        log(f"ERROR {config_name}_{fold_label} (ranking/prep): {type(e).__name__}: {e}")
        return [{"status": "error", "run": fold_stem.format(model=m),
                 "error": f"ranking/prep: {type(e).__name__}: {e}"} for m in MODELS]

    results = []
    for model_name in MODELS:
        result = fit_and_save_one_model(
            model_name, prepped, channels, config_name, scheme, fold, fold_label,
            fit_subjects, tuning_subjects, test_subjects, n_dropped_rank,
        )
        results.append(result)
        log(f"{result['status']} {result['run']} "
            f"({result.get('elapsed_sec', 0):.1f}s)" if result["status"] == "ok"
            else f"{result['status']} {result['run']}")
    return results


def run_all(dry_run=False, only_configs=None):
    cv = load_cv_folds()
    configs = only_configs or list(CONFIGS)
    by_pool = {}
    for c in configs:
        by_pool.setdefault(CONFIG_POOL[c], []).append(c)

    fold_units = [(config_name, scheme, fold)
                  for pool_name, pool_configs in by_pool.items()
                  for config_name in pool_configs
                  for scheme, fold in config_folds(cv, config_name)]
    total_folds = len(fold_units)
    total_model_runs = total_folds * len(MODELS)
    log(f"run_all: {total_folds} folds ({total_model_runs} model runs across {list(by_pool.values())})")

    all_results = []
    folds_done = 0
    started = time.time()
    for pool_name, pool_configs in by_pool.items():
        pool = None if dry_run else load_pool(pool_name)
        for config_name in pool_configs:
            for scheme, fold in config_folds(cv, config_name):
                results = run_fold(config_name, scheme, fold, pool, cv, dry_run=dry_run)
                all_results.extend(results)
                folds_done += 1
                elapsed = time.time() - started
                rate = folds_done / elapsed if elapsed > 0 else 0
                eta_hours = (total_folds - folds_done) / rate / 3600 if rate > 0 else float("nan")
                log(f"progress: {folds_done}/{total_folds} folds, "
                    f"{elapsed / 3600:.2f}h elapsed, ETA {eta_hours:.2f}h")
        del pool
        gc.collect()
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=list(CONFIGS))
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--scheme", choices=["5fold", "loo"])
    parser.add_argument("--fold")
    parser.add_argument("--run-all", action="store_true")
    parser.add_argument("--only-configs", nargs="+", choices=list(CONFIGS))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.run_all or args.only_configs:
        results = run_all(dry_run=args.dry_run, only_configs=args.only_configs)
        n_ok = sum(1 for r in results if r["status"] == "ok")
        n_skip = sum(1 for r in results if r["status"] == "skipped")
        print(f"Done: {n_ok} ok, {n_skip} skipped, {len(results)} total")
    else:
        cv = load_cv_folds()
        pool = load_pool(CONFIG_POOL[args.config])
        fold = int(args.fold) if args.scheme == "5fold" else args.fold
        results = run_fold(args.config, args.scheme, fold, pool, cv, dry_run=args.dry_run)
        print(json.dumps(results, indent=2, ensure_ascii=False, default=str))
