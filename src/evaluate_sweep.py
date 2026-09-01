"""The ~3h approved run: for every (config, model, fold), sweep the full
threshold grid on the TUNING split and score each threshold with both
SzCORE and Ali rules, saving the whole curve. Deliberately decoupled from
picking a single operating point - CLAUDE.md rule 7 (save raw sweep data,
not a single decision) - so which threshold satisfies which sufficiency
criterion (spec 2 section 1 / CLAUDE.md section 14) can be re-derived later
without rerunning any scoring.

merge_gap and min_event_duration are held at one fixed default here
(MERGE_GAP_DEFAULT_SEC, MIN_EVENT_DURATION_DEFAULT_SEC below) for the
threshold sweep - spec 2 section 6's merge_gap/min_event_duration
sensitivity analysis is a separate, smaller sweep (evaluate_postprocessing_sensitivity.py)
against a subset of folds, not this full grid.

Checkpointed per (config, model, fold) - CLAUDE.md section 3 pattern, same
as train.py: skip work whose output already exists, one failure logs and
continues rather than killing the run.

Usage:
    python evaluate_sweep.py --run-all
    python evaluate_sweep.py --config Full-18 --model rf --scheme 5fold --fold 0
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from audit import OUTPUT_DIR
from evaluate import ground_truth_events, file_durations, score_fold
from model_configs import CONFIGS, LOO_CONFIGS, MODELS, THRESHOLD_GRID
from postprocess import make_candidate_events

PRED_DIR = OUTPUT_DIR / "04_predictions"
SWEEP_DIR = OUTPUT_DIR / "05a_threshold_sweep"
CV_FOLDS_PATH = OUTPUT_DIR / "02_cv_folds.json"
LOG_PATH = OUTPUT_DIR / "_sweep_log.txt"

# Fixed for the main threshold sweep - see module docstring. merge_gap=8s (2
# grid windows) and min_event_duration=1s are mid-range, legal choices from
# the locked/proposed grids (spec 2 section 6): min_event_duration=1s is
# inside the <3s ceiling, merge_gap=8s inside the {0,4,8,12,16,20,28}s set.
MERGE_GAP_DEFAULT_SEC = 8.0
MIN_EVENT_DURATION_DEFAULT_SEC = 1.0


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_cv_folds():
    return json.loads(CV_FOLDS_PATH.read_text(encoding="utf-8"))


def all_fold_stems(cv):
    stems = []
    for config_name in CONFIGS:
        for model_name in MODELS:
            for f in range(cv["five_fold"]["n_folds"]):
                stems.append((config_name, model_name, "5fold", f))
            if config_name in LOO_CONFIGS:
                for held_out in cv["loo"]["folds"]:
                    stems.append((config_name, model_name, "loo", held_out))
    return stems


def sweep_threshold_curve(split_df, ref_by_file, durations, merge_gap_sec, min_event_duration_sec,
                           threshold_grid=THRESHOLD_GRID):
    """The reusable core: full threshold-grid sweep against one split's windows
    at one fixed (merge_gap, min_event_duration). Used both by the main sweep
    (fixed defaults, all 312 folds) and by postprocessing_sensitivity.py
    (varies merge_gap/min_event_duration on a representative subset).
    """
    recorded_hours = len(split_df) * 4.0 / 3600
    rows = []
    for threshold in threshold_grid:
        events_by_file = make_candidate_events(
            split_df, threshold, merge_gap_sec, min_event_duration_sec
        )
        totals = score_fold(events_by_file, ref_by_file, durations)
        row = {"threshold": threshold, "recorded_hours": recorded_hours}
        for rule in ("szcore", "ali"):
            t = totals[rule]
            row[f"tp_{rule}"] = t["tp"]
            row[f"fp_{rule}"] = t["fp"]
            row[f"ref_true_{rule}"] = t["ref_true"]
            row[f"sens_{rule}"] = t["tp"] / t["ref_true"] if t["ref_true"] else float("nan")
            row[f"fa_per_day_{rule}"] = t["fp"] / (recorded_hours / 24) if recorded_hours > 0 else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def sweep_one(config_name, model_name, scheme, fold, ref_by_file, durations):
    fold_label = f"5fold{fold}" if scheme == "5fold" else f"loo_{fold}"
    stem = f"{config_name}_{model_name}_{fold_label}"
    pred_path = PRED_DIR / f"{stem}.parquet"
    out_path = SWEEP_DIR / f"{stem}.parquet"
    if out_path.exists():
        return {"status": "skipped", "run": stem}
    if not pred_path.exists():
        return {"status": "missing_pred", "run": stem}

    t0 = time.time()
    try:
        df = pd.read_parquet(pred_path)
        tuning = df[(df.split == "tuning") & (df.on_uniform_grid)][
            ["subject_id", "filename", "window_start_sec", "prob"]
        ]

        out_df = sweep_threshold_curve(
            tuning, ref_by_file, durations, MERGE_GAP_DEFAULT_SEC, MIN_EVENT_DURATION_DEFAULT_SEC
        )
        out_df["config"] = config_name
        out_df["model"] = model_name
        out_df["scheme"] = scheme
        out_df["fold"] = str(fold)

        SWEEP_DIR.mkdir(parents=True, exist_ok=True)
        tmp_path = out_path.with_suffix(".parquet.tmp")
        out_df.to_parquet(tmp_path, index=False)
        tmp_path.replace(out_path)

        elapsed = time.time() - t0
        return {"status": "ok", "run": stem, "elapsed_sec": elapsed}
    except Exception as e:
        log(f"ERROR {stem}: {type(e).__name__}: {e}")
        return {"status": "error", "run": stem, "error": f"{type(e).__name__}: {e}"}


def run_all():
    cv = load_cv_folds()
    stems = all_fold_stems(cv)
    total = len(stems)
    log(f"evaluate_sweep run_all: {total} (config, model, fold) sweeps, "
        f"{len(THRESHOLD_GRID)} thresholds each, merge_gap={MERGE_GAP_DEFAULT_SEC}s "
        f"min_event_duration={MIN_EVENT_DURATION_DEFAULT_SEC}s")

    ref_by_file = ground_truth_events()
    durations = file_durations()

    started = time.time()
    done = 0
    n_ok = n_skip = n_err = 0
    for config_name, model_name, scheme, fold in stems:
        result = sweep_one(config_name, model_name, scheme, fold, ref_by_file, durations)
        done += 1
        if result["status"] == "ok":
            n_ok += 1
        elif result["status"] == "skipped":
            n_skip += 1
        elif result["status"] == "error":
            n_err += 1
        elapsed = time.time() - started
        rate = done / elapsed if elapsed > 0 else 0
        eta_hours = (total - done) / rate / 3600 if rate > 0 else float("nan")
        log(f"{result['status']} {result['run']} "
            f"({result.get('elapsed_sec', 0):.1f}s) | {done}/{total}, "
            f"{elapsed / 3600:.2f}h elapsed, ETA {eta_hours:.2f}h")

    log(f"Done: {n_ok} ok, {n_skip} skipped, {n_err} error, {total} total")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=list(CONFIGS))
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--scheme", choices=["5fold", "loo"])
    parser.add_argument("--fold")
    parser.add_argument("--run-all", action="store_true")
    args = parser.parse_args()

    if args.run_all:
        run_all()
    else:
        ref_by_file = ground_truth_events()
        durations = file_durations()
        fold = int(args.fold) if args.scheme == "5fold" else args.fold
        result = sweep_one(args.config, args.model, args.scheme, fold, ref_by_file, durations)
        print(json.dumps(result, indent=2, default=str))
