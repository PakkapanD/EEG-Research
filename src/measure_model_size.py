"""Model size / inference time, spec 2 section 8 ("will this run on a
microcontroller"). Not reused from training - train.py only persists
predictions, not fitted models (CLAUDE.md rule 7: raw predictions on disk so
post-processing can be redone without retraining; the trade-off is that
anything about the MODEL OBJECT itself needs a small separate refit).

Scoped to RF/Full-18 (largest config, 540 selected features) and RF/Glass-2
(smallest, 60 selected features) - the two ends of the channel ladder, per
the project owner's request (2026-08-16), not the full 7x3 grid: this
question only needs the size/latency range, not every combination.

Refits on fold 0 (5-fold) exactly as train.py would - same hyperparameters,
same undersample/scale/chi2-select pipeline - so the measured model is
representative of what actually produced the results tables, not a fresh
untuned stand-in.
"""

import pickle
import time

import numpy as np
from sklearn.tree._tree import TREE_LEAF

from audit import OUTPUT_DIR
from model_configs import CONFIGS, MODEL_HYPERPARAMS
from train import (
    load_cv_folds, resolve_fold_subjects, load_pool, prep_fold,
    CONFIG_POOL, MODEL_BUILDERS,
)

OUT_PATH = OUTPUT_DIR / "06d_model_size_inference.md"
TARGET_CONFIGS = ["Full-18", "Glass-2"]
SCHEME, FOLD = "5fold", 0
N_INFERENCE_TRIALS = 2000  # single-window predict_proba calls, timed individually
N_RAW_TRIALS = 500  # raw traversal is timed per-tree, so this is 500 x 300 = 150k tree walks


def _traverse_one_tree(children_left, children_right, feature, threshold, value, x):
    """Walk one fitted tree's raw arrays for one row. Arrays are passed in
    already unpacked from the Cython Tree object (see raw_forest_predict_proba)
    rather than accessed as tree_.children_left etc. inside the loop - the
    first version of this function did that and cost ~21ms/prediction despite
    only visiting ~13 nodes x 300 trees; the Python-level attribute lookup on
    a Cython object, repeated on every node visit, was the actual bottleneck,
    not the array indexing itself.
    """
    node = 0
    while children_left[node] != TREE_LEAF:
        if x[feature[node]] <= threshold[node]:
            node = children_left[node]
        else:
            node = children_right[node]
    return value[node][0]


def raw_forest_predict_proba(model, x_row):
    total = np.zeros(model.n_classes_)
    for est in model.estimators_:
        t = est.tree_
        leaf = _traverse_one_tree(t.children_left, t.children_right, t.feature, t.threshold, t.value, x_row)
        total += leaf / leaf.sum()
    return total / len(model.estimators_)


def measure_one(config_name, pool, fit_subjects, tuning_subjects, test_subjects):
    cfg = CONFIGS[config_name]
    channels = cfg["channels"]

    t0 = time.time()
    prepped = prep_fold(pool, channels, fit_subjects, tuning_subjects, test_subjects)
    prep_sec = time.time() - t0

    t0 = time.time()
    model = MODEL_BUILDERS["rf"]()
    model.fit(prepped["X_fit"], prepped["y_fit"])
    fit_sec = time.time() - t0

    n_estimators = len(model.estimators_)
    total_nodes = sum(tree.tree_.node_count for tree in model.estimators_)
    pickled = pickle.dumps(model)
    size_mb = len(pickled) / 1e6

    X_test_full = prepped["scaler"].transform(prepped["test_df"][prepped["cols"]].values)[:, prepped["sel_idx"]]
    n_test = len(X_test_full)

    # Batch throughput (amortizes fixed per-call overhead across many windows)
    t0 = time.time()
    model.predict_proba(X_test_full)
    batch_sec = time.time() - t0
    per_window_batch_us = (batch_sec / n_test) * 1e6

    # Single-window latency (what a real-time deployment actually pays per
    # window: one predict_proba call per new window, not one big batch).
    # n_jobs=-1 (the training config) is wrong for this specific measurement:
    # it makes every predict_proba call dispatch through joblib's parallel
    # backend, and that dispatch overhead is a near-fixed ~90ms regardless of
    # input size - confirmed by a first pass where Full-18 (540 features) and
    # Glass-2 (60 features) came out at 90098.8us and 90139.1us respectively,
    # i.e. identical to within noise, while batch throughput (which amortizes
    # the dispatch cost across many rows) correctly showed Glass-2 costing
    # ~2x Full-18 per window (more total tree nodes - see table). No embedded
    # deployment would spin up a multiprocessing pool per window, so n_jobs=1
    # is what the "will this run on a microcontroller" question actually
    # needs; predict-time parallelism is controlled by the fitted estimator's
    # n_jobs attribute and can be changed after fitting without retraining.
    model.n_jobs = 1
    rng = np.random.RandomState(42)
    idx = rng.choice(n_test, size=min(N_INFERENCE_TRIALS, n_test), replace=False)
    single_times = []
    for i in idx:
        row = X_test_full[i:i + 1]
        t0 = time.perf_counter()
        model.predict_proba(row)
        single_times.append(time.perf_counter() - t0)
    single_times = np.array(single_times)
    model.n_jobs = MODEL_HYPERPARAMS["rf"]["n_jobs"]  # restore before returning/pickling

    # Raw traversal: bypasses sklearn's predict_proba call path (input
    # validation, dtype/shape checks, and - confirmed above - joblib.Parallel
    # dispatch machinery that runs even at n_jobs=1) entirely. n_jobs=1 still
    # measured ~32ms for both configs, next to identical regardless of size
    # and ~9000x the batch-amortized cost - so the sklearn API call itself,
    # not parallelism, was the dominant cost. This is what a hand-ported
    # embedded implementation (walk each tree's array with plain comparisons)
    # would actually cost, which is the number "will this run on a
    # microcontroller" needs - the sklearn-API numbers above answer "how fast
    # is calling this from Python," a different and much less relevant
    # question for embedded deployment.
    raw_idx = rng.choice(n_test, size=min(N_RAW_TRIALS, n_test), replace=False)
    # Correctness check: raw traversal must agree with sklearn's own
    # predict_proba, or the "faster" number would just be a wrong number.
    sklearn_proba = model.predict_proba(X_test_full[raw_idx[0]:raw_idx[0] + 1])[0]
    raw_proba = raw_forest_predict_proba(model, X_test_full[raw_idx[0]])
    assert np.allclose(sklearn_proba, raw_proba, atol=1e-9), (
        f"raw traversal disagrees with sklearn: {raw_proba} vs {sklearn_proba}"
    )
    raw_times = []
    for i in raw_idx:
        t0 = time.perf_counter()
        raw_forest_predict_proba(model, X_test_full[i])
        raw_times.append(time.perf_counter() - t0)
    raw_times = np.array(raw_times)

    return dict(
        config=config_name, n_channels=cfg["n_channels"], n_features_selected=len(prepped["sel_idx"]),
        n_fit_rows=len(prepped["X_fit"]), prep_sec=prep_sec, fit_sec=fit_sec,
        n_estimators=n_estimators, total_nodes=total_nodes,
        avg_nodes_per_tree=total_nodes / n_estimators, size_mb=size_mb,
        n_test_windows=n_test, per_window_batch_us=per_window_batch_us,
        per_window_single_median_us=np.median(single_times) * 1e6,
        per_window_single_p95_us=np.percentile(single_times, 95) * 1e6,
        per_window_single_mean_us=np.mean(single_times) * 1e6,
        per_window_raw_median_us=np.median(raw_times) * 1e6,
        per_window_raw_p95_us=np.percentile(raw_times, 95) * 1e6,
    )


def main():
    cv = load_cv_folds()
    fit_subjects, tuning_subjects, test_subjects, _ = resolve_fold_subjects(cv, SCHEME, FOLD)

    pool_name = CONFIG_POOL[TARGET_CONFIGS[0]]
    assert all(CONFIG_POOL[c] == pool_name for c in TARGET_CONFIGS), "expected both configs from one pool"
    pool = load_pool(pool_name)

    results = []
    for config_name in TARGET_CONFIGS:
        print(f"Measuring {config_name}...")
        r = measure_one(config_name, pool, fit_subjects, tuning_subjects, test_subjects)
        results.append(r)
        print(f"  {r['n_estimators']} trees, {r['total_nodes']:,} nodes, {r['size_mb']:.2f} MB, "
              f"{r['per_window_raw_median_us']:.1f} us/window (raw traversal median), "
              f"{r['per_window_single_median_us']:.1f} us/window (sklearn single-call, n_jobs=1)")

    lines = [
        "# Model size / inference time (spec 2 section 8)",
        "",
        f"RF only ({MODEL_HYPERPARAMS['rf']}), refit on fold {FOLD} ({SCHEME}) exactly as train.py "
        "would - same undersample/scale/chi2-select pipeline, same hyperparameters. Scoped to "
        f"{' and '.join(TARGET_CONFIGS)} (largest/smallest configs) per project owner request "
        "2026-08-16, not the full config x model grid.",
        "",
        "| config | channels | features selected | n_estimators | total nodes | avg nodes/tree | "
        "pickled size (MB) | raw traversal, median (us) | raw traversal, p95 (us) | "
        "sklearn batch call, n_jobs=-1 (us) | sklearn single call, n_jobs=1 median (us) |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['config']} | {r['n_channels']} | {r['n_features_selected']} | {r['n_estimators']} | "
            f"{r['total_nodes']:,} | {r['avg_nodes_per_tree']:.0f} | {r['size_mb']:.2f} | "
            f"{r['per_window_raw_median_us']:.1f} | {r['per_window_raw_p95_us']:.1f} | "
            f"{r['per_window_batch_us']:.1f} | {r['per_window_single_median_us']:.1f} |"
        )
    batch_us = [r["per_window_batch_us"] for r in results]
    single_us = [r["per_window_single_median_us"] for r in results]
    raw_us = [r["per_window_raw_median_us"] for r in results]
    by_config = {r["config"]: r for r in results}
    a, b = TARGET_CONFIGS[0], TARGET_CONFIGS[1]
    raw_vs_batch_ratio = [raw_us[i] / batch_us[i] for i in range(len(results))]
    single_vs_batch_ratio = [single_us[i] / batch_us[i] for i in range(len(results))]
    raw_pct_diff = abs(by_config[a]["per_window_raw_median_us"] - by_config[b]["per_window_raw_median_us"]) \
        / min(by_config[a]["per_window_raw_median_us"], by_config[b]["per_window_raw_median_us"]) * 100
    node_ratio = max(r["avg_nodes_per_tree"] for r in results) / min(r["avg_nodes_per_tree"] for r in results)
    size_summary = ", ".join(f"{r['config']} {r['size_mb']:.1f}MB" for r in results)

    lines += [
        "",
        "**Two sklearn-API measurements were tried first and rejected as misleading before landing on "
        "raw traversal.** Both call `predict_proba` on one row at a time, differing only in "
        "`model.n_jobs`: n_jobs=-1 (the training config) measured ~90ms for both Full-18 and Glass-2 "
        "- identical regardless of 9x fewer input features, because it is timing joblib's parallel-"
        f"dispatch setup/teardown, not tree traversal. Switching to n_jobs=1 measured "
        f"{min(single_us):,.1f}-{max(single_us):,.1f} us (rightmost column), *still* next to identical "
        "between configs and roughly "
        f"{min(single_vs_batch_ratio):,.0f}-{max(single_vs_batch_ratio):,.0f}x the batch-amortized cost - "
        "sklearn's ensemble `predict_proba` routes through `joblib.Parallel` internally regardless of "
        "n_jobs, so a large slice of Python/joblib call overhead survives even with parallelism "
        "disabled. This number is still reported since it's real - calling this model from Python one "
        "window at a time does cost that much - but it does not answer \"will this run on a "
        "microcontroller\", since no embedded port would carry Python or joblib.",
        "",
        "**Raw traversal (leftmost timing columns)** walks each tree's raw `feature`/`threshold`/"
        "`children_left`/`children_right` arrays directly with plain comparisons - the shape a hand-"
        "ported embedded implementation would take - bypassing sklearn's input validation and joblib "
        "entirely. Checked against sklearn's own `predict_proba` on the same row before timing "
        "(assert, not just eyeballed) to confirm it computes the identical result, not just a faster "
        "wrong one. An earlier version of this loop re-accessed `tree_.children_left` etc. as Cython-"
        "object attributes on every node visit and measured 21ms; caching them as local arrays once "
        f"per tree (still pure Python, just fewer attribute lookups) dropped that to "
        f"~{min(raw_us) / 1000:.1f}-{max(raw_us) / 1000:.1f}ms - a 2.6x win from removing lookup "
        "overhead alone, with the actual arithmetic unchanged.",
        "",
        "**None of the single-window numbers are the right proxy for embedded feasibility, including "
        "the optimized raw-traversal one.** sklearn n_jobs=1 and raw Python traversal sit within a "
        f"factor of {max(single_us) / min(raw_us):.0f} of each other "
        f"(~{min(raw_us) / 1000:.0f}-{max(single_us) / 1000:.0f}ms) while batch throughput is "
        f"{min(batch_us):.1f}-{max(batch_us):.1f} us - roughly a "
        f"{min(raw_vs_batch_ratio):,.0f}-{max(single_vs_batch_ratio):,.0f}x gap between \"any "
        "pure-Python call, however written\" and \"the same computation with per-call Python/numpy "
        "overhead amortized away\". That gap is Python interpreter cost (function calls, numpy scalar "
        "boxing, attribute resolution), not real arithmetic, and no amount of further Python-loop "
        f"optimization removes it - a compiled C/C++ port pays none of it. **Batch throughput "
        f"({min(batch_us):.1f}-{max(batch_us):.1f} us/window) is therefore the more honest proxy for "
        "compiled-code inference cost**, not the raw-traversal number, even though raw traversal is "
        "structurally closer to what an embedded port's logic would look like. Raw traversal's real "
        "value here is structural, not its absolute timing: it shows total work scales with tree depth "
        f"(~log of node count) rather than raw node count or input width - {b}'s per-window cost is "
        f"within {raw_pct_diff:.1f}% of {a}'s ({by_config[b]['per_window_raw_median_us']:.1f} vs "
        f"{by_config[a]['per_window_raw_median_us']:.1f} us) despite having "
        f"~{node_ratio:.1f}x the average nodes/tree "
        f"({by_config[b]['avg_nodes_per_tree']:.0f} vs {by_config[a]['avg_nodes_per_tree']:.0f}), "
        "consistent with binary-tree traversal cost growing logarithmically, not linearly, with node "
        "count. Both figures are measured on the training/evaluation machine, not embedded hardware, "
        "and account for no fixed-point arithmetic, cache effects, or clock-speed differences - useful "
        "for relative comparison across configs and as evidence the compute itself is small, not an "
        "absolute microcontroller-cycle estimate.",
        "",
        "**Reporting guidance, confirmed with project owner 2026-08-17:** cite batch throughput "
        f"({min(batch_us):.1f}-{max(batch_us):.1f} us/window) as the compiled/embedded-implementation "
        f"proxy, explicitly framed as a proxy (not a measurement on target hardware), paired with the "
        f"sklearn single-call n_jobs=1 number (~{min(single_us) / 1000:.0f}-{max(single_us) / 1000:.0f}ms/window) "
        "as \"what naive Python deployment costs today\" - the gap between the two is itself the "
        "finding (a compiled inference path is needed for real-time use; out-of-the-box sklearn is not "
        "sufficient). Pickled model size "
        f"({size_summary}) belongs in "
        "Limitations as a real storage constraint for a glasses form factor, not in the main results. "
        f"The {b} deeper-trees-despite-fewer-features finding "
        f"({by_config[b]['avg_nodes_per_tree']:.0f} vs {by_config[a]['avg_nodes_per_tree']:.0f} avg "
        "nodes/tree) is worth one line in Results, not a subsection.",
        "",
        "Fit/prep time and training-set size included for context (not a deployment-time cost):",
        "",
        "| config | fit rows (undersampled) | prep time (s) | RF fit time (s) |",
        "|---|---|---|---|",
    ]
    for r in results:
        lines.append(f"| {r['config']} | {r['n_fit_rows']:,} | {r['prep_sec']:.1f} | {r['fit_sec']:.1f} |")

    OUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
