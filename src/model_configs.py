"""Channel-set configs and model/post-processing hyperparameters for spec 2
(docs/06-spec2-cv-training-postprocessing-evaluation-th.md). Written as a
Python module, not JSON, because two of the seven channel configs (Best-n)
are data-driven per fold, not a fixed list - train.py imports CONFIGS and
resolves Best-n at fit time. A companion output/03_model_configs.json is
still written for the static parts (hyperparameters, grids) so the report-
writing side has a plain-data file to read without importing this module.

Design decisions made here that the spec left open, both flagged for
objection rather than decided silently:

  - 5-fold tuning subset (spec 2 section 4's open question was scoped to LOO
    only - section 12 item 2 - and left the 5-fold case unstated). Reusing a
    per-subject ring for a whole fold-vs-fold split adds a mechanism the spec
    never asked for, so instead: tuning subjects for test fold i = fold
    (i+1) % 5 from cv_folds.py's five_fold assignment; fit subjects = the
    other 3 folds. Deterministic, reuses the existing fold structure, keeps
    the tuning subjects out of both the model fit and the test fold.
  - Best-n channel ranking pool = the 18 double-banana channels only (not the
    21 unique derivations), because Best-n is reported against Tier A
    alongside Full-18 (CLAUDE.md section 8, `01_dataset_summary.md` section
    1), and the three Glass-7-dependency channels are NaN outside Tier B.
    Ranked by a preliminary RF's feature_importances_ summed per channel,
    fit on the fold's undersampled fit-subject data only (CLAUDE.md rule 1 -
    channel ranking is a fit, so it never sees tuning/test windows).
"""

import json

from audit import OUTPUT_DIR
from preprocess import DOUBLE_BANANA_CHANNELS

OUT_PATH = OUTPUT_DIR / "03_model_configs.json"

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Channel configs (CLAUDE.md section 8)
# ---------------------------------------------------------------------------

GLASS_CHANNELS = {
    7: ["FP1-F7", "F7-T7", "T7-FT9", "FT9-FT10", "FT10-T8", "F8-T8", "FP2-F8"],
    4: ["FP1-F7", "F7-T7", "F8-T8", "FP2-F8"],
    2: ["F7-T7", "F8-T8"],
}

CONFIGS = {
    "Full-18": {
        "n_channels": 18,
        "channels": DOUBLE_BANANA_CHANNELS,
        "data_driven": False,
        "tier": "A",
    },
    "Glass-7": {
        "n_channels": 7,
        "channels": GLASS_CHANNELS[7],
        "data_driven": False,
        "tier": "B",
    },
    "Glass-4": {
        "n_channels": 4,
        "channels": GLASS_CHANNELS[4],
        "data_driven": False,
        "tier": "A",
    },
    "Glass-2": {
        "n_channels": 2,
        "channels": GLASS_CHANNELS[2],
        "data_driven": False,
        "tier": "A",
    },
    "Best-7": {"n_channels": 7, "channels": None, "data_driven": True, "tier": "A",
               "ranking_pool": DOUBLE_BANANA_CHANNELS},
    "Best-4": {"n_channels": 4, "channels": None, "data_driven": True, "tier": "A",
               "ranking_pool": DOUBLE_BANANA_CHANNELS},
    "Best-2": {"n_channels": 2, "channels": None, "data_driven": True, "tier": "A",
               "ranking_pool": DOUBLE_BANANA_CHANNELS},
}

LOO_CONFIGS = ["Full-18", "Glass-2", "Glass-7"]  # spec 2 section 3, confirmed/locked (decision A8); not extended to all 7 configs by design
MODELS = ["rf", "lr", "mlp"]

# ---------------------------------------------------------------------------
# Locked (spec 2 section 1 / CLAUDE.md section 7)
# ---------------------------------------------------------------------------

N_FEATURES_PER_CHANNEL_SELECTED = 30  # chi2, per channel, fit in-fold
UNDERSAMPLE_RATIO = 10  # majority:minority after undersampling, training fold only

# ---------------------------------------------------------------------------
# Models (spec 2 section 5: "RF (primary) - record the full hyperparameter
# table"; no hyperparameter search is specified, so these are fixed, reported
# values, not a grid - flagged for objection like the other open items).
# ---------------------------------------------------------------------------

MODEL_HYPERPARAMS = {
    "rf": {
        "n_estimators": 300,
        "max_depth": None,
        "min_samples_leaf": 1,
        "class_weight": "balanced",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    },
    "lr": {
        "penalty": "l2",
        "C": 1.0,
        "solver": "lbfgs",
        "max_iter": 1000,
        "class_weight": "balanced",
        "random_state": RANDOM_STATE,
    },
    "mlp": {
        "hidden_layer_sizes": [64, 32],
        "activation": "relu",
        "alpha": 1e-4,
        "max_iter": 200,
        "early_stopping": True,
        "n_iter_no_change": 10,
        "random_state": RANDOM_STATE,
        # sklearn's MLPClassifier has no class_weight param; class balance is
        # handled the same way as the other two models, by undersampling the
        # training fold (spec 2 section 1), not by a per-model mechanism.
    },
}

# ---------------------------------------------------------------------------
# Threshold sweep (spec 2 section 4 step 6): window-level, on the unbalanced
# tuning set, to trace the full ROC/PR curve. Fine enough near the low end
# where the two target operating points (CLAUDE.md section 14: sens>=60%/
# FA<=5/day, sens>=80%/FA<=2/day) are expected to sit, given the 1:307 class
# ratio on the evaluation timeline.
# ---------------------------------------------------------------------------

THRESHOLD_GRID = sorted(set(round(t, 4) for t in
                             [i / 10000 for i in range(1, 100)] +   # 0.0001-0.0099
                             [i / 1000 for i in range(1, 100)] +    # 0.001-0.099
                             [i / 100 for i in range(10, 100)]))    # 0.10-0.99

# ---------------------------------------------------------------------------
# Post-processing sweep (spec 2 section 6). k and min_event_duration ceilings
# are hard limits computed from real data (output/01_dataset_summary.md
# section 3), not free parameters.
# ---------------------------------------------------------------------------

POSTPROCESSING_GRID = {
    "smoothing_k": [1],  # capped at 1 - k=2 spans 8s, longer than the 6s shortest seizure
    "min_event_duration_sec": [0, 1, 2],  # must stay < 3s (half the shortest seizure)
    "merge_gap_sec": [0, 4, 8, 12, 16, 20, 28],  # 4s-grid-aligned; 28 = 7 grid steps (30 would behave identically at 4s spacing)
    "max_event_duration_sec": 300,  # SzCORE's own 5-minute split ceiling, not a free parameter
}

SUFFICIENCY_CRITERIA = {
    "medication_titration": {"sensitivity_min": 0.60, "fa_per_day_max": 5},
    "realtime_alert": {"sensitivity_min": 0.80, "fa_per_day_max": 2},
}


def main():
    out = {
        "random_state": RANDOM_STATE,
        "configs": {
            name: {k: v for k, v in cfg.items() if k != "ranking_pool"}
            for name, cfg in CONFIGS.items()
        },
        "loo_configs": LOO_CONFIGS,
        "loo_configs_status": "confirmed/locked - spec 2 section 3, decision A8 (not extended to all 7 configs by design)",
        "models": MODELS,
        "model_hyperparams": MODEL_HYPERPARAMS,
        "n_features_per_channel_selected": N_FEATURES_PER_CHANNEL_SELECTED,
        "undersample_ratio": UNDERSAMPLE_RATIO,
        "five_fold_tuning_rule": "tuning subjects for test fold i = fold (i+1) % 5; "
                                  "fit subjects = the other 3 folds. Not specified by "
                                  "spec 2 (section 4's open question was scoped to LOO "
                                  "only) - proposed here, flagged for objection.",
        "best_n_ranking": "preliminary RF (same hyperparams as model_hyperparams.rf) fit "
                           "on the fold's undersampled fit-subject data over all 18 "
                           "double-banana channels; feature_importances_ summed per "
                           "channel; top n taken. Fit only on fit-subjects, never "
                           "tuning/test (CLAUDE.md rule 1).",
        "threshold_grid_n": len(THRESHOLD_GRID),
        "threshold_grid_range": [min(THRESHOLD_GRID), max(THRESHOLD_GRID)],
        "postprocessing_grid": POSTPROCESSING_GRID,
        "sufficiency_criteria": SUFFICIENCY_CRITERIA,
    }
    tmp_path = OUT_PATH.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(OUT_PATH)
    print(f"Wrote {OUT_PATH}")
    print(f"Configs: {list(CONFIGS)}")
    print(f"Threshold grid: {len(THRESHOLD_GRID)} points, "
          f"{min(THRESHOLD_GRID)}-{max(THRESHOLD_GRID)}")


if __name__ == "__main__":
    main()
