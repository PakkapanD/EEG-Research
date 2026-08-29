"""Subject-wise CV fold assignment, shared by every config (spec 2 section 3).

Two schemes, both keyed on the merged subject_id (23 subjects, chb01+21 as one
entity - CLAUDE.md section 6.1):

  - 5-fold: greedy load-balanced partition on seizure count per subject, not on
    windows. StratifiedKFold on a seizure-count quartile bin was tried first
    and rejected: it only preserves the low/medium/high *proportion* per fold,
    not the summed total, so chb12 alone (27 of 181 seizures, the single
    largest subject) landed in one fold and produced fold totals of
    54/38/38/22/29 - a >2x spread that would inject fold-to-fold noise into
    the channel-ladder curve (spec 2 section 10.1, the project's main plot)
    unrelated to channel count. Greedy load-balancing (subjects sorted by
    seizure count descending, each assigned to the currently-lightest fold
    under a size cap of ceil(23/5)=5) directly minimizes that spread instead.
    Ties in seizure count are broken by a random_state=42 shuffle so equal-
    count subjects aren't always resolved in the same (e.g. alphabetical)
    order.
  - LOO: a fixed, seeded (random_state=42) permutation of the 23 subjects,
    independent of any subject's own characteristics. For a fold with
    held-out subject S, the tuning pool is the next TUNING_POOL_SIZE subjects
    after S in that ring (wrap-around), excluded from that fold's RF fit.
    Approved 2026-08-15: this avoids the leakage in "2 subjects nearest by
    seizure count to the held-out subject" (spec 2 section 12 item 2), where
    the tuning-set choice would depend on the very subject being held out.

Fold membership is identical for Tier A and Tier B (spec 2 section 3) - the
tiers differ in which files/windows each subject contributes, not in who's in
which fold.
"""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from audit import OUTPUT_DIR
from preprocess import tier_a_files

RANDOM_STATE = 42
N_FOLDS = 5
TUNING_POOL_SIZE = 3

# Recommended in spec 2 section 3; must be confirmed, not silently assumed.
LOO_CONFIGS = ["Full-18", "Glass-2", "Glass-7"]

OUT_PATH = OUTPUT_DIR / "02_cv_folds.json"


def subject_seizure_counts():
    files = tier_a_files()
    counts = files.groupby("subject_id")["n_seizures"].sum().astype(int)
    return counts.sort_index()


def five_fold_assignment(counts):
    """Greedy load-balanced partition by seizure count (LPT bin-packing): the
    subject with the most seizures goes to the currently lightest fold,
    breaking ties with a seeded shuffle, under a per-fold size cap so no fold
    is starved of subjects while another overflows.
    """
    rng = np.random.RandomState(RANDOM_STATE)
    subjects = counts.index.to_numpy().copy()
    rng.shuffle(subjects)  # tie-break order for equal seizure counts
    order = sorted(subjects, key=lambda s: -counts.loc[s])

    max_per_fold = math.ceil(len(order) / N_FOLDS)
    fold_totals = [0] * N_FOLDS
    fold_sizes = [0] * N_FOLDS
    fold_of = {}
    for subject in order:
        candidates = [f for f in range(N_FOLDS) if fold_sizes[f] < max_per_fold]
        fold = min(candidates, key=lambda f: fold_totals[f])
        fold_of[subject] = fold
        fold_totals[fold] += int(counts.loc[subject])
        fold_sizes[fold] += 1

    folds = {str(f): sorted(s for s, ff in fold_of.items() if ff == f) for f in range(N_FOLDS)}
    return folds, fold_of


def loo_assignment(subjects_sorted):
    rng = np.random.RandomState(RANDOM_STATE)
    order = list(rng.permutation(subjects_sorted))

    loo = {}
    n = len(order)
    for i, held_out in enumerate(order):
        tuning_pool = [order[(i + k) % n] for k in range(1, TUNING_POOL_SIZE + 1)]
        fit_subjects = sorted(s for s in order if s != held_out and s not in tuning_pool)
        loo[held_out] = {
            "held_out": held_out,
            "tuning_pool": tuning_pool,
            "fit_subjects": fit_subjects,
        }
    return order, loo


def main():
    counts = subject_seizure_counts()
    subjects_sorted = sorted(counts.index.tolist())
    assert len(subjects_sorted) == 23, f"expected 23 merged subjects, got {len(subjects_sorted)}"

    folds, fold_of = five_fold_assignment(counts)
    loo_order, loo = loo_assignment(subjects_sorted)

    fold_seizure_totals = {
        f: int(counts.loc[subs].sum()) for f, subs in folds.items()
    }

    out = {
        "random_state": RANDOM_STATE,
        "n_subjects": len(subjects_sorted),
        "subjects": subjects_sorted,
        "seizure_counts_tier_a": {s: int(counts.loc[s]) for s in subjects_sorted},
        "five_fold": {
            "method": "greedy load-balanced partition (LPT bin-packing) on each subject's "
                      "Tier A seizure count, random_state=42 tie-break shuffle, "
                      f"max {math.ceil(23 / N_FOLDS)} subjects/fold",
            "n_folds": N_FOLDS,
            "folds": folds,
            "fold_seizure_totals": fold_seizure_totals,
        },
        "loo": {
            "method": "fixed seeded permutation (random_state=42) of the 23 subjects; "
                      f"tuning pool for held-out S = next {TUNING_POOL_SIZE} subjects after S "
                      "in the ring, wrap-around, excluded from that fold's RF fit. Blind to "
                      "the held-out subject's own characteristics (spec 2 section 12 item 2).",
            "fixed_order": loo_order,
            "tuning_pool_size": TUNING_POOL_SIZE,
            "recommended_configs": LOO_CONFIGS,
            "recommended_configs_status": "proposed in spec 2 section 3, not yet confirmed",
            "folds": loo,
        },
    }

    tmp_path = OUT_PATH.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(OUT_PATH)

    fold_subject_counts = {f: len(s) for f, s in folds.items()}
    print(f"Wrote {OUT_PATH}")
    print(f"5-fold seizure totals per fold: {fold_seizure_totals}")
    print(f"5-fold subject counts per fold: {fold_subject_counts}")
    print(f"chb01+21 position in LOO fixed order: {loo_order.index('chb01+21')}")


if __name__ == "__main__":
    main()
