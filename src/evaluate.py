"""SzCORE (primary) and Ali et al. 2024 (secondary) event scoring, spec 2
section 7. Uses `timescoring` (the reference SzCORE implementation, PyPI
package `timescoring`) directly for the scoring mechanics rather than
reimplementing tolerance/merge/split logic by hand - CLAUDE.md section 13's
"cite the original source, not something citing it" applies to code too.

Ali's rule reuses the same EventScoring engine with different parameters
(no tolerance, 10s merge, >=70% overlap) plus one extra step the library
doesn't have: discarding hypothesis events <=10s before scoring, done here.

Both scoring passes see the SAME candidate events out of postprocess.py
(spec 2 section 6's note: post-processing is one shared pre-scoring step,
SzCORE's and Ali's own merge/duration rules are scoring-time definitions
layered on top, not separate post-processing runs).
"""

import numpy as np
from timescoring.annotations import Annotation
from timescoring.scoring import EventScoring

from preprocess import parse_seizure_intervals, summary_path_for, tier_a_files

SZCORE_PARAMS = EventScoring.Parameters(
    toleranceStart=30, toleranceEnd=60, minOverlap=0,
    maxEventDuration=300, minDurationBetweenEvents=90,
)
ALI_MIN_EVENT_DURATION_SEC = 10  # Ali discards hyp events <=10s before scoring
ALI_PARAMS = EventScoring.Parameters(
    toleranceStart=0, toleranceEnd=0, minOverlap=0.70,
    maxEventDuration=300, minDurationBetweenEvents=10,
)


def file_durations():
    files = tier_a_files()
    return {(row.subject_id, row.filename): row.duration_sec for row in files.itertuples(index=False)}


def ground_truth_events():
    files = tier_a_files()
    cache = {}
    for row in files.itertuples(index=False):
        cache[(row.subject_id, row.filename)] = parse_seizure_intervals(
            summary_path_for(row.subject), row.filename
        )
    return cache


def score_fold(events_by_file, ref_by_file, durations):
    """Aggregates SzCORE and Ali scoring across every file in events_by_file.

    Returns {"szcore": {"tp":.., "fp":.., "ref_true":..},
             "ali":    {"tp":.., "fp":.., "ref_true":..}}
    ref_true can differ between the two rulesets even for identical raw
    ground truth: each ruleset merges close-together reference events with
    its own gap (90s SzCORE, 10s Ali) before counting, per the library.
    """
    totals = {"szcore": {"tp": 0, "fp": 0, "ref_true": 0},
              "ali": {"tp": 0, "fp": 0, "ref_true": 0}}

    for key, hyp_events in events_by_file.items():
        duration = durations.get(key)
        if duration is None:
            continue
        ref_events = ref_by_file.get(key, [])
        n_samples = max(1, round(duration))

        ref_ann = Annotation(ref_events, fs=1, numSamples=n_samples)
        hyp_ann = Annotation(hyp_events, fs=1, numSamples=n_samples)

        sc = EventScoring(ref_ann, hyp_ann, SZCORE_PARAMS)
        totals["szcore"]["tp"] += sc.tp
        totals["szcore"]["fp"] += sc.fp
        totals["szcore"]["ref_true"] += sc.refTrue

        ali_hyp_events = [(s, e) for s, e in hyp_events if (e - s) > ALI_MIN_EVENT_DURATION_SEC]
        ali_hyp_ann = Annotation(ali_hyp_events, fs=1, numSamples=n_samples)
        ac = EventScoring(ref_ann, ali_hyp_ann, ALI_PARAMS)
        totals["ali"]["tp"] += ac.tp
        totals["ali"]["fp"] += ac.fp
        totals["ali"]["ref_true"] += ac.refTrue

    return totals


def sensitivity(totals):
    return totals["tp"] / totals["ref_true"] if totals["ref_true"] else float("nan")


def fa_per_day(totals, recorded_hours):
    return totals["fp"] / (recorded_hours / 24) if recorded_hours > 0 else float("nan")
