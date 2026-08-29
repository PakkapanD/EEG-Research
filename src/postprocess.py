"""Threshold -> candidate events: the pre-scoring half of spec 2 section 6.

Turns raw per-window probabilities (output/04_predictions/) into candidate
seizure events per recording: threshold -> smoothing (k<=1, a no-op at the
locked value, kept explicit rather than special-cased away) -> merge nearby
positive stretches -> drop stretches shorter than min_event_duration -> split
stretches longer than the SzCORE 5-minute ceiling. This is entirely our own
pre-scoring step, distinct from the merge/duration rules built into SzCORE's
and Ali's own scoring definitions (evaluate.py) - CLAUDE.md's own note to keep
the two layers separate applies here.

Operates only on on_uniform_grid rows (spec 2's evaluation timeline - dense
rows are training-only and excluded here as everywhere else downstream of
training).
"""

import numpy as np
import pandas as pd

GRID_STEP_SEC = 4.0
MAX_EVENT_DURATION_SEC = 300.0  # SzCORE's own ceiling, not a free parameter


def positive_runs_to_events(window_starts_sorted, merge_gap_sec, min_event_duration_sec,
                             max_event_duration_sec=MAX_EVENT_DURATION_SEC,
                             window_sec=GRID_STEP_SEC):
    """window_starts_sorted: sorted array of window_start_sec for POSITIVE grid
    windows in one file (already thresholded). Returns [(start_sec, end_sec), ...].

    Smoothing at k<=1 is a no-op: every positive grid window is already an
    eligible 4s stretch on its own, so there is no run-length gate here - only
    merge_gap (bridging near stretches) and min_event_duration (dropping short
    ones) do any filtering, per the locked k=1 ceiling (CLAUDE.md section 12).
    """
    if len(window_starts_sorted) == 0:
        return []

    # Each positive window is an interval [start, start+window_sec). Merge two
    # such intervals if the gap between them is <= merge_gap_sec.
    starts = window_starts_sorted
    ends = starts + window_sec

    merged = []
    cur_start, cur_end = starts[0], ends[0]
    for i in range(1, len(starts)):
        gap = starts[i] - cur_end
        if gap <= merge_gap_sec:
            cur_end = max(cur_end, ends[i])
        else:
            merged.append((cur_start, cur_end))
            cur_start, cur_end = starts[i], ends[i]
    merged.append((cur_start, cur_end))

    kept = [(s, e) for s, e in merged if (e - s) >= min_event_duration_sec]

    split = []
    for s, e in kept:
        while e - s > max_event_duration_sec:
            split.append((s, s + max_event_duration_sec))
            s += max_event_duration_sec
        split.append((s, e))
    return split


def percentile_to_threshold(probs, fraction_flagged):
    """Absolute probability threshold flagging at least
    round(fraction_flagged * len(probs)) windows as positive - chosen as an
    actual observed value (the k-th largest, via np.partition), never an
    interpolated one.

    np.quantile(probs, 1 - fraction_flagged) looks equivalent but isn't safe
    here: for a fraction very close to 0 (our operating points routinely flag
    <0.1% of windows), linear interpolation can land one ULP *above* the true
    maximum observed probability. `prob >= threshold` then matches nothing -
    zero flagged windows instead of the intended k - and the miss is silent
    (no error, just a wrong answer). Caught for real on Full-18/chb22/
    realtime_alert: np.quantile returned 0.9966666666666668 against an actual
    max of 0.9966666666666667, undercounting a fold from sens=1.0 to sens=0.0.
    Picking an observed order statistic instead makes undershoot structurally
    impossible.
    """
    n = len(probs)
    if n == 0:
        return float("inf")
    if fraction_flagged <= 0:
        return float(np.max(probs)) + 1e-9
    k = max(1, min(n, round(fraction_flagged * n)))
    return float(np.partition(probs, -k)[-k])


def make_candidate_events(df, threshold, merge_gap_sec, min_event_duration_sec,
                           max_event_duration_sec=MAX_EVENT_DURATION_SEC):
    """df: predictions for one split (tuning or test) of one fold, already
    filtered to on_uniform_grid rows - subject_id, filename, window_start_sec,
    prob columns required.

    Returns {(subject_id, filename): [(start_sec, end_sec), ...]} - every file
    present in df gets an entry, empty list if no candidate survives.
    """
    events_by_file = {}
    positive = df[df["prob"] >= threshold]
    grouped = positive.groupby(["subject_id", "filename"])["window_start_sec"]
    for key in df[["subject_id", "filename"]].drop_duplicates().itertuples(index=False):
        k = (key.subject_id, key.filename)
        events_by_file[k] = []
    for key, starts in grouped:
        window_starts_sorted = np.sort(starts.to_numpy())
        events_by_file[key] = positive_runs_to_events(
            window_starts_sorted, merge_gap_sec, min_event_duration_sec, max_event_duration_sec
        )
    return events_by_file
