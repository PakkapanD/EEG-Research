# Post-processing sensitivity analysis (spec 2 section 6)

Representative subset: [('Full-18', '5fold', 0), ('Full-18', 'loo', 'chb02'), ('Glass-7', '5fold', 0), ('Glass-7', 'loo', 'chb02'), ('Glass-2', '5fold', 0), ('Glass-2', 'loo', 'chb02')], model=rf. Smoothing fixed at k<=1 (the only value the shortest-seizure ceiling allows - CLAUDE.md section 12). Sweeps merge_gap x min_event_duration (7 x 3 = 21 combos), scored with the locked percentile-calibrated threshold method, event-level per-subject pass rate (primary metric, spec 2 open item raised 2026-08-16).

## Per-subject pass rate, pooled across the subset (SzCORE), by (merge_gap, min_event_duration)

| merge_gap (s) | min_event_duration (s) | n_subject-criterion instances | n_met | pct_met |
|---|---|---|---|---|
| 0 | 0 | 24 | 2 | 8.3% |
| 0 | 1 | 24 | 2 | 8.3% |
| 0 | 2 | 24 | 2 | 8.3% |
| 4 | 0 | 24 | 2 | 8.3% |
| 4 | 1 | 24 | 2 | 8.3% |
| 4 | 2 | 24 | 2 | 8.3% |
| 8 | 0 | 24 | 2 | 8.3% |
| 8 | 1 | 24 | 2 | 8.3% **<- locked default** |
| 8 | 2 | 24 | 2 | 8.3% |
| 12 | 0 | 24 | 2 | 8.3% |
| 12 | 1 | 24 | 2 | 8.3% |
| 12 | 2 | 24 | 2 | 8.3% |
| 16 | 0 | 24 | 2 | 8.3% |
| 16 | 1 | 24 | 2 | 8.3% |
| 16 | 2 | 24 | 2 | 8.3% |
| 20 | 0 | 24 | 2 | 8.3% |
| 20 | 1 | 24 | 2 | 8.3% |
| 20 | 2 | 24 | 2 | 8.3% |
| 30 | 0 | 24 | 2 | 8.3% |
| 30 | 1 | 24 | 2 | 8.3% |
| 30 | 2 | 24 | 2 | 8.3% |

Best combo in this subset: merge_gap=0s, min_event_duration=0s -> 8.3% pass rate. Locked default (merge_gap=8s, min_event_duration=1s, used for the main 06_results run) -> 8.3%.

## By config

### Full-18

min_event_duration (rows) x merge_gap (cols), pct pass rate:

| min_dur \ merge_gap | 0s | 4s | 8s | 12s | 16s | 20s | 30s |
|---|---|---|---|---|---|---|---|
| 0s | 12.5% | 12.5% | 12.5% | 12.5% | 12.5% | 12.5% | 12.5% |
| 1s | 12.5% | 12.5% | 12.5% | 12.5% | 12.5% | 12.5% | 12.5% |
| 2s | 12.5% | 12.5% | 12.5% | 12.5% | 12.5% | 12.5% | 12.5% |

### Glass-7

min_event_duration (rows) x merge_gap (cols), pct pass rate:

| min_dur \ merge_gap | 0s | 4s | 8s | 12s | 16s | 20s | 30s |
|---|---|---|---|---|---|---|---|
| 0s | 12.5% | 12.5% | 12.5% | 12.5% | 12.5% | 12.5% | 12.5% |
| 1s | 12.5% | 12.5% | 12.5% | 12.5% | 12.5% | 12.5% | 12.5% |
| 2s | 12.5% | 12.5% | 12.5% | 12.5% | 12.5% | 12.5% | 12.5% |

### Glass-2

min_event_duration (rows) x merge_gap (cols), pct pass rate:

| min_dur \ merge_gap | 0s | 4s | 8s | 12s | 16s | 20s | 30s |
|---|---|---|---|---|---|---|---|
| 0s | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| 1s | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| 2s | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |

## Robustness check: is SzCORE's flatness real, or is merge_gap/min_duration not being applied?

The SzCORE pass rate above is bit-identical across all 21 combos for every one of the 12
(config, scheme, fold, criterion) groups scored under the `szcore` rule - a flat line strong
enough to warrant checking whether merge_gap/min_duration are actually reaching the scorer at
all, rather than trusting the flatness at face value.

They are: the same 21-combo sweep scored under the `ali` rule (0s/0s tolerances, 70% overlap -
much less forgiving of exact event boundaries than SzCORE's 30s/60s onset/offset tolerance)
shows real variation in 3 of its 12 groups. Example - Full-18/LOO(chb02)/medication_titration/ali
flips 0.0% -> 100.0% at merge_gap=4s and stays there (non-monotonically dipping back to 0.0% at
merge_gap=8s before recovering at 12s+) as candidate-event merging changes which windows get
counted as one event vs several. This confirms the merge_gap/min_duration parameters are live and
do change scoring outcomes when the scoring rule is sensitive enough to notice - SzCORE's
generous tolerance windows are why *this* subset's SzCORE pass/fail calls don't move, not a
broken sweep.

**Update 2026-08-18:** this script had the same silent-denominator bug as A12
(`select_operating_points_percentile.py`, fixed 2026-08-17) - `pick_threshold()` returned `None`
whenever the tuning pool couldn't reach the FA ceiling on any grid threshold, which silently
zeroed out that (config, scheme, fold, criterion, rule) combo's subjects instead of scoring them.
Confirmed live in the previous version of this file: `Glass-2/LOO(chb02)/realtime_alert/szcore`
had `n_subjects=0` across all 21 combos. Fixed with the same `fallback_to_best_achievable=True`
parameter and rerun in full. Effect: pooled denominator corrected from 23 to **24**
subject-criterion instances (the recovered chb02 instance itself did not pass, `met=False`), so
the headline number moved from 8.7% to **8.3%** - both flatness findings above (SzCORE flat
across all 21 combos; Ali varies in exactly 3/12 groups, same groups, same example) are
unchanged. The locked default (merge_gap=8s, min_duration=1s) is unaffected by this correction.
