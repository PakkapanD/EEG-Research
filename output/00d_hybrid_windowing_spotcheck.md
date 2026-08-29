# Hybrid windowing spot-check (spec 2 section 2ka)

Pilot run: fit=['chb01+21', 'chb03', 'chb24'], tune=chb08, test=['chb16', 'chb12', 'chb15', 'chb17']. Full-18 config, RF(n_estimators=300, class_weight=balanced), chi2 30 features/channel, threshold 0.200 tuned on chb08 (unbalanced) targeting sens>=0.80. This is NOT the real CV/training pipeline - single split, 8 of 23 subjects, no hyperparameter search. Its only purpose is validating the hybrid-windowing measurement, per docs/06-spec2-cv-training-postprocessing-evaluation-th.md section 2ka.

## Data-quality finding surfaced by this pilot (unrelated to windowing)

Non-finite (+-inf) feature cells found and patched to column median for this pilot only (NOT fixed at the source):

- chb17: 25343 cells

Root cause (chb17b_69.edf, ~1520-1540s, FP1-F7): a zero-variance (flat/saturated) channel segment makes kurtosis/skewness divide by std=0 and come out as +-inf instead of NaN. Per CLAUDE.md section 9, features that cannot be computed should be NaN and counted, never a silent non-finite value. This should be fixed in `features.py` (guard kurtosis/skewness/etc. against zero variance -> NaN) before the real CV/training pipeline runs, and the affected file(s) re-extracted.

## Check 1: FA/h, near-seizure grid zone vs far grid zone

- Near-seizure grid negatives: 1790 windows (1.99 h), 90 false alarms -> 45.25 FA/h
- Far grid negatives: 87985 windows (97.76 h), 4223 false alarms -> 43.20 FA/h
- Relative difference: 4.5% (PASS, within +-20%)

## Check 2: does the +-60s dense margin reach seizure onset AND offset

Computed directly from the window schedule against all 181 Tier A seizures (not just the pilot's 4 test subjects) - no model needed.

## Check 3: false-positive cluster length, dense (0.5s) vs grid (4s) resolution

- FP runs observed at native resolution inside near-seizure zones: 82
  - duration stats (s): min=0.5, median=2.2, max=27.0
  - runs shorter than one grid window (4s): 56 (68.3%)
- FP runs observed at grid-only resolution, same spans: 43

## Check 4: short seizures (<=2 grid windows) - grid-level detection

- Short seizures (<=2 grid windows) in test subjects: 6, detected (>=1 grid window predicted positive): 4
- Longer seizures (>2 grid windows) in test subjects: 54, detected: 32
- Detection rate: short=67%, long=59%
