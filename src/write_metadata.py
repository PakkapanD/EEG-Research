"""Writes output/metadata.json. Run after (or independently of) the feature
extraction pipeline - this only records parameters/config, not run totals
(those go in 01_dataset_summary.md, generated from the actual output).

Every deviation from the locked parameters in CLAUDE.md section 7 must appear
here with its reason, its approval date and who approved it (CLAUDE.md section
11): a silent deviation that nobody notices while writing the Methods is the
worst outcome this project can produce. There are currently three - D1, D2, D3
below - plus one performance rewrite that changed no feature values.
"""

import json
import platform
import sys
from pathlib import Path

import antropy
import mne
import numba
import numpy as np
import pandas as pd
import scipy
import sklearn

from audit import REQUIRED_CHANNELS, GLASS7_DEPENDENCY
from features import BANDS, AZC_THRESHOLDS_UV, FEATURE_NAMES, N_FEATURES
from preprocess import (
    FS, WINDOW_SEC, LABEL_OVERLAP_FRACTION,
    GRID_STEP_SEC, DENSE_STEP_SEC, DENSE_MARGIN_SEC,
    BANDPASS, BUTTER_ORDER, NOTCH_FREQ, NOTCH_Q,
    SUBJECT_MERGE_MAP, DOUBLE_BANANA_CHANNELS,
    tier_a_files, AUDIT_CSV,
)

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"

RANDOM_STATE = 42

# D1 - approved 2026-08-14.
SUBJECT_SET_DEVIATION = {
    "what": "Files are admitted individually (Tier A) instead of admitting only subjects whose every "
            "file carries all 21 derivations. Tier A = all 18 double-banana derivations present and the "
            "file is not one of chb12's three monopolar recordings. Configurations that need "
            "T7-FT9/FT9-FT10/FT10-T8 (only Glass-7) are evaluated on the Tier B subset of those files; "
            "every other configuration uses the full Tier A base.",
    "why": "The previous rule - impose the Glass-7 channel requirement on every configuration so that all "
           "configurations share one subject base (CLAUDE.md section 8) - excluded 7 of 23 subjects and 81 "
           "of 181 seizures (45%). CLAUDE.md section 11 requires stopping and asking above 3 excluded "
           "subjects. The cost of the change is that Glass-7 is reported against a slightly smaller base "
           "(640 files / 177 seizures) than the other configurations (668 files / 181 seizures); this must "
           "be stated wherever the two are compared, and the Best-n vs Glass-n curves that carry the "
           "study's argument all sit on the identical 181-seizure base.",
    "who_approved": "Project owner, 2026-08-14, in an interactive session with Claude Code.",
    "deviates_from": "CLAUDE.md section 8: 'every configuration should be evaluated on the same subject base'.",
    "must_be_stated_in_paper": True,
}

# D2 - approved 2026-08-14.
WINDOWING_DEVIATION = {
    "what": f"Windows are taken on a uniform {GRID_STEP_SEC:g} s grid (contiguous, non-overlapping, 100% "
            f"temporal coverage) instead of the locked {DENSE_STEP_SEC:g} s step, plus a {DENSE_STEP_SEC:g} s "
            f"dense schedule inside seizure_start - {DENSE_MARGIN_SEC:g} s to seizure_end + {DENSE_MARGIN_SEC:g} s. "
            "Every row carries on_uniform_grid; the dense rows exist only to give training folds enough "
            "positive windows and are excluded from every evaluation.",
    "why": "The chosen schedule is 907,755 windows and took 13.34 h wall (measured, 12 workers on a 6P+8E "
           "core laptop, 666 files, 0 failures) for 9.1 GB of feature parquet. The locked 0.5 s step over "
           "the same Tier A base is 6,802,824 windows - 7.5x more - which extrapolates to ~100 h and "
           "62.6 GB, against ceilings of 12 h and 20 GB in CLAUDE.md section 11. A 4 s grid with a 4 s "
           "window is the coarsest schedule that still covers 100% of the recordings, which is what keeps "
           "recorded hours a valid denominator for false alarms per day (rule 5.2).",
    "measured_runtime": {
        "wall_hours": 13.34,
        "windows": 905675,
        "files": 666,
        "failures": 0,
        "workers": 12,
        "mean_windows_per_sec": 18.9,
        "note": "Throughput was strongly bimodal: ~8.5 windows/s over the 86 recordings longer than 2.5 h "
                "and ~32 windows/s over the 552 recordings of about 1 h. The cause is I/O, not compute. To "
                "keep memory bounded, each recording is read one channel at a time, so a file is read 21 "
                "times; 12 workers on 4-hour files hold ~7.2 GB of EDF against a 15.7 GB machine and every "
                "re-read reaches the disk, while 1-hour files stay in the page cache. Anyone re-estimating "
                "this pipeline must weight by file duration - a probe on a 1-hour file underestimates the "
                "long recordings by about 4x.",
    },
    "runtime_estimate_history": "Two earlier estimates for this schedule were wrong and must not be quoted. "
           "~14 h projected wall time from the single-core cost (19.1 ms per channel per window) scaled by a "
           "7.4x parallel speedup measured on the entropy kernel in isolation, which the full pipeline does "
           "not reach. ~29.6 h extrapolated from the first 12 completed files, which were the longest "
           "recordings in the study and the slowest per window by 4x. The measured figure is 13.34 h.",
    "evaluation_rule": "All sensitivity / false-alarm figures are computed on on_uniform_grid == True rows "
                       "only, so the evaluation timeline is uniform and no region of the recording is "
                       "over-represented. Training may use every row.",
    "cost": "Post-processing at grid resolution is quantized to 4 s. The shortest seizure in the Tier A base "
            "is 6 s and every one of the 181 seizures gets at least 2 grid windows labelled positive "
            "(median 12), so a 'k consecutive positive windows' rule is capped at k=1 before short seizures "
            "become undetectable by construction (CLAUDE.md section 12). Finer post-processing sweeps are "
            "possible only inside the dense zone and must be reported as a secondary analysis.",
    "who_approved": "Project owner, 2026-08-14, in an interactive session with Claude Code.",
    "deviates_from": "CLAUDE.md section 7: step 0.5 s (87.5% overlap).",
    "must_be_stated_in_paper": True,
}

# D3 - originally taken 2026-08-12, re-confirmed 2026-08-14 with corrected numbers.
FEATURE_SCOPE_DEVIATION = {
    "what": "The 9 entropyProfiled_* features (entropyProfiled_total/average/maximum/minimum/median/"
            "standardDeviation/variance/kurtosis/skewness_sampleEntropy) are dropped from the 92-feature "
            f"DIHC_FeatureManager-equivalent library. 83 of 92 DIHC features + all 6 AZC features are "
            f"extracted ({N_FEATURES} features x {len(REQUIRED_CHANNELS)} channels = "
            f"{N_FEATURES * len(REQUIRED_CHANNELS)} feature columns).",
    "why": "Benchmarked on real EEG (not synthetic data): the entropy-profile computation costs ~496 ms per "
           "channel per window, against 19.1 ms for all 89 retained features combined. It would roughly "
           "triple the extraction cost even after the same kind of rewrite the other O(N^2) entropies "
           "received, and would add a second bespoke kernel to audit for fidelity to the upstream source.",
    "correction_to_earlier_record": "The 2026-08-12 version of this entry justified the drop with '~68 days "
            "for all 92 features' and '~1 ms for the other 83 features combined', and predicted ~2.4 days "
            "for the 83. Both were wrong: the 83 features measured 100.3 ms per channel per window, and the "
            "run they predicted at 2.4 days was on track for ~17.5 days when it was killed. The decision "
            "stands on the corrected numbers above; the superseded figures must not be quoted in the paper.",
    "who_approved": "Project owner, 2026-08-12; re-confirmed 2026-08-14 after the numbers were corrected.",
    "deviates_from": "The preprocessing spec's 'extract all 92, no selection at this stage'.",
    "must_be_stated_in_paper": True,
}

# Not a deviation - no feature value changed - but it explains why features.py
# looks the way it does and where the evidence for that claim lives.
PERFORMANCE_OPTIMIZATION = {
    "what": "features.py was rewritten for speed: 100.3 -> 19.1 ms per channel per window (5.3x), measured "
            "on real CHB-MIT data on a single core. Three changes account for almost all of it: "
            "(a) approximateEntropy, fuzzyEntropy and distributionEntropy now share one pass over the "
            "pairwise Chebyshev distances of the delay embedding instead of three separate scipy pdist "
            "calls (63.5 -> 17.0 ms); (b) the AZC Douglas-Peucker approximation is numba-compiled "
            "(19.3 -> 0.11 ms); (c) the frequency-domain block computes moments directly instead of through "
            "scipy.stats and reuses its Welch PSD for spectral entropy (16.4 -> 4.5 ms).",
    "verification": "The pre-optimization implementation is frozen verbatim in src/features_reference.py and "
                    "src/validate_features.py recomputes all 89 features both ways on 252 real filtered "
                    "CHB-MIT windows spanning 3 subjects, ictal and interictal. Result: 86 of 89 features "
                    "bit-identical; the other 3 (fuzzyEntropy, approximateEntropy, fd_kurtosis) agree to a "
                    "largest relative difference of 6.7e-14, i.e. floating-point accumulation order only. "
                    "See output/00c_feature_parity.md.",
    "parallel_scaling_caveat": "The 19.1 ms figure is single-core and unloaded. Running 12 workers, each "
                               "costs ~68 ms per channel per window, so aggregate throughput is ~3.4x a "
                               "single core rather than ~12x. The fused kernel materialises a 4 MB distance "
                               "buffer per window for distributionEntropy's histogram; 12 of those exceed "
                               "the machine's 24 MB shared L3. Anyone reporting per-window cost must say "
                               "which of the two numbers they mean.",
    "no_feature_definition_changed": True,
}

DEGENERATE_FEATURES = [
    {
        "feature": "sampleEntropy",
        "issue": "DIHC_FeatureManager only computes real sample entropy for len(window) >= 5000 samples "
                 "(antropy.sample_entropy is O(N^2) and the source guards against it). Our windows are "
                 "1024 samples (4s @ 256Hz), so this feature is identically 0.0 for every window.",
        "action": "Kept as 0.0 (faithful to source), not removed.",
    },
    {
        "feature": "positiveToNegativePeakRatio",
        "issue": "Source computes len(scipy.signal.find_peaks(x)) / len(scipy.signal.find_peaks(-x)). "
                 "find_peaks() returns a 2-tuple (indices, properties); len() of that tuple is always 2 "
                 "regardless of input, so this feature is identically 1.0 for every window. Almost "
                 "certainly an upstream bug (missing [0] indexing into the peak-index array).",
        "action": "Reproduced faithfully. Emitted as the constant 1.0 it provably is, rather than by calling "
                  "find_peaks twice per window to rediscover it; verified identical in 00c_feature_parity.md.",
    },
    {
        "feature": "lempelZivComplexity",
        "issue": "antropy.lziv_complexity casts float input to uint32 without binarizing first (documented "
                 "antropy behaviour, not a bug in antropy - the caller is expected to binarize). DIHC calls "
                 "it directly on continuous uV amplitudes; negative values wrap around under the uint32 "
                 "cast. The result is deterministic given the input but does not measure the intended "
                 "complexity notion.",
        "action": "Reproduced faithfully (ant.lziv_complexity(x) on raw amplitudes), not fixed.",
    },
    {
        "feature": "shannonEntropy",
        "issue": "Computed via collections.Counter on raw continuous amplitude values (source code), which "
                 "are almost all unique per window -> the resulting distribution is close to uniform over N "
                 "samples regardless of signal content, so this feature is close to a constant "
                 "log2(N) = log2(1024) = 10 for nearly every window.",
        "action": "Reproduced faithfully, not fixed.",
    },
    {
        "feature": "fisherInfo",
        "issue": "DIHC wraps pyeeg.fisher_info. pyeeg's source for fisher_info is -sum(W*log(W)) on the "
                 "normalized SVD spectrum of the delay embedding (tau=1, m=2) - mathematically identical to "
                 "pyeeg's own svd_entropy formula. Expected to be highly correlated with "
                 "singularValueDecompositionEntropy (which uses antropy's svd_entropy, same mathematical "
                 "form, possibly different order/delay defaults).",
        "action": "Reproduced the literal pyeeg formula faithfully. Flag for possible redundancy in feature "
                  "selection results.",
    },
    {
        "feature": "fd_*_other / fd_bandPower_other",
        "issue": "This project's bandpass filter is 0.5-40 Hz. The 'other' band (30-128 Hz, standing in for "
                 "DIHC's own default gamma band of 31-100 Hz) sits almost entirely outside the filter "
                 "passband, so these 9 features are expected to be near-zero by construction, not by signal "
                 "content. Anticipated by CLAUDE.md section 7.",
        "action": "Kept (not removed) per CLAUDE.md instruction; report near-zero values.",
    },
]


def file_admission():
    """Which files are in, which are out, and the counts for both bases (D1)."""
    audit = pd.read_csv(AUDIT_CSV)
    admitted = tier_a_files()
    admitted_names = set(admitted["filename"])

    excluded = {}
    for row in audit.itertuples(index=False):
        if row.filename in admitted_names:
            continue
        if row.is_monopolar:
            excluded[row.filename] = ("monopolar montage (Zanetti et al. 2022); auto-detected, none of the "
                                      "21 required bipolar derivations present")
        else:
            missing = [c for c in DOUBLE_BANANA_CHANNELS if not getattr(row, f"has_{c}")]
            excluded[row.filename] = f"missing double-banana derivation(s): {missing}"

    tier_b = admitted[admitted["has_glass7"]]
    return {
        "rule": "per file (D1), decided from output/00_channel_audit.csv - nothing hardcoded",
        "tier_a_definition": "all 18 double-banana derivations present and the file is not monopolar",
        "tier_b_definition": f"Tier A plus {GLASS7_DEPENDENCY}; required only by Glass-7",
        "tier_a": {
            "n_files": len(admitted),
            "n_subjects_after_merge": int(admitted["subject_id"].nunique()),
            "recording_hours": round(float(admitted["duration_sec"].sum()) / 3600, 2),
            "n_seizures": int(admitted["n_seizures"].sum()),
            "used_by": ["Full-18", "Best-7", "Best-4", "Best-2", "Glass-4", "Glass-2"],
        },
        "tier_b": {
            "n_files": len(tier_b),
            "n_subjects_after_merge": int(tier_b["subject_id"].nunique()),
            "recording_hours": round(float(tier_b["duration_sec"].sum()) / 3600, 2),
            "n_seizures": int(tier_b["n_seizures"].sum()),
            "used_by": ["Glass-7"],
        },
        "excluded_files": excluded,
        "n_excluded_files": len(excluded),
        "excluded_subjects": [],
    }


def build_metadata():
    admission = file_admission()
    admitted = tier_a_files()
    return {
        "generated_at": pd.Timestamp.now().isoformat(),
        "random_state": RANDOM_STATE,
        "sampling_rate_hz": FS,
        "filter": {
            "type": "Butterworth",
            "order_requested": BUTTER_ORDER,
            "order_actual_note": (
                "scipy.signal.butter(4, ..., btype='band') returns an order-2N=8 bandpass filter; "
                "sosfiltfilt applies it forward and backward, squaring the magnitude response again. "
                "The nominal 'order 4' describes the design request, not the realized filter order."
            ),
            "passband_hz": list(BANDPASS),
            "implementation": "scipy.signal.butter(output='sos') + scipy.signal.sosfiltfilt (zero-phase)",
            "notch_hz": NOTCH_FREQ,
            "notch_q": NOTCH_Q,
            "notch_implementation": "scipy.signal.iirnotch + scipy.signal.filtfilt",
            "notch_form_note": (
                "iirnotch only produces the (b, a) form. Rule 4's sos requirement targets the 0.5 Hz "
                "highpass edge (normalized frequency 0.0039), where (b, a) is numerically unstable; at "
                "60 Hz / 256 Hz the normalized frequency is 0.469 and (b, a) is well conditioned."
            ),
            "filtered_whole_recording_before_windowing": True,
            "filtered_one_channel_at_a_time": (
                "Each channel is read and filtered over its full length independently, then all of its "
                "windows are extracted, before the next channel is read. This is a memory layout change "
                "only - the filter still sees the entire recording, never a single window (rule 3)."
            ),
            "realtime_limitation": (
                "filtfilt (both bandpass and notch) is zero-phase and non-causal - it uses future "
                "samples. Not usable in a real-time/streaming device; a real device would need a causal "
                "one-directional IIR filter, which introduces phase distortion affecting detection "
                "latency. Record this in the paper's Limitations section."
            ),
        },
        "windowing": {
            "window_sec": WINDOW_SEC,
            "grid_step_sec": GRID_STEP_SEC,
            "grid_overlap_fraction": 1 - GRID_STEP_SEC / WINDOW_SEC,
            "dense_step_sec": DENSE_STEP_SEC,
            "dense_margin_sec": DENSE_MARGIN_SEC,
            "label_rule": (f"window labeled 1 if overlap with a seizure interval >= "
                           f"{LABEL_OVERLAP_FRACTION * 100:.0f}% of window duration"),
            "label_rule_locked_level": "3 - arbitrary choice, requires sensitivity analysis in spec 2",
            "deviation": WINDOWING_DEVIATION,
        },
        "welch_psd": {"nperseg": 512, "noverlap": 256, "frequency_resolution_hz": 256 / 512},
        "frequency_bands_hz": BANDS,
        "band_note": (
            "Project-locked bands (delta 1-4, theta 4-8, alpha 8-13, beta 13-30) are used in place of "
            "DIHC_FeatureManager's own defaults (delta 0-5, theta 5-8, alpha 8-14, beta 14-31) because "
            "the library defaults touch the filter's transition/stop bands (e.g. delta starting at 0 Hz "
            "is inside our 0.5 Hz highpass stopband). 'other' stands in for DIHC's gamma band and "
            "intentionally spans beyond our 40 Hz passband (30-128 Hz) - see feature docstring."
        ),
        "channels": {
            "required_21": REQUIRED_CHANNELS,
            "double_banana_18": DOUBLE_BANANA_CHANNELS,
            "dropped_duplicates_at_load_time": {
                "P7-T7": "polarity-flipped copy of T7-P7 (signal = -1 x original)",
                "T8-P8 (second occurrence)": "byte-identical duplicate of the first T8-P8; recorder configuration error",
            },
            "glass7_dependency_channels": GLASS7_DEPENDENCY,
            "missing_channel_policy": (
                "A required derivation absent from a file yields NaN for all of that channel's feature "
                "columns in that file - never 0 (CLAUDE.md section 9). Counts are reported in "
                "01_dataset_summary.md."
            ),
        },
        "features": {
            "dihc_library_source": "https://github.com/WWM-EMRAN/DIHC_FeatureManager (no PyPI package; reverse-engineered from source, no official version tag)",
            "dihc_features_extracted": 83,
            "dihc_features_in_full_library": 92,
            "azc_features": 6,
            "azc_thresholds_uv": AZC_THRESHOLDS_UV,
            "azc_source": "Zanetti et al. 2022 (J. Neural Eng.) - own reimplementation, no reference code found. "
                          "Method: raw zero-crossing count, plus Douglas-Peucker polygonal approximation "
                          "(vertical/amplitude distance criterion) at thresholds 16/32/64/128/256 uV, "
                          "each followed by a zero-crossing count on the approximated signal.",
            "total_features_per_channel": N_FEATURES,
            "total_feature_columns": N_FEATURES * len(REQUIRED_CHANNELS),
            "feature_names_per_channel": list(FEATURE_NAMES),
            "feature_selection": "NONE at this stage - raw features only, chi-squared selection happens inside each CV fold in spec 2 (not yet written)",
            "scaling_normalization": "NONE at this stage - to be fit inside each CV fold in spec 2",
            "feature_scope_deviation": FEATURE_SCOPE_DEVIATION,
            "performance_optimization": PERFORMANCE_OPTIMIZATION,
            "known_degenerate_or_redundant_features": DEGENERATE_FEATURES,
        },
        "subjects": {
            "subject_merge_map": SUBJECT_MERGE_MAP,
            "merge_reason": "chb21 is the same patient as chb01, recorded ~1.5 years later (Ali et al. 2024 note this but do not merge). Merging prevents patient leakage across LOSO folds.",
            "n_final_subjects_after_merge": int(admitted["subject_id"].nunique()),
            "final_subject_ids": sorted(admitted["subject_id"].unique()),
            "selection_deviation": SUBJECT_SET_DEVIATION,
        },
        "file_admission": admission,
        "software_versions": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit-learn": sklearn.__version__,
            "pandas": pd.__version__,
            "mne": mne.__version__,
            "antropy": antropy.__version__,
            "numba": numba.__version__,
        },
    }


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    meta = build_metadata()
    out_path = OUTPUT_DIR / "metadata.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"wrote {out_path}")
    print(f"  deviations recorded: D1 subject set, D2 windowing, D3 feature scope")


if __name__ == "__main__":
    main()
