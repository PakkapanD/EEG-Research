"""Parity gate: the fast features.py must return the same values as the frozen
features_reference.py on real EEG.

features.py was rewritten for speed (4x) after the original cost estimate for
this study turned out to be wrong by a factor of seven. The rewrite is only
legitimate if it changed nothing else, so this script re-derives every feature
both ways on real CHB-MIT windows and fails loudly on any disagreement. Its
output, output/00c_feature_parity.md, is the evidence the Methods section can
point at instead of asserting "we optimized it and it's fine".

Synthetic signals are not used deliberately: several features here (the entropy
and complexity group especially) branch on properties real EEG has and white
noise does not, so a synthetic check would exercise the wrong paths.

Run: python validate_features.py     (exit code 0 = parity holds)
"""

import sys
import time
from pathlib import Path

import numpy as np

import features as fast
import features_reference as reference
from audit import REQUIRED_CHANNELS
from preprocess import (
    edf_path_for,
    filtered_channel,
    open_recording,
    parse_seizure_intervals,
    summary_path_for,
    tier_a_files,
    FS,
    WINDOW_SAMPLES,
)

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
REPORT_PATH = OUTPUT_DIR / "00c_feature_parity.md"

# A feature passes if |new - ref| <= RELATIVE_TOLERANCE * max(1, |ref|). The
# rewrites that reorder floating-point accumulation (the fused entropy kernel)
# were measured at ~1e-14; this leaves four orders of magnitude of headroom
# while still being far tighter than any difference that could move a result.
RELATIVE_TOLERANCE = 1e-10

N_FILES = 3
WINDOWS_PER_FILE = 4  # x 21 channels


def pick_files():
    """One seizure-bearing file from each of three well-separated subjects.

    Chosen from the audit rather than hardcoded so this keeps working if the
    admitted file set changes.
    """
    files = tier_a_files()
    with_seizures = files[files["n_seizures"] > 0]
    subjects = sorted(with_seizures["subject"].unique())
    chosen_subjects = [subjects[0], subjects[len(subjects) // 2], subjects[-1]]
    picked = []
    for subject in chosen_subjects[:N_FILES]:
        row = with_seizures[with_seizures["subject"] == subject].iloc[0]
        picked.append((row.subject, row.filename))
    return picked


def pick_window_starts(n_samples, seizure_intervals):
    """Half the windows inside a seizure, half spread over the background."""
    last_start = n_samples - WINDOW_SAMPLES
    ictal = []
    if seizure_intervals:
        start_sec, end_sec = seizure_intervals[0]
        span = int(end_sec * FS) - int(start_sec * FS) - WINDOW_SAMPLES
        for fraction in (0.25, 0.75):
            if span > 0:
                ictal.append(int(start_sec * FS) + int(span * fraction))
    background = [int(last_start * fraction) for fraction in (0.1, 0.6)]
    starts = (ictal + background)[:WINDOWS_PER_FILE]
    return [s for s in starts if 0 <= s <= last_start]


def collect_window_pairs():
    """(label, window) for every real window this check runs on."""
    windows = []
    for raw_subject, filename in pick_files():
        intervals = parse_seizure_intervals(summary_path_for(raw_subject), filename)
        raw, resolved = open_recording(edf_path_for(raw_subject, filename))
        starts = pick_window_starts(raw.n_times, intervals)
        for channel in REQUIRED_CHANNELS:
            if resolved[channel] is None:
                continue
            signal = filtered_channel(raw, resolved[channel])
            for start in starts:
                windows.append((
                    f"{filename}:{channel}:{start / FS:.1f}s",
                    np.ascontiguousarray(signal[start:start + WINDOW_SAMPLES]),
                ))
    return windows


def compare(windows):
    """Per-feature worst-case disagreement across all windows."""
    worst = {name: {"abs": 0.0, "rel": 0.0, "identical": True, "where": ""}
             for name in fast.FEATURE_NAMES}
    nan_disagreements = []

    for label, window in windows:
        new_feats = fast.extract_channel_features(window)
        ref_feats = reference.extract_channel_features(window)
        if set(new_feats) != set(ref_feats):
            raise SystemExit(
                "Feature name sets differ: "
                f"only in fast={sorted(set(new_feats) - set(ref_feats))}, "
                f"only in reference={sorted(set(ref_feats) - set(new_feats))}"
            )
        for name, ref_value in ref_feats.items():
            new_value = new_feats[name]
            if np.isnan(ref_value) or np.isnan(new_value):
                if not (np.isnan(ref_value) and np.isnan(new_value)):
                    nan_disagreements.append((name, label, ref_value, new_value))
                continue
            absolute = abs(new_value - ref_value)
            relative = absolute / max(1.0, abs(ref_value))
            entry = worst[name]
            if new_value != ref_value:
                entry["identical"] = False
            if relative > entry["rel"]:
                entry.update(abs=absolute, rel=relative, where=label)
    return worst, nan_disagreements


def write_report(worst, nan_disagreements, n_windows, failures, elapsed):
    identical = [n for n, e in worst.items() if e["identical"]]
    within = [n for n, e in worst.items() if not e["identical"]]

    lines = [
        "# Feature parity: fast `features.py` vs frozen `features_reference.py`",
        "",
        f"- Windows checked: {n_windows} (real CHB-MIT, filtered exactly as the pipeline does)",
        f"- Features per window: {fast.N_FEATURES}",
        f"- Pass rule: `|new - ref| <= {RELATIVE_TOLERANCE:g} * max(1, |ref|)`",
        f"- Bit-identical features: **{len(identical)} / {fast.N_FEATURES}**",
        f"- Within tolerance but not bit-identical: **{len(within)}**",
        f"- Failures: **{len(failures)}**",
        f"- NaN disagreements: **{len(nan_disagreements)}**",
        f"- Elapsed: {elapsed:.1f} s",
        "",
    ]

    if failures:
        lines += ["## FAILURES", "",
                  "| feature | max abs err | max rel err | worst window |",
                  "|---|---|---|---|"]
        for name in failures:
            e = worst[name]
            lines.append(f"| {name} | {e['abs']:.3e} | {e['rel']:.3e} | {e['where']} |")
        lines.append("")

    if nan_disagreements:
        lines += ["## NaN disagreements", "",
                  "| feature | window | reference | fast |", "|---|---|---|---|"]
        for name, label, ref_value, new_value in nan_disagreements[:50]:
            lines.append(f"| {name} | {label} | {ref_value} | {new_value} |")
        lines.append("")

    lines += [
        "## Features that are not bit-identical",
        "",
        "These are the ones whose floating-point accumulation order changed. All",
        "come from the fused Chebyshev-distance kernel or from computing moments",
        "directly instead of through `scipy.stats`; none is a change of definition.",
        "",
        "| feature | max abs err | max rel err | worst window |",
        "|---|---|---|---|",
    ]
    for name in sorted(within, key=lambda n: -worst[n]["rel"]):
        e = worst[name]
        lines.append(f"| {name} | {e['abs']:.3e} | {e['rel']:.3e} | {e['where']} |")
    if not within:
        lines.append("| _(none - every feature is bit-identical)_ | | | |")

    lines += [
        "",
        "## Bit-identical features",
        "",
        ", ".join(f"`{n}`" for n in identical) if identical else "_(none)_",
        "",
    ]

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main():
    started = time.time()

    if tuple(reference.extract_channel_features(np.linspace(-50, 50, WINDOW_SAMPLES))) != fast.FEATURE_NAMES:
        raise SystemExit("FEATURE_NAMES does not match the reference emission order")

    print("Loading real EEG windows...")
    windows = collect_window_pairs()
    print(f"Comparing {len(windows)} windows x {fast.N_FEATURES} features...")

    worst, nan_disagreements = compare(windows)
    failures = [n for n, e in worst.items() if e["rel"] > RELATIVE_TOLERANCE]

    elapsed = time.time() - started
    write_report(worst, nan_disagreements, len(windows), failures, elapsed)
    print(f"wrote {REPORT_PATH}")

    n_identical = sum(1 for e in worst.values() if e["identical"])
    worst_name = max(worst, key=lambda n: worst[n]["rel"])
    print(f"bit-identical: {n_identical}/{fast.N_FEATURES}; "
          f"largest relative difference: {worst[worst_name]['rel']:.3e} ({worst_name})")

    if failures or nan_disagreements:
        print(f"PARITY FAILED: {len(failures)} features over tolerance, "
              f"{len(nan_disagreements)} NaN disagreements")
        return 1
    print("PARITY OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
