"""Feature extraction: 83 of the 92 DIHC_FeatureManager-equivalent features + 6 AZC features.

SCOPE DEVIATION ("extract all 92, no selection at this stage"): the 9
entropyProfiled_* features are dropped, leaving 89 features per channel.
Re-confirmed by the project owner 2026-08-14 after the original cost estimate
was found to be wrong - see metadata.json "feature_scope_deviation" for the
corrected, measured numbers. This must be stated plainly in the paper's
Methods/Limitations, not hidden.

The 92 features and their exact formulas were reverse-engineered from the public
source of https://github.com/WWM-EMRAN/DIHC_FeatureManager (no PyPI package
exists). Confirmed against the library's own example output
(eeg_all_features_python.csv, 92 columns) and its source files
(DIHC_FeatureExtractor.py, DIHC_FeatureExtrantor_Helper.py, DIHC_EntropyProfile.py).
Most features wrap the `antropy` package exactly as DIHC does; a handful
(fuzzyEntropy, distributionEntropy, entropyProfiled_*, shannonEntropy,
renyiEntropy, fisherInfo) are DIHC's own code and are reimplemented here.

The 6 AZC features follow Zanetti et al. 2022: a raw zero-crossing count plus
five zero-crossing counts on a Douglas-Peucker polygonal approximation of the
window at thresholds 16/32/64/128/256 uV (their reported values). This is our
own implementation (no reference code was found) - see AZC_THRESHOLDS_UV below.


PERFORMANCE - why parts of this file look strange
-------------------------------------------------
The straightforward implementation costs 100.3 ms per channel per window on real
CHB-MIT data, which is ~132 CPU-hours for this study's window schedule. This file
is a 4x faster rewrite that must produce *identical* feature values; the original
is frozen verbatim in features_reference.py and validate_features.py checks every
feature of this file against it on real EEG windows (output/00c_feature_parity.md).

Three rewrites carry almost all of the speedup:

  1. approximateEntropy, fuzzyEntropy and distributionEntropy each walked all
     ~523k pairs of embedding vectors separately (63.5 ms combined, three
     scipy pdist allocations). _chebyshev_entropy_kernel() walks the pairs once
     and accumulates all three (17.0 ms), using the fact that the m=3 Chebyshev
     distance is max(d_m2, |x[i+2] - x[j+2]|) so it costs one extra comparison
     rather than a second distance matrix.

     Inside that kernel the fuzzy weight is written `u = d * inv_r; exp(-LOG2 * u * u)`
     and NOT `exp(-LOG2 * (d / r) ** 2)`. The `**` operator emits a pow() call
     that numba does not fold into a multiply: measured 36 ms vs 17 ms for the
     same result. Do not "simplify" this back.

  2. The AZC Douglas-Peucker pass was a Python-level recursion over numpy slices
     (19.3 ms). It is now njit with an explicit index stack (0.11 ms).

  3. fd_features called scipy.stats.kurtosis/skew 12 times per window at ~0.84 ms
     of dispatch overhead each (10.2 of its 16.8 ms) and recomputed Welch's PSD a
     second time inside antropy.spectral_entropy. Both are now computed directly
     from the single PSD, using the same formulas scipy/antropy use (including
     scipy's NaN-on-zero-variance behaviour, which is reproduced deliberately).

scipy.integrate.simpson is deliberately NOT reimplemented (2.95 ms/window): its
handling of an even number of intervals has changed across scipy versions and
matching it by hand would be a silent-divergence risk for a 3% gain.


Known degenerate / redundant features inherited faithfully from the upstream
library (documented, not "fixed" - see metadata.json and CLAUDE.md section 5.4
for the analogous gamma-band case):
  - sampleEntropy: the source only computes real sample entropy for
    len(window) >= 5000; our windows are 1024 samples, so this is always 0.0.
  - positiveToNegativePeakRatio: the source computes
    len(scipy.signal.find_peaks(x)) / len(scipy.signal.find_peaks(-x)) - i.e.
    len() of the 2-tuple find_peaks() returns, not len() of the peak-index
    array. This is always 2/2 = 1.0 regardless of input.
  - lempelZivComplexity: antropy.lziv_complexity casts float input to uint32
    without binarizing first (documented antropy behaviour). Negative uV
    amplitudes wrap around during that cast, so this feature reflects the
    wraparound pattern rather than a principled complexity measure. Still
    deterministic given the input, so kept faithfully.
  - shannonEntropy: computed via collections.Counter on raw continuous
    amplitude values, which are almost all unique -> the resulting
    distribution is close to uniform over N samples regardless of signal
    content, so this feature is close to a constant log2(N).
  - fisherInfo: DIHC wraps pyeeg.fisher_info, whose source
    (-sum(W*log(W)) on the normalized SVD spectrum of the delay embedding) is
    mathematically identical to pyeeg's own svd_entropy - i.e. this and
    singularValueDecompositionEntropy are expected to be highly correlated.
  - fd_*_other / fd_bandPower_other: this project's bandpass filter is
    0.5-40 Hz, so the "other"/gamma band (30-128 Hz, DIHC's own default is
    31-100 Hz) sits almost entirely outside the passband and is expected to be
    near-zero by construction (CLAUDE.md section 5.4).
"""

import warnings

import numpy as np
from numba import njit
from scipy import stats as sp_stats
from scipy.integrate import simpson
from scipy.signal import welch

import antropy as ant

FS = 256
NPERSEG = 512
NOVERLAP = 256
WINDOW_SAMPLES = 1024  # 4 s at 256 Hz (CLAUDE.md section 7)

# Project-locked band edges (CLAUDE.md section 7), used in place of DIHC's own
# defaults (alpha 8-14, beta 14-31, delta 0-5, theta 5-8) because those touch
# the filter's transition bands (delta starting at 0 Hz is inside our stopband).
# "other" stands in for DIHC's gamma band; it is intentionally left spanning
# beyond our passband (see module docstring).
BANDS = {
    "alpha": (8, 13),
    "beta": (13, 30),
    "delta": (1, 4),
    "theta": (4, 8),
    "other": (30, FS / 2),
}

AZC_THRESHOLDS_UV = [0, 16, 32, 64, 128, 256]  # Zanetti et al. 2022

# Embedding dimension and tolerance factor shared by approximateEntropy,
# fuzzyEntropy and distributionEntropy - all three use m=2, tau=1 and
# r = 0.2 * std(x), which is what makes the single fused pass possible.
ENTROPY_ORDER = 2
ENTROPY_R_FACTOR = 0.2
DISTRIBUTION_ENTROPY_BINS = 500  # DIHC's M

STAT_NAMES = (
    "maximum", "minimum", "mean", "median",
    "standardDeviation", "variance", "kurtosis", "skewness",
)

# Frequency axis of every Welch PSD in this project is fixed by (FS, NPERSEG),
# so the band masks never change - computing them once removes them from the
# per-window path.
PSD_FREQS = np.fft.rfftfreq(NPERSEG, 1.0 / FS)
BAND_SLICES = {
    band: (int(np.searchsorted(PSD_FREQS, lo, side="left")),
           int(np.searchsorted(PSD_FREQS, hi, side="left")))
    for band, (lo, hi) in BANDS.items()
}


# ---------------------------------------------------------------------------
# Shared statistics
# ---------------------------------------------------------------------------

def _stats(a):
    """The 8 STAT_NAMES for one array, in order.

    kurtosis/skewness reproduce scipy.stats.kurtosis(fisher=True, bias=True) and
    scipy.stats.skew(bias=True) including their NaN result when the variance is
    zero, which was verified against scipy directly. Calling scipy here instead
    costs 0.84 ms per call and this runs 6 times per window.

    The `m2 == 0` guard catches an *exactly* flat array, but a near-flat one
    (e.g. filtfilt's output on a saturated/disconnected channel, ~1e-70 rather
    than exactly 0) slips past it: m2 is nonzero, but m2 ** 2 underflows to
    exactly 0.0 in float64 before the division, so the ratio silently becomes
    +-inf (or 0/0 -> NaN) instead of a meaningful number. Seen for real on
    chb17b_69.edf ~1520-1540s (FP1-F7/P3-O1). Per CLAUDE.md section 9 a
    feature that cannot be meaningfully computed must be NaN and counted, not
    a silent +-inf flowing into the scaler/model - so the result is checked
    for finiteness after the fact rather than trusting the m2==0 pre-check
    alone.
    """
    mean = a.mean()
    dev = a - mean
    m2 = np.mean(dev ** 2)
    if m2 == 0:
        kurtosis = np.nan
        skewness = np.nan
    else:
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            kurtosis = np.mean(dev ** 4) / m2 ** 2 - 3.0
            skewness = np.mean(dev ** 3) / m2 ** 1.5
        if not np.isfinite(kurtosis):
            kurtosis = np.nan
        if not np.isfinite(skewness):
            skewness = np.nan
    return (a.max(), a.min(), mean, np.median(a),
            np.sqrt(m2), m2, kurtosis, skewness)


def _safe_ratio(numerator, denominator):
    try:
        return numerator / denominator
    except ZeroDivisionError:
        return 0.0


# ---------------------------------------------------------------------------
# Time-domain linear
# ---------------------------------------------------------------------------

def td_linear_features(x):
    feats = dict(zip(STAT_NAMES, _stats(x)))
    feats["numberOfZeroCrossing"] = ant.num_zerocross(x)
    feats["positiveToNegativeSampleRatio"] = _safe_ratio(
        np.sum(x >= 0), np.sum(x < 0)
    )
    # Faithful to the upstream bug: DIHC divides len() of the 2-tuple that
    # find_peaks() returns, not len() of the peak-index array, so the value is
    # 2/2 for every possible input. Kept as the constant it provably is rather
    # than paying 1.8 ms per window to rediscover it. See module docstring.
    feats["positiveToNegativePeakRatio"] = 1.0
    feats["meanAbsoluteValue"] = np.mean(np.abs(x))
    return feats


# ---------------------------------------------------------------------------
# Time-domain nonlinear entropy
# ---------------------------------------------------------------------------

@njit(cache=True)
def _chebyshev_entropy_kernel(x, r, d2_out):
    """One pass over all pairs of delay-embedding vectors (m=2 and m=3, tau=1).

    Returns (approximateEntropy, fuzzyEntropy) and fills d2_out with the m=2
    Chebyshev distances in pair order for distribution_entropy to histogram.

    approximateEntropy reproduces antropy.app_entropy, which counts neighbours
    with distance <= r inclusive of the self-match (sklearn KDTree.query_radius)
    and returns phi(m) - phi(m+1) with phi = mean(log(count / n_vectors)).

    fuzzyEntropy reproduces DIHC's own formulation: -log(mean(w_{m+1}) / mean(w_m))
    with w = exp(-ln2 * (d/r)^2) over all unordered pairs.
    """
    n = x.shape[0]
    n_m = n - 1          # number of m=2 embedding vectors
    n_m1 = n - 2         # number of m=3 embedding vectors

    count_m = np.ones(n_m)      # 1.0 = the self-match KDTree also counts
    count_m1 = np.ones(n_m1)
    weight_sum_m = 0.0
    weight_sum_m1 = 0.0

    log2_value = np.log(2.0)
    inv_r = 1.0 / r
    pair_index = 0

    for i in range(n_m):
        x_i0 = x[i]
        x_i1 = x[i + 1]
        for j in range(i + 1, n_m):
            diff0 = abs(x_i0 - x[j])
            diff1 = abs(x_i1 - x[j + 1])
            dist_m = diff0 if diff0 > diff1 else diff1

            d2_out[pair_index] = dist_m
            pair_index += 1

            if dist_m <= r:
                count_m[i] += 1.0
                count_m[j] += 1.0

            # See module docstring: u*u, never (d/r)**2.
            u = dist_m * inv_r
            weight_sum_m += np.exp(-log2_value * u * u)

            if i < n_m1 and j < n_m1:
                diff2 = abs(x[i + 2] - x[j + 2])
                dist_m1 = dist_m if dist_m > diff2 else diff2
                if dist_m1 <= r:
                    count_m1[i] += 1.0
                    count_m1[j] += 1.0
                u = dist_m1 * inv_r
                weight_sum_m1 += np.exp(-log2_value * u * u)

    phi_m = 0.0
    for i in range(n_m):
        phi_m += np.log(count_m[i] / n_m)
    phi_m1 = 0.0
    for i in range(n_m1):
        phi_m1 += np.log(count_m1[i] / n_m1)
    approximate_entropy = phi_m / n_m - phi_m1 / n_m1

    n_pairs_m = n_m * (n_m - 1) // 2
    n_pairs_m1 = n_m1 * (n_m1 - 1) // 2
    mean_m = weight_sum_m / n_pairs_m
    mean_m1 = weight_sum_m1 / n_pairs_m1
    if mean_m == 0.0 or mean_m1 == 0.0:
        fuzzy_entropy = 0.0
    else:
        fuzzy_entropy = -np.log(mean_m1 / mean_m)

    return approximate_entropy, fuzzy_entropy


def pair_buffer(window_samples=WINDOW_SAMPLES):
    """Scratch space for _chebyshev_entropy_kernel, allocated once per channel
    instead of once per window (4 MB at the locked 4 s window).
    """
    n_vectors = window_samples - (ENTROPY_ORDER - 1)
    return np.empty(n_vectors * (n_vectors - 1) // 2, dtype=np.float64)


def _distribution_entropy_from_distances(distances, bins=DISTRIBUTION_ENTROPY_BINS):
    min_distance = distances.min()
    max_distance = distances.max()
    if min_distance == max_distance:
        return 0.0
    histogram, _ = np.histogram(
        distances, bins=np.linspace(min_distance, max_distance, bins + 1)
    )
    prob = histogram / len(distances)
    prob = prob[prob > 0]
    return -np.sum(prob * np.log2(prob)) / np.log2(bins)


def fused_entropies(x, d2_buffer=None):
    """(approximateEntropy, fuzzyEntropy, distributionEntropy) in one pass.

    A constant window (std == 0) makes r == 0: every distance is then <= r so
    every count saturates and all three entropies are 0, which is also what the
    separate implementations return (antropy via log(1), DIHC via its own r == 0
    guard). Short-circuited here to keep the division out of the kernel.
    """
    if d2_buffer is None:
        d2_buffer = pair_buffer(len(x))
    r = ENTROPY_R_FACTOR * np.std(x)
    if r == 0:
        return 0.0, 0.0, 0.0
    approximate_entropy, fuzzy_entropy = _chebyshev_entropy_kernel(x, r, d2_buffer)
    return (approximate_entropy, fuzzy_entropy,
            _distribution_entropy_from_distances(d2_buffer))


def shannon_entropy(x, m=2):
    _, counts = np.unique(x, return_counts=True)
    dist = counts / counts.sum()
    return sp_stats.entropy(dist, base=m)


def renyi_entropy(x, alpha=2):
    if np.isclose(alpha, 1):
        p = x * np.log2(x, out=np.zeros_like(x), where=x > 0)
        return -np.sum(p)
    return (1.0 / (1.0 - alpha)) * np.log2(np.sum(x ** alpha))


def td_entropy_features(x, d2_buffer=None):
    feats = {}
    approximate_entropy, fuzzy_entropy, distribution_entropy = fused_entropies(x, d2_buffer)
    feats["approximateEntropy"] = approximate_entropy

    # DIHC only computes real sample entropy for len(x) >= 5000; our windows
    # are 1024 samples, so this is always 0.0 by the library's own design.
    feats["sampleEntropy"] = 0.0 if len(x) < 5000 else ant.sample_entropy(x)

    feats["permutationEntropy"] = ant.perm_entropy(x)
    feats["singularValueDecompositionEntropy"] = ant.svd_entropy(x)
    feats["fuzzyEntropy"] = fuzzy_entropy
    feats["distributionEntropy"] = distribution_entropy
    feats["shannonEntropy"] = shannon_entropy(x)
    feats["renyiEntropy"] = renyi_entropy(x)

    # entropyProfiled_* (9 features) intentionally dropped - see module
    # docstring and metadata.json "feature_scope_deviation".
    return feats


# ---------------------------------------------------------------------------
# Time-domain nonlinear complexity / fractal dimension
# ---------------------------------------------------------------------------

def _embed(x, m, tau=1):
    return np.lib.stride_tricks.sliding_window_view(x, m)[::tau]


def fisher_info(x, tau=1, m=2):
    """Literal port of pyeeg.fisher_info: -sum(W*log(W)) on the normalized SVD
    spectrum of the delay embedding. See module docstring for why this is
    mathematically identical to singularValueDecompositionEntropy.
    """
    try:
        emb = _embed(x, m, tau)
        w = np.linalg.svd(emb, compute_uv=False)
        w = w / w.sum()
        w = w[w > 0]
        return -np.sum(w * np.log(w))
    except (ZeroDivisionError, np.linalg.LinAlgError):
        return 0.0


def td_complexity_features(x):
    feats = {}
    # Faithful to DIHC: antropy.lziv_complexity casts floats to uint32 without
    # binarizing (see module docstring - negative uV values wrap around).
    feats["lempelZivComplexity"] = ant.lziv_complexity(x)
    mobility, complexity = ant.hjorth_params(x)
    feats["hjorthMobility"] = mobility
    feats["hjorthComplexity"] = complexity
    feats["fisherInfo"] = fisher_info(x)
    feats["petrosianFd"] = ant.petrosian_fd(x)
    feats["katzFd"] = ant.katz_fd(x)
    feats["higuchiFd"] = ant.higuchi_fd(x)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        feats["detrendedFluctuation"] = ant.detrended_fluctuation(x)
    return feats


# ---------------------------------------------------------------------------
# Frequency-domain
# ---------------------------------------------------------------------------

def _spectral_entropy_from_psd(psd):
    """antropy.spectral_entropy(method='welch', nperseg=NPERSEG, normalize=False)
    computed from a PSD we already have.

    antropy calls welch(x, sf, nperseg=nperseg) whose noverlap defaults to
    nperseg // 2 = NOVERLAP, so its PSD is bit-identical to ours and only the
    redundant second transform is skipped. log(x)/log(2) rather than log2(x)
    reproduces antropy._xlogx exactly, down to the last ulp.
    """
    psd_norm = psd / psd.sum()
    positive = psd_norm > 0
    xlogx = np.zeros(psd_norm.shape)
    xlogx[positive] = psd_norm[positive] * np.log(psd_norm[positive]) / np.log(2)
    return -xlogx.sum()


def fd_features(x, fs=FS):
    freqs, psd = welch(x, fs=fs, nperseg=NPERSEG, noverlap=NOVERLAP)
    feats = {}

    for name, value in zip(STAT_NAMES, _stats(psd)):
        feats[f"fd_{name}"] = value

    for band, (start, stop) in BAND_SLICES.items():
        band_psd = psd[start:stop]
        if band_psd.size == 0:
            for name in STAT_NAMES:
                feats[f"fd_{name}_{band}"] = np.nan
            feats[f"fd_bandPower_{band}"] = np.nan
            continue
        for name, value in zip(STAT_NAMES, _stats(band_psd)):
            feats[f"fd_{name}_{band}"] = value
        feats[f"fd_bandPower_{band}"] = simpson(band_psd, x=freqs[start:stop])

    feats["fd_bandPower"] = simpson(psd, x=freqs)
    feats["spectralEntropy"] = _spectral_entropy_from_psd(psd)
    return feats


# ---------------------------------------------------------------------------
# AZC (Zanetti et al. 2022)
# ---------------------------------------------------------------------------

@njit(cache=True)
def _rdp_keep_mask(y, epsilon):
    """Douglas-Peucker simplification using vertical (amplitude) distance from
    the straight-line interpolation, matching AZC's amplitude-threshold
    (uV) semantics. x-axis is the sample index.

    The pending segments are held in a preallocated index stack rather than a
    Python list; each split adds one kept point and at most two segments, so
    2*n slots can never overflow.
    """
    n = y.shape[0]
    keep = np.zeros(n, dtype=np.bool_)
    keep[0] = True
    keep[n - 1] = True

    stack_start = np.empty(2 * n, dtype=np.int64)
    stack_end = np.empty(2 * n, dtype=np.int64)
    stack_start[0] = 0
    stack_end[0] = n - 1
    top = 1

    while top > 0:
        top -= 1
        start = stack_start[top]
        end = stack_end[top]
        if end - start < 2:
            continue
        y_start = y[start]
        slope = (y[end] - y_start) / (end - start)
        max_distance = -1.0
        split = -1
        for t in range(start + 1, end):
            distance = abs(y[t] - (y_start + slope * (t - start)))
            if distance > max_distance:
                max_distance = distance
                split = t
        if max_distance > epsilon:
            keep[split] = True
            stack_start[top] = start
            stack_end[top] = split
            top += 1
            stack_start[top] = split
            stack_end[top] = end
            top += 1
    return keep


@njit(cache=True)
def _zero_crossings(y):
    """Sign changes with sign(0) treated as +1, matching np.sign() followed by
    replacing zeros with 1 in the original implementation.
    """
    count = 0
    previous = 1.0 if y[0] >= 0 else -1.0
    for i in range(1, y.shape[0]):
        current = 1.0 if y[i] >= 0 else -1.0
        if current != previous:
            count += 1
        previous = current
    return count


def azc_features(x):
    feats = {}
    for threshold in AZC_THRESHOLDS_UV:
        if threshold == 0:
            feats[f"azc_{threshold}"] = _zero_crossings(x)
        else:
            mask = _rdp_keep_mask(x, float(threshold))
            feats[f"azc_{threshold}"] = _zero_crossings(x[mask])
    return feats


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _build_feature_names():
    """The 89 feature names in the exact order extract_channel_features() emits
    them. Derived from the same constants the extractors loop over so the two
    can never drift apart; validate_features.py asserts the match.
    """
    names = list(STAT_NAMES) + [
        "numberOfZeroCrossing", "positiveToNegativeSampleRatio",
        "positiveToNegativePeakRatio", "meanAbsoluteValue",
        "approximateEntropy", "sampleEntropy", "permutationEntropy",
        "singularValueDecompositionEntropy", "fuzzyEntropy",
        "distributionEntropy", "shannonEntropy", "renyiEntropy",
        "lempelZivComplexity", "hjorthMobility", "hjorthComplexity",
        "fisherInfo", "petrosianFd", "katzFd", "higuchiFd",
        "detrendedFluctuation",
    ]
    names += [f"fd_{name}" for name in STAT_NAMES]
    for band in BANDS:
        names += [f"fd_{name}_{band}" for name in STAT_NAMES]
        names.append(f"fd_bandPower_{band}")
    names += ["fd_bandPower", "spectralEntropy"]
    names += [f"azc_{threshold}" for threshold in AZC_THRESHOLDS_UV]
    return tuple(names)


FEATURE_NAMES = _build_feature_names()
N_FEATURES = len(FEATURE_NAMES)  # 89


def extract_channel_features(x, d2_buffer=None):
    """All 89 features (83 DIHC-equivalent + 6 AZC) for one channel window.

    Final safety net: a handful of features besides kurtosis/skewness divide
    by a variance-derived quantity internally (renyiEntropy's log2 of a sum of
    squares, antropy's hjorth_params and detrended_fluctuation) and can come
    out +-inf on the same near-zero-variance windows described in _stats().
    We do not own those implementations, so rather than special-casing each
    one, any non-finite value that reaches here is converted to NaN - CLAUDE.md
    section 9: a feature that cannot be meaningfully computed is NaN, never a
    silent +-inf. This is what makes run_pipeline.py's existing n_nan_features
    count (np.isnan on the assembled array) catch these cases too.
    """
    x = np.ascontiguousarray(x, dtype=np.float64)
    feats = {}
    feats.update(td_linear_features(x))
    feats.update(td_entropy_features(x, d2_buffer))
    feats.update(td_complexity_features(x))
    feats.update(fd_features(x))
    feats.update(azc_features(x))
    for name, value in feats.items():
        if not np.isfinite(value):
            feats[name] = np.nan
    return feats


def channel_feature_matrix(signal, start_samples, window_samples=WINDOW_SAMPLES):
    """Features for every window of ONE channel: array (n_windows, N_FEATURES).

    Channel-major rather than window-major so only one channel's filtered signal
    and one pair buffer are ever resident - the window-major version held every
    channel of the recording plus a dict per window and exhausted memory on the
    4-hour files (see run_pipeline.py).
    """
    d2_buffer = pair_buffer(window_samples)
    out = np.empty((len(start_samples), N_FEATURES), dtype=np.float64)
    for row, start in enumerate(start_samples):
        window = np.ascontiguousarray(signal[start:start + window_samples], dtype=np.float64)
        feats = extract_channel_features(window, d2_buffer)
        for col, name in enumerate(FEATURE_NAMES):
            out[row, col] = feats[name]
    return out


def warm_jit():
    """Compile (or load from numba's on-disk cache) every njit kernel once.

    Called in the parent process before the worker pool starts: with spawn on
    Windows each worker would otherwise compile these itself, and simultaneous
    first-time writes to the same cache files are a known race.
    """
    x = np.linspace(-100.0, 100.0, WINDOW_SAMPLES)
    extract_channel_features(x)


if __name__ == "__main__":
    import time

    signal = np.random.default_rng(42).standard_normal(WINDOW_SAMPLES) * 50
    warm_jit()
    start = time.time()
    features = extract_channel_features(signal)
    print(f"{len(features)} features in {(time.time() - start) * 1000:.2f} ms")
    print(f"FEATURE_NAMES covers all: {set(features) == set(FEATURE_NAMES)}")
