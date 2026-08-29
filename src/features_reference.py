"""FROZEN REFERENCE IMPLEMENTATION - do not use in the pipeline, do not optimize.

This is the original, straightforward (slow) implementation of every feature.
features.py was later rewritten for speed; this file is kept verbatim as the
ground truth that validate_features.py checks the fast version against, so the
claim "the optimization did not change any feature value" is reproducible by an
outside reviewer rather than asserted. See output/00c_feature_parity.md.

Nothing here may be "fixed" or sped up. If a feature definition ever legitimately
changes, change it in features.py and record the mismatch in metadata.json.

--- original docstring follows ---

Feature extraction: 83 of the 92 DIHC_FeatureManager-equivalent features + 6 AZC features.

SCOPE DEVIATION from docs/05-spec-preprocessing.md ("extract all 92, no
selection at this stage"): the 9 entropyProfiled_* features are dropped.
Benchmarked on real EEG (not synthetic test data) at ~496ms/channel/window -
vs ~1ms for the other 83 features combined - which put the full 17-subject,
21-channel, 92-feature run at an estimated ~68 days on this machine's 20
cores. Approved by the project owner 2026-08-12 in exchange for ~50x speedup
(~1-2 days). This must be stated plainly in the paper's Methods/Limitations,
not hidden - see metadata.json "feature_scope_deviation".

The 92 features and their exact formulas were reverse-engineered from the public
source of https://github.com/WWM-EMRAN/DIHC_FeatureManager (no PyPI package
exists). Confirmed against the library's own example output
(eeg_all_features_python.csv, 92 columns) and its source files
(DIHC_FeatureExtractor.py, DIHC_FeatureExtrantor_Helper.py, DIHC_EntropyProfile.py).
Most features wrap the `antropy` package exactly as DIHC does; a handful
(fuzzyEntropy, distributionEntropy, entropyProfiled_*, shannonEntropy,
renyiEntropy, fisherInfo) are DIHC's own code and are reimplemented here,
vectorized for speed (the source uses numba @njit with an O(N^3)-ish histogram
loop that is far slower than a scipy.spatial.distance.pdist + searchsorted
formulation for our window length).

The 6 AZC features follow Zanetti et al. 2022: a raw zero-crossing count plus
five zero-crossing counts on a Douglas-Peucker polygonal approximation of the
window at thresholds 16/32/64/128/256 uV (their reported values). This is our
own implementation (no reference code was found) - see AZC_THRESHOLDS_UV below.

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
from numba import njit, prange
from scipy import stats as sp_stats
from scipy.integrate import simpson
from scipy.signal import find_peaks, welch
from scipy.spatial.distance import pdist, squareform

import antropy as ant

FS = 256
NPERSEG = 512
NOVERLAP = 256

# Project-locked band edges (CLAUDE.md section 5), used in place of DIHC's own
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

STAT_FUNCS = {
    "maximum": np.max,
    "minimum": np.min,
    "mean": np.mean,
    "median": np.median,
    "standardDeviation": np.std,
    "variance": np.var,
    "kurtosis": sp_stats.kurtosis,
    "skewness": sp_stats.skew,
}


# ---------------------------------------------------------------------------
# Time-domain linear
# ---------------------------------------------------------------------------

def _safe_ratio(numerator, denominator):
    try:
        return numerator / denominator
    except ZeroDivisionError:
        return 0.0


def td_linear_features(x):
    feats = {name: fn(x) for name, fn in STAT_FUNCS.items()}
    feats["numberOfZeroCrossing"] = ant.num_zerocross(x)
    feats["positiveToNegativeSampleRatio"] = _safe_ratio(
        np.sum(x >= 0), np.sum(x < 0)
    )
    # Faithful to the upstream bug: len() of the find_peaks() 2-tuple, not the
    # peak-index array - this is always 2/2 = 1.0. See module docstring.
    feats["positiveToNegativePeakRatio"] = _safe_ratio(
        len(find_peaks(x)), len(find_peaks(-x))
    )
    feats["meanAbsoluteValue"] = np.mean(np.abs(x))
    return feats


# ---------------------------------------------------------------------------
# Time-domain nonlinear entropy
# ---------------------------------------------------------------------------

def shannon_entropy(x, m=2):
    _, counts = np.unique(x, return_counts=True)
    dist = counts / counts.sum()
    return sp_stats.entropy(dist, base=m)


def renyi_entropy(x, alpha=2):
    if np.isclose(alpha, 1):
        p = x * np.log2(x, out=np.zeros_like(x), where=x > 0)
        return -np.sum(p)
    return (1.0 / (1.0 - alpha)) * np.log2(np.sum(x ** alpha))


@njit(cache=True)
def _fuzzy_similarity(vec1, vec2, r, log2_val):
    max_diff = 0.0
    for i in range(len(vec1)):
        diff = abs(vec1[i] - vec2[i])
        if diff > max_diff:
            max_diff = diff
    return np.exp(-log2_val * (max_diff / r) ** 2)


def _embed(x, m, tau=1):
    return np.lib.stride_tricks.sliding_window_view(x, m)[::tau]


def fuzzy_entropy(x, m=2, tau=1, r_factor=0.2):
    n = len(x)
    if n - (m + 1) * tau + 1 <= 0:
        return 0.0
    r = r_factor * np.std(x)
    if r == 0:
        return 0.0
    emb_m = _embed(x, m, tau)
    emb_m1 = _embed(x, m + 1, tau)
    dm = pdist(emb_m, metric="chebyshev")
    dm1 = pdist(emb_m1, metric="chebyshev")
    log2v = np.log(2.0)
    total_m = np.mean(np.exp(-log2v * (dm / r) ** 2))
    total_m1 = np.mean(np.exp(-log2v * (dm1 / r) ** 2))
    if total_m1 == 0.0 or total_m == 0.0:
        return 0.0
    return -np.log(total_m1 / total_m)


def distribution_entropy(x, m=2, M=500):
    emb = _embed(x, m)
    d = pdist(emb, metric="chebyshev")
    min_d, max_d = d.min(), d.max()
    if min_d == max_d:
        return 0.0
    bin_edges = np.linspace(min_d, max_d, M + 1)
    hist, _ = np.histogram(d, bins=bin_edges)
    prob = hist / len(d)
    prob = prob[prob > 0]
    return -np.sum(prob * np.log2(prob)) / np.log2(M)


@njit(parallel=True, cache=True)
def _cum_hist_searchsorted(mat, bins):
    n = mat.shape[0]
    M = len(bins)
    out = np.empty((n, M))
    for i in prange(n):
        row = np.empty(n - 1)
        k = 0
        for j in range(n):
            if j != i:
                row[k] = mat[i, j]
                k += 1
        row = np.sort(row)
        counts = np.searchsorted(row, bins, side="right")
        out[i, :] = counts / (n - 1)
    return out


def sample_entropy_profile(x, m=2):
    """DIHC's 'sample entropy profile': a per-lag-bin log-ratio of cumulative
    Chebyshev-distance histograms between embedding dimensions m and m+1.
    Returns the raw profile array (reduced to 9 summary stats by the caller).
    """
    emb_m = _embed(x, m)
    emb_m1 = _embed(x, m + 1)

    def cheb_matrix(e):
        d = pdist(e, metric="chebyshev")
        d = np.floor(d * 1000 + 0.5) / 1000
        return squareform(d)

    dist_m = cheb_matrix(emb_m)
    dist_m1 = cheb_matrix(emb_m1)
    combined = np.concatenate([dist_m.ravel(), dist_m1.ravel()])
    range_vals = np.unique(combined)
    if len(range_vals) < 2:
        return np.zeros(1)

    hist_m = _cum_hist_searchsorted(dist_m, range_vals)
    hist_m1 = _cum_hist_searchsorted(dist_m1, range_vals)
    b = hist_m.sum(axis=0) / hist_m.shape[0]
    a = hist_m1.sum(axis=0) / hist_m1.shape[0]
    eps = 1e-12
    return np.log((b + eps) / (a + eps))


def entropy_profile_stats(profile):
    stats = {
        "total": np.sum(profile),
        "average": np.mean(profile),
        "maximum": np.max(profile),
        "minimum": np.min(profile),
        "median": np.median(profile),
        "standardDeviation": np.std(profile),
        "variance": np.var(profile),
        "kurtosis": sp_stats.kurtosis(profile),
        "skewness": sp_stats.skew(profile),
    }
    return {f"entropyProfiled_{k}_sampleEntropy": v for k, v in stats.items()}


def td_entropy_features(x):
    feats = {}
    try:
        feats["approximateEntropy"] = ant.app_entropy(x)
    except ZeroDivisionError:
        feats["approximateEntropy"] = 0.0

    # DIHC only computes real sample entropy for len(x) >= 5000; our windows
    # are 1024 samples, so this is always 0.0 by the library's own design.
    feats["sampleEntropy"] = 0.0 if len(x) < 5000 else ant.sample_entropy(x)

    feats["permutationEntropy"] = ant.perm_entropy(x)
    feats["singularValueDecompositionEntropy"] = ant.svd_entropy(x)
    feats["fuzzyEntropy"] = fuzzy_entropy(x)
    feats["distributionEntropy"] = distribution_entropy(x)
    feats["shannonEntropy"] = shannon_entropy(x)
    feats["renyiEntropy"] = renyi_entropy(x)

    # entropyProfiled_* (9 features) intentionally dropped - benchmarked at
    # ~496ms/channel on real EEG (vs ~1ms for the rest of this function
    # combined), which put the full-dataset run at ~68 days on this 20-core
    # machine. Scope change approved by project owner 2026-08-12; see
    # metadata.json "feature_scope_deviation". sample_entropy_profile() and
    # entropy_profile_stats() are kept below, unused, in case this is revisited.
    return feats


# ---------------------------------------------------------------------------
# Time-domain nonlinear complexity / fractal dimension
# ---------------------------------------------------------------------------

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

def fd_features(x, fs=FS):
    freqs, psd = welch(x, fs=fs, nperseg=NPERSEG, noverlap=NOVERLAP)
    feats = {}

    for name, fn in STAT_FUNCS.items():
        feats[f"fd_{name}"] = fn(psd)

    for band, (lo, hi) in BANDS.items():
        mask = (freqs >= lo) & (freqs < hi)
        band_psd = psd[mask]
        if band_psd.size == 0:
            for name in STAT_FUNCS:
                feats[f"fd_{name}_{band}"] = np.nan
            feats[f"fd_bandPower_{band}"] = np.nan
            continue
        for name, fn in STAT_FUNCS.items():
            feats[f"fd_{name}_{band}"] = fn(band_psd)
        feats[f"fd_bandPower_{band}"] = simpson(band_psd, x=freqs[mask])

    feats["fd_bandPower"] = simpson(psd, x=freqs)
    feats["spectralEntropy"] = ant.spectral_entropy(x, sf=fs, method="welch", nperseg=NPERSEG)
    return feats


# ---------------------------------------------------------------------------
# AZC (Zanetti et al. 2022)
# ---------------------------------------------------------------------------

def _rdp_keep_mask(y, epsilon):
    """Douglas-Peucker simplification using vertical (amplitude) distance from
    the straight-line interpolation, matching AZC's amplitude-threshold
    (uV) semantics. x-axis is the sample index.
    """
    n = len(y)
    keep = np.zeros(n, dtype=bool)
    keep[0] = True
    keep[-1] = True
    x = np.arange(n, dtype=np.float64)
    stack = [(0, n - 1)]
    while stack:
        start, end = stack.pop()
        if end - start < 2:
            continue
        x0, y0 = x[start], y[start]
        x1, y1 = x[end], y[end]
        seg_x = x[start + 1:end]
        seg_y = y[start + 1:end]
        if x1 == x0:
            dist = np.abs(seg_y - y0)
        else:
            slope = (y1 - y0) / (x1 - x0)
            interp = y0 + slope * (seg_x - x0)
            dist = np.abs(seg_y - interp)
        idx_max_local = np.argmax(dist)
        dmax = dist[idx_max_local]
        if dmax > epsilon:
            split = start + 1 + idx_max_local
            keep[split] = True
            stack.append((start, split))
            stack.append((split, end))
    return keep


def _zero_crossings(y):
    signs = np.sign(y)
    signs[signs == 0] = 1
    return int(np.sum(np.diff(signs) != 0))


def azc_features(x):
    feats = {}
    for thr in AZC_THRESHOLDS_UV:
        if thr == 0:
            feats[f"azc_{thr}"] = _zero_crossings(x)
        else:
            mask = _rdp_keep_mask(x, thr)
            feats[f"azc_{thr}"] = _zero_crossings(x[mask])
    return feats


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def extract_channel_features(x):
    """All 98 features (92 DIHC-equivalent + 6 AZC) for one channel window."""
    x = np.ascontiguousarray(x, dtype=np.float64)
    feats = {}
    feats.update(td_linear_features(x))
    feats.update(td_entropy_features(x))
    feats.update(td_complexity_features(x))
    feats.update(fd_features(x))
    feats.update(azc_features(x))
    return feats


def extract_window_features(window_data, channel_names):
    """window_data: array[n_channels, n_samples]. Returns {channel__feature: value}."""
    out = {}
    for ch_idx, ch_name in enumerate(channel_names):
        ch_feats = extract_channel_features(window_data[ch_idx])
        for feat_name, value in ch_feats.items():
            out[f"{ch_name}__{feat_name}"] = value
    return out
