# Feature parity: fast `features.py` vs frozen `features_reference.py`

- Windows checked: 252 (real CHB-MIT, filtered exactly as the pipeline does)
- Features per window: 89
- Pass rule: `|new - ref| <= 1e-10 * max(1, |ref|)`
- Bit-identical features: **86 / 89**
- Within tolerance but not bit-identical: **3**
- Failures: **0**
- NaN disagreements: **0**
- Elapsed: 43.3 s

## Features that are not bit-identical

These are the ones whose floating-point accumulation order changed. All
come from the fused Chebyshev-distance kernel or from computing moments
directly instead of through `scipy.stats`; none is a change of definition.

| feature | max abs err | max rel err | worst window |
|---|---|---|---|
| fuzzyEntropy | 6.734e-14 | 6.734e-14 | chb13_19.edf:P8-O2:2157.6s |
| approximateEntropy | 1.554e-14 | 1.554e-14 | chb24_01.edf:P8-O2:495.8s |
| fd_kurtosis | 1.421e-14 | 2.341e-16 | chb13_19.edf:P4-O2:2157.6s |

## Bit-identical features

`maximum`, `minimum`, `mean`, `median`, `standardDeviation`, `variance`, `kurtosis`, `skewness`, `numberOfZeroCrossing`, `positiveToNegativeSampleRatio`, `positiveToNegativePeakRatio`, `meanAbsoluteValue`, `sampleEntropy`, `permutationEntropy`, `singularValueDecompositionEntropy`, `distributionEntropy`, `shannonEntropy`, `renyiEntropy`, `lempelZivComplexity`, `hjorthMobility`, `hjorthComplexity`, `fisherInfo`, `petrosianFd`, `katzFd`, `higuchiFd`, `detrendedFluctuation`, `fd_maximum`, `fd_minimum`, `fd_mean`, `fd_median`, `fd_standardDeviation`, `fd_variance`, `fd_skewness`, `fd_maximum_alpha`, `fd_minimum_alpha`, `fd_mean_alpha`, `fd_median_alpha`, `fd_standardDeviation_alpha`, `fd_variance_alpha`, `fd_kurtosis_alpha`, `fd_skewness_alpha`, `fd_bandPower_alpha`, `fd_maximum_beta`, `fd_minimum_beta`, `fd_mean_beta`, `fd_median_beta`, `fd_standardDeviation_beta`, `fd_variance_beta`, `fd_kurtosis_beta`, `fd_skewness_beta`, `fd_bandPower_beta`, `fd_maximum_delta`, `fd_minimum_delta`, `fd_mean_delta`, `fd_median_delta`, `fd_standardDeviation_delta`, `fd_variance_delta`, `fd_kurtosis_delta`, `fd_skewness_delta`, `fd_bandPower_delta`, `fd_maximum_theta`, `fd_minimum_theta`, `fd_mean_theta`, `fd_median_theta`, `fd_standardDeviation_theta`, `fd_variance_theta`, `fd_kurtosis_theta`, `fd_skewness_theta`, `fd_bandPower_theta`, `fd_maximum_other`, `fd_minimum_other`, `fd_mean_other`, `fd_median_other`, `fd_standardDeviation_other`, `fd_variance_other`, `fd_kurtosis_other`, `fd_skewness_other`, `fd_bandPower_other`, `fd_bandPower`, `spectralEntropy`, `azc_0`, `azc_16`, `azc_32`, `azc_64`, `azc_128`, `azc_256`
