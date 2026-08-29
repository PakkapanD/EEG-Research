# entropyProfiled_* chi-squared pilot (CLAUDE.md section 7 / spec 2 section 2 #3)

Stratified sample: 2000 windows (1000 label=1, 1000 label=0) across 13 subjects, 79 files. 89 existing features (output/features/) + 9 entropyProfiled_* features (freshly extracted, entropy_pilot.py) = 98/channel, ranked per channel with chi2 (MinMaxScaler first, same method as train.py's select_top_features_per_channel), top-30 per channel.

**Headline: entropyProfiled_* placed in the chi-squared top-30 in 21 of 21 scoreable channels.**

## Per-channel result

| channel | n_rows | entropy in top-30? | best entropy rank (of 98) | best feature |
|---|---|---|---|---|
| FP1-F7 | 2000 | YES | 2 | entropyProfiled_total_sampleEntropy |
| F7-T7 | 2000 | YES | 3 | entropyProfiled_total_sampleEntropy |
| T7-P7 | 2000 | YES | 2 | entropyProfiled_total_sampleEntropy |
| P7-O1 | 2000 | YES | 1 | entropyProfiled_total_sampleEntropy |
| FP1-F3 | 2000 | YES | 5 | entropyProfiled_total_sampleEntropy |
| F3-C3 | 2000 | YES | 8 | entropyProfiled_total_sampleEntropy |
| C3-P3 | 2000 | YES | 2 | entropyProfiled_total_sampleEntropy |
| P3-O1 | 2000 | YES | 2 | entropyProfiled_total_sampleEntropy |
| FP2-F4 | 2000 | YES | 15 | entropyProfiled_total_sampleEntropy |
| F4-C4 | 2000 | YES | 10 | entropyProfiled_total_sampleEntropy |
| C4-P4 | 2000 | YES | 1 | entropyProfiled_total_sampleEntropy |
| P4-O2 | 2000 | YES | 13 | entropyProfiled_total_sampleEntropy |
| FP2-F8 | 2000 | YES | 1 | entropyProfiled_total_sampleEntropy |
| F8-T8 | 2000 | YES | 7 | entropyProfiled_total_sampleEntropy |
| T8-P8 | 2000 | YES | 15 | entropyProfiled_total_sampleEntropy |
| P8-O2 | 2000 | YES | 3 | entropyProfiled_total_sampleEntropy |
| FZ-CZ | 2000 | YES | 25 | entropyProfiled_total_sampleEntropy |
| CZ-PZ | 2000 | YES | 25 | entropyProfiled_total_sampleEntropy |
| T7-FT9 | 1970 | YES | 1 | entropyProfiled_total_sampleEntropy |
| FT9-FT10 | 1970 | YES | 2 | entropyProfiled_total_sampleEntropy |
| FT10-T8 | 1970 | YES | 1 | entropyProfiled_total_sampleEntropy |

**Conclusion: entropyProfiled_* would have mattered in 21 channel(s).** The cost-only deviation (2026-08-12) is NOT validated as harmless - this needs to be written up as a real limitation in Methods, not folded into the cost justification. Consider whether affected channels overlap with any locked channel-ladder config (CLAUDE.md section 8) before deciding how much this matters for the headline results.