# Dataset Summary (preprocessing + feature extraction)

## 1. Evaluation bases

Files are admitted individually (deviation D1 in `metadata.json`). Glass-7 needs `T7-FT9`, `FT9-FT10`, `FT10-T8`, which 28 admitted files do not carry, so it is reported against the Tier B subset. Every other configuration uses Tier A.

| base | subjects | files | hours | seizures | used by |
|---|---|---|---|---|---|
| Tier A | 23 | 668 | 945.49 | 181 | Full-18, Best-7/4/2, Glass-4, Glass-2 |
| Tier B | 23 | 640 | 917.48 | 177 | Glass-7 |

**Our seizure denominator is 181** (Tier A). Ali 2024 uses 175, Zanetti 2022 185, Zanetti 2025 198 - percentages are not comparable across those without restating the denominator (CLAUDE.md section 6.4).

Seizure events recovered from the feature files: 181 (must equal 181).

## 2. Windows

- Total windows: **907,755**
- On the uniform 4 s grid (the evaluation timeline): 850,907 = 945.45 recorded hours
- Dense 0.5 s windows around seizures (training only): 56,848
- Positive windows: 21,969 total, 2,760 on the grid
- Class ratio on the evaluation timeline: 1:307
- Class ratio including dense windows (what training sees): 1:40

## 3. Seizure durations and the post-processing ceiling

- Seizures: 181
- Duration: shortest **6 s**, median 47 s, longest 752 s
- Uniform-grid positive windows per seizure: min **2**, median 12, max 189
- Seizures with 0 grid windows: 0 (any non-zero here means a seizure is undetectable by construction)
- Seizures with <= 2 grid windows: 6

**Ceiling for spec 2:** a 'k consecutive positive grid windows' rule spans k x 4 s. At k=2 that is 8 s, longer than the shortest seizure (6 s), so **k is capped at 1** at grid resolution. Do not copy Ali's 10 s minimum-duration rule - it would discard 10 of our 181 seizures before detection is even attempted.

### Shortest 10 seizures

| subject | file | duration (s) | grid windows |
|---|---|---|---|
| chb16 | chb16_16.edf | 6 | 2 |
| chb16 | chb16_17.edf | 6 | 2 |
| chb16 | chb16_18.edf | 7 | 2 |
| chb16 | chb16_17.edf | 8 | 3 |
| chb16 | chb16_17.edf | 8 | 3 |
| chb16 | chb16_18.edf | 8 | 2 |
| chb02 | chb02_19.edf | 9 | 3 |
| chb16 | chb16_10.edf | 9 | 3 |
| chb16 | chb16_11.edf | 9 | 2 |
| chb16 | chb16_17.edf | 9 | 2 |

## 4. Missing channels (NaN, never 0)

| channel | windows with NaN | % of all windows |
|---|---|---|
| T7-FT9 | 26,238 | 2.9% |
| FT9-FT10 | 26,238 | 2.9% |
| FT10-T8 | 26,238 | 2.9% |

Each NaN channel accounts for 89 NaN feature columns in those windows. Structural NaN feature cells (whole channel absent): 7,005,546.

On top of that, **25,343 computation-failure NaN feature cells** occur inside channels that ARE present: a handful of features (kurtosis, skewness, renyiEntropy, hjorthMobility/Complexity, detrended_fluctuation) divide by a variance-derived quantity, and a near-zero-variance segment (flat/saturated channel, not literally absent) can make that quantity underflow to exactly 0.0 after squaring - features.py converts the result to NaN rather than letting a silent +-inf reach the scaler/model (CLAUDE.md section 9). Found and fixed on 2026-08-15: all 25,343 of these cells are on chb17, `chb17b_69.edf` ~1520-1540s (FP1-F7/P3-O1). A full scan of every other subject's feature parquet found zero further occurrences (`output/00e_nonfinite_feature_scan.md`).

**Total NaN feature cells (structural + computation-failure): 7,030,889.**

## 5. Per subject

| subject | files | hours | seizures | shortest (s) | longest (s) | windows | grid | dense | pos | grid pos | files w/o Glass-7 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| chb01+21 | 74 | 72.38 | 10 | 12 | 101 | 68,271 | 65,142 | 3,129 | 1,180 | 147 | 0 |
| chb02 | 35 | 34.27 | 3 | 9 | 82 | 31,772 | 30,839 | 933 | 347 | 44 | 0 |
| chb03 | 37 | 37.00 | 7 | 47 | 69 | 35,479 | 33,301 | 2,178 | 811 | 103 | 0 |
| chb04 | 39 | 146.62 | 4 | 49 | 116 | 133,463 | 131,959 | 1,504 | 760 | 95 | 0 |
| chb05 | 38 | 38.00 | 5 | 96 | 120 | 36,232 | 34,202 | 2,030 | 1,121 | 140 | 0 |
| chb06 | 16 | 58.73 | 8 | 12 | 20 | 54,759 | 52,860 | 1,899 | 250 | 32 | 0 |
| chb07 | 17 | 59.05 | 2 | 86 | 96 | 53,883 | 53,143 | 740 | 366 | 46 | 0 |
| chb08 | 20 | 20.01 | 5 | 134 | 264 | 20,667 | 18,005 | 2,662 | 1,843 | 231 | 0 |
| chb09 | 19 | 67.87 | 4 | 62 | 79 | 62,405 | 61,079 | 1,326 | 556 | 69 | 0 |
| chb10 | 24 | 48.02 | 7 | 35 | 89 | 45,479 | 43,220 | 2,259 | 901 | 114 | 0 |
| chb11 | 35 | 34.79 | 3 | 22 | 752 | 33,356 | 31,313 | 2,043 | 1,615 | 203 | 0 |
| chb12 | 21 | 20.69 | 27 | 13 | 97 | 25,969 | 18,617 | 7,352 | 2,005 | 256 | 0 |
| chb13 | 33 | 33.00 | 12 | 17 | 70 | 33,165 | 29,700 | 3,465 | 1,082 | 138 | 22 |
| chb14 | 26 | 26.00 | 8 | 14 | 41 | 25,382 | 23,400 | 1,982 | 346 | 43 | 0 |
| chb15 | 40 | 40.01 | 20 | 31 | 205 | 43,706 | 36,008 | 7,698 | 4,004 | 501 | 1 |
| chb16 | 19 | 19.00 | 10 | 6 | 14 | 19,354 | 17,100 | 2,254 | 178 | 25 | 2 |
| chb17 | 21 | 21.01 | 3 | 88 | 115 | 20,050 | 18,906 | 1,144 | 589 | 74 | 1 |
| chb18 | 35 | 34.63 | 6 | 30 | 68 | 32,990 | 31,170 | 1,820 | 640 | 80 | 1 |
| chb19 | 29 | 28.93 | 3 | 77 | 81 | 27,080 | 26,036 | 1,044 | 475 | 58 | 1 |
| chb20 | 29 | 27.60 | 8 | 29 | 49 | 27,040 | 24,839 | 2,201 | 596 | 76 | 0 |
| chb22 | 30 | 30.00 | 3 | 58 | 74 | 27,991 | 27,002 | 989 | 411 | 50 | 0 |
| chb23 | 9 | 26.56 | 7 | 20 | 113 | 26,118 | 23,900 | 2,218 | 855 | 108 | 0 |
| chb24 | 22 | 21.30 | 16 | 16 | 70 | 23,144 | 19,166 | 3,978 | 1,038 | 127 | 0 |
| **TOTAL** | **668** | **945.45** | **181** | **6** | **752** | **907,755** | **850,907** | **56,848** | **21,969** | **2,760** | **28** |

Per-subject numbers are reported because the distribution across subjects in this dataset is bimodal, not bell-shaped; a macro average alone hides that (CLAUDE.md section 12).
