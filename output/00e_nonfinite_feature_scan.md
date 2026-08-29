# Non-finite (+-inf) feature cell scan (pre-fix data)

Scans the existing `output/features/*.parquet` (extracted before the
zero-variance overflow fix in `features.py`) for +-inf cells - the
symptom of dividing by a variance that underflows to exactly 0.0 after
squaring on a near-zero-variance (flat/saturated) channel segment.
This is a report only; no files are re-extracted here.

Total non-finite cells: **0** of 1,696,594,095 feature cells scanned (0.000000%)

| subject | non-finite cells | affected windows | affected files/channels |
|---|---|---|---|
| chb01+21 | 0 | - | - |
| chb02 | 0 | - | - |
| chb03 | 0 | - | - |
| chb04 | 0 | - | - |
| chb05 | 0 | - | - |
| chb06 | 0 | - | - |
| chb07 | 0 | - | - |
| chb08 | 0 | - | - |
| chb09 | 0 | - | - |
| chb10 | 0 | - | - |
| chb11 | 0 | - | - |
| chb12 | 0 | - | - |
| chb13 | 0 | - | - |
| chb14 | 0 | - | - |
| chb15 | 0 | - | - |
| chb16 | 0 | - | - |
| chb17 | 0 | - | - |
| chb18 | 0 | - | - |
| chb19 | 0 | - | - |
| chb20 | 0 | - | - |
| chb22 | 0 | - | - |
| chb23 | 0 | - | - |
| chb24 | 0 | - | - |

## Detail (first 200 rows across all affected subjects)

| subject | filename | window_start_sec | column |
|---|---|---|---|