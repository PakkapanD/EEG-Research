# CHB-MIT Header Audit Summary

- Subjects (raw folders, before chb01/chb21 merge): 24
- Files: 671
- Total recorded duration: 948.49 hours (3414574 s)
- Total seizures (sum of 'Number of Seizures in File' across all files): 194

## 1. Channel presence

| Channel | Subjects (of 24) | Files (of 671) | % of total duration |
|---|---|---|---|
| FP1-F7 | 24 | 668 | 99.7% |
| F7-T7 | 24 | 668 | 99.7% |
| T7-P7 | 24 | 668 | 99.7% |
| P7-O1 | 24 | 668 | 99.7% |
| FP1-F3 | 24 | 668 | 99.7% |
| F3-C3 | 24 | 668 | 99.7% |
| C3-P3 | 24 | 668 | 99.7% |
| P3-O1 | 24 | 668 | 99.7% |
| FP2-F4 | 24 | 668 | 99.7% |
| F4-C4 | 24 | 668 | 99.7% |
| C4-P4 | 24 | 668 | 99.7% |
| P4-O2 | 24 | 668 | 99.7% |
| FP2-F8 | 24 | 668 | 99.7% |
| F8-T8 | 24 | 668 | 99.7% |
| T8-P8 | 24 | 668 | 99.7% |
| P8-O2 | 24 | 668 | 99.7% |
| FZ-CZ | 24 | 668 | 99.7% |
| CZ-PZ | 24 | 668 | 99.7% |
| T7-FT9 | 24 | 640 | 96.7% |
| FT9-FT10 | 24 | 640 | 96.7% |
| FT10-T8 | 24 | 640 | 96.7% |

## 2. Subjects missing T7-FT9, FT9-FT10, or FT10-T8 (Glass-7 dependency)

**This determines whether Glass-7 survives as a configuration.**

- Missing `T7-FT9` (in at least one file): none
- Missing `FT9-FT10` (in at least one file): none
- Missing `FT10-T8` (in at least one file): none

- Subjects missing at least one of these channels in **every** file (cannot support Glass-7 at all): none

## 3. Subjects with all 21 required channels present in every file

- Count: 17 of 24
- Subjects: ['chb01', 'chb02', 'chb03', 'chb04', 'chb05', 'chb06', 'chb07', 'chb08', 'chb09', 'chb10', 'chb11', 'chb14', 'chb20', 'chb21', 'chb22', 'chb23', 'chb24']

## 4. Non-standard channels found (not in the 21 required + known duplicate P7-T7)

- `chb04`: ['ECG']
- `chb09`: ['VNS']
- `chb11`: ['-']
- `chb12`: ['-', '01', 'C2', 'C2-CS2', 'C3', 'C3-CS2', 'C4', 'C4-CS2', 'C6', 'C6-CS2', 'CP2', 'CP2-CS2', 'CP4', 'CP4-CS2', 'CP6', 'CP6-CS2', 'CZ', 'CZ-CS2', 'EKG1-CHIN', 'F3', 'F3-CS2', 'F4', 'F4-CS2', 'F7', 'F7-CS2', 'F8', 'F8-CS2', 'FP1', 'FP1-CS2', 'FP2', 'FP2-CS2', 'FZ', 'FZ-CS2', 'LOC-ROC', 'O1-CS2', 'O2', 'O2-CS2', 'P3', 'P3-CS2', 'P4', 'P4-CS2', 'P7', 'P7-CS2', 'P8', 'P8-CS2', 'PZ', 'PZ-CS2', 'T7', 'T7-CS2', 'T8', 'T8-CS2']
- `chb13`: ['-', 'EKG1-EKG2', 'LUE-RAE']
- `chb14`: ['-']
- `chb15`: ['-', 'CP1-Ref', 'CP2-Ref', 'CP5-Ref', 'CP6-Ref', 'FC1-Ref', 'FC2-Ref', 'FC5-Ref', 'FC6-Ref', 'PZ-OZ']
- `chb16`: ['-']
- `chb17`: ['-']
- `chb18`: ['-', '.']
- `chb19`: ['-']
- `chb20`: ['.']
- `chb21`: ['-']
- `chb22`: ['-']

## 5. Files where sampling rate != 256 Hz

- None. All files are 256 Hz.

## 6. Recording totals and chb12 monopolar exclusion

- Total duration: 948.49 hours
- Total seizures (all files, all subjects, before any exclusion): 194
- Files auto-detected as monopolar (none of the 21 required bipolar derivations present): 3 -> ['chb12_27.edf', 'chb12_28.edf', 'chb12_29.edf']
- Seizures in those monopolar files (removed if excluded per CLAUDE.md section 4.2): 13
- Total seizures after excluding monopolar files: 181

## Notes

- `subject` here is the raw CHB-MIT folder id. chb01 and chb21 are the same patient (CLAUDE.md section 4.1) and will be merged into `chb01+21` starting from the preprocessing step, not in this audit.
- `is_monopolar` is derived purely from header content (absence of all 21 required bipolar channel names), not from a hardcoded filename list.