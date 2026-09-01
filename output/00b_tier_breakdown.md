# Per-file Tier Breakdown (Tier A vs Tier B)

> ⚠️ **SUPERSEDED — pre-decision snapshot.** Written while the two-tier question was
> still open. The "17 subjects / 470 files" `INCLUDED_RAW_SUBJECTS` rollup, the
> "7 subjects excluded" note, and the "Tension with CLAUDE.md section 6" section all
> describe the **rejected** approach (impose the Glass-7 channel requirement on every
> config). Final design: **two-tier subject base — 23 subjects in BOTH tiers, none
> excluded.** Glass-7 runs on the Tier B subset of *files*; every other config uses
> full Tier A. See `output/metadata.json` -> `subjects.selection_deviation` and
> `file_admission`, and `output/01_dataset_summary.md` section 1.
> The per-file Tier A/B counts in the table below are still correct; only the framing is stale.

Computed from `output/00_channel_audit.csv`, per-file (not per-subject).

- **Tier A** = all 18 double-banana channels present, file not monopolar. Sufficient for
  `Full-18`, `Best-7`, `Best-4`, `Best-2`, `Glass-4`, `Glass-2` (none of these need
  `T7-FT9` / `FT9-FT10` / `FT10-T8`).
- **Tier B** = Tier A + `T7-FT9`, `FT9-FT10`, `FT10-T8` present. Required only for `Glass-7`.

Current `run_pipeline.py` (`INCLUDED_RAW_SUBJECTS`, 17 subjects, 470 files) is exactly the
set of subjects whose files are **100% Tier-B-compliant** — i.e. it applies the Glass-7
requirement to every configuration, per CLAUDE.md section 6 ("same subject set for all
configurations").

| Subject | Files (total) | Tier A files | Tier A hrs | Tier A seizures | Tier B files | Tier B hrs | Tier B seizures |
|---|---|---|---|---|---|---|---|
| chb01 | 42 | 42 | 40.55 | 7 | 42 | 40.55 | 7 |
| chb02 | 35 | 35 | 34.27 | 3 | 35 | 34.27 | 3 |
| chb03 | 37 | 37 | 37.00 | 7 | 37 | 37.00 | 7 |
| chb04 | 39 | 39 | 146.63 | 4 | 39 | 146.63 | 4 |
| chb05 | 38 | 38 | 38.00 | 5 | 38 | 38.00 | 5 |
| chb06 | 16 | 16 | 58.73 | 8 | 16 | 58.73 | 8 |
| chb07 | 17 | 17 | 59.05 | 2 | 17 | 59.05 | 2 |
| chb08 | 20 | 20 | 20.01 | 5 | 20 | 20.01 | 5 |
| chb09 | 19 | 19 | 67.87 | 4 | 19 | 67.87 | 4 |
| chb10 | 24 | 24 | 48.02 | 7 | 24 | 48.02 | 7 |
| chb11 | 35 | 35 | 34.79 | 3 | 35 | 34.79 | 3 |
| **chb12** | 24 | 21 | 20.69 | 27 | 21 | 20.69 | 27 |
| **chb13** | 33 | 33 | 33.00 | 12 | **11** | **11.00** | **10** |
| chb14 | 26 | 26 | 26.00 | 8 | 26 | 26.00 | 8 |
| **chb15** | 40 | 40 | 40.01 | 20 | 39 | 39.01 | 20 |
| **chb16** | 19 | 19 | 19.00 | 10 | 17 | 17.00 | 8 |
| **chb17** | 21 | 21 | 21.01 | 3 | 20 | 20.01 | 3 |
| **chb18** | 35 | 35 | 34.63 | 6 | 34 | 33.63 | 6 |
| **chb19** | 29 | 29 | 28.93 | 3 | 28 | 27.93 | 3 |
| chb20 | 29 | 29 | 27.60 | 8 | 29 | 27.60 | 8 |
| chb21 | 32 | 32 | 31.83 | 3 | 32 | 31.83 | 3 |
| chb22 | 30 | 30 | 30.00 | 3 | 30 | 30.00 | 3 |
| chb23 | 9 | 9 | 26.56 | 7 | 9 | 26.56 | 7 |
| chb24 | 22 | 22 | 21.30 | 16 | 22 | 21.30 | 16 |
| **TOTAL** | **671** | **668** | **945.49** | **181** | **640** | **917.48** | **177** |

Bold rows = the 7 subjects currently excluded entirely from the study
(`chb12, chb13, chb15, chb16, chb17, chb18, chb19`).

## Subject-level rollup

| Set | Subjects | Files | Hours | Seizures |
|---|---|---|---|---|
| Tier B, subject fully qualifies (**current `INCLUDED_RAW_SUBJECTS`**) | 17 | 470 | 748.22 | 100 |
| Tier A, subject fully qualifies (excludes only chb12, which has 3 monopolar files) | 23 | 647 | 924.80 | 154 |
| Tier A, per-file inclusion (keep every subject, drop only disqualifying files) | 24 | 668 | 945.49 | 181 |
| Tier B, per-file inclusion (keep every subject, drop only disqualifying files) | 24 | 640 | 917.48 | 177 |

**chb12 is a special case**: it fails Tier A/B only on the 3 known monopolar files
(CLAUDE.md section 4.2, already excluded per Zanetti et al. 2022). Its other 21 files are
fully Tier-B-compliant. So chb12 could be added back into the *current* 17-subject set at
essentially no extra methodological cost — the monopolar exclusion is separate from why
it's currently missing from `INCLUDED_RAW_SUBJECTS`.

**chb13** is the sharpest Tier A/B split: 33/33 files pass Tier A but only 11/33 pass Tier
B — it has FT9/FT10/T8 in barely a third of its recording.

## Tension with CLAUDE.md section 6

> "All configurations must be evaluated on the same subject set — if a subject lacks any
> required channel, exclude that subject from the entire study rather than varying n per
> configuration."

That rule is why the current pipeline uses the strict 17-subject Tier B set for
*everything*, even for configs (`Full-18`, `Best-n`, `Glass-4`, `Glass-2`) that don't need
`T7-FT9/FT9-FT10/FT10-T8` at all. Splitting into two subject sets (23-24 for
Tier-A-only configs, 17 for `Glass-7`) would recover 6-7 subjects and ~30 more seizures for
6 of the 7 configurations, at the cost of no longer having one fixed subject set across the
whole study. This is a methodology decision, not a data question — flagging it here rather
than deciding unilaterally.
