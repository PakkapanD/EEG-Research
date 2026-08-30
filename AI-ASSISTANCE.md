# AI-ASSISTANCE.md

<!-- เจ้าของงานเขียนประโยคเปิดของเอกสารนี้เองตรงนี้ (หนึ่งย่อหน้า) -->

เอกสารนี้ระบุขอบเขตการใช้เครื่องมือ AI ในโปรเจกต์ ผู้อ่านที่ตั้งใจไว้คือกรรมการ ผู้ตรวจ
และผู้ที่ต้องการรันงานซ้ำ ทุกข้อในหัวข้อ 2–3 อ้างอิงไฟล์ในโปรเจกต์ได้

---

## 1. ขอบเขตการใช้ AI

เครื่องมือ: Claude (โมเดล Sonnet) ผ่าน 2 ช่องทาง แบ่งหน้าที่ตาม CLAUDE.md §2

| ช่องทาง | หน้าที่ |
|---|---|
| **Claude Code** (รันในโฟลเดอร์ repo) | เขียนโค้ดทั้งหมดใน `src/` (27 ไฟล์), รันการทดลอง (preprocessing, feature extraction, training sweep 312 runs, evaluation, sensitivity analysis, entropy pilot), สร้างตารางผลและกราฟทั้ง 4 รูป, เขียนไฟล์ `.md` สรุปผลและ `metadata.json` ใน `output/` |
| **Claude (แชท / เอกสาร)** | งานเขียน: รายงานภาษาอังกฤษ, Related Work, แก้รายงานไทยฉบับเดิม, สไลด์, การทำความเข้าใจเนื้อหาเพื่อตอบคำถาม — ทำนอก repo นี้ |

- ตัว **เอกสาร** `CLAUDE.md`, `Consolidated-Context.md` และสเปกใน `docs/` ถูกร่างโดย Claude — `CLAUDE.md` เป็นการเรียบเรียงข้อกำหนดและการตัดสินใจของเจ้าของงานให้อยู่ในรูปที่ Claude Code ใช้เป็นบริบทถาวรได้ ตัวการตัดสินใจเชิงวิจัยที่บันทึกอยู่ในนั้น (หัวข้อ 2) เป็นของเจ้าของงาน — หลักฐานการอนุมัติแยกอยู่ใน `metadata.json` (`who_approved`) และ docstring ("approved", "flagged for objection")
- repo นี้ยังไม่มี `git init` ณ วันที่เขียนเอกสารนี้

---

## 2. การตัดสินใจเชิงวิจัยที่เป็นของเจ้าของงาน

| การตัดสินใจ | ข้อจำกัด / เหตุผลที่บังคับให้ตัดสินใจแบบนั้น | อ้างอิง |
|---|---|---|
| แยกตัวแปรต้นเป็น 2 มิติ: จำนวนช่อง (18/7/4/2) กับ ข้อจำกัดตำแหน่ง — `Best-n` (เลือกจากข้อมูล) เทียบ `Glass-n` (ตำแหน่งถูกกรอบแว่นบังคับ) ช่องว่างระหว่างสองเส้นคือคำตอบ | ถ้าลดช่องแล้ววัดแค่ประสิทธิภาพตก จะแยกไม่ออกว่าตกเพราะจำนวนช่องน้อยลง หรือเพราะตำแหน่งที่กรอบแว่นบังคับไม่เหมาะ | CLAUDE.md §1, §8 |
| ใช้หลายโมเดล (RF, LogReg, MLP) เป็นตัวควบคุม ห้ามวางกรอบผลเป็น "ML vs DL" | ถ้าเส้นโค้งรูปร่างเหมือนกันข้ามโมเดล แปลว่าเป็นสมบัติของข้อมูล ไม่ใช่ของโมเดลตัวใดตัวหนึ่ง | CLAUDE.md §1 |
| ใช้ derivation สาย F: `F7-T7` / `F8-T8` เป็นช่องหลักของ `Glass-*` (ไม่ใช่ `T7-P7` / `T8-P8`) | e-Glass จริง (Sopic 2018) วางอิเล็กโทรดที่ F7, T3(=T7), F8, T4(=T8); P7/P8 อยู่นอกระยะที่ขากรอบแว่นเอื้อมถึง; รายงานไทยฉบับเดิมใช้ `T7-P7`/`T8-P8` ใน Abstract + บทที่ 3 ขัดกับ objective ของตัวเอง | CLAUDE.md §8; `03_model_configs.json` → `configs` |
| รวม `chb01` + `chb21` เป็น subject เดียว → เหลือ 23 subject | เป็นผู้ป่วยคนเดียวกันบันทึกห่างกัน ~1.5 ปี ถ้าไม่รวม เวลา chb21 อยู่ชุดทดสอบ chb01 จะอยู่ชุดเทรน = patient leakage (รายงานเดิมมีบั๊กนี้จริง บทที่ 3.2.4) | CLAUDE.md §6.1; `metadata.json` → `subjects` |
| ตัด 3 ไฟล์ monopolar ของ chb12 | montage ต่างจากไฟล์อื่น ตัดตาม Zanetti et al. 2022 (auto-detect จาก header) | CLAUDE.md §6.2; `metadata.json` → `file_admission.excluded_files` |
| ตัดช่องซ้ำ `P7-T7` (กลับขั้วของ `T7-P7`) และ `T8-P8` ตัวที่สอง ตั้งแต่ตอนโหลดไฟล์ | ถ้าไม่ตัด `Best-2` อาจเลือกทั้ง `T7-P7` และ `P7-T7` แล้วกลายเป็น `Best-1` โดยไม่มีใครรู้ | CLAUDE.md §6.3; `metadata.json` → `channels.dropped_duplicates_at_load_time` |
| CV แบบ subject-wise: 5-fold sweep + LOO สำหรับ config หลัก (Full-18, Glass-2, Glass-7) | รายงานเดิมแบ่ง train/tuning/test แบบ fixed patient list ซึ่ง leak; subject-wise บังคับให้ทดสอบข้ามผู้ป่วยจริง | CLAUDE.md §7; docs/06 §3 |
| Pre-register เกณฑ์ "พอ" ก่อนเห็นผล และห้ามแก้หลังเห็นผล (ปรับยา: Sens≥60%/FA≤5·วัน⁻¹; แจ้งเตือนเรียลไทม์: Sens≥80%/FA≤2·วัน⁻¹) | อ้าง Beniczky & Ryvlin 2018 (Phase 0); ตั้งเกณฑ์หลังเห็นผลคือการขยับเป้า | CLAUDE.md §14; docs/06 §1; `03_model_configs.json` → `sufficiency_criteria` |
| ปฏิเสธข้อเสนอเปลี่ยน metric หลักของกราฟ channel-ladder ไปเป็น AUC-ROC หลังเห็นว่า pass rate noisy — คงไว้ที่ event-level per-subject pass rate (SzCORE) เป็นหลัก AUC-ROC เป็นกราฟเสริม | (1) pipeline ออกแบบมาตอบคำถามคลินิก; (2) เปลี่ยน metric หลักหลังเห็นผลเสี่ยงถูกมองว่าขยับเป้า; (3) LOO กับ 5-fold-per-subject ให้ตัวเลขใกล้กัน = ไม่ใช่ noise ล้วน | Consolidated-Context.md §7 A8 |
| ตัดสินใจไม่ทำ FA-margin (option 2 ของ threshold selection) | near-miss analysis พบมีแค่ 2/168 fold ที่เป็น near-miss จริง margin ช่วยได้ 2–5 fold ไม่คุ้มเวลา | Consolidated-Context.md §5.12, §7 A6 |
| กฎ tuning-pool ของ LOO: seeded ring permutation ของ 23 subject → tuning = 3 คนถัดไปใน ring (ไม่ใช่ "2 คนที่ seizure count ใกล้ที่สุด") | การเลือกตาม seizure count ใกล้เคียง leak ลักษณะของ held-out subject เข้าไปในการเลือก tuning pool | docs/06 §4 (A3, อนุมัติ 2026-08-15); `cv_folds.py` docstring |
| ตัด 9 ฟีเจอร์ `entropyProfiled_*` ออกจากคลัง (เหลือ 83 DIHC + 6 AZC = 89 ต่อช่อง) | ต้นทุนเวลา entropyProfiled ~1.4–1.5 s/channel/window (วัดจริง) รันเต็มชุดกินระดับปี — เจ้าของงานอนุมัติ 12 ส.ค. 2026 ยืนยันซ้ำ 14 ส.ค. 2026 หลังพบตัวเลขประมาณเดิมผิด | CLAUDE.md §7; `metadata.json` → `features.feature_scope_deviation` (`who_approved`) |
| ตัดสินใจไม่รัน entropyProfiled เต็มชุดข้อมูล (~259 วัน) — ทำ pilot chi-squared + lightweight impact test แทน แล้วเขียนเป็น limitation สองชั้น | pilot พบ `entropyProfiled_total_sampleEntropy` ติด chi-squared top-30 ครบ 21/21 ช่อง (การตัดจึงไม่ใช่ deviation ไร้ผล) แต่ impact test GroupKFold(5) พบ pooled AUC แทบไม่ต่างจาก baseline (+0.0010) | Consolidated-Context.md §9 A17, A17b (ตัดสินใจร่วม 19 ส.ค. 2026) |
| two-tier subject base: Tier A (6 config, 668 files / 945.49 ชม. / 181 seizures) กับ Tier B (Glass-7 เท่านั้น, 640 files / 917.48 ชม. / 177 seizures) แทนการบังคับเงื่อนไข Glass-7 กับทุก config | เงื่อนไข "ทุกไฟล์ต้องมีครบ 21 ช่อง" จะตัด 7/23 subject และ 81/181 seizure (45%) เกินเพดาน "ตัด subject เกิน 3 คนต้องหยุดถาม" | CLAUDE.md §8, §11; `metadata.json` → `subjects.selection_deviation`, `file_admission`; docs/06 §2 |
| hybrid windowing: grid 4 s ไม่ overlap ทั้งชุด + dense 0.5 s เฉพาะ ±60 s รอบ seizure — แทน uniform step 0.5 s (ซ้อน 87.5%) ที่ CLAUDE.md §7 ล็อกไว้ | uniform 0.5 s ≈ 100 ชม. / 62.6 GB เกินเพดาน 12 ชม. / 20 GB; hybrid = 907,755 window / 13.34 ชม. / 8.8 GB และยังคุม 100% ของเวลาบันทึกเป็นตัวหาร FA/day | `metadata.json` → `windowing.deviation` (`who_approved`, 14 ส.ค. 2026); docs/06 §2, §2ก |
| merge_gap เปลี่ยนจากช่วงต่อเนื่อง 5–30 s เป็น discrete set {0,4,8,12,16,20,28} s (relabel 30→28) | scoring ทำงานบน grid-only prediction ที่เว้น 4 s ค่า merge_gap ที่ไม่ใช่ผลคูณ 4 จึงไม่มีความหมาย | docs/06 §6 (A4); Consolidated-Context.md §7 A4 |
| min_event_duration < 3 s คำนวณจากชักสั้นสุดจริงในชุดข้อมูล (6 s, chb16) — ห้ามลอกเลข 10 s ของ Ali ใน pass หลัก | เกณฑ์ min-duration ที่เกินครึ่งของชักสั้นสุดทำให้ชักสั้นตรวจจับไม่ได้ตั้งแต่ต้น (10 s จะทิ้งชัก 10/181 ครั้งก่อนให้คะแนน) | CLAUDE.md §12; docs/06 §6 |
| ไม่ใช้ SVM แบบ RBF kernel | ต้นทุน inference โตตามจำนวน support vector ขยายไปประเมินบนหน้าต่าง test ~7M อันไม่ไหว | CLAUDE.md §7; docs/06 §5 |
| ไม่ใช้รายการ 32/34 ฟีเจอร์ที่ Ali คัดไว้แล้ว — คัดฟีเจอร์เองภายในแต่ละ fold ด้วย chi-squared | Ali คัดภายใต้เงื่อนไขต่างจากเรา (หน้าต่าง 5 s ไม่ซ้อน, 22 ช่อง, ไม่กรองสัญญาณ) คะแนน chi-squared ย้ายมาใช้ไม่ได้ | CLAUDE.md §7 |
| ตัดสินใจไม่ overlay จุด operating point ของ Ali 2024 บนกราฟหลัก (เก็บเป็นตาราง/footnote) | หน่วยเป็นต่อชั่วโมง แปลงแล้ว ~115–128/วัน เกินเกณฑ์ Beniczky >20 เท่า ทำให้แกน x ยืดจนโซนคลินิก 0–5/วัน อ่านไม่ออก; Ali ไม่ได้ทำ threshold sweep เทียบบนเส้นเดียวกันไม่ได้ | docs/06 §10 (A13) |
| ตัดสินใจไม่ทำ detection latency | `timescoring`'s EventScoring ไม่เปิด per-event ref↔hyp matching ต้องเขียน matching logic เองแยก — deferred อย่างมีเหตุผล เขียนใน Limitations | docs/06 §8; `build_final_results.py` docstring |
| คง chb17 ไว้เป็น onset-zone case study แม้ข้อมูลจริงขัดกับสมมติฐานเดิม (Glass-7 กลับทำได้ดีที่ chb17) | ผลที่ขัดสมมติฐานน่าสนใจกว่าเดิม เขียนเป็น open/contradictory finding | docs/06 §9 (B1); Consolidated-Context.md §8 |
| ยังไม่ `git init` ตอนนี้ (ทำทีหลัง) `.gitignore` เตรียมไว้แล้ว | เจ้าของงานตัดสินใจ 18 ส.ค. 2026 | Consolidated-Context.md §3, §11 |

ข้อที่หลักฐานในเอกสารระบุว่า AI เป็นผู้เสนอทางเลือกทางเทคนิคแล้วเจ้าของงานอนุมัติ (บันทึกเพื่อความครบถ้วน):

| เรื่อง | หลักฐาน |
|---|---|
| percentile-calibrated threshold แทน absolute-probability threshold | `select_operating_points_percentile.py` docstring: "option 1 from the threshold-transfer discussion" / "Fix tried here"; Consolidated-Context.md §5.12 A5 |
| 5-fold tuning subset = fold (i+1)%5 | `model_configs.py` docstring: "flagged for objection rather than decided silently"; `03_model_configs.json` → `five_fold_tuning_rule` |
| grid หยาบ 45 จุดสำหรับกราฟ tradeoff (ไม่แตะ grid 279 จุดที่ล็อก) | `build_tradeoff_curve.py` docstring: "Project owner chose (2026-08-17) a coarser grid"; Consolidated-Context.md §9 A14 |

---

## 3. ส่วนที่ AI เขียนภายใต้สเปกของเจ้าของงาน

`src/` มี 27 ไฟล์ `.py` แบ่งเป็น 3 กลุ่มตามบทบาท — (3.1) โมดูลฐานที่ **นิยามฟังก์ชัน/ค่าคงที่**
ถูกไฟล์อื่น import ไม่ได้ "รัน" เป็นขั้นตอน; (3.2) **สคริปต์ที่รันเป็นขั้นตอน** เรียงตามลำดับ
data flow แต่ละตัวอ่านผลของขั้นก่อนแล้วเขียนไฟล์ผลของตัวเอง; (3.3) **สคริปต์ตรวจสอบ**
ที่รันเมื่อไรก็ได้เพื่อยืนยันผลของขั้นหนึ่ง โดยไม่อยู่ในเส้นทางข้อมูลหลัก

### 3.1 โมดูลฐาน (library — ถูก import, ไม่ใช่ขั้นตอน)

| โมดูล | นิยามอะไร | ถูกใช้โดย | สเปกที่กำกับ |
|---|---|---|---|
| `audit.py` (ส่วนค่าคงที่) | `OUTPUT_DIR`, `DATA_DIR`, `REQUIRED_CHANNELS`, `GLASS7_DEPENDENCY` | แทบทุกไฟล์ | — |
| `preprocess.py` | เลือกไฟล์ (`tier_a_files`), เปิด/กรอง EDF (`open_recording`, `filtered_channel`), ตารางหน้าต่าง + ฉลาก (`window_schedule`, `window_metadata`, `label_window`, `parse_seizure_intervals`), ค่าคงที่หน้าต่าง | `run_pipeline`, `cv_folds`, `model_configs`, `evaluate`, `train`, `write_*` | CLAUDE.md §7; `metadata.json` → D1/D2 |
| `features.py` | สูตรฟีเจอร์ 89 ตัว + `channel_feature_matrix` (ตัดหน้าต่าง+สกัดในรอบเดียว), `FEATURE_NAMES`, `BANDS`, `AZC_THRESHOLDS_UV` | `run_pipeline`, `train`, `validate_features`, `write_metadata` | CLAUDE.md §7; `metadata.json` → `features` |
| `features_reference.py` | ฝาแฝดช้าดั้งเดิม แช่แข็ง ห้ามแก้/optimize | `validate_features`, `entropy_pilot` | — (หลักฐานว่า optimize แล้วค่าไม่เปลี่ยน) |
| `postprocess.py` | `make_candidate_events` (threshold → event: smooth k≤1 → merge → drop-short → split-long), `percentile_to_threshold` | `evaluate_sweep`, `select_operating_points*`, `rescore_per_subject`, `build_final_results`, `build_tradeoff_curve`, `postprocessing_sensitivity` | docs/06 §6 |
| `evaluate.py` | `score_fold`, `ground_truth_events`, `file_durations`, `sensitivity`, `fa_per_day` — ครอบ library `timescoring` (SzCORE หลัก + Ali รอง) | ทุกสคริปต์ที่ให้คะแนน | docs/06 §7; CLAUDE.md §13 |
| `model_configs.py` | `CONFIGS` (7 channel set), `MODELS`, `LOO_CONFIGS`, `THRESHOLD_GRID` (279 จุด), `SUFFICIENCY_CRITERIA`, `MODEL_HYPERPARAMS` — `main()` เขียน `03_model_configs.json` | `train`, `evaluate_sweep`, `select_operating_points*`, `build_*` | docs/06 §4–5; CLAUDE.md §8 |

หมายเหตุ: `evaluate_sweep.py` และ `select_operating_points.py` ทำสองบทบาท — เป็นทั้งขั้นตอนที่รัน (ดู 3.2)
และโมดูลที่ไฟล์อื่น import ค่าคงที่/ฟังก์ชันไปใช้ (`MERGE_GAP_DEFAULT_SEC`, `MIN_EVENT_DURATION_DEFAULT_SEC`,
`load_cv_folds`, `sweep_threshold_curve`, `pick_threshold`) — ดูข้อสังเกตเรื่องนี้ในรายงานแชทที่แนบมากับเอกสารนี้

### 3.2 ลำดับการรัน (ต้นน้ำ → ปลายน้ำ)

แต่ละแถว = สคริปต์ 1 ตัวที่สั่งรัน คอลัมน์ "อ่าน / เขียน" คือ dependency ของข้อมูล

| ขั้น | สคริปต์ | อ่าน | เขียน | สเปก |
|---|---|---|---|---|
| 0 | `audit.py` | CHB-MIT EDF headers | `00_channel_audit.csv` / `.md` | CLAUDE.md §10 ขั้น 1 |
| 1 | `run_pipeline.py` (เรียก `preprocess` + `features`, checkpoint ต่อไฟล์) | CHB-MIT + `00_channel_audit.csv` | `output/features/{subject}.parquet` | CLAUDE.md §7, §3 |
| 1b | `write_dataset_summary.py` | `output/features/*.parquet` | `01_dataset_summary.md` | CLAUDE.md §10 ขั้น 5, §6.4 |
| 1c | `write_metadata.py` | พารามิเตอร์ (import จาก `features`, `preprocess`) | `metadata.json` | CLAUDE.md §11 |
| 2 | `cv_folds.py` | `01` / features | `02_cv_folds.json` | docs/06 §3 |
| 2 | `model_configs.py` | — | `03_model_configs.json` | docs/06 §4–5 |
| 3 | `train.py --run-all` | `features/`, `02`, `03` | `04_predictions/{config}_{model}_{fold}.parquet` (+ `.meta.json` จำนวนอิเล็กโทรดจริง) | docs/06 §5, §11; CLAUDE.md rule 7, §8 |
| 4 | `evaluate_sweep.py --run-all` (~3 ชม.) | `04_predictions/`, `02`, `03` | `05a_threshold_sweep/*.parquet` (279-pt grid บน tuning, SzCORE+Ali) | docs/06 §7; CLAUDE.md rule 7 |
| 4b | `postprocessing_sensitivity.py` | `05a` subset | `05_postprocessing_sensitivity.md`, `05b_..._raw.csv` | docs/06 §6, §12 |
| 5 | `select_operating_points.py` | `05a` | `06_operating_points.csv` — absolute-threshold, **documented-failure finding เก็บไว้อ้างเป็น limitation ไม่ใช่ผลหลัก** | docs/06 §12; Consolidated-Context.md §5.12 A5 |
| 5 | `select_operating_points_percentile.py` | `05a` | `06_operating_points_percentile.csv` **← pipeline ที่ใช้จริง** (+ `fallback_to_best_achievable`, แก้บั๊ก A12) | Consolidated-Context.md §5.12, §9 A12 |
| 5b | `rescore_per_subject.py` | `06_operating_points_percentile.csv` | `06_operating_points_percentile_per_subject.csv` (5-fold คิดต่อ subject) | Consolidated-Context.md §7 A7 |
| 6 | `build_final_results.py` | `06_operating_points_percentile*.csv`, `04`, features | `06_results_szcore.csv` **(แหล่งความจริงหลัก)**, `06_results_ali.csv`, `06a_..._raw.csv` | docs/06 §7–8 |
| 6b | `measure_model_size.py` (refit RF/Full-18, RF/Glass-2 — import จาก `train`) | `features/`, `03` | `06d_model_size_inference.md` | docs/06 §8; CLAUDE.md rule 7 |
| 7 | `build_figure_data.py` | `06_results_*.csv`, `06a` | `07_figures/data_channel_ladder.csv`, `data_per_patient_heatmap.csv`, `data_szcore_vs_ali.csv` | docs/06 §10–11 |
| 7 | `build_tradeoff_curve.py` | `04_predictions/` (pooled test, grid หยาบ 45 จุด) | `07_figures/data_tradeoff_curve.csv` | docs/06 §10; Consolidated-Context.md §9 A14 |
| 7 | `make_figures.py` | `07_figures/data_*.csv` | `07_figures/fig1–4.svg` | docs/06 §10–11 |

### 3.3 สคริปต์ตรวจสอบ / วินิจฉัย (ยืนยันผลของขั้นก่อน ไม่อยู่ในเส้นทางข้อมูลหลัก)

| สคริปต์ | ทำอะไร | ผลออก | ยืนยันขั้นไหน |
|---|---|---|---|
| `validate_features.py` | recompute ทั้ง 89 ฟีเจอร์สองทาง (`features` vs `features_reference`) บนหน้าต่าง EEG จริง | `00c_feature_parity.md` | ขั้น 1 (parity gate) |
| `scan_nonfinite_features.py` | สแกน cell ±inf ในไฟล์ฟีเจอร์ (one-off — เจอบั๊ก `chb17b_69` แก้แล้ว 15 ส.ค. 2026) | `00e_nonfinite_feature_scan.md` | ขั้น 1 |
| `spotcheck_hybrid_windowing.py` | RF pilot 8 subject / Full-18, 4 acceptance check ว่า hybrid windowing ไม่ทำให้ FA/day เพี้ยน | `00d_hybrid_windowing_spotcheck.md` | ขั้น 1 (windowing) |

### 3.4 สาขาแยก — entropy pilot (ไม่อยู่ในเส้นทางหลัก — ตรวจฟีเจอร์กลุ่มที่ตัดออก ดูหัวข้อ 2)

| สคริปต์ | ทำอะไร | ผลออก |
|---|---|---|
| `entropy_pilot.py` | สกัดเฉพาะ 9 คอลัมน์ `entropyProfiled_*` บน 2000 window / 79 ไฟล์ | `00f_entropy_pilot_features.parquet` |
| `entropy_pilot_analyze.py` | chi-squared ต่อช่อง (89+9=98 ฟีเจอร์) เช็คว่า entropyProfiled ติด top-30 ไหม | `00f_entropy_pilot.md` |
| `entropy_pilot_impact_test.py` | GroupKFold(5) Full-18 เทียบ AUC-ROC pool 89 vs 90 ฟีเจอร์ | `00f_entropy_pilot_impact_test.md` |

---

## 4. ข้อจำกัดของเอกสารฉบับนี้

- เป็น snapshot **ณ วันที่ 29 สิงหาคม 2026** สังเคราะห์จาก: `CLAUDE.md`, `Consolidated-Context.md`, `docs/06-spec2-cv-training-postprocessing-evaluation-th.md`, module docstring ของทั้ง 27 ไฟล์ใน `src/`, `output/metadata.json`, `output/03_model_configs.json`
- ไม่ได้อ่านโค้ดบรรทัดต่อบรรทัด — หัวข้อ 3 อ้างจาก docstring ระดับ module เป็นหลัก
- คอลัมน์อ้างอิงในหัวข้อ 2 ยึดถ้อยคำในเอกสาร (`who_approved`, "approved", "flagged for objection", "Project owner chose") — จุดที่เอกสารไม่ได้ระบุว่าใครริเริ่มข้อเสนอ ไม่เดาแทน
- อัปเดตครั้งถัดไปเมื่อ: เริ่มร่างรายงาน, `git init`, หรือมีการเบี่ยงเบนจากพารามิเตอร์ที่ล็อกเพิ่ม
