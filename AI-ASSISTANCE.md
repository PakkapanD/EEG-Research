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

ไล่ตาม `src/` ตามลำดับ data flow (Consolidated-Context.md §15.1)

| ไฟล์ | ทำอะไร | สเปกที่กำกับ |
|---|---|---|
| `audit.py` | header audit ทุกไฟล์ EDF (ไม่โหลดสัญญาณ) → `00_channel_audit.csv/.md` | CLAUDE.md §10 ขั้น 1 |
| `preprocess.py` | เลือกไฟล์, โหลด, กรองทั้งไฟล์, หั่นหน้าต่าง (hybrid), ติดฉลาก, ตัดช่องซ้ำตอนโหลด | พารามิเตอร์ล็อก CLAUDE.md §7; deviation D1/D2 ใน `metadata.json`; สเปก preprocessing ฉบับเต็ม (`docs/05-spec-preprocessing.md`) **หาไม่เจอใน repo — ต้องยืนยันกับเจ้าของงาน** |
| `features.py` | สกัด 83 DIHC + 6 AZC = 89 ฟีเจอร์/ช่อง (เวอร์ชัน optimize) | CLAUDE.md §7; `metadata.json` → `features` |
| `features_reference.py` | เวอร์ชันช้าดั้งเดิม แช่แข็งไว้เป็น ground truth ห้ามแก้/optimize | ใช้พิสูจน์ว่า optimize แล้วค่าฟีเจอร์ไม่เปลี่ยน |
| `validate_features.py` | เทียบ `features.py` กับ `features_reference.py` บนหน้าต่าง EEG จริง → `00c_feature_parity.md` | parity gate |
| `scan_nonfinite_features.py` | สแกนหา cell ±inf ในไฟล์ฟีเจอร์ → `00e` | diagnostic ไม่แก้ข้อมูล |
| `spotcheck_hybrid_windowing.py` | pilot 8 subject / Full-18 / RF เดียว เช็ค 4 เกณฑ์ว่า hybrid windowing ไม่ทำให้ FA/day เพี้ยน → `00d` | docs/06 §2ก |
| `run_pipeline.py` | runner checkpoint ต่อไฟล์ แล้ว consolidate เป็น `features/{subject}.parquet` | CLAUDE.md §3 |
| `write_dataset_summary.py` | `01_dataset_summary.md` — ชักสั้นสุด, จำนวน window ของชักสั้น, จำนวน seizure ของเราเอง | CLAUDE.md §10 ขั้น 5, §6.4 |
| `write_metadata.py` | `metadata.json` — บันทึกทุก deviation พร้อมเหตุผล/วันที่/ผู้อนุมัติ | CLAUDE.md §11 |
| `cv_folds.py` | นิยาม fold (5-fold load-balanced ตาม seizure count + LOO ring) → `02_cv_folds.json` | docs/06 §3 |
| `model_configs.py` | นิยาม 7 channel config + hyperparameter + grid → `03_model_configs.json` | docs/06 §4, §5; CLAUDE.md §8 |
| `train.py` | เทรน RF/LR/MLP ต่อ (config, model, fold) → `04_predictions/` (probability ดิบ, บันทึกจำนวนอิเล็กโทรดจริงต่อ fold ใน meta.json) | docs/06 §5, §11; CLAUDE.md rule 7, §8 |
| `postprocess.py` | threshold → candidate event (smoothing k≤1 → merge → drop-short → split-long) | docs/06 §6 |
| `evaluate.py` | scoring SzCORE (หลัก) + Ali's rule (รอง) ผ่าน library `timescoring` | docs/06 §7; CLAUDE.md §13 |
| `evaluate_sweep.py` | กวาด threshold grid 279 จุดบน tuning split ทุก (config, model, fold) เก็บทั้งเส้น | docs/06 §7; CLAUDE.md rule 7 |
| `postprocessing_sensitivity.py` | sensitivity ของ merge_gap × min_duration บน subset เล็ก → `05_...md`, `05b_...raw.csv` | docs/06 §6, §12 (open item 4) |
| `select_operating_points.py` | absolute-threshold pipeline → `06_operating_points.csv` (**documented failure finding, ห้ามแก้**) | docs/06 §12; Consolidated-Context.md §5.12 A5 |
| `select_operating_points_percentile.py` | **pipeline ที่ใช้จริง** — percentile-calibrated threshold + `fallback_to_best_achievable` (แก้บั๊ก A12) | Consolidated-Context.md §5.12, §9 A12 |
| `rescore_per_subject.py` | คิด pass/fail ของ 5-fold ต่อ subject (เหมือน LOO) ไม่ใช่ต่อกลุ่ม fold — แก้ floor effect | Consolidated-Context.md §7 A7 |
| `build_final_results.py` | รวมผลเป็น `06_results_szcore.csv` (**แหล่งความจริงหลัก**) + `06_results_ali.csv` + `06a_..._raw.csv` | docs/06 §7–8 |
| `measure_model_size.py` | refit RF/Full-18 กับ RF/Glass-2 วัดขนาดโมเดล + inference time → `06d` | docs/06 §8; CLAUDE.md rule 7 |
| `build_figure_data.py` | เตรียมข้อมูล 3 กราฟ (channel ladder, per-patient heatmap, SzCORE-vs-Ali) → `07_figures/data_*.csv` | docs/06 §10–11 |
| `build_tradeoff_curve.py` | เตรียมข้อมูลกราฟ tradeoff (grid หยาบ 45 จุดแยกต่างหาก) → `data_tradeoff_curve.csv` | docs/06 §10; Consolidated-Context.md §9 A14 |
| `make_figures.py` | render `fig1–4.svg` จากไฟล์ `data_*.csv` | docs/06 §10–11 |
| `entropy_pilot.py` | สกัดเฉพาะ 9 คอลัมน์ `entropyProfiled_*` บน 2000 window / 79 ไฟล์ | CLAUDE.md §7 (open item); Consolidated-Context.md §9 A17 |
| `entropy_pilot_analyze.py` | chi-squared ต่อช่อง (89+9=98 ฟีเจอร์) เช็คว่า entropyProfiled จะติด top-30 ไหม → `00f_entropy_pilot.md` | Consolidated-Context.md §9 A17 |
| `entropy_pilot_impact_test.py` | GroupKFold(5) Full-18 เทียบ AUC-ROC pool 89 vs 90 ฟีเจอร์ → `00f_..._impact_test.md` | Consolidated-Context.md §9 A17b |

**เอกสารสเปกที่ยังหาไม่เจอใน repo (ต้องยืนยันกับเจ้าของงานว่าอยู่ที่อื่นหรือหายจริง):**
`docs/05-spec-preprocessing.md`, ฉบับภาษาอังกฤษของสเปก 2, literature review, reading plan —
โฟลเดอร์ `docs/` มีไฟล์เดียว: `06-spec2-cv-training-postprocessing-evaluation-th.md`

---

## 4. ข้อจำกัดของเอกสารฉบับนี้

- เป็น snapshot **ณ วันที่ 29 สิงหาคม 2026** สังเคราะห์จาก: `CLAUDE.md`, `Consolidated-Context.md`, `docs/06-spec2-cv-training-postprocessing-evaluation-th.md`, module docstring ของทั้ง 27 ไฟล์ใน `src/`, `output/metadata.json`, `output/03_model_configs.json`
- ไม่ได้อ่านโค้ดบรรทัดต่อบรรทัด — หัวข้อ 3 อ้างจาก docstring ระดับ module เป็นหลัก
- คอลัมน์อ้างอิงในหัวข้อ 2 ยึดถ้อยคำในเอกสาร (`who_approved`, "approved", "flagged for objection", "Project owner chose") — จุดที่เอกสารไม่ได้ระบุว่าใครริเริ่มข้อเสนอ ไม่เดาแทน
- อัปเดตครั้งถัดไปเมื่อ: เริ่มร่างรายงาน, `git init`, หรือมีการเบี่ยงเบนจากพารามิเตอร์ที่ล็อกเพิ่ม
