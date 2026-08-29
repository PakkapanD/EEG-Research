# entropyProfiled_total_sampleEntropy impact test (lightweight, see module docstring)

Full-18 (18 double-banana channels), GroupKFold(5) by subject over the pilot's 13 subjects, 2000 windows (1000 label=1, 1000 label=0 - stratified pilot sample, NOT natural prevalence). Window-level AUC-ROC only - no sens/FA/day claim is possible from this sample. NOT a substitute for a real CV run.

## Per-fold AUC-ROC

| fold | n_test | test subjects | baseline (89 feat/ch) | +entropyProfiled_total (90 feat/ch) | delta |
|---|---|---|---|---|---|
| 0 | 412 | chb12, chb20 | 0.6466 | 0.6145 | -0.0321 |
| 1 | 420 | chb03, chb08, chb14 | 0.9043 | 0.8987 | -0.0056 |
| 2 | 397 | chb09, chb10, chb19 | 0.9270 | 0.9230 | -0.0040 |
| 3 | 360 | chb01+21, chb05 | 0.9129 | 0.9178 | +0.0049 |
| 4 | 411 | chb06, chb13, chb22 | 0.4243 | 0.4373 | +0.0130 |

**Pooled (all folds' held-out predictions concatenated): baseline=0.6629, with entropyProfiled_total=0.6638, delta=+0.0010**

Mean per-fold delta: -0.0048 (2/5 folds improved).
