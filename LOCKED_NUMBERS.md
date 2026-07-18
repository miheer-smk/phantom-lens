# LOCKED_NUMBERS.md — PRISM honest leakage-free baseline

**Single source of truth.** Every number destined for the manuscript lives here with full
provenance. Nothing enters the paper that is not in this file, regenerable from a committed script.

## Provenance (baseline block)
- **script:** `Major Revision Results/00_logs/baseline_clean.py`
- **git commit:** `e8f5b7c` (baseline protocol) — rerun regenerates identical numbers
- **seed:** 42 (splits, model, bootstrap)
- **date:** 2026-07-17
- **classifier:** LightGBM (n_estimators=200, max_depth=6, lr=0.05, num_leaves=31,
  min_child_samples=20, class_weight=balanced) — PRISM's documented classifier; **no test-set
  model selection**
- **protocol:** identity-disjoint official FF++ 720/140/140 (`splits/ffpp_official_split.json`);
  overlap assertion enforced (`src/protocol.py`), `identity_overlap=0` on all 9 evaluations
  (`results_clean/protocol_matrix.csv`)
- **n_features:** 50
- **feature CSV SHA-256 (16-char):** original_c23 `fd7c0ac90c046e33` · deepfakes_c23
  `730f064e3a414466` · face2face_c23 `e8c58381b1c14418` · faceswap_c23 `20be0632cf726e23` ·
  neuraltextures_c23 `e569316d209cb5c7` · celebdf `d1be8a4a75515174`
- **CI:** bootstrap 95% (2000 resamples, seed 42)

## Regime 1 — In-distribution, per manipulation (identity-disjoint)
| Manipulation | AUC | 95% CI | F1 | Prec | Recall | MCC | n_train | n_test |
|---|---|---|---|---|---|---|---|---|
| Deepfakes | **0.971** | [0.954, 0.985] | 0.899 | 0.921 | 0.872 | 0.799 | 1383 | 267 |
| FaceSwap | **0.963** | [0.942, 0.980] | 0.892 | 0.853 | 0.948 | 0.789 | 1385 | 269 |
| Face2Face | **0.810** | [0.758, 0.858] | 0.754 | 0.758 | 0.746 | 0.508 | 1386 | 268 |
| NeuralTextures | **0.787** | [0.731, 0.837] | 0.699 | 0.690 | 0.726 | 0.398 | 1386 | 269 |

## Regime 2 — Cross-manipulation, leave-one-manipulation-out (honest generalization)
| Held-out manip | AUC | 95% CI | F1 | MCC | n_train | n_test |
|---|---|---|---|---|---|---|
| Deepfakes | 0.704 | [0.644, 0.765] | 0.616 | 0.253 | 2771 | 267 |
| Face2Face | 0.690 | [0.624, 0.755] | 0.631 | 0.273 | 2768 | 268 |
| FaceSwap | 0.598 | [0.531, 0.666] | 0.553 | 0.109 | 2769 | 269 |
| NeuralTextures | 0.522 | [0.455, 0.592] | 0.521 | 0.058 | 2768 | 269 |

## Regime 3 — Zero-shot cross-dataset, Celeb-DF v2 (never seen in training)
| Train → Test | AUC | 95% CI | F1(macro) | real-rec | fake-rec | MCC | n_train | n_test |
|---|---|---|---|---|---|---|---|---|
| FF++ → Celeb-DF v2 | **0.632** | [0.613, 0.654] | 0.557 | — | 0.781 | 0.140 | 3461 | 6121 (798R/5323F) |

## Headline summary (honest, lockable)
- **In-distribution:** strong on structural manipulations (Deepfakes 0.97, FaceSwap 0.96),
  moderate on subtle ones (Face2Face 0.81, NeuralTextures 0.79).
- **Cross-manipulation generalization:** degrades to 0.52–0.70 — honest, expected, informative.
- **Zero-shot Celeb-DF v2:** 0.63 (above chance, physics-only, no deep learning).

## Retired (leaky — never reported)
FaceSwap 0.9999, NeuralTextures 0.9991, combined "0.9939", Celeb-DF 0.6989. Leakage artifacts /
unreproducible; permanently retired (see `07_summary/repro_gap_report.md`, `PROTOCOL.md`).

## Notes toward Track C targets
Regime 1 F2F/NT (0.81/0.79) and Regime 2 (esp. FaceSwap 0.60, NeuralTextures 0.52) are the
improvement targets. Any Track C gain must be shown on the **validation** split first, then
confirmed once on **test**, never tuned on test or Celeb-DF.

---

## TRACK C — Region-localized ROI features (extended set; original 50 unchanged)
Script `track_c_measure.py` + `track_c_test_confirm.json` · commit `e2a5cb9`+ · seed 42 ·
fit=train ids, decision on val, confirmed ONCE on test · ROI features max_frames=150.
ROI CSV sha256: roi_original `see track_c.json` / roi_face2face / roi_neuraltextures.

**DECISION (on val + Cohen's d):** KEEP G1 (mouth-instability, 3 feats:
roi_mouth_dct_midband_std, roi_mouth_hf_residual_energy, roi_mouth_texture_flicker).
DROP G2 (inner/outer seam) and G3 (motion-texture coupling) — negligible Δ, negligible d.

**Val incremental Δ:** F2F +G1 +0.044 / +ALL +0.046 · NT +G1 +0.105 / +ALL +0.104.
**Cohen's d (val):** roi_mouth_texture_flicker F2F −0.76 / NT −1.37; roi_mouth_dct_midband_std NT −0.99.

**TEST-CONFIRMED (one-time, identity-disjoint, bootstrap 95% CI):**
| Manip | 50-D | 50+G1 | Δ |
|---|---|---|---|
| Face2Face | 0.810 [0.758,0.858] | **0.875 [0.833,0.914]** | +0.065 |
| NeuralTextures | 0.787 [0.731,0.837] | **0.905 [0.866,0.940]** | +0.118 |

Original 50-D baseline (Regime 1) UNCHANGED and remains the locked reference. G1 is an
additive extended set. DF/FS with G1 pending (ROI extraction) to complete the 53-D table.

## c40 EXTRACTION COMPLETE (all 5 sets, for Track D compression + reviewer Exp 3)
ffpp_{original,deepfakes,face2face,faceswap,neuraltextures}_c40.csv — 950–959 rows each.

---

## 53-D EXTENDED SET — FINAL in-distribution (original 50 + G1 mouth-instability)
Script `track_c_53D_full.json` · commit see git · seed 42 · identity-disjoint TEST · bootstrap 95% CI.
G1 = 3 mouth-region features (roi_mouth_dct_midband_std, roi_mouth_hf_residual_energy, roi_mouth_texture_flicker).
G1 HELPS ALL 4 manipulations (never hurts) -> consistent 53-D set justified, no per-manip note needed.

| Manipulation | 50-D (pre-G1 baseline) | 53-D (50+G1, FINAL) | Δ |
|---|---|---|---|
| Deepfakes | 0.971 [0.954,0.985] | **0.978 [0.963,0.989]** | +0.007 |
| FaceSwap | 0.963 [0.942,0.980] | **0.969 [0.949,0.984]** | +0.006 |
| Face2Face | 0.810 [0.758,0.858] | **0.875 [0.833,0.914]** | +0.065 |
| NeuralTextures | 0.787 [0.731,0.837] | **0.905 [0.866,0.940]** | +0.118 |
| **mean per-manip** | 0.883 | **0.932** | +0.049 |

NOTE: the pillar-ablation "full-50 (ref)" rows use the 50-D PRE-G1 baseline (correct for ablation).
The 53-D numbers above are the extended-set in-distribution results. Keep the two clearly distinct.

---

## XCEPTION BASELINE (fair DL comparison, R5.2/R3.4) — identity-disjoint, same protocol as PRISM
Script `xception_train.py` · commit 1337bc8 · seed 42 · legacy_xception (ImageNet-pretrained,
20.8M params, 83MB) · GPU NVIDIA GB10 · video-level mean aggregation · CelebDF zero-shot (complete: 875 real/5612 fake).

| Metric | Xception (DL) | PRISM 53-D (physics) |
|---|---|---|
| FF++ Deepfakes | 0.994 | 0.978 |
| FF++ Face2Face | 0.994 | 0.875 |
| FF++ FaceSwap | 0.994 | 0.969 |
| FF++ NeuralTextures | 0.977 | 0.905 |
| FF++ test overall | 0.990 | (mean per-manip 0.932) |
| **CelebDF zero-shot AUC** | **0.821** (real-rec 0.817, fake-rec 0.689) | **0.632** (real-rec low, fake-rec 0.78) |
| Hardware | GPU (GB10), 83 MB | CPU-only, ~KB model |
| Explainable | No (black box) | Yes (physics + SHAP) |

HONEST FINDING: Xception OUTPERFORMS PRISM on BOTH in-distribution (0.99 vs 0.88-0.98) AND
cross-dataset (0.821 vs 0.632). PRISM does NOT beat a standard deep baseline on accuracy.
=> The paper's contribution must be positioned on INTERPRETABILITY + CPU EFFICIENCY, not
accuracy/generalization superiority. Also note: Xception's cross-dataset failure mode differs
(real-rec HIGH 0.817) — the real-class-mismatch is PRISM-specific, not universal.

---

## EXP-4 THRESHOLD CALIBRATION (R1/R5.3) — CelebDF, thresholds from FF++ VAL only
Script `exp4_calibration.py` · commit 81c6067 · seed 42 · 50-D · celebdf sha d1be8a4a75515174 ·
identity-disjoint assertion PASSED · thresholds/calibrators derived on FF++ val ONLY (test labels never used).

| config | AUC | macro-F1 | real-rec | fake-rec | MCC |
|---|---|---|---|---|---|
| θ=0.50 | 0.632 | 0.557 | 0.397 | 0.781 | 0.140 |
| Youden-J (val) | 0.632 | 0.430 | 0.729 | 0.439 | 0.115 |
| val macro-F1 max | 0.632 | 0.555 | 0.412 | 0.771 | **0.142** |
| val bal-acc max | 0.632 | 0.428 | 0.734 | 0.434 | 0.115 |
| Platt (val) | 0.632 | 0.559 | 0.148 | 0.953 | 0.142 |
| isotonic (val) | 0.632 | 0.545 | 0.109 | 0.971 | 0.138 |

FINDING (null-ish): AUC fixed at 0.632 (threshold is ranking-independent). No threshold/calibrator
FIXES the real-recall collapse — it only TRADES real vs fake recall along the fixed ROC. Youden/
bal-acc raise real-rec to ~0.73 but drop fake-rec to ~0.44 (MCC no better); Platt/isotonic push the
other way. Best MCC (val-macroF1) = 0.142 vs θ=0.5's 0.140 — negligible. => real-recall collapse is a
DOMAIN-SHIFT/ranking problem, not a threshold problem; better thresholding cannot solve it.

---

## EXP-12 FEATURE REDUNDANCY (R3.12) — 50-D, identity-disjoint
Script `exp12_redundancy.py` · commit 92a4235 · seed 42 · drop-decision on train+val importance only; test eval once.

- **Highly-correlated pairs |r|>0.90: only 2 of 1225** →
  s_noise_res_std ~ s_prnu_energy (r=0.958, drop s_prnu_energy);
  s_noise_vmr ~ s_shadow_score (r=0.907, drop s_shadow_score)
- **Near-zero-variance features: none.** VIF max=30.1 (6 features VIF>10 — moderate, not severe).
- **Dropping the 2 redundant features (→48-D) is negligible:**

| manip | full-50 | dedup-48 | Δ |
|---|---|---|---|
| Deepfakes | 0.975 | 0.977 | +0.001 |
| Face2Face | 0.826 | 0.826 | +0.000 |
| FaceSwap | 0.969 | 0.966 | −0.003 |
| NeuralTextures | 0.804 | 0.796 | −0.008 |

FINDING: the 50-feature set is largely NON-redundant — only 2/1225 pairs strongly correlated,
no zero-variance features, negligible AUC change on dedup. Complements: (i) pillar-only (each
domain has standalone power) and (ii) remove-one ablation (functional compensation redundancy).
So: linearly near-independent features, with functional compensation at the pillar level.
Figures: 03_figures/exp12_feature_redundancy/{corr_heatmap.png, dendrogram.png}.

---

## EXP-10 CASE-LEVEL SHAP (R4, R5.6) — 4 principled cases, identity-disjoint
Script `exp10_case_shap.py` + `exp10_signals.py` · commit ce496db · seed 42 ·
Selection: TP=highest-conf correct fake; TN=lowest P_fake correct real; FN=fake w/ lowest P_fake;
FP=CelebDF real w/ highest P_fake. Caveat on every figure: "SHAP explains the classifier's output;
it does not prove a feature causally establishes manipulation."

| case | video | true | P(fake) | top push→fake | top push→real |
|---|---|---|---|---|---|
| TP | 739_865 (Deepfakes) | fake | 0.9996 | roi_mouth_texture_flicker +1.10 | t_skin_texture_corr −0.07 |
| TN | 949 (real) | real | 0.022 | t_dct_temporal_autocorr +0.13 | roi_mouth_texture_flicker −2.12 |
| FN | 128_896 (Face2Face) | fake | 0.078 | s_noise_vmr +0.19 | roi_mouth_texture_flicker −0.98 |
| FP | 00111 (CelebDF real) | real | 0.993 | t_coupling_consistency +1.28 | t_boundary_color_disc −0.30 |

FINDINGS (visible forensic evidence): (TP) G1 mouth-instability drives correct fake detection.
(FN) the F2F fake was MISSED because its mouth stayed stable (mouth features pushed it toward
real) — honest failure mode. (FP) a real CelebDF video is flagged fake because coupling/texture/
noise features read as fake to the FF++-trained model — the cross-dataset real-class mismatch,
made visible. Figures: 03_figures/exp10_case_level_shap/{case_shap_*,case_signal_*,case_signals_all}.png
