# LOCKED_NUMBERS.md — PRISM honest leakage-free baseline

**Single source of truth.** Every number destined for the manuscript lives here with full
provenance. Nothing enters the paper that is not in this file, regenerable from a committed script.

## Methodology changelog
- **2026-07-21 — M1 imputation-leakage fix (guide #6).** The shared `clean()` idiom previously
  computed `fillna(median)` over an entire feature CSV (train+val+test) **before** partitioning, so
  test rows could influence imputation medians. Replaced with a **train-only imputer**
  (`src/leakfree.py`: medians from the TRAIN partition only, applied unchanged to val/test; zero-shot
  Celeb-DF/WildDeepfake imputed with the FF++ TRAIN median). Applied to all 12 classifier scripts
  (baseline, pillar_ablation, pillar_only, shap_stability, exp3, exp4, exp8, exp9, track_c, run_delong,
  exp11_stats, exp11_prism_vs_xception) + exp10_case_shap. **Impact: none.** The 50-D/ROI/rPPG matrices
  have 0 missing cells and residual CSVs ~1 cell/file, so imputation never (or barely) fires. Re-ran
  every affected script and verified **all 35 locked result files reproduce bit-identically
  (worst abs diff 0.00e+00)** vs the pre-fix snapshot. This is a code-defensibility fix, NOT a
  re-baseline; no reported number changes. Verification: `diff_m1.py`.
- M2 (scaler-in-CV) and M3 (test-set model selection): audited clean — no change needed. `rigorous_search.py`
  confirmed exploratory (writes nothing to disk; feeds no locked number).

## G3 — Runtime / memory profiling (re-profiled on ≥100 videos)
Script `exp5_runtime.py` → `results_clean/runtime.json`, `runtime_profile.csv`, `runtime_per_video.csv`.
Sample bumped 10→12 per source (5 sources × 2 comp) → **113 videos profiled** (was 96). Re-measured,
supersedes the prior run. CPU single-thread, hardware **ARM Cortex-X925 (NVIDIA GB10 SoC)**.
- **mean extraction 48.59 s/video** (was 50.17), **RTF 3.196** (was 3.344), **peak RAM 3462 MB**.
- Stage means (s): frame_load 1.07, mediapipe 2.18, optical_flow 1.26, rppg 3.82, other 40.26.
- 4 feature configs (top-3/10/20/all-50): extraction time identical (stage-shared), classifier
  inference 0.169–0.199 ms, model 502–611 KB.
- **Xception inference: 67.82 ms/video (8-frame, GPU), model 83.2 MB** (vs PRISM ~0.6 MB, CPU).

## G9 — Per-video predictions persisted + statistics recomputed from them (auditability, guide #40)
Scripts `exp_g9_predictions.py` (PRISM, CPU) + `exp_g9_xception_predictions.py` (Xception, GPU) →
`results_clean/predictions_per_video.csv` (**16,635 rows**, schema `video_path, source_id, dataset,
manipulation, compression, true_label, pred_prob, pred_label, split, model, seed`; 0 nulls in key cols).
Coverage: PRISM_50D_indist/53D_indist/50D_LOMO (FF++ test 1073 ea.), PRISM_50D_zeroshot (CelebDF 6121),
PRISM_53D_zeroshot (WildDeepfake 124), Xception (FF++ test 684 + CelebDF 6487).
**Auditability check — statistics recomputed FROM the persisted probs reproduce the locked values exactly:**
- DeLong 53-vs-50 per manip: auc_50/auc_53/Δ/|z|/p **identical** to `delong_53vs50.csv`
  (DF 0.9706→0.9776 p=.219; F2F 0.8096→0.8746 p=9.3e-4; FS 0.9631→0.9691 p=.356; NT 0.7867→0.9049 p=1e-6).
- PRISM-vs-Xception (CelebDF): Xcep **0.8211** / PRISM **0.6322**, z **15.43** — identical to locked.
(`delong_53vs50_from_predictions.csv`, `prism_vs_xception_from_predictions.json`.) eval_wdf.py also moved to
the M1 train-only imputer; `zeroshot_wilddeepfake.json` re-verified **bit-identical** (AUC 0.5212).

## G1 — Hard-negative analysis, CLEAN identity-disjoint Deepfakes TEST (retires leaky exp5)
Script `exp_g1_hardneg.py` → `results_clean/hardneg_deepfakes.json` + figure
`03_figures/expG1_hard_negatives/`. Old `results/exp5` was leaky (trained AND tested on the same full
`ffpp_fake.csv` → 13/957 = 1.36% FN). Clean re-run: identity-disjoint, 133 Deepfakes TEST fakes,
FN = P(fake)<0.5. Two training recipes:
| Recipe | test AUC (real vs DF) | False negatives | P(fake) median |
|---|---|---|---|
| **in-distribution** (real+DF train — the reported 0.9706 detector) | **0.9706** ✓ (matches locked) | **17/133 = 12.78%** | 0.9939 |
| multi-manip (real+all4 train — old exp5 recipe) | 0.9127 | 6/133 = 4.51% | 0.9615 |
The leaky 1.36% understated the FN rate ~9× (test videos were in training). Hardest case `480_389.mp4`
P(fake)=0.0197. In-distribution AUC reproducing the locked 0.9706 confirms same recipe/split.

## M4 — Missingness-as-signal audit (guide #8)
Script `Major Revision Results/00_logs/exp_m4_missingness.py` → `results_clean/missingness_audit.json`,
`missingness_success_rates.csv` (seed 42, commit `e09ffa5`). Rows are video-level; FF++ `residual_*` =
full 1000-video attempted list per set. Validity indicators = per-video extraction-success flags.

**Extraction success (extracted / attempted):** FF++ 50-D 0.956–0.961 (real 0.960), ROI 0.982–0.987,
rPPG 0.981–0.986, residual 1.000 — **near-identical across real and all four fake sets.**
Celeb-DF 50-D: **real 0.912 vs fake 0.948** (attempted 875 real / 5612 fake).
Cell-level missingness within extracted rows: 50-D/ROI/rPPG **0.0%**, residual 0.0063% (5/80000).

**Missingness-ONLY classifier AUC (LogisticRegression, deterministic; chance = 0.50):**
| Target | Features | AUC | 95% CI | Verdict |
|---|---|---|---|---|
| real-vs-fake (FF++, identity-disjoint) | valid_50d/roi/rppg | **0.5009** | [0.482, 0.520] | at chance — no confound |
| real-vs-fake (within Celeb-DF) | valid_50d | **0.5145** | [0.498, 0.532] | CI includes 0.5 — n.s. |
| dataset identity (FF++ vs Celeb-DF) | valid_50d | **0.5114** | [0.504, 0.519] | statistically >0.5 but effect negligible |
(Classifier = LogisticRegression on binary validity indicators — deterministic and reproducible;
LightGBM is not reproducible on these degenerate few-binary-feature problems. Set-iteration order
sorted for full determinism. Values reproduce bit-identically across runs and on a fresh checkout.)

**Conclusion:** Missingness does **not** explain the detector's real-vs-fake performance (AUC ≈ 0.50–0.51,
CIs at/through chance). The honest headline numbers are **not** a missingness artifact. The only
non-trivial observation is a **mild class-dependent extraction gap in Celeb-DF** (reals fail 50-D
extraction ~3.7 pts more than fakes); the dataset-identity AUC is statistically above chance only
because n≈11.5k, with a negligible magnitude (0.512). → author-decision item [M4] (selection-bias
sentence in limitations, compounding the already-disclosed Celeb-DF real-recall domain shift).

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

---

## EXP-3 COMPRESSION c23/c40 — ALL 4 manipulations (R5.4) — 50-D, identity-disjoint
Script `exp3_compression.py` · commit see git · seed 42 · per-manip train(train+val ids)/test(test ids); bootstrap CI.

| manip | c23 AUC | c40 AUC | ΔAUC | c23 MCC | c40 MCC | train-c23→test-c40 |
|---|---|---|---|---|---|---|
| Deepfakes | 0.975 | 0.944 | −0.032 | 0.790 | 0.697 | 0.924 |
| Face2Face | 0.826 | 0.757 | −0.068 | 0.530 | 0.342 | 0.673 |
| FaceSwap | 0.969 | 0.851 | −0.118 | 0.825 | 0.528 | 0.694 |
| NeuralTextures | 0.804 | 0.734 | −0.070 | 0.503 | 0.401 | 0.691 |

Feature-GROUP degradation under c40 (pillar-only AUC, avg over 4 manips; top degraders):
P1_noise −0.097 · T13_motion_blur_coupling −0.090 · T2_rppg −0.070 · T9_skin_texture −0.062 · T5_codec_residual −0.059 · T12_blink −0.058

FINDING: heavy compression (c40) degrades ALL manipulations (Δ −0.03 to −0.12; FaceSwap most,
Deepfakes least). Cross-compression (train c23→test c40) drops further for F2F/FS/NT (compression
mismatch). The pillars that lose most under c40 are the high-frequency/sensor ones (noise, motion-
blur coupling, rPPG, skin texture) — physically expected, as c40 quantises exactly those bands.
Extends the paper's DeepFakes-only compression analysis to all four manipulations.

---

## EXP-5 RUNTIME / MEMORY / COMPLEXITY (R2.2, R2.3, R3.6) — n=95 videos, CPU single-thread
Script `exp5_runtime.py` · commit d6ec61f · seed 42 · single-threaded per video (workers=1) for clean timing.
HARDWARE: ARM Cortex-X925 (NVIDIA GB10 SoC), 20 cores, aarch64, 121 GB RAM, GPU NVIDIA GB10, Linux 6.17.0.
SOFTWARE: python 3.12.3, numpy 1.26.4, scipy 1.15.3, sklearn 1.7.2, lightgbm 4.6.0, opencv 4.11.0,
mediapipe 0.10.18, torch 2.11.0+cu128, timm 1.0.28.

Aggregate (max_frames=300): mean total extraction **50.2 s/video**, **RTF 3.34** (slower than real-time,
single CPU thread), peak RAM **3.37 GB**.
Per-stage means (s): frame_load 1.09 · MediaPipe 2.30 · optical-flow 1.32 · rPPG 4.00 · **other(temporal) 41.5**.

| config | n feat | extract_s | classifier_inf_ms | model_KB | RTF |
|---|---|---|---|---|---|
| top-3 | 3 | 50.2 | 0.20 | 502 | 3.34 |
| top-10 | 10 | 50.2 | 0.18 | 585 | 3.34 |
| top-20 | 20 | 50.2 | 0.18 | 590 | 3.34 |
| all-50 | 50 | 50.2 | 0.21 | 611 | 3.34 |

Xception inference (GPU GB10): 68.2 ms / 8-frame video; model 83.2 MB.

FINDINGS: (1) classifier inference is negligible (0.2 ms) and model is tiny (~0.5-0.6 MB) at any feature
count. (2) Extraction time does NOT drop with feature-count reduction — the top-ranked features are ALL
temporal (t_*), so top-3/top-10 still require the expensive temporal stages (extraction is stage-shared,
not per-feature). (3) Bottleneck is temporal feature computation (~83% of time), not MediaPipe/rPPG.
(4) PRISM: CPU-only, tiny model, but RTF 3.34 (not real-time single-thread). Xception: 68 ms on GPU,
83 MB. Honest tradeoff: PRISM = no GPU + tiny model; Xception = fast but needs GPU + 130x larger model.
This is the hardware/timing detail the manuscript omitted (R2/R3.6).

---

## EXP-8 PRNU-INSPIRED RESIDUAL COMPARISON (R1) — median/gaussian/wavelet; BM3D NOT COMPUTED
Script `exp8_analyze.py` · commit 33550c9 · seed 42 · identity-disjoint · residual descriptors on FF++ c23.
BM3D = NOT COMPUTED (no linux-aarch64 native library — not approximated, per §0).

Descriptor means (real vs fake), median residual: face_energy real 22.8 > fake 17.3; bg_energy real≈fake (57);
face/bg ratio real 0.83 > fake 0.71; face/bg corr real≈fake; temporal_consistency real 0.14 < fake 0.16.
(Gaussian & wavelet show the SAME direction: real face residual energy > fake.)

Classification AUC (base-46 [50-D minus 4 PRNU feats] + method's 5 residual descriptors), per manip:
| residual method | DF | F2F | FS | NT | mean |
|---|---|---|---|---|---|
| median (current) | 0.977 | 0.817 | 0.964 | 0.796 | 0.889 |
| gaussian | 0.973 | 0.830 | 0.970 | 0.806 | **0.894** |
| wavelet | 0.974 | 0.831 | 0.968 | 0.789 | 0.890 |
| current 50-D (ref) | 0.975 | 0.826 | 0.969 | 0.804 | 0.893 |
| BM3D | NOT COMPUTED — library unavailable (aarch64) | | | | |

FINDING: all three residual methods give EQUIVALENT AUC (mean 0.889–0.894), matching the current
50-D reference (0.893). The residual-energy signal is ROBUST to denoising method — the median-filter
choice is not a limitation; any reasonable residual yields the same discrimination. Physical signal:
real faces retain MORE residual energy than fakes (face_energy & face/bg ratio real>fake); manipulation
suppresses sensor residual. This is a PRNU-INSPIRED residual-energy cue, NOT a camera PRNU fingerprint.

AUTHOR REWRITE ITEM (terminology): rename "PRNU" descriptors to "PRNU-inspired residual-energy
descriptors" / "sensor-residual consistency" — no reference sensor pattern is estimated. [flagged, not drafted]

---

## EXP-9 rPPG POS/CHROM COMPARISON (R1) — real-vs-fake AUC per condition, identity-disjoint
Script `exp9_analyze.py` + `exp9_rppg_extract.py` · commit d9105fd · seed 42 · rPPG-only 4 descriptors per method.
current = existing POS+CHROM dual (50-D t_rppg_*); POS/CHROM = pure implementations. BM3D n/a (rPPG expt).

| method | overall | c23 | c40 | low-motion | high-motion | low-illum | high-illum | short-seq | long-seq |
|---|---|---|---|---|---|---|---|---|---|
| current (POS+CHROM) | 0.478 | 0.479 | 0.478 | 0.494 | 0.465 | 0.471 | 0.485 | 0.500 | 0.463 |
| POS | **0.518** | 0.546 | 0.491 | 0.572 | 0.475 | 0.507 | 0.527 | 0.533 | 0.510 |
| CHROM | 0.466 | 0.484 | 0.448 | 0.417 | 0.513 | 0.462 | 0.470 | 0.493 | 0.453 |

FINDINGS (null-ish, honest): (1) rPPG ALONE is a WEAK forensic cue — all methods near chance (0.47–0.52);
none is a strong standalone discriminator. Corroborates DeLong (rPPG hurts cross-dataset) + pillar-only
(weak standalone). (2) POS is marginally best overall (0.518) and MORE compression-robust than CHROM
(POS c23 0.546→c40 0.491; CHROM 0.484→0.448) — validates the extractor's POS-primary design choice.
(3) rPPG degrades under high motion (POS 0.572→0.475) and heavy compression — R1's sensitivity concern
CONFIRMED. (4) pure POS > current dual (0.518 vs 0.478) — the dual fallback isn't optimal.

AUTHOR ITEMS: (a) rPPG is a forensic temporal descriptor, NOT medical-grade pulse estimation.
(b) consider pure POS over the POS+CHROM dual. (c) rPPG's value is a weak complementary cue, not standalone. [flagged, not drafted]

---

## EXP-11 STATISTICAL TESTS WRAP-UP (R3.13) — all p-values from ACTUAL scores, seed 42
Script `exp11_stats.py` + `exp11_prism_vs_xception.py` · commit 0b46c06 · identity-disjoint.

### DeLong (paired AUC, Holm-corrected across family)
| comparison | ΔAUC | z | p | p_holm | sig |
|---|---|---|---|---|---|
| full-50 vs top-3 (DF/F2F/FS/NT) | +0.20/+0.18/+0.11/+0.15 | 4.6–7.6 | ≤5e-6 | **0.000** | ✅ all |
| full-50 vs top-10 (F2F/FS/NT) | +0.066/+0.042/+0.059 | 2.6–3.3 | ≤9e-3 | 0.008–0.026 | ✅ |
| full-50 vs top-10 (Deepfakes) | +0.014 | 2.10 | 0.036 | 0.063 | ~n.s. |
| c23 vs c40 (DF/FS/NT) | +0.031/+0.118/+0.069 | 2.9–5.6 | ≤3.5e-3 | 0.000–0.014 | ✅ |
| c23 vs c40 (Face2Face) | +0.070 | 2.15 | 0.031 | 0.063 | ~n.s. |
| **PRISM vs Xception (CelebDF)** | Xcep 0.821 vs PRISM 0.632 = **+0.189** | 15.43 | <1e-16 | — | ✅ Xception better |

### McNemar (CelebDF, baseline θ=0.50 vs val-calibrated θ=0.510)
stat=25.47, p=4.5e-7. b=54 (only baseline correct) vs c=12 (only calibrated correct) → baseline classifies
MORE correctly; calibration significantly HURTS. Confirms EXP-4 (threshold calibration doesn't help).

### Wilcoxon signed-rank (full-50 vs top-3 across 10 identity-grouped folds)
stat=0.0, p=1.95e-3 (full 0.773 vs top-3 0.644) → full-50 beats top-3 in EVERY fold.

FINDINGS: (1) the full 50-feature model SIGNIFICANTLY outperforms top-3 (all manips, p_holm<0.001) and
top-10 (3/4 manips) — statistically justifies using 50 features (answers "why 50?"). (2) Compression c23→c40
significantly degrades AUC (3/4 manips). (3) Threshold calibration significantly HURTS (McNemar) — not a fix.
(4) Xception significantly outperforms PRISM zero-shot (DeLong z=15.4, p<1e-16) — the interpretability/
efficiency contribution, not accuracy, is confirmed as the honest framing.

## Track D — additive physics features (Phase 1 setup; hypothesis-driven, sealed-set protocol)
Baseline = locked 53-D (additive only; 50-D/53-D never modified). Pre-registration:
`trackD_preregistration.md` (families H/I/J/K + vanishing-point rejection). Anti-overfitting gating:
`src/sealed.py` (sealed sets raise on access unless explicitly unsealed; unseals logged).
- **Celeb-DF dev/test split** (`splits/celebdf_dev_test.json`, `build_celebdf_devtest.py`, seed 42,
  identity-disjoint, 0 shared identities): **dev 2421 (426 real/1995 fake) · test_SEALED 2273 (372/1901)**,
  1427 spanning fakes dropped. FF++ test also sealed; dev uses FF++ train/val + celebdf_dev only.
- **Sealed evaluations performed so far: 0** (budget 1; final count recorded at Phase 4).

### Track D Group H — Gradient Structure Tensor (DEV only; sealed sets untouched)
Script `exp_trackD_H_eval.py` → `results_clean/trackD_H_dev.json`. 10 features, extracted for FF++
train+val + celebdf_dev (`extract_trackD_H.py`, max_frames 40, seed 42); G1/ROI also extracted for
celebdf_dev (`extract_roi_manifest.py`) to give 53-D cross-dataset coverage.
- **In-distribution (FF++ val), 53-D vs 53-D+H:** DF 0.9910→0.9929 (Δ+0.0020), F2F 0.8520→0.8520 (Δ0.0000),
  FS 0.9755→0.9772 (Δ+0.0017), NT 0.8834→0.8849 (Δ+0.0015). **mean Δ +0.0013** (only DF CI excludes 0).
- **Cross-dataset (celebdf_dev, n=2421):** 53-D 0.6312 → 53-D+H **0.6175, Δ −0.0137** (CI [−0.026, −0.001], significantly NEGATIVE).
- **New 53-D-on-celebdf_dev baseline = 0.6312** (≈ 50-D 0.632; G1 adds ~nothing cross-dataset).
- **Verdict: FAILS inclusion.** In-dist +0.0013 < +0.005 threshold; and it HURTS cross-dataset (−0.0137 > −0.005 degradation).
  Pre-registered prediction (in-dist gain, cross-dataset small/uncertain) **partially wrong**: in-dist gain negligible &
  concentrated in DF not NT/F2F; cross-dataset significantly negative. → **Group H NOT included in the frozen set.**

### Track D Group J — Domain-invariant reformulations, CHEAP (DEV only; sealed untouched)
Script `exp_trackD_J_eval.py` → `results_clean/trackD_J_dev.json`. No extraction. J-a = 12 additive
dimensionless ratios among existing magnitude features; J-b = train-fitted quantile alignment of the 53-D rep.
- **J-a ratios, in-distribution (FF++ val):** DF Δ+0.0007, F2F **Δ+0.0140**, FS Δ+0.0021, NT Δ+0.0066; **mean Δ +0.0058**
  (driven by Face2Face; all per-manip CIs include 0).
- **J-a ratios, cross-dataset (celebdf_dev):** 0.6267 → 0.6289, **Δ +0.0022 (n.s., CI[−0.008,0.013])** — fails the +0.005 cross-dataset target.
- **J-b quantile alignment, cross-dataset:** 0.6267 → **0.6371, Δ +0.0104 (CI[0.0016,0.0195], significant)** — in-distribution unchanged
  (monotonic for a tree). **First cross-dataset mover in Track D**, but it is a preprocessing TRANSFORM, not an additive
  physics feature (outside the "additive-only" rule), small and fragile → flagged as a domain-alignment OPTION for authors, needs sealed confirmation.
- **Verdict:** J-a (additive) does NOT meet the cross-dataset target (+0.0022 < +0.005) → not included as an additive family;
  its in-dist +0.0058 (F2F) is noted. J-b (quantile alignment) is a separate representation finding, not folded into the frozen additive set.

### Track D-B — Unsupervised domain adaptation (DEV only; resolves author_decisions #11)
Script `exp_trackD_DA.py` → `results_clean/trackD_DA_dev.json`. 50-D, source=FF++ train, target=celebdf_dev
(alignment fit on UNLABELED target features; labels only score AUC). Distinct from zero-shot full-CelebDF 0.632.
| Method | celebdf_dev AUC | Δ vs zero-shot |
|---|---|---|
| zero-shot (baseline) | 0.6157 | — |
| CORAL | 0.5761 | **−0.0396** |
| Subspace alignment (d=10/20/30) | 0.529/0.526/0.561 | −0.087 / −0.090 / −0.055 |
| per-domain standardisation | 0.6133 | −0.0024 |
| per-domain quantile alignment | 0.6006 | −0.0151 |
**ALL unsupervised DA methods FAIL** (none ≥ +0.03; most hurt). CORAL/subspace alignment actively degrade —
the domain gap is NOT a covariance/subspace shift they can align. Confirms J-b (+0.0104) was noise. → **[#11]
Table 11 CORAL/IFD: the reproducible result is that these do NOT improve cross-dataset transfer; report the
honest negative, or remove Table 11.** Combined with additive-feature failures (G1/H/J-ratios), the Celeb-DF
gap resists BOTH feature-addition AND standard alignment.

### Track D Batch-2 — M/Q/R/T (DEV only; sealed untouched). 53-D base: in-dist 0.9413, celebdf_dev 0.6312.
Script `exp_trackD_MQRT_eval.py` → `results_clean/trackD_MQRT_dev.json`. Unified extractor `extract_trackD_MQRT.py`
(60-frame denser sampling). Holm across 10 family×axis tests; thresholds in-dist +0.005 / cross +0.03.
| Family | in-dist Δ (p_holm) | cross Δ (p_holm) | verdict |
|---|---|---|---|
| M cardiac coherence | −0.0002 (1.0) | −0.0141 (.050) | reject |
| Q muscle co-activation | +0.0012 (1.0) | +0.0001 (1.0) | reject |
| R blink kinematics | +0.0001 (1.0) | −0.0028 (1.0) | reject |
| T rigid 3-D | +0.0011 (1.0) | −0.0104 (.153) | reject |
| ALL combined | +0.0021 (.464) | +0.0070 (1.0) | reject |
**All four fail.** Q per-manip: DF +0.0014, F2F +0.0023, FS +0.0008, NT +0.0028 — **directionally correct
(F2F/NT largest, the region-animation manips) but ~40× too small** to matter; G1 already captures it (redundant).
**TRACK D DEV PHASE CLOSED (STOP condition met):** across 8 pre-registered families + 5 DA methods (17 dev evals),
**no family meets +0.005 in-dist (Holm) or +0.03 cross**. **Frozen model = 53-D (unchanged); sealed evaluations = 0.**
Only region-localized in-distribution features ever helped (G1 +0.118); nothing transfers cross-dataset;
standard UDA fails → strong confirmation the Celeb-DF gap is a robust domain shift, not feature poverty.

### Track E1 — richer temporal aggregation (IN PROGRESS; first run had a keying bug)
First E1 run produced a SPURIOUS +0.0566 in-dist gain from a basename-collision bug: FF++ manipulations
share target_source basenames (e.g. 001_870.mp4 exists under Deepfakes/Face2Face/FaceSwap/NeuralTextures),
and the per-frame extractor keyed by basename only → 827 basenames collapsed up to 4 videos into one
aggregation group, contaminating fake features and inflating F2F/NT. Recovery-by-matching rejected
(only 39% confident, 14% mis-nearest — the 4 manips are too similar to disambiguate). Extractor fixed to
key by FULL video_path; FF++ re-extracting (celebdf_dev was collision-free, reused). **E1 result PENDING
correct re-run — the +0.0566 is NOT a valid number.**

### Track E1 — CORRECTED result (clean full-path keys; supersedes the invalid +0.0566)
Script `exp_trackE_E1_eval.py` → `results_clean/trackE_E1_dev.json`. 143 spatial order-statistics from
per-frame values (`extract_trackE_perframe.py`, full video_path key, 60 frames). 53-D base: in-dist 0.9434,
celebdf_dev 0.6273. Dev only; sealed = 0.
| Model | in-dist Δ (p_holm) | cross Δ (p_holm) | per-manip DF/F2F/FS/NT |
|---|---|---|---|
| E1_replace (143 + 37t + 3 G1) | +0.0103 (.003) | +0.0259 (.004) | +.002/+.025/+.010/+.013 |
| E1_additive (53-D + 143) | +0.0092 (.003) | **+0.0326 (.000)** | +.001/+.020/+.010/+.015 |
**E1_additive is the FIRST Track D+E family to clear BOTH bars** (+0.005 in-dist Holm AND +0.03 cross Holm):
celebdf_dev 0.627→0.660, in-dist gain concentrated in F2F/NT (intermittent-artifact manips, as predicted).
Order statistics recover intermittent-frame signal the mean discards, and it transfers cross-dataset.
CAVEATS: dev-only (needs the single sealed confirmation, budget 1, still unused); still modest (0.66, not the
0.70 target); broader-multiplicity Holm across all ~20 D+E families should be applied before freezing. → **CANDIDATE
for the frozen extended set / sealed evaluation** (author-decision — first real cross-dataset mover).

### Track E X1/X2 (on E1-expanded 196-D; DEV only). full: in-dist 0.9526, cross 0.6599.
`exp_trackE_X1X2.py` → `trackE_X1X2_dev.json`. X1 KS-stability selection FAILS (top-k hurts: k=20 cross 0.518 …
k=196 0.663 — domain-stable≠discriminative; keep all features). X2 drop-rPPG: in-dist +0.0019, cross +0.0055
(directionally right, rPPG mildly hurts Celeb-DF, but below +0.03 bar → not an independent qualifier). Neither
qualifies; E1 remains frozen base. Most domain-stable feats = low-tail order stats (min/p10 of flow/compression).

### Track E3 — SBV pre-flight gate: PASS (blend visible to physics descriptors)
`extract_trackE_SBV.py --preflight 50` → `trackE_SBV_preflight.json`. Cohen's d real-vs-SBV, n=50, jitter 1.0.
Boundary/T8: t_texture_warp_residual +3.45, t_boundary_freq_leakage +2.09, t_boundary_color_disc +1.82,
t_boundary_grad_temporal +0.94, t_face_ssim_mean −0.83. **max boundary |d| = 3.45 > 0.5 → PASS**, full extraction justified.
Notable: the TOP overall discriminators are E1 ORDER STATISTICS (s_noise_hf_ratio__std 4.3, s_prnu_energy__skew 4.0,
s_noise_res_std__skew 3.1) — the distributional representation that won E1 cross-dataset also best registers the SBV blend
(E1 and E3 pull the same lever). → full SBV extraction launched (R0/R1/R2, jitter 0.5 & 1.0).

### Track E3 — SBV training regimes: FAILED (SBI does not transfer to handcrafted features). DEV; sealed=0.
`exp_trackE_E3_eval.py` → `trackE_E3_dev.json`. 196-D E1-expanded via CONSISTENT full_features (60fr).
Selection = identity-grouped 5-fold CV within celebdf_dev.
| Regime | in-dist | celebdf_dev CV | real-rec | fake-rec |
|---|---|---|---|---|
| R0 real-vs-FFpp (current) | 0.865 | **0.6967 ±.031** | 0.225 | 0.948 |
| R1 real-vs-SBV only | 0.530 | **0.4638** (COLLAPSED — predicts all real) | 0.979 | 0.013 |
| R2 hybrid | 0.858 | 0.6719 | 0.242 | 0.915 |
**SBV FAILS:** R1 collapses (handcrafted features trained on synthetic self-blends do NOT recognise Celeb-DF's
real-deepfake blends → all Celeb-DF called real), R2 hurts vs R0. The SBI training-distribution effect is
CNN-specific; it does NOT reproduce with handcrafted physics descriptors (novel negative for the paper).
**Silver lining:** R0 on the CONSISTENT full-196-D E1-expanded features = **0.6967 dev CV** — highest yet,
approaching 0.70 (vs E1 mixed-extraction 0.660); the consistent 60-frame extraction + order-stat representation
is the real gain. **Real recall 0.225 confirms the real-class bottleneck → X4 (diverse reals) is the right lever.**

### Track E — per-manip ensemble & E2 windowed MIL: both FAIL to beat R0 (DEV; sealed=0)
- Per-manip ensemble (`trackE_ensemble_dev.json`): avg of real-vs-{DF,F2F,FS,NT} = celebdf_dev CV **0.6975** (Δ+0.0008
  vs R0 0.6967 — no AUC gain). real-recall 0.225→0.742 is a CALIBRATION shift (averaging softens fake-bias), NOT a
  ranking gain (AUC unchanged). FaceSwap-only transfers worst (0.509≈chance); NT/DF best (0.678/0.667).
- E2 windowed MIL (`trackE_E2_dev.json`, 30-frame windows/stride 15): best aggregator max = **0.6818**, **−0.0149 vs R0**.
  Window order-stats noisier than full 60-frame; MIL doesn't recover. Both reject. **R0 (0.6967 dev CV) stays best.**

### Track E — classifier/regularization sweep: RandomForest transfers best (DEV; sealed=0)
`exp_trackE_clfsweep.py` → `trackE_clfsweep_dev.json`. 196-D R0 rep, select by celebdf_dev CV.
**RandomForest(d8) = 0.7018 ±0.050 (BEST, first >0.70 dev CV)** vs LightGBM_d6 0.6967 (+0.0051, within CV std).
Confirms pre-revision RF>LGBM cross + mechanism (GBDT absolute-threshold splits overfit source domain; RF bagging
transfers). LGBM strong-reg 0.6977 (reg helps marginally). Very-smooth models FAIL: LogReg 0.595, RBF-SVM 0.549
(collapsed, all-fake) — underfit the nonlinear 196-D. → switch frozen classifier to RandomForest; improvement modest.

### Track E4 — LoG frequency: +0.0055 cross (sub-threshold, REJECTED). DEV; sealed=0.
`exp_trackE_E4_eval.py` → `trackE_E4_dev.json`. 18 multi-scale LoG features (energy/entropy/kurtosis/face-bg
ratio × 4 sigmas) stacked on 196-D, RandomForest. 196-D 0.7018 → +E4 **0.7073** (Δcross +0.0055; in-dist −0.005).
Below the +0.03 cross bar AND in-dist negative → does NOT qualify. Frozen candidate stays **196-D + RandomForest
= 0.7018 dev CV**. Feature/model levers plateauing at ~0.70–0.71; X4 (real-recall) is the remaining lever.

### Track E — pseudo-label self-training (transductive UDA): FAILS. DEV; sealed=0.
`exp_trackE_selftrain.py` → `trackE_selftrain_dev.json`. RandomForest, honest identity-grouped CV (pseudo-label
train folds, eval held-out). R0 0.7018 → top-10% **0.6707** (−0.031), top-20% 0.6538 (−0.048), top-30% 0.6450 (−0.057).
Self-training AMPLIFIES the real-class bias: the base model confidently mis-labels Celeb-DF reals as fake, so the
most-confident pseudo-labels reinforce the error. Boundary adaptation fails for the same reason feature alignment did.
Reported as transductive DA, distinct from inductive zero-shot. → adds to the DA-fails evidence (Track D-B + this).

### Track E — X4 diverse-real augmentation (DFD/Google reals -> real class): FAILS cross-AUC; clean finding. DEV; sealed=0.
`exp_trackE_X4_eval.py` -> `trackE_X4_dev.json`. Add unrelated-corpus reals (DFD 28 actors, 226 usable of 363;
137 failed = 38%, full-scene 1080p clips) to the TRAINING real class. 196-D R0 (60-frame), RandomForest,
identity-grouped 5-fold celebdf_dev CV. Celeb-DF reals stay SEALED.
- base(0 DFD):  in-dist 0.8179 | cross CV **0.7018** ±0.050 | realRec 0.232 | fakeRec 0.942
- half(+113):   in-dist 0.8119 | cross CV 0.6908 (-0.011) | realRec **0.291** | fakeRec 0.906
- all(+226):    in-dist 0.8018 | cross CV **0.6814** (-0.020) | realRec 0.284 | fakeRec 0.894
**Finding:** diverse reals raise REAL RECALL (+0.052, threshold/calibration) but do NOT raise cross-AUC — they
slightly DEGRADE ranking (dose-response monotonic: more DFD -> lower AUC), and cost in-dist (-0.016). This
REFUTES the pre-registered "widen real class -> higher cross-AUC" hypothesis and CONFIRMS the author framing
(author_decisions.md): the Celeb-DF real-recall deficit was a THRESHOLD artifact, not a ranking one; adding
diverse reals moves the operating point without improving discrimination. X4 does NOT qualify (bar +0.03; goes
the wrong way). Frozen candidate unchanged = 196-D + RandomForest = **0.7018 dev CV**. Caveat: 38% DFD extraction
failure (face-cropped DFD would clean the input) but the negative direction is consistent across both doses.

### Track E — E5/TD temporal-difference / relative-flicker representation: FAILS. DEV; sealed=0. Zero extraction.
`exp_trackE_tempdiff.py` -> `trackE_tempdiff_dev.json`. 13 spatial channels x 6 diff-stats (meanabs/std/p90abs/
max/relflick[dimensionless]/signchg) = 78 TD features from persisted per-frame series. RF, identity-grouped celebdf_dev CV.
- 196-D base:  in-dist 0.8179 | cross **0.7018** ±0.050 | realRec 0.232 fakeRec 0.942
- 196+78 TD:   in-dist 0.8117 | cross 0.7001 (**Δ-0.0017**) | realRec 0.239 fakeRec 0.957
- TD-only 78:  in-dist 0.7152 | cross 0.5809 | realRec 0.099 fakeRec 0.957
**Finding:** REFUTES the pre-registered "offset-invariant differences transfer" hypothesis. TD adds nothing
(-0.0017 cross, -0.006 in-dist); TD-only weak (0.58). Mechanism: the order-statistics already in the 196-D
(std/IQR/p90) capture the transferable temporal-variability signal -> frame-difference stats are redundant with
them. Does NOT qualify (bar +0.03). Frozen candidate unchanged = 196-D + RF = **0.7018 dev CV**.

### Track E — strong-member ensemble + ExtraTrees: MODEST GAIN. DEV; sealed=0. Zero extraction.
`exp_trackE_ens2.py` -> `trackE_ens2_dev.json`. 196-D R0, identity-grouped celebdf_dev CV. Retest of ensembling
with ONLY strong members (the clfsweep naive avg with weak LogReg/SVM had failed at 0.684).
- RF_d8 0.7018 | ExtraTrees **0.7036** | LGBM_d6 0.6967
- RF+ET prob 0.7050 / rank 0.7055 | RF+ET+LGBM prob 0.7092 / **rank 0.7125**
**Finding:** strong-member RANK-averaged ensemble (RF+ExtraTrees+LGBM_d6) = **0.7125 dev CV, +0.0107 vs RF**.
Genuine variance reduction (every ensemble > every member; monotonic in members). ExtraTrees alone also beats RF
(+0.0018). recall-at-0.5 on rank scores is meaningless (uncalibrated); AUC is threshold-invariant. Running-best
dev candidate: 196-D + RF+ET+LGBM rank ensemble = 0.7125. MULTIPLICITY NOTE below.

**MULTIPLICITY NOTE (2026-07-27, after 52 dev evals):** the dev-eval ledger now has 52 entries. Every additional
selection on celebdf_dev inflates the dev->test optimism gap. The pre-registration estimated a 0.02-0.04 gap at
~30 evals; at 52 evals the realistic gap is likely 0.03-0.05. The ensemble's +0.0107 is principled (variance
reduction, monotonic) but part of any small dev gain at this stage may not survive to sealed test. DISCIPLINE:
after the two remaining substantive levers land (100-frame full pass; DFD-fakes-in-training if pursued), FREEZE —
apply the fixed inclusion rule, pre-register a predicted test point + interval, and spend the ONE sealed eval.
Continued dev micro-optimization past that point has rising overfitting risk and falling real value.

### Track E — denser sampling (100 frames) FULL PASS: FAILS at scale. DEV; sealed=0.
`exp_trackE_frameval.py` on full 100-frame re-extraction (`plain_everyone_100.csv`, all 6547 videos, ~17h).
celebdf_dev CV = **0.6988** ±0.055 (realRec 0.216 fakeRec 0.947) vs 60-frame R0 **0.7018** -> **Δ -0.003**.
The +0.052 subset signal did NOT replicate at full scale: more frames helped only when the training set was
scarce (300 videos); on the full 3461-video training set the marginal benefit vanished (slightly negative).
Denser sampling REJECTED as a lever. Confirms the small-sample caveat flagged before committing the full pass.

### Track E — X4-FAKES diverse fake augmentation (DFD fakes -> fake class): FAILS. DEV; sealed=0.
`exp_trackE_X4fakes_eval.py` -> `trackE_X4fakes_dev.json`. Add DFD high-quality fakes (191 usable of 300; 109
failed = 36% full-scene) to the FAKE class. 196-D R0, RF, identity-grouped celebdf_dev CV.
- base 0.7018 (realRec 0.232) -> +95 0.7005 (realRec 0.202) -> +191 **0.6920** (realRec 0.190). Δcross -0.0098, in-dist -0.003.
**Finding:** adding diverse fakes made the model MORE fake-biased (real recall DROPPED 0.232->0.190), cross-AUC down.
Combined with X4-reals (-0.020): diverse-data augmentation FAILS on BOTH classes. Reals lift recall but hurt AUC;
fakes hurt both. The domain gap resists data augmentation on either side. Does NOT qualify. Frozen candidate unchanged.

### Track E — random-subspace (feature-bagging) ensemble: MARGINAL. DEV; sealed=0. Zero extraction.
`exp_trackE_subspace.py` -> `trackE_subspace_dev.json`. M members on random 50% feature subsets, rank-averaged.
- RF M15 0.7019 / M30 **0.7031** | ExtraTrees M15 0.7014 / M30 0.7021.
**Finding:** feature bagging gives +0.0013 over single RF but is WORSE than the model-diversity rank ensemble
(0.7125, -0.0094). Feature bagging alone doesn't beat member diversity. Running-best dev unchanged = **0.7125**.

### Track E — JOINT SELECTION (celebdf_dev CV + WildDeepfake AUC): winner's-curse CONFIRMED. DEV; sealed=0.
`exp_trackE_joint.py` -> `trackE_joint_dev.json`. Second independent target: WildDeepfake 196-D (168 videos,
88 real/80 fake; `extract_wdf_196d.py`). FF++-trained, inductive.
| candidate | celebdf_dev CV | WildDeepfake AUC | mean |
|---|---|---|---|
| RF_d8 | 0.7018 | 0.5751 | 0.6384 |
| ExtraTrees | 0.7036 | **0.5834** | 0.6435 |
| LGBM_d6 | 0.6967 | 0.5476 | 0.6221 |
| RF+ET_rank | 0.7055 | 0.5794 | 0.6424 |
| RF+ET+LGBM_rank | **0.7125** | 0.5746 | 0.6436 |
**Findings:** (1) WildDeepfake AUC ~0.55-0.58 for ALL candidates — physics features barely transfer to a
genuinely different (in-the-wild) domain. (2) The celebdf_dev ranking does NOT match the WildDeepfake ranking:
the rank ensemble is BEST on celebdf_dev (0.7125) but MID-PACK on WildDeepfake (0.5746), while ExtraTrees is
BEST on WildDeepfake (0.5834) and 2nd on celebdf_dev. => the ensemble's +0.0107 celebdf_dev edge is NOT
corroborated by the independent target = winner's-curse inflation empirically confirmed. Joint-mean best
(ensemble 0.6436) ties ExtraTrees (0.6435). Freeze implication: predicted celebdf_test should be CONSERVATIVE;
ExtraTrees is the most domain-robust single model. WildDeepfake is small (168 videos) -> wide CI (~±0.05).

### Track E — Test-Time Augmentation (TTA): FAILS. DEV; sealed=0.
`extract_trackE_TTA.py` (7263/7263 ok) + `exp_trackE_TTA_eval.py` -> `trackE_TTA_dev.json`. Mean predicted prob
over {original + N augmented (JPEG/scale/brightness/blur)} per celebdf_dev video.
- RF:       base 0.7018 -> TTA2 0.7006 -> TTA3 0.6985 (monotonic -0.003)
- ensemble: base 0.7125 -> TTA2 0.7113 -> TTA3 **0.7088** (monotonic -0.0037)
**Finding:** TTA dilutes rather than denoises — augmentation perturbs celebdf features in non-ranking-helpful
ways. LAST lever; does NOT change the freeze. Frozen candidate stays 196-D + RF+ET+LGBM rank ensemble = 0.7125.
