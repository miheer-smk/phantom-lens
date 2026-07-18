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
