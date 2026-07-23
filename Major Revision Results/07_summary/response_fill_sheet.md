# Response-table FILL SHEET (values only — NO prose/framing; that is author-owned)

Maps the co-author's response tables **H1–H9** and the **19 `[INSERT VERIFIED RESULT]` placeholders**
to exact locked values. Canonical source: `LOCKED_NUMBERS.md`. Machine-readable twin:
`response_fill_sheet.json`. Protocol: identity-disjoint FF++ 720/140/140, seed 42, bootstrap 95% CI.

## H1 — Leakage-free FF++  (`baseline.json`, `track_c_53D_full.json`)
| Manipulation | AUC 50-D | AUC 53-D | 95% CI (53-D) | Δ(53−50) |
|---|---|---|---|---|
| Deepfakes | 0.9706 | **0.9776** | [0.9632, 0.9893] | +0.0070 |
| Face2Face | 0.8096 | **0.8746** | [0.8326, 0.9142] | +0.0650 |
| FaceSwap | 0.9631 | **0.9691** | [0.9494, 0.9843] | +0.0060 |
| NeuralTextures | 0.7867 | **0.9049** | [0.8662, 0.9403] | +0.1181 |
| **mean** | **0.883** | **0.932** | — | — |
Cross-manip LOMO: DF 0.7039 · F2F 0.6904 · FS 0.5978 · NT 0.5221.
Cross-dataset zero-shot: **Celeb-DF 0.632 [0.613, 0.654]** (n=6121) · WildDeepfake 0.5212.

## H2 — Group ablation  (`pillar_ablation.csv`, `pillar_only.csv`)
Remove-one-pillar: **no single pillar significant after Holm** (redundancy). Standalone examples
(Deepfakes): P1_noise 0.7299 [0.671,0.788] · P2_prnu 0.7068 · T12_blink 0.7013 (full-50 ref 0.9706).

## H3 — Compression c23/c40  (`compression_all_manips.csv`)
| Manip | c23 AUC | c40 AUC | Δ | c23→c40 cross |
|---|---|---|---|---|
| Deepfakes | 0.9752 | 0.9437 | −0.0315 | 0.9242 |
| Face2Face | 0.8255 | 0.7574 | −0.0681 | 0.6727 |
| FaceSwap | 0.9689 | 0.8507 | **−0.1182** | 0.6943 |
| NeuralTextures | 0.8039 | 0.7343 | −0.0696 | 0.6910 |

## H4 — Threshold calibration  (`calibration.json`; thresholds from FF++ val ONLY)
**NULL result:** AUC fixed at **0.6322** across every threshold; thresholds only trade recall.
θ=0.50 → real-rec 0.3972 / fake-rec 0.7813; Youden(val) → real-rec 0.7293 / fake-rec 0.4390.
McNemar baseline-vs-calibrated: χ²=25.47, **p=4.49e-07 (calibration WORSE, not better)**.

## H5 — SHAP stability  (`shap_stability.json`)
Mean cross-fold Spearman **0.911**. Cross-manipulation Spearman **0.069–0.358**
(DF~F2F .358 · DF~FS .212 · DF~NT .069 · F2F~FS .256 · F2F~NT .131 · FS~NT .072).

## H6 — Runtime / memory  (`runtime.json`, `runtime_profile.csv`; n=113 videos)
**48.59 s/video, RTF 3.196, peak RAM 3462 MB.** Stages (s): frame_load 1.07 · mediapipe 2.18 ·
optical_flow 1.26 · rppg 3.82 · other 40.26. Configs (extract time identical): top-3 502 KB/0.169 ms ·
top-10 585 KB/0.173 ms · top-20 590 KB/0.177 ms · all-50 611 KB/0.199 ms.
Xception: **67.82 ms/video (8-frame, GPU), 83.2 MB.**

## H7 — External datasets + DL baseline  (`xception_baseline.json`, `zeroshot_wilddeepfake.json`)
PRISM: Celeb-DF 0.632 · WildDeepfake 0.5212. Xception: FF++ **0.990** (0.9898) · Celeb-DF **0.821** (0.8207),
83.2 MB, not explainable. DeLong PRISM-vs-Xception (Celeb-DF): Xception +0.1889, **z=15.43, p<1e-16**.

## H8 — Feature redundancy  (`redundancy.json`)
**2 of 1225 pairs** with |r|>0.90; VIF max 30.1; dropping the 2 → AUC change negligible
(e.g. DF 0.9752→0.9766, FS 0.9689→0.9657).

## H9 — Statistical comparisons  (`delong_53vs50.csv`, `statistical_tests.json`, `prism_vs_xception_from_predictions.json`)
DeLong 53-vs-50: DF Δ+0.007 z1.229 p.219 · F2F Δ+0.065 z3.312 **p9.3e-4** · FS Δ+0.006 z0.923 p.356 ·
NT Δ+0.118 z4.878 **p1e-6**. Full-50 vs top-3 (DF): Δ+0.20, z7.626, **p_holm<0.001**;
Wilcoxon (10 folds) mean 0.7728 vs 0.6443, p1.95e-3. McNemar calibration χ²25.47 p4.5e-7.
PRISM-vs-Xception (Celeb-DF, from persisted per-video probs): 0.8211/0.6322, z15.43.

---

## The 19 `[INSERT VERIFIED RESULT]` placeholders
| # | Value | Provenance |
|---|---|---|
| P1 | FF++ in-dist mean 50-D = **0.883** | baseline.json |
| P2 | FF++ in-dist mean 53-D = **0.932** | track_c_53D_full.json |
| P3 | Celeb-DF zero-shot = **0.632** [0.613,0.654] | baseline.json regime3 |
| P4 | WildDeepfake zero-shot = **0.5212** | zeroshot_wilddeepfake.json |
| P5 | Xception FF++ = **0.990** (0.9898) | xception_baseline.json |
| P6 | Xception Celeb-DF = **0.821** (0.8207) | xception_baseline.json |
| P7 | PRISM-vs-Xception: Δ+0.189, z=15.43, p<1e-16 | prism_vs_xception_from_predictions.json |
| P8 | NT 53-D gain 0.787→0.905 (Δ+0.118, p=1e-6) | delong_53vs50.csv |
| P9 | F2F 53-D gain 0.810→0.875 (Δ+0.065, p=9.3e-4) | delong_53vs50.csv |
| P10 | Runtime 48.59 s/video, RTF 3.196 | runtime.json |
| P11 | Peak RAM 3462 MB | runtime.json |
| P12 | Model size PRISM 611 KB vs Xception 83.2 MB | runtime_profile.csv, xception_baseline.json |
| P13 | Redundancy 2/1225 pairs \|r\|>0.9; dedup negligible | redundancy.json |
| P14 | Calibration null: AUC 0.6322 fixed; McNemar p=4.5e-7 (worse) | calibration.json, statistical_tests.json |
| P15 | SHAP cross-fold Spearman 0.911; cross-manip 0.069–0.358 | shap_stability.json |
| P16 | c40 worst FaceSwap Δ−0.118; DF Δ−0.032 | compression_all_manips.csv |
| P17 | Full-50 vs top-3: Δ+0.20, z=7.63, p_holm<0.001 | statistical_tests.json |
| P18 | Missingness-only real-vs-fake AUC **0.5009** [0.482,0.520] (at chance) | missingness_audit.json |
| P19 | Clean hard-negatives **17/133** FN (retires 13/957) | hardneg_deepfakes.json |

*Assignment of P-numbers to the document's placeholder positions is by topic; if the co-author's
placeholders are numbered/located differently, match by the value description. Prose/framing is theirs.*
