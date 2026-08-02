# McNemar + Wilcoxon significance tests (reviewer request)

Post-freeze descriptive; no tuning, no model changes. PRISM = frozen 196-D prob-avg ensemble. Xception =
frozen `xception_best.pt` re-scored by **inference** on the saved crops (the persisted Celeb-DF Xception
predictions are keyed by an unmappable sequential index, so the frozen weight was re-scored to obtain
basename-keyed per-video probabilities — inference, not retraining). Paired by inner-join on shared videos.
McNemar with continuity correction (all discordant counts b+c ≥ 25, so exact binomial was not triggered).
Thresholds: θ=0.50 and each model's **F1-optimal threshold derived on the FF++ validation partition only**
(196-D 0.443 · 53-D 0.384 · 50-D 0.363 · Xception 0.47). Seed 42. Script `exp_mcnemar_wilcoxon.py`.

## T-M1 — McNemar (b = PRISM-correct/baseline-wrong; c = PRISM-wrong/baseline-correct)
| comparison | dataset | θ | n | b | c | χ²(cc) | p | direction | acc PRISM / baseline |
|---|---|---|---|---|---|---|---|---|---|
| 196-D vs **Xception** | Celeb-DF test | 0.50 | 2273 | 587 | 312 | 83.51 | <1e-18 | *PRISM higher-acc* ⚠ | 0.834 / 0.713 |
| 196-D vs **Xception** | Celeb-DF test | F1-opt | 2273 | 536 | 292 | 71.32 | <1e-16 | *PRISM higher-acc* ⚠ | 0.841 / 0.733 |
| 196-D vs **Xception** | FF++ test | 0.50 | 684 | 13 | 91 | 57.01 | 4.3e-14 | **Xception better** | 0.855 / 0.969 |
| 196-D vs **Xception** | FF++ test | F1-opt | 684 | 14 | 94 | 57.79 | 2.9e-14 | **Xception better** | 0.854 / 0.971 |
| 196-D vs 50-D | Celeb-DF test | 0.50 | 2273 | 106 | 89 | 1.31 | 0.252 | n.s. | 0.834 / 0.826 |
| 196-D vs 50-D | Celeb-DF test | F1-opt | 2273 | 45 | 41 | 0.10 | 0.746 | n.s. | 0.841 / 0.839 |
| 196-D vs 53-D | Celeb-DF test | 0.50 | 2273 | 133 | 42 | 46.29 | 1.0e-11 | **196-D better** | 0.834 / 0.794 |
| 196-D vs 53-D | Celeb-DF test | F1-opt | 2273 | 45 | 28 | 3.51 | 0.061 | 196-D (borderline) | 0.841 / 0.833 |

⚠ = accuracy artifact — see the caveat below; this is **not** evidence PRISM detects better than Xception.

## T-M2 — Wilcoxon signed-rank (PRISM-196 vs Xception AUC per fold)
| target | folds | PRISM-196 fold AUC | Xception fold AUC | statistic | p |
|---|---|---|---|---|---|
| Celeb-DF sealed test, identity-grouped CV | 5 | 0.742, 0.747, 0.725, 0.683, 0.687 (mean 0.717) | 0.810, 0.757, 0.834, 0.867, 0.859 (mean 0.825) | 0.0 | 0.0625 |

**Honest note (mandatory):** with 5 folds the minimum achievable two-sided Wilcoxon p is **0.0625**, so this test
**cannot** reach p<0.05 regardless of the effect. Xception's per-fold AUC exceeds PRISM's in **all 5 folds**
(consistent direction), but the test cannot establish significance at n=5. The AUC gap is separately significant by
paired DeLong on the full test set (locked: z=15.4, p<1e-16).

## Factual statement per comparison
1. **PRISM vs Xception, Celeb-DF test:** By **AUC** (ranking), **Xception is better** (0.825 vs 0.717; DeLong
   p<1e-16; Xception wins all 5 CV folds). By **McNemar accuracy** the numbers favour PRISM (0.834 vs 0.713,
   p<1e-16), but this is a **class-imbalance/threshold artifact** (see caveat) — **not** a PRISM win.
2. **PRISM vs Xception, FF++ test:** **Xception significantly better** on both metrics (McNemar p≈3–4e-14;
   accuracy 0.97 vs 0.85; AUC 0.990 vs 0.842). Report plainly.
3. **196-D vs 50-D, Celeb-DF test:** McNemar accuracy **not significant** (p=0.25 / 0.75). The genuine 196-D>50-D
   gain is a **ranking** result (DeLong AUC +0.056, p=1.7e-6), which washes out at fixed thresholds on this set.
4. **196-D vs 53-D, Celeb-DF test:** 196-D significantly better on McNemar accuracy at θ=0.50 (p=1.0e-11);
   borderline at the F1-optimal threshold (p=0.061). Consistent with the DeLong AUC result (+0.030, p=2.2e-9).

## ⚠ CRITICAL CAVEAT (accuracy vs ranking) — flag for authors
McNemar tests **thresholded accuracy**, not ranking. The Celeb-DF sealed test is **83.6% fake**, so a detector that
predicts "fake" for nearly everything scores accuracy ≈ prevalence (0.836). At θ=0.50, PRISM's real-recall is
**0.183** (it flags ~82% of real videos as fake) → its accuracy (0.834) ≈ predict-all-fake. Xception's real-recall
is **0.836** (it genuinely detects reals) but pays an accuracy penalty on the imbalanced set → accuracy 0.713.
**Therefore the Celeb-DF McNemar "PRISM higher-accuracy" reflects the imbalance, not better detection.** By AUC —
the metric the paper reports — **Xception ranks substantially better cross-dataset** (0.825 vs 0.717).
[AUTHORS: reconcile the accuracy(McNemar) vs ranking(AUC/DeLong) divergence in the framing; do not present the
Celeb-DF McNemar as PRISM outperforming Xception.]

## Deliverables
- `results_clean/mcnemar_wilcoxon_results.json` (this table + full 2×2 counts + interpretation block)
- `196D_FINAL/03_results/prism_vs_xception_predictions.csv` — 2,957 rows: `video_name, dataset, ground_truth,
  prism_pred_label, prism_prob, xception_pred_label, xception_prob` (the exact schema requested).
