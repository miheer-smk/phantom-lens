# Publication-ready tables — Track E (196-D) programme

All cross-dataset numbers are **sealed Celeb-DF-v2 test** unless labelled *dev*. Frozen model: 196-D E1-expanded
representation + RF+ExtraTrees+LGBM_d6 rank ensemble, trained on FF++ train only, seed 42. Post-freeze
measurements are descriptive (no selection after unseal).

## T1 — Cross-dataset, sealed Celeb-DF-v2 test (identical data, all models)
| representation | AUC | 95% CI | real rec | fake rec | DeLong vs 196-D |
|---|---|---|---|---|---|
| 50-D | 0.6573 | [0.630, 0.685] | 0.183 | 0.952 | p = 1.7×10⁻⁶ |
| 53-D | 0.6830 | [0.656, 0.718] | 0.274 | 0.895 | p = 2.2×10⁻⁹ |
| 196-D | 0.7133 | [0.687, 0.746] | 0.183 | 0.961 | — |

*Caption:* n = 2,273 sealed-test videos (372 real / 1,901 fake), 27 identities disjoint from the dev half.
DeLong is a **paired** test on the shared test set (same videos scored by both models); it accounts for the
correlation between the two AUC estimates, which is why the differences are significant even though the marginal
95% CIs overlap. Recall is at θ = 0.5 on the mean-probability ensemble; AUC is threshold-free.

## T2 — In-distribution FF++ test: per-manipulation and pooled (both models)
| model | DF | F2F | FS | NT | mean-of-4 | pooled | pooled 95% CI |
|---|---|---|---|---|---|---|---|
| 196-D | 0.907 | 0.796 | 0.831 | 0.833 | 0.8420 | 0.8420 | [0.802, 0.880] |
| 53-D | 0.911 | 0.787 | 0.807 | 0.838 | 0.8358 | 0.8358 | [0.796, 0.874] |

*Caption:* FF++ official test split, 685 videos (137 real / 548 fake). Per-manipulation AUC = real vs that
manipulation. "mean-of-4" is the mean of the four per-manipulation AUCs (the quantity the paper's 0.932 reports);
"pooled" is the single AUC over all fakes vs reals. Frozen ensemble; post-freeze descriptive.

## T3 — In-distribution vs cross-dataset (descriptive trade-off)
| representation | FF++ test (in-dist, mean-of-4) | Celeb-DF-v2 sealed test (cross) |
|---|---|---|
| 53-D | 0.8358 | 0.6830 |
| 196-D | 0.8420 | 0.7133 |

*Caption:* both axes are held-out test sets; classifier held constant (frozen ensemble). [AUTHORS: interpretation
of the in-distribution / cross-dataset trade-off]

## T4 — Negative results (systematic; the tested-and-rejected levers)
Δ = change vs the relevant base on celebdf_dev CV (cross) / FF++ val (in-dist). Bars: in-dist +0.005 (Holm),
cross +0.03 (Holm across the 57-eval ledger). Dev-set measurements.
| category | lever | Δ in-dist | Δ cross | Holm p (where run) | verdict |
|---|---|---|---|---|---|
| feature addition | Group H structure tensor | — | −0.0137 | — | ✗ |
| feature addition | Group J ratios / quantile | — | +0.0022 / +0.0104 | — | ✗ |
| feature addition | M cardiac coherence | — | −0.0141 | .050 | ✗ |
| feature addition | Q muscle co-activation | +0.0012 | +0.0001 | — | ✗ |
| feature addition | R blink kinematics | — | −0.0028 | — | ✗ |
| feature addition | T rigid 3-D | — | −0.0104 | .153 | ✗ |
| feature addition | M+Q+R+T combined | +0.0021 | +0.0070 | 1.0 | ✗ |
| feature addition | E4 multi-scale LoG | −0.005 | +0.0055 | — | ✗ below bar |
| feature addition | E5 temporal-difference | −0.006 | −0.0017 | — | ✗ |
| domain adaptation | CORAL | — | −0.0396 | — | ✗ |
| domain adaptation | subspace align (d10/20/30) | — | −0.087/−0.090/−0.055 | — | ✗ |
| domain adaptation | per-domain standardise | — | −0.0024 | — | ✗ |
| domain adaptation | per-domain quantile | — | −0.0151 | — | ✗ |
| domain adaptation | self-training (k=10/20/30%) | — | −0.031/−0.048/−0.057 | — | ✗ |
| training-augmentation | SBV self-blended video (R1/R2) | — | R1 collapse 0.464 / R2 −0.025 | — | ✗ |
| training-augmentation | X4 diverse reals (DFD) | −0.016 | −0.020 | — | ✗ |
| training-augmentation | X4 diverse fakes (DFD) | −0.003 | −0.0098 | — | ✗ |
| training-augmentation | denser sampling 100-frame (full) | — | −0.003 | — | ✗ |
| inference-time | TTA (N=2/3) | — | −0.001 / −0.004 | — | ✗ |
| ensembling | random-subspace bagging | — | +0.0013 | — | ✗ marginal |
| ensembling | per-manip ensemble | — | +0.0008 | — | ✗ calibration only |
| aggregation | E2 windowed MIL | — | −0.0149 (best) | — | ✗ |
| selection | X1 KS-stability (top-k) | — | up to −0.145 | — | ✗ |
| selection | X2 drop-rPPG | +0.0019 | +0.0055 | — | ✗ below bar |

*Caption:* qualified levers (not shown here): E1 order-statistics (cross +0.033, Holm-sig) and the rank ensemble.
[AUTHORS: framing of the negative-results contribution]

## T5 — Protocol disclosure
| item | value |
|---|---|
| dev-evaluation count | 57 (ledger: `trackD_dev_evals.txt`) |
| sealed evaluations | 1 of 1 (spent; crash-and-recapture, see SEALED_PROVENANCE.md) |
| pre-registered prediction | 0.68, 80% interval [0.65, 0.71] |
| actual sealed test | 0.7133 (above interval; under-predicted) |
| Celeb-DF split | identity-disjoint; dev 2,421 / test_SEALED 2,273 / 1,427 spanning fakes dropped; balanced spectral (Fiedler) component cut |
| FF++ split | official identity split (train 720 / val 140 / test 140 identities) |
| seed | 42 (RF/ET random_state 42; LGBM deterministic) |
| feature imputation | train-only median; StandardScaler fit on train |
