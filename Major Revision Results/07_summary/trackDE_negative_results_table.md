# Track D + E — complete cross-dataset lever table (the contribution)

56 celebdf_dev evaluations (identity-grouped 5-fold CV; identity-disjoint; sealed test untouched). ΔAUC vs the
relevant base. Bars: in-dist +0.005 (Holm), cross +0.03 (Holm across the full ledger). "cross" = celebdf_dev.

## The headline
**A distributional (order-statistic) representation of per-frame physics features is the ONLY lever that closes
part of the domain gap (cross +0.033, Holm-sig). Feature ADDITION, unsupervised domain ADAPTATION, blending
AUGMENTATION, and diverse-data AUGMENTATION on both classes all FAIL.** Mechanism: the residual gap lives in the
**feature space** (handcrafted physics descriptors don't span Celeb-DF's generative style), not in the classifier
or the quantity of data — which is why more data, alignment, and model tricks don't move it, but a richer
*encoding* of the same signal does.

## Representation (what works)
| # | Lever | Δ in-dist | Δ cross | Verdict |
|---|---|---|---|---|
| 20 | **E1: 53-D + 143 spatial order-statistics (196-D)** | **+0.0092** (Holm) | **+0.0326** (Holm) | ✅ QUALIFIES — the one gain |
| 19 | E1: order-stats replace means | +0.0103 | +0.0259 | ✅ (subsumed by #20) |
| 37–45 | Classifier sweep → RandomForest_d8 | — | 0.7018 (best single) | ✅ RF > LGBM cross |
| 52 | Strong-member RANK ensemble (RF+ET+LGBM) | — | +0.0107 → 0.7125 | ✅ variance reduction (see curse caveat) |

## Feature addition (fails)
| # | Lever | Δ in-dist | Δ cross | Verdict |
|---|---|---|---|---|
| 2 | Group H structure tensor | — | −0.0137 | ✗ |
| 4 | Group J ratios | — | +0.0022 | ✗ |
| 13 | M cardiac coherence | — | −0.0141 | ✗ |
| 14 | Q muscle co-activation | +0.0012 | +0.0001 | ✗ |
| 15 | R blink kinematics | — | −0.0028 | ✗ |
| 16 | T rigid 3-D | — | −0.0104 | ✗ |
| 17 | M+Q+R+T combined | +0.0021 | +0.0070 | ✗ below bar |
| 46 | E4 multi-scale LoG frequency | −0.005 | +0.0055 | ✗ below +0.03 |
| 51 | E5 temporal-difference / rel-flicker (78-D) | −0.006 | −0.0017 | ✗ redundant w/ order-stats |

## Unsupervised domain adaptation (fails — alignment AND boundary)
| # | Lever | Δ cross | Verdict |
|---|---|---|---|
| 7 | CORAL | −0.0396 | ✗ |
| 8–10 | Subspace alignment (d10/20/30) | −0.087/−0.090/−0.055 | ✗ |
| 11 | Per-domain standardisation | −0.0024 | ✗ |
| 12 | Per-domain quantile alignment | −0.0151 | ✗ |
| 21–26 | KS-stability feature selection (top-k) | up to −0.145 | ✗ selection hurts, keep all |
| 47–49 | Pseudo-label self-training (k=10/20/30%) | −0.031/−0.048/−0.057 | ✗ confident errors reinforce |

## Training-distribution & data augmentation (fails on both classes)
| # | Lever | Δ cross | Verdict |
|---|---|---|---|
| 28–30 | Self-blended videos (SBV): R1 only / R2 hybrid | R1 collapses 0.464 / R2 −0.025 | ✗ CNN-specific, not handcrafted |
| 50 | X4 diverse REALS → real class (DFD 226) | −0.020 (realRec +0.052) | ✗ fixes threshold, not ranking |
| 54 | X4 diverse FAKES → fake class (DFD 191) | −0.0098 (realRec −0.042) | ✗ |
| 53 | Denser sampling (100 frames, full pass) | −0.003 | ✗ subset +0.052 didn't replicate |

## Score aggregation & model tricks (fails / marginal)
| # | Lever | Δ cross | Verdict |
|---|---|---|---|
| 31 | Per-manipulation ensemble | +0.0008 | ✗ calibration only |
| 32–36 | Windowed MIL aggregators (E2) | −0.015 (best) | ✗ |
| 55 | Random-subspace feature-bagging | +0.0013 | ✗ member diversity beats it |
| 27 | Drop rPPG ablation | +0.0055 | ✗ below bar (directionally right) |

## Winner's-curse check (methodology)
| # | Lever | Result |
|---|---|---|
| 56 | Joint: celebdf_dev CV vs WildDeepfake AUC | WDF ~0.55–0.58 all candidates; dev ranking ≠ WDF ranking → the ensemble's dev edge does not replicate → curse confirmed (but WDF n=168, SE ±0.045, cannot rank 0.01-separated candidates) |

## Summary counts
- Total dev evaluations: **56**. Qualified (cleared a pre-registered bar): **1** (E1 order-statistics).
- Frozen: 196-D E1-expanded + RF+ET+LGBM rank ensemble. celebdf_dev CV 0.7125.
- Sealed Celeb-DF test evaluations spent: **0** (budget 1). Predicted test 0.68 [0.65, 0.71].
