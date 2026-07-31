# LOCKED_NUMBERS_196D — every reported number with full provenance

Seed 42 throughout. Feature SHA-256 (first 16 hex): plain_everyone_E3 `6da42d5d2d723c64`, plain_celebdf_test
`df2e42adec4cfa90`, plain_ffpp_test `70f5ab0d0d67b6e8`, wdf_196d `d23cf8b09dbf3705` (full in
`02_features/SHA256_MANIFEST.txt`). "cross/dev" = celebdf_dev identity-grouped 5-fold CV.

## Frozen configuration
- Representation: 196-D E1-expanded (13 spatial means + 37 temporal + 3 G1 + 143 order-statistics).
- Model: RF(400,d8,leaf5) + ExtraTrees(600,d10,leaf4) + LGBM(300,lr.05,leaves31,d6), rank-averaged; FF++-train only.
- Freeze doc: `00_protocol/trackE_FREEZE.md`. Pre-reg: `00_protocol/trackE_preregistration.md`.

## ★ Sealed Celeb-DF-v2 test (budget 1, spent once)
| number | value | provenance |
|---|---|---|
| Celeb-DF-v2 test AUC | **0.7133**, 95% CI [0.687, 0.746] | SEALED_final.json · exp_trackE_SEALED_eval.py · commit d35064a · 2026-07-30 |
| n | 2,273 (372 real / 1,901 fake), 27 identities | splits/celebdf_dev_test.json |
| single-model ref (test) | ET 0.7143 · RF 0.7059 · LGBM 0.691 | SEALED_final.json |
| pre-registered prediction | 0.68, [0.65, 0.71] | trackE_preregistration.md · commit 84854b2 |
| audit | 2 unseal log lines = 1 evaluation (crash-recapture) | SEALED_PROVENANCE.md · sealed_eval_log.txt |

## Post-freeze descriptive — celebdf_test (identical data)
| representation | AUC | 95% CI | real rec | fake rec | provenance |
|---|---|---|---|---|---|
| 50-D | 0.6573 | [0.630, 0.685] | 0.183 | 0.952 | POSTFREEZE_compare.json · exp_trackE_postfreeze_compare.py · 122f956 |
| 53-D | 0.6830 | [0.656, 0.718] | 0.274 | 0.895 | POSTFREEZE_compare.json · 122f956 |
| 196-D | 0.7133 | [0.687, 0.746] | 0.183 | 0.961 | POSTFREEZE_compare.json · 122f956 |
| DeLong 196 vs 50 | Δ+0.0561, z=4.79, p=1.7×10⁻⁶ | paired, shared test | POSTFREEZE_compare.json |
| DeLong 196 vs 53 | Δ+0.0303, z=5.98, p=2.2×10⁻⁹ | paired, shared test | POSTFREEZE_compare.json |

## Post-freeze descriptive — FF++ test (in-distribution)
| number | value | provenance |
|---|---|---|
| 196-D pooled | 0.842, [0.802, 0.880] | POSTFREEZE_ffpptest.json / POSTFREEZE_permanip.json · bfc301c / (local) |
| 196-D per-manip (DF/F2F/FS/NT) | 0.907 / 0.796 / 0.831 / 0.833; mean 0.842 | POSTFREEZE_permanip.json · exp_trackE_permanip.py |
| 53-D pooled / mean-of-4 | 0.8358 / 0.8358 | POSTFREEZE_permanip.json |
| 196-D real/fake recall | 0.562 / 0.927 | POSTFREEZE_ffpptest.json |
| n | 685 (137 real / 548 fake) | FF++ official test split |

## Key dev numbers (selection; full set in LOCKED_NUMBERS.md + ledger)
| number | value | provenance |
|---|---|---|
| E1 additive (196-D) | in-dist +0.0092, cross +0.0326 (Holm) | trackE_E1_dev.json · 474d780 |
| RF_d8 classifier | dev CV 0.7018 | trackE_clfsweep_dev.json · 8dfcece |
| rank ensemble | dev CV 0.7125 (+0.0107) | trackE_ens2_dev.json · ee3d6cc |
| WildDeepfake (2nd target) | ~0.55–0.58 all candidates | trackE_joint_dev.json · 84854b2 |
| dev-eval count | 57 | trackD_dev_evals.txt · c1a77bb |

## Datasets / splits
- Celeb-DF-v2: dev 2,421 / test_SEALED 2,273 / dropped 1,427; split `celebdf_dev_test.json` (git_commit 0009956, 2026-07-25).
- FF++ official split `ffpp_official_split.json`; test = 700 videos (140 real / 560 fake), 685 extracted.
