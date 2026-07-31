# Track D + E — Complete Inventory Audit

Branch `best-revision`. Every measurement in the Track D/E (196-D) programme, with provenance. Generated 2026-07-31.
Seed 42 throughout. Single source of truth for numbers: `07_summary/LOCKED_NUMBERS.md`. Dev-eval ledger:
`07_summary/trackD_dev_evals.txt` (57 entries). "cross" = celebdf_dev identity-grouped 5-fold CV unless noted.

| # | experiment | what it tested | result (headline) | verdict | result file | script | commit | in LOCKED_NUMBERS? |
|---|---|---|---|---|---|---|---|---|
| **Representation / aggregation** |
| 1 | E1 order-stats (replace) | 143 order-stats replace means | in-dist +0.0103, cross +0.0259 (Holm) | ✅ qualifies | trackE_E1_dev.json | exp_trackE_E1_eval.py | 474d780 | Y |
| 2 | E1 order-stats (additive) | 53-D + 143 order-stats = 196-D | in-dist +0.0092, cross **+0.0326** (Holm) | ✅ **the gain** | trackE_E1_dev.json | exp_trackE_E1_eval.py | 474d780 | Y |
| 3 | R0 consistent 60-frame | real-vs-manips baseline, full_features | cross 0.6967 | baseline | trackE_E3_dev.json | exp_trackE_E3_eval.py | 4439c79 | Y |
| 4 | Denser 100-frame (subset) | 60 vs 100 frames, 300-vid subset | +0.0517 (subset only) | ⚠ subset | trackE_denser_dev.json | exp_trackE_denser_compare.py | ee3d6cc | Y |
| 5 | Denser 100-frame (full) | full 6547-vid re-extraction | 0.6988 vs 0.7018 (**−0.003, no replication**) | ✗ | trackE_100frame_dev.json | exp_trackE_frameval.py | **UNTRACKED→§6** | Y |
| **Feature families** |
| 6 | Group H | gradient structure tensor | cross −0.0137 | ✗ | trackD_H_dev.json | (Track D) | e353e92 | Y |
| 7 | Group I | ocular physics | **NOT RUN** (not in ledger/results) | — | — | — | — | — (gap noted) |
| 8 | Group J-a | additive dimensionless ratios | cross +0.0022 | ✗ | trackD_J_dev.json | (Track D) | a319a4e | Y |
| 9 | Group J-b | quantile alignment | cross +0.0104 (noise under multiplicity) | ✗ | trackD_J_dev.json | (Track D) | a319a4e | Y |
| 10 | Group M | cardiac coherence | cross −0.0141 (p_holm .050) | ✗ | trackD_MQRT_dev.json | (Track D) | f1458f8 | Y |
| 11 | Group Q | muscle co-activation | cross +0.0001; in-dist +0.0012 | ✗ | trackD_MQRT_dev.json | (Track D) | f1458f8 | Y |
| 12 | Group R | blink kinematics | cross −0.0028 | ✗ | trackD_MQRT_dev.json | (Track D) | f1458f8 | Y |
| 13 | Group T | rigid 3-D | cross −0.0104 (p_holm .153) | ✗ | trackD_MQRT_dev.json | (Track D) | f1458f8 | Y |
| 14 | M+Q+R+T combined | all four added | in-dist +0.0021, cross +0.0070 (p_holm 1.0) | ✗ | trackD_MQRT_dev.json | (Track D) | f1458f8 | Y |
| 15 | E4 multi-scale LoG | 18 frequency feats stacked | cross +0.0055, in-dist −0.005 | ✗ below bar | trackE_E4_dev.json | exp_trackE_E4_eval.py | 715813b | Y |
| 16 | E5 temporal-difference | 78 diff/rel-flicker feats | cross −0.0017 | ✗ | trackE_tempdiff_dev.json | exp_trackE_tempdiff.py | ee3d6cc | Y |
| **Training-distribution / data** |
| 17 | E3 SBV pre-flight | Cohen's d real-vs-SBV boundary | boundary max\|d\|=3.45 (PASS gate) | gate | trackE_SBV_preflight.json | exp_trackE_SBV (extract) | ef33dd7 | Y |
| 18 | E3 SBV R0/R1/R2 | self-blended-video training regimes | R0 0.6967 / R1 0.4638 collapse / R2 0.6719 | ✗ | trackE_E3_dev.json | exp_trackE_E3_eval.py | 4439c79 | Y |
| 19 | X4 diverse reals | DFD originals → real class | cross −0.020 (realRec +0.052) | ✗ | trackE_X4_dev.json | exp_trackE_X4_eval.py | ee3d6cc | Y |
| 20 | X4 diverse fakes | DFD fakes → fake class | cross −0.0098 (realRec −0.042) | ✗ | trackE_X4fakes_dev.json | exp_trackE_X4fakes_eval.py | 84854b2 | Y |
| **Domain adaptation** |
| 21 | CORAL | correlation alignment | cross −0.0396 | ✗ | trackD_DA_dev.json | (Track D) | 9d70f5f | Y |
| 22 | Subspace align (d10/20/30) | subspace alignment | cross −0.087/−0.090/−0.055 | ✗ | trackD_DA_dev.json | (Track D) | 9d70f5f | Y |
| 23 | Per-domain standardise | per-domain z-score | cross −0.0024 | ✗ | trackD_DA_dev.json | (Track D) | 9d70f5f | Y |
| 24 | Per-domain quantile | quantile alignment | cross −0.0151 | ✗ | trackD_DA_dev.json | (Track D) | 9d70f5f | Y |
| 25 | Self-training (k=10/20/30%) | pseudo-label transductive | −0.031/−0.048/−0.057 (all hurt) | ✗ | trackE_selftrain_dev.json | exp_trackE_selftrain.py | 8670192 | Y |
| **Classifier / ensembling** |
| 26 | Classifier sweep | LGBM d2-6, RF, LogReg, SVM | RF_d8 0.7018 (best); LR .595 SVM .549 | ✅ RF wins | trackE_clfsweep_dev.json | exp_trackE_clfsweep.py | 8dfcece | Y |
| 27 | Strong-member ensemble | RF+ET+LGBM prob/rank | rank **0.7125** (+0.0107); ET 0.7036 | ✅ | trackE_ens2_dev.json | exp_trackE_ens2.py | ee3d6cc | Y |
| 28 | Random-subspace bagging | M15/M30 50% feats | RF_M30 0.7031 (+0.0013 vs RF) | ✗ marginal | trackE_subspace_dev.json | exp_trackE_subspace.py | 84854b2 | Y |
| 29 | Per-manip ensemble | avg real-vs-{DF,F2F,FS,NT} | 0.6975 (+0.0008, calibration only) | ✗ | trackE_ensemble_dev.json | exp_trackE_ensemble.py | b0b9b12 | Y |
| 30 | E2 windowed MIL | window aggregators mean/max/topk/p90/frac | best 0.682 (−0.0149 vs R0) | ✗ | trackE_E2_dev.json | exp_trackE_E2.py | b0b9b12 | Y |
| **Inference-time** |
| 31 | TTA (N=2,3) | augment celebdf_dev, avg probs | RF 0.7018→0.6985; ens 0.7125→0.7088 | ✗ | trackE_TTA_dev.json | exp_trackE_TTA_eval.py | c1a77bb | Y |
| **Selection / validation infrastructure** |
| 32 | X1 KS-stability (k) | domain-stable feature subsets | top-k 0.518..0.663, all < keep-all | ✗ | trackE_X1X2_dev.json | exp_trackE_X1X2.py | b9e0a11 | Y |
| 33 | X2 drop-rPPG | ablate rPPG features | cross +0.0055 (below bar) | ✗ | trackE_X1X2_dev.json | exp_trackE_X1X2.py | b9e0a11 | Y |
| 34 | WildDeepfake 196-D | 2nd validation target, all candidates | WDF ~0.55–0.58 all; dev≠WDF ranking | curse check | trackE_joint_dev.json | exp_trackE_joint.py | 84854b2 | Y |
| 35 | Celeb-DF dev/test split | identity-disjoint construction + assert | dev 2421 / test 2273 / dropped 1427 | infra | splits/celebdf_dev_test.json | build_celebdf_devtest.py | (0009956) | Y |
| **Endgame** |
| 36 | Pre-registration | predicted sealed test | 0.68, [0.65, 0.71] | pre-reg | trackE_preregistration.md | — | 84854b2 | Y |
| 37 | Freeze document | frozen config + rule | 196-D + RF+ET+LGBM rank ens | freeze | trackE_FREEZE.md | — | 84854b2 | Y |
| 38 | **Sealed eval** | Celeb-DF-v2 test (budget 1) | **0.7133 [0.687, 0.746]** | ★ SEALED | SEALED_final.json | exp_trackE_SEALED_eval.py | d35064a | Y |
| 39 | Sealed provenance | crash-and-recapture disclosure | 2 log lines, 1 evaluation | disclosure | SEALED_PROVENANCE.md | — | d35064a | Y |
| 40 | Post-freeze 50/53-D | baselines on same test + DeLong | 50-D 0.6573, 53-D 0.6830; p=1.7e-6/2.2e-9 | descriptive | POSTFREEZE_compare.json | exp_trackE_postfreeze_compare.py | 122f956 | Y |
| 41 | FF++ test (frozen) | in-dist pooled | 0.842 [0.810, 0.872] | descriptive | POSTFREEZE_ffpptest.json | exp_trackE_ffpptest_eval.py | bfc301c | Y |
| 42 | FF++ per-manip (§2) | per-manip vs pooled, both models | 196-D 0.842/0.842; 53-D 0.836/0.836 | descriptive | POSTFREEZE_permanip.json | exp_trackE_permanip.py | **UNTRACKED→§6** | Y |
| 43 | Dev-eval ledger | multiplicity count | 57 entries | disclosure | trackD_dev_evals.txt | — | c1a77bb | Y |

## Gaps found (to resolve in Step 6 or flag)
1. **`trackE_100frame_dev.json` UNTRACKED** (local-only) — the full 100-frame non-replication result. Must be committed (§6).
2. **`POSTFREEZE_permanip.json` UNTRACKED** (local-only, just produced) — Step 2 per-manip result. Must be committed (§6).
3. **`sealed_eval_log.txt` UNTRACKED** — the sealed audit trail (2 unseal entries). Must be committed (§6).
4. **Group I (ocular physics): NOT RUN.** No ledger entry, no result file. Recorded here as not executed (not a missing artifact — it was never part of the run). Flag for authors if it was intended.
5. Older Track D result JSONs (H/J/MQRT/DA) are committed with provenance but their producing scripts live in the earlier Track D tooling (not in `00_logs/exp_trackE_*`); numbers are in LOCKED_NUMBERS and the ledger. No missing numbers.

All other results located, committed, and in LOCKED_NUMBERS. No reported number lacks a result file + commit except the three untracked files above (resolved in §6).
