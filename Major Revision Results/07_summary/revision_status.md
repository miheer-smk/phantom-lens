# Master Status (updated after each experiment)

## Original 12 experiments + Track C
| # | Experiment | Answers | Status |
|---|---|---|---|
| 1 | Group ablation (2-view) | R1,R3,R5 | ✅ done (851551a) |
| 2 | SHAP stability (+NT correction) | R5 | ✅ done (c9afcd9) |
| 3 | Compression all 4 manips (c23/c40) | R5.4 | ✅ done (830a780) |
| 4 | Threshold calibration (val-only) | R1,R5.3 | ✅ done (81c6067) |
| 5 | Runtime/memory profiling | R2,R3.6 | ✅ done — re-profiled ≥100 vid (3c2f80c) |
| 6 | Extra dataset (WildDF ✅ / DFDC 🔴) | R3.7,R5.1 | ◑ partial (WildDF 0.5212; DFDC blocked — no valid labels) |
| 7 | Xception baseline | R5.2,R3.4 | ✅ done (1337bc8) |
| 8 | PRNU comparison | R1 | ✅ done (33550c9) |
| 9 | rPPG POS/CHROM | R1 | ✅ done (d9105fd) |
| 10 | Case-level SHAP | R4,R5.6 | ✅ done (ce496db) |
| 11 | Statistical tests | R3.13 | ✅ done (af80bd5) |
| 12 | Feature redundancy | R3.12 | ✅ done (92a4235) |
| Track C | ROI mouth (G1) 53-D | new | ✅ done (de15506) |

## Guide-audit methodology fixes & gaps (2026-07-21→23)
| Item | Task | Status | Commit |
|---|---|---|---|
| M1 | Imputation-leakage → train-only imputer (`src/leakfree.py`) | ✅ fixed; all 35 locked files bit-identical (0.00e+00) | e09ffa5 |
| M2 | Scaler-in-CV | ✅ audited clean (headline train-only; no CV-derived AUC) | — |
| M3 | Test-set model selection | ✅ audited clean; `rigorous_search.py` confirmed exploratory (no disk writes) | — |
| M4 | Missingness-as-signal audit + missingness-only classifier | ✅ done — real-vs-fake AUC 0.50/0.51 (at chance); deterministic (LogReg) | bc409a3 (+PhaseC) |
| G1 | Hard-negative clean Deepfakes TEST (retire leaky 13/957) | ✅ done — 17/133 FN in-distribution | 13b8550 |
| G2 | c40 all 4 manips | ✅ already done in EXP-3 | 830a780 |
| G3 | Runtime ≥100 videos | ✅ done — 113 videos, RTF 3.196 | 3c2f80c |
| G4 | Case-level SHAP (4 cases) | ✅ already done | ce496db |
| G5 | Feature redundancy (VIF/clustering) | ✅ already done | 92a4235 |
| G6 | PRNU (BM3D unavailable, disclosed) | ✅ already done | 33550c9 |
| G7 | rPPG POS/CHROM | ✅ already done | d9105fd |
| G8 | CelebDF 3-way reconciliation; lock 0.632 | ✅ done — `celebdf_reconciliation.md` | 3f94536 |
| G9 | Per-video predictions + stats-from-persisted-probs | ✅ done — 16,635 rows; DeLong reproduces exactly | 3c2f80c |

## Phase C — packaging / reproducibility
| Item | Status |
|---|---|
| Hardcoded absolute paths → env-var/relative | ✅ fixed (exp5_runtime, wilddeepfake_extract, xception_prep, gate_exp1, exp10_signals) |
| Zenodo package: all scripts + src + splits + per-video preds + manifests + requirements/env/seed + README + DATASET_ACCESS | ✅ synced |
| `REPRODUCE.md` — clean regeneration commands | ✅ added |
| Fresh-checkout regeneration of final tables | ✅ verified bit-identical (baseline/M4/G1 0.00e+00; full CPU table set confirmed) |
| Fill sheet for co-author response tables (H1–H9 + 19 placeholders) | ✅ `07_summary/response_fill_sheet.{md,json}` |

## Out of scope for the assistant (author-owned)
- All manuscript framing / claims prose (see `author_decisions.md`).
- DFDC (EXP-6): blocked pending a valid mixed-label DFDC file.
- Domain adaptation (response Table 11): **no reproducing script exists** — author decision run-or-remove (see `author_decisions.md`).
