# Experiment-Readiness Summary (post-recon, 2026-07-15)

**Overall status: ALL 12 EXPERIMENTS BLOCKED.** The repo copy is code-only — the extracted
50-feature CSVs, the trained LightGBM model, and all raw video data are absent (gitignored,
not in the zip, not anywhere on the machine). No scientific-Python env is installed either.

Legend: 🔴 blocked · 🟡 blocked-but-unblockable-with-one-input · CSV = needs only the
50-feature CSVs · VID = needs raw videos · GPU = needs GPU training · ENV = needs package env.

| # | Experiment | Phase | Doc says | Reality now | Unblock requires |
|---|---|---|---|---|---|
| 1 | Group-wise ablation | 1 | CSV-only | 🔴 blocked | 50-feature CSVs + trained-model config + ENV |
| 2 | SHAP ranking stability | 1 | CSV-only | 🔴 blocked | 50-feature CSVs + per-manip labels + ENV |
| 3 | Compression robustness (all manip) | 1 | CSV + c40 | 🔴 blocked | c23 **and c40** CSVs (c40 needs VID re-extract) + ENV |
| 4 | Threshold calibration | 1 | CSV-only | 🔴 blocked | FF++ val CSV + frozen model + CelebDF CSV + ENV |
| 5 | Runtime / memory profiling | 1 | needs videos | 🔴 blocked | VID (≥100 videos) + extractor deps (mediapipe…) + ENV |
| 10 | Case-level SHAP | 1 | CSV + some signals | 🔴 blocked | CSVs + model; signal plots also need VID + ENV |
| 6 | Additional dataset (WildDeepfake…) | 2 | needs videos | 🔴 blocked | VID (new dataset, licensed) + ENV + Miheer's dataset choice |
| 7 | Xception baseline | 2 | needs videos+GPU | 🟡 GPU present | VID (FF++ frames) + ENV (torch) — GPU OK (GB10) |
| 8 | PRNU residual comparison | 2 | needs videos | 🔴 blocked | VID (residual recompute) + ENV (+ optional BM3D lib) |
| 9 | rPPG POS/CHROM robustness | 2 | needs videos | 🔴 blocked | VID + ENV |
| 11 | Statistical significance tests | 3 | depends on 1–7 | 🔴 blocked | stored prediction scores from Exp 1–7 (none exist yet) |
| 12 | Feature redundancy | 3 | CSV-only | 🔴 blocked | 50-feature CSVs + ENV |

## Resolved during recon (no longer ambiguous)
- **Classifier = LightGBM** (primary; LR/RF for comparison). [recon #5]
- **50-feature grouping** = fully mapped to 20 pillars from extractor code. [recon #3, `00_logs/feature_group_mapping.md`]
- **Primary seed = 42** (matches Rule 2 default; `config/training.py`). [recon #4]
- **FF++ All-50 AUC 0.9939** corroborated exactly by `results/exp3/ablation_summary.csv`. [recon #6]

## Open, needs Miheer (see 07_summary/open_ambiguities.md)
- **A1** — Experiment 1 group granularity (20-pillar vs reviewers' coarse ~5-group vs both).
- **A2** — CelebDF number mismatch: repo shows AUC 0.6867 / fake-recall 0.9224, reviewer doc
  says 0.6989 / 0.8745. Which is the frozen ground-truth model?
- **A3** — Do c40 feature CSVs exist elsewhere, or is Exp 3's c40 arm a re-extraction (VID)?
- **A4** — Where are the feature CSVs / trained model / videos? (the master blocker)
- **A5** — Which Phase-2 dataset for Exp 6, and is its license cleared?
- **A6** — Authorize building the Python env from requirements.txt (aarch64 wheel risk)?
