# Section 2 Reconnaissance Report — PRISM / Phantom Lens Major Revision

**Date:** 2026-07-15
**Repo copy examined:** `/home/iiitn/Downloads/phantom-lens-main` (extracted from `phantom-lens-main.zip`)
**Author of report:** Claude Code (recon phase only — no experiments run)

---

## TL;DR — HARD BLOCKER (read first)

This repo copy is **code-only**. Every input required to compute a single number is
**absent**:

- **No extracted 50-feature CSVs** — the `features/` directory is `.gitignore`d and was
  never in the zip. Searched the entire machine (`/home/iiitn`): no PRISM feature CSV,
  pickle, parquet, or npz exists anywhere.
- **No trained model** — `checkpoints/` and all `*.pkl`/`*.pt` are gitignored and absent.
- **No raw video data** — `data/` is gitignored and absent (FF++, CelebDF-v2, none present).
- **No Python environment** — system `python3` (3.12.3, aarch64) has none of the required
  packages (numpy, pandas, sklearn, lightgbm, shap, mediapipe, cv2, scipy). No project
  venv exists.

**Consequence:** Not one of the 12 experiments can run — not even the "CSV-only, ready
now" Phase-1 items (Exp 1, 2, 3, 4, 5, 10), because they all depend on the 50-feature
CSVs, which are the missing artifact. Recon item #6 (reproduce baseline AUC 0.9939 /
0.6989) is likewise blocked.

Per **Rule 1 (never fabricate)** and **Rule 10 (stop and ask, don't guess)**, work is
halted pending the data. See `07_summary/open_ambiguities.md` for the decision needed.

---

## Recon item 1 — Repo structure

Top-level: `analysis/  archive/  config/  data_prep/  evaluation/  experiments/  results/
src/  training/` plus `README.md`, `requirements*.txt`, `pyproject.toml`, `dockerfile`,
`LICENSE`. 227 files, all source/scripts + small result artifacts (JSON/PNG/PDF/small CSV).

- **Feature-extraction code:** `src/precompute_features_best.py` (the 50-feature PRISM
  extractor; the README's `features/` path is aspirational — actual file is under `src/`).
  Also `src/pillars/pillar1_noise.py`, `pillar2_light.py`, `pillar3_compression.py`.
- **Training code:** `training/train_v3_best.py` (best-model CV+test), `train.py`,
  `train_v3.py`, plus `train_v4.py` (a separate 20-feature attention model — NOT the
  manuscript's 50-feature model).
- **Evaluation code:** `evaluation/cross_dataset_eval.py`, `evaluate_v3_best.py`,
  `validate_ffpp_indistribution.py`.
- **Existing results (artifacts only, no underlying data):** `results/exp1` (multi-manip
  FF++), `results/exp2` (compression), `results/exp3` (SHAP ablation), `results/exp5`
  (false negatives), `results/exp_celebdf` (cross-dataset).

## Recon item 2 — The 50-feature CSV(s)

**NOT PRESENT.** `.gitignore` excludes `features/` ("Extracted features (large CSV files,
100s MB)") and `data/*.pkl`. The zip contains no `features/` or `data/` entries. Machine-
wide search for feature CSV/pkl/parquet/npz found nothing PRISM-related.

The feature schema is nonetheless fully recoverable from surviving artifacts:
- 50 features = **13 spatial (`s_` prefix) + 37 temporal (`t_` prefix)**, confirmed by the
  canonical `FEATURE_NAMES_SPATIAL` / `FEATURE_NAMES_TEMPORAL` lists in
  `src/precompute_features_best.py` (lines ~66–100) and by `results/exp3/feature_ranking.csv`
  (all 50 named with SHAP importances).
- Label convention: `0 = real`, `1 = fake` (from extractor `--label` usage in README).
- The intended on-disk form is a per-video pickle (`data/best_train.pkl`,
  `data/v3_train.pkl`, `data/celebdf_eval.pkl`) per `config/datasets.py` — **all absent.**
- Whether a `split`/`fold` column exists cannot be confirmed (file absent).

## Recon item 3 — Five (actually 20-pillar) physical feature groups — RESOLVED from code

The extractor header (`src/precompute_features_best.py` lines 9–33) defines the **ground-
truth grouping** as 19 active pillars over the 50 features. Full mapping (see
`00_logs/feature_group_mapping.md` for the machine-readable version):

**SPATIAL (13):** P1 Noise Physics (3: s_noise_vmr, s_noise_res_std, s_noise_hf_ratio) ·
P2 PRNU/Camera (2: s_prnu_energy, s_prnu_face_periph) · P4 Shadow/Light (2:
s_shadow_score, s_face_bg_diff) · P6 Compression (3: s_benford_dev, s_block_artifact,
s_dbl_compress) · P8 Motion Blur (1: s_blur_mag) · P9 Optical Flow (2: s_flow_mag,
s_flow_dir_consist).

**TEMPORAL (37):** T1 Temporal Noise (3) · T2 rPPG (4) · T3 Temporal PRNU (2) · T4 Face
SSIM Stability (3) · T5 Codec Temporal Residual (2) · T6 Landmark Trajectory (4) · T7
Rigid Geometry (3) · T8 Face-BG Edge (3) · T9 Skin Texture (2) · T10 Color Transfer (2) ·
T11 Specular (2) · T12 Blink Dynamics (3) · T13 Motion-Blur Coupling (2) · T14 DCT
Temporal (2).

Sums: 3+2+2+3+1+2 = **13 spatial**; 3+4+2+3+2+4+3+3+2+2+2+3+2+2 = **37 temporal**; total **50 ✓**.

> **Ambiguity (flagged, not blocking recon):** The reviewer response's example table uses a
> COARSER 5-group scheme (Noise physics 3 / PRNU-inspired residual 2 / rPPG 4 / Landmark
> geometry 7 / Blink dynamics 3). "Landmark geometry (7)" = T6(4)+T7(3); the others map 1:1
> to pillars above. So the reviewer's grouping is a documented merge of the 20-pillar
> scheme. **Experiment 1 must decide: ablate at 20-pillar granularity, at the reviewers'
> ~5–6 coarse-group granularity, or both.** See open_ambiguities.md #A1.

## Recon item 4 — Train/test split & CV folds

- **Seed:** `config/training.py` defines `SEEDS = [42, 123, 777, 999, 2024]`. **42 is the
  primary seed** → consistent with Rule 2 default (RANDOM_SEED = 42). Confirm against
  `training/train_v3_best.py` internals when data is available.
- **Saved split file:** none found on disk (data/ absent). Must be created once per Rule 3
  **after** the CSVs are supplied — cannot be created now (no rows to split).
- **10-fold CV:** referenced by `results/exp3/ablation_summary.csv` (CV AUC + std reported)
  and README ("10-fold CV AUC ... LGBM = 0.921"). Fold assignment code lives in the
  training scripts; fold indices are not saved to disk in this copy.

## Recon item 5 — The classifier — RESOLVED

**LightGBM** is the primary classifier (README badge + "LightGBM = 0.921" CV + Pillar
tables; requirements list `lightgbm` as "Primary classifier"). LR and RandomForest are also
trained for comparison (`final_results.csv` shows per-transfer best-of {LR, RF, LGBM}). The
headline 50-feature FF++ AUC 0.9939 corresponds to the LightGBM "All 50" row in
`results/exp3/ablation_summary.csv`. **Exact LightGBM hyperparameters** are in
`training/train_v3_best.py` — must be read and pinned before any ablation retrain (not yet
extracted in this recon; deferred until env+data exist, since retraining is impossible now).

## Recon item 6 — Reproduce baseline as sanity check — BLOCKED

Cannot run (no CSVs, no model, no env). However the target numbers are **partially
corroborated by surviving artifacts**, and one discrepancy is already visible:

| Metric | Reviewer-doc ground truth | Surviving artifact | Match? |
|---|---|---|---|
| FF++ c23 All-50 Test AUC | 0.9939 | `exp3/ablation_summary.csv` → 0.9939 | ✅ exact |
| CelebDF-v2 zero-shot AUC | 0.6989 | `exp_celebdf/results.json` → **0.6867** | ⚠️ **differs by 0.012** |
| CelebDF fake recall | 0.8745 | `exp_celebdf/results.json` → **0.9224** | ⚠️ differs |
| CelebDF real recall | 0.4020 | `exp_celebdf/results.json` → **0.3002** (TNR) | ⚠️ differs |
| CelebDF macro-F1 | 0.6252 | (Real f1 0.3313 + Fake f1 0.9095)/2 = **0.6204** | ≈ close |
| CelebDF MCC | 0.2537 | not stored | — |

> **The CelebDF numbers in this repo copy are a DIFFERENT run from the reviewer doc's
> ground-truth CelebDF numbers.** Possible causes: different model version/seed, different
> feature-extraction parameters (max_frames), different test-set composition (this run:
> 806 real + 5323 fake = 6129, matching README), or the reviewer-doc numbers come from a
> later/frozen model not in this copy. This MUST be reconciled before any Phase-1 experiment,
> because Rule 4/Exp 4 threshold work and Exp 6/11 all assume a single frozen model whose
> CelebDF numbers are known. See open_ambiguities.md #A2.

## Recon item 7 — Raw video data present locally — NONE

No `data/` directory, no FF++, no CelebDF-v2, no WildDeepfake/DFDC/etc. anywhere on the
machine. **Experiments 6, 7, 8, 9 (Phase 2) are hard-blocked on video access**, AND so is
any re-extraction fallback for Phase-1 CSVs.

## Recon item 8 — c40 feature availability — NOT PRESENT / UNKNOWN

No feature CSVs exist, so c40 features exist for zero manipulation types in this copy.
`results/exp2/summary.csv` (compression experiment) + README both state only **c23** was
tested ("Compression c0/c40 unavailable — only c23 tested"). So Experiment 3's c40 arm
requires **new extraction from c40 videos for all four manipulations** → hard-blocked on
video access. Confirm whether Miheer has c40 CSVs elsewhere (open_ambiguities.md #A3).

---

## Environment (partial — see 00_logs/environment.txt)

- Python 3.12.3, Linux 6.17.0 aarch64 (ARM64), glibc 2.39.
- CPU: 20 cores. RAM: 121 GiB. GPU: **NVIDIA GB10** (Grace Blackwell; present → Exp 7
  Xception is hardware-feasible *if* videos are supplied).
- **No scientific Python packages installed in system python; no project venv.** A pinned
  env must be built from `requirements.txt` (note: aarch64 wheel availability for mediapipe
  / lightgbm / opencv must be verified when env setup is authorized).

## What I could NOT determine without the data/model (deferred, not skipped)

- Exact LightGBM hyperparameters (in `train_v3_best.py`; will pin when retraining is possible).
- Whether a persisted split/fold column exists (file absent).
- Whether the reviewer-doc CelebDF numbers (0.6989 etc.) are reproducible (needs frozen model).
