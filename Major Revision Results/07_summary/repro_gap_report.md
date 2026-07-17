# Reproduction-Gap Report (Track B)

**Date:** 2026-07-17 · **Question:** are the regenerated features faithful to the originals,
and which numbers are authoritative?

## 1. Search for original artifacts — NOT RECOVERABLE

Exhaustive system-wide search (`/`, `/media`, `/mnt`, `/home`, `/scratch`) for:
- original feature CSVs (`features/ffpp_face2face.csv`, `ffpp_neuraltextures.csv`,
  `ffpp_faceswap.csv`, `ffpp_real*.csv`, `ffpp_celebdf*.csv`)
- trained models (`model_LightGBM.pkl`, `scaler.pkl`, `best_train.pkl`, any `*lgbm*.pkl`)

**Result: none found.** The only `.pkl` files on the system are 7 *older-iteration feature
dicts* (`precomputed_features*.pkl`, 30/24/20-feature, frame-level) in `~/Downloads` — not
models, not the 50-feature set. No trained model, no scaler, no original 50-feature CSV
survives. The original data root `/home/iiitn/Miheer_Project` was deleted (`rm -rf`, confirmed
in shell history). `/media/iiitn` and `/mnt` are empty (no HDD mounted at report time).

## 2. Faithfulness of regeneration — VERIFIED where ground truth exists

The single surviving 50-feature ground truth is the committed
`results/exp5/false_negatives_with_features.csv` (13 Deepfakes c23 videos). Regenerated
features for those exact 13 videos vs the committed originals:
- **49 / 50 features machine-identical** (abs diff 1e-9…1e-14 = floating-point noise).
- 1 feature (`t_noise_spectral_entropy`, FFT-spectral) differs by ~1% (opencv 4.11 vs the
  original opencv; benign — a one-line dtype fix keeps the mask numerically identical).
- Environment match confirmed: mediapipe **0.10.18** (the repo Dockerfile pins the extractor to
  0.10.18, so this is the exact original landmark model); numpy 1.26.4, scipy 1.15.3 exact.

**Conclusion for Deepfakes (and, by shared extractor, FaceSwap): regeneration is faithful.**

## 3. Face2Face / NeuralTextures — NOT independently verifiable

No original F2F/NT feature CSV survives, so their regenerated features cannot be compared
row-for-row. The extractor code is identical and Deepfakes reproduced exactly, so there is no
evidence of an F2F/NT-specific regeneration bug — but this cannot be *proven* without the
originals. Per the prime directive, we do **not** assume the higher published numbers.

## 4. Authoritative numbers (decision)

- **Deepfakes / FaceSwap in-distribution:** regenerated values are authoritative (verified faithful).
- **Face2Face / NeuralTextures:** regenerated values are authoritative *by necessity*; flagged
  as an open reproducibility question (originals unrecoverable), not a resolved discrepancy.
- **Celeb-DF v2:** the published 0.6989 is **not reproducible** from code + regenerated
  features (best honest zero-shot ≈ 0.62–0.65). The published number relied on a pipeline whose
  artifacts do not survive; the honest reproduced value is authoritative.
- **Retired leaky numbers** (0.9991 / 0.9999 / 0.9939): permanently retired; not reported.

## 5. Reproducibility note for the manuscript

The original trained model and feature tables were lost; all numbers in this revision are
regenerated end-to-end from raw video by committed, deterministic scripts (seed 42), with
input hashes recorded in `LOCKED_NUMBERS.md`. Feature extraction was validated machine-identical
on the one surviving ground-truth sample (Deepfakes).
