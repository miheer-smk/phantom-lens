# LOCKED HONEST NUMBERS — PRISM (post rigorous search, 2026-07-17)

Method: 7 model families (LogReg, RF, ExtraTrees, HistGB, LightGBM×3, XGBoost) ×
3 transforms (Standard/Quantile/Power) × ensembling. All model/hyperparam/transform
selection by TRAINING-CV ONLY (never CelebDF). No leakage, no cherry-picking, seed=42.
Feature extraction validated machine-identical to originals on 13 Deepfakes ground-truth videos.

## IN-DISTRIBUTION (FF++ c23, clean 5-fold video-level CV — NO leakage)
| Manipulation   | Best model | AUC   |
|----------------|------------|-------|
| Deepfakes      | XGBoost    | 0.981 |
| FaceSwap       | XGBoost    | 0.961 |
| Face2Face      | XGBoost    | 0.790 |
| NeuralTextures | LogReg     | 0.762 |
| Combined multi | LogReg     | 0.778 |

## CROSS-DATASET (FF++ -> CelebDF v2, zero-shot, model selected by FF++ CV)
Best legitimate: PowerTransform + LogisticRegression -> CelebDF AUC = 0.6505
(baseline was 0.617; honest gain +0.034 from transform+model selection)

## CEILINGS CONFIRMED (all 7 models agree)
- F2F/NT ~0.76-0.79: feature-SEPARABILITY ceiling, not model choice. Needs new features.
- Cross-dataset ~0.65: domain-shift ceiling. Foreign real data could help but = domain shift.
- 0.99+/0.70+ across the board is NOT achievable on current features by any legit optimization.

## vs ORIGINAL PAPER CLAIMS
- Paper "0.9939 in-distribution" = leaky multi-manip protocol. Honest combined = 0.778;
  honest best per-manip = 0.981 (Deepfakes). Report per-manip, not the leaky 0.9939.
- Paper "0.9991/0.9999" (NT/FS) = train=test leakage. Honest = 0.76/0.96.
- Paper "0.6989 CelebDF" = not reproducible; honest best = 0.65 (zero-shot, clean).

## STATUS: these are the defensible numbers to build the revision on.
Extraction of c40 continues in background (for reviewer Experiment 3 / compression).
