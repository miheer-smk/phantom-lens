# Track D — Pre-registration (written BEFORE any extraction/measurement)

Date: 2026-07-25 · Protocol: hypothesis-driven additive feature engineering under the anti-overfitting
rules (sealed sets, dev-only iteration, one final evaluation). Baseline model = locked **53-D**
(50-D + G1 ROI mouth). New features are **additive** (53+X); locked 50-D/53-D vectors are never modified.
If no family helps on dev, the frozen model remains 53-D (null result is a valid outcome).

## Sealed sets (evaluated exactly ONCE, Phase 4)
- **FF++ test** identities (official split `partition=="test"`). Dev uses FF++ **train/val** only.
- **Celeb-DF `test` half** — `splits/celebdf_dev_test.json` (identity-disjoint; built + asserted by
  `build_celebdf_devtest.py`). **dev = 2421 (426 real / 1995 fake)**, **test_SEALED = 2273 (372 / 1901)**,
  1427 spanning fakes dropped; 0 identities shared. Gated by `src/sealed.py::unseal()` (raises by default;
  every unseal logged to `sealed_eval_log.txt`; count must be 1 at the end).
- WildDeepfake = **secondary dev signal** only (also reported at the sealed eval for completeness).

## Pre-committed decision rule (fixed before seeing ANY sealed result)
1. Each group is measured on dev only: **FF++ val ΔAUC** (in-distribution) and **celebdf_dev ΔAUC**
   (cross-dataset), always as 53-D vs 53-D+group, plus per-feature Cohen's d.
2. A group is **INCLUDED** in the frozen set iff, on its **target axis**, dev ΔAUC ≥ **+0.005**, AND it does
   not degrade the other axis by more than **−0.005**. Target axis: H, I, K → in-distribution; J → cross-dataset.
3. Frozen set = 53-D + every included group (evaluated together on dev before freezing).
4. **STOP condition:** if, after all pre-registered families, the best dev gain on either axis is
   **< +0.01**, stop, freeze = 53-D, and report the null result. No further variants.
5. Sealed evaluation (Phase 4) runs **once** on the frozen set: FF++ test (per-manip + mean),
   celebdf_test, WildDeepfake, + DeLong vs the 53-D locked model. Report whatever the numbers are.

## Candidate feature families (predicted effects)

### Group H — Gradient Structure Tensor  (priority; target axis = in-distribution)
Features (per-frame, face ROI + background for a ratio; + temporal std/lag-1 autocorr of anisotropy & orientation entropy):
`h_anisotropy=(λ1−λ2)/(λ1+λ2+ε)`, `h_coherence=λ2/(λ1+ε)`, `h_tensor_trace`, `h_eig_ratio_log=log(λ1/(λ2+ε))`,
`h_orientation_entropy` (Shannon entropy of gradient-orientation histogram), `h_face_bg_aniso_ratio` (ratio → more domain-invariant),
`h_anisotropy_tstd`, `h_anisotropy_lag1`, `h_orient_entropy_tstd`, `h_orient_entropy_lag1`.
- **Physical rationale:** real sensor output has isotropic high-frequency photon/read noise → characteristic
  luminance-gradient anisotropy/coherence; GAN/rendered content is smoother, more locally-coherent/oriented.
- **Pre-registered prediction:** **in-distribution ΔAUC positive** (esp. NeuralTextures/FaceSwap, texture-driven);
  cross-dataset small/uncertain except the `face_bg_aniso_ratio` variant (ratio → some transfer).

### Group I — Ocular Physics  (target axis = in-distribution)
Features: `i_corneal_highlight_iou` (L/R specular-highlight shape consistency), `i_corneal_position_consistency`,
`i_pupil_circularity=4π·area/perimeter²`, `i_iris_boundary_sharpness`, + temporal std of each.
- **Physical rationale:** corneal specular highlights reflect the same light field in two spheres a fixed distance
  apart → must be geometrically consistent between eyes; generators have no explicit reflection model; real pupils near-circular.
- **Pre-registered prediction:** **in-distribution ΔAUC positive** on face-reenactment manips (Deepfakes/Face2Face,
  eyes preserved/animated); may be **weak/noisy** where eye crops are low-res; cross-dataset uncertain.

### Group J — Domain-Invariant Reformulations  (target axis = CROSS-DATASET — most likely to move Celeb-DF)
Features: per-video z-normalised variants of magnitude features (normalise by that video's own background/global
statistic → cancels camera/codec scale); face-vs-background ratio variants where currently absent; rank/quantile-transformed
variants (shape not scale); within-video temporal contrast (mouth-ROI feature vs same feature in a non-manipulated region, e.g. forehead/cheek).
- **Physical rationale:** pillar-only analysis showed all pillars collapse to ~0.50–0.56 on Celeb-DF; absolute
  feature magnitudes shift across datasets (cameras/encoders/resolutions). Ratios / within-video normalised /
  within-video control quantities are inherently more transferable than absolutes.
- **Pre-registered prediction:** **celebdf_dev ΔAUC positive** (primary hoped-for effect); in-distribution
  roughly neutral (may slightly reduce in-dist if it removes discriminative absolute scale).

### Group K — Higher-Order Noise Statistics  (target axis = in-distribution)
Features: skewness & kurtosis of the noise residual (face region); cross-band wavelet coefficient correlations
(steganalysis-style); residual histogram entropy.
- **Physical rationale:** current noise pillars use variance-based descriptors only; shot noise has a specific
  higher-moment signature not captured by second moments.
- **Pre-registered prediction:** **in-distribution ΔAUC small-positive**; cross-dataset uncertain (higher moments
  are also scale/codec sensitive).

## Considered and REJECTED (not implemented) — documented for transparency
- **Vanishing-point / scene-perspective geometry:** designed for full-scene synthetic images with architectural
  perspective. Our inputs are 224×224 **face crops** from real video — no usable scene geometry, and in a face-swap
  the **background is genuinely real**, so scene-perspective cues are absent/misleading. Facial 3-D consistency is
  already covered by pillar **T7 (rigid geometry)**. → rejected a priori.

## Multiple-comparisons transparency
Families pre-registered: **4** (H, I, J, K). Any feature tried but not listed above will be marked **post-hoc**
in `trackD_report.md`. Number of sealed-set evaluations budget: **1**.
