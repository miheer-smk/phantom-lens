# Track E — Pre-registration (written BEFORE measuring). Additive/alternative to locked 53-D; never a rewrite.

Motivation (from author literature review): Track D showed the Celeb-DF bottleneck is NOT feature poverty
(5+ additive families + 5 DA methods all failed cross-dataset). Cross-dataset SOTA wins come from the
TRAINING DISTRIBUTION (SBI/FSBI self-blended images ~0.93–0.95), not better features. Track E tests whether
that mechanism reproduces in a non-deep, handcrafted-physics pipeline. Same hard rules: identity-disjoint,
sealed sets untouched (budget 1, currently 0), Holm across ALL Track D+E families, cross threshold +0.03,
in-dist +0.005, train-only imputer/scaler, seed 42. Locked 50-D/53-D and all locked numbers untouched.

## E1 — Richer temporal aggregation (order statistics over per-frame spatial values)
**Hypothesis.** Reducing each spatial feature to a single frame-mean destroys distributional information;
manipulation artifacts are often INTERMITTENT (a few bad frames). Order statistics (std/min/max/percentiles/
skew/kurtosis) should recover that signal at ~zero extraction cost beyond one per-frame pass.
**Method.** Persist per-frame values of the 13 spatial features (`extract_spatial_features_single_frame`),
then replace each feature's single mean with **11 statistics** [mean, std, min, max, p10, p25, p75, p90, IQR,
skew, kurtosis] → **13×11 = 143 spatial-stat features**. E1 model = 143 spatial-stats + 37 temporal (existing)
+ 3 G1 (= 183-D), vs 53-D baseline. Also test additive 53-D + 143 stats. Denser sampling (60 frames) to expose
intermittent traces. Per-frame values are persisted (long CSV) so **E2 (windows) reuses them** — no re-extraction.
**Pre-registered prediction.** In-distribution **positive, concentrated in Face2Face / NeuralTextures**
(intermittent re-render artifacts); Deepfakes/FaceSwap smaller (near-ceiling / whole-face swaps). Cross-dataset
**uncertain–small** (order stats of absolute magnitudes are still scale-sensitive; the domain gap likely persists).
Meets bar only if in-dist ΔAUC ≥ +0.005 (Holm) on F2F/NT, or cross ≥ +0.03.

## E2 — Frame/window-level MIL scoring (reuses E1 per-frame values)
Windows (30 frames, stride 15) → per-window P(fake) → aggregate {mean, max, top-k mean, p90, frac>thr},
aggregator chosen on val only. Windows grouped by identity (never span train/val). **Prediction:** in-dist
positive (F2F/NT intermittent); cross uncertain.

## E3 — Self-Blended Videos (SBV) — main cross-dataset lever
Train real vs self-blended (STG source/target from same frame + soft landmark-hull mask + TEMPORAL artifact
injection: boundary jitter, per-frame transform flicker, single-frame outliers, landmark perturbation;
`--temporal_jitter` 2–3 levels). Regimes: R0 real-vs-FFpp (current), R1 real-vs-SBV only, R2 hybrid.
**Prediction:** R1 sacrifices some in-dist AUC, substantially improves cross-dataset (the SBI mechanism);
R2 balances. Honest caveat: SBI's 0.93–0.95 are frame-level deep-CNN; with ~50 handcrafted descriptors the
ceiling is lower — the scientific question is whether the training-distribution effect reproduces at all in a
non-deep pipeline (novel: SBI not previously tested with handcrafted forensic features). If it fails, that is
itself publishable (the blending signal SBI exploits is not captured by physics descriptors).

## E4 — Multi-scale Laplacian-of-Gaussian frequency descriptors
LoG pyramid (4–5 scales) over face ROI: per-scale energy, entropy, kurtosis, face/bg energy ratio (dimensionless).
~20 features. **Prediction:** in-dist small-positive; cross uncertain (ratio variant the only transfer hope).

Order: E1 → E2 → E3 → E4. One at a time: implement → measure (val + celebdf_dev) → report → wait for go.
Multiplicity: dev-eval ledger continues from Track D (17 so far); Holm across the combined Track D+E set.
