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

## FREEZE CRITERIA (locked 2026-07-26, BEFORE X1/X2/E3/E4/Y1 measured — no best-of-N drift)
The frozen model is built by a FIXED inclusion rule, applied ONCE, not by picking the max of many runs:
1. **Base (already qualified): 53-D + E1 order-statistics** (E1_additive) — cleared both bars on dev
   (in-dist +0.0092 Holm-sig, cross +0.0326 Holm-sig). This is the confirmed base representation.
2. **A further component** (E3 SBV regime, E4 LoG-frequency, X1 KS-stable subset, X2 rPPG-drop, Y1 regional)
   is INCLUDED iff, added to the current-best dev model, it improves its **target axis** by ≥ threshold
   (in-dist +0.005 / **cross +0.03**) with **Holm significance across the FULL Track D+E dev ledger**, AND
   does not degrade the other axis > −0.005. Components are evaluated additively on top of the base.
3. **Frozen set = 53-D + E1 + {every component that independently qualifies by rule 2}.** No cherry-picking:
   include-if-qualifies, fixed thresholds, Holm over all ~20+ dev evals. If nothing else qualifies, frozen = 53-D + E1.
4. For E3 (a training-distribution change, not a feature): the winning **regime** (R0/R1/R2 or SBV+hybrid) is chosen
   on **celebdf_dev cross-AUC only**, decided before the sealed run; whichever regime maximises celebdf_dev is frozen.
5. **Sealed evaluation: exactly 1**, on the frozen set, scoring locked 50-D + 53-D + frozen-set together on
   celebdf_test AND FF++ test (identical data; DeLong). celebdf_dev/FFval never used after freeze.

## Contribution to log (author-decision item)
**E1 finding refutes the "cross-dataset is dead" read:** a distributional (order-statistic) representation of the
per-frame physics features transfers cross-dataset materially better than mean aggregation (celebdf_dev 0.627→0.660,
+0.033 Holm-sig) — the first cross-dataset gain in the program. Frame it as a contribution in its own right
(representation, not new features, closes part of the domain gap). [author framing — flagged, not drafted]

## Y1 (QUEUED, not yet run) — Regional temporal decomposition
Compute the existing 37 temporal features PER facial region (mouth, L-eye, R-eye, nose, cheeks, forehead, boundary
band) instead of whole-face-averaged → ~37×7 regional features. Same mechanism that gave G1 +0.118 (region-localised
intermittency). **Prediction:** best remaining IN-DISTRIBUTION play (F2F/NT); cross uncertain. Needs a video pass.

## Plan refinements (locked 2026-07-26, before E3 lands)
1. **Model-selection metric → identity-grouped 5-fold CV within `celebdf_dev`** (not single-split). After ~30 dev
   evals the single-split max is optimistically biased; the CV-averaged AUC tracks the sealed test much more
   closely and shrinks the dev→test gap. All regime/component selection from here uses **`celebdf_dev_cv_mean`**.
   (E3 eval updated accordingly; reports CV mean±std + single-split for reference.)
2. **Pre-registered predicted test number (REQUIRED before unsealing).** Before the single sealed evaluation,
   write a point estimate + interval for `celebdf_test` AUC into the freeze document, then report predicted vs
   actual. Demonstrates the sealed eval was not gamed. Sealed test = **27 identities / ~2,273 videos → bootstrap
   CI ≈ ±0.02–0.03**; expect actual test ≈ dev-CV − (0.02–0.04) optimism gap.
3. **Report REAL recall and FAKE recall separately** on every cross-dataset eval — the core failure is
   Celeb-DF **real recall 0.40 vs fake recall 0.87**; every lever so far improved fake discrimination, none widened
   the real class. (E3 eval now reports both.)

## X4 (QUEUED after E3) — Diverse real augmentation (targets the real-recall failure)
**Hypothesis.** The real class is under-represented/over-fit to FF++ real distribution → Celeb-DF reals flagged fake
(real recall 0.40). Add real videos from **unrelated corpora** to the real training class to widen it:
WildDeepfake reals (already extracted; need `full_features` 196-D), + **VoxCeleb2 / CelebV-HQ reals if obtainable**.
**Celeb-DF reals stay SEALED** (never in training). Compose with SBV: also generate self-blends FROM the diverse reals
(so the fake class covers diverse-real blending too). **Prediction:** widens real recall on celebdf_dev (0.40→higher),
raising cross-AUC via the real class; fake recall roughly maintained. Report real/fake recall separately.
Prep: extract diverse-real `full_features` (196-D) + obtain VoxCeleb2/CelebV-HQ; run after E3 to avoid CPU contention.

## X1/X2 (run now — cheap, no extraction, on the E1-expanded 196-D set)
- **X1 KS-stability selection:** rank features by KS distance between FF++-train and UNLABELED celebdf_dev marginals;
  train on top-k most domain-stable subsets k∈{20,30,50,80,120}. **Prediction:** cross-AUC peaks at an intermediate k
  (dropping unstable features helps transfer); in-dist may dip at small k.
- **X2 rPPG-drop ablation:** drop the rPPG temporal features entirely. **Prediction:** small cross-dataset GAIN
  (leave-one-pillar showed rPPG ≈ −0.014 on Celeb-DF), in-dist ~neutral.
