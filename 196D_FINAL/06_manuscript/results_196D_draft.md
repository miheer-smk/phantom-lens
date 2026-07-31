# Results — 196-D representation and cross-dataset evaluation (factual draft)

> Factual reporting only. Numbers, protocol, and table captions are provided. Every interpretive, comparative-
> claim, positioning, or limitation-framing sentence is left as `[AUTHORS: ...]`.

## 1. Experimental protocol
Detection used the physics-grounded PRISM feature pipeline. FaceForensics++ (FF++, c23) was partitioned by the
official identity split (720 train / 140 validation / 140 test identities); target and source identities of any
manipulated clip share a split. For cross-dataset evaluation on Celeb-DF-v2 we constructed an identity-disjoint
dev/test partition: identity components were formed and the dominant component was cut by a balanced spectral
(Fiedler) sweep, yielding a dev half (2,421 videos; 426 real / 1,995 fake; 32 identities) and a sealed test half
(2,273 videos; 372 real / 1,901 fake; 27 identities); 1,427 fakes whose two source identities spanned the dev/test
boundary were dropped. All model development used the dev half; the test half was sealed behind an access gate and
evaluated once. Seed 42 throughout; feature imputation used the training-partition median and a StandardScaler fit
on training data only. The classifier was a rank-averaged ensemble of a random forest, an extra-trees classifier,
and a gradient-boosted tree (LightGBM), trained on FF++ train (real vs the four manipulation families).
[AUTHORS: relationship of this evaluation protocol to the paper's primary in-distribution protocol]

## 2. Representation
Each of 13 per-frame spatial forensic features was summarised over the sampled frames by 11 order statistics
(mean, standard deviation, minimum, maximum, the 10th/25th/75th/90th percentiles, inter-quartile range, skewness,
kurtosis), producing 143 order-statistic features. Concatenated with the 37 temporal features, the 3 mouth-ROI
(G1) features, and the 13 per-frame spatial means, this yields the 196-dimensional representation. The 50-D and
53-D configurations are the leading 50 and 53 columns of this vector (53-D = 50-D + the 3 mouth-ROI features).

## 3. In-distribution results (FF++ test)
On the FF++ official test split (685 videos; 137 real / 548 fake), the frozen 196-D ensemble obtained the
per-manipulation and pooled AUCs in **Table T2**: Deepfakes 0.907, Face2Face 0.796, FaceSwap 0.831,
NeuralTextures 0.833; mean-of-four 0.842; pooled 0.842 (95% CI [0.802, 0.880]). The 53-D configuration obtained
mean-of-four 0.836 and pooled 0.836 ([0.796, 0.874]). [AUTHORS: relationship to the paper's existing 53-D
in-distribution results and how the two configurations are presented]

## 4. Cross-dataset results (sealed Celeb-DF-v2 test)
On the sealed Celeb-DF-v2 test half (2,273 videos), the frozen 196-D ensemble obtained AUC 0.7133 (95% CI
[0.687, 0.746]); the 53-D configuration 0.6830 ([0.656, 0.718]); the 50-D configuration 0.6573 ([0.630, 0.685])
(**Table T1**). Paired DeLong tests on the shared test set gave p = 2.2×10⁻⁹ (196-D vs 53-D) and p = 1.7×10⁻⁶
(196-D vs 50-D). The pre-registered prediction for the sealed test AUC, recorded before unsealing, was 0.68 with
an 80% interval [0.65, 0.71]. [AUTHORS: interpretation of the cross-dataset result and of the pre-registered
prediction vs actual]

## 5. In-distribution / cross-dataset trade-off
**Table T3** reports both models on both axes: 53-D obtained 0.836 in-distribution and 0.683 cross-dataset; 196-D
obtained 0.842 in-distribution and 0.713 cross-dataset. [AUTHORS: interpretation of the in-distribution /
cross-dataset trade-off]

## 6. Negative results
Beyond the representation change, the programme evaluated the levers in **Table T4** on the dev half (57 dev
evaluations, multiplicity-controlled by Holm correction; inclusion bars +0.005 in-distribution and +0.03
cross-dataset). Feature-addition families (gradient structure tensor; dimensionless ratios; cardiac coherence;
muscle co-activation; blink kinematics; rigid 3-D geometry; multi-scale Laplacian-of-Gaussian frequency;
temporal-difference features) returned cross-dataset deltas between −0.014 and +0.006. Unsupervised domain
adaptation (CORAL; subspace alignment; per-domain standardisation; per-domain quantile alignment; pseudo-label
self-training) returned cross-dataset deltas between −0.090 and −0.002. Training-distribution changes (self-blended
videos; diverse-real augmentation; diverse-fake augmentation; denser frame sampling) returned cross-dataset deltas
between −0.020 and −0.003. Inference-time augmentation (test-time augmentation, N=2 and N=3) returned −0.001 and
−0.004. Ensembling and aggregation variants (random-subspace feature bagging; per-manipulation ensemble; windowed
multiple-instance aggregation) returned between −0.015 and +0.001. A second independent cross-dataset target
(WildDeepfake, 168 videos) yielded AUC 0.55–0.58 for every candidate. [AUTHORS: framing of the negative-results
contribution and its mechanism]

## 7. Protocol transparency
**Table T5.** 57 dev evaluations preceded the sealed test; the sealed budget was one, spent once. The pre-registered
prediction (0.68 [0.65, 0.71]) and the actual sealed result (0.7133) are reported together. Split construction,
seed, and imputation are as in §1.

## 8. Limitations (factual)
- The 95% CI lower bound for the sealed 196-D cross-dataset AUC is 0.687; the interval therefore includes values
  below 0.70. The result is reported as 0.713 [0.687, 0.746].
- The sealed half is a custom identity split, not Celeb-DF-v2's official test protocol; these numbers are not
  directly comparable to published Celeb-DF figures.
- The sealed half (2,273 videos / 27 identities) is measurably easier than the full Celeb-DF-v2 set: the 50-D
  baseline scores 0.657 on this half versus 0.632 on the full set. No "0.632 → 0.713" comparison is made; the
  like-for-like figures are the sealed-test deltas (+0.030 over 53-D, +0.056 over 50-D).
- No full-Celeb-DF number is reported for the 196-D model because the full set contains the dev half used for
  model selection.
- The single sealed evaluation was invoked twice in the audit log: the first invocation crashed in the FF++-test
  reporting branch before emitting or saving any metric; the second re-captured the identical deterministic result
  on the pre-committed frozen model (see `SEALED_PROVENANCE.md`).
- Real-class recall on Celeb-DF remains low (0.183 at θ = 0.5); AUC is threshold-free, and thresholding does not
  address the ranking (EXP-4).
- [AUTHORS: relationship to the paper's existing 53-D in-distribution results and how the two configurations are
  presented]
