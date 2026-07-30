# Handoff briefing — PRISM Track E cross-dataset push (to Claude Opus 5)

**From:** Opus 4.8 working session, 2026-07-30 · **Branch:** `best-revision` (frozen `major-revision` untouched)
**Purpose:** you review the full evidence and help design end-game strategy. TTA is extracting (~10h) — the last
un-landed lever. Everything below is DEV-only; the Celeb-DF **sealed test budget is 1, still unspent (0 used).**

## 1. What we're trying to do
PRISM is a physics-grounded (handcrafted forensic) deepfake detector. In-distribution (FF++) is strong
(~0.93 AUC). The Scientific Reports major revision needs a credible **cross-dataset** number on Celeb-DF-v2.
The whole of Track D/E is a disciplined hunt for cross-dataset lift under strict rules: identity-disjoint,
sealed test never touched, seed 42, pre-registration, Holm across a shared dev-eval ledger, report all failures.

## 2. The state of play (celebdf_dev CV, identity-grouped 5-fold)
- Base representation win (early): E1 order-statistics of per-frame physics features took cross-dataset from
  **0.627 → 0.702** — the ONE big gain, a *representation* change (distributional > mean aggregation).
- Best classifier: RandomForest 0.7018 → **RF+ExtraTrees+LGBM rank ensemble = 0.7125** (running-best dev).
- ExtraTrees alone: 0.7036, and it is the **most domain-robust** model (see §4).

## 3. Levers tried and their verdicts (56 dev evals total)
| lever | result | verdict |
|---|---|---|
| E1 order-stats representation | 0.627→0.702 | ✅ the one big win |
| Classifier sweep (RF wins) | 0.7018 | ✅ |
| Strong-member rank ensemble | 0.7125 (+0.0107) | ✅ modest, but see §4 |
| Additive physics families (H/I/J/K/M/Q/R/T) | ~0 | ✗ |
| Unsupervised DA (CORAL/subspace/quantile/per-domain std) | fail | ✗ |
| Pseudo-label self-training (k=10/20/30%) | −0.031/−0.048/−0.057 | ✗ all hurt |
| Self-blended videos (SBV) training aug | collapsed | ✗ CNN-specific, not handcrafted |
| Windowed MIL (E2) | −0.015 | ✗ |
| E4 multi-scale LoG frequency | +0.0055 | ✗ sub-threshold |
| X4 diverse REALS → real class | −0.020 (realRec +0.052) | ✗ fixes threshold, not ranking |
| X4 diverse FAKES → fake class | −0.0098 (realRec −0.042) | ✗ |
| E5 temporal-difference / rel-flicker | −0.0017 | ✗ redundant w/ order-stats |
| Denser sampling (100 frames, full) | −0.003 | ✗ subset +0.052 didn't replicate |
| Random-subspace feature-bagging | +0.0013 | ✗ member diversity beats it |
| TTA (N=3 augment celebdf_dev) | pending (~10h) | ? |

**Read:** every substantive lever except representation (E1) and member-diversity ensembling has failed.
Data augmentation fails on BOTH classes. UDA fails on both alignment and boundary-adaptation. The domain gap
is stubborn and, we now believe, **fundamental to the handcrafted-physics feature space**, not a tuning miss.

## 4. The winner's-curse check (WildDeepfake, second independent target) — the key methodological result
We extracted 196-D on WildDeepfake (168 videos) as a second cross-dataset target and re-scored every candidate:
| candidate | celebdf_dev CV | WildDeepfake AUC |
|---|---|---|
| RF_d8 | 0.7018 | 0.5751 |
| ExtraTrees | 0.7036 | **0.5834** |
| RF+ET+LGBM_rank | **0.7125** | 0.5746 |

Two things: (a) WildDeepfake AUC is ~0.55–0.58 for everything → physics features barely generalize to a truly
different (in-the-wild) domain. (b) The celebdf_dev ranking ≠ the WildDeepfake ranking — the ensemble is best on
celebdf_dev but mid-pack on WildDeepfake; ExtraTrees is best on WildDeepfake. **So the ensemble's +0.0107
celebdf_dev edge is partly dev-overfitting** — empirically confirmed winner's curse over 56 evals.

## 5. My predicted sealed Celeb-DF-v2 TEST AUC (based on all dev numbers)
Reasoning: celebdf_test is the SAME domain as celebdf_dev (held-out identities), so the identity-grouped CV is a
mostly-unbiased estimator EXCEPT for (i) argmax selection bias over 56 evals (≈ −0.01 to −0.02 on the selected
0.7125) and (ii) dev-split vs sealed-split variance (≈ ±0.02–0.03; sealed = 27 identities / ~2273 videos).
WildDeepfake tells us the ensemble edge is fragile, so I weight toward the robust models (~0.70–0.71 dev).

- **Point estimate: celebdf_test AUC ≈ 0.675**
- **80% interval: [0.65, 0.70]** — most likely JUST UNDER the 0.70 target.
- Best realistic case ~0.70 (touches target); downside ~0.64.
- I would NOT predict > 0.70; the honest center is ~0.67.

## 6. My opinion / recommended strategy
1. **We have essentially hit the ceiling of handcrafted-physics cross-dataset transfer (~0.70–0.71 dev).**
   Further dev micro-optimization only inflates the optimism gap (winner's curse is now measurable). Stop fishing.
2. **Freeze the robust choice, not the dev-argmax.** The ensemble and ExtraTrees tie on joint mean; ExtraTrees
   is simpler and most robust across targets. Defensible to freeze either — I lean ensemble for the paper (it is
   the pre-registered "prefer ensembles" choice) but with a CONSERVATIVE predicted test number (~0.67), or
   ExtraTrees if we want the most honest transfer story. Worth a human decision.
3. **The negatives ARE the contribution.** "Distributional representation closes part of the gap; feature
   addition, UDA, blending-aug, and diverse-data-aug do NOT" is a genuine, publishable finding with a clean
   mechanism (the gap is in the feature space, not the classifier or the data quantity).
4. **The only thing likely to materially raise TEST AUC is a hybrid**: physics features + a lightweight learned
   embedding (e.g. a small CNN / frozen face-recognition or self-supervised feature) on the same crops, late-fused.
   That is the SOTA mechanism (SBI-style training-distribution effects live in learned features, not handcrafted).
   BUT it changes the paper's "physics-grounded" framing — a scope decision for the authors, not a tuning step.
5. **Freeze protocol (agreed):** joint selection over celebdf_dev + WildDeepfake; prefer rank ensemble; pre-register
   predicted test point+interval; spend the ONE sealed eval on celebdf_test + FF++ test; report predicted vs actual;
   disclose the 56 dev-eval count in Methods.

## 7. Open items for you to weigh in on
- Does the TTA result (pending) change the freeze pick? (Predict: no — it's transductive inference-time smoothing;
  I expect +0.00 to +0.01, not curse-proof.)
- Ensemble vs ExtraTrees as the frozen model, given the WildDeepfake disagreement.
- Whether to keep the paper physics-only (freeze at ~0.71 dev / ~0.67 test, lead with the negative-results
  contribution) or open a hybrid track (higher test AUC, changed framing).
- What predicted-test point + interval to pre-register (my proposal: 0.675, [0.65, 0.70]).

## UPDATE (TTA landed, 2026-07-30): last lever FAILS
TTA (mean prob over original + N augmented) is monotonically negative: RF 0.7018->0.6985, ensemble
0.7125->0.7088. All levers now resolved. Freeze stands: 196-D E1-expanded + RF+ET+LGBM rank ensemble, dev CV
0.7125. Predicted sealed test 0.68 [0.65,0.71]. Ready for the single sealed evaluation.
