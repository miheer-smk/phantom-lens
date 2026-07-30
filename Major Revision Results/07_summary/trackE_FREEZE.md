# Track E — FREEZE DOCUMENT (locked 2026-07-30, before sealed unseal)

The frozen model is fixed here by an **a priori rule**, not by picking the dev maximum. Sealed budget: 1, unspent.

## Frozen feature set
**196-D E1-expanded representation:** 13 spatial per-frame means + 37 temporal features + 3 G1 mouth-ROI
features + **143 spatial order-statistics** (11 stats × 13 spatial channels). The order-statistics component is
the one qualified representation gain (celebdf_dev in-dist +0.0092 Holm-sig, cross +0.0326 Holm-sig; ledger #20).
Extraction: `extract_trackE_SBV.py --plain --max_frames 60`, train-only median imputation, StandardScaler on train.

## Frozen model
**RF + ExtraTrees + LGBM_d6, RANK-averaged ensemble** (equal weight, rank of each member's P(fake) averaged).
- RF: 400 trees, depth 8, min_leaf 5, class_weight balanced.
- ExtraTrees: 600 trees, depth 10, min_leaf 4, class_weight balanced.
- LGBM_d6: 300 est, lr 0.05, num_leaves 31, max_depth 6, class_weight balanced, deterministic.
- Trained on FF++ train ONLY (real vs 4 manipulation families). seed 42 throughout.

## Selection rationale — a priori robustness, NOT dev-argmax
Rank-averaging a small set of strong, diverse learners is a **standard variance-reduction procedure** chosen on
general grounds (ensembles reduce estimator variance and are less sensitive to any single model's domain-specific
quirks). We commit to it a priori, **not** because it scored 0.7125 on celebdf_dev. Framing it as "it topped dev"
would be curse-prone; the dev number is a consequence of the choice, not its justification.

**The choice is not outcome-determining.** Ensemble 0.7125 vs ExtraTrees 0.7036 on celebdf_dev differ by 0.009 —
well inside the sealed-test noise (27 identities → bootstrap CI ≈ ±0.025–0.03). Either freezes to statistically
indistinguishable test performance. We freeze the ensemble; had we frozen ExtraTrees the conclusion would not change.

## WildDeepfake evidence — what it does and does NOT license (caveat)
Second target WildDeepfake (168 videos) shows all candidates at AUC ~0.55–0.58 and shows the ensemble's dev edge
does **not** replicate there (ensemble 0.5746 vs ExtraTrees 0.5834) — valid evidence of **winner's-curse
inflation** in the celebdf_dev argmax. BUT WildDeepfake n=168 → SE ≈ ±0.045, so it **cannot adjudicate between
candidates separated by ~0.01**. It is NOT evidence that ExtraTrees is truly better than the ensemble. We use it
only to (a) justify a conservative predicted test number and (b) avoid over-trusting the dev argmax — **not** to
swap one noisy argmax (celebdf_dev) for another (WildDeepfake). The a priori robustness choice stands on its own.

## Pre-registered prediction (see trackE_preregistration.md)
Celeb-DF-v2 sealed TEST AUC: **point 0.68, 80% interval [0.65, 0.71]**. FF++ test ~0.90–0.94.

## THE sealed evaluation — exact commands (freeze → unseal in 3 steps)
```
cd /home/iiitn/Downloads/phantom-lens-main
# 1) build authoritative sealed-test manifest (label-agnostic; asserts == 2273 videos)  [ALREADY DONE]
.venv/bin/python "Major Revision Results/00_logs/exp_trackE_SEALED_eval.py" --make_test_manifest
# 2) PREREQUISITE extraction (label-agnostic, does NOT spend budget, ~3-4h):
.venv/bin/python src/extract_trackE_SBV.py --plain \
    --manifest features/trackD/manifest_celebdf_test.csv \
    --output features/trackE/plain_celebdf_test.csv --max_frames 60
# 3) dry-run check (spends nothing):
.venv/bin/python "Major Revision Results/00_logs/exp_trackE_SEALED_eval.py"
# 4) THE single sealed evaluation (spends budget 1; logs the unseal):
.venv/bin/python "Major Revision Results/00_logs/exp_trackE_SEALED_eval.py" --unseal
```
Reports Celeb-DF-v2 test AUC + identity-grouped bootstrap 95% CI, FF++ test AUC, single-model reference, and
predicted-vs-actual. Writes `results_clean/SEALED_final.json`. After this, dev iteration does NOT re-open.

## Methods disclosure (required for credibility)
Report in the paper's Methods: **56 celebdf_dev evaluations** preceded the single sealed test; model and
representation were fixed by the a priori rule above; the sealed Celeb-DF test half (27 identities / 2273 videos,
identity-disjoint from dev) was evaluated exactly once.
