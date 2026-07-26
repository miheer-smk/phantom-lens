# ⚗️ best-revision — exploratory Track D / Track E work

**This branch is NOT the resubmission state.** It is branched from `major-revision` (the leakage-free,
resubmission-ready revision) and adds exploratory feature-engineering and training-distribution experiments
(Track D + Track E) on top of that foundation. Read all four points below before using any number here.

## (a) What this branch is
Exploratory work **beyond** the frozen revision: richer temporal aggregation (E1), KS-stability / rPPG-drop
ablations (X1/X2), unsupervised domain adaptation (Track D-B), the physics feature families H/M/Q/R/T,
the Self-Blended-Video generator + extractor (E3), the Celeb-DF identity-disjoint dev/test split, all
pre-registrations, the freeze criteria, and everything downstream. It sits on the same leakage-free base
(identity-disjoint protocol, official splits, `LOCKED_NUMBERS.md`, `results_clean/`) as `major-revision`.

## (b) Every cross-dataset number here is DEV, not sealed
All Track D/E cross-dataset AUCs on this branch — including the E1 gain **Celeb-DF 0.627 → 0.660** — are
**`celebdf_dev` results** (the ~50% development half of `splits/celebdf_dev_test.json`). They are **NOT**
sealed-test results. **The sealed evaluation budget is 1 and remains UNSPENT** (`src/sealed.py`;
`sealed_eval_count() == 0`). No number here has been confirmed on held-out sealed data.

## (c) The paper's locked numbers are unchanged
The manuscript's authoritative results remain those on **`major-revision`**:
**in-distribution 0.932 (53-D) · cross-dataset Celeb-DF 0.632 (full set).**
Nothing on this branch changes them unless and until a **single sealed evaluation** is run on the frozen
extended set and reported. Until then, treat every best-revision cross-dataset figure as a promising
dev-only signal, not a paper result.

## (d) Pointers
- `Major Revision Results/07_summary/LOCKED_NUMBERS.md` — single source of truth (all numbers + provenance; Track D/E sections at the end).
- `Major Revision Results/07_summary/trackD_preregistration.md` · `trackE_preregistration.md` — hypotheses, predicted directions, **freeze criteria**, multiplicity rules.
- `Major Revision Results/07_summary/trackD_report.md` · `trackD_dev_evals.txt` — Track D results + the running dev-evaluation ledger (multiplicity accounting).
- `Major Revision Results/07_summary/author_decisions.md` — flagged author-decision / contribution items (incl. E1 and X1 findings).

**Branch discipline:** `main` (pre-leak-fix) · `major-revision` (frozen, resubmission-ready) · `best-revision` (this, exploratory). Kept distinct; do not merge.
