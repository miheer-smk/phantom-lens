# G8 — Celeb-DF v2 zero-shot AUC reconciliation (three-way)

**Authoritative value (locked): AUC = 0.632, 95% CI [0.613, 0.654].**
Everything else on this page is superseded and must NOT appear as a current result.

## The three numbers and why they differ
| Value | Source | Protocol | Fake-recall | Status |
|---|---|---|---|---|
| **0.6989** | Manuscript / reviewer doc | pre-fix, **NOT identity-disjoint** (test fakes reachable in training regime); one frozen model snapshot | 0.8745 | **retired (leaky/inflated)** |
| **0.6867** | Old repo (`results/…`) | pre-fix, **NOT identity-disjoint**; a *different* saved run (different model snapshot / feature build) than the manuscript one | 0.9224 | **retired (leaky; also internally inconsistent with 0.6989)** |
| **0.632** | `results_clean/baseline.json` (`baseline_clean.py`, seed 42) | **identity-disjoint** official FF++ 720/140/140; M1 train-only imputer; bootstrap 95% CI | recall(fake)=0.781 | **AUTHORITATIVE** |

## Explanation of the discrepancies
1. **0.6989 vs 0.6867 (manuscript vs old repo):** both were produced by the *same* leakage-prone
   (non-identity-disjoint) pipeline but from **two different frozen runs** — they disagree on fake-recall
   (0.8745 vs 0.9224), i.e. they are not even the same model/threshold. Which was "ground truth" was an
   open ambiguity (recon A2) that was **never resolved**; the gap is run/config drift within the old
   leaky setup, not a meaningful methodological difference. Neither is defensible.
2. **→ 0.632 (clean):** re-evaluating under the identity-disjoint official split (no source identity shared
   between FF++ train and any evaluation set) with the locked LightGBM and the M1 train-only imputer
   yields **0.632 [0.613, 0.654]** over n=6121 Celeb-DF videos (798 real / 5323 fake). The drop from
   ~0.69 to 0.63 is the honest cost of removing the leakage/optimistic-config, consistent with the
   overall re-baselining. WildDeepfake (~0.52) corroborates the "moderate, real-class-domain-mismatched"
   cross-dataset behaviour.

## Decision
- **Lock 0.632 as the single authoritative Celeb-DF v2 zero-shot AUC.**
- Retire 0.6989 and 0.6867 entirely; record here that they came from an unresolved, leakage-prone
  pipeline and disagreed with each other (fake-recall 0.8745 vs 0.9224).
- Framing "strong generalization" → "moderate zero-shot with real-class domain mismatch" (author text;
  see `author_decisions.md` items 2 & 11).
