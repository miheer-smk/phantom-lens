# Track D — Report (dev phase closed 2026-07-25). Values/tables only; framing is authors'.

Goal (aspirational): raise in-dist mean (53-D 0.932) and/or zero-shot Celeb-DF (0.632) via additive
physically-grounded features, under a sealed-set anti-overfitting protocol. **Result: no family qualifies;
frozen model = 53-D unchanged; sealed sets never touched (0 evaluations).**

## Every family tried (dev only; FF++ val + celebdf_dev)
| Family | type | in-dist Δ | cross Δ | verdict |
|---|---|---|---|---|
| G1 mouth ROI (Track C, prior) | additive | **+0.118** (NT) | 0.000 | included earlier (53-D) |
| H gradient structure tensor | additive | +0.0013 | −0.0137 | reject |
| J-a domain-invariant ratios | additive | +0.0058 (F2F) | +0.0022 | reject (cross target) |
| J-b quantile alignment | transform | 0 | +0.0104 → **noise** | reject (multiplicity) |
| M cardiac cross-modal coherence | additive | −0.0002 | −0.0141 | reject |
| Q muscle co-activation | additive | +0.0012 (F2F+.0023, NT+.0028) | +0.0001 | reject |
| R blink kinematics | additive | +0.0001 | −0.0028 | reject |
| T rigid 3-D (Tomasi–Kanade) | additive | +0.0011 | −0.0104 | reject |
| M+Q+R+T combined | additive | +0.0021 (p_holm .46) | +0.0070 (p_holm 1.0) | reject |
| Group I ocular physics | — | skipped (author) | — | — |

## Unsupervised domain adaptation (celebdf_dev, vs zero-shot 0.6157)
| CORAL | Subspace align (10/20/30) | per-domain standardise | per-domain quantile |
|---|---|---|---|
| −0.0396 | −0.087 / −0.090 / −0.055 | −0.0024 | −0.0151 |
All fail; most hurt. (Resolves author_decisions #11: Table 11 CORAL/IFD → negative, reproducible.)

## Protocol accounting
- **Sealed-set evaluations performed: 0** (`src/sealed.py` gate; budget 1, unused — nothing to confirm since frozen = 53-D).
- **Dev evaluations: 17** (`trackD_dev_evals.txt`), Holm-corrected; cross inclusion threshold +0.03, in-dist +0.005.
- Pre-registered families: 8 (H,I,J,K,M,Q,R,T); K not reached / I skipped; J-b post-hoc-flagged as noise.
- Thresholds & corrections were TIGHTENED mid-track after multiplicity was raised (documented in `trackD_preregistration.md`).

## Findings (factual)
1. Only **region-localized in-distribution** features ever helped (G1 +0.118 on NeuralTextures). Every family since is < +0.006 in-dist and none survives Holm.
2. **Nothing transfers cross-dataset** — 4 additive families ≈0/negative, and 5 standard UDA methods all negative.
3. Q's mechanism prediction was **directionally correct** (F2F/NT largest) but ~40× too small; redundant with G1.
4. Frozen model remains **53-D**; the paper's locked numbers (in-dist 0.932, Celeb-DF 0.632) are unchanged and remain authoritative.
