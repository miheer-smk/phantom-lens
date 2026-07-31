# 196D_FINAL — consolidated deliverables (Track E, 196-D programme)

Branch `best-revision`. **This directory is the single consolidated record of the 196-D cross-dataset programme.**
Do not merge into `major-revision` or `main`.

> **All cross-dataset AUC numbers in this directory are SEALED Celeb-DF-v2 TEST numbers unless explicitly
> labelled `dev`.** Dev numbers come from identity-grouped 5-fold CV on the Celeb-DF dev half and were used for
> model selection; the sealed test half was evaluated exactly once.

## Headline
- **Frozen model:** 196-D E1-expanded representation + RF+ExtraTrees+LGBM_d6 rank ensemble, FF++-train only, seed 42.
- **Sealed Celeb-DF-v2 test AUC = 0.7133, 95% CI [0.687, 0.746]** (n=2,273; 27 identities disjoint from dev).
- Same-data baselines: 53-D 0.6830, 50-D 0.6573; paired DeLong 196-vs-53 p=2.2×10⁻⁹, 196-vs-50 p=1.7×10⁻⁶.
- In-distribution FF++ test (this frozen model): mean-of-4 0.842 (distinct from the paper's dedicated in-dist model).
- Pre-registered prediction 0.68 [0.65,0.71] → actual 0.7133. 57 dev evals; sealed budget 1, spent once.

## Directory map
```
196D_FINAL/
  README.md                 ← this file
  INVENTORY.md              ← complete audit table (every measurement + provenance)
  LOCKED_NUMBERS_196D.md    ← every reported number with full provenance (script/commit/seed/SHA/date)
  00_protocol/   freeze doc, pre-registration, sealed provenance, dev-eval ledger,
                 sealed audit log, celebdf & ffpp split JSONs, protocol.py, sealed.py
  01_scripts/    every script that produced a reported number + make_figures.py
  02_features/   model-critical feature CSVs + SHA256_MANIFEST.txt (full canonical set referenced)
  03_results/    all Track D/E result JSONs (dev + sealed + post-freeze)
  04_tables/     publication_tables.md (T1–T5) + T1/T2 CSVs
  05_figures/    fig1 ROC (sealed test) · fig2 in-dist/cross trade-off · fig3 negative-results bar
  06_manuscript/ results_196D_draft.md (factual; interpretation left as [AUTHORS: ...])
```

## Reproduce the reported numbers (from committed features)
```
cd <repo root>
.venv/bin/python "Major Revision Results/00_logs/exp_trackE_postfreeze_compare.py"   # T1 + DeLong
.venv/bin/python "Major Revision Results/00_logs/exp_trackE_permanip.py"             # T2 per-manip
.venv/bin/python 196D_FINAL/01_scripts/make_figures.py                               # figures
```
The sealed evaluation itself (`exp_trackE_SEALED_eval.py --unseal`) has already been spent (budget 1/1) and must
not be re-run for new numbers; its result is in `03_results/SEALED_final.json`.

## Integrity notes
- Environment pinned in `requirements_snapshot.txt` (repo root); Python 3.12.3.
- The sealed unseal log (`00_protocol/sealed_eval_log.txt`) shows two entries — a crash-and-recapture of one
  deterministic evaluation; see `00_protocol/SEALED_PROVENANCE.md`.
- Feature CSV SHA-256s: `02_features/SHA256_MANIFEST.txt`.
