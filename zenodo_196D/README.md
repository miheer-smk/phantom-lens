# Phantom Lens (PRISM) — 196-D representation & sealed cross-dataset evaluation (code & data deposit)

Reproducibility deposit for the physics-grounded deepfake-detection results on branch `outstanding-results` of
github.com/miheer-smk/phantom-lens. Deposited to satisfy the editor's code-deposition requirement. Seed 42.

> **Upload-ready staging only.** The DOI is to be minted by the authors (author list and metadata are permanent).
> `zenodo_metadata.json` holds draft metadata; complete the author list/ORCIDs and the related-identifier (paper
> DOI) before publishing.

## Headline
- **Celeb-DF-v2 sealed test AUC = 0.713 [0.687, 0.746]** — zero-shot, identity-disjoint, single pre-registered
  evaluation of a frozen model (196-D order-statistic representation + RF+ExtraTrees+LightGBM rank ensemble).
- Like-for-like on the same sealed half: 50-D 0.6573, 53-D 0.6830, 196-D 0.7133; paired DeLong p=1.7×10⁻⁶ / 2.2×10⁻⁹.
- In-distribution FF++ test (this frozen model): 0.842 [0.810, 0.872].
- **Caveats:** the sealed half is a custom identity split (not Celeb-DF's official protocol) and is easier than the
  full set (50-D 0.657 here vs 0.632 full); report only the like-for-like deltas (+0.030 over 53-D, +0.056 over 50-D);
  CI lower bound is 0.687 — do not write "above 0.70".

## Contents
```
zenodo_196D/
├── README.md              this file
├── zenodo_metadata.json   draft deposit metadata (DOI unminted)
├── DATASET_ACCESS.md      how to obtain FaceForensics++, Celeb-DF-v2, DFD, WildDeepfake (not redistributable)
├── requirements.txt       pinned environment (Python 3.12.3)
├── MANIFEST_SHA256.txt     SHA-256 of every file in this deposit
├── scripts/               extraction + evaluation + figure scripts
├── splits/                celebdf_dev_test.json, ffpp_official_split.json
├── features/              model-critical 196-D feature CSVs (real videos not redistributed)
├── predictions/           per_video_predictions.csv (frozen model, sealed test + FF++ test)
└── results/               SEALED_final.json + post-freeze + joint-target result JSONs
```

## Reproduce (CPU, no raw video needed for the tables)
```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python scripts/exp_trackE_postfreeze_compare.py   # 50/53/196-D sealed-test AUC + paired DeLong
python scripts/exp_trackE_permanip.py             # FF++ per-manipulation
python scripts/make_figures.py                    # figures
```
Scripts expect the feature CSVs at `features/trackE/` — either symlink `features/` here or edit the `TE` path.
The sealed evaluation (`scripts/exp_trackE_SEALED_eval.py --unseal`) is already spent (budget 1/1); its result is
`results/SEALED_final.json` and must not be re-run for a new number.

## Notes on data
Feature CSVs are derived descriptors, not raw video. The underlying video datasets are **not redistributed**
(license-restricted); obtain them per `DATASET_ACCESS.md` and regenerate features with `scripts/extract_trackE_SBV.py`.
