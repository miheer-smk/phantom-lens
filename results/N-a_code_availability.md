# N-a — Code Availability section (ready to paste)

**Placement:** immediately after Data Availability, before Image Source Statement.
**The statement below is the agreed wording for the Code Availability section**, with only the
three bracketed placeholders filled where they can be. The DOI stays bracketed until Zenodo
issues it.

---

**Code Availability**

> The custom code used to implement the PRISM-50 pipeline, extract the spatial and temporal
> forensic descriptors, reproduce the identity-disjoint data partitions, train and evaluate the
> statistical classifiers, perform group-wise ablation, bootstrap confidence-interval estimation,
> calibration, SHAP analysis and generate the reported experimental outputs has been archived in
> Zenodo at DOI: **[INSERT ZENODO DOI]**. The archived release corresponds to version
> **v1.0.0-prism-srep-revision2** used for the results reported in this manuscript and is also
> maintained at **[GITHUB REPOSITORY]**. The repository includes environment specifications, fixed
> configuration files, evaluation manifests and instructions for reproducing the principal
> experiments. Third-party benchmark videos are not redistributed and must be obtained directly
> from their respective dataset providers subject to the applicable access conditions and licences.

---

## Verification — every claim in that paragraph, checked against the archive

The paragraph is a factual statement about what the archive contains, so I checked it clause by
clause rather than pasting it on trust.

| Clause | Where it is satisfied | ✓ |
|---|---|---|
| implement the PRISM-50 pipeline | `src/classifiers/prism_pipeline.py` | ✓ |
| extract the spatial and temporal forensic descriptors | `src/spatial_features/`, `src/temporal_features/`, `src/landmarks/`, `src/preprocessing/precompute_features_seeded.py` | ✓ |
| reproduce the identity-disjoint data partitions | `splits/ffpp_identity_split.csv`, `splits/ffpp_official_split.json`, `load_split()`/`assign_partition()` | ✓ |
| train and evaluate the statistical classifiers | `experiments/run_prism50.py`, `run_classifier_comparison.py`; `configs/{lightgbm,random_forest,logistic_regression}.yaml` | ✓ |
| group-wise ablation | `experiments/run_group_ablation.py` | ✓ |
| bootstrap confidence-interval estimation | `experiments/run_bootstrap.py`, `grouped_ci()` | ✓ |
| **calibration** | `experiments/run_calibration.py` | ✓ **added — see below** |
| SHAP analysis | `experiments/run_shap.py` | ✓ |
| generate the reported experimental outputs | eleven runners writing to `results/table_values/` | ✓ |
| environment specifications | `requirements.txt`, `requirements-gpu.txt`, `environment.yml` | ✓ |
| fixed configuration files | `configs/` (4 YAMLs) | ✓ |
| evaluation manifests | `splits/evaluation_manifest.csv` (20,524 rows, 21 columns) | ✓ |
| instructions for reproducing the principal experiments | `README.md` reproduction section; clean-room test in `scripts/p12_cleanroom.sh` | ✓ |
| third-party videos not redistributed | archive holds extracted **feature matrices only** — no video, no frame imagery | ✓ |

### One clause was not true when I checked it, and is now

**"calibration" had no runner in the archive.** Threshold calibration lived only in the internal
`scripts/p1_6/exp11_stats.py`, which is not part of the release. Pasting the paragraph as supplied
would have made the manuscript assert something about the archive that was false.

I ported it as `experiments/run_calibration.py` rather than deleting the word from the agreed
wording. It derives the operating threshold by maximising macro-F1 on the FF++ **validation partition
only**, applies it unchanged to Celeb-DF v2, and runs the paired McNemar test. The leakage
constraint is asserted in code, not commented.

**It reproduces the canonical result exactly:**

| Quantity | `results/P1_6_trainonly/statistical_tests.json` | `experiments/run_calibration.py` | Δ |
|---|---|---|---|
| calibrated threshold | 0.510 | 0.510 | 0 |
| McNemar χ² | 25.4697 | 25.4697 | 0 |
| p | 4.493860940242397e-07 | 4.493860940242397e-07 | 0 |
| b (only baseline correct) | 54 | 54 | 0 |
| c (only calibrated correct) | 12 | 12 | 0 |

Registered in `README.md` and added to the clean-room test, which now exercises **eleven** runners.

## Still to fill

| Placeholder | Blocked on |
|---|---|
| `[INSERT ZENODO DOI]` | Zenodo deposition (N-b depends on this too) |
| `[GITHUB REPOSITORY]` | the author's decision on the public GitHub URL |
| version string | filled: `v1.0.0-prism-srep-revision2` — tag held for 5 September |

## Editor response — ready to paste once the DOI exists

The Editor-facing response is held with the response letter and needs only the DOI.
