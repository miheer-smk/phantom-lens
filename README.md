# PRISM-50 — reference implementation, frozen protocol and per-video results

Code archive for *PRISM: A Physics-Reality Integrated Signal Multistream Framework for Explainable
Deepfake Detection Using Handcrafted Forensic Features* (Scientific Reports).

This archive contains everything needed to reproduce every number in the paper **from the released
feature matrices**, plus the extraction code needed to regenerate those matrices from raw video.
Read [§ Reproducibility caveats](#reproducibility-caveats) before doing the latter — the two paths
are not identical, and we say exactly how they differ.

---

## What PRISM-50 is

Fifty handcrafted forensic descriptors computed per video — 13 spatial, 37 temporal — scored by a
LightGBM classifier. No learned features, no pretrained backbone. The representation is designed to
be inspectable: every descriptor has a stated physical or forensic motivation, and the group ablation
and SHAP analyses in the paper attribute performance to named descriptor families.

## Operating domain — read this first

**PRISM-50 is defined for full-frame video at native frame rate with intact acquisition and encoding
history.** It is not a general-purpose face-video representation.

| Precondition | Descriptors that depend on it |
|---|---|
| background context present | 7 (PRNU face-vs-periphery, face–background luminance, PRNU face-vs-background, 3 boundary-coherence, skin–background decorrelation) |
| original codec intact | 5 (Benford deviation, blocking artifact, double compression, 2 codec temporal residual) |
| native frame rate known | 6 (blink rate, blink duration, 4 rPPG) |
| sequence length ≥ gates | up to 37 (30 frames temporal, 60 frames rPPG) |

Datasets distributed as **pre-cropped face sequences** violate the first three simultaneously.
Critically the affected descriptors **do not fail detectably** — the background mask degenerates to a
rim of crop-edge pixels and the codec descriptors read the export encoding, so both return plausible
values computed against the wrong substrate.

Run the checker before applying PRISM to a new corpus:

```python
from src.evaluation.substrate import check_substrate, summarise
r = check_substrate("clip.mp4")
r.in_operating_domain      # False if any precondition fails
r.n_undefined, r.n_unreliable
```

Measured compliance: FaceForensics++, Celeb-DF v2 and the five video-native DF40 subsets satisfy all
four preconditions (≥ 99.2% of videos). WildDeepfake satisfies **none**.

## Layout

```
configs/     lightgbm.yaml  prism50.yaml  logistic_regression.yaml  random_forest.yaml
splits/      ffpp_identity_split.csv  evaluation_manifest.csv
src/         preprocessing/ landmarks/ spatial_features/ temporal_features/
             aggregation/ classifiers/ evaluation/
experiments/ run_prism50.py run_classifier_comparison.py run_group_ablation.py
             run_bootstrap.py run_shap.py run_domain_shift.py run_rppg_analysis.py
             run_compression_analysis.py run_df40_eval.py run_attrition_report.py
             run_calibration.py run_table19_comparison.py
results/     final_video_scores/  table_values/  figure_source_data/
tests/       test_determinism.py
```

## The frozen protocol

- **Split.** Official FaceForensics++ 720 / 140 / 140 **identity-disjoint** split of the 1000 source
  sequences (`splits/ffpp_identity_split.csv`, seed 42). Every clip is assigned by source identity.
  `assert_no_identity_overlap()` runs before every fit and fails loudly on any leak.
- **Classifier.** `configs/lightgbm.yaml` is the source of truth:
  `n_estimators=200, learning_rate=0.05, num_leaves=31, min_child_samples=20, max_depth=6,
  class_weight=balanced, random_state=42`.
  **`max_depth=6` is load-bearing** — omitting it gives 6200 leaves instead of 5579 and different
  scores.
- **Feature order is alphabetical**, not CSV column order:
  `sorted(c for c in cols if c[:2] in ("s_","t_"))`. Sorting is not cosmetic; get it wrong and every
  tree split moves.
- **Scaler and imputation fit on the training partition only.** Zero-shot targets (Celeb-DF v2,
  WildDeepfake, DF40) are imputed with the pooled FaceForensics++ **training** median and never
  contribute to any statistic.
- **Primary results are fit on the training partition only** (720 identities).

## Data licensing — what this archive does and does not contain

**This archive contains derived measurements only.** It redistributes **no video, no frames, no
face crops and no model checkpoints** from any dataset.

What is here, and why each is derived rather than redistributed:

| Artifact | What it is | Why it is derived data |
|---|---|---|
| `features/` | 50 scalar descriptors per video, plus a path and a label | An irreversibly lossy statistical summary — noise variance-mean ratio, PRNU energy, landmark temporal stability and so on. No frame, face or identity can be reconstructed from 50 numbers. |
| `results/per_video_scores/` | one probability per video, per method | Strictly less information than the feature matrices: a single scalar per video. |
| `splits/` | video ids, identity ids, partitions, exclusion reasons | Metadata describing how the corpora were partitioned and which videos were evaluated. Contains no pixel data. |
| `figures/` | aggregate plots | Population-level summaries; no per-subject imagery. |

**The underlying datasets remain governed by their own agreements.** FaceForensics++, Celeb-DF v2,
WildDeepfake and DF40 each require you to obtain access directly from their providers under their
own terms — see [`docs/DATASET_ACCESS.md`](docs/DATASET_ACCESS.md) for the access forms and the
expected directory layout. Nothing in this archive grants any right to the underlying data, and
possessing this archive is not a substitute for the required agreements.

The MIT licence in `LICENSE` covers **the code and the derived measurements in this repository
only**. It does not and cannot extend to the datasets.

## Citing this archive

This release is archived on Zenodo with a permanent DOI:

> **DOI: [INSERT ZENODO DOI]**

Cite the software record alongside the paper:

> [Author list], "PRISM-50: reference implementation, frozen evaluation protocol and per-video
> results," Zenodo, 2026, version `v1.0.0-prism-srep-revision2`. doi: **[INSERT ZENODO DOI]**

The same DOI appears in the paper's Code Availability statement and reference list. `CITATION.cff`
carries it in machine-readable form — update both when Zenodo issues it.

## Baseline comparison — Table 19 and Section 6.8

**Table 19 and the Section 6.8 comparison are fully recomputable from the released per-video
scores.** `results/per_video_scores/` holds video-level probabilities for Xception, LSDA and
PRISM-50 across all five evaluation targets, and

```bash
./venv/bin/python experiments/run_table19_comparison.py
```

recomputes every published value from them — including the paired DeLong statistic — and checks
each against the published figure. It needs no checkpoint, no face crops and no dataset access.
At the time of release it reports **7/7 published values reproducing within 5e-4**.

**Regenerating the score files themselves requires a locally trained checkpoint.** The training
script and configuration are provided for that purpose:

| Path | What it is |
|---|---|
| `baselines/scripts/xception_train.py` | Xception baseline training, incl. the normalisation constants |
| `baselines/scripts/xception_prep.py` | face-crop extraction, 8 frames/video at 299×299 |
| `baselines/scripts/exp_g9_xception_predictions.py` | per-video scoring and aggregation |
| `baselines/scripts/pD1_lsda_eval.py` | LSDA evaluation (DeepfakeBench) |
| `baselines/scripts/pD2_xception_rerun.py` | Xception re-scoring under the unified aggregation |
| `baselines/configs/lsda_prism.yaml` | LSDA training configuration |

> **Checkpoints and face crops are deliberately not distributed** — size, and the
> FaceForensics++ terms do not permit redistribution of derived imagery. Obtain the datasets from
> their providers and train with the scripts above.
>
> The two `pD*` scripts are included **verbatim, as the record of exactly what was run**. Their
> paths are specific to the authors' machine and must be adapted; they are provenance, not a
> turnkey entry point. The turnkey path is `run_table19_comparison.py` above.
>
> **A note on normalisation.** `pD2_xception_rerun.py` reads its normalisation constants directly
> out of `xception_train.py` rather than restating them. An earlier version hard-coded different
> values, and the model silently accepted inputs it had never been trained on and returned
> plausible-looking scores — every AUC depressed by 0.026–0.062, with no error. If you adapt these
> scripts, keep that coupling.

## Released feature matrices — start here

The archive ships the **extracted 50-D feature matrices** (`features/`, ~27 MB), so every table can
be reproduced with **no dataset access at all**:

> **Path convention.** `video_path` (and `file_path` in `df40_prism50.jsonl`) is stored
> **dataset-relative**, e.g. `FaceForensics++/original_sequences/youtube/c23/videos/000.mp4`.
> Evaluation depends only on the basename — identity and partition are derived from it — so the
> prefix is irrelevant to every result in this archive, and the reproduction gate returns the same
> 9/9 either way.
>
> **One consequence, if you re-extract rather than use the shipped matrices.** The per-video RNG
> seed is derived by hashing the video path *string* (`_video_seed` in
> `src/preprocessing/precompute_features_seeded.py`). At extraction time that string was an
> absolute path on the authors' filesystem, so **the seed is not portable**: re-extracting the same
> video from a different location produces a different seed and therefore different values for the
> sampler-dependent temporal noise-stability descriptor. Extraction is in any case not bit-exact
> across runs — see the reproducibility statement — so this does not add a new *kind* of gap, but
> it does mean a re-extraction will not match the shipped matrices exactly. **Use the shipped
> matrices to reproduce the published tables.**

```bash
python -m venv venv && ./venv/bin/pip install -r requirements.txt
export PRISM_FEATURES=$PWD/features
./venv/bin/python experiments/run_prism50.py            # Tables 7, 8, 13 - checks each against published
./venv/bin/python experiments/run_attrition_report.py   # R1-C5
./venv/bin/python experiments/run_bootstrap.py          # R1-C7
./venv/bin/python experiments/run_classifier_comparison.py   # R1-C6
./venv/bin/python experiments/run_group_ablation.py
./venv/bin/python experiments/run_shap.py --ablation results/table_values/group_ablation.csv  # R1-C8
./venv/bin/python experiments/run_domain_shift.py       # R1-C3B
./venv/bin/python experiments/run_rppg_analysis.py
./venv/bin/python experiments/run_compression_analysis.py
./venv/bin/python experiments/run_df40_eval.py          # R1-C4, gate runs before any AUC
./venv/bin/python experiments/run_calibration.py        # operating threshold + McNemar
./venv/bin/python experiments/run_table19_comparison.py # Table 19 + Section 6.8, from released scores
```

Raw video is needed only to regenerate the matrices themselves.

## Datasets are not redistributed

Place them yourself; see `docs/DATASET_ACCESS.md` for the expected layout of FaceForensics++,
Celeb-DF v2, WildDeepfake and DF40, and for where to request each.

## Reproducibility caveats

Stated plainly, because they matter to anyone re-running this.

1. **Extraction was non-deterministic in the version that produced the published results.** The
   extractor contained a single unseeded `np.random.choice` sampling 100 face-mask pixels for
   temporal spectral entropy. Exactly one of the fifty descriptors — `t_noise_spectral_entropy` —
   varied run to run. **This archive ships the seeded fix**
   (`src/preprocessing/precompute_features_seeded.py`) and a regression test
   (`tests/test_determinism.py`) that asserts bit-identical output in-process and across processes.
   The **released feature matrices were not re-extracted**; they are the matrices behind the paper.
   Re-extracting from raw video with the fixed code therefore produces slightly different values for
   that one descriptor. The measured effect on reported AUCs is given in the paper's
   reproducibility statement.
2. **A single NaN descriptor discards the whole video** in the version used for the reported results:
   `extract_codec_temporal_residual` can histogram an all-NaN array, and the surrounding
   `try/except` returns `None` for the entire video. This archive ships a per-descriptor guard as a
   **documented improvement**; it would retain some videos the reported run excluded, so it is
   **off by default** and the reported population is preserved. See `docs/EXCLUSIONS.md`.
3. **Environment.** `requirements.txt` pins the exact versions used. LightGBM minor-version drift
   changes tree structure; do not substitute.

## Citation

See `CITATION.cff`.
