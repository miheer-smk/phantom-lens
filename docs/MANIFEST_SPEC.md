# Evaluation manifest — specification conformance

`splits/evaluation_manifest.csv`, 20,524 rows × 23 columns. Every reported table is generated
from this manifest, so sample populations cannot drift between tables.

## Columns specified for this release, and their status

### Frozen-manifest columns

| Required | Present | Note |
|---|---|---|
| `dataset` | ✅ | |
| `video_id` | ✅ | |
| `source_id` | ✅ | **alias of `source_video_id`**; both are kept, because existing result files reference `source_video_id`. The columns are identical by construction. |
| `identity_id` | ✅ | |
| `class` | ✅ | |
| `manipulation` | ✅ | |
| `split` | ✅ | |
| `filepath` | ✅ | **dataset-relative**, matching the layout in `DATASET_ACCESS.md`. Absolute paths are deliberately not stored. |

### Master-manifest columns (protocol audit)

`dataset`, `video_id`, `source_video_id`, `identity_id`, `manipulation`, `class`, `split`,
`original_available`, `landmark_success_fraction`, `feature_valid`, `exclusion_reason`,
`final_model_evaluation` — all ✅, in that order.

### Per-video preprocessing fields

| Required | Present |
|---|---|
| `total_frames` | ✅ |
| `sampled_frames` | ✅ |
| `frames_with_landmarks` | ✅ |
| `landmark_success_ratio` | ✅ |
| `spatial_valid_frames` | ❌ **absent — see below** |
| `temporal_valid_frames` | ❌ **absent — see below** |
| `rppg_valid_frames` | ❌ **absent — see below** |
| `video_retained` | ✅ |
| `exclusion_reason` | ✅ |

## Why three per-stage frame counts are absent

**They were never logged.** The extraction pipeline records validity at **video** granularity, not
per descriptor family per frame: a video is retained or not, and the stage that failed is recorded
in `exclusion_reason` under a controlled vocabulary that keeps the four causes distinct —
`excl_file_missing`, `excl_decode_failure`, `excl_mediapipe_no_face`,
`excl_insufficient_valid_frames`, `excl_feature_computation_failure`.

Recovering `spatial_valid_frames`, `temporal_valid_frames` and `rppg_valid_frames` would require
**re-extracting all 20,524 videos**. That is not done, for two reasons:

1. It is out of scope for this revision.
2. It would not answer the question cleanly anyway. Extraction is **not bit-exact across runs**
   (`DEFECTS.md` DEFECT-008), and the evaluated population is itself reproducible only to about
   ±1.5% on Celeb-DF (DEFECT-010). A re-extraction would produce per-stage counts for a *slightly
   different population* than the one behind the published numbers, which would be more misleading
   than the gap it closes.

**How the underlying question is answered instead.** Selection bias is addressed at the level the
data supports — full per-dataset, per-class retention accounting in
`results/P2_attrition_table.csv` (22 rows, 7 specified columns including median landmark success),
the distinct-cause breakdown above, and Fisher's exact tests for real-vs-fake exclusion imbalance
with Holm correction (`results/P2_exclusion_mechanisms.json`): Celeb-DF OR 0.5149, p_holm 1.8e-06;
WildDeepfake OR 0.2517, p_holm 5.7e-04; FF++ and DF40 not significant.

This gap is recorded here rather than left silent, so that anyone diffing the manifest against the
specification finds the reason instead of an unexplained omission.
