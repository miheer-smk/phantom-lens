# Backup and recovery

What exists, where, and what it would take to rebuild anything that does not.

## What this archive contains

Everything needed to reproduce every published number **without dataset access**:

| | |
|---|---|
| `features/` | extracted 50-D descriptor matrices, 19 files |
| `results/per_video_scores/` | video-level probabilities for PRISM-50, Xception and LSDA |
| `results/table_values/`, `results/R2_MASTER_RESULTS.json` | every reported value with provenance |
| `splits/`, `configs/` | frozen partitions and frozen hyperparameters |
| `experiments/` | twelve runners |

Reproducing Tables 7, 8, 13, 19 and Section 6.8 needs nothing beyond this directory and
`requirements.txt`.

## What this archive deliberately does **not** contain

| Excluded | Why | How to obtain or rebuild |
|---|---|---|
| Dataset videos (FF++, Celeb-DF v2, WildDeepfake, DF40) | Redistribution is not permitted under their terms | Request from each provider; see `docs/DATASET_ACCESS.md` |
| Extracted face crops and preprocessed frames | Derived imagery, same restriction; ~30 GB | `baselines/scripts/xception_prep.py` regenerates them from source video |
| Model checkpoints (Xception, LSDA) | Size; and they encode training data | Train with `baselines/scripts/xception_train.py` and `baselines/configs/lsda_prism.yaml` |

**Consequence, stated plainly:** the per-video scores in this archive are sufficient to *verify*
every published number, but *regenerating them from raw video* requires the datasets and a
locally trained checkpoint. Those are two different reproducibility claims and only the first is
satisfied offline.

## Reproducibility limits you should know before relying on this

Three are documented in full in `DEFECTS.md`; the short version:

- **Extraction is not bit-exact across runs** (DEFECT-008). Re-extracting features from video will
  not reproduce the shipped matrices exactly. A 500-draw Monte Carlo places every published value
  inside its 95% interval, but the values are not identical.
- **A single NaN descriptor discards the whole video** (DEFECT-009), so the evaluated population is
  itself sensitive to extraction noise — reproducible to roughly ±1.5% on Celeb-DF (DEFECT-010).
- **The per-video RNG seed is derived from the video path string**, which was absolute at extraction
  time, so the seed is not portable. See the path-convention note in `README.md`.

**Use the shipped feature matrices and per-video scores to reproduce the published tables.**
Re-extraction is for extending the work, not for verifying it.

## Author-side backup

The working repository is larger than this archive and is not fully captured by it. For the
authors' own recovery:

| Artifact | Contents | Recoverable from the archive? |
|---|---|---|
| `git bundle create prism_r2_full.bundle --all` | all source, results, drafts, manifests, figures, full history and tags | No — the archive is a snapshot, not the history |
| Trained checkpoints | LSDA (~676 MB, ~14 GPU-hours), Xception (~80 MB) | No — retrain |
| `legacy/` provenance tree | the original scoring scripts and locked numbers that document the published protocol | No |
| Datasets, preprocessed frames, virtual environments | ~56 GB | Re-obtainable; environments rebuild from `requirements.txt` |

Verify a bundle before trusting it:

```bash
git bundle create prism_r2_full.bundle --all
git clone prism_r2_full.bundle /tmp/verify
git -C /tmp/verify rev-parse HEAD    # must equal the source HEAD
```

A bundle that has never been cloned has not been tested.
