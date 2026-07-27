# Reproducibility Snapshot — Track D/E (best-revision)

**Captured:** 2026-07-27 · **Branch:** `best-revision` · **Remote:** `git@github.com:miheer-smk/phantom-lens.git`
This snapshot pins the exact state needed to reproduce every Track D/E cross-dataset result. Paper numbers
(0.932 in-dist / 0.632 cross) live on `major-revision` and are untouched by this branch.

## Environment
- Python **3.12.3**, venv at `./.venv` (full freeze: `requirements_snapshot.txt`, 88 packages)
- Key pins: numpy 1.26.4 · pandas 3.0.3 · scikit-learn 1.7.2 · scipy 1.15.3 · lightgbm 4.6.0 ·
  opencv-contrib 4.11.0.86 / headless 4.9.0.80 · mediapipe 0.10.18 · torch 2.11.0+cu128
- Determinism: `seed=42` everywhere; RandomForest/ExtraTrees `random_state=42`; LightGBM `deterministic=True, force_row_wise=True`.

## Data inventory (identity-disjoint; sealed test never touched)
- Celeb-DF-v2: `/home/iiitn/Datasets/Celeb-DF-v2` (6529 mp4). Dev/test split: `splits/celebdf_dev_test.json`
  (dev 2421 = 426r/1995f · test_SEALED 2273 = 372r/1901f · 1427 identity-spanning fakes dropped).
- FaceForensics++ c23: `/home/iiitn/Datasets/FaceForensics++` (10525 mp4). Training = youtube reals + 4 manip families.
- DFD (Google DeepFakeDetection) real actors: `original_sequences/actors/c23/videos` (363 mp4, 28 actors) —
  downloaded via `data_prep/download.py <FFpp_root> -d DeepFakeDetection_original -c c23 -t videos`.

## Frozen-candidate lineage (dev CV, identity-grouped 5-fold on celebdf_dev)
- 53-D locked → +E1 order-statistics (196-D) → RF_d8 = **0.7018**
- + strong-member rank ensemble (RF+ExtraTrees+LGBM_d6) = **0.7125** (current running-best dev)
- PENDING: 100-frame full pass; DFD-fakes-in-training.
- **Sealed evaluations spent: 0** (budget 1). No test tuning. See `sealed.py` gate.

## Feature artifacts (gitignored — too large for git; back up separately)
- `features/trackE/plain_everyone_E3.csv` — 196-D, 60-frame, all 6547 videos (the R0 representation).
- `features/trackE/plain_dfd_reals.csv` — 196-D DFD reals (226 usable of 363).
- `features/trackE/perframe_{ffpp_trainval_fixed,celebdf_dev}.csv` — persisted per-frame spatial series (E1/E2/E5 reuse).
- `features/trackE/plain_everyone_100.csv` — 100-frame full pass (IN PROGRESS at snapshot time).

## Reproduce each result (all write to `results_clean/*.json`)
| Result | Script | dev CV |
|---|---|---|
| Classifier sweep (RF wins) | `exp_trackE_clfsweep.py` | RF 0.7018 |
| E4 LoG frequency (rejected) | `exp_trackE_E4_eval.py` | +0.0055 |
| Self-training (fails) | `exp_trackE_selftrain.py` | −0.031… |
| X4 diverse reals (fails) | `exp_trackE_X4_eval.py` | −0.020 |
| E5 temporal-difference (fails) | `exp_trackE_tempdiff.py` | −0.0017 |
| Denser sampling subset (+) | `exp_trackE_denser_compare.py` | +0.052 subset |
| **Strong-member ensemble (+)** | `exp_trackE_ens2.py` | **0.7125** |
| 100-frame full eval | `exp_trackE_frameval.py <csv>` | pending |
Scripts live in `Major Revision Results/00_logs/`; run with `./.venv/bin/python`. Multiplicity ledger:
`trackD_dev_evals.txt` (52 evals). Single source of truth: `LOCKED_NUMBERS.md`. Pre-registrations:
`trackD_preregistration.md`, `trackE_preregistration.md`.

## To rebuild from scratch
1. `python -m venv .venv && ./.venv/bin/pip install -r requirements_snapshot.txt`
2. Place datasets at the paths above; build split: `python build_celebdf_devtest.py`.
3. Extract 196-D: `./.venv/bin/python src/extract_trackE_SBV.py --plain --manifest features/trackD/manifest_everyone_E3.csv --output features/trackE/plain_everyone_E3.csv --max_frames 60`
4. Run any eval script above; compare JSON to `LOCKED_NUMBERS.md`.
