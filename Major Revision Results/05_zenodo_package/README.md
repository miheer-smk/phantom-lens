# PRISM / PhantomLens — Reproducibility Package (Scientific Reports Major Revision)

Physics-grounded, interpretable deepfake detection. All results are leakage-free (identity-disjoint
official FaceForensics++ split), deterministic (seed 42), and regenerable from the scripts here.

## Layout
- scripts/            feature extraction + all experiment scripts (one per experiment)
- splits/             ffpp_official_split.json (720/140/140 identity split), pillar_map.json (20 pillars)
- results/            all result tables/JSON per experiment (02_tables mirror)
- features_regenerated/  the extracted feature CSVs (50-D spatial+temporal, ROI/G1, PRNU-residual,
                     rPPG, c40-compressed). These are numeric descriptors — NOT video content — so
                     every headline number regenerates directly from them WITHOUT re-downloading the
                     license-gated FF++/Celeb-DF videos. 32 files.
- requirements.txt, ENVIRONMENT.txt, SEED.txt, DATASET_ACCESS.md

## Reproduce numbers directly from deposited features (no video download needed)
The scripts default to reading `features/*.csv`; point them at `features_regenerated/` (or symlink it
to `features/`) and re-run any experiment below to reproduce the exact locked numbers, seed 42.
NOT deposited (license): raw FF++/Celeb-DF videos and the derived face crops (see DATASET_ACCESS.md);
trained Xception checkpoint (regenerable; author's private off-machine backup).

## Protocol (critical)
Identity-disjoint splitting via scripts/protocol.py (assert_no_identity_overlap runs at the start of
every experiment). Thresholds/calibration derived on FF++ validation ONLY. Test sets (FF++ test,
Celeb-DF, WildDeepfake, DFDC) never used for tuning/selection.

## Reproduce a table (examples; after obtaining videos + extracting features)
- Clean baseline:            python scripts/baseline_clean.py         -> baseline.json
- Track C (ROI G1):          python scripts/track_c_measure.py        -> track_c*.json
- Per-pillar ablation:       python scripts/pillar_ablation.py ; scripts/pillar_only.py
- SHAP stability:            python scripts/shap_stability.py
- Threshold calibration:     python scripts/exp4_calibration.py
- Compression c23/c40:       python scripts/exp3_compression.py
- Runtime profiling:         python scripts/exp5_runtime.py
- PRNU residual comparison:  python scripts/exp8_analyze.py
- rPPG POS/CHROM:            python scripts/exp9_analyze.py
- Case-level SHAP:           python scripts/exp10_case_shap.py
- Statistical tests:         python scripts/exp11_stats.py
- Feature redundancy:        python scripts/exp12_redundancy.py
- Xception baseline:         python scripts/xception_prep.py ; scripts/xception_train.py

## Key honest numbers (identity-disjoint)
In-dist per-manip (53-D): Deepfakes 0.978, FaceSwap 0.969, Face2Face 0.875, NeuralTextures 0.905.
Cross-dataset zero-shot: Celeb-DF v2 0.632, WildDeepfake ~0.52 (face-crop caveat).
Xception baseline: FF++ 0.990, Celeb-DF 0.821 (GPU, 83MB, not explainable).

SEED = 42 everywhere. Retired leaky numbers (0.9939/0.9991/0.9999/0.6989) are NOT reported.
Actual video upload + Zenodo DOI = author action (not performed here).
