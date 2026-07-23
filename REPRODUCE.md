# Reproducing the PRISM / PhantomLens final tables

All numbers are identity-disjoint, seed 42, and regenerate deterministically from the **committed
feature CSVs** in `features/` — **no raw videos or GPU needed** for the tables below. Run every command
from the **repo root** (scripts do `sys.path.insert(0,"src")` and read `features/`, `splits/`,
write `results_clean/`).

## 0. Environment
```bash
python -m venv .venv && . .venv/bin/activate
pip install -r "Major Revision Results/05_zenodo_package/requirements.txt"
```
Seed is fixed to 42 in every script (`SEED=42`). Splits: `splits/ffpp_official_split.json` (720/140/140).

## 1. Tables reproducible from committed feature CSVs (CPU-only, no raw data)
```bash
L="Major Revision Results/00_logs"
python "$L/baseline_clean.py"        # -> results_clean/baseline.json      (Table: leakage-free FF++, 50-D; CelebDF)
python "$L/track_c_measure.py"       # -> track_c_53D_full.json            (53-D +G1)
python "$L/pillar_ablation.py"       # -> pillar_ablation.csv/json         (group ablation)
python "$L/pillar_only.py"           # -> pillar_only.csv/json
python "$L/shap_stability.py"        # -> shap_stability.json              (SHAP stability)
python "$L/exp3_compression.py"      # -> compression*.json/csv            (c23/c40)
python "$L/exp4_calibration.py"      # -> calibration.csv/json             (threshold calibration)
python "$L/exp8_analyze.py"          # -> prnu_comparison*                 (residual descriptors)
python "$L/exp9_analyze.py"          # -> rppg_comparison*                 (rPPG POS/CHROM)
python "$L/exp12_redundancy.py"      # -> redundancy*                      (feature redundancy)
python "$L/run_delong.py"            # -> delong*.csv/json                 (DeLong 53-vs-50, pillars)
python "$L/exp11_stats.py"           # -> statistical_tests.*              (DeLong/McNemar/Wilcoxon/Holm)
python "$L/exp_m4_missingness.py"    # -> missingness_audit.json           (M4)
python "$L/exp_g1_hardneg.py"        # -> hardneg_deepfakes.json           (G1 hard negatives)
python "$L/eval_wdf.py"              # -> zeroshot_wilddeepfake.json        (WildDeepfake zero-shot)
python "$L/exp_g9_predictions.py"    # -> predictions_per_video.csv        (G9 per-video, PRISM regimes)
```

## 2. Steps that additionally need raw data / GPU (set env vars; not needed for the tables above)
```bash
export FFPP_ROOT=/path/to/FaceForensics++         # exp5_runtime.py, extractors
export WILDDEEPFAKE_ROOT=/path/to/WildDeepfake/test
export DATASETS_ROOT=/path/to/datasets_parent     # exp10_signals.py, xception_prep.py
python "$L/exp5_runtime.py"                        # runtime profiling (needs FF++ videos)
python "$L/xception_prep.py" && python "$L/xception_train.py"   # Xception baseline (crops + GPU)
python "$L/exp_g9_xception_predictions.py"        # Xception per-video (crops + GPU)
python "$L/pub_figures.py"                        # publication figures
```
No script contains a hardcoded absolute path; dataset roots come from the env vars above (relative
defaults under `data/`). Dataset download/access: see
`Major Revision Results/05_zenodo_package/DATASET_ACCESS.md`.

## 3. Auditability
`results_clean/predictions_per_video.csv` stores per-video predictions for every regime; the DeLong /
McNemar statistics recompute from it (`delong_53vs50_from_predictions.csv`,
`prism_vs_xception_from_predictions.json`) and match the locked values exactly.
