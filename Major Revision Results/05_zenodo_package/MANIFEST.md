# Zenodo Package Manifest (updated 2026-07-23)
scripts: 36 (+ 7 src/ modules) | result dirs: 17 | feature CSVs: 32 | manifests: 2
last repo commit at packaging: 3c2f80c (branch major-revision, github.com/miheer-smk/phantom-lens)

Contents:
- scripts/ all experiment + extraction + stats scripts; scripts/src/ = importable modules
  (protocol.py identity split, leakfree.py train-only imputer, delong.py, roi_config.py)
- splits/ ffpp_official_split.json (720/140/140 identity), pillar_map.json
- features_regenerated/ all 32 extracted feature CSVs (numbers regenerate WITHOUT raw videos)
- manifests/ manifest_ffpp.csv, manifest_celebdf.csv (video/label/split — needed by M4 + Xception per-video)
- results/ all locked result tables/JSON incl. expM4 (missingness), expG1 (hard-neg),
  expG9_per_video_predictions (predictions_per_video.csv + DeLong-from-probs)
- requirements.txt, ENVIRONMENT.txt, SEED.txt (42), DATASET_ACCESS.md, README.md, REPRODUCE.md
- zenodo_metadata.json, zenodo_upload.sh (draft-only)

Reproducibility: every table regenerates from committed feature CSVs via repo-root REPRODUCE.md
(verified on a fresh checkout, bit-identical). No script has a hardcoded absolute path; dataset roots
come from env vars (FFPP_ROOT / WILDDEEPFAKE_ROOT / DATASETS_ROOT).

Exploratory (NOT a source of any locked number, included for completeness): rigorous_search.py,
gate_exp1.py, gate_celebdf.py.
NOT included: raw videos + derived face crops (license-restricted), trained model binaries (regenerable;
author private off-machine backup). Upload/DOI mint = author action.
