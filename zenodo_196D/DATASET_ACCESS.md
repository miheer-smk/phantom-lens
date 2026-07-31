# Dataset Access (raw videos NOT included — require signed licenses from providers)
- FaceForensics++ (c23 & c40): https://github.com/ondyari/FaceForensics — request access, use download script.
  Official identity split (720/140/140) provided in splits/ffpp_official_split.json.
- Celeb-DF v2: https://github.com/yuezunli/celeb-deepfakeforensics — signed agreement required.
- WildDeepfake: https://github.com/OpenTAI/wild-deepfake (Kaggle mirror) — face-crop frames.
- DFDC: https://ai.meta.com/datasets/dfdc/ — test-set ground-truth labels required (n_distinct==2).
Feature CSVs (extracted 50-D + ROI + residual + rPPG) are regenerable from raw videos via
scripts/precompute_features_best.py + the per-experiment extractors (seed 42, deterministic).
