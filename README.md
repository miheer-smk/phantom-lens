# Phantom Lens — PRISM Framework

**Physics-Anchored Deepfake Detection**  
Miheer Satish Kulkarni · IIIT Nagpur · 2026

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square)](https://python.org)
[![Framework](https://img.shields.io/badge/Framework-LightGBM%20%7C%20RF%20%7C%20LR-orange?style=flat-square)]()
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)]()

---

## About

Phantom Lens (PRISM — *Physics-Reality Integrated Signal Multistream*) is a deepfake detection framework that uses **50 hand-crafted physics-grounded features** (37 temporal + 13 spatial) to distinguish real from AI-generated faces — with **no deep learning, no domain adaptation, and no data augmentation tricks**.

The core idea: generative models fool human eyes, but they consistently violate the laws of physics. Sensor noise, rigid facial geometry, skin colour consistency, and compression behaviour are all measurable signals that deepfakes suppress poorly.

---

## Results

### FF++ In-Distribution Validation (5-Fold CV)

| Classifier | AUC | F1 | Precision | Recall | MCC |
|------------|-----|----|-----------|--------|-----|
| Logistic Regression | 0.9653 | 0.9139 | 0.9125 | 0.9154 | 0.8275 |
| Random Forest | 0.9680 | 0.9048 | 0.9110 | 0.8986 | 0.8108 |
| **LightGBM ★** | **0.9742** | **0.9197** | **0.9241** | **0.9154** | **0.8401** |

- Dataset: FaceForensics++ · 956 real + 957 fake · 50/50 balanced
- LightGBM 95% bootstrap CI: AUC [0.968–0.979], F1 [0.906–0.932]
- Optimal threshold θ = 0.50 (no recalibration required)
- Brier Score: 0.0633 · Log Loss: 0.2186

### Cross-Dataset Evaluation (Trained FF++ → Tested Celeb-DF v2)

| Version | Features | Cross-Dataset AUC | vs Baseline |
|---------|----------|-------------------|-------------|
| v2 PRISM (spatial only) | 24 spatial | 0.48 | baseline |
| **v3 PRISM (current)** | **50 (37T + 13S)** | **0.6132** | **+27.7%** |

> Test set: 5894 samples (767 real + 5127 fake). AUC is the primary metric due to class imbalance.  
> Context: SOTA deep learning methods reach 0.82–0.88 using massive augmentation + domain adaptation. PRISM achieves 0.61 with pure physics — above-chance generalisation on the hardest benchmark in the field.

---

## Top Physics Features

By SHAP importance (FF++) and Cohen's d effect size (Celeb-DF v2):

| Feature | Description | Domain |
|---------|-------------|--------|
| `t_nose_bridge_std` | Nose bridge geometry instability across frames | Geometry |
| `t_skin_color_jitter` | Frame-to-frame colour variance from neural rendering | Colour |
| `t_texture_warp_residual` | Blending artefacts at face boundary | GAN Artifacts |
| `s_noise_res_std` | Shot noise residual — strongest cross-dataset signal | Sensor |
| `s_prnu_energy` | Photo-response non-uniformity energy | Sensor |
| `t_dct_temporal_std` | DCT coefficient instability over time | Compression |
| `t_blink_symmetry` | Asymmetric blink patterns | Biological |

---

## Installation

```bash
git clone https://github.com/miheer-smk/phantom-lens.git
cd phantom-lens

python -m venv phantomlens_env
phantomlens_env\Scripts\activate        # Windows
# source phantomlens_env/bin/activate   # Linux/Mac

pip install -r requirements_best.txt
python test_pillars.py                  # verify setup
```

---

## Usage

```bash
# 1. Prepare dataset
python prepare_ffpp_official.py --data_path /path/to/ffpp

# 2. Precompute physics features
python precompute_features_best.py --dataset ffpp --split train

# 3. Train
python train_v3_best.py

# 4. Evaluate in-distribution
python validate_ffpp_indistribution.py

# 5. Cross-dataset evaluation
python cross_dataset_eval.py --source ffpp --target celebdf
```

---

## Supported Datasets

| Dataset | Script |
|---------|--------|
| FaceForensics++ | `prepare_ffpp_official.py` |
| Celeb-DF v2 | `prepare_celebdf.py` |
| CelebVHQ | `prepare_celebvhq_v2.py` |
| WildDeepfake | `prepare_wilddeepfake_v2.py` |
| DeeperForensics | `prepare_deeperforensics.py` |
| DFFD | `prepare_dffd.py` |

---

## Project Structure

```
phantom-lens/
├── src/
│   ├── pillars/               # Core physics feature extractors
│   ├── models/                # Classifier definitions
│   └── utils/                 # Dataset & video utilities
├── notebooks/                 # Exploratory analysis
├── train_v3_best.py           # Best training pipeline
├── precompute_features_best.py
├── evaluate_v3_best.py
├── validate_ffpp_indistribution.py
├── cross_dataset_eval.py
├── rppg_extractor.py
└── requirements_best.txt
```

---

## Technical Report

Full validation details, confusion matrices, ROC curves, SHAP analysis, and cross-dataset breakdown:

📎 [`Phantom_Lens_Updated_References.pdf`](./Phantom_Lens_Updated_References.pdf)

---

## Citation

```bibtex
@misc{kulkarni2026phantomlens,
  title      = {Phantom Lens: Physics-Anchored Deepfake Detection via PRISM Framework},
  author     = {Kulkarni, Miheer Satish},
  year       = {2026},
  institution = {IIIT Nagpur},
  url        = {https://github.com/miheer-smk/phantom-lens}
}
```

---

MIT License · IIIT Nagpur · 2026
