<!--
================================================================
  FILE: README.md
  GOES IN: github.com/miheer-smk/phantom-lens
  HOW: Replace the existing README.md with this file
  NOTE: License badge says "Research Only" — DO NOT add an MIT LICENSE
        file to this repo without discussing with your supervisor.
================================================================
-->

<h1 align="center"> Phantom Lens</h1>

<p align="center">
  <b>Physics-Anchored Deepfake Detection Framework (PRISM)</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/LightGBM-Classifier-brightgreen" />
  <img src="https://img.shields.io/badge/MediaPipe-Face%20Mesh-orange" />
  <img src="https://img.shields.io/badge/License-Research%20Only-lightgrey" />
  <img src="https://img.shields.io/badge/Status-Active%20Research-yellow" />
</p>

<p align="center">
  <b>Researcher:</b> Miheer Satish Kulkarni — IIIT Nagpur, 2026<br>
  <b>Supervisor:</b> Dr. Nileshchandra K. Pikle, Assistant Professor, CSE, IIIT Nagpur
</p>

---

## About

Phantom Lens is a physics-grounded deepfake detection framework that takes a fundamentally different approach from CNN-based detectors. Instead of learning texture artifacts that break when a new generator is released, it checks whether a video obeys the laws of real-world physics.

**The core asymmetry:** A generative model must simultaneously replicate dozens of physical constraints — sensor noise statistics, light transport, physiological signals, lens optics, compression traces. A detector only needs to catch *one* violation. Phantom Lens exploits this.

The system is built on **PRISM V3** (Physics-Reality Integrated Signal Multistream), a 50-feature extractor anchored to MediaPipe facial landmarks across 19 active physics pillars (13 spatial + 37 temporal). A LightGBM classifier trained on FaceForensics++ achieves near-perfect in-distribution detection and meaningful cross-dataset generalisation to CelebDF v2 without any target-domain data.

---

## Key Results

### In-Distribution — FaceForensics++ (c23, multi-manipulation)

| Manipulation | Best Model | AUC | F1 |
|---|---|---|---|
| Deepfakes | LightGBM | 0.9709 | 0.9633 |
| Face2Face | LightGBM | 0.8818 | 0.7671 |
| FaceSwap | Random Forest | 0.9999 | 0.9971 |
| NeuralTextures | Random Forest | 0.9991 | 0.9946 |

10-fold CV AUC (training set): LR = 0.895 · RF = 0.900 · **LGBM = 0.921**

### Cross-Dataset — CelebDF v2 (zero-shot, never seen during training)

| Metric | Value |
|---|---|
| AUC-ROC | **0.6867** |
| Average Precision | 0.9222 |
| Fake Recall (TPR) | 0.9224 |
| Fake F1 | 0.9095 |
| Test videos | 6,129 (806 real + 5,323 fake) |

> Cross-dataset degradation is expected and well-studied; the high Average Precision (0.9222) confirms strong ranking quality. Domain adaptation is in progress.

---

## Physics Pillars

### Pillar 1 — Sensor Noise (`src/pillars/pillar1_noise.py`)
Real cameras produce signal-dependent Poisson shot noise and a unique Photo-Response Non-Uniformity (PRNU) fingerprint. GAN and diffusion models rarely replicate these correctly.

| Feature Group | What it measures |
|---|---|
| Noise VMR / ResStd / HFRatio | Variance-to-mean ratio; noise residual standard deviation; high-frequency noise fraction |
| PRNU Energy / Face-Periph ratio | Camera sensor fingerprint energy; face vs background PRNU disparity |
| Temporal Noise Stability | Frame-to-frame noise correlation; spectral entropy of noise band |

### Pillar 2 — Light Transport & Geometry (`src/pillars/pillar2_light.py`)
Deepfakes struggle to maintain physically consistent illumination, specular reflections, and facial micro-geometry across time.

| Feature Group | What it measures |
|---|---|
| Shadow score / Face-BG diff | Illumination physics consistency; face-to-background lighting discontinuity |
| Specular stability / symmetry | Temporal coherence of highlight positions; bilateral symmetry of specular reflections |
| Landmark trajectory / rigidity | Facial motion jitter; jaw-chin rigidity; interpupillary stability |
| Blink dynamics | Blink rate, duration, and symmetry — neuromuscular physics hard to fake |

### Pillar 3 — Compression Forensics (`src/pillars/pillar3_compression.py`)
Video codecs leave statistical fingerprints that disappear or change character in synthesized content.

| Feature Group | What it measures |
|---|---|
| Benford deviation | DCT coefficient distribution vs Benford's Law — detects double-compression |
| Block artifact / DCT temporal | H.264 blocking artifacts; temporal DCT coefficient stability |
| Codec residual entropy | Inter-frame residual distribution — changes under GAN synthesis |

### Pillar 4 — Physiological Signals (rPPG)
Remote photoplethysmography (rPPG) extracted via the CHROM method measures the cardiac pulse signal embedded in skin colour variation. Deepfakes cannot synthesize a coherent rPPG signal because it requires physically accurate blood-flow simulation.

| Feature | What it measures |
|---|---|
| rPPG SNR / Peak prominence | Signal-to-noise of cardiac frequency; prominence of heartbeat peak |
| Inter-region correlation | rPPG coherence across forehead, cheeks, nose — real faces are correlated |
| rPPG–Motion coupling consistency | Physics law: head motion and rPPG are coupled in real video |

---

## Project Structure

```
phantom-lens/
│
├── src/                              # Core library
│   ├── pillars/
│   │   ├── pillar1_noise.py          # Sensor noise physics features
│   │   ├── pillar2_light.py          # Light transport & geometry features
│   │   └── pillar3_compression.py    # Compression forensics features
│   ├── models/                       # Classifier wrappers
│   └── utils/
│       ├── video_utils.py            # Video decoding and frame sampling
│       └── dataset.py                # Dataset loaders
│
├── features/                         # PRISM V3 feature extractor (50 features)
│   ├── precompute_features_best.py   # Main extractor — 13 spatial + 37 temporal
│   ├── precompute_features_v3.py     # V3 extractor variant
│   └── rppg_extractor.py             # Standalone rPPG signal extractor
│
├── training/                         # Training scripts
│   ├── train.py                      # LR / RF / LGBM training pipeline
│   ├── train_v3.py                   # V3 multi-manipulation training
│   └── train_v3_best.py              # Best-model training with CV + test eval
│
├── evaluation/                       # Evaluation and analysis scripts
│   ├── cross_dataset_eval.py         # Zero-shot cross-dataset evaluation
│   ├── evaluate_v3_best.py           # Full metrics on best model
│   └── validate_ffpp_indistribution.py  # In-distribution FF++ validation
│
├── data_prep/                        # Dataset preparation
│   ├── prepare_ffpp.py               # FaceForensics++ preprocessing
│   ├── prepare_celebdf.py            # CelebDF v2 preprocessing
│   └── download.py                   # Dataset download utility
│
├── analysis/                         # EDA, t-SNE, visualisation
│   ├── eda/                          # Exploratory data analysis
│   ├── tsne/                         # t-SNE feature space visualisation
│   └── visualization/                # Publication-quality plot generators
│
├── experiments/                      # Ablation and pillar experiments
│   ├── test_pillars.py               # Unit tests for each physics pillar
│   └── test_no_codec_norm.py         # Codec normalisation ablation
│
├── config/                           # Dataset paths and training hyperparameters
│   ├── datasets.py
│   └── training.py
│
├── results/                          # All experiment outputs
│   ├── exp1/                         # Multi-manipulation evaluation (FF++)
│   │   ├── run_exp1.py               # Reproducer script
│   │   ├── results.json              # Full metrics
│   │   ├── roc_combined.png          # ROC curves per manipulation
│   │   └── confusion_matrices.png
│   ├── exp2/                         # Compression comparison (c23 / c0 / c40)
│   ├── exp3/                         # Feature ablation (SHAP, top-k subsets)
│   ├── exp5/                         # Hard-negative analysis (false negatives)
│   ├── exp_celebdf/                  # Cross-dataset CelebDF v2 evaluation
│   │   ├── run_celebdf_eval.py       # Reproducer — extracts features + evaluates
│   │   ├── results.json
│   │   ├── threshold_sweep.csv       # Metrics at t=0.20…0.80
│   │   ├── confusion_matrix_pr_curve.png
│   │   └── threshold_analysis.png
│   ├── generate_final_pdf.py         # 10-page PDF report generator
│   └── final_report.pdf              # Latest compiled report
│
├── dockerfile                        # GPU-enabled Docker environment
├── pyproject.toml                    # Package metadata (setuptools)
├── requirements.txt                  # Full dependency list
└── requirements_gpu.txt              # GPU/CUDA specific dependencies
```

---

## Installation

### Prerequisites
- Python 3.9+
- CUDA-capable GPU recommended (tested on NVIDIA RTX 4060)
- FFmpeg installed system-wide

### Install dependencies

```bash
git clone https://github.com/miheer-smk/phantom-lens.git
cd phantom-lens

# CPU
pip install -r requirements.txt

# GPU (CUDA 12.x)
pip install -r requirements_gpu.txt
```

### Docker (recommended)

```bash
docker build -f dockerfile -t phantom-lens .
docker run --gpus all -v /path/to/data:/data phantom-lens
```

---

## Usage

### 1 — Extract PRISM features from a video directory

```bash
# Real videos (label=0), recursive, 8 parallel workers
python features/precompute_features_best.py \
    --video_dir /data/celebdf/real \
    --output features/celebdf_real.csv \
    --label 0 \
    --max_frames 150 \
    --workers 8

# Fake videos (label=1)
python features/precompute_features_best.py \
    --video_dir /data/celebdf/fake \
    --output features/celebdf_fake.csv \
    --label 1 \
    --max_frames 150 \
    --workers 8
```

### 2 — Run the multi-manipulation experiment (FF++)

```bash
python results/exp1/run_exp1.py
# Outputs: results/exp1/results.json, roc_combined.png, confusion_matrices.png
```

### 3 — Cross-dataset evaluation on CelebDF v2

```bash
python results/exp_celebdf/run_celebdf_eval.py
# Extracts features (auto-resumes if interrupted), trains on FF++, tests on CelebDF
# Outputs: results/exp_celebdf/results.json, threshold_analysis.png
```

### 4 — Feature ablation (SHAP)

```bash
python results/exp3/run_exp3.py
# Outputs: SHAP rankings, ablation AUC curves, CV fold distribution
```

### 5 — Generate the full PDF report

```bash
python results/generate_final_pdf.py
# Outputs: results/final_report.pdf  (10 pages, no recomputation)
```

---

## Requirements

Core dependencies (see `requirements.txt` for pinned versions):

| Package | Purpose |
|---|---|
| `torch`, `torchvision` | Deep learning backend |
| `lightgbm` | Primary classifier |
| `scikit-learn` | LR, RF, metrics, CV |
| `mediapipe` | 478-point facial landmark tracking |
| `opencv-python` | Video decoding, image processing |
| `numpy`, `scipy` | Numerical signal processing |
| `matplotlib` | Plots and report figures |
| `pandas` | Feature CSV management |
| `shap` | Feature importance (TreeExplainer) |
| `tqdm` | Progress bars |

---

## Reproducing Results

All experiments are fully reproducible from the extracted feature CSVs in `features/`. No video data is needed after extraction.

| Script | Reproduces |
|---|---|
| `results/exp1/run_exp1.py` | Multi-manipulation FF++ evaluation (AUC 0.97–0.999) |
| `results/exp2/run_exp2.py` | Compression comparison (c23 available; c0/c40 require download) |
| `results/exp3/run_exp3.py` | SHAP feature ablation |
| `results/exp5/run_exp5.py` | Hard-negative (false positive) analysis |
| `results/exp_celebdf/run_celebdf_eval.py` | Full CelebDF v2 cross-dataset pipeline |
| `results/generate_final_pdf.py` | 10-page PDF report |

---

## Report

A compiled research report covering all experiments is available at:

📄 [`results/final_report.pdf`](results/final_report.pdf)

The report includes:
- PRISM V3 pipeline methodology
- Per-manipulation ROC curves and confusion matrices
- SHAP feature importance rankings and ablation results
- Hard-negative analysis (13/957 missed DeepFakes, P(fake) ∈ [0.24, 0.46])
- CelebDF v2 cross-dataset results and threshold sweep
- Key insights and limitations

---

## Limitations & Future Work

- **Cross-dataset real-class boundary**: 70% of CelebDF real videos are falsely flagged at threshold=0.50 — caused by domain shift between FF++ c23 and YouTube-compressed CelebDF real videos. Planned fix: domain-adapted real boundary using physics-invariant feature subset.
- **Compression c0/c40 unavailable**: Only c23 tested. Full compression generalisation study pending dataset download.
- **Temporal window**: Features computed over ≤150 frames (~6s). Longer-sequence modelling is a future direction.

---

## Citation

If you use this work, please cite:

```bibtex
@misc{kulkarni2026phantomlens,
  author       = {Kulkarni, Miheer Satish},
  title        = {Phantom Lens: Physics-Anchored Deepfake Detection Framework (PRISM)},
  year         = {2026},
  institution  = {Indian Institute of Information Technology, Nagpur},
  note         = {Active research, manuscript in preparation}
}
```

---

## Author

**Miheer Satish Kulkarni**
B.Tech CSE, IIIT Nagpur
[github.com/miheer-smk](https://github.com/miheer-smk)

*Supervised by Dr. Nileshchandra K. Pikle — Assistant Professor, CSE Department, IIIT Nagpur (PhD, IIT Bombay)*
