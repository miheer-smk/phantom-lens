<div align="center">

<img src="https://img.shields.io/badge/PHANTOM_LENS-v2.0-0d1117?style=for-the-badge&labelColor=0d1117&color=7c3aed" />
<img src="https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
<img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
<img src="https://img.shields.io/badge/Research-IIIT_Nagpur-orange?style=for-the-badge" />
<img src="https://img.shields.io/badge/Status-Active-22c55e?style=for-the-badge" />

<br/><br/>

```
██████╗ ██╗  ██╗ █████╗ ███╗   ██╗████████╗ ██████╗ ███╗   ███╗    ██╗     ███████╗███╗   ██╗███████╗
██╔══██╗██║  ██║██╔══██╗████╗  ██║╚══██╔══╝██╔═══██╗████╗ ████║    ██║     ██╔════╝████╗  ██║██╔════╝
██████╔╝███████║███████║██╔██╗ ██║   ██║   ██║   ██║██╔████╔██║    ██║     █████╗  ██╔██╗ ██║███████╗
██╔═══╝ ██╔══██║██╔══██║██║╚██╗██║   ██║   ██║   ██║██║╚██╔╝██║    ██║     ██╔══╝  ██║╚██╗██║╚════██║
██║     ██║  ██║██║  ██║██║ ╚████║   ██║   ╚██████╔╝██║ ╚═╝ ██║    ███████╗███████╗██║ ╚████║███████║
╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝   ╚═╝    ╚═════╝ ╚═╝     ╚═╝    ╚══════╝╚══════╝╚═╝  ╚═══╝╚══════╝
```

### *Physics-Anchored Deepfake Detection — PRISM Framework*

**Miheer Satish Kulkarni** · IIIT Nagpur · 2026

[Paper PDF](./Phantom_Lens_Updated_References.pdf) · [Architecture](#-architecture) · [Quickstart](#-quickstart) · [Results](#-results) · [Cite](#-citation)

</div>

---

## What is Phantom Lens?

Deepfakes fool neural networks by mimicking *appearance* — but they cannot fool *physics*.

**Phantom Lens (PRISM)** detects synthetic media by interrogating three fundamental physical laws that every real camera obeys and every GAN/diffusion model subtly violates:

| Pillar | Physical Law | What Gets Measured |
|--------|-------------|-------------------|
| `pillar1_noise.py` | **Sensor noise** follows predictable statistical distributions | Variance-mean ratio, residual std, high-frequency energy |
| `pillar2_light.py` | **Illumination** obeys inverse-square law & geometric consistency | Shadow geometry, specular highlights, lighting direction |
| `pillar3_compression.py` | **Real media** carries JPEG/codec compression fingerprints | Benford's law on DCT coefficients, quantisation tables, block artefacts |

A lightweight **4-layer MLP** fuses the 24-dimensional physics feature vector into a binary real/fake prediction — no heavy backbone, no transformer, no pretrained vision model.

---

## 🏛 Architecture

```
Video Input
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│                    FEATURE EXTRACTION                        │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │  PILLAR  1   │  │  PILLAR  2   │  │    PILLAR  3     │  │
│  │ Sensor Noise │  │  Lighting &  │  │   Compression    │  │
│  │              │  │   Shadows    │  │   Fingerprints   │  │
│  │ f₁ vmr       │  │ f₄ illum     │  │ f₇  benford      │  │
│  │ f₂ res_std   │  │ f₅ specular  │  │ f₈  block_art    │  │
│  │ f₃ hi_freq   │  │ f₆ shadow    │  │ f₉  quant_err    │  │
│  │              │  │              │  │ f₁₀ dct_ratio    │  │
│  └──────┬───────┘  └──────┬───────┘  └───────┬──────────┘  │
│         └─────────────────┴──────────────────┘             │
│                           │                                  │
│              [ 12-dim physics vector ]                       │
│               + 12 cross-pillar terms                        │
│                    = 24-dim input                            │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                  PhysicsClassifier MLP                       │
│                                                             │
│   BatchNorm(24) → Linear(24→64) → ReLU → BN → Dropout(0.3) │
│              → Linear(64→32) → ReLU → BN → Dropout(0.2)     │
│              → Linear(32→16) → ReLU → BN → Dropout(0.1)     │
│              → Linear(16→1)  → Sigmoid                      │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
                    P(fake) ∈ [0, 1]
```

---

## 📁 Repository Structure

```
phantom-lens/
│
├── 📄 README.md
├── 📄 Phantom_Lens_Updated_References.pdf
│
├── 📦 phantom_lens/                  ← installable core package
│   ├── __init__.py
│   ├── pillars/                      ← physics feature extractors
│   │   ├── pillar1_noise.py          ✦ sensor noise fingerprints
│   │   ├── pillar2_light.py          ✦ illumination consistency
│   │   └── pillar3_compression.py   ✦ codec artefact analysis
│   └── data/                         ← data I/O utilities
│       ├── dataset.py
│       └── video_utils.py
│
├── 🚀 v2/                            ← LATEST PIPELINE (6 core files)
│   ├── precompute_features_v2.py     ✦ extract & cache 24-dim vectors
│   ├── train_v2.py                   ✦ full training w/ ablation & eval
│   ├── analyze_eda.py                ✦ exploratory data analysis
│   ├── analyze_pca.py                ✦ PCA feature visualisation
│   ├── analyze_tsne_iterations.py    ✦ t-SNE convergence study
│   └── analyze_tsne_perplexity.py    ✦ t-SNE perplexity sweep
│
├── 🧪 tests/
│   ├── test_pillars.py               ← unit tests for all 3 pillars
│   └── check_auc.py                  ← quick AUC sanity check
│
└── 🗄 legacy/                        ← archived earlier pipelines
    ├── precompute_features.py
    ├── precompute_fake_only.py
    ├── train.py
    └── train_fast.py
```

> **`v2/`** is the canonical entry point for all experiments. Everything in `legacy/` is preserved for reproducibility but superseded by v2.

---

## ⚡ Quickstart

### 1 · Install dependencies

```bash
git clone https://github.com/miheer-smk/phantom-lens.git
cd phantom-lens

pip install torch torchvision numpy scikit-learn matplotlib opencv-python
```

### 2 · Precompute physics features

```bash
# Expects your dataset at data/  with real/ and fake/ subdirectories
python v2/precompute_features_v2.py \
    --data_dir   data/ \
    --output     data/precomputed_features.pkl \
    --workers    4
```

This extracts all 24 physics features and caches them to a `.pkl` for fast training iteration.

### 3 · Train the classifier

```bash
python v2/train_v2.py
```

The training script runs with:
- **Video-level train/val split** (no frame leakage)
- **3-seed ensemble** (seeds 42, 123, 777)
- **Physics-aware augmentation** (noise, brightness, compression simulation)
- **Early stopping** (patience = 20 epochs)
- **Ablation study** (each pillar tested independently)

Outputs land in `checkpoints/` and `results/`.

### 4 · Explore the feature space

```bash
python v2/analyze_pca.py               # 2D PCA of real vs fake
python v2/analyze_tsne_perplexity.py   # t-SNE perplexity sweep
python v2/analyze_eda.py               # full EDA report
```

---

## 📊 Results

| Configuration | AUC | F1 | Accuracy |
|---|---|---|---|
| Pillar 1 only (Sensor Noise) | — | — | — |
| Pillar 2 only (Lighting) | — | — | — |
| Pillar 3 only (Compression) | — | — | — |
| Pillars 1 + 2 | — | — | — |
| Pillars 1 + 2 + 3 (Full Model) | **0.XXXX ± 0.XXXX** | **0.XXXX** | **XX.X%** |

> Fill in with your trained values from `results/training_report.txt`. Ablation numbers are auto-generated by `train_v2.py`.

**Training configuration:**

```
Batch size  : 512      Learning rate : 1e-3 (CosineAnnealing)
Max epochs  : 100      Weight decay  : 1e-4
Val split   : 20%      Early stop    : patience 20
Seeds       : 42, 123, 777
```

---

## 🔬 Physics Pillars — Deep Dive

<details>
<summary><strong>Pillar 1 · Sensor Noise Fingerprinting</strong></summary>

Real camera sensors produce noise that follows a Poisson-Gaussian mixture model. GAN generators and diffusion models produce noise with fundamentally different statistics — too uniform, too structured, or with high-frequency energy concentrated in unphysical frequency bands.

**Features extracted:**
- `f1_vmr` — variance-to-mean ratio of residual noise
- `f1_residual_std` — standard deviation of high-pass residual
- `f1_high_freq` — proportion of power in the top octave

</details>

<details>
<summary><strong>Pillar 2 · Illumination & Shadow Consistency</strong></summary>

Real scenes have a single dominant light source (or a physically consistent mix). Deepfake faces are often composited under inconsistent lighting — the face may appear lit from a different direction than the background, or specular highlights may not correspond to any real light source.

**Features extracted:**
- `f2_lighting` — illumination gradient consistency score
- `f2_specular` — specular highlight plausibility (Phong model fit)
- `f2_shadow` — shadow direction angular deviation

</details>

<details>
<summary><strong>Pillar 3 · Compression Fingerprints</strong></summary>

Every real JPEG/video frame carries a forensic record of compression history in its DCT coefficient distribution. Benford's law holds for natural images but is systematically violated by synthesised media. Block artefact energy and quantisation table signatures provide additional discriminating signals.

**Features extracted:**
- `f3_benford` — Benford's law deviation in DCT magnitudes
- `f3_block` — 8×8 block boundary artefact energy
- `f3_quant` — quantisation table fingerprint residual

</details>

---

## 🛠 How to Reorganise the Repo

If you've cloned the original flat layout, run the provided migration script to adopt the modular structure:

```bash
chmod +x reorganize.sh
./reorganize.sh
```

This creates the `v2/`, `phantom_lens/`, `tests/`, and `legacy/` folders and moves every file to its correct location without deleting anything.

---

## 📜 Citation

If Phantom Lens or the PRISM framework informs your research, please use the following reference:

```
╔══════════════════════════════════════════════════════════════════╗
║  PHANTOM LENS — PRISM Framework                                  ║
║  Physics-Anchored Deepfake Detection                             ║
╠══════════════════════════════════════════════════════════════════╣
║  Author   Miheer Satish Kulkarni                                  ║
║  Affil.   Indian Institute of Information Technology, Nagpur      ║
║  Year     2026                                                    ║
║  Repo     github.com/miheer-smk/phantom-lens                     ║
╚══════════════════════════════════════════════════════════════════╝
```

BibTeX:

```bibtex
@software{kulkarni2026phantomlens,
  author       = {Kulkarni, Miheer Satish},
  title        = {{Phantom Lens}: Physics-Anchored Deepfake Detection
                  ({PRISM} Framework)},
  year         = {2026},
  institution  = {Indian Institute of Information Technology, Nagpur},
  url          = {https://github.com/miheer-smk/phantom-lens},
  note         = {Independent research. All rights reserved.}
}
```

For the accompanying reference list see [`Phantom_Lens_Updated_References.pdf`](./Phantom_Lens_Updated_References.pdf).

---

## ⚖️ License & Rights

© 2026 Miheer Satish Kulkarni — IIIT Nagpur. All rights reserved.

This repository contains original independent research. The code and methodology may not be reproduced, modified, or distributed without explicit written permission from the author.

---

<div align="center">
<sub>Built with physics, not just parameters.</sub>
</div>
