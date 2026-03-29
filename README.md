# Phantom Lens — Physics-Anchored Deepfake Detection

A deepfake detection framework that flips the conventional approach: instead of hunting for AI-generated artifacts, it checks whether a video obeys real-world physics. Fakes have to get *everything* right — this system only needs to catch *one* slip.

## Core Insight

Generative models must simultaneously replicate dozens of physical constraints — sensor noise statistics, light transport, lens optics, compression traces. But a detector only needs to find a single inconsistency. This asymmetry is the foundation of Phantom Lens.

## How It Works

The pipeline extracts a **24-dimensional physics-based feature vector** from video frames, organized into 3 pillar groups:

### Pillar 1 — Sensor Noise Analysis (`pillar1_noise.py`)
- **Poisson shot noise** — real cameras produce signal-dependent noise following Poisson statistics. GANs/diffusion models rarely replicate this correctly.
- **PRNU fingerprinting** — every sensor has a unique photo-response non-uniformity pattern. Generated images lack this.
- **Bayer demosaicing artifacts** — real images pass through a color filter array. Synthetic images skip this step, leaving detectable traces.

### Pillar 2 — Light Transport & Geometry (`pillar2_light.py`)
- **Light transport consistency** — checks if illumination follows the Kajiya rendering equation.
- **Specular coherence** — verifies that highlights behave according to physical reflection models.
- **Motion blur directionality** — real motion blur has consistent direction tied to camera/object movement.
- **Optical flow boundary analysis** — checks temporal consistency at object boundaries.
- **Chromatic aberration** — real lenses produce predictable color fringing that's hard to fake.

### Pillar 3 — Compression Forensics (`pillar3_compression.py`)
- **DCT compression history (Benford's Law)** — JPEG/video compression leaves statistical fingerprints in DCT coefficients.
- **Codec temporal residuals** — video codecs produce predictable inter-frame residual patterns.

## Results

| Dataset | Split | AUC | Variance |
|---------|-------|-----|----------|
| FaceForensics++ | Video-level train/test | 0.9745 | — |
| WildDeepfake | Video-level train/test | 0.9745* | — |
| CelebV-HQ | Video-level train/test | 0.9745* | — |
| **CelebDF-v2 (cross-dataset, unseen)** | **Zero-shot** | **0.8961** | **0.0007** |

*\*Combined AUC across 80,000 samples from all three datasets with strict video-level splits.*

Cross-dataset generalization on CelebDF-v2 is the harder test — the model never saw this data during training. Currently working on improving this.

## Project Structure

```
phantom-lens/
├── pillar1_noise.py              # Sensor noise features (Poisson, PRNU, Bayer)
├── pillar2_light.py              # Light transport & geometric consistency
├── pillar3_compression.py        # Compression forensics (DCT, codec residuals)
├── dataset.py                    # Video dataset loader with frame sampling
├── video_utils.py                # Video I/O and frame extraction utilities
├── precompute_features.py        # Batch feature extraction from video datasets
├── precompute_features_v2.py     # Optimized feature extraction pipeline
├── precompute_fake_only.py       # Feature extraction for fake-only analysis
├── train.py                      # Main training script (Random Forest + SVM)
├── train_fast.py                 # Fast training variant for experimentation
├── train_v2.py                   # Training with cross-dataset evaluation
├── check_auc.py                  # AUC computation and threshold analysis
├── analyze_eda.py                # Exploratory data analysis on features
├── analyze_pca.py                # PCA visualization of feature space
├── analyze_tsne_iterations.py    # t-SNE analysis (iteration sweep)
├── analyze_tsne_perplexity.py    # t-SNE analysis (perplexity sweep)
├── test_pillars.py               # Unit tests for pillar feature extractors
└── Phantom_Lens_Updated_References.pdf  # Reference paper and citations
```

## Quick Start

### Requirements
```bash
pip install torch torchvision opencv-python numpy scikit-learn ffmpeg-python matplotlib
```

### Feature Extraction
```bash
# Extract physics features from a video dataset
python precompute_features.py --data_dir /path/to/dataset --output features.npy
```

### Training
```bash
# Train classifier on extracted features
python train.py --features features.npy --labels labels.npy
```

### Evaluation
```bash
# Check AUC on test set
python check_auc.py --features test_features.npy --labels test_labels.npy
```

## What Makes This Different

Most deepfake detectors learn texture-level artifacts that break when they encounter a new generator. Phantom Lens targets **physics violations** that are generator-agnostic — the laws of optics and sensor electronics don't change just because someone released a new model.

This is why cross-dataset generalization (AUC 0.8961 on completely unseen CelebDF-v2) works without any fine-tuning.

## Status

**Active research — work in progress.**

Currently working on:
- Improving cross-dataset AUC on CelebDF-v2 (target: 0.93+)
- Adding temporal consistency analysis across longer video sequences
- Manuscript in preparation for arXiv submission

## Tech Stack

Python, PyTorch, OpenCV, FFmpeg, scikit-learn, NumPy | NVIDIA RTX 4060 GPU

## Research Context

Built under the mentorship of **Dr. Nileshchandra K. Pikle** (PhD, IIT Bombay), Assistant Professor, CSE Department, IIIT Nagpur.

## Author

Miheer Kulkarni — [github.com/miheer-smk](https://github.com/miheer-smk)
