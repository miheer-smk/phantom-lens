<!--
================================================================
  FILE: README.md — github.com/miheer-smk/phantom-lens
  NOTE: License badge says "Research Only" — DO NOT add an MIT LICENSE
        file without discussing with the supervisor.
================================================================
-->

<h1 align="center"> Phantom Lens</h1>

<p align="center">
  <b>Physics-Anchored Deepfake Detection Framework (PRISM)</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/LightGBM-Classifier-brightgreen" />
  <img src="https://img.shields.io/badge/MediaPipe-Face%20Mesh-orange" />
  <img src="https://img.shields.io/badge/Protocol-Identity--disjoint-blueviolet" />
  <img src="https://img.shields.io/badge/License-Research%20Only-lightgrey" />
</p>

<p align="center">
  <b>Researcher:</b> Miheer Satish Kulkarni — IIIT Nagpur, 2026<br>
  <b>Supervisor:</b> Dr. Nileshchandra K. Pikle, Assistant Professor, CSE, IIIT Nagpur
</p>

---

> ### ⚠️ Results correction notice
> An earlier version of this repository reported **leakage-inflated** numbers (test fakes were present
> in the training set; no identity-disjoint split). Those figures — FF++ **0.9939 / 0.9991 / 0.9999**,
> Celeb-DF **0.6989 / 0.6867**, hard-negative **13/957** — are **RETIRED** and must not be used.
> The current numbers below are **leakage-free** (identity-disjoint official FaceForensics++
> 720/140/140 split, seed 42). The superseded code/results are quarantined in
> [`archive/deprecated_leaky/`](archive/deprecated_leaky/).
>
> **Authoritative numbers & provenance:**
> [`Major Revision Results/07_summary/LOCKED_NUMBERS.md`](Major%20Revision%20Results/07_summary/LOCKED_NUMBERS.md) ·
> [`AUTHOR_HANDOFF.md`](Major%20Revision%20Results/07_summary/AUTHOR_HANDOFF.md) ·
> current tables in [`results_clean/`](results_clean/) · clean commands in [`REPRODUCE.md`](REPRODUCE.md).

---

## About

Phantom Lens is a physics-grounded deepfake detection framework that takes a different approach from
CNN-based detectors. Instead of learning texture artifacts that break when a new generator is released,
it checks whether a video obeys the statistics of real-world physics.

**The core asymmetry:** a generative model must simultaneously replicate dozens of physical
constraints — sensor noise statistics, light transport, physiological signals, lens optics, compression
traces. A detector only needs to catch *one* violation.

The system is built on **PRISM** (Physics-Reality Integrated Signal Multistream), a 50-feature extractor
anchored to MediaPipe facial landmarks (extended to a 53-feature variant with ROI mouth-instability
descriptors). A LightGBM classifier trained on FaceForensics++ gives **strong in-distribution detection
(mean AUC 0.932, 53-D)** and **moderate zero-shot cross-dataset generalisation (Celeb-DF v2 AUC 0.632)**.
Its contribution is interpretability and CPU-efficiency, not raw accuracy over a deep baseline (see Xception below).

---

## Key Results  (leakage-free, identity-disjoint, seed 42)

### In-distribution — FaceForensics++ c23 (per manipulation)
| Manipulation | AUC 50-D | AUC 53-D (+ROI G1) | 95% CI (53-D) |
|---|---|---|---|
| Deepfakes | 0.971 | **0.978** | [0.963, 0.989] |
| FaceSwap | 0.963 | **0.969** | [0.949, 0.984] |
| Face2Face | 0.810 | **0.875** | [0.833, 0.914] |
| NeuralTextures | 0.787 | **0.905** | [0.866, 0.940] |
| **mean** | **0.883** | **0.932** | — |

### Cross-dataset — zero-shot (never seen in training)
| Dataset | AUC | 95% CI |
|---|---|---|
| **Celeb-DF v2** | **0.632** | [0.613, 0.654] |
| WildDeepfake | 0.521 | — |

Cross-dataset degradation is expected; the drop reflects real-class domain shift (see limitations).
Celeb-DF reconciliation of the retired vs honest values:
[`celebdf_reconciliation.md`](Major%20Revision%20Results/07_summary/celebdf_reconciliation.md).

### Deep-learning baseline & other analyses
| Item | Result |
|---|---|
| Xception baseline (same protocol) | FF++ **0.990**, Celeb-DF **0.821** (GPU, 83 MB, not explainable) |
| DeLong PRISM vs Xception (Celeb-DF) | Xception +0.189, z=15.4, p<1e-16 |
| Hard-negative (clean Deepfakes test) | **17/133 false negatives (12.78%)** |
| Runtime (CPU, per video) | 48.6 s, RTF 3.20, peak RAM 3.5 GB, model ~0.6 MB |
| SHAP ranking stability (cross-fold) | Spearman 0.911 |
| Missingness-only classifier (real vs fake) | AUC 0.50 — detection is **not** a missingness artifact |

Full per-experiment numbers with provenance: **`LOCKED_NUMBERS.md`**.

---

## Physics Pillars (summary)

- **P1 — Sensor noise:** signal-dependent noise statistics + PRNU-inspired residual-energy descriptors.
- **P2 — Light transport & geometry:** illumination consistency, specular stability, landmark rigidity, blink dynamics.
- **P3 — Compression forensics:** Benford deviation, block artifacts, DCT temporal stability.
- **P4 — Physiological (rPPG):** POS/CHROM temporal descriptors (forensic cue, not medical-grade pulse).

Feature → pillar mapping: `splits/pillar_map.json`.

---

## Project Structure

```
phantom-lens/
├── src/                              # Core library (protocol.py, leakfree.py, delong.py, roi_config.py, pillars/)
├── features/                         # Extracted PRISM feature CSVs (50-D, ROI, residual, rPPG, c40) — committed
├── splits/                           # ffpp_official_split.json (720/140/140), pillar_map.json
├── results_clean/                    # ★ CURRENT leakage-free result tables (JSON/CSV)
├── REPRODUCE.md                      # ★ Clean regeneration commands (fresh-checkout verified)
├── Major Revision Results/
│   ├── 00_logs/                      # All experiment scripts (baseline_clean, exp_m4_missingness, exp_g1_hardneg, …)
│   ├── 02_tables/                    # Consolidated result tables
│   ├── 03_figures/                   # Publication figures
│   ├── 05_zenodo_package/            # Reproducibility package (scripts+splits+features+results+README)
│   └── 07_summary/                   # ★ LOCKED_NUMBERS.md, AUTHOR_HANDOFF.md, author_decisions.md,
│                                     #   celebdf_reconciliation.md, response_fill_sheet.{md,json}
└── archive/
    └── deprecated_leaky/             # ⛔ RETIRED leakage-inflated results/scripts — DO NOT USE
```

---

## Installation

```bash
git clone https://github.com/miheer-smk/phantom-lens.git
cd phantom-lens
python -m venv .venv && . .venv/bin/activate
pip install -r "Major Revision Results/05_zenodo_package/requirements.txt"
```
Python 3.9+, FFmpeg system-wide. GPU only needed for the Xception baseline and crop-based steps.

---

## Reproducing the current (honest) results

Every table regenerates deterministically (seed 42) from the committed feature CSVs in `features/` —
**no raw videos or GPU needed**. Run from the repo root. Full command list: [`REPRODUCE.md`](REPRODUCE.md).

```bash
L="Major Revision Results/00_logs"
python "$L/baseline_clean.py"      # -> results_clean/baseline.json        (FF++ 50-D + Celeb-DF)
python "$L/track_c_measure.py"     # -> track_c_53D_full.json              (53-D +ROI)
python "$L/run_delong.py"          # -> delong*.csv/json                   (DeLong significance)
python "$L/exp_m4_missingness.py"  # -> missingness_audit.json             (missingness-as-signal)
python "$L/exp_g1_hardneg.py"      # -> hardneg_deepfakes.json             (clean hard negatives)
python "$L/exp_g9_predictions.py"  # -> predictions_per_video.csv          (per-video predictions)
```
Steps needing raw data/GPU (set `FFPP_ROOT` / `WILDDEEPFAKE_ROOT` / `DATASETS_ROOT`): `exp5_runtime.py`,
`xception_prep.py`+`xception_train.py`, `exp_g9_xception_predictions.py`, `pub_figures.py`.
Fresh-checkout reproduction of all CPU tables has been verified bit-identical.

---

## Superseded results

The original, **leakage-inflated** experiment tree (`exp1`, `exp2`, `exp3`, `exp5`, `exp_celebdf`,
per-/cross-manipulation dirs, old reports) and the recon-phase gate scripts have been moved to
[`archive/deprecated_leaky/`](archive/deprecated_leaky/) with an explanatory README. They produced the
retired numbers (0.9939 / 0.9991 / 0.9999 / 0.6989 / 0.6867 / 13-957) and **must not be run or cited**.
The correction is a data-leakage fix: the earlier pipeline trained and tested on the same manipulation
CSVs without identity-disjoint splitting.

---

## Limitations & Future Work

- **Cross-dataset real-class boundary:** Celeb-DF real videos are disproportionately flagged at θ=0.50
  (domain shift, FF++ c23 vs YouTube-compressed reals). Threshold calibration does **not** fix this
  (AUC-invariant; see EXP-4). A mild class-dependent extraction gap is disclosed in the missingness audit.
- **Domain adaptation:** under author decision (no reproducing script yet — see `author_decisions.md`).
- **Temporal window:** features over ≤150 frames (~6 s); longer-sequence modelling is future work.

---

## Citation

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

**Miheer Satish Kulkarni** — B.Tech CSE, IIIT Nagpur — [github.com/miheer-smk](https://github.com/miheer-smk)

*Supervised by Dr. Nileshchandra K. Pikle — Assistant Professor, CSE Department, IIIT Nagpur.*
