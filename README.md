<!--
  README.md — github.com/miheer-smk/phantom-lens  (branch: outstanding-results)
  License badge says "Research Only" — do not add an MIT LICENSE without the supervisor.
-->

<h1 align="center">Phantom Lens (PRISM)</h1>

<p align="center"><b>Physics-Anchored Deepfake Detection — 196-D representation & sealed cross-dataset evaluation</b></p>

<p align="center">
  <img src="https://img.shields.io/badge/Protocol-Identity--disjoint-blueviolet" />
  <img src="https://img.shields.io/badge/Evaluation-Sealed%20%C2%B7%20Pre--registered-success" />
  <img src="https://img.shields.io/badge/LightGBM%20%2B%20RF%20%2B%20ExtraTrees-Rank%20ensemble-brightgreen" />
  <img src="https://img.shields.io/badge/License-Research%20Only-lightgrey" />
</p>

<p align="center">
  <b>Researcher:</b> Miheer Satish Kulkarni — IIIT Nagpur, 2026 ·
  <b>Supervisor:</b> Dr. Nileshchandra K. Pikle, CSE, IIIT Nagpur
</p>

---

## Headline result — sealed Celeb-DF-v2 cross-dataset test

**Celeb-DF-v2 sealed test AUC = 0.713 [0.687, 0.746].** Zero-shot (no Celeb-DF data in training), identity-disjoint,
a **single pre-registered evaluation** of a **frozen** model: a 196-D order-statistic physics representation with a
RandomForest + ExtraTrees + LightGBM rank-averaged ensemble, trained on FaceForensics++ only (seed 42).

### Like-for-like on the same sealed half (identical 2,273 test videos)
| representation | AUC | 95% CI | paired DeLong vs 196-D |
|---|---|---|---|
| 50-D | 0.6573 | [0.630, 0.685] | p = 1.7×10⁻⁶ |
| 53-D | 0.6830 | [0.656, 0.718] | p = 2.2×10⁻⁹ |
| **196-D** | **0.7133** | [0.687, 0.746] | — |

DeLong is a **paired** test on the shared test set; it accounts for the correlation between the two AUC estimates,
which is why the differences are significant even though the marginal CIs overlap.

### In-distribution — FaceForensics++ test (same frozen model)
Pooled AUC **0.842 [0.810, 0.872]**; per-manipulation: Deepfakes 0.907 · Face2Face 0.796 · FaceSwap 0.831 ·
NeuralTextures 0.833 (mean-of-4 0.842). This frozen model is cross-dataset-optimised; its in-distribution number is
lower than the paper's dedicated 53-D in-distribution model on `major-revision` (mean-of-4 0.932) and is reported
separately — the two are not the same configuration.

### Protocol disclosure
57 dev-set evaluations preceded the sealed test; the sealed budget was **1, spent once**. The pre-registered
prediction (written before unsealing) was **0.68, 80% interval [0.65, 0.71]**; the actual sealed result was
**0.7133** (above the interval — the prediction was conservative).

### Negative-results summary (57-eval programme)
A **distributional (order-statistic) representation** of the per-frame physics features is the only lever that
improves cross-dataset AUC (+0.033 dev, Holm-significant). **Feature addition** (8 families), **unsupervised domain
adaptation** (CORAL, subspace/quantile alignment, per-domain standardisation, self-training), **training-distribution
augmentation** (self-blended video), and **diverse-data augmentation on both the real and fake classes** (DFD
originals and fakes) each failed to improve cross-dataset transfer. Full table: `196D_FINAL/04_tables/`.

### Caveats (read before quoting any number)
- The 95% CI lower bound is **0.687**, so the interval includes values below 0.70. Report as **0.713 [0.687, 0.746]** —
  **never** as "above 0.70".
- The sealed half is a **custom identity-disjoint split, not Celeb-DF-v2's official test protocol**; these numbers are
  **not directly comparable** to published Celeb-DF figures.
- The sealed half is **measurably easier** than the full set (the 50-D baseline scores **0.657 here vs 0.632 on full
  Celeb-DF**). Do **not** present any "0.632 → 0.713" comparison. The valid, like-for-like figures are the sealed-test
  deltas: **+0.030 over 53-D** and **+0.056 over 50-D**.
- No full-Celeb-DF number is reported for the 196-D model, because the full set contains the dev half used for selection.
- The single sealed evaluation appears twice in the audit log — a crash-and-recapture of one deterministic evaluation
  on the pre-committed frozen model (`00_protocol/SEALED_PROVENANCE.md`).
- Real-class recall on Celeb-DF stays low (0.183 at θ=0.5); AUC is threshold-free and thresholding does not fix it.

### Where the full record lives
- [`196D_FINAL/`](196D_FINAL/) — consolidated deliverables (inventory, tables T1–T5, figures, results draft).
- [`196D_FINAL/LOCKED_NUMBERS_196D.md`](196D_FINAL/LOCKED_NUMBERS_196D.md) — every reported number with full provenance (script · commit · seed · feature-CSV SHA-256).
- [`196D_FINAL/00_protocol/trackE_FREEZE.md`](196D_FINAL/00_protocol/trackE_FREEZE.md) — the freeze document (frozen config + a-priori selection rule).
- [`196D_FINAL/00_protocol/SEALED_PROVENANCE.md`](196D_FINAL/00_protocol/SEALED_PROVENANCE.md) — sealed-evaluation record.
- [`BRANCHES.md`](BRANCHES.md) — what each branch contains and which is authoritative.

---

> ### ⚠️ Retired numbers (do not cite)
> An earlier version of this repository reported **leakage-inflated** figures (test fakes present in training; no
> identity-disjoint split): FF++ **0.9939 / 0.9991 / 0.9999**, Celeb-DF **0.6989 / 0.6867**, hard-negative **13/957**.
> These are **RETIRED**. The superseded code/results are quarantined in
> [`archive/deprecated_leaky/`](archive/deprecated_leaky/). See `main`'s README for the visitor-facing correction.

---

## About
Phantom Lens is a physics-grounded deepfake detector: instead of learning generator-specific texture artifacts, it
tests whether a video obeys the statistics of real-world physics (a generator must replicate many physical
constraints; a detector needs to catch one violation). PRISM (Physics-Reality Integrated Signal Multistream) extracts
landmark-anchored physics features; the 196-D representation summarises each per-frame spatial feature by 11 order
statistics (mean/std/min/max/percentiles/IQR/skew/kurtosis), plus 37 temporal and 3 mouth-ROI features.

## Physics pillars
- **P1 — Sensor noise:** signal-dependent noise statistics + PRNU-inspired residual energy.
- **P2 — Light transport & geometry:** illumination consistency, specular stability, landmark rigidity, blink dynamics.
- **P3 — Compression forensics:** Benford deviation, block artifacts, DCT temporal stability.
- **P4 — Physiological (rPPG):** POS/CHROM temporal descriptors (forensic cue, not medical-grade pulse).

## Reproduce the reported numbers (CPU, seed 42, from committed features)
```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements_snapshot.txt
L="Major Revision Results/00_logs"
python "$L/exp_trackE_postfreeze_compare.py"   # T1: 50/53/196-D on sealed test + paired DeLong
python "$L/exp_trackE_permanip.py"             # T2: FF++ per-manipulation
python 196D_FINAL/01_scripts/make_figures.py   # ROC / trade-off / negative-results figures
```
The sealed evaluation (`exp_trackE_SEALED_eval.py --unseal`) is already spent (budget 1/1); its result is in
`196D_FINAL/03_results/SEALED_final.json` and must not be re-run for new numbers. Fresh-checkout reproduction of
T1 has been verified bit-identical.

## Citation
```bibtex
@misc{kulkarni2026phantomlens,
  author = {Kulkarni, Miheer Satish},
  title  = {Phantom Lens: Physics-Anchored Deepfake Detection Framework (PRISM)},
  year   = {2026},
  institution = {Indian Institute of Information Technology, Nagpur},
  note   = {Manuscript under revision}
}
```

**Author:** Miheer Satish Kulkarni — B.Tech CSE, IIIT Nagpur · Supervised by Dr. Nileshchandra K. Pikle.
