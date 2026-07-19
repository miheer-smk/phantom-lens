# AUTHOR HANDOFF — PRISM / PhantomLens, Scientific Reports Major Revision
**Facts, tables, and decisions for the authors (Miheer + co-author). Not drafted prose.**
Framing/claims paragraphs are deliberately left to the authors. Generated 2026-07-19.

All numbers below are: identity-disjoint (official FF++ 720/140/140 split), leakage-free,
seed 42, computed by committed deterministic scripts. Full per-value provenance in
`LOCKED_NUMBERS.md`. Retired leaky numbers (0.9939 / 0.9991 / 0.9999 / 0.6989) are NOT reported.

---

## 1. THE HONEST HEADLINE NUMBERS

### In-distribution, per manipulation (identity-disjoint test, bootstrap 95% CI)
| Manipulation | 50-D | 53-D (final, +G1 mouth) |
|---|---|---|
| Deepfakes | 0.971 | **0.978** |
| FaceSwap | 0.963 | **0.969** |
| Face2Face | 0.810 | **0.875** |
| NeuralTextures | 0.787 | **0.905** |
| mean per-manip | 0.883 | **0.932** |

### Cross-manipulation (leave-one-manipulation-out)
Deepfakes 0.704 · Face2Face 0.690 · FaceSwap 0.598 · NeuralTextures 0.522

### Cross-dataset zero-shot
Celeb-DF v2 **0.632** [0.613,0.654] · WildDeepfake **0.521** (face-crop caveat; real-recall 0.11)

### Deep-learning baseline (Xception, same identity-disjoint protocol)
FF++ 0.990 · Celeb-DF **0.821** · GPU / 83 MB / not explainable.
DeLong PRISM vs Xception on Celeb-DF: Xception +0.189, z=15.4, p<1e-16 (Xception significantly better).

---

## 2. EVERY EXPERIMENT (13) — headline + reviewer + file/commit

| # | Experiment | Answers | Headline result | File / commit |
|---|---|---|---|---|
| 1 | Group ablation (remove-one) | R1,R3,R5 | no single pillar sig. after Holm (redundancy) | pillar_ablation.csv / 9b7eaa1 |
| 1b| Pillar-only (standalone) | R1,R3,R5 | P1 noise 0.73–0.86; T7 rigid-geom 0.82(DF) | pillar_only.csv / a347b04 |
| 2 | SHAP stability | R5 | cross-fold 0.911; cross-manip 0.07–0.36 | shap_stability.json / c9afcd9 |
| 3 | Compression c23/c40 (all 4) | R5.4 | Δ −0.03…−0.12 (FS most); c40 hits HF/sensor pillars | compression.json / (EXP-3) |
| 4 | Threshold calibration | R1,R5.3 | NULL: only trades real/fake recall (AUC fixed) | calibration.csv / 81c6067 |
| 5 | Runtime/memory | R2,R3.6 | 50.2 s/vid CPU, RTF 3.34, 3.4 GB, model ~0.6 MB | runtime.json / d6ec61f |
| 6 | WildDeepfake zero-shot | R3.7,R5.1 | AUC 0.52; real-recall collapse GENERALIZES | zeroshot_wilddeepfake.json / 98d5d27 |
| 6b| DFDC | R3.7,R5.1 | BLOCKED — no valid labels (all-zeros template only) | — |
| 7 | Xception baseline | R5.2,R3.4 | FF++ 0.990 / CelebDF 0.821 (beats PRISM both) | xception_baseline.json / a0a457b |
| 8 | PRNU residual comparison | R1 | median≈gaussian≈wavelet (0.889–0.894); BM3D NOT COMPUTED | prnu_comparison.csv / 33550c9 |
| 9 | rPPG POS/CHROM | R1 | rPPG weak (~chance); POS best+compression-robust | rppg_comparison.csv / d9105fd |
| 10| Case-level SHAP | R4,R5.6 | 4 cases (TP/TN/FN/FP) waterfalls + signals | case_shap.json / ce496db |
| 11| Statistical tests | R3.13 | full-50>top-k (p_holm<0.001); calibration hurts; Xcep>PRISM | statistical_tests.csv / 0b46c06 |
| 12| Feature redundancy | R3.12 | 2/1225 pairs |r|>0.9; dedup negligible | redundancy_pairs.csv / 92a4235 |
| Track C | ROI mouth features (G1) | new | F2F 0.81→0.875, NT 0.79→0.905 (DeLong p≤9e-4) | track_c_53D_full.json / de15506 |
| — | Zenodo package | editor | staged (upload=author action) | 05_zenodo_package/ |

---

## 3. MANUSCRIPT CORRECTIONS / AUTHOR-DECISION ITEMS (flagged, NOT drafted)

1. **In-distribution headline:** retire "0.9939" (leaky). Report per-manip 50-D mean 0.883 / 53-D 0.932. Retire 0.9991/0.9999. [B1]
2. **Cross-dataset:** "0.6989 / strong" → "0.632 / moderate zero-shot with real-class domain mismatch"; WildDeepfake corroborates. [B2]
3. **NT top-3 features:** "universally dominant" → "manipulation-specific, fold-stable"; corrected clean top-3. (rewrite item 08) [B3]
4. **Xception repositioning:** contribution = interpretability + CPU-efficiency, NOT accuracy/generalization superiority. **[TEXT HELD for co-author]** [B4]
5. **Leakage disclosure (Methods):** identity-disjoint re-run; assertion enforced. (PROTOCOL.md) [B5]
6. **Threshold/limitations:** "calibration will improve real recall" is UNSUPPORTED (EXP-4 null + McNemar). Real-recall collapse = domain shift.
7. **PRNU terminology:** "PRNU" → "PRNU-inspired residual-energy descriptors" / "sensor-residual consistency" (no reference sensor pattern estimated).
8. **rPPG:** call it a forensic temporal descriptor, NOT medical-grade pulse; weak standalone (complementary only); consider pure POS.
9. **Runtime/efficiency:** honest efficiency = tiny model + CPU-only (NOT "fewer features → faster"; extraction is stage-shared).
10. **Also pending from plan (not yet actioned):** "Without Deep Learning" / MediaPipe clarification; blink-claim moderation; title. [author edits]

Rewrite drafts prepared: `04_manuscript_rewrite/PROTOCOL.md`, `08_shap_nt_feature_claim_correction.md`.

---

## 4. REPRODUCIBILITY & SAFETY

- Every number regenerable from a committed script (seed 42). Git HEAD at handoff: see `git rev-parse HEAD`.
- **Backups:** 2 on-disk copies (`/home/iiitn/phantom_lens_revision_backups/` + `/home/iiitn/Datasets/_phantomlens_backups/`),
  timestamped, gzip-verified, md5-identical, include full git history + all feature CSVs + all results/figures.
  ⚠️ Both on nvme0n1 — an OFFSITE copy (14–19 MB tarball) is still recommended (author action).
- Zenodo staging: `05_zenodo_package/` (scripts, split, results, requirements, env, README, dataset-access). Upload = author action.

## 5. WHAT REMAINS
- **DFDC** (EXP-6): blocked pending a real DFDC test-label file (needs both classes; the found `sample_submission.csv` was an all-zeros template). If unavailable, report Celeb-DF + WildDeepfake, DFDC as future work.
- **Author framing decisions** (§3, esp. item 4) — the contribution/positioning paragraphs.
- **Point-by-point response letter** — map §2 experiments + §3 corrections to each reviewer comment (author task).
