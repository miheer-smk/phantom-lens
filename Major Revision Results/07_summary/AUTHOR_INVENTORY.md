# Revision Inventory — RAW FACTS FOR AUTHORS TO DECIDE
*(Not drafted prose. Facts + deltas only. Framing decisions are the authors' — Miheer + co-author.)*

Generated 2026-07-18. Git HEAD `a0a457b`. All numbers identity-disjoint, leakage-free,
bootstrap 95% CI where shown, seed 42, full provenance in `results_clean/*.json`.

---

## (A) EXPERIMENTS DONE — headline numbers

| # | Experiment | Headline result | File | Commit |
|---|---|---|---|---|
| 1 | Clean baseline (50-D, identity-disjoint) — **in-distribution per manip** | DF 0.971 · FS 0.963 · F2F 0.810 · NT 0.787 | baseline.json | 96a0c85 |
| 1b | Baseline — **cross-manip leave-one-out** | DF 0.704 · F2F 0.690 · FS 0.598 · NT 0.522 | baseline.json | 96a0c85 |
| 1c | Baseline — **zero-shot Celeb-DF v2** | AUC 0.632 [0.613,0.654] | baseline.json | 96a0c85 |
| 2 | Track C — **ROI mouth features (G1)**, 53-D extended | F2F 0.810→**0.875** · NT 0.787→**0.905** (DF/FS +~0.006) | track_c_53D_full.json | de15506 |
| 3 | Per-pillar ablation — **remove-one** (20 pillars × 5 sets) | no single pillar significant after Holm (redundancy) | pillar_ablation.csv | 9b7eaa1 |
| 3b | Per-pillar ablation — **pillar-only standalone** | P1 noise 0.73–0.86 · T7 rigid-geom 0.82 (DF) — each pillar has power | pillar_only.csv | a347b04 |
| 4 | DeLong — **53-D vs 50-D significance** | NT p=1e-6 · F2F p=9e-4 (sig) · DF/FS n.s. | delong_53vs50.csv | 810673e |
| 4b | DeLong — **per-pillar Holm** | CelebDF: P6 compression helps (sig), T2 rppg HURTS (sig) | delong_pillars.csv | 810673e |
| 5 | Zero-shot **WildDeepfake** (53-D) | AUC 0.521 [0.41,0.63]; real-rec 0.11 / fake-rec 0.91 | zeroshot_wilddeepfake.json | 98d5d27 |
| 6 | **SHAP stability** | cross-fold Spearman 0.911; cross-manip 0.07–0.36 | shap_stability.json | c9afcd9 |
| 7 | **Xception baseline** (fair DL, same protocol) | FF++ 0.990 · CelebDF zero-shot **0.821** | xception_baseline.json | a0a457b |
| — | DFDC | ON HOLD — no verified mixed-label file; candidate was all-zeros template | — | — |

---

## (B) MANUSCRIPT CORRECTIONS the clean pipeline surfaced (deltas vs current manuscript)

### B1 — In-distribution headline: 0.9939 → honest numbers
- **Was:** "in-distribution AUC 0.9939" (from leaky multi-manip protocol; test fakes in training).
- **Now (clean, identity-disjoint):** per-manip 50-D DF 0.971 / FS 0.963 / F2F 0.810 / NT 0.787;
  53-D (with G1) F2F 0.875 / NT 0.905; mean per-manip 0.932.
- **Delta:** −0.06 on the single headline; retired 0.9991/0.9999 per-manip (leakage artifacts).
- Evidence: PROTOCOL.md, repro_gap_report.md, baseline.json, track_c_53D_full.json.

### B2 — Cross-dataset: 0.6989 → 0.632 (and reframed)
- **Was:** Celeb-DF "0.6989", framed near "strong cross-dataset."
- **Now:** 0.632 [0.613,0.654] zero-shot (0.6989 not reproducible; honest reproduced value).
  Reframe to "moderate zero-shot transfer with real-class domain mismatch."
- Second dataset (WildDeepfake) confirms the real-class mismatch pattern (real-rec 0.11).
- **Delta:** −0.067; framing "strong" → "moderate, with real-class mismatch."

### B3 — NeuralTextures top-3 feature claim (rewrite item 08)
- **Was:** NT top-3 (t_noise_spectral_entropy, t_coupling_consistency, t_nose_bridge_std) "dominant."
- **Now:** feature importance is **manipulation-specific and fold-stable, not universally dominant**;
  clean NT top-3 = s_noise_hf_ratio, t_face_ssim_mean, t_skin_texture_corr.
- Evidence: shap_stability.json (cross-fold 0.911, cross-manip 0.07–0.36). File: 04_manuscript_rewrite/08_*.md.

### B4 — Xception repositioning (RAW FACTS ONLY — authors to write the paragraph)
- **Fact:** fair Xception (same identity-disjoint protocol) beats PRISM in-distribution
  (0.990 vs 0.932 mean) AND cross-dataset (0.821 vs 0.632).
- **Implication (authors decide):** contribution repositions to INTERPRETABILITY + CPU-EFFICIENCY,
  not accuracy/generalization superiority. Xception real-rec 0.82 (high) → real-class mismatch is
  PRISM-specific, not universal. **Text HELD pending co-author.** File: xception_baseline.json.

### B5 — Leakage disclosure (methods)
- **Fact:** original per-manip protocol placed all test fakes in training (results/exp1/run_exp1.py).
  All results re-run under identity-disjoint official FF++ split (720/140/140) with overlap assertion.
- Evidence: PROTOCOL.md, src/protocol.py, protocol_matrix.csv (identity_overlap=0 asserted).

---

## Provenance / safety
- Every number regenerable from committed script (seed 42). Backups: 2 on-disk copies verified
  (phantomlens_revision_20260718_133723.tar.gz, 15M) + git history (25+ commits). Offsite copy
  still recommended (both on nvme0n1).
