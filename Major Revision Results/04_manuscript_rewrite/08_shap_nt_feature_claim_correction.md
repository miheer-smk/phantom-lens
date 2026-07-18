# Rewrite Item 08 — SHAP / NeuralTextures feature-importance claim correction

**Trigger:** Reviewer 5's caution — "do not claim the NeuralTextures top-three features are
universally dominant unless the rankings are reasonably stable." Confirmed by the clean,
identity-disjoint SHAP stability analysis (`results_clean/shap_stability.json`, commit c9afcd9).

**Evidence (clean, leakage-free):**
- Cross-fold SHAP ranking stability: mean Spearman = **0.911** (10 identity-grouped folds) → rankings are reproducible WITHIN a distribution.
- Cross-manipulation SHAP Spearman = **0.07–0.36**; top-5 overlap **0–2 / 5** → rankings DIVERGE across manipulations; feature importance is manipulation-specific.
- The clean NeuralTextures top-3 (`s_noise_hf_ratio`, `t_face_ssim_mean`, `t_skin_texture_corr`)
  differ from the previously reported top-3, which came from the retired leaky ablation
  (`results/exp3/feature_ranking.csv`). Those NT top-3 are not dominant elsewhere
  (cross-manipulation rank SD 8.8–17.3).

---

## BEFORE (to be removed / retired)
> "The NeuralTextures top-three features — t_noise_spectral_entropy, t_coupling_consistency,
> and t_nose_bridge_std — are the dominant discriminative cues [implying universal importance]."
(Source: leaky exp3 SHAP ranking; not reproducible under identity-disjoint evaluation.)

## AFTER (drop-in replacement)
> "Feature importance is **manipulation-specific and fold-stable, not universally dominant**.
> SHAP rankings are highly reproducible across cross-validation folds (mean Spearman 0.911),
> confirming the analysis is stable within a given distribution. However, rankings diverge
> sharply across manipulation types (cross-manipulation Spearman 0.07–0.36; top-5 overlap
> 0–2 of 5), indicating that each manipulation is detected via a different subset of physical
> cues rather than a single universal feature set. For NeuralTextures specifically, the most
> important cues under leakage-free evaluation are the spatial high-frequency noise ratio
> (`s_noise_hf_ratio`), temporal face structural stability (`t_face_ssim_mean`), and skin
> texture temporal coherence (`t_skin_texture_corr`); these are not the dominant features for
> other manipulations."

## Why this is a strength, not a retraction
The corrected claim is more defensible and more interesting: it demonstrates that the
physics-grounded feature bank provides **complementary, attack-specific cues** — a redundancy
already independently supported by the per-pillar ablation (no single pillar significant after
Holm-correction) and the pillar-only standalone-power analysis. It also directly answers
Reviewer 5's stability concern with computed evidence.

## Cross-references
- `results_clean/shap_stability.json`, `results_clean/shap_ranking_by_manip.csv`
- Pillar ablation (redundancy): `results_clean/pillar_ablation.csv` + `pillar_only.csv`
- PROTOCOL.md (identity-disjoint, leakage-free evaluation)
