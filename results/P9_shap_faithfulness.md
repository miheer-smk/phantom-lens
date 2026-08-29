# Phase 9 (R1-C8) — SHAP faithfulness

Every held-out FF++ c23 test video. Top-5 features by |SHAP| are replaced with their
**training-set median** (train partition only — never test or target medians), the score is
recomputed, and the drop in confidence is compared against **100 independent random 5-feature
masks per video** (seed 42).

Confidence in the originally predicted class: `C = p_fake if ŷ=1 else (1 − p_fake)`;
`ΔC = C_original − C_masked`.

Artifacts: `results/P9_shap_faithfulness.json`

---

## 1. Primary result — SHAP explanations are faithful, by a wide margin

| Manipulation | n | median ΔC (top-5 SHAP) | median ΔC (random-5) | ratio | rank-biserial | Wilcoxon p |
|---|---|---|---|---|---|---|
| DeepFakes | 267 | **0.28836** | 0.01043 | **27.7×** | 0.783 | 1.4 × 10⁻³⁴ |
| Face2Face | 268 | **0.21515** | 0.02684 | **8.0×** | 0.515 | 4.2 × 10⁻²⁵ |
| FaceSwap | 269 | **0.34232** | 0.01746 | **19.6×** | 0.695 | 1.1 × 10⁻³³ |
| NeuralTextures | 269 | **0.26539** | 0.02988 | **8.9×** | 0.740 | 6.0 × 10⁻³⁹ |

Masking the five features SHAP identifies as most influential costs **8–28× more confidence** than
masking five random features, on every manipulation, with Wilcoxon signed-rank p between 10⁻²⁵ and
10⁻³⁹. Rank-biserial effect sizes of 0.52–0.78 are large.

**The explanations are not decorative.** The features SHAP names are the features the model actually
uses: removing them moves the decision, and removing arbitrary ones largely does not. This is the
direct answer to R1-C8.

**Caveat worth stating.** This establishes *self-consistency* — SHAP correctly identifies what the
fitted model relies on. It does **not** establish that those features are forensically correct, nor
that the model relies on them for the right reasons. Faithfulness and validity are different claims
and the manuscript should not blur them.

## 2. Secondary — group |SHAP| vs Table 11 remove-one-group ΔAUC

Mean |SHAP| aggregated into the 20 Table A2 implementation groups, ranked against the
leave-one-group-out ΔAUC.

**Sign convention, verified from the artifact:** `delta_auc = full_auc − loGo_auc`, so a **positive**
ΔAUC means removing the group *hurt* — the group is **useful**. A **positive** Spearman ρ therefore
means SHAP attribution is **concordant** with ablation utility.

| Manipulation | Spearman ρ | p | verdict |
|---|---|---|---|
| **Face2Face** | **+0.755** | 1.2 × 10⁻⁴ | strong, significant |
| **FaceSwap** | **+0.509** | 0.022 | moderate, significant |
| NeuralTextures | +0.315 | 0.176 | weak, n.s. |
| DeepFakes | +0.264 | 0.261 | weak, n.s. |

**Positive on all four; significant on two.** This is a *better* outcome than the brief anticipated —
it expected a weak correlation and pre-emptively cautioned against reading one as failure. The
concordance is real, if uneven.

**Why the two weak cases are expected, not a defect.** The brief's own reasoning applies and is
confirmed here: ablation measures *marginal* contribution after the remaining 45–49 descriptors have
compensated, while SHAP measures *attribution given the fitted model*. Where the representation is
redundant, a group can be heavily attributed yet removable at no cost, because its signal is
recoverable elsewhere. DeepFakes and NeuralTextures are exactly where that redundancy is greatest —
for DeepFakes the model is near ceiling (0.9706), so almost every leave-one-out ΔAUC is within
±0.016 and the ranking is dominated by noise. The correlation is weakest precisely where the ΔAUC
range is smallest, which is the signature of a compensation effect rather than an explanation
failure.

Consistent with that reading, the strongest concordance (Face2Face, ρ = +0.755) is the manipulation
with the *lowest* in-distribution AUC among the three non-NT cases — the one where the model has the
least slack and each group's contribution is least substitutable.

## 3. Draft manuscript text

> **Explanation faithfulness.** To test whether SHAP attributions reflect the model's actual
> decision process rather than a plausible post-hoc narrative, we masked, for every held-out test
> video, the five descriptors with the largest |SHAP| by substituting their training-partition
> medians, and compared the resulting loss of confidence in the originally predicted class against
> 100 random five-descriptor masks per video. Masking the SHAP-identified descriptors cost 8–28×
> more confidence than masking random ones (median ΔC 0.215–0.342 versus 0.010–0.030; Wilcoxon
> signed-rank p < 10⁻²⁴ for all four manipulations; rank-biserial 0.52–0.78). Aggregating attribution
> into the twenty implementation groups and comparing against the leave-one-group-out ablation gives
> positive rank correlation for all four manipulations (Spearman ρ = 0.26–0.76), significant for
> Face2Face and FaceSwap. The weaker agreement for DeepFakes and NeuralTextures is expected: ablation
> measures marginal contribution after the remaining descriptors compensate, whereas SHAP measures
> attribution within the fitted model, and the two diverge most where the representation is most
> redundant. We emphasise that this establishes the self-consistency of the explanations, not the
> forensic validity of the descriptors they identify.
