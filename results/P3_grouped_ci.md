# Phase 3 (R1-C7) — identity-grouped cluster bootstrap

The i.i.d. bootstrap is replaced by a cluster bootstrap over source identities. Duplicated groups
**replicate their rows**; they are never deduplicated by a boolean mask. Resamples missing a class
are skipped and counted. 2000 resamples, seed 42.

Artifacts: `results/P3_grouped_ci.json` · script `scripts/p3_grouped_bootstrap.py`

---

## 1. Point estimates are unchanged — verified

Every point AUC is bit-identical to the published value: 0.9706 / 0.8096 / 0.9631 / 0.7867 (Table 7),
0.7039 / 0.6904 / 0.5978 / 0.5221 (Table 8), 0.6322 (Table 13), 0.5212 (Table 15), 0.8207 (Xception
Celeb-DF). The bootstrap affects intervals only, as required.

## 2. Results

| Row | n | groups | AUC | i.i.d. CI (width) | grouped CI (width) | ratio | skipped |
|---|---|---|---|---|---|---|---|
| Table 7 DeepFakes | 267 | 137 | 0.9706 | [0.9539, 0.9845] (0.0306) | [0.9532, 0.9851] (0.0318) | **1.039** | 0 |
| Table 7 Face2Face | 268 | 137 | 0.8096 | [0.7576, 0.8580] (0.1004) | [0.7598, 0.8571] (0.0972) | 0.968 | 0 |
| Table 7 FaceSwap | 269 | 137 | 0.9631 | [0.9416, 0.9799] (0.0384) | [0.9418, 0.9807] (0.0389) | **1.015** | 0 |
| Table 7 NeuralTextures | 269 | 137 | 0.7867 | [0.7306, 0.8372] (0.1067) | [0.7353, 0.8357] (0.1004) | 0.942 | 0 |
| Table 8 LOMO DeepFakes | 267 | 137 | 0.7039 | [0.6440, 0.7653] (0.1213) | [0.6494, 0.7573] (0.1078) | 0.889 | 0 |
| Table 8 LOMO Face2Face | 268 | 137 | 0.6904 | [0.6243, 0.7545] (0.1302) | [0.6438, 0.7420] (0.0982) | 0.754 | 0 |
| Table 8 LOMO FaceSwap | 269 | 137 | 0.5978 | [0.5310, 0.6659] (0.1349) | [0.5352, 0.6642] (0.1290) | 0.957 | 0 |
| Table 8 LOMO NeuralTextures | 269 | 137 | 0.5221 | [0.4554, 0.5919] (0.1365) | [0.4657, 0.5803] (0.1145) | 0.839 | 0 |
| **Table 13 Celeb-DF v2** | 6121 | 60 | 0.6322 | [0.6125, 0.6537] (0.0412) | **[0.6080, 0.6538] (0.0458)** | **1.113** | 0 |
| Table 19 Xception FF++ | 684 | 137 | 0.9898 | [0.9827, 0.9955] (0.0128) | [0.9810, 0.9966] (0.0156) | **1.220** | 0 |
| Table 19 Xception Celeb-DF | 6487 | 350 | 0.8207 | [0.8058, 0.8342] (0.0284) | [0.7886, 0.8473] (0.0587) | **2.069** | 0 |
| Table 15 WildDeepfake | 124 | **2** | 0.5212 | [0.4139, 0.6297] | **degenerate — not reported** | — | **498** |

## 3. Six rows are NARROWER — investigated, and it is a real effect with a known cause

The brief said to investigate if grouped intervals came out narrower. Six did. **It is not a bug and
not Monte Carlo noise.**

**Stability check.** Re-running the ratio over 8 independent seeds:

| Row | mean ratio | SD | range | verdict |
|---|---|---|---|---|
| Table 8 LOMO Face2Face | 0.784 | 0.026 | [0.742, 0.809] | consistently < 1 |
| Table 7 NeuralTextures | 0.906 | 0.035 | [0.863, 0.967] | consistently < 1 |
| Table 7 DeepFakes | 1.062 | 0.033 | [1.020, 1.115] | consistently > 1 |

**Cause: the FF++ grouping is not a conventional same-class cluster — it is a matched real–fake
pair.** An FF++ group is a source identity, and a manipulated clip `A_B` carries `source_id = A`, so
each group holds the pristine video *and the fakes derived from it*. Within-group real-vs-fake score
correlation is **positive in all eight cases**:

| Row | within-group real–fake r | p | width ratio |
|---|---|---|---|
| Table 7 DeepFakes | +0.119 | 0.336 | 1.039 |
| Table 7 FaceSwap | +0.268 | 0.029 | 1.015 |
| Table 8 LOMO FaceSwap | +0.285 | 0.019 | 0.957 |
| Table 7 Face2Face | +0.326 | 0.007 | 0.968 |
| Table 7 NeuralTextures | +0.420 | 0.0004 | 0.942 |
| Table 8 LOMO DeepFakes | +0.466 | 7.1e-05 | 0.889 |
| Table 8 LOMO NeuralTextures | +0.480 | 3.9e-05 | 0.839 |
| Table 8 LOMO Face2Face | +0.503 | 1.4e-05 | 0.754 |

**corr(within-group correlation, width ratio) = −0.883 (Pearson, p = 0.0037), −0.976 (Spearman, p = 3.3 × 10⁻⁵).** The stronger the
real–fake pairing inside a group, the narrower the grouped interval — a textbook matched-pairs
variance reduction. Resampling a group carries its real and its fakes together, preserving their
relative ranking, and AUC is a purely rank-based between-class statistic.

**This does not mean the grouped interval is wrong — it means the i.i.d. interval was too wide for
FF++.** The i.i.d. bootstrap breaks the real–fake pairing that the identity-disjoint design creates,
and so overstates uncertainty. The grouped interval respects the dependency structure and is the one
to report, in both directions.

**Celeb-DF behaves the opposite way and confirms the account.** 58 of its 60 groups also span both
classes, but they are large (median 86 videos, max 259) and dominated by *within-class* clustering,
so the classic variance-inflating effect wins: **ratio 1.113**, interval widening from 0.0412 to
0.0458. Xception's Celeb-DF row, grouped over 350 ids, widens most of all (**2.069**).

**Manuscript note.** R1-C7's premise — that identity clustering will widen the intervals — holds for
the cross-dataset results but **not** for FF++, where the grouping is a matched pair. Reporting this
honestly is better than presenting only the cross-dataset rows where the expected widening appears.

## 4. Where grouping is not defensible

**WildDeepfake: the grouping is degenerate and no grouped CI is reported.** The persisted `source_id`
takes only two values (the class labels `0`/`1`), giving **2 groups**, and 498 of 2000 resamples had
to be skipped for missing a class. The resulting interval [0.1091, 0.5212] is an artifact. This is
consistent with the standing decision (DEFECT-006) that **no CI of any kind is reported for
WildDeepfake**, since a CI would imply a precision the measurement does not have. The row is kept in
the JSON marked degenerate, and appears in no table.

**Xception Celeb-DF** is grouped on the anonymised sequential ids of DEFECT-002, which are one row
per group — so its grouping is nominal rather than genuine. Phase 7.4 regenerates those scores with
real video names and the row will be recomputed then.

**Celeb-DF v2** has no true identity grouping across the corpus: `idN` is the celebrity subject, and
YouTube-real carries no subject id at all and forms a single 253-video group. This is the coarsest
defensible grouping and both intervals are reported, per the brief.
