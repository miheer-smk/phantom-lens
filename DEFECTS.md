# DEFECTS.md — PRISM R2 discrepancy register

Numbered, dated register of every discrepancy found between the manuscript / published artifacts and
what is reproducible on disk. Status is one of OPEN · UNDER INVESTIGATION · RESOLVED · WONTFIX.

---

## DEFECT-001 — Three different PRISM-50 FF++ c23 AUC vectors in the manuscript
**Opened:** 2026-08-27 · **Status:** ✅ **CLOSED 2026-08-27** — Option B executed (Phase 1.6) · **Severity:** HIGH — reviewer-visible

The manuscript reports three mutually inconsistent PRISM-50 FF++ c23 AUC vectors:

| Location | DF | F2F | FS | NT |
|---|---|---|---|---|
| Table 7 (identity-disjoint) | 0.9706 | 0.8096 | 0.9631 | 0.7867 |
| Tables 10/16/17 ("full-50") | 0.9752 | 0.8255 | 0.9689 | 0.8039 |
| Table 4 (median residual) | 0.9771 | 0.8174 | 0.9637 | 0.7960 |

### Resolution — all three vectors reproduced exactly (max |Δ| = 0.000000)

Six single-variable arms were tested against the artifacts (`results/P1_defect001.json`,
`results/P1_defect001_h5.json`). The mechanism for each vector:

| Vector | Mechanism | Reproduced by |
|---|---|---|
| Table 7 | PRISM-50, fit on **train only** (720 ids, n_train 1383) | H0 |
| Tables 10/16/17 | PRISM-50, fit on **train + val** (860 ids, n_train 1649) | H1 |
| Table 4 | **51 features** — 46 PRISM (4 PRNU-residual removed) + 5 median-residual — train+val | H5 |

**Ruled out as mechanisms:** imputation (H2 ≡ H0 bit-identical — the 50-D matrices have no missing
cells, so the imputer never fires) and sample population (H3 ≡ H0 — the 50-D ∩ residual inner join
returns the same rows). CV protocol and classifier fit are identical throughout. **The single
operative variable between Table 7 and Tables 10/16/17 is the training partition**; the test
partition is byte-identical (n = 267–269), so this is 19% more training data, not a population effect.
The lift is uniform across all four manipulations (+0.0046 to +0.0172), as that mechanism predicts.

**Table 4 is not a PRISM-50 result at all** — it is a residual-extractor comparison over 51 features.
Its caption attributes it to PRISM-50 incorrectly.

**Blast radius:** six locked experiments fit on train+val — `exp8_analyze` (T4), `exp11_stats` (T10),
`exp3_compression` (T16/17), `exp9_analyze` (**T5**), `exp12_redundancy`, `exp5_runtime` (T18).
Everything else, including Tables 7/8/13/15/19, is train-only.

### Closure — Option B executed (Phase 1.6, 2026-08-27)

All six train+val experiments refitted train-only by mechanical patch (15 sites, 5 scripts, diffs
verified to contain nothing else). `current_50D_ref` now returns **0.9706 / 0.8096 / 0.9631 /
0.7867** — bit-identical to Table 7. The three vectors are one vector; the defect is closed by
construction.

**215 cells compared, 187 changed.** Change table: `results/P1_defect001_change_table.csv`
(response-letter ready). Tables 7, 8, 13 and 19 **asserted unchanged in code**
(`results/P1_6_assert_unaffected.json`, all 10 checks UNCHANGED).

**Conclusion stability: 9 reversals in 51 checks** — see `results/P1_defect001_rerun.md`. The
consequential one is **four DeLong SIG→NS flips**: under the primary protocol the full 50-D
representation is **no longer significantly better than a top-10 subset** on face2face, faceswap and
neuraltextures. Diagnosed as effect shrinkage, not variance growth — the AUC gaps more than halve
(e.g. face2face 0.0662 → 0.0252) on an identical test partition. Full-50 still beats top-3
significantly everywhere (p_holm ≤ 0.0003).

**Blocked on:** Table 4's caption correction (DEFECT-007) before its recomputed values are used.

---

## DEFECT-002 — Xception↔PRISM Celeb-DF join key destroyed in the persisted predictions file
**Opened:** 2026-08-27 · **Status:** RESOLVED 2026-08-29 · **Severity:** MEDIUM — affected Table 19 auditability

> **RESOLUTION (D2, 2026-08-29).** The join key is repaired and **all five published Xception
> values plus the paired DeLong statistic reproduce at Δ = 0** on the rebuilt 6,121-video
> intersection:
>
> | Quantity | Published | Re-derived | Δ |
> |---|---|---|---|
> | Xception DeepFakes | 0.9939 | 0.9939 | +0.0000 |
> | Xception Face2Face | 0.9943 | 0.9943 | −0.0000 |
> | Xception FaceSwap | 0.9937 | 0.9937 | −0.0000 |
> | Xception NeuralTextures | 0.9772 | 0.9772 | +0.0000 |
> | Xception Celeb-DF (shared n=6121) | 0.8211 | 0.8211 | 0 |
> | PRISM Celeb-DF (shared n=6121) | 0.6322 | 0.6322 | 0 |
> | **Paired DeLong z** | **15.426** | **15.426** | **0** |
>
> **Table 19 is unchanged. No disclosure entry.**
>
> **Transient mismatch, and its cause.** The first D2 run reported Xception 0.7922 and z = 12.248,
> and was reported as a non-reproduction. That was wrong. D2 hard-coded normalisation
> `[0.5,0.5,0.5]` while the checkpoint was trained with ImageNet constants
> `[0.485,0.456,0.406] / [0.229,0.224,0.225]`, documented at
> `legacy/phantomlens/Major Revision Results/00_logs/xception_train.py:14`. The model accepted the
> wrong inputs silently and returned plausible scores. Recorded as **DEV-010**; no published value
> was ever affected.

`legacy/phantomlens/results_clean/predictions_per_video.csv` stores the Xception Celeb-DF rows with
anonymised sequential ids (`00000`…`06486`) in the `video_path` column instead of video names, while the
PRISM rows use real names (`id10_0000.mp4`). Basename overlap between the two sets is **0**.

**Consequence.** The published Table 19 pairing — Xception 0.8211 on "the shared 6121" and the paired
DeLong z = 15.426 — **cannot be independently re-derived from saved artifacts**. It is not contradicted,
merely unverifiable.

**What does reproduce:** Xception full-set Celeb-DF AUC over its own n=6487 → 0.8207209 vs published
0.8207 (PASS). PRISM 6121 → 0.6321941 vs 0.6322 (PASS).

**Evidence.** The producing script `Major Revision Results/00_logs/exp_g9_xception_predictions.py:43`
*does* write `video_path=str(r["video"])`, and `data_xception/manifest_celebdf.csv` retains real video
names (`id0_0004`). So the anonymisation happened after that script ran; the mapping is recoverable in
principle from the crop manifest but the ordering assumption would be unverified.

**Next action:** Phase 7.4 re-runs Xception per-video scoring with the common frame-average aggregation
and an intact join key, regenerating 0.8211 and the DeLong z from scratch. Do **not** reconstruct the
join by assuming sort order.

---

## DEFECT-003 — WildDeepfake evaluated on 124 of 174 available sequences, and as PRISM-53 not PRISM-50
**Opened:** 2026-08-27 · **Status:** OPEN (Phase 5) · **Severity:** MEDIUM — reviewer R1-C3A

On disk, `data/wilddeepfake/test/` is **not videos**. It is a flat directory of 224×224 PNG face crops
named `{sequence}_{frame}.png`: 3370 real PNGs over **93** real sequences, 3398 fake PNGs over **81**
fake sequences — **174 sequences available**, not 124.

The published Table 15 used 124 (55 real / 69 fake). So **38 real and 12 fake sequences were dropped**,
and the drop is strongly class-asymmetric (41% of real sequences vs 15% of fake). Frames per sequence
range 2–90 (median 30), and PRISM's temporal descriptors need ≥30 frames (rPPG ≥60) — the likely
mechanism, which would make the attrition mechanically class-dependent.

This is a **selection-bias finding in its own right** and must be reported, not smoothed over
(interacts with Phase 2's Fisher test).

### Update 2026-08-27 (GATE 0 amendments C1 + C2) — escalated to HIGH

**C1 — the attrition is significantly class-dependent.** Fisher's exact test on retention,
93 real / 81 fake input against 55 / 69 retained:

|  | retained | excluded | rate |
|---|---|---|---|
| real | 55 | 38 | **59.1%** |
| fake | 69 | 12 | **85.2%** |

**odds ratio 0.2517, p = 1.886 × 10⁻⁴** (two-sided). This is a **class-dependent selection bias in an
already-published table** and must be stated in the manuscript, not merely logged.

**The ≥30-frame threshold does not explain it** — 12 real vs 10 fake sequences fall below 30 frames,
near-identical.

> **Correction (2026-08-27, after the Phase 1.1 gating scan).** An earlier revision of this entry
> attributed the asymmetry to MediaPipe's `MIN_FACE_DETECTIONS` gate. **That was wrong.** Replaying
> the frozen extractor's gates over all 174 sequences and reconciling against the published matrix
> gives the true mechanism breakdown:
>
> | class | feature_computation_failure | insufficient_valid_frames | mediapipe_no_face | retained |
> |---|---|---|---|---|
> | real (93) | **31** | 2 | 5 | 55 |
> | fake (81) | **11** | 1 | 0 | 69 |
>
> MediaPipe accounts for only **5 of the 50 exclusions**. The dominant mechanism is **failure inside
> the 50-D feature computation itself** — 33% of real sequences (31/93) versus 14% of fake (11/81).
> Real WildDeepfake sequences disproportionately *crash the extractor*, they do not merely fail face
> detection. The Fisher test on retention is unaffected (it tests the final outcome): OR 0.2517,
> p = 1.886 × 10⁻⁴ stands.

Evidence: `results/P1_wdf_retention_fisher.json`, `manifest/master_manifest.csv`.

**C2 — undefined descriptors were silently zeroed, not imputed.** `MIN_FRAMES_TEMPORAL` (30) and
`MIN_FRAMES_RPPG` (60) do not exclude a sequence; the extractor emits a default and
`np.nan_to_num(..., nan=0.0)` writes exactly 0.0. No NaN survives to reach the M1 train-only imputer,
so **the FF++ training median is never applied to a single WildDeepfake cell**.

**19.44% of the published 124×50 matrix is zero-fill**, class-asymmetrically (real 21.89% vs fake
17.48%). Background- and codec-dependent descriptors are *not* undefined as the brief anticipated —
worse, they are **computed on a degenerate substrate** (the "background" mask is a thin rim of crop
edge pixels) and emit plausible-looking numbers. 23.9% of cells lie beyond |z| > 3 of the FF++
distribution the scaler was fitted on, 6.1% beyond |z| > 10, worst case −56.5 σ.

**Decisive test.** Discarding every feature *value* and keeping only which cells were zeroed:

| Predictor | AUC |
|---|---|
| zero-count alone, one integer, nothing fitted | 0.3715 → **0.6285 inverted** |
| zero-pattern, 5-fold CV | **0.5635** |
| **published PRISM-53 Table 15** | **0.5212** |

A single bookkeeping integer separates the classes better than the whole 53-D representation. This
does not make 0.5212 *wrong* — it reproduces exactly, and the confound points the opposite way — but
the measurement is **confounded by an effect larger than the reported one**.
Evidence: `results/P1_wdf_descriptor_availability.md`, `.csv`, `results/P1_wdf_zerofill_audit.json`.

**Next action:** Phase 5 must extract on all 174 sequences, persist a three-state availability mask
per cell (computed / undefined-insufficient-frames / undefined-substrate-absent) instead of collapsing
to 0.0, report per-descriptor availability as a primary result, and carry the zero-pattern AUC as a
floor against which any PRISM-50 number is compared. Manuscript needs a stated caveat on Table 15.

---

## DEFECT-004 — Celeb-DF v2 authentic-video count (806) not reconcilable with disk or with the pipeline
**Opened:** 2026-08-27 · **Status:** ✅ **RESOLVED 2026-08-27** (text-only fix; 0.6322 stands) · **Severity:** HIGH — reviewer R1

The manuscript states 806 authentic / 5323 manipulated, then 798 / 5323 processed. Neither the disk
contents nor the legacy audit supports 806.

**On disk** (`data/celebdf_v2/`): Celeb-real 590 + YouTube-real 300 = **890 authentic**;
Celeb-synthesis **5639**; total 6529. `List_of_testing_videos.txt` present (518 entries: 178 real —
108 Celeb-real + 70 YouTube-real — and 340 fake).

**Legacy audit** (`LOCKED_NUMBERS.md` §M4) reports Celeb-DF 50-D extraction *attempted* **875 real /
5612 fake** at success rates 0.912 / 0.948 → 798 / 5320 ≈ the 798 / 5323 processed, and
`features/celebdf_features.csv` has exactly 6121 rows. So the pipeline arithmetic is
875 → 798 and 5612 → 5323, **not** 806 → 798.

**Three unexplained figures:** 890 on disk vs 875 attempted (15 authentic unaccounted); 875 attempted vs
806 claimed (69 unaccounted); 5639 on disk vs 5612 attempted (27 unaccounted). The official test list
was evidently **not** used as the evaluation population (6121 ≫ 518).

**Consequence.** Table 6 currently describes the evaluated subset as if it were the complete dataset.
The AUC 0.6322 itself is unaffected and reproduces exactly — this is a *reporting* defect, not a
results defect.

### Resolution — 806 is a stale output count from the retired pipeline

The hypothesis that 806 was back-computed as *processed + excluded* is **rejected**.
`archive/deprecated_leaky/results/exp_celebdf/results.json` records `n_test_real: 806` alongside the
retired `cross_dataset_auc: 0.6867`. **806 is the authentic count *retained* by the superseded
leakage-prone pipeline**; 798 is the count retained by the current one. The prior authors knew —
`open_ambiguities.md:85`: *"the ~1% face-detection survival difference (my CelebDF real=798 vs
original 806)"*. The manuscript subtracts two incommensurable output counts and reports the
difference as attrition. Corroboration: the fake count is identical (5323) across both pipelines,
which a genuine 8-video input→output attrition could not produce.

**Measured arithmetic** (all 6121 retained paths classified against the release folders):

| | Manuscript | On disk | Δ |
|---|---|---|---|
| authentic input | 806 | **890** (590 Celeb-real + 300 YouTube-real) | −84 |
| authentic excluded | 8 | **92** (Celeb-real 45, YouTube-real 47) | −84 |
| manipulated input | 5323 | **5639** | −316 |
| manipulated excluded | 0 | **316** | −316 |
| **total excluded** | **8** | **408** | understated **51×** |

Both *processed* counts (798 / 5323) are correct.

**Answered:** the official 518-video test list was **not** used — evaluation covers the full release,
11.8× larger. **YouTube-real was included** (253 of 300 retained).

**Second, independent counting error found.** `LOCKED_NUMBERS.md` §M4's "attempted 875 / 5612" is
also wrong: `exp_m4_missingness.py:66` takes the denominator from `data_xception/manifest_celebdf.csv`
— the **Xception crop manifest**, a downstream artifact — so success rates are inflated
(0.912/0.9485 reported vs 0.897/0.944 true).

**Next action:** text-only manuscript fix — replace 806→890 and 5323→5639 as inputs, report 408
exclusions, state that the official test list was not used and that YouTube-real is included, and
flag the class-dependent attrition (authentic 10.3% vs manipulated 5.6%) as a caveat on the 0.3972
real recall. Replacement Table 6 drafted in `results/P1_celebdf_reconciliation.md`. **No re-run
needed; 0.6322 stands.**

---

## DEFECT-005 — Manuscript §2 LightGBM config omits `max_depth=6`
**Opened:** 2026-08-27 · **Status:** OPEN (documentation) · **Severity:** LOW but reproducibility-relevant

The manuscript / R1 letter specifies LightGBM as "200 estimators, lr 0.05, 31 leaves,
min_child_samples 20, balanced class weights". The authoritative producing script
`Major Revision Results/00_logs/baseline_clean.py:41` also sets **`max_depth=6`**, as do
`exp5_runtime.py` and every other locked script.

**The omission is material, measured.** Refitting the Table 18 all-50 profiling model on identical
data, varying only this one parameter:

| `max_depth` | total leaves over 200 trees | serialised size |
|---|---|---|
| `6` (actual) | 5,579 | 613.5 KB |
| unset / `-1` (as the manuscript reads) | 6,200 | 678.2 KB |

An 11% difference in model complexity. A reader reproducing from the manuscript text alone gets
different trees and therefore different scores. The published Table 18 sizes (501.9 / 585.0 / 590.1 /
**611.2** KB) reproduce **bit-exactly** only with `max_depth=6` present in the full exp5 recipe.

**Next action:** add `max_depth=6` to the manuscript's stated config and to `configs/lightgbm.yaml` in
the Phase 12 release repo. No numbers change.

---

## DEFECT-006 — WildDeepfake: the missingness control exceeds the reported PRISM AUC, and no clean subset exists
**Opened:** 2026-08-27 (Phase 1.4) · **Status:** OPEN — author decision taken (caveat + control), Phase 5 scoping affected · **Severity:** HIGH

The only dataset flagged by the Phase 1.4 audit. Distinct from DEFECT-003 — different mechanism,
different descriptors (see `results/P1_missingness_audit.md` §"one root cause or two").

| | AUC |
|---|---|
| **Published PRISM-53 (Table 15)** | **0.5212** |
| PRISM-50, same frozen model, same population *(new measurement)* | **0.4862** — below chance |
| Missingness control, within-dataset 5-fold CV | **0.5689** |
| Missingness control, zero-count alone, nothing fitted | 0.4108 → **0.5892 inverted** |
| **control ÷ PRISM ratio** | **1.130** |

**21.40% of the 124×50 matrix is missing** (real 23.2%, fake 19.9%), against 0.64% for Celeb-DF and
~1% for FF++ — an outlier by a factor of ~33.

**Residualisation (1.4e) cannot be performed.** Complete cases number **9 of 124**; per-pattern
strata are too small to pool. **There is no subset of WildDeepfake on which PRISM can be evaluated
free of the confound.** This is materially worse than a large confound with a clean holdout.

**Mechanism, verified — length, not substrate.** Missingness concentrates in the *length-gated*
descriptors: rPPG 0.909/0.873 (real/fake) and blink up to 0.909/0.826, versus background-dependent
0.117/0.124 and codec-dependent 0.065/0.070. Retained real sequences are significantly shorter —
mean 32.7 vs 41.2 frames, Mann–Whitney **p = 0.034** — so 90.9% of real vs 75.4% of fake fall below
the 60-frame rPPG gate. The class asymmetry traces to `t_blink_duration` (real−fake gap +0.184),
`t_blink_rate` (+0.083) and the three rPPG terms (+0.054).

The background/codec descriptors show *low* missingness precisely because, per DEFECT-003, they do
not fail — they are computed on a degenerate substrate and emit plausible but meaningless values.
Silent wrongness does not register as missingness. **Two defects, two mechanisms, reported separately.**

**Author decision (GATE 1):** caveat **plus** control. Table 15 keeps its AUC and reports the
missingness control beside it.

**Required manuscript changes:**
1. Report the control AUC (0.5689 CV; 0.5892 zero-count inverted) directly beside Table 15's 0.5212,
   with the ratio.
2. State that complete-case residualisation is impossible at n = 9, so the confound cannot be
   partialled out.
3. State the length asymmetry (p = 0.034) and the class-dependent attrition (OR 0.2517,
   p = 1.89 × 10⁻⁴) as selection bias.
4. Do **not** claim WildDeepfake demonstrates cross-domain generalisation in either direction.

**Phase 5 scoping:** an AUC alone is not a reportable result on this dataset. Phase 5 must lead with
per-descriptor availability, report the control AUC alongside any PRISM figure, and re-extract on all
174 sequences with a persisted three-state availability mask.

---

## DEFECT-007 — Table 4 dimensionality: 46 + 5 = 51, not 50
**Opened:** 2026-08-27 (Phase 1.3) · **Status:** ✅ **RESOLVED 2026-08-27 (Phase 1.5)** — caption fix required before recomputation · **Severity:** MEDIUM

`exp8_analyze.py:24,55` builds Table 4's feature set as `base46 + 5` residual descriptors = **51
features**, where `base46 = FC − PRNU_RESID` and
`PRNU_RESID = {s_prnu_energy, s_prnu_face_periph, t_prnu_temporal_stability, t_prnu_face_vs_bg}`
(4 descriptors), so 50 − 4 = 46.

### Resolution: (a) — Table 4 is a mislabelled 51-D variant

**(b) ruled out:** the 50 descriptors are unique (0 duplicates) and the 5 residual names are disjoint
from all 50. **(c) ruled out:** exactly 4 PRNU descriptors exist in PRISM-50 and all 4 are removed,
so the base is unambiguously 46.

**It is not a clean 4-for-4 swap.** Correlating each PRISM PRNU descriptor against its nominal
`median_*` counterpart over 960 c23 pristine videos (PRISM's own estimator *is* median, so these
should agree):

| PRISM descriptor | counterpart | Pearson r | verdict |
|---|---|---|---|
| `s_prnu_energy` | `median_face_energy` | **0.9983** | equivalent |
| `s_prnu_face_periph` | `median_face_bg_ratio` | **0.9347** | close |
| `t_prnu_temporal_stability` | `median_temporal_consistency` | **0.7641** | related |
| `t_prnu_face_vs_bg` | `median_face_bg_corr` | **0.0052** | **unrelated** |

Only 3 of 4 map. `median_face_bg_corr` is a *new* descriptor despite its name, and
`median_bg_energy` has no PRISM-50 counterpart at all (PRISM carries only the ratio). The 51-D set is
46 PRISM + 3 re-parameterised + **2 genuinely new** descriptors.

**Exact algebraic redundancy:** `median_face_bg_ratio ≡ median_face_energy / median_bg_energy`,
verified to `max |diff| = 1.78e-15`. The 51 columns carry **50 independent degrees of freedom** —
though not the *same* 50 as PRISM-50, and trees can split on the ratio directly, so the fitted models
still differ.

**Consequence:** Table 4 mixes dimensionalities — three 51-D rows against one 50-D reference row. The
three method rows are mutually comparable; **`current_50D_ref` is not comparable to them**. The
published conclusion ("estimator choice barely matters", spread 0.8886–0.8944) is probably safe, but
is currently supported by an uncontrolled comparison.

**Required before Phase 1.6 recomputes it:** caption must state 51 features for the method rows and
50 for the reference; the two new descriptors and the exact redundancy must be disclosed.
**Recommended addition:** a dimensionality-matched 50-D controlled row (base46 + the 4 mapping
residual descriptors, dropping `bg_energy`), which makes the estimator comparison controlled. One
extra fit per estimator on saved features. Full analysis: `results/P1_table4_identity.md`.

---

## DEFECT-008 — The PRISM feature extractor is not deterministic: unseeded `np.random.choice`
**Opened:** 2026-08-27 (Phase 6.3) · **Status:** OPEN — fix required for the Zenodo release · **Severity:** MEDIUM (reproducibility), LOW (published values)

**Found by accident, then characterised deliberately.** Three DF40 extraction processes were briefly
running concurrently and re-extracted 1,889 videos more than once. Comparing those duplicate
extractions showed 1,847 of 1,849 pairs differing — the same video, the same code, different features.

**Root cause:** `src/precompute_features_best.py:489`, inside `extract_temporal_noise_stability`:

```python
idx = np.random.choice(len(mask_coords), 100, replace=False)   # <-- unseeded
```

100 face-mask pixels are sampled at random to compute temporal spectral entropy. **The extractor sets
no seed anywhere** (`grep -E "np.random.seed|default_rng|RandomState"` returns nothing), and this is
its **only** stochastic call. The analysis scripts seed themselves; the extractor never does.

**Scope, measured on a controlled back-to-back rerun of 8 videos in one process:**
**exactly 1 of 50 descriptors ever differs** — `t_noise_spectral_entropy`, max |Δ| = **0.1069**,
which is **0.232 SD** of that feature's FF++ pooled spread (SD 0.4617).

**Impact on published values: bounded and small.** Leave-one-out — deleting the feature entirely,
the largest possible perturbation of it:

| Result | with feature (published) | without | Δ | importance rank |
|---|---|---|---|---|
| Table 7 DeepFakes | 0.9706 | 0.9724 | **+0.0018** | 39/50 |
| Table 7 Face2Face | 0.8096 | 0.8133 | **+0.0037** | 33/50 |
| Table 7 FaceSwap | 0.9631 | 0.9642 | **+0.0011** | 42/50 |
| Table 7 NeuralTextures | 0.7867 | 0.7872 | **+0.0004** | 24/50 |
| Table 13 Celeb-DF | 0.6322 | 0.6332 | **+0.0010** | — |

The descriptor ranks 24–42 of 50 in importance and is marginally *harmful* in every case. Since
run-to-run resampling perturbs it by ~0.23 SD — far less than deleting it — **≤ 0.0037 is a hard
upper bound on the effect, and the true effect is much smaller.** No published number is
invalidated.

**But the reproducibility claim is affected.** `LOCKED_NUMBERS.md` asserts bit-identical
reproduction; that holds only because every published result is computed from the **saved feature
CSVs**. **Re-extracting from raw video does not reproduce those CSVs.** A third party following the
release pipeline end-to-end will get different `t_noise_spectral_entropy` values — and this is
exactly what the Editor's mandated archive invites someone to do.

**Required fix (Phase 12):** thread the seed into the extractor —
`rng = np.random.default_rng(seed); idx = rng.choice(...)` — with `seed` from config, defaulting to 42.
Do **not** silently re-extract the published matrices with the fix; the shipped CSVs stay as they are,
and the release documents that the fix changes `t_noise_spectral_entropy` by ≤ 0.24 SD and AUCs by
≤ 0.0037. Add a determinism regression test that extracts one video twice and asserts equality.

**Manuscript:** one sentence in the reproducibility statement.

---

## DEFECT-009 — Unhandled NaN in `extract_codec_temporal_residual` discards the whole video
**Opened:** 2026-08-27 (Phase 6.3) · **Status:** ✅ **CONFIRMED BY INSTRUMENTATION 2026-08-28** — fix deferred to Phase 12 · **Severity:** MEDIUM (raised from LOW–MEDIUM)

Directly observed during DF40 extraction on `inswap/cdf/id0_id1_0000.mp4`:

```
extract_codec_temporal_residual -> np.histogram(r_norm, bins=20, density=True)
ValueError: autodetected range of [nan, nan] is not finite
```

`precompute_features_best.py:906` histograms a residual array that can be all-NaN. The exception
propagates to `process_single_video`'s bare `except`, which returns `None` — so **one NaN in one of
50 descriptors discards the entire video**, rather than yielding that descriptor as missing.

This is the concrete mechanism behind the `feature_computation_failure` bucket that Phase 1.1
inferred from absence (FF++ 240, Celeb-DF 353, WildDeepfake 42 videos). It is now directly observed
with a root cause, and it is **not** class-neutral in general: whichever class has more degenerate
residuals loses more videos, which is how DEFECT-003's class-asymmetric attrition arises.

### Confirmed by instrumentation (Phase 2 / item 3, 2026-08-28)

An instrumented extractor with per-descriptor `try/except` was run over the **635 already-excluded
videos only** (features discarded, production behaviour untouched):

| Failing block | videos | share |
|---|---|---|
| **`t5_codec_residual`** | **540** | **85.0%** |
| none — re-ran clean | 95 | 15.0% |
| all 14 other blocks | **0** | 0% |

All 540 raise the identical `ValueError: autodetected range of [nan, nan] is not finite`, and **no
video fails more than one block**. All 635 passed the gates on re-run, confirming the Phase 1.1
two-stage attrition model exactly.

**Failure rate by dataset:** FF++ 240/10000 = **2.4%**, Celeb-DF 258/6529 = **4.0%**,
WildDeepfake 42/174 = **24.1%**. The 10× elevation on PNG crops confirms the DEFECT-003 substrate
mechanism by observation — but the 2.4% on ordinary FF++ video shows this is **a latent NaN bug
first and a substrate symptom second**.

**Quantified release impact:** with the guard enabled, **540 of 635 currently-excluded videos would
be retained** (FF++ +240, Celeb-DF +258, WildDeepfake +42), each carrying 1 missing descriptor of 50
rather than being dropped. That is a materially different evaluation population — hence the guard
ships **off by default** and the reported results keep the original behaviour.
Evidence: `results/P2_exclusion_mechanisms.md`.


---

## DEFECT-010 — 95 Celeb-DF exclusions are not reproducible: the evaluated population itself varies
**Opened:** 2026-08-28 (Phase 2 / item 3) · **Status:** OPEN · **Severity:** MEDIUM

Of the 635 already-excluded videos re-run under instrumentation, **95 Celeb-DF videos (16 real /
79 fake) raised no exception at all** — they were dropped from the published matrix, yet recompute
cleanly.

This is a **third**, distinct reproducibility defect:
- DEFECT-008 varies one descriptor's *value* (unseeded sampler).
- DEFECT-009 is *deterministic* given the input (all-NaN histogram).
- **DEFECT-010 varies the evaluated *population***.

Re-running extraction would retain ~95 Celeb-DF videos the published run dropped, moving n from
**6121 to ~6216** (+1.5%).

**Likely mechanism, not yet confirmed:** MediaPipe landmark jitter alters the face mask, which alters
the residual, which decides whether the array is all-NaN and therefore whether `t5_codec_residual`
raises. It is **not** the DEFECT-008 sampler — `t5` does not use it.

**Bounding note.** Phase 1.4 showed the Celeb-DF AUC is stable under population perturbation
(complete cases n = 5137 → 0.6382 vs full 6121 → 0.6322), so a 1.5% shift is not *expected* to move
0.6322 materially. **This has not been measured, and no claim is made that it would not.**

**Next action:** (i) confirm the mechanism by re-extracting the 95 under a fixed landmark seed;
(ii) if confirmed, state in the reproducibility statement that the evaluated population is
reproducible to ~±1.5% on Celeb-DF; (iii) the Phase 12 per-descriptor guard removes the sensitivity
entirely, since a NaN descriptor would no longer void a video.

---

## DEFECT-011 — provenance dicts printed into the reviewer-facing response letter
**Opened:** 2026-08-28 (drafting) · **Status:** CLOSED same day · **Severity:** MEDIUM (deliverable-facing)

`scripts/n1_response_letter.py` accessor `v()` returns the **cell dict** when the addressed leaf is
itself a group of cells, and `f()` fell through to `str()` on anything that was not a float. Four
sentences in `R2_response_letter_DRAFT.md` therefore read, literally:

> Extraction `{'value': 48.586, 'status': 'MEASURED', 'source_file': ..., 'source_sha256': ...}`
> s/video and peak memory are unchanged

Affected: the Table 18 runtime paragraph (3 sites) and the standalone-rPPG POS/POS+CHROM/CHROM
sentence (1 site). **This was live in the draft**, not a hypothetical.

**Why it survived:** the earlier N1 «PENDING» accessor fix addressed the same root cause at a
different call site. I fixed the symptom there rather than the accessor, so the second instance was
never going to be caught by inspection.

**Fix.** `f()` unwraps a cell and **raises** on any other dict rather than stringifying it. All five
generators now abort if `'status':`, `"status":` or `'source_sha256'` appears anywhere in their own
output. Regenerated: 0 occurrences across all six drafts.

**Lesson recorded:** every generated deliverable needs an assertion about its *output*, not only
correct code. The guard is what makes this class of bug impossible rather than merely absent.

---

## DEFECT-012 — the claim-registry generator was never committed, so the no-drift guarantee was unenforceable
**Opened:** 2026-08-28 · **Status:** CLOSED same day · **Severity:** MEDIUM (process)

Commit `9805c18` states that `CLAIM_REGISTRY.csv` "is now GENERATED FROM the master results file
rather than maintained alongside it, so the two cannot drift." **The generator was not in the
commit.** The registry could therefore not be regenerated, and the stated guarantee held only for as
long as nobody changed the master file — which happened the same day.

**Fix.** `scripts/p14_claim_registry.py`, validated by reproducing the existing registry *before*
adding anything: all 108 prior rows regenerate with 0 removed and 0 value changes.

One schema change, verified lossless: the reason for an UNAVAILABLE cell moved out of the `status`
column (where it was concatenated as `UNAVAILABLE :: <reason>`, making `status` unfilterable) into
its own `reason` column. All 7 reasons confirmed present; `status` is now exactly
{MEASURED, UNAVAILABLE}. The generator asserts every MEASURED row has a source file and every
UNAVAILABLE cell has a stated reason.

**Lesson recorded:** a commit message asserting a property is not the same as a committed artifact
that enforces it.
