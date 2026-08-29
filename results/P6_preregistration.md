# P6.1 — DF40 evaluation pre-registration

**Committed before any DF40 feature was extracted and before any DF40 AUC was computed.**
Nothing below may be added, dropped, or substituted after seeing a result. Method availability was
established from the file inventory only (counts and formats), which is population information, not
outcome information.

Date: 2026-08-27 · Inventory source: `data/df40/` unpacked from
`Supply/Datasets/DF40 Dataset/*.zip` · Git: committed at tag `P6-prereg` before extraction.

---

## 1. The six pre-registered methods

Chosen a priori to span the face-swap and reenactment families, per the brief.

| # | Method | Directory | Family | Available as video? | Fake n (ff / cdf) | Status |
|---|---|---|---|---|---|---|
| 1 | **InSwapper** | `inswap/` | face swap | ✅ mp4 | 542 (118 / 424) | **INCLUDED** |
| 2 | **FaceDancer** | `facedancer/` | face swap | ✅ mp4 | 788 (140 / 648) | **INCLUDED** |
| 3 | **Wav2Lip** | `wav2lip/` | lip-sync reenactment | ✅ mp4 | 808 (140 / 668) | **INCLUDED** |
| 4 | **SadTalker** | `sadtalker/` | audio-driven reenactment | ✅ mp4 | 802 (138 / 664) | **INCLUDED** |
| 5 | **HyperReenact** | `hyperreenact/` | reenactment | ✅ mp4 | 781 (140 / 641) | **INCLUDED** |
| 6 | **HeyGen** | `heygen/` | commercial avatar | ❌ mixed by class | 50 fake mp4 / 50 real PNG dirs | **NOT EVALUABLE** |

All six were located on disk. Five are evaluable. Family balance among the evaluable five:
2 face-swap, 3 reenactment.

### HeyGen — NOT EVALUABLE (amended 2026-08-27, before any extraction)

**Amendment.** HeyGen was initially registered as "included, constrained — excluded from the
macro-average". That was insufficient. It is now recorded as **not evaluable, and no HeyGen AUC will
be computed or reported.**

**Reason — the format difference is *between classes*, not merely unusual.** HeyGen's fake class
ships as **50 mp4 videos**; its real class ships as **50 directories of PNG frames**. The two classes
therefore pass through different substrates:

- **Codec-dependent descriptors** (`s_benford_dev`, `s_block_artifact`, `s_dbl_compress`,
  `t_residual_flow_corr`, `t_residual_entropy`) read a native H.264 bitstream for the fake class and
  a lossless PNG re-encode for the real class.
- **Background-dependent descriptors** (`s_prnu_face_periph`, `s_face_bg_diff`, `t_prnu_face_vs_bg`,
  the three boundary-coherence terms, `t_skin_bg_decorrelation`) have full-frame background available
  for one class and, on cropped frame exports, a degenerate crop rim for the other.

**Any AUC computed on this pairing would separate container format, not manipulation.** It would
almost certainly be high, and it would be meaningless — precisely the artifact the Phase 1
WildDeepfake audit (DEFECT-003) exposed, but here perfectly confounded with the label because the
format difference *is* the class boundary. A high HeyGen number would be the most misleading figure
in the paper.

**Pre-committed:** no HeyGen AUC, macro-F1, MCC or CI is computed or reported. HeyGen appears in the
methods table as *not evaluable* with this reason stated. The macro-average is over the **five
video-native methods**. This is recorded before any HeyGen feature was extracted.

## 2. Real-video pairing — the leakage question, settled in advance

DF40 fakes derive from two source corpora, held in `ff/` and `cdf/` subdirectories. Naive pairing
with FF++ reals would risk exactly the target-domain leakage Rule 4 forbids, since 720 of the 1000
FF++ identities are PRISM training identities.

**Verified before extraction:** every DF40 `ff/` fake maps to an FF++ **test-partition** identity.
Checked against `splits/ffpp_official_split.json`:

| Method | ff fakes | in FF++ train | in FF++ val | **in FF++ test** |
|---|---|---|---|---|
| InSwapper | 118 | 0 | 0 | **118** |
| FaceDancer | 140 | 0 | 0 | **140** |
| Wav2Lip | 140 | 0 | 0 | **140** |
| SadTalker | 138 | 0 | 0 | **138** |
| HyperReenact | 140 | 0 | 0 | **140** |

DF40 already respects the official FF++ split. (Wav2Lip/SadTalker use a
`{id}_{youtube_hash}_{n}.mp4` scheme; the leading numeric field is the FF++ source id.)

**Pre-committed pairing rule:**
- `ff/` fakes → paired with FF++ c23 pristine videos **restricted to the 140 test identities**.
- `cdf/` fakes → paired with Celeb-DF v2 authentic videos, which are evaluation-only throughout.
- The two subsets are evaluated **separately as well as pooled**, since they have different real
  populations and different native compression.
- An identity-overlap assertion runs in code before every DF40 fit, mirroring
  `assert_no_identity_overlap`. Any DF40 fake resolving to an FF++ train or val identity is a hard
  failure, not a warning.

## 3. Frozen protocol — no DF40 quantity may influence anything

- The **frozen FF++-trained PRISM-50 model** is used unchanged: `models/prism50_scaler_plus_lgbm.joblib`,
  fitted on FF++ train identities only, config `configs/lightgbm.yaml` (`max_depth=6`).
- **Zero DF40 training. Zero DF40 imputation statistics. Zero DF40 threshold fitting. Zero DF40
  feature normalisation.** The StandardScaler and any imputation median come from the FF++ **train**
  partition and are applied unchanged, as for Celeb-DF.
- Feature order is the alphabetical 50-name list in `results/P0_refit_gate.json → feature_order`.
- Seed 42 throughout; 2000 resamples; identity-grouped bootstrap per Phase 3.

## 4. Metrics fixed in advance

Per method: **AUC** with grouped 95% CI (grouped on source identity where recoverable — `ff/` by FF++
source id, `cdf/` by Celeb-DF `idN`), **macro-F1**, **MCC**, real n, fake n, valid n, and full
attrition accounting in the Phase 1.1 controlled vocabulary.

Then an **unweighted macro-average across the five video-native methods** (InSwapper, FaceDancer,
Wav2Lip, SadTalker, HyperReenact), with every per-method value reported so the average is auditable.
HeyGen contributes nothing, having no reported value.

Per-descriptor availability rates are reported for **every** method, not only HeyGen — the Phase 1
WildDeepfake audit (DEFECT-003) showed that an AUC on cropped or re-encoded material is
uninterpretable without them.

## 5. Declared expectation

Zero-shot transfer from a handcrafted 2019-era representation to 2024-era generators is expected to be
**weak — plausibly at or near chance for several methods**. A low AUC is a legitimate, publishable
finding that reinforces the paper's careful framing, and will be reported exactly as measured.

**A result at or below 0.5 will not be inverted, re-paired, re-subset, or otherwise adjusted.** If a
method's AUC is below chance, that is reported as below chance. This sentence exists so that no such
adjustment can later be presented as a methodological refinement.

## 6. Registered outputs

`results/P6_df40_eval.json` (per-method + macro), `results/P6_df40_attrition.json`,
`results/P6_df40_descriptor_availability.csv`, `figures/df40_per_method_auc.{pdf,svg}` + source CSV.
