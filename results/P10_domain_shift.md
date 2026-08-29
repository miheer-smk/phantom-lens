# Phase 10 (R1-C3B) — class-conditional cross-domain feature shift

For each of the 50 descriptors, FF++ is compared to each target domain **separately by class**, using
standardised mean difference, Wasserstein-1 on standardised values, and the KS statistic.

Artifacts: `results/P10_domain_shift.csv` (all 1,050 descriptor × domain × class rows),
`results/P10_domain_shift.json`

---

## 1. The brief's hypothesis is only half right — and the half that fails matters

The brief anticipated: *"if rPPG descriptors show the largest real-class shift, that mechanically
explains why removing rPPG improved Celeb-DF AUC by 0.0139 in Table 11."*

Two parts, and they come apart:

**Supported — rPPG does shift heavily.** It is the **second-largest** shifting family for Celeb-DF
(mean |SMD| 0.73 real / 0.75 fake), and `t_rppg_peak_prominence` shifts by **+1.7 SD** and
`t_rppg_snr` by **+1.1 SD**. Descriptors displaced that far from their training distribution cannot
contribute usefully, so removing them helping is entirely consistent.

**Refuted — the shift is NOT real-class-specific. It is almost perfectly class-symmetric.**

| descriptor | real SMD | fake SMD | real − fake |
|---|---|---|---|
| `t_rppg_peak_prominence` | +1.715 | +1.773 | **−0.058** |
| `t_rppg_snr` | +1.129 | +1.092 | **+0.038** |
| `t_rppg_harmonic_ratio` | +0.048 | +0.044 | +0.004 |
| `t_rppg_interregion_corr` | −0.017 | −0.075 | +0.058 |

And across all 50 descriptors on Celeb-DF, **the fake class shifts marginally more than the real
class** — mean |SMD| **0.553 fake vs 0.533 real, ratio 0.962**.

**Consequence for the manuscript.** Removing rPPG helps Celeb-DF because those descriptors are
**displaced in both classes simultaneously** — they inject domain noise without carrying
class-discriminative information — **not** because they differentially distort the authentic class.
Any wording that attributes the Table 11 rPPG gain to a *real-class* domain mismatch is not supported
by this analysis and should be corrected.

This does **not** contradict the low real recall (0.3972). That is a decision-threshold and
class-balance effect, and — per Phase 2 — is compounded by genuine selection bias in *which* real
videos survive extraction (OR 0.515, p_holm 1.8 × 10⁻⁶). Those are real findings. What is not
supported is a *feature-distribution* explanation specific to the real class.

## 2. Largest class-conditional shifts, FF++ → Celeb-DF v2

| Descriptor | Group | real SMD | fake SMD | interpretation |
|---|---|---|---|---|
| `s_benford_dev` | codec/compression | **+2.418** | **+2.504** | largest shift in the study; Celeb-DF's encoding history differs wholesale from FF++ c23 |
| `t_noise_spectral_entropy` | noise statistics | −1.416 | **−2.142** | most class-asymmetric of the large shifts (Δ 0.73) |
| `t_rppg_peak_prominence` | rPPG | +1.715 | +1.773 | pulse-quality proxy inflated in both classes |
| `t_landmark_jitter` | landmark geometry | +1.084 | +1.185 | Celeb-DF has more head motion |
| `s_dbl_compress` | codec/compression | +1.145 | +1.126 | double-compression signature differs |
| `t_rppg_snr` | rPPG | +1.129 | +1.092 | |
| `t_rigid_dist_var` | landmark geometry | +0.902 | +1.182 | |
| `t_interpupillary_std` | landmark geometry | +0.884 | +1.102 | |
| `t_skin_bg_decorrelation` | boundary/background | +1.005 | +1.093 | |
| `s_noise_res_std` | noise statistics | −1.014 | −1.066 | |

## 3. Group-level roll-up — which forensic families travel worst

Mean |SMD| by family:

| Family | Celeb-DF real | Celeb-DF fake | WildDeepfake real | WildDeepfake fake |
|---|---|---|---|---|
| **codec/compression** | **0.868** | **0.893** | 1.115 | 1.052 |
| **rPPG physiological** | **0.727** | **0.746** | **1.761** | **1.433** |
| noise statistics | 0.595 | 0.780 | 1.509 | 1.456 |
| landmark geometry | 0.601 | 0.706 | 0.540 | 0.728 |
| structural/frequency | 0.647 | 0.528 | 1.402 | 1.562 |
| specular | 0.501 | 0.603 | 0.358 | 0.756 |
| PRNU residual | 0.576 | 0.518 | 0.976 | 0.666 |
| skin texture | 0.462 | 0.172 | 0.717 | 0.870 |
| motion/flow | 0.275 | 0.380 | 1.202 | 1.387 |
| boundary/background | 0.316 | 0.330 | 0.445 | 0.464 |
| blink dynamics | 0.205 | 0.212 | **1.468** | 0.748 |

**Codec/compression is the worst-travelling family on Celeb-DF**, ahead of rPPG. This connects
directly to the compression analysis (Tables 16/17): descriptors that read encoding history are the
least portable across corpora, because corpora differ in encoding before they differ in anything
forensic. It also independently corroborates DEFECT-009 — the codec-residual descriptor is both the
most fragile (85% of all exclusions) *and* the least portable.

**WildDeepfake shifts about twice as far as Celeb-DF on every family** (mean |SMD| ~1.0 vs ~0.54) and
**here the real class does shift more** (1.035 vs 1.003), driven by blink dynamics (1.468 vs 0.748)
and rPPG (1.761 vs 1.433) — both frame-rate-dependent families, consistent with DEFECT-006's
length/rate mechanism rather than with a content difference.

## 4. Manuscript corrections required

1. **Correct the rPPG framing.** Removing rPPG helps Celeb-DF because the descriptors are displaced
   in *both* classes, not because of a real-class-specific mismatch. State the SMDs.
2. **Report codec/compression as the worst-travelling family**, ahead of rPPG. This is a new
   result and it ties the domain-shift, compression and exclusion analyses into one account.
3. Keep the low real-recall discussion, but attribute it to threshold/class-balance plus the
   Phase 2 selection bias — **not** to feature-distribution asymmetry.
4. WildDeepfake's shift is roughly double Celeb-DF's and is concentrated in frame-rate-dependent
   families, consistent with the operating-domain account rather than with content novelty.
