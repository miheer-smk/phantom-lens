# Author-Decision Items (flagged facts, NOT drafted prose)
1. In-distribution 0.9939 (leaky) → honest per-manip (50-D mean 0.883 / 53-D 0.932). Retired 0.9991/0.9999.
2. Cross-dataset 0.6989 → 0.632; framing "strong" → "moderate w/ real-class mismatch" (WildDeepfake corroborates).
   **[G8 reconciliation locked]** manuscript 0.6989 (leaky, fake-rec 0.8745) ≠ old-repo 0.6867 (leaky,
   fake-rec 0.9224) ≠ clean **0.632 [0.613,0.654]** (identity-disjoint, AUTHORITATIVE). The two old values
   came from an unresolved, non-identity-disjoint pipeline and disagreed with each other. Full note:
   `07_summary/celebdf_reconciliation.md`. Retire both old values; 0.632 is the single reported number.
3. NT top-3 feature claim → "manipulation-specific, fold-stable" (rewrite item 08).
4. Xception repositioning → interpretability + CPU-efficiency, NOT accuracy superiority. [TEXT HELD for co-author]
5. Leakage disclosure (methods) — identity-disjoint re-run.
6. EXP-4: real-recall collapse is domain-shift, not fixable by threshold — informs the "limitations/threshold" text.
7. Pending author rewrite items (from plan): "Without Deep Learning"/MediaPipe clarification; PRNU→"PRNU-inspired
   residual-energy descriptors"; blink-claim moderation; title. [to surface as EXP-8/9 land]
8. EXP-8 terminology: rename "PRNU" → "PRNU-inspired residual-energy descriptors" / "sensor-residual
   consistency" (no reference sensor pattern estimated). Residual method is robust (median≈gaussian≈wavelet).
9. EXP-9 rPPG: (a) call rPPG a forensic temporal descriptor, NOT medical-grade pulse. (b) rPPG is a WEAK
   standalone cue (~chance); value is complementary only. (c) pure POS marginally beats the POS+CHROM dual
   and is more compression-robust — consider switching. Sensitivity to motion/compression CONFIRMED (R1).
11. **Domain adaptation (response Table 11) — RESOLVED with a reproducing script; result is NEGATIVE.**
    `exp_trackD_DA.py` now implements CORAL, subspace alignment, per-domain standardisation, and per-domain
    quantile alignment (unsupervised; alignment fit on UNLABELED celebdf_dev target features, seed 42,
    identity-disjoint). **Every method FAILS to improve cross-dataset transfer** (celebdf_dev, vs zero-shot
    0.6157): CORAL −0.040, subspace align −0.055…−0.090, standardisation −0.002, quantile −0.015 — most
    actively hurt. → **Do NOT claim CORAL/IFD improves cross-dataset.** Options: (a) report the honest
    negative DA result (strengthens the "domain gap is fundamental" analysis), or (b) remove Table 11.
    This is now reproducible; the earlier unreproducible Table 11 numbers must not be used. (`trackD_DA_dev.json`)
10. M4 missingness audit — REASSURING result to state positively: missingness alone does NOT predict
    real-vs-fake (AUC 0.50 FF++ / 0.51 CelebDF, CIs at chance) → detector performance is not a missingness
    artifact. BUT disclose one mild caveat: Celeb-DF real videos fail 50-D extraction more than fakes
    (0.912 vs 0.948), a small class-dependent selection bias in the zero-shot test set that compounds the
    already-reported Celeb-DF real-recall domain shift (item 2/6). Dataset-identity from missingness is
    statistically >chance (AUC 0.512, n≈11.5k) but negligible in magnitude. Suggest one limitations
    sentence. [author framing — do not overstate as "no effect"; state it as "negligible/at-chance for
    label, mild real-class extraction gap disclosed"]

12. **E1 finding (contribution):** a distributional (order-statistic) representation of the per-frame physics
    features transfers cross-dataset materially better than mean aggregation (Celeb-DF_dev 0.627->0.660, +0.033
    Holm-sig) — the first cross-dataset gain in the program; refutes the "cross-dataset is dead" read. Frame as a
    representation contribution (not new features). [author framing]
13. **X1 finding (one sentence for the paper):** DOMAIN-STABLE != DISCRIMINATIVE. Selecting features whose
    marginals match across FF++/Celeb-DF (low KS distance) HURTS both axes (top-20 stable -> cross 0.518 vs 0.663
    for all features); the features carrying transferable signal are NOT the ones whose distributions match across
    domains. Cautions against naive KS/MMD feature-stability selection for cross-domain forensics. [author framing]

14. **E3 finding (contribution):** SBI's self-blending augmentation — which reaches ~0.93 on Celeb-DF with a
    deep CNN — is **CNN-specific and does NOT transfer to handcrafted physics descriptors**. Training real-vs-
    self-blended (R1) collapses on Celeb-DF (dev CV 0.46; the detector calls all Celeb-DF real), and the hybrid
    (R2) underperforms the standard regime (R0 0.697). Handcrafted features trained on synthetic blends do not
    recognise Celeb-DF's real-deepfake blends. Novel negative: the SBI effect requires a learned (CNN) blend
    representation. [author framing]
15. **Real-class framing correction (for accuracy):** the Celeb-DF weakness is a RANKING problem — reals are
    scored too fake-like — NOT a thresholding artifact. Real recall 0.225 @ θ=0.5 is threshold-dependent and does
    NOT bound AUC (EXP-4: AUC is threshold-invariant). Report it as "Celeb-DF reals ranked too high"; X4 targets
    the ranking (AUC), not the threshold. [author framing]
