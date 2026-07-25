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
