# `src/aggregation/` — no separate module

**The implementation is monolithic.** Frame-to-video aggregation is a few lines, not a package.

| What | Where |
|---|---|
| PRISM-50 descriptors | aggregated to one vector per video during extraction, in `../preprocessing/precompute_features_seeded.py` |
| Frame-level deep baselines (Xception, LSDA) | `p_V = (1/m) Σ_t p_t` — the unweighted mean over sampled frames, applied **identically to every frame-level baseline** so the Section 6.8 comparison is protocol-matched |
| Where that aggregation is implemented | `../../baselines/scripts/pD1_lsda_eval.py`, `../../baselines/scripts/pD2_xception_rerun.py`, and re-derived from released scores in `../../experiments/run_table19_comparison.py` |

This directory exists so the layout matches the structure suggested in review. It does not
indicate that the code is split — it is not.
