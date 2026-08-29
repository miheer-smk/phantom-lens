# `src/temporal_features/` — no separate module

**The implementation is monolithic.** The 37 temporal descriptors (`t_*`) are computed inside
`../preprocessing/precompute_features_seeded.py`, not in a separate package.

| What | Where |
|---|---|
| Temporal noise stability (`t1`) — the seeded sampler | `precompute_features_seeded.py` → `extract_temporal_noise_stability()`, seeded by `_video_seed()` |
| Codec-residual descriptor (`t5_codec_residual`) | same file; a NaN here voids the whole video — see `DEFECTS.md` DEFECT-009 |
| rPPG descriptors (POS / CHROM families) | same file |

> **Reproducibility note.** `_video_seed()` hashes the video **path string**, which was absolute at
> extraction time, so the per-video seed is not portable. Re-extraction elsewhere will not
> reproduce `t1` bit-exactly. See `README.md` and `DEFECTS.md` DEFECT-008.

This directory exists so the layout matches the structure suggested in review. It does not
indicate that the code is split — it is not.
