# `src/spatial_features/` — no separate module

**The implementation is monolithic.** The 13 spatial descriptors (`s_*`) are computed inside
`../preprocessing/precompute_features_seeded.py`, not in a separate package.

| What | Where |
|---|---|
| Sensor-noise and PRNU descriptors (`s_noise_*`, `s_prnu_*`) | `precompute_features_seeded.py` → `process_single_video()` |
| Remaining spatial descriptors | same function, same file |
| Canonical descriptor order | **alphabetical**: `sorted(c for c in cols if c[:2] in ("s_","t_"))` — see `classifiers/prism_pipeline.py::feature_columns()`. It is *not* CSV column order. |

This directory exists so the layout matches the structure suggested in review. It does not
indicate that the code is split — it is not.
