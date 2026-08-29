# `src/landmarks/` — no separate module

**The implementation is monolithic.** Landmark localisation is not a standalone module; it is a
stage inside `../preprocessing/precompute_features_seeded.py`.

| What | Where |
|---|---|
| MediaPipe FaceMesh initialisation and per-frame landmarking | `precompute_features_seeded.py` → `load_video_frames()` and the face-mask construction that follows it |
| Face mask derivation used by every downstream descriptor | same file, applied before spatial and temporal extraction |
| Landmark-failure accounting | recorded per video as `frames_with_landmarks` and `landmark_success_ratio` in `splits/evaluation_manifest.csv` |

This directory exists so the layout matches the structure suggested in review. It does not
indicate that the code is split — it is not.
