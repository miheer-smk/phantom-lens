#!/usr/bin/env python3
"""Track C ROI definitions — MediaPipe FaceMesh (478-landmark) index sets.
Single documented source for all region-localized features. Reproducible & inspectable.

Manipulation physics motivating each ROI:
  - NeuralTextures re-renders the MOUTH region -> MOUTH_REGION / LIPS_INNER.
  - Face2Face transfers expression to the INNER face while outer identity stays
    -> INNER_FACE vs OUTER_FACE seam.
"""

# --- Mouth (NeuralTextures target; also F2F) ---
LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
              409, 270, 269, 267, 0, 37, 39, 40, 185]
LIPS_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
              415, 310, 311, 312, 13, 82, 81, 80, 191]
MOUTH_REGION = LIPS_OUTER            # convex hull -> mouth ROI mask

# --- Inner face (Face2Face reenacted region): eyes + brows + nose + mouth central ---
INNER_FACE = [
    # brows
    70, 63, 105, 66, 107, 336, 296, 334, 293, 300,
    # eyes (outer/inner corners + lids)
    33, 133, 159, 145, 362, 263, 386, 374,
    # nose bridge/tip
    168, 6, 197, 195, 5, 4, 1, 19, 94,
    # mouth corners + central lips
    61, 291, 0, 17, 84, 314,
    # inner cheeks
    205, 425, 50, 280,
]

# --- Outer face: full oval (boundary band = OUTER_FACE ring minus INNER_FACE hull) ---
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361,
             288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149,
             150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

# ROI registry (name -> landmark index list) for logging/provenance
ROIS = {
    "mouth_region": MOUTH_REGION,
    "lips_inner":   LIPS_INNER,
    "inner_face":   INNER_FACE,
    "face_oval":    FACE_OVAL,
}

# Candidate feature GROUPS (each measured separately for incremental val-AUC, Track C)
CANDIDATE_GROUPS = {
    "G1_mouth_instability":  ["roi_mouth_dct_midband_std",
                              "roi_mouth_hf_residual_energy",
                              "roi_mouth_texture_flicker"],
    "G2_inner_outer_seam":   ["roi_seam_gradient_disc",
                              "roi_seam_texture_mismatch",
                              "roi_seam_color_mismatch"],
    "G3_motion_tex_coupling":["roi_mouth_motion_texture_corr",
                              "roi_mouth_motion_texture_lag"],
}
ROI_FEATURE_NAMES = [f for g in CANDIDATE_GROUPS.values() for f in g]
