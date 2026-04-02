"""
Phantom Lens V3 — New Physics Pillars
rPPG SKIPPED (ratio=0.788, fakes have higher SNR than real)

P11: Eye Reflection Symmetry   (3 dims)
P12: Global Illumination Residue (3 dims)

Total new dims: 6
New feature vector: 24 (original) - 1 (dead dim20) + 6 (new) = 29 dims

Author: Miheer Satish Kulkarni, IIIT Nagpur
"""

import cv2
import numpy as np


# ─────────────────────────────────────────────────────────────
# PILLAR 11 — Eye Reflection Symmetry
# dims: [reflection_asymmetry, temporal_stability,
#        geometry_consistency]
# ─────────────────────────────────────────────────────────────

def compute_pillar11_eye_symmetry(frames_bgr):
    """
    Real faces: corneal reflections in left and right eye
    are geometrically consistent with a single light source.
    Face-swapped videos: eye reflections are composited
    incorrectly and show geometric inconsistency.
    Returns 3 values: [asymmetry, temporal_stability,
                        geometry_score]
    """
    if len(frames_bgr) == 0:
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)

    asymmetries         = []
    reflection_positions = []

    for frame in frames_bgr:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        cy = h // 2
        cx = w // 2
        eye_h = h // 8
        eye_w = w // 6

        # Left eye patch
        ly1 = max(0,   cy - h//4)
        ly2 = min(h,   ly1 + eye_h)
        lx1 = max(0,   cx - w//4)
        lx2 = min(w,   lx1 + eye_w)

        # Right eye patch
        ry1 = max(0,   cy - h//4)
        ry2 = min(h,   ry1 + eye_h)
        rx1 = max(0,   cx + w//8)
        rx2 = min(w,   rx1 + eye_w)

        if (ly2-ly1) < 4 or (lx2-lx1) < 4:
            continue
        if (ry2-ry1) < 4 or (rx2-rx1) < 4:
            continue

        left_eye  = gray[ly1:ly2, lx1:lx2].astype(np.float32)
        right_eye = gray[ry1:ry2, rx1:rx2].astype(np.float32)

        l_bright = float(left_eye.max())
        r_bright = float(right_eye.max())

        asymmetry = abs(l_bright - r_bright) / (
            max(l_bright, r_bright) + 1e-6)
        asymmetries.append(asymmetry)

        l_pos = np.unravel_index(
            left_eye.argmax(), left_eye.shape)
        r_pos = np.unravel_index(
            right_eye.argmax(), right_eye.shape)
        l_pos_norm = (l_pos[0] / max(ly2-ly1, 1),
                      l_pos[1] / max(lx2-lx1, 1))
        r_pos_norm = (r_pos[0] / max(ry2-ry1, 1),
                      r_pos[1] / max(rx2-rx1, 1))

        vertical_diff = abs(l_pos_norm[0] - r_pos_norm[0])
        reflection_positions.append(vertical_diff)

    if not asymmetries:
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)

    mean_asymmetry = float(np.mean(asymmetries))

    temporal_stability = float(np.std(reflection_positions)) \
        if len(reflection_positions) > 1 else 0.0

    geometry = float(np.mean(reflection_positions)) \
        if reflection_positions else 0.0

    return np.array(
        [mean_asymmetry, temporal_stability, geometry],
        dtype=np.float32
    )


# ─────────────────────────────────────────────────────────────
# PILLAR 12 — Global Illumination Residue
# dims: [color_temp_mismatch, fill_light_mismatch,
#        saturation_consistency]
# ─────────────────────────────────────────────────────────────

def compute_pillar12_illumination(frame_bgr):
    """
    Real video: face color temperature is physically consistent
    with the scene lighting.
    Face-swap: the composited face has a different color
    profile than the scene predicts.
    Returns 3 values: [color_temp_mismatch, fill_mismatch,
                        saturation_consistency]
    """
    img  = frame_bgr.astype(np.float32)
    h, w = img.shape[:2]

    cy, cx = h // 2, w // 2
    h4, w4 = h // 4, w // 4

    cy_a = max(h4, cy - h4)
    cy_b = min(h - h4, cy + h4)
    cx_a = max(w4, cx - w4)
    cx_b = min(w - w4, cx + w4)

    face = img[cy_a:cy_b, cx_a:cx_b]

    top   = img[:max(1, h//5), :]
    bot   = img[min(h-1, h - h//5):, :]
    left  = img[:, :max(1, w//5)]
    right = img[:, min(w-1, w - w//5):]

    bg = np.concatenate([
        top.reshape(-1, 3),
        bot.reshape(-1, 3),
        left.reshape(-1, 3),
        right.reshape(-1, 3)
    ], axis=0)

    if len(bg) == 0 or face.size == 0:
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)

    face_flat = face.reshape(-1, 3)

    def color_temp_ratio(pixels):
        r = float(pixels[:, 2].mean()) + 1e-6
        b = float(pixels[:, 0].mean()) + 1e-6
        return b / r

    face_temp = color_temp_ratio(face_flat)
    bg_temp   = color_temp_ratio(bg)
    color_temp_mismatch = abs(face_temp - bg_temp)

    face_gray = cv2.cvtColor(
        face.astype(np.uint8), cv2.COLOR_BGR2GRAY
    ).astype(np.float32)

    shadow_mask = face_gray < np.percentile(face_gray, 30)
    face_flat_shadow = face_flat[shadow_mask.flatten()]

    if len(face_flat_shadow) > 10:
        shadow_temp  = color_temp_ratio(face_flat_shadow)
        fill_mismatch = abs(shadow_temp - bg_temp)
    else:
        fill_mismatch = 0.0

    def mean_saturation(pixels):
        if len(pixels) == 0:
            return 0.0
        sats = []
        for px in pixels[::10]:
            b, g, r = float(px[0]), float(px[1]), float(px[2])
            mx = max(r, g, b)
            mn = min(r, g, b)
            if mx > 0:
                sats.append((mx - mn) / mx)
        return float(np.mean(sats)) if sats else 0.0

    face_sat = mean_saturation(face_flat)
    bg_sat   = mean_saturation(bg)
    sat_consistency = abs(face_sat - bg_sat)

    return np.array(
        [color_temp_mismatch, fill_mismatch, sat_consistency],
        dtype=np.float32
    )


# ─────────────────────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing pillar_new.py...")

    # Dummy test with black frame
    dummy_frame  = np.zeros((224, 224, 3), dtype=np.uint8)
    dummy_frames = [dummy_frame] * 8

    p11 = compute_pillar11_eye_symmetry(dummy_frames)
    p12 = compute_pillar12_illumination(dummy_frame)

    print(f"P11 eye symmetry   : {p11}  shape={p11.shape}")
    print(f"P12 illumination   : {p12}  shape={p12.shape}")

    assert p11.shape == (3,), f"P11 wrong shape: {p11.shape}"
    assert p12.shape == (3,), f"P12 wrong shape: {p12.shape}"

    print("\npillar_new.py OK — 6 new dims total")
    print("P11: dims 24-26  (eye reflection symmetry)")
    print("P12: dims 27-29  (global illumination residue)")
    print("New total: 24 original + 6 new = 30 dims in PKL")
    print("Training: 30 - 1 dead (dim20) = 29 live dims")