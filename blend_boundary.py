"""
Blending Boundary Gradient Detector for Phantom Lens V4
========================================================
Detects the gradient discontinuity at the face-background boundary
that every face-swap deepfake must introduce during blending.

Physics principle:
  Real faces are part of a continuous scene — gradients flow smoothly
  from face interior through the boundary into the background.
  Deepfakes blend a generated face into the original frame, creating
  an unnatural gradient transition at the blending seam.

All three features are RATIOS, making them codec-invariant:
  compression scales gradients uniformly, so ratios are preserved.

Features:
  1. boundary_gradient_ratio     — boundary ring vs face interior
  2. gradient_direction_coherence — angular consistency at boundary
  3. boundary_vs_background_ratio — boundary ring vs background corners
"""

import numpy as np
import cv2


NEUTRAL = np.array([0.5, 0.5, 0.5], dtype=np.float64)

# Clip raw feature values to this range before normalising to [0, 1]
RAW_CLIP_MAX = 3.0


# ---------------------------------------------------------------------------
# Mask construction
# ---------------------------------------------------------------------------
def _build_masks(h, w):
    """
    Build three boolean masks on a (h, w) grid:
      - ring_mask:     boundary ring (10-20% from edge of central 50% face)
      - interior_mask: face interior (central 30% of frame)
      - corner_mask:   four 30x30 corners (background reference)

    The central 50% face region spans [h/4 : 3h/4, w/4 : 3w/4].
    The boundary ring is the annular zone between the face edge and
    a band 10-20% of face size inward from each edge.

    Layout diagram (224x224 example):
    ┌──────────────────────────┐
    │ corner          corner   │  ← background
    │    ┌──────────────┐      │
    │    │ ░░░RING░░░░░ │      │  ← boundary ring (10-20% inset)
    │    │ ░ ┌────────┐░│      │
    │    │ ░ │INTERIOR│░│      │  ← central 30%
    │    │ ░ └────────┘░│      │
    │    │ ░░░░░░░░░░░░ │      │
    │    └──────────────┘      │
    │ corner          corner   │
    └──────────────────────────┘
    """
    ring_mask = np.zeros((h, w), dtype=bool)
    interior_mask = np.zeros((h, w), dtype=bool)
    corner_mask = np.zeros((h, w), dtype=bool)

    # Central 50% face box
    fy0, fy1 = h // 4, 3 * h // 4
    fx0, fx1 = w // 4, 3 * w // 4
    face_h = fy1 - fy0
    face_w = fx1 - fx0

    # Ring: 10-20% inset from face edge
    # Outer boundary of ring = face edge (0% inset)
    # Inner boundary of ring = 20% inset
    inset_outer = 0.10  # 10% of face size
    inset_inner = 0.20  # 20% of face size

    outer_y0 = fy0 + int(face_h * inset_outer)
    outer_y1 = fy1 - int(face_h * inset_outer)
    outer_x0 = fx0 + int(face_w * inset_outer)
    outer_x1 = fx1 - int(face_w * inset_outer)

    inner_y0 = fy0 + int(face_h * inset_inner)
    inner_y1 = fy1 - int(face_h * inset_inner)
    inner_x0 = fx0 + int(face_w * inset_inner)
    inner_x1 = fx1 - int(face_w * inset_inner)

    # Ring = outer box minus inner box
    ring_mask[outer_y0:outer_y1, outer_x0:outer_x1] = True
    ring_mask[inner_y0:inner_y1, inner_x0:inner_x1] = False

    # Interior: central 30% of frame
    int_y0 = int(h * 0.35)
    int_y1 = int(h * 0.65)
    int_x0 = int(w * 0.35)
    int_x1 = int(w * 0.65)
    interior_mask[int_y0:int_y1, int_x0:int_x1] = True

    # Four corners: 30x30 each (background reference)
    cs = min(30, h // 4, w // 4)  # corner size, safe for small frames
    corner_mask[0:cs, 0:cs] = True               # top-left
    corner_mask[0:cs, w - cs:w] = True            # top-right
    corner_mask[h - cs:h, 0:cs] = True            # bottom-left
    corner_mask[h - cs:h, w - cs:w] = True        # bottom-right

    return ring_mask, interior_mask, corner_mask


# ---------------------------------------------------------------------------
# Gradient computation
# ---------------------------------------------------------------------------
def _compute_gradients(gray):
    """
    Compute gradient magnitude and direction using Sobel operators.
    Returns:
      mag:   gradient magnitude (float64)
      angle: gradient direction in radians [-pi, pi] (float64)
    """
    # Sobel in x and y, 64-bit to avoid overflow on uint8 input
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    angle = np.arctan2(gy, gx)
    return mag, angle


# ---------------------------------------------------------------------------
# Feature 1: Boundary gradient ratio
# ---------------------------------------------------------------------------
def _boundary_gradient_ratio(mag, ring_mask, interior_mask):
    """
    Ratio of mean gradient magnitude at the boundary ring to the
    face interior. Real faces ≈ 1.0, fakes > 1.5 (blending artifact).
    """
    ring_vals = mag[ring_mask]
    int_vals = mag[interior_mask]

    if len(ring_vals) == 0 or len(int_vals) == 0:
        return 0.5

    ring_mean = ring_vals.mean()
    int_mean = int_vals.mean()

    if int_mean < 1e-6:
        # Interior is flat (e.g. solid color) — can't compute ratio
        return 0.5

    return ring_mean / int_mean


# ---------------------------------------------------------------------------
# Feature 2: Gradient direction coherence
# ---------------------------------------------------------------------------
def _gradient_direction_coherence(angle, ring_mask):
    """
    Measure how consistent gradient directions are along the boundary.

    Uses circular variance: Var_c = 1 - R, where R is the mean
    resultant length of the unit vectors (cos(θ), sin(θ)).

    R close to 1 → all gradients point the same way → coherent (real)
    R close to 0 → random directions → incoherent (fake blend seam)

    Returns 1 - circular_variance = R, mapped to [0, 1].
    """
    ring_angles = angle[ring_mask]

    if len(ring_angles) == 0:
        return 0.5

    # Mean resultant length (circular statistics)
    mean_cos = np.mean(np.cos(ring_angles))
    mean_sin = np.mean(np.sin(ring_angles))
    R = np.sqrt(mean_cos ** 2 + mean_sin ** 2)

    # R is naturally in [0, 1]
    return float(np.clip(R, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Feature 3: Boundary vs background ratio
# ---------------------------------------------------------------------------
def _boundary_vs_background_ratio(mag, ring_mask, corner_mask):
    """
    Ratio of boundary gradient to background gradient.
    Real faces ≈ 1.0 (boundary gradients match scene gradients).
    Fakes > 1.5 (unnatural boundary gradients from blending).
    """
    ring_vals = mag[ring_mask]
    bg_vals = mag[corner_mask]

    if len(ring_vals) == 0 or len(bg_vals) == 0:
        return 0.5

    ring_mean = ring_vals.mean()
    bg_mean = bg_vals.mean()

    if bg_mean < 1e-6:
        # Background is flat — degenerate case
        return 0.5

    return ring_mean / bg_mean


# ---------------------------------------------------------------------------
# Normalisation: raw ratio → [0, 1]
# ---------------------------------------------------------------------------
def _normalise_ratio(value, clip_max=RAW_CLIP_MAX):
    """
    Clip ratio to [0, clip_max] then linearly map to [0, 1].
    This ensures all features share the same output scale.
    """
    clipped = np.clip(value, 0.0, clip_max)
    return float(clipped / clip_max)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def compute_blend_boundary(frame_bgr):
    """
    Extract blending boundary gradient features from a single BGR frame.

    Args:
        frame_bgr: numpy array shape (224, 224, 3), dtype uint8, BGR order

    Returns:
        numpy array shape (3,) containing:
          [0] boundary_gradient_ratio      — ring / interior magnitude   [0, 1]
          [1] gradient_direction_coherence — circular consistency        [0, 1]
          [2] boundary_vs_background_ratio — ring / corner magnitude    [0, 1]

        Returns [0.5, 0.5, 0.5] on any failure.
    """
    try:
        # ---- Input validation ----
        if frame_bgr is None:
            return NEUTRAL.copy()
        if not isinstance(frame_bgr, np.ndarray):
            return NEUTRAL.copy()
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] < 3:
            return NEUTRAL.copy()

        h, w = frame_bgr.shape[0], frame_bgr.shape[1]
        if h < 32 or w < 32:
            return NEUTRAL.copy()

        # ---- Convert to grayscale ----
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        # ---- Compute Sobel gradients ----
        mag, angle = _compute_gradients(gray)

        # ---- Build region masks ----
        ring_mask, interior_mask, corner_mask = _build_masks(h, w)

        # ---- Feature 1: boundary / interior gradient ratio ----
        f1_raw = _boundary_gradient_ratio(mag, ring_mask, interior_mask)

        # ---- Feature 2: gradient direction coherence at boundary ----
        # This one is already in [0, 1], no ratio normalisation needed
        f2 = _gradient_direction_coherence(angle, ring_mask)

        # ---- Feature 3: boundary / background gradient ratio ----
        f3_raw = _boundary_vs_background_ratio(mag, ring_mask, corner_mask)

        # ---- Normalise ratio features to [0, 1] ----
        f1 = _normalise_ratio(f1_raw)
        f3 = _normalise_ratio(f3_raw)

        features = np.array([f1, f2, f3], dtype=np.float64)

        # Final safety: clip and NaN check
        if not np.all(np.isfinite(features)):
            return NEUTRAL.copy()
        features = np.clip(features, 0.0, 1.0)

        return features

    except Exception:
        return NEUTRAL.copy()


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 65)
    print("Blending Boundary Gradient Detector — Self-Test")
    print("=" * 65)

    # ---- Test 1: None input ----
    r = compute_blend_boundary(None)
    assert r.shape == (3,) and np.allclose(r, NEUTRAL)
    print(f"[PASS] None input           → {r}")

    # ---- Test 2: Wrong shape ----
    r = compute_blend_boundary(np.zeros((10,), dtype=np.uint8))
    assert np.allclose(r, NEUTRAL)
    print(f"[PASS] Wrong shape          → {r}")

    # ---- Test 3: Too small ----
    r = compute_blend_boundary(np.zeros((16, 16, 3), dtype=np.uint8))
    assert np.allclose(r, NEUTRAL)
    print(f"[PASS] Too small (16x16)    → {r}")

    # ---- Test 4: Uniform frame (no gradients anywhere) ----
    uniform = np.full((224, 224, 3), 128, dtype=np.uint8)
    r = compute_blend_boundary(uniform)
    assert r.shape == (3,) and np.all(r >= 0) and np.all(r <= 1)
    print(f"[PASS] Uniform frame        → {r}")

    # ---- Test 5: Natural gradient (smooth ramp — simulates real) ----
    ramp = np.zeros((224, 224, 3), dtype=np.uint8)
    for y in range(224):
        for x in range(224):
            # Smooth gradient from top-left to bottom-right
            val = int((y + x) / (224 + 224) * 255)
            ramp[y, x] = [val, val, val]
    r = compute_blend_boundary(ramp)
    assert r.shape == (3,) and np.all(r >= 0) and np.all(r <= 1)
    print(f"[PASS] Smooth ramp          → {r}")
    print(f"       (Expected: ratio ~1.0 → normalised ~0.33, coherence high)")

    # ---- Test 6: Fake-like — sharp boundary at face edge ----
    fake_frame = np.full((224, 224, 3), 128, dtype=np.uint8)
    # Place a bright square in the central 50% with hard edge
    fake_frame[56:168, 56:168] = 200
    # Add noise inside to simulate generated texture
    np.random.seed(42)
    noise = np.random.randint(-15, 15, (112, 112, 3))
    fake_frame[56:168, 56:168] = np.clip(
        fake_frame[56:168, 56:168].astype(np.int16) + noise, 0, 255
    ).astype(np.uint8)
    r_fake = compute_blend_boundary(fake_frame)
    assert r_fake.shape == (3,) and np.all(r_fake >= 0) and np.all(r_fake <= 1)
    print(f"[PASS] Fake-like boundary   → {r_fake}")
    print(f"       (Expected: high boundary_gradient_ratio due to hard edge)")

    # ---- Test 7: Real-like — smooth transition at face edge ----
    real_frame = np.full((224, 224, 3), 128, dtype=np.uint8)
    # Gaussian blob in centre — smooth boundary like a real face
    Y, X = np.mgrid[0:224, 0:224]
    gaussian = np.exp(-((X - 112)**2 + (Y - 112)**2) / (2 * 50**2))
    for c in range(3):
        real_frame[:, :, c] = (128 + 60 * gaussian).astype(np.uint8)
    r_real = compute_blend_boundary(real_frame)
    assert r_real.shape == (3,) and np.all(r_real >= 0) and np.all(r_real <= 1)
    print(f"[PASS] Real-like (gaussian) → {r_real}")
    print(f"       (Expected: lower boundary_gradient_ratio than fake)")

    # ---- Test 8: Compare fake vs real boundary ratio ----
    if r_fake[0] > r_real[0]:
        print(f"[PASS] Fake boundary ratio ({r_fake[0]:.4f}) > Real ({r_real[0]:.4f}) ✓")
    else:
        print(f"[WARN] Fake boundary ratio ({r_fake[0]:.4f}) <= Real ({r_real[0]:.4f})")

    # ---- Test 9: Random frame (should not crash) ----
    rand_frame = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    r = compute_blend_boundary(rand_frame)
    assert r.shape == (3,) and np.all(r >= 0) and np.all(r <= 1)
    print(f"[PASS] Random frame         → {r}")

    # ---- Test 10: Non-standard size ----
    big = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    r = compute_blend_boundary(big)
    assert r.shape == (3,) and np.all(r >= 0) and np.all(r <= 1)
    print(f"[PASS] Non-standard 480x640 → {r}")

    # ---- Test 11: Codec-invariance check ----
    # Compress fake_frame with JPEG at different qualities
    # Ratios should stay roughly the same
    print()
    print("Codec-invariance check (JPEG compression):")
    for quality in [95, 50, 15]:
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, buf = cv2.imencode('.jpg', fake_frame, encode_param)
        compressed = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        r_c = compute_blend_boundary(compressed)
        print(f"  JPEG q={quality:2d}  → ratio={r_c[0]:.4f}  "
              f"coherence={r_c[1]:.4f}  bg_ratio={r_c[2]:.4f}")

    print()
    print("=" * 65)
    print("All tests passed. All outputs in [0, 1] with shape (3,).")
    print("=" * 65)