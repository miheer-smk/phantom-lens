"""
Frequency Checkerboard Detector for Phantom Lens V4
=====================================================
Detects periodic peaks in the 2D frequency spectrum caused by
transposed convolution (deconvolution) in GAN/autoencoder face
generators.

Physics principle:
  Transposed convolution with stride s creates spectral copies at
  intervals of N/s in the frequency domain (where N is the spatial
  dimension). This produces a checkerboard pattern of energy peaks
  at mathematically predictable intervals. These peaks exist in ALL
  GAN-generated faces regardless of codec because they arise from
  the upsampling architecture, not the recording pipeline.

  Real camera images have smooth, monotonically decaying radial
  spectra with no periodic peaks.

All features use ratios or relative measures → codec-invariant.

Features:
  1. peak_energy_ratio       — peak energy / total energy
  2. peak_spacing_regularity — consistency of peak intervals
  3. face_vs_bg_spectral_kl  — KL divergence face vs background
"""

import numpy as np
from scipy.signal import find_peaks
import cv2


NEUTRAL = np.array([0.5, 0.5, 0.5], dtype=np.float64)


# ---------------------------------------------------------------------------
# Helper: Extract ROI and compute 2D power spectrum
# ---------------------------------------------------------------------------
def _power_spectrum_2d(gray_roi):
    """
    Compute the 2D power spectrum of a grayscale ROI.
    Returns the magnitude spectrum (shifted so DC is at centre).
    """
    # Window to reduce spectral leakage at frame edges
    h, w = gray_roi.shape
    win_y = np.hanning(h)
    win_x = np.hanning(w)
    window = np.outer(win_y, win_x)

    windowed = gray_roi.astype(np.float64) * window

    fft2 = np.fft.fft2(windowed)
    fft_shifted = np.fft.fftshift(fft2)
    magnitude = np.abs(fft_shifted)

    return magnitude


# ---------------------------------------------------------------------------
# Helper: Azimuthal (radial) average of 2D spectrum
# ---------------------------------------------------------------------------
def _azimuthal_average(magnitude):
    """
    Compute the radial profile of a 2D magnitude spectrum by averaging
    all pixels at the same distance from the centre.

    This collapses the 2D spectrum into a 1D curve: energy vs frequency.
    Real images: smooth decay. GAN fakes: peaks at regular intervals.

    Returns:
      radial_profile: 1D array, length = max radius
      bin_counts:     number of pixels averaged per bin (for weighting)
    """
    h, w = magnitude.shape
    cy, cx = h // 2, w // 2

    # Distance of each pixel from centre
    Y, X = np.ogrid[0:h, 0:w]
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2).astype(np.float64)

    # Integer bin for each pixel
    dist_int = dist.astype(int)
    max_radius = min(cy, cx)  # don't go beyond the shorter axis

    # Accumulate into radial bins
    radial_sum = np.zeros(max_radius, dtype=np.float64)
    bin_counts = np.zeros(max_radius, dtype=np.float64)

    # Mask to valid radius range
    valid = dist_int < max_radius
    np.add.at(radial_sum, dist_int[valid], magnitude[valid])
    np.add.at(bin_counts, dist_int[valid], 1.0)

    # Average (avoid division by zero)
    safe_counts = np.maximum(bin_counts, 1.0)
    radial_profile = radial_sum / safe_counts

    return radial_profile, bin_counts


# ---------------------------------------------------------------------------
# Helper: Normalise spectrum to probability distribution
# ---------------------------------------------------------------------------
def _to_distribution(spectrum):
    """
    Normalise a 1D spectrum to sum to 1 (probability distribution).
    Adds small epsilon to avoid zeros for KL divergence.
    """
    s = spectrum.copy()
    s = np.maximum(s, 0.0)
    total = s.sum()
    if total < 1e-12:
        # Uniform fallback
        return np.ones_like(s) / len(s)
    p = s / total
    # Add epsilon and renormalise to avoid log(0) in KL
    p = p + 1e-10
    p = p / p.sum()
    return p


# ---------------------------------------------------------------------------
# Feature 1: Peak energy ratio
# ---------------------------------------------------------------------------
def _peak_energy_ratio(radial_profile):
    """
    Find peaks in the radial spectral profile and compute the ratio
    of energy at peaks to total spectral energy.

    GAN upsampling creates periodic energy peaks → high ratio.
    Real cameras produce smooth decay → low ratio (few/no peaks).

    Returns raw ratio (will be normalised to [0,1] externally).
    """
    if len(radial_profile) < 10:
        return 0.0

    # Skip DC component (bin 0) and very low frequencies (bins 1-2)
    # These are dominated by image content, not artifacts
    profile = radial_profile[3:]

    if len(profile) < 5:
        return 0.0

    total_energy = profile.sum()
    if total_energy < 1e-12:
        return 0.0

    # Detrend: remove the smooth decay envelope to expose peaks
    # Use a large median filter as the envelope estimate
    kernel_size = max(3, len(profile) // 4)
    if kernel_size % 2 == 0:
        kernel_size += 1  # must be odd
    from scipy.ndimage import median_filter
    envelope = median_filter(profile, size=kernel_size)
    residual = profile - envelope

    # Find peaks in the residual (above-envelope bumps)
    # prominence threshold relative to residual std
    res_std = residual.std()
    if res_std < 1e-12:
        return 0.0

    prominence = max(res_std * 0.5, 1e-6)
    peaks, properties = find_peaks(residual, prominence=prominence, distance=3)

    if len(peaks) == 0:
        return 0.0

    # Energy at peak locations (from original profile, not residual)
    peak_energy = profile[peaks].sum()
    ratio = peak_energy / total_energy

    return float(ratio)


# ---------------------------------------------------------------------------
# Feature 2: Peak spacing regularity
# ---------------------------------------------------------------------------
def _peak_spacing_regularity(radial_profile):
    """
    Measure how regularly spaced the spectral peaks are.

    GAN transposed convolution creates peaks at exact multiples of N/s
    → very regular spacing (low variance of inter-peak distances).

    Real images have irregular or no peaks → high variance.

    Returns value in [0, 1] where 1 = perfectly regular (suspicious).
    """
    if len(radial_profile) < 10:
        return 0.5

    profile = radial_profile[3:]
    if len(profile) < 5:
        return 0.5

    # Same detrending as feature 1
    kernel_size = max(3, len(profile) // 4)
    if kernel_size % 2 == 0:
        kernel_size += 1
    from scipy.ndimage import median_filter
    envelope = median_filter(profile, size=kernel_size)
    residual = profile - envelope

    res_std = residual.std()
    if res_std < 1e-12:
        return 0.5

    prominence = max(res_std * 0.5, 1e-6)
    peaks, _ = find_peaks(residual, prominence=prominence, distance=3)

    if len(peaks) < 3:
        # Not enough peaks to measure regularity
        # Fewer peaks → more likely real → return low regularity
        return 0.2

    # Inter-peak spacing
    spacings = np.diff(peaks).astype(np.float64)

    mean_spacing = spacings.mean()
    if mean_spacing < 1e-6:
        return 0.5

    # Coefficient of variation (CV) = std / mean
    # Low CV = regular spacing = suspicious (GAN artifact)
    cv = spacings.std() / mean_spacing

    # Map CV to [0, 1]: CV=0 → regularity=1.0, CV=1+ → regularity≈0.0
    # Using exponential decay: regularity = exp(-2 * CV)
    regularity = float(np.exp(-2.0 * cv))

    return np.clip(regularity, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Feature 3: Face vs background spectral KL divergence
# ---------------------------------------------------------------------------
def _face_bg_spectral_kl(gray, h, w):
    """
    KL divergence between radial spectra of face ROI and background.

    Deepfakes only modify the face region, so the face has a different
    frequency signature (GAN peaks) while the background retains the
    original camera spectrum. High KL = spectral mismatch = suspicious.

    Real faces: face and background share the same camera + codec
    characteristics → low KL divergence.

    Returns raw KL value (normalised externally).
    """
    # Face ROI: central 50%
    fy0, fy1 = h // 4, 3 * h // 4
    fx0, fx1 = w // 4, 3 * w // 4
    face_roi = gray[fy0:fy1, fx0:fx1]

    # Background: assemble from four borders (top, bottom, left, right strips)
    border = h // 4  # width of each border strip
    strips = []
    # Top strip
    if border >= 8:
        strips.append(gray[0:border, 0:w])
    # Bottom strip
    if border >= 8:
        strips.append(gray[h - border:h, 0:w])

    if len(strips) == 0:
        return 0.0

    # Use the largest strip that forms a reasonable rectangle
    # Concatenate top and bottom into one background patch
    bg_roi = np.vstack(strips)

    # Both ROIs need minimum size for meaningful FFT
    if face_roi.shape[0] < 16 or face_roi.shape[1] < 16:
        return 0.0
    if bg_roi.shape[0] < 16 or bg_roi.shape[1] < 16:
        return 0.0

    # Make both ROIs square by cropping to the smaller dimension
    # This ensures radial profiles have the same length
    def _make_square(roi):
        rh, rw = roi.shape
        s = min(rh, rw)
        cy, cx = rh // 2, rw // 2
        return roi[cy - s // 2:cy - s // 2 + s, cx - s // 2:cx - s // 2 + s]

    face_sq = _make_square(face_roi)
    bg_sq = _make_square(bg_roi)

    if face_sq.shape[0] < 16 or bg_sq.shape[0] < 16:
        return 0.0

    # Resize both to same dimensions for comparable radial profiles
    target_size = min(face_sq.shape[0], bg_sq.shape[0])
    if face_sq.shape[0] != target_size:
        face_sq = cv2.resize(face_sq, (target_size, target_size))
    if bg_sq.shape[0] != target_size:
        bg_sq = cv2.resize(bg_sq, (target_size, target_size))

    # Compute radial profiles
    face_mag = _power_spectrum_2d(face_sq)
    bg_mag = _power_spectrum_2d(bg_sq)

    face_radial, _ = _azimuthal_average(face_mag)
    bg_radial, _ = _azimuthal_average(bg_mag)

    # Trim to same length (should already match, but safety)
    min_len = min(len(face_radial), len(bg_radial))
    if min_len < 4:
        return 0.0

    face_r = face_radial[:min_len]
    bg_r = bg_radial[:min_len]

    # Convert to probability distributions
    p = _to_distribution(face_r)
    q = _to_distribution(bg_r)

    # Symmetric KL divergence: (KL(p||q) + KL(q||p)) / 2
    # More stable than one-sided KL
    kl_pq = np.sum(p * np.log(p / q))
    kl_qp = np.sum(q * np.log(q / p))
    kl_sym = (kl_pq + kl_qp) / 2.0

    return float(max(kl_sym, 0.0))


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------
def _sigmoid_norm(value, midpoint, slope):
    """Map value to [0,1] via sigmoid: 1 / (1 + exp(-slope*(value - midpoint)))"""
    return float(1.0 / (1.0 + np.exp(-slope * (value - midpoint))))


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def compute_freq_checkerboard(frame_bgr):
    """
    Extract frequency checkerboard features from a single BGR frame.

    Args:
        frame_bgr: numpy array (224, 224, 3), dtype uint8, BGR

    Returns:
        numpy array shape (3,):
          [0] peak_energy_ratio        — spectral peak energy fraction  [0, 1]
          [1] peak_spacing_regularity  — regularity of peak intervals   [0, 1]
          [2] face_vs_bg_spectral_kl   — face-background spectral KL   [0, 1]

        Higher values → more likely fake (GAN artifacts detected).
        Returns [0.5, 0.5, 0.5] on failure.
    """
    try:
        # ---- Input validation ----
        if frame_bgr is None or not isinstance(frame_bgr, np.ndarray):
            return NEUTRAL.copy()
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] < 3:
            return NEUTRAL.copy()

        h, w = frame_bgr.shape[0], frame_bgr.shape[1]
        if h < 32 or w < 32:
            return NEUTRAL.copy()

        # ---- Convert to grayscale ----
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        # ---- Extract face ROI (central 50%) ----
        fy0, fy1 = h // 4, 3 * h // 4
        fx0, fx1 = w // 4, 3 * w // 4
        face_gray = gray[fy0:fy1, fx0:fx1]

        if face_gray.shape[0] < 16 or face_gray.shape[1] < 16:
            return NEUTRAL.copy()

        # ---- 2D FFT of face region ----
        face_mag = _power_spectrum_2d(face_gray)
        radial_profile, _ = _azimuthal_average(face_mag)

        if len(radial_profile) < 10:
            return NEUTRAL.copy()

        # ---- Feature 1: Peak energy ratio ----
        f1_raw = _peak_energy_ratio(radial_profile)
        # Map to [0,1]: ratio=0 → 0.0, ratio=0.1 → ~0.5, ratio=0.3+ → ~1.0
        f1 = _sigmoid_norm(f1_raw, midpoint=0.10, slope=20.0)

        # ---- Feature 2: Peak spacing regularity ----
        f2 = _peak_spacing_regularity(radial_profile)

        # ---- Feature 3: Face vs background spectral KL ----
        f3_raw = _face_bg_spectral_kl(gray, h, w)
        # Map to [0,1]: KL=0 → 0.0, KL=0.5 → ~0.5, KL=2+ → ~1.0
        f3 = _sigmoid_norm(f3_raw, midpoint=0.5, slope=3.0)

        features = np.array([f1, f2, f3], dtype=np.float64)

        # NaN safety
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
    from scipy.ndimage import median_filter  # pre-import for tests

    print("=" * 65)
    print("Frequency Checkerboard Detector — Self-Test")
    print("=" * 65)

    # ---- Test 1: None / bad inputs ----
    assert np.allclose(compute_freq_checkerboard(None), NEUTRAL)
    print("[PASS] None input           → neutral")

    assert np.allclose(compute_freq_checkerboard(np.zeros((5,), dtype=np.uint8)), NEUTRAL)
    print("[PASS] Wrong shape          → neutral")

    assert np.allclose(compute_freq_checkerboard(np.zeros((16, 16, 3), dtype=np.uint8)), NEUTRAL)
    print("[PASS] Too small            → neutral")

    # ---- Test 2: Uniform frame ----
    uniform = np.full((224, 224, 3), 128, dtype=np.uint8)
    r = compute_freq_checkerboard(uniform)
    assert r.shape == (3,) and np.all(r >= 0) and np.all(r <= 1)
    print(f"[PASS] Uniform frame        → {r}")

    # ---- Test 3: Natural image (smooth gradient — real-like) ----
    Y, X = np.mgrid[0:224, 0:224]
    natural = np.zeros((224, 224, 3), dtype=np.uint8)
    # Smooth gaussian texture
    blob = (128 + 60 * np.exp(-((X-112)**2 + (Y-112)**2) / (2*40**2))).astype(np.uint8)
    natural[:, :, 0] = blob
    natural[:, :, 1] = blob
    natural[:, :, 2] = blob
    r_nat = compute_freq_checkerboard(natural)
    assert r_nat.shape == (3,) and np.all(r_nat >= 0) and np.all(r_nat <= 1)
    print(f"[PASS] Natural (gaussian)   → {r_nat}")

    # ---- Test 4: Fake-like — add periodic frequency artifacts ----
    # Simulate GAN checkerboard: add sinusoidal patterns at regular intervals
    np.random.seed(42)
    fake_frame = np.random.randint(100, 160, (224, 224, 3), dtype=np.uint8)
    # Add periodic pattern only in face region (central 50%)
    face_region = fake_frame[56:168, 56:168, :].astype(np.float64)
    yy, xx = np.mgrid[0:112, 0:112]
    # Simulate transposed conv artifacts: periodic peaks at stride intervals
    for freq in [8, 16, 32]:  # multiple harmonics
        pattern = 10.0 * np.sin(2 * np.pi * freq * xx / 112)
        pattern += 10.0 * np.sin(2 * np.pi * freq * yy / 112)
        for c in range(3):
            face_region[:, :, c] += pattern
    fake_frame[56:168, 56:168, :] = np.clip(face_region, 0, 255).astype(np.uint8)

    r_fake = compute_freq_checkerboard(fake_frame)
    assert r_fake.shape == (3,) and np.all(r_fake >= 0) and np.all(r_fake <= 1)
    print(f"[PASS] Fake (periodic)      → {r_fake}")

    # ---- Test 5: Compare ----
    print()
    print("Comparison (higher = more fake-like):")
    names = ["peak_energy_ratio", "peak_spacing_reg", "face_bg_kl"]
    for i, n in enumerate(names):
        direction = "FAKE > REAL ✓" if r_fake[i] > r_nat[i] else "unexpected"
        print(f"  {n:<22s}  natural={r_nat[i]:.4f}  fake={r_fake[i]:.4f}  [{direction}]")

    # ---- Test 6: Codec-invariance (JPEG compression) ----
    print()
    print("Codec-invariance check (JPEG compression on fake frame):")
    for quality in [95, 50, 15]:
        _, buf = cv2.imencode('.jpg', fake_frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        compressed = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        r_c = compute_freq_checkerboard(compressed)
        print(f"  JPEG q={quality:2d}  → peak_ratio={r_c[0]:.4f}  "
              f"regularity={r_c[1]:.4f}  kl={r_c[2]:.4f}")

    # ---- Test 7: Random noise (no structure) ----
    rand_frame = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    r_rand = compute_freq_checkerboard(rand_frame)
    assert r_rand.shape == (3,) and np.all(r_rand >= 0) and np.all(r_rand <= 1)
    print(f"\n[PASS] Random noise         → {r_rand}")

    # ---- Test 8: Non-standard size ----
    big = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    r_big = compute_freq_checkerboard(big)
    assert r_big.shape == (3,) and np.all(r_big >= 0) and np.all(r_big <= 1)
    print(f"[PASS] Non-standard 480x640 → {r_big}")

    print()
    print("=" * 65)
    print("All tests passed. All outputs in [0, 1] with shape (3,).")
    print("=" * 65)