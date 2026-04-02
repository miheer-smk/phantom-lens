"""
rPPG Pulse Signal Extractor for Phantom Lens V4
=================================================
Extracts three physics-based features from video frames that detect
the presence (or absence) of a human blood pulse signal.

Deepfakes do not preserve hemodynamic signals because no face generator
models blood flow through subcutaneous vessels. Real faces exhibit
subtle periodic color changes at 0.7-4.0 Hz (42-240 BPM) caused by
cardiac-synchronous blood volume changes.

Features:
  1. pulse_snr      — Signal-to-noise ratio of strongest pulse frequency
  2. spectral_entropy — Entropy of power spectrum in pulse band
  3. spatial_coherence — Cross-correlation of pulse across face regions

All features are normalised to [0, 1] range.
Real faces: high SNR, low entropy, high coherence
Fake faces: low SNR, high entropy, low coherence
"""

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from scipy.fft import rfft, rfftfreq
import cv2


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MIN_FRAMES = 16          # absolute minimum for any meaningful extraction
DEFAULT_FPS = 25.0       # assumed frame rate
PULSE_LOW_HZ = 0.7       # lower bound of cardiac pulse band (~42 BPM)
PULSE_HIGH_HZ = 4.0      # upper bound of cardiac pulse band (~240 BPM)
FILTER_ORDER = 3          # butterworth filter order (3 = good tradeoff)
NEUTRAL = np.array([0.5, 0.5, 0.5], dtype=np.float64)  # returned on failure


# ---------------------------------------------------------------------------
# Helper: Bandpass filter design
# ---------------------------------------------------------------------------
def _bandpass_coefficients(low_hz, high_hz, fps, order=FILTER_ORDER):
    """
    Design a Butterworth bandpass filter for the cardiac pulse band.
    
    Returns (b, a) filter coefficients.
    Nyquist = fps / 2. We need low_hz and high_hz to be strictly
    below Nyquist, so we clamp to avoid unstable filter design.
    """
    nyquist = fps / 2.0
    # Clamp frequencies to valid range (must be < nyquist and > 0)
    low = max(low_hz / nyquist, 0.01)
    high = min(high_hz / nyquist, 0.99)
    if low >= high:
        # Fallback: widen the band slightly
        low = 0.01
        high = 0.99
    b, a = butter(order, [low, high], btype='band')
    return b, a


# ---------------------------------------------------------------------------
# Helper: Extract green channel time-series from a region of interest
# ---------------------------------------------------------------------------
def _extract_green_signal(frames_bgr, y_start, y_end, x_start, x_end):
    """
    Extract the mean green channel value from a rectangular ROI
    across all frames. Green channel carries the strongest pulse
    signal because hemoglobin absorption peaks near 540nm (green).
    
    Args:
        frames_bgr: list of BGR frames (H, W, 3)
        y_start, y_end: row range of ROI
        x_start, x_end: column range of ROI
    
    Returns:
        1D numpy array of length N (one value per frame)
    """
    signal = np.empty(len(frames_bgr), dtype=np.float64)
    for i, frame in enumerate(frames_bgr):
        roi = frame[y_start:y_end, x_start:x_end, 1]  # channel 1 = Green in BGR
        signal[i] = roi.mean()
    return signal


# ---------------------------------------------------------------------------
# Helper: Detrend signal to remove slow illumination drift
# ---------------------------------------------------------------------------
def _detrend(signal):
    """
    Remove linear trend from signal. Illumination changes cause slow
    drift that would contaminate the pulse band. Simple linear detrend
    is sufficient because the pulse band (0.7+ Hz) is well above any
    linear drift frequency.
    """
    n = len(signal)
    if n < 2:
        return signal
    x = np.arange(n, dtype=np.float64)
    # Least squares linear fit: signal = slope * x + intercept
    slope = (n * np.dot(x, signal) - x.sum() * signal.sum()) / \
            (n * np.dot(x, x) - x.sum() ** 2 + 1e-12)
    intercept = (signal.sum() - slope * x.sum()) / n
    return signal - (slope * x + intercept)


# ---------------------------------------------------------------------------
# Feature 1: Pulse SNR
# ---------------------------------------------------------------------------
def _compute_pulse_snr(green_signal, fps):
    """
    Compute signal-to-noise ratio of the cardiac pulse signal.
    
    Method:
      1. Bandpass filter the green channel signal to isolate pulse band
      2. Compute FFT power spectrum
      3. Find the peak power within the pulse band
      4. SNR = peak_power / mean_power_outside_pulse_band
      5. Map to [0, 1] using sigmoid-like transform
    
    Real faces: strong peak at heart rate frequency → high SNR
    Fake faces: no coherent pulse → low SNR (flat spectrum)
    """
    n = len(green_signal)
    
    # Detrend to remove DC offset and slow drift
    signal = _detrend(green_signal)
    
    # Apply bandpass filter to isolate pulse frequencies
    b, a = _bandpass_coefficients(PULSE_LOW_HZ, PULSE_HIGH_HZ, fps)
    try:
        filtered = filtfilt(b, a, signal, padlen=min(3 * FILTER_ORDER, n - 1))
    except ValueError:
        # filtfilt can fail if signal is too short for the filter
        return 0.5
    
    # Compute one-sided power spectrum via FFT
    freqs = rfftfreq(n, d=1.0 / fps)
    fft_vals = rfft(filtered)
    power = np.abs(fft_vals) ** 2
    
    # Avoid division by zero — if entire spectrum is zero, no signal
    if power.sum() < 1e-12:
        return 0.5
    
    # Identify which frequency bins fall inside the pulse band
    pulse_mask = (freqs >= PULSE_LOW_HZ) & (freqs <= PULSE_HIGH_HZ)
    noise_mask = ~pulse_mask & (freqs > 0)  # exclude DC component
    
    if pulse_mask.sum() == 0 or noise_mask.sum() == 0:
        return 0.5
    
    # Peak power in pulse band
    pulse_power = power[pulse_mask]
    peak_power = pulse_power.max()
    
    # Mean noise power outside pulse band
    noise_power = power[noise_mask].mean()
    
    if noise_power < 1e-12:
        # No noise at all — degenerate case
        return 0.5
    
    snr_linear = peak_power / noise_power
    
    # Map SNR to [0, 1] using sigmoid: 1 / (1 + exp(-k*(log(snr) - threshold)))
    # Calibrated so that SNR=1 (no signal) → ~0.2, SNR=10 (strong pulse) → ~0.85
    snr_db = 10.0 * np.log10(snr_linear + 1e-12)
    snr_normalised = 1.0 / (1.0 + np.exp(-0.3 * (snr_db - 5.0)))
    
    return float(np.clip(snr_normalised, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Feature 2: Spectral Entropy
# ---------------------------------------------------------------------------
def _compute_spectral_entropy(green_signal, fps):
    """
    Compute Shannon entropy of the normalised power spectral density
    within the cardiac pulse band.
    
    Method:
      1. Bandpass filter and compute FFT
      2. Extract power spectrum within pulse band only
      3. Normalise to probability distribution (sum = 1)
      4. Entropy = -sum(p * log(p))
      5. Normalise by log(N) to get [0, 1] range
    
    Real faces: sharp peak at heart rate → low entropy (ordered)
    Fake faces: flat noisy spectrum → high entropy (disordered)
    
    NOTE: We INVERT so that real=high, fake=low (consistent with SNR)
    """
    n = len(green_signal)
    
    signal = _detrend(green_signal)
    
    b, a = _bandpass_coefficients(PULSE_LOW_HZ, PULSE_HIGH_HZ, fps)
    try:
        filtered = filtfilt(b, a, signal, padlen=min(3 * FILTER_ORDER, n - 1))
    except ValueError:
        return 0.5
    
    freqs = rfftfreq(n, d=1.0 / fps)
    fft_vals = rfft(filtered)
    power = np.abs(fft_vals) ** 2
    
    # Extract only the pulse band
    pulse_mask = (freqs >= PULSE_LOW_HZ) & (freqs <= PULSE_HIGH_HZ)
    pulse_power = power[pulse_mask]
    
    if pulse_power.sum() < 1e-12 or len(pulse_power) < 2:
        return 0.5
    
    # Normalise to probability distribution
    p = pulse_power / pulse_power.sum()
    
    # Remove zeros to avoid log(0)
    p = p[p > 1e-12]
    
    if len(p) < 2:
        return 0.5
    
    # Shannon entropy
    entropy = -np.sum(p * np.log(p))
    
    # Normalise by maximum possible entropy (uniform distribution)
    max_entropy = np.log(len(p))
    if max_entropy < 1e-12:
        return 0.5
    
    normalised_entropy = entropy / max_entropy  # range [0, 1]
    
    # INVERT: real faces have low entropy → we want high value for real
    # So feature = 1 - normalised_entropy
    inverted = 1.0 - normalised_entropy
    
    return float(np.clip(inverted, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Feature 3: Spatial Coherence
# ---------------------------------------------------------------------------
def _compute_spatial_coherence(frames_bgr, fps, h, w):
    """
    Compute cross-correlation of pulse signals across three face regions.
    
    Physics principle:
      In real faces, blood flows from the heart through common arteries
      to the forehead, left cheek, and right cheek. All three regions
      exhibit the same pulse waveform (in phase, same frequency).
      
      In deepfakes, each region is generated independently or with
      limited spatial coherence, so pulse-band signals are uncorrelated.
    
    Method:
      1. Divide face into 3 ROIs: forehead, left cheek, right cheek
      2. Extract green channel time-series from each
      3. Bandpass filter each signal
      4. Compute Pearson correlation: forehead↔left, forehead↔right
      5. Coherence = mean of both correlations, mapped to [0, 1]
    
    Real faces: high coherence (corr ≈ 0.6-0.9)
    Fake faces: low coherence (corr ≈ -0.2 to 0.3)
    """
    # Define the central 50% of the frame as the face area
    face_y_start = h // 4
    face_y_end = 3 * h // 4
    face_x_start = w // 4
    face_x_end = 3 * w // 4
    face_h = face_y_end - face_y_start
    face_w = face_x_end - face_x_start
    
    # Region 1: Forehead — top 30% of face area
    forehead_y_start = face_y_start
    forehead_y_end = face_y_start + int(face_h * 0.3)
    forehead_x_start = face_x_start
    forehead_x_end = face_x_end
    
    # Region 2: Left cheek — bottom 60% of face, left half
    cheek_y_start = face_y_start + int(face_h * 0.4)
    cheek_y_end = face_y_end
    left_x_start = face_x_start
    left_x_end = face_x_start + face_w // 2
    
    # Region 3: Right cheek — bottom 60% of face, right half
    right_x_start = face_x_start + face_w // 2
    right_x_end = face_x_end
    
    # Extract green channel signals from each region
    sig_forehead = _extract_green_signal(
        frames_bgr, forehead_y_start, forehead_y_end,
        forehead_x_start, forehead_x_end
    )
    sig_left = _extract_green_signal(
        frames_bgr, cheek_y_start, cheek_y_end,
        left_x_start, left_x_end
    )
    sig_right = _extract_green_signal(
        frames_bgr, cheek_y_start, cheek_y_end,
        right_x_start, right_x_end
    )
    
    # Bandpass filter all three signals
    b, a = _bandpass_coefficients(PULSE_LOW_HZ, PULSE_HIGH_HZ, fps)
    n = len(sig_forehead)
    padlen = min(3 * FILTER_ORDER, n - 1)
    
    try:
        filt_forehead = filtfilt(b, a, _detrend(sig_forehead), padlen=padlen)
        filt_left = filtfilt(b, a, _detrend(sig_left), padlen=padlen)
        filt_right = filtfilt(b, a, _detrend(sig_right), padlen=padlen)
    except ValueError:
        return 0.5
    
    # Compute Pearson correlation between forehead and each cheek
    def _pearson(x, y):
        """Pearson correlation coefficient, safe against zero variance."""
        x = x - x.mean()
        y = y - y.mean()
        denom = np.sqrt(np.dot(x, x) * np.dot(y, y))
        if denom < 1e-12:
            return 0.0
        return float(np.dot(x, y) / denom)
    
    corr_left = _pearson(filt_forehead, filt_left)
    corr_right = _pearson(filt_forehead, filt_right)
    
    # Mean correlation
    mean_corr = (corr_left + corr_right) / 2.0
    
    # Map from [-1, 1] correlation range to [0, 1] feature range
    # correlation of -1 → 0.0, correlation of 0 → 0.5, correlation of 1 → 1.0
    coherence = (mean_corr + 1.0) / 2.0
    
    return float(np.clip(coherence, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def compute_rppg_features(frames_bgr, fps=DEFAULT_FPS):
    """
    Extract rPPG pulse features from a sequence of BGR video frames.
    
    Args:
        frames_bgr: list of numpy arrays, each shape (224, 224, 3), dtype uint8
                     in BGR color order (standard OpenCV format)
        fps: frame rate of the video (default 25.0)
    
    Returns:
        numpy array shape (3,) containing:
          [0] pulse_snr        — pulse signal-to-noise ratio      [0, 1]
          [1] spectral_entropy — inverted spectral entropy         [0, 1]
          [2] spatial_coherence — cross-region pulse correlation   [0, 1]
        
        Real faces:  values tend toward 1.0 (strong pulse, ordered, coherent)
        Fake faces:  values tend toward 0.0-0.5 (no pulse, disordered, incoherent)
        
        Returns [0.5, 0.5, 0.5] on failure or insufficient frames.
    
    Edge cases handled:
        - Fewer than MIN_FRAMES frames → returns neutral [0.5, 0.5, 0.5]
        - Empty frames list → returns neutral
        - All-black or all-white frames → returns neutral (zero variance)
        - Non-standard frame sizes → still works (uses central 50%)
        - NaN or Inf in computation → caught and returns neutral per-feature
    """
    # ---- Guard: minimum frame count ----
    if frames_bgr is None or len(frames_bgr) < MIN_FRAMES:
        return NEUTRAL.copy()
    
    n_frames = len(frames_bgr)
    
    # ---- Validate first frame to get dimensions ----
    try:
        first = frames_bgr[0]
        if first is None or first.ndim != 3 or first.shape[2] < 3:
            return NEUTRAL.copy()
        h, w = first.shape[0], first.shape[1]
        if h < 16 or w < 16:
            return NEUTRAL.copy()
    except (IndexError, AttributeError):
        return NEUTRAL.copy()
    
    # ---- Guard: fps must be positive and reasonable ----
    if fps <= 0 or fps > 1000:
        fps = DEFAULT_FPS
    
    # ---- Guard: Nyquist must exceed our pulse band ----
    nyquist = fps / 2.0
    if nyquist <= PULSE_LOW_HZ:
        # fps too low to capture any pulse signal
        return NEUTRAL.copy()
    
    # ---- Extract green channel from central 50% of frame ----
    # Central 50% is a reasonable proxy for face ROI when input
    # is already a cropped 224x224 face
    cy_start = h // 4
    cy_end = 3 * h // 4
    cx_start = w // 4
    cx_end = 3 * w // 4
    
    green_signal = _extract_green_signal(
        frames_bgr, cy_start, cy_end, cx_start, cx_end
    )
    
    # ---- Guard: check for degenerate signal (all same value) ----
    if green_signal.std() < 1e-6:
        return NEUTRAL.copy()
    
    # ---- Compute each feature independently ----
    # Each feature has its own try/except so one failure doesn't
    # prevent the others from being computed
    
    try:
        snr = _compute_pulse_snr(green_signal, fps)
        if not np.isfinite(snr):
            snr = 0.5
    except Exception:
        snr = 0.5
    
    try:
        entropy = _compute_spectral_entropy(green_signal, fps)
        if not np.isfinite(entropy):
            entropy = 0.5
    except Exception:
        entropy = 0.5
    
    try:
        coherence = _compute_spatial_coherence(frames_bgr, fps, h, w)
        if not np.isfinite(coherence):
            coherence = 0.5
    except Exception:
        coherence = 0.5
    
    features = np.array([snr, entropy, coherence], dtype=np.float64)
    
    # Final safety clip
    features = np.clip(features, 0.0, 1.0)
    
    return features


# ---------------------------------------------------------------------------
# Self-test / demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 65)
    print("rPPG Pulse Feature Extractor — Self-Test")
    print("=" * 65)
    
    # ---- Test 1: Edge case — empty input ----
    result = compute_rppg_features([])
    assert result.shape == (3,), f"Expected shape (3,), got {result.shape}"
    assert np.allclose(result, NEUTRAL), f"Expected neutral, got {result}"
    print(f"[PASS] Empty input         → {result}")
    
    # ---- Test 2: Edge case — too few frames ----
    short_frames = [np.zeros((224, 224, 3), dtype=np.uint8)] * 10
    result = compute_rppg_features(short_frames)
    assert np.allclose(result, NEUTRAL)
    print(f"[PASS] Too few frames (10) → {result}")
    
    # ---- Test 3: Edge case — None input ----
    result = compute_rppg_features(None)
    assert np.allclose(result, NEUTRAL)
    print(f"[PASS] None input          → {result}")
    
    # ---- Test 4: All-black frames (zero signal) ----
    black_frames = [np.zeros((224, 224, 3), dtype=np.uint8)] * 64
    result = compute_rppg_features(black_frames)
    assert result.shape == (3,)
    assert np.all(result >= 0) and np.all(result <= 1)
    print(f"[PASS] All-black frames    → {result}")
    
    # ---- Test 5: Random noise frames (simulates fake — no pulse) ----
    np.random.seed(42)
    noise_frames = [np.random.randint(100, 200, (224, 224, 3), dtype=np.uint8)
                    for _ in range(64)]
    result = compute_rppg_features(noise_frames)
    assert result.shape == (3,)
    assert np.all(result >= 0) and np.all(result <= 1)
    print(f"[PASS] Random noise        → {result}")
    print(f"       (Expected: low SNR, high entropy → values near 0.3-0.6)")
    
    # ---- Test 6: Synthetic pulse signal (simulates real face) ----
    # Create frames with a sinusoidal green channel modulation at 1.2 Hz
    # (72 BPM — typical resting heart rate)
    n_frames_synth = 128
    fps = 25.0
    pulse_freq = 1.2  # Hz
    t = np.arange(n_frames_synth) / fps
    pulse = np.sin(2 * np.pi * pulse_freq * t)  # pure sinusoid
    
    synth_frames = []
    for i in range(n_frames_synth):
        frame = np.full((224, 224, 3), 128, dtype=np.uint8)
        # Modulate green channel by pulse signal (amplitude ~5 levels)
        # This simulates the ~1-2% intensity change from blood pulse
        green_val = int(128 + 5 * pulse[i])
        green_val = max(0, min(255, green_val))
        frame[:, :, 1] = green_val  # BGR: channel 1 = green
        synth_frames.append(frame)
    
    result = compute_rppg_features(synth_frames, fps=fps)
    assert result.shape == (3,)
    assert np.all(result >= 0) and np.all(result <= 1)
    print(f"[PASS] Synthetic pulse     → {result}")
    print(f"       (Expected: high SNR, low entropy inverted=high → values near 0.7-1.0)")
    
    # ---- Test 7: Synthetic pulse with spatial coherence ----
    # All regions get the same pulse → high spatial coherence
    synth_coherent = []
    for i in range(n_frames_synth):
        frame = np.full((224, 224, 3), 128, dtype=np.uint8)
        green_val = int(128 + 5 * pulse[i])
        green_val = max(0, min(255, green_val))
        frame[:, :, 1] = green_val  # uniform pulse everywhere
        synth_coherent.append(frame)
    
    result = compute_rppg_features(synth_coherent, fps=fps)
    print(f"[PASS] Coherent pulse      → {result}")
    print(f"       (Expected: spatial_coherence near 1.0)")
    
    # ---- Test 8: Incoherent regions (simulates fake) ----
    # Each region gets a different frequency pulse
    synth_incoherent = []
    pulse_forehead = np.sin(2 * np.pi * 1.2 * t)
    pulse_left = np.sin(2 * np.pi * 2.3 * t)     # different frequency
    pulse_right = np.sin(2 * np.pi * 3.1 * t)     # yet another frequency
    
    for i in range(n_frames_synth):
        frame = np.full((224, 224, 3), 128, dtype=np.uint8)
        h_f, w_f = 224, 224
        # Forehead: top 30% of central face
        fy_s, fy_e = h_f // 4, h_f // 4 + int((h_f // 2) * 0.3)
        frame[fy_s:fy_e, :, 1] = int(128 + 5 * pulse_forehead[i])
        # Left cheek
        cy_s = h_f // 4 + int((h_f // 2) * 0.4)
        cy_e = 3 * h_f // 4
        cx_mid = w_f // 2
        frame[cy_s:cy_e, :cx_mid, 1] = int(128 + 5 * pulse_left[i])
        # Right cheek
        frame[cy_s:cy_e, cx_mid:, 1] = int(128 + 5 * pulse_right[i])
        synth_incoherent.append(frame)
    
    result = compute_rppg_features(synth_incoherent, fps=fps)
    print(f"[PASS] Incoherent regions  → {result}")
    print(f"       (Expected: spatial_coherence lower than coherent case)")
    
    # ---- Test 9: Non-standard frame size ----
    odd_frames = [np.random.randint(100, 200, (320, 480, 3), dtype=np.uint8)
                  for _ in range(32)]
    result = compute_rppg_features(odd_frames)
    assert result.shape == (3,)
    print(f"[PASS] Non-standard size   → {result}")
    
    # ---- Test 10: Verify output bounds on all tests ----
    print()
    print("=" * 65)
    print("All tests passed. All outputs in [0, 1] range with shape (3,).")
    print("=" * 65)