#!/usr/bin/env python3
"""
PHANTOM LENS V2 — PRISM Feature Extractor (Landmark-Based)
===========================================================
Physics-Reality Integrated Signal Multistream feature extraction.
Extracts 50 physics-grounded features (13 spatial + 37 temporal)
from face videos using MediaPipe landmark anchoring.

Locked Physics Pillars (19 active):
  SPATIAL (per-frame, landmark-anchored):
    P1  Noise Physics         — VMR, ResStd, HFRatio
    P2  PRNU/Camera           — Energy, Face/Periph
    P4  Shadow/Light          — Shadow score, Face-BG diff
    P6  Compression Forensics — Benford, BlockArt, DblCompress
    P8  Motion Blur           — Blur magnitude
    P9  Optical Flow          — Flow magnitude, Dir consistency

  TEMPORAL (across-frame consistency):
    T1  Temporal Noise Stability     — 3 features
    T2  rPPG Cardiac Signal          — 4 features
    T3  Temporal PRNU Persistence    — 2 features
    T4  Face Structural Stability    — 3 features
    T5  Codec Temporal Residual      — 2 features
    T6  Landmark Trajectory          — 4 features
    T7  Rigid Geometry Consistency   — 3 features
    T8  Face-BG Edge Coherence       — 3 features
    T9  Skin Texture Coherence       — 2 features
    T10 Color Transfer Consistency   — 2 features
    T11 Specular Reflection Temporal — 2 features
    T12 Blink Dynamics               — 3 features
    T13 Motion-Blur Coupling         — 2 features
    T14 DCT Temporal Stability       — 2 features

Usage:
    python precompute_features_v3.py \
        --video_dir /path/to/videos \
        --output features_v3.csv \
        --label 0   # 0=real, 1=fake
        --max_frames 300 \
        --workers 4
"""

import argparse
import csv
import os
import sys
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
from scipy import signal as sp_signal
from scipy import stats as sp_stats
from scipy.ndimage import uniform_filter
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION
# ============================================================================

FEATURE_NAMES_SPATIAL = [
    "s_noise_vmr", "s_noise_res_std", "s_noise_hf_ratio",
    "s_prnu_energy", "s_prnu_face_periph",
    "s_shadow_score", "s_face_bg_diff",
    "s_benford_dev", "s_block_artifact", "s_dbl_compress",
    "s_blur_mag",
    "s_flow_mag", "s_flow_dir_consist",
]

FEATURE_NAMES_TEMPORAL = [
    "t_noise_temporal_corr", "t_noise_corr_std", "t_noise_spectral_entropy",
    "t_rppg_snr", "t_rppg_peak_prominence", "t_rppg_interregion_corr",
    "t_rppg_harmonic_ratio",
    "t_prnu_temporal_stability", "t_prnu_face_vs_bg",
    "t_face_ssim_mean", "t_face_ssim_std", "t_face_ssim_min",
    "t_residual_flow_corr", "t_residual_entropy",
    "t_landmark_jitter", "t_landmark_accel_var",
    "t_landmark_velocity_autocorr", "t_jaw_chin_rigidity",
    "t_rigid_dist_var", "t_interpupillary_std", "t_nose_bridge_std",
    "t_boundary_grad_temporal", "t_boundary_color_disc",
    "t_boundary_freq_leakage",
    "t_skin_texture_corr", "t_texture_warp_residual",
    "t_skin_color_jitter", "t_skin_bg_decorrelation",
    "t_specular_stability", "t_specular_symmetry",
    "t_blink_rate", "t_blink_duration", "t_blink_symmetry",
    "t_motion_blur_coupling", "t_coupling_consistency",
    "t_dct_temporal_std", "t_dct_temporal_autocorr",
]

ALL_FEATURE_NAMES = FEATURE_NAMES_SPATIAL + FEATURE_NAMES_TEMPORAL
N_FEATURES = len(ALL_FEATURE_NAMES)

# MediaPipe landmark indices for face regions
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361,
             288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149,
             150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
FOREHEAD = [10, 67, 109, 338, 297, 21, 54, 103, 68, 104]
LEFT_CHEEK = [116, 117, 118, 119, 120, 121, 187, 207, 206]
RIGHT_CHEEK = [345, 346, 347, 348, 349, 350, 411, 427, 426]
NOSE_BRIDGE = [168, 6, 197, 195, 5, 4, 1]
JAW = [152, 148, 176, 149, 150, 136, 172, 58, 132, 377, 400, 378, 379, 365, 397]
CHIN = [152, 175, 199, 200]
LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
              409, 270, 269, 267, 0, 37, 39, 40, 185]
LEFT_PUPIL = [468]   # MediaPipe iris landmarks (if available)
RIGHT_PUPIL = [473]

# Minimum requirements
MIN_FRAMES_SPATIAL = 10
MIN_FRAMES_TEMPORAL = 30
MIN_FRAMES_RPPG = 60
MIN_FACE_DETECTIONS = 0.5   # at least 50% of frames must have face detected


# ============================================================================
# LANDMARK & ROI UTILITIES
# ============================================================================

def init_face_mesh():
    """Initialize MediaPipe Face Mesh."""
    import mediapipe as mp
    return mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


def get_landmarks(face_mesh, frame_rgb):
    """Extract 478 landmarks from a frame. Returns (478, 2) array or None."""
    results = face_mesh.process(frame_rgb)
    if not results.multi_face_landmarks:
        return None
    lm = results.multi_face_landmarks[0]
    h, w = frame_rgb.shape[:2]
    coords = np.array([(l.x * w, l.y * h) for l in lm.landmark], dtype=np.float32)
    return coords


def landmarks_to_mask(landmarks, indices, shape, margin=0):
    """Create binary mask from landmark polygon with optional margin expansion."""
    h, w = shape[:2]
    pts = landmarks[indices].astype(np.int32)
    hull = cv2.convexHull(pts)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 1)
    if margin > 0:
        kernel = np.ones((margin * 2 + 1, margin * 2 + 1), np.uint8)
        mask = cv2.dilate(mask, kernel)
    return mask


def get_face_mask(landmarks, shape):
    """Get face region mask using face oval landmarks."""
    return landmarks_to_mask(landmarks, FACE_OVAL, shape)


def get_face_bbox(landmarks, shape, padding=0.1):
    """Get bounding box of face from landmarks with padding."""
    h, w = shape[:2]
    pts = landmarks[FACE_OVAL]
    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)
    fw, fh = x_max - x_min, y_max - y_min
    x_min = max(0, int(x_min - fw * padding))
    y_min = max(0, int(y_min - fh * padding))
    x_max = min(w, int(x_max + fw * padding))
    y_max = min(h, int(y_max + fh * padding))
    return x_min, y_min, x_max, y_max


def get_roi_pixels(frame, landmarks, indices):
    """Get pixel values within a landmark-defined region."""
    mask = landmarks_to_mask(landmarks, indices, frame.shape)
    return frame[mask > 0]


def get_roi_mean_rgb(frame, landmarks, indices):
    """Get mean RGB of a landmark-defined region."""
    pixels = get_roi_pixels(frame, landmarks, indices)
    if len(pixels) < 10:
        return np.array([0.0, 0.0, 0.0])
    return pixels.mean(axis=0)


def compute_ear(landmarks, eye_indices):
    """Eye Aspect Ratio from 6 landmarks."""
    p = landmarks[eye_indices]
    A = np.linalg.norm(p[1] - p[5])
    B = np.linalg.norm(p[2] - p[4])
    C = np.linalg.norm(p[0] - p[3])
    if C < 1e-6:
        return 0.0
    return (A + B) / (2.0 * C)


def get_face_size(landmarks):
    """Return face height for normalization."""
    pts = landmarks[FACE_OVAL]
    return np.linalg.norm(pts.max(axis=0) - pts.min(axis=0))


# ============================================================================
# VIDEO LOADING
# ============================================================================

def load_video_frames(video_path, max_frames=300, target_size=None):
    """Load frames from video file.
    
    Returns: list of BGR frames, fps
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        total = 10000
    
    step = max(1, total // max_frames) if total > max_frames else 1
    
    frames = []
    idx = 0
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            if target_size:
                frame = cv2.resize(frame, target_size)
            frames.append(frame)
        idx += 1
    
    cap.release()
    return frames, fps


# ============================================================================
# SPATIAL PILLAR EXTRACTORS (per-frame, landmark-anchored)
# ============================================================================

def _noise_residual(gray):
    """Extract noise residual via median filter subtraction."""
    denoised = cv2.medianBlur(gray, 5)
    return gray.astype(np.float64) - denoised.astype(np.float64)


def extract_noise_physics(frame_gray, face_mask):
    """P1: Noise Physics — VMR, ResStd, HFRatio in face ROI."""
    roi = frame_gray[face_mask > 0].astype(np.float64)
    if len(roi) < 100:
        return 0.0, 0.0, 0.0
    
    mean_val = roi.mean()
    var_val = roi.var()
    vmr = var_val / (mean_val + 1e-8)
    
    residual = _noise_residual(frame_gray)
    res_roi = residual[face_mask > 0]
    res_std = res_roi.std()
    
    f = np.fft.fft2(residual * face_mask)
    fshift = np.fft.fftshift(f)
    mag = np.abs(fshift)
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    r = min(h, w) // 4
    total_energy = mag.sum() + 1e-8
    center_mask = np.zeros_like(mag)
    cv2.circle(center_mask, (cx, cy), r, 1, -1)
    lf_energy = (mag * center_mask).sum()
    hf_ratio = 1.0 - (lf_energy / total_energy)
    
    return vmr, res_std, hf_ratio


def extract_prnu(frame_gray, face_mask, bg_mask):
    """P2: PRNU — Energy in face, Face/Periph ratio."""
    residual = _noise_residual(frame_gray)
    
    face_res = residual[face_mask > 0]
    prnu_energy = np.mean(face_res ** 2) if len(face_res) > 50 else 0.0
    
    bg_res = residual[bg_mask > 0]
    bg_energy = np.mean(bg_res ** 2) if len(bg_res) > 50 else 1e-8
    face_periph = prnu_energy / (bg_energy + 1e-8)
    
    return prnu_energy, face_periph


def extract_shadow_light(frame_gray, face_mask, landmarks, shape):
    """P4: Shadow/Light — shadow score, face-bg diff."""
    face_pixels = frame_gray[face_mask > 0].astype(np.float64)
    if len(face_pixels) < 100:
        return 0.0, 0.0
    
    face_mean = face_pixels.mean()
    face_std = face_pixels.std()
    shadow_score = face_std / (face_mean + 1e-8)
    
    bg_mask = 1 - face_mask
    bg_pixels = frame_gray[bg_mask > 0].astype(np.float64)
    bg_mean = bg_pixels.mean() if len(bg_pixels) > 50 else face_mean
    face_bg_diff = abs(face_mean - bg_mean) / (255.0 + 1e-8)
    
    return shadow_score, face_bg_diff


def extract_compression(frame_gray, face_mask):
    """P6: Compression — Benford deviation, block artifacts, double compression."""
    roi = frame_gray[face_mask > 0].astype(np.float64)
    if len(roi) < 100:
        return 0.0, 0.0, 0.0
    
    # Benford's Law on DCT coefficients
    h, w = frame_gray.shape
    bh, bw = (h // 8) * 8, (w // 8) * 8
    block_img = frame_gray[:bh, :bw].astype(np.float64)
    dct_coeffs = []
    for i in range(0, bh, 8):
        for j in range(0, bw, 8):
            block = block_img[i:i+8, j:j+8]
            dct_block = cv2.dct(block)
            coeffs = dct_block.flatten()[1:]
            dct_coeffs.extend(coeffs[coeffs != 0])
    
    if len(dct_coeffs) < 100:
        benford_dev = 0.0
    else:
        first_digits = np.abs(np.array(dct_coeffs))
        first_digits = first_digits[first_digits >= 1]
        if len(first_digits) > 0:
            fd = (first_digits / (10 ** np.floor(np.log10(first_digits + 1e-10)))).astype(int)
            fd = fd[(fd >= 1) & (fd <= 9)]
            if len(fd) > 0:
                observed = np.bincount(fd, minlength=10)[1:] / len(fd)
                expected = np.log10(1 + 1.0 / np.arange(1, 10))
                benford_dev = np.sqrt(np.mean((observed - expected) ** 2))
            else:
                benford_dev = 0.0
        else:
            benford_dev = 0.0
    
    # Block artifact: measure discontinuity at 8x8 block boundaries
    if bh > 16 and bw > 16:
        h1 = block_img[7::8, :]
        h2 = block_img[8::8, :]
        min_h = min(h1.shape[0], h2.shape[0])
        h_boundaries = h1[:min_h] - h2[:min_h]
        v1 = block_img[:, 7::8]
        v2 = block_img[:, 8::8]
        min_v = min(v1.shape[1], v2.shape[1])
        v_boundaries = v1[:, :min_v] - v2[:, :min_v]
        block_art = (np.abs(h_boundaries).mean() + np.abs(v_boundaries).mean()) / 2.0
    else:
        block_art = 0.0
    
    # Double compression: histogram of DCT coefficient (1,1) position
    ac11_vals = []
    for i in range(0, bh, 8):
        for j in range(0, bw, 8):
            block = block_img[i:i+8, j:j+8]
            dct_block = cv2.dct(block)
            ac11_vals.append(dct_block[1, 1])
    
    if len(ac11_vals) > 50:
        hist, _ = np.histogram(ac11_vals, bins=50)
        hist = hist / (hist.sum() + 1e-8)
        peaks = sp_signal.find_peaks(hist, height=0.02)[0]
        dbl_compress = len(peaks) / 10.0
    else:
        dbl_compress = 0.0
    
    return benford_dev, block_art, dbl_compress


def extract_blur(frame_gray, face_mask):
    """P8: Motion Blur — Laplacian variance in face ROI."""
    masked = frame_gray.copy()
    masked[face_mask == 0] = 0
    lap = cv2.Laplacian(masked, cv2.CV_64F)
    face_lap = lap[face_mask > 0]
    if len(face_lap) < 50:
        return 0.0
    return face_lap.var()


def extract_optical_flow(prev_gray, curr_gray, face_mask):
    """P9: Optical Flow — magnitude, direction consistency in face ROI."""
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )
    
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    face_mag = mag[face_mask > 0]
    face_ang = ang[face_mask > 0]
    
    if len(face_mag) < 50:
        return 0.0, 0.0
    
    flow_mag = face_mag.mean()
    
    # Direction consistency: circular variance (low = consistent direction)
    sin_sum = np.sin(face_ang).mean()
    cos_sum = np.cos(face_ang).mean()
    R = np.sqrt(sin_sum**2 + cos_sum**2)
    dir_consist = R  # 1 = all same direction, 0 = random
    
    return flow_mag, dir_consist


def extract_spatial_features_single_frame(frame_bgr, frame_gray, prev_gray,
                                           landmarks, face_mask):
    """Extract all spatial features for a single frame."""
    h, w = frame_gray.shape
    bg_mask = (1 - face_mask).astype(np.uint8)
    
    vmr, res_std, hf_ratio = extract_noise_physics(frame_gray, face_mask)
    prnu_e, prnu_fp = extract_prnu(frame_gray, face_mask, bg_mask)
    shadow, fb_diff = extract_shadow_light(frame_gray, face_mask, landmarks, (h, w))
    benford, block_a, dbl_c = extract_compression(frame_gray, face_mask)
    blur = extract_blur(frame_gray, face_mask)
    
    if prev_gray is not None:
        flow_m, flow_d = extract_optical_flow(prev_gray, frame_gray, face_mask)
    else:
        flow_m, flow_d = 0.0, 0.0
    
    return np.array([
        vmr, res_std, hf_ratio,
        prnu_e, prnu_fp,
        shadow, fb_diff,
        benford, block_a, dbl_c,
        blur,
        flow_m, flow_d,
    ], dtype=np.float64)


# ============================================================================
# TEMPORAL PILLAR EXTRACTORS (across-frame consistency)
# ============================================================================

def extract_temporal_noise_stability(frames_gray, face_masks):
    """T1: Temporal Noise Stability — 3 features."""
    n = len(frames_gray)
    if n < MIN_FRAMES_TEMPORAL:
        return np.array([0.0, 0.0, 0.0])
    
    residuals = []
    for g, m in zip(frames_gray, face_masks):
        res = _noise_residual(g)
        residuals.append(res * m)
    
    step = max(1, n // 30)
    correlations = []
    for i in range(0, n - step, step):
        r1 = residuals[i].flatten()
        r2 = residuals[i + step].flatten()
        valid = (r1 != 0) & (r2 != 0)
        if valid.sum() > 200:
            c = np.corrcoef(r1[valid], r2[valid])[0, 1]
            if np.isfinite(c):
                correlations.append(c)
    
    if len(correlations) < 3:
        return np.array([0.0, 0.0, 0.0])
    
    corr_mean = np.mean(correlations)
    corr_std = np.std(correlations)
    
    # Spectral entropy of temporal noise at sample pixels
    sample_pixels = []
    mask_coords = np.argwhere(face_masks[0] > 0)
    if len(mask_coords) > 100:
        idx = np.random.choice(len(mask_coords), 100, replace=False)
        for ci in idx:
            y, x = mask_coords[ci]
            ts = [residuals[t][y, x] for t in range(n)]
            sample_pixels.append(ts)
    
    if len(sample_pixels) > 0:
        pixel_ts = np.array(sample_pixels)
        entropies = []
        for ts in pixel_ts:
            f = np.fft.rfft(ts)
            psd = np.abs(f) ** 2
            psd = psd / (psd.sum() + 1e-10)
            psd = psd[psd > 0]
            ent = -np.sum(psd * np.log2(psd + 1e-10))
            entropies.append(ent)
        spectral_entropy = np.mean(entropies)
    else:
        spectral_entropy = 0.0
    
    return np.array([corr_mean, corr_std, spectral_entropy])


def extract_rppg(frames_rgb, landmarks_list, fps):
    """T2: rPPG Cardiac Signal — 4 features.
    
    HARDENED for H.264 compressed video (CelebDF-v2):
      1. Bilateral filter preprocessing to suppress codec block noise
         while preserving the subtle skin color oscillation
      2. Dual-algorithm: POS (primary, more compression-robust) + CHROM (fallback)
      3. Longer temporal windows (use all available frames, overlap segments)
      4. Per-segment quality gating: discard low-quality segments
      5. Windowed overlap extraction for more robust spectral estimation
    
    POS (Plane Orthogonal to Skin) is more robust to compression than CHROM
    because it projects onto a plane orthogonal to the skin-tone vector,
    which is less affected by uniform compression artifacts.
    """
    n = len(frames_rgb)
    if n < MIN_FRAMES_RPPG:
        return np.array([0.0, 0.0, 0.0, 0.0])
    
    # ------------------------------------------------------------------
    # STEP 1: Preprocessing — bilateral filter each frame to suppress
    # H.264 block artifacts WITHOUT destroying the cardiac color signal.
    # Bilateral preserves edges (skin boundaries) while smoothing blocks.
    # ------------------------------------------------------------------
    def preprocess_for_rppg(frame_rgb):
        return cv2.bilateralFilter(frame_rgb, d=7, sigmaColor=25, sigmaSpace=25)
    
    # ------------------------------------------------------------------
    # STEP 2: Extract ROI signals from 3 independent skin regions
    # Using landmark-defined ROIs for forehead, left cheek, right cheek
    # ------------------------------------------------------------------
    forehead_signals = []
    lcheek_signals = []
    rcheek_signals = []
    valid_frame_count = 0
    
    for frame, lm in zip(frames_rgb, landmarks_list):
        if lm is None:
            forehead_signals.append([0, 0, 0])
            lcheek_signals.append([0, 0, 0])
            rcheek_signals.append([0, 0, 0])
            continue
        
        clean_frame = preprocess_for_rppg(frame)
        forehead_signals.append(get_roi_mean_rgb(clean_frame, lm, FOREHEAD))
        lcheek_signals.append(get_roi_mean_rgb(clean_frame, lm, LEFT_CHEEK))
        rcheek_signals.append(get_roi_mean_rgb(clean_frame, lm, RIGHT_CHEEK))
        valid_frame_count += 1
    
    if valid_frame_count < MIN_FRAMES_RPPG:
        return np.array([0.0, 0.0, 0.0, 0.0])
    
    # ------------------------------------------------------------------
    # STEP 3: POS algorithm (primary) + CHROM (fallback)
    # POS: Wang et al., "Algorithmic Principles of Remote PPG" (2017)
    # More robust to illumination changes and compression artifacts.
    # ------------------------------------------------------------------
    def _bandpass(signal_1d, fps_val, low_hz=0.7, high_hz=4.0):
        """Bandpass filter with safety checks."""
        nyq = fps_val / 2.0
        if nyq <= high_hz or len(signal_1d) < 20:
            return None
        lo = low_hz / nyq
        hi = min(high_hz / nyq, 0.99)
        if lo >= hi:
            return None
        b, a = sp_signal.butter(3, [lo, hi], btype='band')
        try:
            return sp_signal.filtfilt(b, a, signal_1d)
        except Exception:
            return None
    
    def pos_rppg(rgb_signals, fps_val, window_sec=1.6):
        """POS (Plane Orthogonal to Skin) rPPG extraction.
        More robust to compression than CHROM because it uses a 
        temporal normalization window that adapts to local color changes.
        """
        sig = np.array(rgb_signals, dtype=np.float64)
        if sig.shape[0] < 30 or sig.max() < 1:
            return None
        
        N = sig.shape[0]
        win_len = int(window_sec * fps_val)
        win_len = max(20, min(win_len, N))
        
        pulse = np.zeros(N)
        
        for t in range(0, N - win_len + 1):
            segment = sig[t:t + win_len]
            
            # Temporal normalization (key to compression robustness)
            means = segment.mean(axis=0, keepdims=True)
            means[means < 1] = 1
            Cn = segment / means
            
            # POS projection
            S1 = Cn[:, 1] - Cn[:, 2]    # G - B
            S2 = Cn[:, 1] + Cn[:, 2] - 2 * Cn[:, 0]  # G + B - 2R
            
            std_s1 = S1.std()
            std_s2 = S2.std()
            if std_s2 < 1e-10:
                continue
            alpha = std_s1 / std_s2
            h = S1 + alpha * S2
            
            # Overlap-add
            pulse[t:t + win_len] += h - h.mean()
        
        # Normalize by overlap count
        overlap_count = np.minimum(
            np.arange(1, N + 1),
            np.minimum(win_len, np.arange(N, 0, -1))
        ).astype(np.float64)
        overlap_count[overlap_count < 1] = 1
        pulse /= overlap_count
        
        # Bandpass filter
        filtered = _bandpass(pulse, fps_val)
        return filtered
    
    def chrom_rppg(rgb_signals, fps_val):
        """CHROM rPPG extraction (fallback)."""
        sig = np.array(rgb_signals, dtype=np.float64)
        if sig.shape[0] < 30 or sig.max() < 1:
            return None
        
        means = sig.mean(axis=0, keepdims=True)
        means[means < 1] = 1
        sig_n = sig / means
        
        Xs = 3 * sig_n[:, 0] - 2 * sig_n[:, 1]
        Ys = 1.5 * sig_n[:, 0] + sig_n[:, 1] - 1.5 * sig_n[:, 2]
        
        std_y = Ys.std()
        if std_y < 1e-10:
            return None
        alpha = Xs.std() / std_y
        pulse = Xs - alpha * Ys
        
        filtered = _bandpass(pulse, fps_val)
        return filtered
    
    def compute_spectral_features(pulse_signal, fps_val):
        """Compute spectral features from a filtered pulse signal."""
        if pulse_signal is None or len(pulse_signal) < 20:
            return None, None, None, None
        
        # Welch's method for smoother PSD (more robust to noise)
        nperseg = min(len(pulse_signal), int(4 * fps_val))
        if nperseg < 16:
            nperseg = len(pulse_signal)
        
        try:
            freqs, psd = sp_signal.welch(
                pulse_signal, fs=fps_val,
                nperseg=nperseg, noverlap=nperseg // 2,
                window='hann'
            )
        except Exception:
            return None, None, None, None
        
        cardiac_mask = (freqs >= 0.7) & (freqs <= 4.0)
        cardiac_psd = psd[cardiac_mask]
        cardiac_freqs = freqs[cardiac_mask]
        
        if len(cardiac_psd) < 3:
            return None, None, None, None
        
        # SNR: peak power / mean power
        snr = cardiac_psd.max() / (cardiac_psd.mean() + 1e-10)
        
        # Peak prominence
        peaks, _ = sp_signal.find_peaks(cardiac_psd, distance=2)
        if len(peaks) > 0:
            prominences = sp_signal.peak_prominences(cardiac_psd, peaks)[0]
            peak_prom = prominences.max() / (cardiac_psd.mean() + 1e-10)
        else:
            peak_prom = 0.0
        
        # Harmonic ratio
        if len(peaks) >= 2:
            sorted_pk = peaks[np.argsort(cardiac_psd[peaks])[::-1]]
            fund = cardiac_psd[sorted_pk[0]]
            harm = cardiac_psd[sorted_pk[1]]
            harmonic = fund / (harm + 1e-10)
        else:
            harmonic = 0.0
        
        return snr, peak_prom, harmonic, pulse_signal
    
    # ------------------------------------------------------------------
    # STEP 4: Extract from each region using POS (primary) then CHROM
    # ------------------------------------------------------------------
    def extract_best_pulse(rgb_signals, fps_val):
        """Try POS first, fall back to CHROM, return best signal."""
        pos_pulse = pos_rppg(rgb_signals, fps_val)
        chrom_pulse = chrom_rppg(rgb_signals, fps_val)
        
        # Pick the one with higher cardiac-band SNR
        pos_feats = compute_spectral_features(pos_pulse, fps_val)
        chrom_feats = compute_spectral_features(chrom_pulse, fps_val)
        
        pos_snr = pos_feats[0] if pos_feats[0] is not None else 0
        chrom_snr = chrom_feats[0] if chrom_feats[0] is not None else 0
        
        if pos_snr >= chrom_snr and pos_feats[3] is not None:
            return pos_feats
        elif chrom_feats[3] is not None:
            return chrom_feats
        elif pos_feats[3] is not None:
            return pos_feats
        else:
            return (None, None, None, None)
    
    forehead_feats = extract_best_pulse(forehead_signals, fps)
    cheek_combined = (np.array(lcheek_signals) + np.array(rcheek_signals)) / 2.0
    cheek_feats = extract_best_pulse(cheek_combined.tolist(), fps)
    
    snr_fh, prom_fh, harm_fh, pulse_fh = forehead_feats
    snr_ck, prom_ck, harm_ck, pulse_ck = cheek_feats
    
    # ------------------------------------------------------------------
    # STEP 5: Signal quality gating
    # If SNR < 1.5 in BOTH regions, the rPPG signal is unreliable.
    # In this case we output features that explicitly signal "no pulse found"
    # rather than outputting noisy garbage that would hurt classification.
    # ------------------------------------------------------------------
    QUALITY_THRESHOLD = 1.5
    fh_ok = snr_fh is not None and snr_fh > QUALITY_THRESHOLD
    ck_ok = snr_ck is not None and snr_ck > QUALITY_THRESHOLD
    
    if not fh_ok and not ck_ok:
        # No reliable cardiac signal — output discriminative zeros
        # For real video: rPPG should be extractable → absence is suspicious
        # For fake video: rPPG absence is expected
        # A zero vector here IS informative — the classifier can learn that
        # "rPPG extraction failed" itself correlates with fake content.
        return np.array([0.0, 0.0, 0.0, 0.0])
    
    # Use best available SNR
    final_snr = max(snr_fh or 0, snr_ck or 0)
    final_prom = max(prom_fh or 0, prom_ck or 0)
    final_harm = max(harm_fh or 0, harm_ck or 0)
    
    # ------------------------------------------------------------------
    # STEP 6: Inter-region correlation (most compression-robust feature)
    # This feature survives compression because it measures RELATIVE
    # consistency between two regions, not absolute signal quality.
    # Even if compression attenuates the cardiac signal, the correlation
    # between forehead and cheek should persist for real video (same heart)
    # and be absent for fake video (no real physiological coupling).
    # ------------------------------------------------------------------
    interregion_corr = 0.0
    if pulse_fh is not None and pulse_ck is not None:
        min_len = min(len(pulse_fh), len(pulse_ck))
        if min_len > 20:
            # Compute correlation on overlapping windowed segments
            # for more robust estimate
            window = min(90, min_len)
            step = max(1, window // 3)
            segment_corrs = []
            
            for start in range(0, min_len - window + 1, step):
                seg_fh = pulse_fh[start:start + window]
                seg_ck = pulse_ck[start:start + window]
                if seg_fh.std() > 1e-8 and seg_ck.std() > 1e-8:
                    c = np.corrcoef(seg_fh, seg_ck)[0, 1]
                    if np.isfinite(c):
                        segment_corrs.append(c)
            
            if segment_corrs:
                # Use median (robust to outlier segments from motion)
                interregion_corr = np.median(segment_corrs)
            else:
                # Direct correlation as fallback
                c = np.corrcoef(pulse_fh[:min_len], pulse_ck[:min_len])[0, 1]
                interregion_corr = c if np.isfinite(c) else 0.0
    
    return np.array([final_snr, final_prom, interregion_corr, final_harm])


def extract_temporal_prnu(frames_gray, face_masks, bg_masks):
    """T3: Temporal PRNU Persistence — 2 features."""
    n = len(frames_gray)
    if n < MIN_FRAMES_TEMPORAL:
        return np.array([0.0, 0.0])
    
    residuals = [_noise_residual(g) for g in frames_gray]
    
    gaps = [5, 10, min(30, n // 3)]
    face_corrs = []
    bg_corrs = []
    
    for gap in gaps:
        for i in range(0, n - gap, max(1, gap)):
            r1_face = residuals[i][face_masks[i] > 0]
            r2_face = residuals[i + gap][face_masks[i + gap] > 0]
            min_len = min(len(r1_face), len(r2_face))
            if min_len > 200:
                c = np.corrcoef(r1_face[:min_len], r2_face[:min_len])[0, 1]
                if np.isfinite(c):
                    face_corrs.append(c)
            
            r1_bg = residuals[i][bg_masks[i] > 0]
            r2_bg = residuals[i + gap][bg_masks[i + gap] > 0]
            min_len_bg = min(len(r1_bg), len(r2_bg))
            if min_len_bg > 200:
                c_bg = np.corrcoef(r1_bg[:min_len_bg], r2_bg[:min_len_bg])[0, 1]
                if np.isfinite(c_bg):
                    bg_corrs.append(c_bg)
    
    prnu_stability = np.mean(face_corrs) if face_corrs else 0.0
    bg_stability = np.mean(bg_corrs) if bg_corrs else 1e-8
    face_vs_bg = prnu_stability / (bg_stability + 1e-8)
    
    return np.array([prnu_stability, face_vs_bg])


def extract_face_structural_stability(frames_gray, landmarks_list):
    """T4: Face Structural Stability — SSIM across frames (physics-based)."""
    n = len(frames_gray)
    if n < MIN_FRAMES_TEMPORAL:
        return np.array([0.0, 0.0, 0.0])
    
    # Crop and resize face to fixed template for comparison
    face_crops = []
    target_size = (64, 64)
    
    for g, lm in zip(frames_gray, landmarks_list):
        if lm is None:
            continue
        x1, y1, x2, y2 = get_face_bbox(lm, g.shape, padding=0.05)
        crop = g[y1:y2, x1:x2]
        if crop.size < 100:
            continue
        crop_resized = cv2.resize(crop, target_size)
        face_crops.append(crop_resized.astype(np.float64))
    
    if len(face_crops) < 10:
        return np.array([0.0, 0.0, 0.0])
    
    # SSIM between consecutive pairs
    ssim_values = []
    for i in range(len(face_crops) - 1):
        c1, c2 = face_crops[i], face_crops[i + 1]
        mu1, mu2 = c1.mean(), c2.mean()
        s1, s2 = c1.std(), c2.std()
        cov = np.mean((c1 - mu1) * (c2 - mu2))
        C1, C2 = 6.5025, 58.5225
        ssim = ((2*mu1*mu2+C1)*(2*cov+C2)) / ((mu1**2+mu2**2+C1)*(s1**2+s2**2+C2))
        ssim_values.append(ssim)
    
    return np.array([np.mean(ssim_values), np.std(ssim_values), np.min(ssim_values)])


def extract_codec_temporal_residual(frames_gray, face_masks):
    """T5: Codec Temporal Residual — 2 features.
    Approximates prediction residual using frame differencing.
    In real video, large differences correlate with motion; in fake, they don't.
    """
    n = len(frames_gray)
    if n < MIN_FRAMES_TEMPORAL:
        return np.array([0.0, 0.0])
    
    residual_mags = []
    flow_mags = []
    
    for i in range(1, n, 2):
        diff = np.abs(frames_gray[i].astype(np.float64) - frames_gray[i-1].astype(np.float64))
        mask = face_masks[i] if i < len(face_masks) else face_masks[-1]
        res = diff[mask > 0].mean()
        residual_mags.append(res)
        
        flow = cv2.calcOpticalFlowFarneback(
            frames_gray[i-1], frames_gray[i], None,
            0.5, 2, 10, 2, 5, 1.1, 0
        )
        mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        fm = mag[mask > 0].mean()
        flow_mags.append(fm)
    
    if len(residual_mags) < 5:
        return np.array([0.0, 0.0])
    
    r = np.array(residual_mags)
    f = np.array(flow_mags)
    
    corr = np.corrcoef(r, f)[0, 1] if f.std() > 1e-8 else 0.0
    if not np.isfinite(corr):
        corr = 0.0
    
    # Temporal entropy of residuals
    r_norm = r / (r.max() + 1e-10)
    hist, _ = np.histogram(r_norm, bins=20, density=True)
    hist = hist / (hist.sum() + 1e-10)
    hist = hist[hist > 0]
    entropy = -np.sum(hist * np.log2(hist + 1e-10))
    
    return np.array([corr, entropy])


def extract_landmark_trajectory(landmarks_list, fps):
    """T6: Landmark Trajectory Smoothness — 4 features."""
    valid = [lm for lm in landmarks_list if lm is not None]
    n = len(valid)
    if n < MIN_FRAMES_TEMPORAL:
        return np.array([0.0, 0.0, 0.0, 0.0])
    
    # Use a subset of stable landmarks
    key_lm = FACE_OVAL[:20] + NOSE_BRIDGE + [33, 133, 362, 263]
    
    # Normalize by face size each frame
    positions = []
    for lm in valid:
        fsize = get_face_size(lm)
        if fsize < 1:
            fsize = 1
        positions.append(lm[key_lm] / fsize)
    
    positions = np.array(positions)  # (n, num_lm, 2)
    
    # Velocity: frame-to-frame displacement
    velocity = np.diff(positions, axis=0)  # (n-1, num_lm, 2)
    vel_mag = np.linalg.norm(velocity, axis=2)  # (n-1, num_lm)
    
    # Jitter: mean displacement magnitude
    jitter = vel_mag.mean()
    
    # Acceleration variance
    if velocity.shape[0] > 2:
        accel = np.diff(velocity, axis=0)
        accel_mag = np.linalg.norm(accel, axis=2)
        accel_var = accel_mag.var()
    else:
        accel_var = 0.0
    
    # Velocity autocorrelation (lag-1)
    vel_flat = vel_mag.mean(axis=1)  # average over landmarks
    if len(vel_flat) > 5:
        autocorr = np.corrcoef(vel_flat[:-1], vel_flat[1:])[0, 1]
        if not np.isfinite(autocorr):
            autocorr = 0.0
    else:
        autocorr = 0.0
    
    # Jaw-chin rigidity
    jaw_idx = [152, 148, 176, 377, 400, 378]
    chin_idx = [152, 175]
    available_jaw = [i for i in jaw_idx if i < valid[0].shape[0]]
    available_chin = [i for i in chin_idx if i < valid[0].shape[0]]
    
    if len(available_jaw) >= 2 and len(available_chin) >= 1:
        jaw_motion = []
        chin_motion = []
        for i in range(1, n):
            jm = np.linalg.norm(valid[i][available_jaw] - valid[i-1][available_jaw])
            cm = np.linalg.norm(valid[i][available_chin] - valid[i-1][available_chin])
            jaw_motion.append(jm)
            chin_motion.append(cm)
        jaw_motion = np.array(jaw_motion)
        chin_motion = np.array(chin_motion)
        if chin_motion.std() > 1e-8:
            rigidity = np.corrcoef(jaw_motion, chin_motion)[0, 1]
            if not np.isfinite(rigidity):
                rigidity = 0.0
        else:
            rigidity = 1.0
    else:
        rigidity = 0.0
    
    return np.array([jitter, accel_var, autocorr, rigidity])


def extract_rigid_geometry(landmarks_list):
    """T7: Rigid Geometry Consistency — 3 features."""
    valid = [lm for lm in landmarks_list if lm is not None]
    n = len(valid)
    if n < MIN_FRAMES_TEMPORAL:
        return np.array([0.0, 0.0, 0.0])
    
    interpupillary = []
    nose_bridge_len = []
    face_widths = []
    
    for lm in valid:
        fsize = get_face_size(lm)
        if fsize < 1:
            fsize = 1
        
        # Inter-pupillary distance (normalized)
        l_eye_center = lm[[33, 133]].mean(axis=0)
        r_eye_center = lm[[362, 263]].mean(axis=0)
        ipd = np.linalg.norm(l_eye_center - r_eye_center) / fsize
        interpupillary.append(ipd)
        
        # Nose bridge length (normalized)
        nb = np.linalg.norm(lm[168] - lm[1]) / fsize
        nose_bridge_len.append(nb)
        
        # Face width at cheekbones
        fw = np.linalg.norm(lm[234] - lm[454]) / fsize
        face_widths.append(fw)
    
    interpupillary = np.array(interpupillary)
    nose_bridge_len = np.array(nose_bridge_len)
    face_widths = np.array(face_widths)
    
    # Rigid distances should have near-zero variance
    rigid_vars = [interpupillary.std(), nose_bridge_len.std(), face_widths.std()]
    rigid_dist_var = np.mean(rigid_vars)
    
    return np.array([rigid_dist_var, interpupillary.std(), nose_bridge_len.std()])


def extract_boundary_coherence(frames_gray, frames_bgr, landmarks_list):
    """T8: Face-BG Edge Coherence — 3 features (temporal variance)."""
    n = len(frames_gray)
    if n < MIN_FRAMES_TEMPORAL:
        return np.array([0.0, 0.0, 0.0])
    
    grad_means = []
    color_discs = []
    freq_leakages = []
    
    for i in range(0, n, max(1, n // 30)):
        lm = landmarks_list[i]
        if lm is None:
            continue
        
        gray = frames_gray[i]
        bgr = frames_bgr[i]
        h, w = gray.shape
        
        # Face boundary mask (dilated - eroded = boundary band)
        face_mask = get_face_mask(lm, (h, w))
        kernel = np.ones((7, 7), np.uint8)
        dilated = cv2.dilate(face_mask, kernel)
        eroded = cv2.erode(face_mask, kernel)
        boundary = dilated - eroded
        
        # Gradient at boundary
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        bnd_grad = grad_mag[boundary > 0]
        if len(bnd_grad) > 10:
            grad_means.append(bnd_grad.mean())
        
        # Color discontinuity in LAB space
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float64)
        inner = lab[eroded > 0].mean(axis=0) if eroded.sum() > 10 else np.zeros(3)
        outer_mask = dilated - face_mask
        outer = lab[outer_mask > 0].mean(axis=0) if outer_mask.sum() > 10 else inner
        color_disc = np.linalg.norm(inner - outer)
        color_discs.append(color_disc)
        
        # High-frequency energy at boundary vs inner face
        f = np.fft.fft2(gray.astype(np.float64) * boundary)
        bnd_hf = np.abs(f).sum()
        f_inner = np.fft.fft2(gray.astype(np.float64) * eroded)
        inner_hf = np.abs(f_inner).sum()
        if inner_hf > 1e-8:
            freq_leakages.append(bnd_hf / inner_hf)
    
    return np.array([
        np.std(grad_means) if grad_means else 0.0,
        np.std(color_discs) if color_discs else 0.0,
        np.std(freq_leakages) if freq_leakages else 0.0,
    ])


def extract_skin_texture(frames_gray, landmarks_list):
    """T9: Skin Texture Coherence — 2 features."""
    n = len(frames_gray)
    if n < MIN_FRAMES_TEMPORAL:
        return np.array([0.0, 0.0])
    
    # Extract cheek texture patches, aligned by landmarks
    patches = []
    target_size = (32, 32)
    
    for g, lm in zip(frames_gray, landmarks_list):
        if lm is None:
            continue
        # Right cheek patch anchored to landmarks
        cheek_center = lm[RIGHT_CHEEK].mean(axis=0).astype(int)
        y, x = cheek_center[1], cheek_center[0]
        half = 16
        y1, y2 = max(0, y-half), min(g.shape[0], y+half)
        x1, x2 = max(0, x-half), min(g.shape[1], x+half)
        patch = g[y1:y2, x1:x2]
        if patch.shape[0] < 8 or patch.shape[1] < 8:
            continue
        patch_resized = cv2.resize(patch, target_size).astype(np.float64)
        patches.append(patch_resized)
    
    if len(patches) < 10:
        return np.array([0.0, 0.0])
    
    corrs = []
    residuals = []
    for i in range(len(patches) - 1):
        p1, p2 = patches[i], patches[i + 1]
        c = np.corrcoef(p1.flatten(), p2.flatten())[0, 1]
        if np.isfinite(c):
            corrs.append(c)
        res = np.mean(np.abs(p1 - p2))
        residuals.append(res)
    
    texture_corr = np.mean(corrs) if corrs else 0.0
    warp_residual = np.mean(residuals) if residuals else 0.0
    
    return np.array([texture_corr, warp_residual])


def extract_color_transfer(frames_bgr, landmarks_list, face_masks):
    """T10: Color Transfer Consistency — 2 features."""
    n = len(frames_bgr)
    if n < MIN_FRAMES_TEMPORAL:
        return np.array([0.0, 0.0])
    
    face_colors = []
    bg_colors = []
    
    for bgr, lm, mask in zip(frames_bgr, landmarks_list, face_masks):
        if lm is None:
            continue
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float64)
        
        face_pix = lab[mask > 0]
        bg_pix = lab[(1 - mask) > 0]
        
        if len(face_pix) > 50:
            face_colors.append(face_pix.mean(axis=0))
        if len(bg_pix) > 50:
            bg_colors.append(bg_pix.mean(axis=0))
    
    if len(face_colors) < 10:
        return np.array([0.0, 0.0])
    
    face_colors = np.array(face_colors)
    bg_colors = np.array(bg_colors)
    
    # Remove global illumination trend (linear detrend)
    face_detrended = sp_signal.detrend(face_colors, axis=0)
    
    # Skin color jitter: variance after detrending
    skin_jitter = np.mean(face_detrended.std(axis=0))
    
    # Skin-BG decorrelation
    min_len = min(len(face_colors), len(bg_colors))
    if min_len > 5:
        face_diff = np.diff(face_colors[:min_len], axis=0)
        bg_diff = np.diff(bg_colors[:min_len], axis=0)
        corrs = []
        for ch in range(3):
            c = np.corrcoef(face_diff[:, ch], bg_diff[:, ch])[0, 1]
            if np.isfinite(c):
                corrs.append(c)
        decorrelation = 1.0 - np.mean(corrs) if corrs else 0.0
    else:
        decorrelation = 0.0
    
    return np.array([skin_jitter, decorrelation])


def extract_specular_temporal(frames_gray, landmarks_list):
    """T11: Specular Reflection Temporal — 2 features."""
    n = len(frames_gray)
    if n < MIN_FRAMES_TEMPORAL:
        return np.array([0.0, 0.0])
    
    # Detect specular highlights as top 1% brightest pixels in face
    highlight_positions = []
    symmetry_scores = []
    
    for g, lm in zip(frames_gray, landmarks_list):
        if lm is None:
            continue
        
        face_mask = get_face_mask(lm, g.shape)
        face_pix = g.copy()
        face_pix[face_mask == 0] = 0
        
        threshold = np.percentile(g[face_mask > 0], 98) if face_mask.sum() > 100 else 200
        highlight_mask = (face_pix > threshold) & (face_mask > 0)
        
        coords = np.argwhere(highlight_mask)
        if len(coords) > 5:
            centroid = coords.mean(axis=0)
            face_center = lm[NOSE_BRIDGE].mean(axis=0)[[1, 0]]
            fsize = get_face_size(lm)
            rel_pos = (centroid - face_center) / (fsize + 1e-8)
            highlight_positions.append(rel_pos)
            
            # Bilateral symmetry of highlights
            nose_x = lm[1, 0]
            left_hl = coords[coords[:, 1] < nose_x]
            right_hl = coords[coords[:, 1] >= nose_x]
            sym = 1.0 - abs(len(left_hl) - len(right_hl)) / (len(coords) + 1e-8)
            symmetry_scores.append(sym)
    
    if len(highlight_positions) < 5:
        return np.array([0.0, 0.0])
    
    positions = np.array(highlight_positions)
    stability = 1.0 / (positions.std(axis=0).mean() + 1e-8)
    symmetry = np.mean(symmetry_scores) if symmetry_scores else 0.0
    
    return np.array([stability, symmetry])


def extract_blink_dynamics(landmarks_list, fps):
    """T12: Blink Dynamics — 3 features."""
    n = len(landmarks_list)
    if n < MIN_FRAMES_TEMPORAL:
        return np.array([0.0, 0.0, 0.0])
    
    ear_left = []
    ear_right = []
    
    for lm in landmarks_list:
        if lm is None:
            ear_left.append(0.3)
            ear_right.append(0.3)
            continue
        ear_left.append(compute_ear(lm, LEFT_EYE))
        ear_right.append(compute_ear(lm, RIGHT_EYE))
    
    ear_left = np.array(ear_left)
    ear_right = np.array(ear_right)
    ear_avg = (ear_left + ear_right) / 2.0
    
    # Detect blinks (EAR drops below threshold)
    threshold = 0.19
    blink_frames = ear_avg < threshold
    
    # Find blink events (contiguous below-threshold segments)
    blinks = []
    in_blink = False
    start = 0
    for i in range(len(blink_frames)):
        if blink_frames[i] and not in_blink:
            in_blink = True
            start = i
        elif not blink_frames[i] and in_blink:
            in_blink = False
            blinks.append((start, i))
    
    # Filter out false blinks: real blinks last 3-13 frames (100-430ms at 30fps)
    min_frames = max(3, int(0.1 * fps))
    max_frames_blink = max(13, int(0.43 * fps))
    blinks = [(s, e) for s, e in blinks if min_frames <= (e - s) <= max_frames_blink]
    
    duration_seconds = n / fps
    blink_rate = len(blinks) / (duration_seconds / 60.0) if duration_seconds > 1 else 0.0
    
    if len(blinks) > 0:
        durations = [(end - start) / fps * 1000 for start, end in blinks]
        blink_duration = np.mean(durations)
    else:
        blink_duration = 0.0
    
    # Blink symmetry: correlation of left and right EAR
    if ear_left.std() > 1e-6 and ear_right.std() > 1e-6:
        blink_sym = np.corrcoef(ear_left, ear_right)[0, 1]
        if not np.isfinite(blink_sym):
            blink_sym = 0.0
    else:
        blink_sym = 1.0
    
    return np.array([blink_rate, blink_duration, blink_sym])


def extract_motion_blur_coupling(frames_gray, face_masks):
    """T13: Motion-Blur Coupling — 2 features."""
    n = len(frames_gray)
    if n < MIN_FRAMES_TEMPORAL:
        return np.array([0.0, 0.0])
    
    couplings = []
    
    for i in range(1, n, 3):
        mask = face_masks[min(i, len(face_masks) - 1)]
        
        # Local blur estimate (Laplacian variance in patches)
        lap = cv2.Laplacian(frames_gray[i], cv2.CV_64F)
        blur_map = -np.abs(lap)  # more negative = sharper
        
        # Optical flow magnitude
        flow = cv2.calcOpticalFlowFarneback(
            frames_gray[i-1], frames_gray[i], None,
            0.5, 2, 10, 2, 5, 1.1, 0
        )
        flow_mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        
        face_blur = blur_map[mask > 0]
        face_flow = flow_mag[mask > 0]
        
        if len(face_blur) > 100 and face_flow.std() > 1e-8:
            c = np.corrcoef(face_blur, face_flow)[0, 1]
            if np.isfinite(c):
                couplings.append(c)
    
    if len(couplings) < 3:
        return np.array([0.0, 0.0])
    
    coupling_mean = np.mean(couplings)
    coupling_consistency = 1.0 / (np.std(couplings) + 1e-8)
    
    return np.array([coupling_mean, coupling_consistency])


def extract_dct_stability(frames_gray, landmarks_list):
    """T14: DCT Temporal Stability — 2 features."""
    n = len(frames_gray)
    if n < MIN_FRAMES_TEMPORAL:
        return np.array([0.0, 0.0])
    
    # Extract DCT from nose bridge region across frames
    dct_series = []
    
    for g, lm in zip(frames_gray, landmarks_list):
        if lm is None:
            continue
        
        center = lm[NOSE_BRIDGE].mean(axis=0).astype(int)
        x, y = center[0], center[1]
        x1, x2 = max(0, x - 16), min(g.shape[1], x + 16)
        y1, y2 = max(0, y - 16), min(g.shape[0], y + 16)
        
        patch = g[y1:y2, x1:x2]
        if patch.shape[0] < 8 or patch.shape[1] < 8:
            continue
        
        patch = cv2.resize(patch, (32, 32)).astype(np.float64)
        dct = cv2.dct(patch)
        # Mid-frequency coefficients (indices 3-10 along diagonal)
        mid_freq = []
        for k in range(3, 11):
            for j in range(k + 1):
                i_idx = k - j
                if i_idx < 32 and j < 32:
                    mid_freq.append(dct[i_idx, j])
        dct_series.append(mid_freq)
    
    if len(dct_series) < 10:
        return np.array([0.0, 0.0])
    
    dct_arr = np.array(dct_series)
    
    # Temporal std of each coefficient
    temporal_std = dct_arr.std(axis=0).mean()
    
    # Temporal autocorrelation (lag-1) of coefficients
    autocorrs = []
    for col in range(dct_arr.shape[1]):
        series = dct_arr[:, col]
        if series.std() > 1e-8 and len(series) > 5:
            ac = np.corrcoef(series[:-1], series[1:])[0, 1]
            if np.isfinite(ac):
                autocorrs.append(ac)
    temporal_autocorr = np.mean(autocorrs) if autocorrs else 0.0
    
    return np.array([temporal_std, temporal_autocorr])


# ============================================================================
# ORCHESTRATOR — Process one video
# ============================================================================

def process_single_video(video_path, label, max_frames=300):
    """Extract all 50 features from a single video.
    
    Returns: dict with feature names as keys, or None on failure.
    """
    try:
        frames_bgr, fps = load_video_frames(video_path, max_frames=max_frames)
        if len(frames_bgr) < MIN_FRAMES_SPATIAL:
            return None
        
        n = len(frames_bgr)
        
        # Convert all frames
        frames_gray = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames_bgr]
        frames_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_bgr]
        
        # Run MediaPipe on all frames
        face_mesh = init_face_mesh()
        landmarks_list = []
        for rgb in frames_rgb:
            lm = get_landmarks(face_mesh, rgb)
            landmarks_list.append(lm)
        face_mesh.close()
        
        # Check minimum face detection rate
        detected = sum(1 for lm in landmarks_list if lm is not None)
        if detected / n < MIN_FACE_DETECTIONS:
            return None
        
        # Build face masks for all frames
        face_masks = []
        bg_masks = []
        for i, (g, lm) in enumerate(zip(frames_gray, landmarks_list)):
            if lm is not None:
                fm = get_face_mask(lm, g.shape)
            else:
                fm = np.zeros(g.shape[:2], dtype=np.uint8)
            face_masks.append(fm)
            bg_masks.append((1 - fm).astype(np.uint8))
        
        # ============================================================
        # SPATIAL FEATURES: sample frames, compute, average
        # ============================================================
        spatial_step = max(1, n // 20)
        spatial_results = []
        prev_gray = None
        
        for i in range(0, n, spatial_step):
            lm = landmarks_list[i]
            if lm is None:
                prev_gray = frames_gray[i]
                continue
            
            feats = extract_spatial_features_single_frame(
                frames_bgr[i], frames_gray[i], prev_gray,
                lm, face_masks[i]
            )
            spatial_results.append(feats)
            prev_gray = frames_gray[i]
        
        if len(spatial_results) < 3:
            return None
        
        spatial_features = np.nanmean(spatial_results, axis=0)
        spatial_features = np.nan_to_num(spatial_features, nan=0.0)
        
        # ============================================================
        # TEMPORAL FEATURES: use full frame sequence
        # ============================================================
        t1 = extract_temporal_noise_stability(frames_gray, face_masks)
        t2 = extract_rppg(frames_rgb, landmarks_list, fps)
        t3 = extract_temporal_prnu(frames_gray, face_masks, bg_masks)
        t4 = extract_face_structural_stability(frames_gray, landmarks_list)
        t5 = extract_codec_temporal_residual(frames_gray, face_masks)
        t6 = extract_landmark_trajectory(landmarks_list, fps)
        t7 = extract_rigid_geometry(landmarks_list)
        t8 = extract_boundary_coherence(frames_gray, frames_bgr, landmarks_list)
        t9 = extract_skin_texture(frames_gray, landmarks_list)
        t10 = extract_color_transfer(frames_bgr, landmarks_list, face_masks)
        t11 = extract_specular_temporal(frames_gray, landmarks_list)
        t12 = extract_blink_dynamics(landmarks_list, fps)
        t13 = extract_motion_blur_coupling(frames_gray, face_masks)
        t14 = extract_dct_stability(frames_gray, landmarks_list)
        
        temporal_features = np.concatenate([
            t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14
        ])
        temporal_features = np.nan_to_num(temporal_features, nan=0.0)
        
        # ============================================================
        # Combine all features
        # ============================================================
        all_features = np.concatenate([spatial_features, temporal_features])
        
        assert len(all_features) == N_FEATURES, \
            f"Expected {N_FEATURES} features, got {len(all_features)}"
        
        # Replace inf
        all_features = np.nan_to_num(all_features, nan=0.0,
                                      posinf=1e6, neginf=-1e6)
        
        result = {"video_path": str(video_path), "label": label}
        for name, val in zip(ALL_FEATURE_NAMES, all_features):
            result[name] = float(val)
        
        return result
    
    except Exception as e:
        print(f"[ERROR] {video_path}: {e}")
        traceback.print_exc()
        return None


def _worker(args):
    """Wrapper for multiprocessing."""
    video_path, label, max_frames = args
    return process_single_video(video_path, label, max_frames)


# ============================================================================
# DATASET DISCOVERY
# ============================================================================

def discover_videos(video_dir, label=None, extensions=(".mp4", ".avi", ".mkv")):
    """Find all video files in a directory recursively."""
    video_dir = Path(video_dir)
    videos = []
    for ext in extensions:
        videos.extend(video_dir.rglob(f"*{ext}"))
    videos.sort()
    return [(str(v), label) for v in videos]


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="PRISM V3 Feature Extractor — Phantom Lens V2"
    )
    parser.add_argument("--video_dir", type=str, required=True,
                        help="Directory containing video files")
    parser.add_argument("--output", type=str, default="features_v3.csv",
                        help="Output CSV file path")
    parser.add_argument("--label", type=int, required=True,
                        help="Label: 0=real, 1=fake")
    parser.add_argument("--max_frames", type=int, default=300,
                        help="Maximum frames to process per video")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel workers (1=sequential)")
    parser.add_argument("--append", action="store_true",
                        help="Append to existing CSV instead of overwriting")
    args = parser.parse_args()
    
    videos = discover_videos(args.video_dir, args.label)
    print(f"Found {len(videos)} videos in {args.video_dir} (label={args.label})")
    
    if len(videos) == 0:
        print("No videos found. Exiting.")
        return
    
    # Prepare CSV
    header = ["video_path", "label"] + ALL_FEATURE_NAMES
    mode = "a" if args.append else "w"
    write_header = not args.append or not os.path.exists(args.output)
    
    outfile = open(args.output, mode, newline="")
    writer = csv.DictWriter(outfile, fieldnames=header)
    if write_header:
        writer.writeheader()
    
    tasks = [(v, l, args.max_frames) for v, l in videos]
    
    success = 0
    failed = 0
    
    if args.workers <= 1:
        for task in tqdm(tasks, desc="Extracting features"):
            result = _worker(task)
            if result is not None:
                writer.writerow(result)
                outfile.flush()
                success += 1
            else:
                failed += 1
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(_worker, t): t for t in tasks}
            for future in tqdm(as_completed(futures), total=len(tasks),
                               desc="Extracting features"):
                result = future.result()
                if result is not None:
                    writer.writerow(result)
                    outfile.flush()
                    success += 1
                else:
                    failed += 1
    
    outfile.close()
    print(f"\nDone. Success: {success}, Failed: {failed}")
    print(f"Features saved to: {args.output}")
    print(f"Feature dimensions: {N_FEATURES} "
          f"({len(FEATURE_NAMES_SPATIAL)} spatial + "
          f"{len(FEATURE_NAMES_TEMPORAL)} temporal)")


if __name__ == "__main__":
    main()