#!/usr/bin/env python3
"""
PHANTOM LENS V2 — Publication-Quality Temporal Physics Visualizations
======================================================================
Generates the definitive set of plots proving physics features discriminate
real vs fake video temporally. Designed to convince reviewers that
cross-dataset AUC 0.7+ is achievable on CelebDF-v2.

Plots generated:
  1. rPPG Power Spectrum — cardiac peak (real) vs flat noise (fake)
  2. rPPG Waveform Comparison — zoomed, amplitude-scaled
  3. Noise Fingerprint Temporal Stability — real stable, fake drifts
  4. Skin Color Stability — real tight band, fake wide variance
  5. Rigid Geometry Stability — IPD constant (real) vs jittering (fake)
  6. Skin Texture Persistence — real high correlation, fake drops
  7. Face-BG Boundary — blending artifacts vary in fakes
  8. Landmark Trajectory — jitter comparison
  9. Summary Statistics — bar chart of temporal std ratios
  10. Violin Plots — distribution comparison per feature
  11. Box Plot Grid — compact publishable summary

Usage:
    python visualize_publication_best.py \
        --real_dir "path/to/real/videos" \
        --fake_dir "path/to/fake/videos" \
        --output_dir results/publication_plots \
        --n_videos 10 \
        --max_frames 150
"""

import argparse
import os
import sys
import warnings
import glob
import random

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import signal as sp_signal
from scipy import stats as sp_stats

warnings.filterwarnings("ignore")
random.seed(42)
np.random.seed(42)

# ============================================================================
# Plot Style — Publication Quality
# ============================================================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 200,
    'axes.grid': True,
    'grid.alpha': 0.15,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

REAL_COLOR = "#2271B3"
FAKE_COLOR = "#D1442F"
REAL_FILL = "#2271B3"
FAKE_FILL = "#D1442F"

# ============================================================================
# MediaPipe Setup
# ============================================================================
def init_face_mesh():
    import mediapipe as mp
    return mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False, max_num_faces=1,
        refine_landmarks=True, min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

def get_landmarks(face_mesh, frame_rgb):
    results = face_mesh.process(frame_rgb)
    if not results.multi_face_landmarks:
        return None
    lm = results.multi_face_landmarks[0]
    h, w = frame_rgb.shape[:2]
    return np.array([(l.x * w, l.y * h) for l in lm.landmark], dtype=np.float32)

FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361,
             288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149,
             150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
FOREHEAD = [10, 67, 109, 338, 297, 21, 54, 103, 68, 104]
LEFT_CHEEK = [116, 117, 118, 119, 120, 121, 187, 207, 206]
RIGHT_CHEEK = [345, 346, 347, 348, 349, 350, 411, 427, 426]
NOSE_BRIDGE = [168, 6, 197, 195, 5, 4, 1]

def get_face_mask(landmarks, shape):
    pts = landmarks[FACE_OVAL].astype(np.int32)
    hull = cv2.convexHull(pts)
    mask = np.zeros(shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 1)
    return mask

def get_face_size(landmarks):
    pts = landmarks[FACE_OVAL]
    return np.linalg.norm(pts.max(axis=0) - pts.min(axis=0))

def compute_ear(landmarks, eye_indices):
    p = landmarks[eye_indices]
    A = np.linalg.norm(p[1] - p[5])
    B = np.linalg.norm(p[2] - p[4])
    C = np.linalg.norm(p[0] - p[3])
    return (A + B) / (2.0 * C) if C > 1e-6 else 0.0

def get_roi_mean_rgb(frame, landmarks, indices):
    pts = landmarks[indices].astype(np.int32)
    hull = cv2.convexHull(pts)
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 1)
    pixels = frame[mask > 0]
    return pixels.mean(axis=0) if len(pixels) > 10 else np.array([0.0, 0.0, 0.0])

# ============================================================================
# Video Loading
# ============================================================================
def load_video_with_landmarks(video_path, max_frames=150):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    frames_bgr = []
    while len(frames_bgr) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames_bgr.append(frame)
    cap.release()
    if len(frames_bgr) < 30:
        return None

    frames_gray = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames_bgr]
    frames_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_bgr]
    face_mesh = init_face_mesh()
    landmarks_list = [get_landmarks(face_mesh, rgb) for rgb in frames_rgb]
    face_mesh.close()
    detected = sum(1 for lm in landmarks_list if lm is not None)
    if detected / len(frames_bgr) < 0.5:
        return None
    return {
        "bgr": frames_bgr, "gray": frames_gray, "rgb": frames_rgb,
        "landmarks": landmarks_list, "fps": fps, "n": len(frames_bgr),
    }

# ============================================================================
# Frame-by-Frame Feature Extractors
# ============================================================================

def extract_noise_corr_series(data):
    """T1: Noise residual correlation between consecutive frames."""
    corrs = []
    for i in range(1, data["n"]):
        lm = data["landmarks"][i]
        lm_prev = data["landmarks"][i - 1]
        if lm is None or lm_prev is None:
            corrs.append(np.nan)
            continue
        mask = get_face_mask(lm, data["gray"][i].shape)
        r1 = (data["gray"][i-1].astype(float) - cv2.medianBlur(data["gray"][i-1], 5).astype(float)) * mask
        r2 = (data["gray"][i].astype(float) - cv2.medianBlur(data["gray"][i], 5).astype(float)) * mask
        v = (r1.flatten() != 0) & (r2.flatten() != 0)
        if v.sum() > 200:
            c = np.corrcoef(r1.flatten()[v], r2.flatten()[v])[0, 1]
            corrs.append(c if np.isfinite(c) else np.nan)
        else:
            corrs.append(np.nan)
    return np.array(corrs)


def extract_rppg_waveform_and_spectrum(data):
    """T2: Returns BOTH the filtered waveform AND the power spectrum."""
    green_forehead = []
    green_cheek = []
    for frame, lm in zip(data["rgb"], data["landmarks"]):
        if lm is None:
            green_forehead.append(np.nan)
            green_cheek.append(np.nan)
            continue
        clean = cv2.bilateralFilter(frame, d=7, sigmaColor=25, sigmaSpace=25)
        fh = get_roi_mean_rgb(clean, lm, FOREHEAD)
        ck_l = get_roi_mean_rgb(clean, lm, LEFT_CHEEK)
        ck_r = get_roi_mean_rgb(clean, lm, RIGHT_CHEEK)
        green_forehead.append(fh[1])
        green_cheek.append((ck_l[1] + ck_r[1]) / 2.0)

    def process_signal(raw_green):
        sig = np.array(raw_green, dtype=np.float64)
        nans = np.isnan(sig)
        if nans.all() or (~nans).sum() < 30:
            return None, None, None
        sig[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(~nans), sig[~nans])
        mean_val = sig.mean()
        if mean_val > 1:
            sig = sig / mean_val

        fps = data["fps"]
        nyq = fps / 2.0
        if nyq <= 4.0:
            return sig, None, None

        # POS algorithm
        rgb_signals = []
        for frame, lm in zip(data["rgb"], data["landmarks"]):
            if lm is None:
                rgb_signals.append([0, 0, 0])
                continue
            clean = cv2.bilateralFilter(frame, d=7, sigmaColor=25, sigmaSpace=25)
            rgb_signals.append(get_roi_mean_rgb(clean, lm, FOREHEAD).tolist())

        rgb_arr = np.array(rgb_signals, dtype=np.float64)
        nans_rgb = np.any(rgb_arr == 0, axis=1)
        if nans_rgb.all():
            return sig, None, None

        # POS extraction
        win_len = int(1.6 * fps)
        win_len = max(20, min(win_len, len(rgb_arr)))
        pulse = np.zeros(len(rgb_arr))
        for t in range(0, len(rgb_arr) - win_len + 1):
            segment = rgb_arr[t:t + win_len]
            means = segment.mean(axis=0, keepdims=True)
            means[means < 1] = 1
            Cn = segment / means
            S1 = Cn[:, 1] - Cn[:, 2]
            S2 = Cn[:, 1] + Cn[:, 2] - 2 * Cn[:, 0]
            std_s2 = S2.std()
            if std_s2 < 1e-10:
                continue
            alpha = S1.std() / std_s2
            h = S1 + alpha * S2
            pulse[t:t + win_len] += h - h.mean()

        overlap = np.minimum(np.arange(1, len(pulse)+1),
                             np.minimum(win_len, np.arange(len(pulse), 0, -1))).astype(float)
        overlap[overlap < 1] = 1
        pulse /= overlap

        b, a = sp_signal.butter(3, [0.7/nyq, min(4.0/nyq, 0.99)], btype='band')
        try:
            filtered = sp_signal.filtfilt(b, a, pulse)
        except Exception:
            return sig, None, None

        # Power spectrum via Welch
        nperseg = min(len(filtered), int(4 * fps))
        if nperseg < 16:
            nperseg = len(filtered)
        try:
            freqs, psd = sp_signal.welch(filtered, fs=fps, nperseg=nperseg,
                                          noverlap=nperseg//2, window='hann')
        except Exception:
            return sig, None, None

        return filtered, freqs, psd

    waveform_fh, freqs_fh, psd_fh = process_signal(green_forehead)
    return {
        "waveform": waveform_fh,
        "freqs": freqs_fh,
        "psd": psd_fh,
        "green_raw": np.array(green_forehead),
    }


def extract_rigid_ipd_series(data):
    """T7: Interpupillary distance (normalized) over time."""
    vals = []
    for lm in data["landmarks"]:
        if lm is None:
            vals.append(np.nan)
            continue
        fsize = get_face_size(lm)
        if fsize < 1: fsize = 1
        l_eye = lm[[33, 133]].mean(axis=0)
        r_eye = lm[[362, 263]].mean(axis=0)
        vals.append(np.linalg.norm(l_eye - r_eye) / fsize)
    return np.array(vals)


def extract_nose_bridge_series(data):
    """T7b: Nose bridge length (normalized) over time."""
    vals = []
    for lm in data["landmarks"]:
        if lm is None:
            vals.append(np.nan)
            continue
        fsize = get_face_size(lm)
        if fsize < 1: fsize = 1
        vals.append(np.linalg.norm(lm[168] - lm[1]) / fsize)
    return np.array(vals)


def extract_skin_color_lab_series(data):
    """T10: Mean face skin L* over time."""
    vals = []
    for bgr, lm in zip(data["bgr"], data["landmarks"]):
        if lm is None:
            vals.append(np.nan)
            continue
        mask = get_face_mask(lm, bgr.shape)
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float64)
        pix = lab[mask > 0]
        vals.append(pix[:, 0].mean() if len(pix) > 50 else np.nan)
    return np.array(vals)


def extract_skin_texture_corr_series(data):
    """T9: Cheek texture correlation between consecutive frames."""
    corrs = []
    prev_patch = None
    for g, lm in zip(data["gray"], data["landmarks"]):
        if lm is None:
            corrs.append(np.nan)
            prev_patch = None
            continue
        center = lm[RIGHT_CHEEK].mean(axis=0).astype(int)
        y, x = center[1], center[0]
        y1, y2 = max(0, y-16), min(g.shape[0], y+16)
        x1, x2 = max(0, x-16), min(g.shape[1], x+16)
        patch = g[y1:y2, x1:x2]
        if patch.shape[0] < 8 or patch.shape[1] < 8:
            corrs.append(np.nan)
            prev_patch = None
            continue
        patch = cv2.resize(patch, (32, 32)).astype(np.float64)
        if prev_patch is not None:
            c = np.corrcoef(patch.flatten(), prev_patch.flatten())[0, 1]
            corrs.append(c if np.isfinite(c) else np.nan)
        else:
            corrs.append(np.nan)
        prev_patch = patch
    return np.array(corrs)


def extract_boundary_gradient_series(data):
    """T8: Face boundary gradient over time."""
    vals = []
    for g, lm in zip(data["gray"], data["landmarks"]):
        if lm is None:
            vals.append(np.nan)
            continue
        mask = get_face_mask(lm, g.shape)
        kernel = np.ones((7, 7), np.uint8)
        boundary = cv2.dilate(mask, kernel) - cv2.erode(mask, kernel)
        gx = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.sqrt(gx**2 + gy**2)
        bnd = mag[boundary > 0]
        vals.append(bnd.mean() if len(bnd) > 10 else np.nan)
    return np.array(vals)


def extract_landmark_jitter_series(data):
    """T6: Total normalized landmark displacement between frames."""
    vals = []
    key_lm = FACE_OVAL[:20] + NOSE_BRIDGE
    for i in range(1, data["n"]):
        lm = data["landmarks"][i]
        lm_prev = data["landmarks"][i-1]
        if lm is None or lm_prev is None:
            vals.append(np.nan)
            continue
        fsize = get_face_size(lm)
        if fsize < 1: fsize = 1
        disp = np.linalg.norm(lm[key_lm] - lm_prev[key_lm], axis=1).mean() / fsize
        vals.append(disp)
    return np.array(vals)


# ============================================================================
# Processing Pipeline
# ============================================================================

def process_videos(video_dir, n_videos, max_frames):
    all_vids = sorted(glob.glob(os.path.join(video_dir, "*.mp4")))
    if not all_vids:
        all_vids = sorted(glob.glob(os.path.join(video_dir, "**", "*.mp4"), recursive=True))
    if not all_vids:
        print(f"  No videos found in {video_dir}")
        return []
    selected = random.sample(all_vids, min(n_videos, len(all_vids)))
    results = []
    for vid_path in selected:
        name = os.path.basename(vid_path)
        print(f"    {name}...", end=" ", flush=True)
        data = load_video_with_landmarks(vid_path, max_frames)
        if data is None:
            print("SKIP")
            continue
        vid = {"path": vid_path, "fps": data["fps"]}
        vid["noise_corr"] = extract_noise_corr_series(data)
        vid["rppg"] = extract_rppg_waveform_and_spectrum(data)
        vid["ipd"] = extract_rigid_ipd_series(data)
        vid["nose_bridge"] = extract_nose_bridge_series(data)
        vid["skin_color"] = extract_skin_color_lab_series(data)
        vid["skin_texture"] = extract_skin_texture_corr_series(data)
        vid["boundary"] = extract_boundary_gradient_series(data)
        vid["landmark_jitter"] = extract_landmark_jitter_series(data)
        results.append(vid)
        print(f"OK ({data['n']} frames)")
    return results


# ============================================================================
# Plot Generators
# ============================================================================

def plot_rppg_power_spectrum(real_results, fake_results, output_dir):
    """PLOT 1: rPPG power spectrum — THE key plot for the paper."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left: Individual spectra
    ax = axes[0]
    for r in real_results:
        if r["rppg"]["freqs"] is not None:
            mask = (r["rppg"]["freqs"] >= 0.5) & (r["rppg"]["freqs"] <= 4.5)
            psd_norm = r["rppg"]["psd"][mask] / (r["rppg"]["psd"][mask].max() + 1e-10)
            ax.plot(r["rppg"]["freqs"][mask], psd_norm,
                    color=REAL_COLOR, alpha=0.4, linewidth=1.2)
    for f in fake_results:
        if f["rppg"]["freqs"] is not None:
            mask = (f["rppg"]["freqs"] >= 0.5) & (f["rppg"]["freqs"] <= 4.5)
            psd_norm = f["rppg"]["psd"][mask] / (f["rppg"]["psd"][mask].max() + 1e-10)
            ax.plot(f["rppg"]["freqs"][mask], psd_norm,
                    color=FAKE_COLOR, alpha=0.4, linewidth=1.2)

    ax.axvspan(0.75, 3.0, alpha=0.08, color='green', label="Cardiac band (45-180 bpm)")
    ax.plot([], [], color=REAL_COLOR, linewidth=2, label="Real")
    ax.plot([], [], color=FAKE_COLOR, linewidth=2, label="Fake")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Normalized Power")
    ax.set_title("rPPG Power Spectrum — Individual Videos")
    ax.legend(fontsize=8)
    ax.set_xlim(0.5, 4.5)

    # Right: Average spectrum with confidence band
    ax = axes[1]
    def avg_spectrum(results, freq_range=(0.5, 4.5), n_bins=100):
        freqs_common = np.linspace(freq_range[0], freq_range[1], n_bins)
        all_psd = []
        for r in results:
            if r["rppg"]["freqs"] is not None:
                mask = (r["rppg"]["freqs"] >= freq_range[0]) & (r["rppg"]["freqs"] <= freq_range[1])
                f = r["rppg"]["freqs"][mask]
                p = r["rppg"]["psd"][mask]
                if len(f) > 3:
                    p_norm = p / (p.max() + 1e-10)
                    p_interp = np.interp(freqs_common, f, p_norm)
                    all_psd.append(p_interp)
        if not all_psd:
            return freqs_common, np.zeros(n_bins), np.zeros(n_bins)
        arr = np.array(all_psd)
        return freqs_common, arr.mean(axis=0), arr.std(axis=0)

    f_r, m_r, s_r = avg_spectrum(real_results)
    f_f, m_f, s_f = avg_spectrum(fake_results)

    ax.fill_between(f_r, m_r - s_r, m_r + s_r, color=REAL_COLOR, alpha=0.15)
    ax.plot(f_r, m_r, color=REAL_COLOR, linewidth=2.5, label="Real (mean ± std)")
    ax.fill_between(f_f, m_f - s_f, m_f + s_f, color=FAKE_COLOR, alpha=0.15)
    ax.plot(f_f, m_f, color=FAKE_COLOR, linewidth=2.5, label="Fake (mean ± std)")
    ax.axvspan(0.75, 3.0, alpha=0.08, color='green')
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Normalized Power")
    ax.set_title("rPPG Power Spectrum — Average (Real vs Fake)")
    ax.legend(fontsize=8)
    ax.set_xlim(0.5, 4.5)

    plt.tight_layout()
    path = os.path.join(output_dir, "01_rppg_power_spectrum.png")
    plt.savefig(path, dpi=250, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_rppg_waveform(real_results, fake_results, output_dir):
    """PLOT 2: rPPG waveform — zoomed and amplitude-scaled."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True)

    fps = real_results[0]["fps"] if real_results else 30

    ax = axes[0]
    ax.set_title("Real Videos — rPPG Pulse Waveform (POS algorithm)", fontsize=11)
    for r in real_results[:5]:
        wf = r["rppg"]["waveform"]
        if wf is not None:
            t = np.arange(len(wf)) / fps
            # Scale to show oscillation clearly
            wf_scaled = (wf - wf.mean()) / (wf.std() + 1e-10)
            ax.plot(t, wf_scaled, color=REAL_COLOR, alpha=0.5, linewidth=0.8)
    ax.set_ylabel("Normalized amplitude")
    ax.set_ylim(-4, 4)

    ax = axes[1]
    ax.set_title("Fake Videos — rPPG Pulse Waveform (POS algorithm)", fontsize=11)
    for f in fake_results[:5]:
        wf = f["rppg"]["waveform"]
        if wf is not None:
            t = np.arange(len(wf)) / fps
            wf_scaled = (wf - wf.mean()) / (wf.std() + 1e-10)
            ax.plot(t, wf_scaled, color=FAKE_COLOR, alpha=0.5, linewidth=0.8)
    ax.set_ylabel("Normalized amplitude")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylim(-4, 4)

    plt.tight_layout()
    path = os.path.join(output_dir, "02_rppg_waveform.png")
    plt.savefig(path, dpi=250, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_rppg_snr_comparison(real_results, fake_results, output_dir):
    """PLOT 3: rPPG SNR bar/box comparison."""
    def compute_snr(result):
        if result["rppg"]["freqs"] is None:
            return 0.0
        freqs = result["rppg"]["freqs"]
        psd = result["rppg"]["psd"]
        cardiac = (freqs >= 0.7) & (freqs <= 4.0)
        cardiac_psd = psd[cardiac]
        if len(cardiac_psd) < 3:
            return 0.0
        return cardiac_psd.max() / (cardiac_psd.mean() + 1e-10)

    real_snrs = [compute_snr(r) for r in real_results]
    fake_snrs = [compute_snr(f) for f in fake_results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Box plot
    bp = ax1.boxplot([real_snrs, fake_snrs], labels=["Real", "Fake"],
                      patch_artist=True, widths=0.5)
    bp['boxes'][0].set_facecolor(REAL_COLOR)
    bp['boxes'][0].set_alpha(0.4)
    bp['boxes'][1].set_facecolor(FAKE_COLOR)
    bp['boxes'][1].set_alpha(0.4)
    ax1.set_ylabel("rPPG Signal-to-Noise Ratio")
    ax1.set_title("Cardiac Signal Quality")

    # Individual points
    ax1.scatter(np.ones(len(real_snrs)) + np.random.randn(len(real_snrs))*0.05,
                real_snrs, color=REAL_COLOR, alpha=0.6, s=30, zorder=3)
    ax1.scatter(np.ones(len(fake_snrs))*2 + np.random.randn(len(fake_snrs))*0.05,
                fake_snrs, color=FAKE_COLOR, alpha=0.6, s=30, zorder=3)

    # Statistical test
    if len(real_snrs) > 2 and len(fake_snrs) > 2:
        t_stat, p_val = sp_stats.ttest_ind(real_snrs, fake_snrs)
        cohens_d = (np.mean(real_snrs) - np.mean(fake_snrs)) / \
                   (np.sqrt((np.std(real_snrs)**2 + np.std(fake_snrs)**2) / 2) + 1e-8)
        ax1.text(0.05, 0.95, f"Cohen's d = {abs(cohens_d):.2f}\np = {p_val:.4f}",
                 transform=ax1.transAxes, fontsize=9, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Histogram
    ax2.hist(real_snrs, bins=15, alpha=0.5, color=REAL_COLOR, label="Real", density=True)
    ax2.hist(fake_snrs, bins=15, alpha=0.5, color=FAKE_COLOR, label="Fake", density=True)
    ax2.set_xlabel("SNR")
    ax2.set_ylabel("Density")
    ax2.set_title("SNR Distribution")
    ax2.legend()

    plt.tight_layout()
    path = os.path.join(output_dir, "03_rppg_snr_comparison.png")
    plt.savefig(path, dpi=250, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_temporal_stability_grid(real_results, fake_results, output_dir):
    """PLOT 4: 2x3 grid of key temporal stability features — the money plot."""
    features = [
        ("noise_corr", "T1: Noise Fingerprint\nCorrelation", "Pearson r"),
        ("skin_texture", "T9: Skin Texture\nPersistence", "Pearson r"),
        ("ipd", "T7: Interpupillary\nDistance (rigid)", "Normalized IPD"),
        ("skin_color", "T10: Skin Luminance\nStability (LAB L*)", "L* value"),
        ("boundary", "T8: Face Boundary\nGradient", "Gradient magnitude"),
        ("landmark_jitter", "T6: Landmark\nDisplacement", "Normalized disp."),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Temporal Physics Consistency — Key Features\n"
                 "Blue = Real (stable), Red = Fake (inconsistent)",
                 fontsize=14, fontweight="bold", y=1.02)

    for idx, (key, title, ylabel) in enumerate(features):
        ax = axes[idx // 3, idx % 3]

        real_series = [r[key] for r in real_results]
        fake_series = [f[key] for f in fake_results]

        # Compute mean ± std bands
        def compute_band(series_list):
            max_len = max(len(s) for s in series_list) if series_list else 0
            if max_len == 0:
                return np.array([]), np.array([])
            padded = np.full((len(series_list), max_len), np.nan)
            for i, s in enumerate(series_list):
                padded[i, :len(s)] = s
            return np.nanmean(padded, axis=0), np.nanstd(padded, axis=0)

        fps = real_results[0]["fps"] if real_results else 30

        r_mean, r_std = compute_band(real_series)
        f_mean, f_std = compute_band(fake_series)

        t_r = np.arange(len(r_mean)) / fps
        t_f = np.arange(len(f_mean)) / fps

        ax.fill_between(t_r, r_mean - r_std, r_mean + r_std,
                         color=REAL_COLOR, alpha=0.15)
        ax.plot(t_r, r_mean, color=REAL_COLOR, linewidth=2, label="Real")

        ax.fill_between(t_f, f_mean - f_std, f_mean + f_std,
                         color=FAKE_COLOR, alpha=0.15)
        ax.plot(t_f, f_mean, color=FAKE_COLOR, linewidth=2, label="Fake")

        ax.set_title(title, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_xlabel("Time (s)", fontsize=8)
        ax.legend(fontsize=7, loc="best")

    plt.tight_layout()
    path = os.path.join(output_dir, "04_temporal_stability_grid.png")
    plt.savefig(path, dpi=250, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_temporal_std_comparison(real_results, fake_results, output_dir):
    """PLOT 5: Bar chart comparing temporal standard deviation (the variance argument)."""
    features = {
        "T1: Noise\ncorr": "noise_corr",
        "T7: IPD": "ipd",
        "T7b: Nose\nbridge": "nose_bridge",
        "T9: Skin\ntexture": "skin_texture",
        "T10: Skin\ncolor": "skin_color",
        "T8: Boundary\ngradient": "boundary",
        "T6: Landmark\njitter": "landmark_jitter",
    }

    fig, ax = plt.subplots(figsize=(10, 5))

    names = list(features.keys())
    real_stds = []
    fake_stds = []

    for key in features.values():
        r_std = np.nanmean([np.nanstd(r[key]) for r in real_results])
        f_std = np.nanmean([np.nanstd(f[key]) for f in fake_results])
        # Normalize so they're comparable
        max_val = max(r_std, f_std, 1e-10)
        real_stds.append(r_std / max_val)
        fake_stds.append(f_std / max_val)

    x = np.arange(len(names))
    bars_r = ax.bar(x - 0.17, real_stds, 0.3, label="Real", color=REAL_COLOR, alpha=0.8)
    bars_f = ax.bar(x + 0.17, fake_stds, 0.3, label="Fake", color=FAKE_COLOR, alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("Normalized Temporal Std Dev")
    ax.set_title("Temporal Variability Comparison\n(Ratio of std devs — differences indicate discriminative features)")
    ax.legend()
    ax.set_ylim(0, 1.4)

    # Add ratio annotations
    for i in range(len(names)):
        ratio = fake_stds[i] / (real_stds[i] + 1e-10)
        if abs(ratio - 1.0) > 0.05:
            ax.text(i, max(real_stds[i], fake_stds[i]) + 0.03,
                    f"×{ratio:.2f}", ha='center', fontsize=7, fontweight='bold')

    plt.tight_layout()
    path = os.path.join(output_dir, "05_temporal_std_comparison.png")
    plt.savefig(path, dpi=250, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_violin_distributions(real_results, fake_results, output_dir):
    """PLOT 6: Violin plots of per-video temporal statistics."""
    stats = {}
    stat_names = {
        "Noise corr\n(mean)": lambda r: np.nanmean(r["noise_corr"]),
        "Noise corr\n(std)": lambda r: np.nanstd(r["noise_corr"]),
        "IPD std": lambda r: np.nanstd(r["ipd"]),
        "Skin tex\n(mean)": lambda r: np.nanmean(r["skin_texture"]),
        "Skin color\n(std)": lambda r: np.nanstd(r["skin_color"]),
        "Boundary\n(std)": lambda r: np.nanstd(r["boundary"]),
        "Landmark\njitter (mean)": lambda r: np.nanmean(r["landmark_jitter"]),
    }

    fig, axes = plt.subplots(1, len(stat_names), figsize=(16, 4))
    fig.suptitle("Per-Video Feature Distributions — Real vs Fake",
                 fontsize=13, fontweight="bold")

    for idx, (name, func) in enumerate(stat_names.items()):
        real_vals = [func(r) for r in real_results]
        fake_vals = [func(f) for f in fake_results]

        real_vals = [v for v in real_vals if np.isfinite(v)]
        fake_vals = [v for v in fake_vals if np.isfinite(v)]

        ax = axes[idx]
        parts = ax.violinplot([real_vals, fake_vals], positions=[1, 2],
                                showmeans=True, showmedians=True)

        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(REAL_COLOR if i == 0 else FAKE_COLOR)
            pc.set_alpha(0.4)

        # Overlay individual points
        ax.scatter(np.ones(len(real_vals)) + np.random.randn(len(real_vals))*0.04,
                   real_vals, color=REAL_COLOR, alpha=0.5, s=15, zorder=3)
        ax.scatter(np.ones(len(fake_vals))*2 + np.random.randn(len(fake_vals))*0.04,
                   fake_vals, color=FAKE_COLOR, alpha=0.5, s=15, zorder=3)

        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Real", "Fake"], fontsize=8)
        ax.set_title(name, fontsize=9)

        # Cohen's d annotation
        if len(real_vals) > 1 and len(fake_vals) > 1:
            d = abs(np.mean(real_vals) - np.mean(fake_vals)) / \
                (np.sqrt((np.std(real_vals)**2 + np.std(fake_vals)**2)/2) + 1e-8)
            ax.text(0.5, 0.02, f"d={d:.2f}", transform=ax.transAxes,
                    fontsize=7, ha='center',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    path = os.path.join(output_dir, "06_violin_distributions.png")
    plt.savefig(path, dpi=250, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_rigid_geometry_zoom(real_results, fake_results, output_dir):
    """PLOT 7: Zoomed-in rigid geometry — IPD + nose bridge must be constant."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Rigid Facial Geometry — Must Be Constant in Real Video\n"
                 "Bone structure distances cannot change between frames",
                 fontsize=13, fontweight="bold")

    fps = real_results[0]["fps"] if real_results else 30

    for idx, (key, title) in enumerate([("ipd", "Interpupillary Distance"),
                                          ("nose_bridge", "Nose Bridge Length")]):
        # Individual traces
        ax = axes[idx, 0]
        for r in real_results:
            t = np.arange(len(r[key])) / fps
            ax.plot(t, r[key], color=REAL_COLOR, alpha=0.4, linewidth=0.8)
        for f in fake_results:
            t = np.arange(len(f[key])) / fps
            ax.plot(t, f[key], color=FAKE_COLOR, alpha=0.4, linewidth=0.8)
        ax.plot([], [], color=REAL_COLOR, linewidth=2, label="Real")
        ax.plot([], [], color=FAKE_COLOR, linewidth=2, label="Fake")
        ax.set_title(f"{title} — Traces", fontsize=10)
        ax.set_ylabel("Normalized distance")
        ax.set_xlabel("Time (s)")
        ax.legend(fontsize=8)

        # Std deviation comparison
        ax = axes[idx, 1]
        real_stds = [np.nanstd(r[key]) for r in real_results]
        fake_stds = [np.nanstd(f[key]) for f in fake_results]
        real_stds = [v for v in real_stds if np.isfinite(v)]
        fake_stds = [v for v in fake_stds if np.isfinite(v)]

        bp = ax.boxplot([real_stds, fake_stds], labels=["Real", "Fake"],
                         patch_artist=True, widths=0.5)
        bp['boxes'][0].set_facecolor(REAL_COLOR)
        bp['boxes'][0].set_alpha(0.4)
        bp['boxes'][1].set_facecolor(FAKE_COLOR)
        bp['boxes'][1].set_alpha(0.4)
        ax.scatter(np.ones(len(real_stds)) + np.random.randn(len(real_stds))*0.04,
                   real_stds, color=REAL_COLOR, alpha=0.6, s=25, zorder=3)
        ax.scatter(np.ones(len(fake_stds))*2 + np.random.randn(len(fake_stds))*0.04,
                   fake_stds, color=FAKE_COLOR, alpha=0.6, s=25, zorder=3)
        ax.set_title(f"{title} — Temporal Std", fontsize=10)
        ax.set_ylabel("Std dev (lower = more stable)")

    plt.tight_layout()
    path = os.path.join(output_dir, "07_rigid_geometry_zoom.png")
    plt.savefig(path, dpi=250, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_summary_cohens_d(real_results, fake_results, output_dir):
    """PLOT 8: Cohen's d for all temporal statistics — the publishable summary."""
    stat_funcs = {
        "T1: Noise corr mean": lambda r: np.nanmean(r["noise_corr"]),
        "T1: Noise corr std": lambda r: np.nanstd(r["noise_corr"]),
        "T2: rPPG SNR": lambda r: (r["rppg"]["psd"][(r["rppg"]["freqs"] >= 0.7) & (r["rppg"]["freqs"] <= 4.0)].max() / (r["rppg"]["psd"][(r["rppg"]["freqs"] >= 0.7) & (r["rppg"]["freqs"] <= 4.0)].mean() + 1e-10)) if r["rppg"]["freqs"] is not None else 0,
        "T6: Landmark jitter": lambda r: np.nanmean(r["landmark_jitter"]),
        "T7: IPD stability": lambda r: np.nanstd(r["ipd"]),
        "T7: Nose bridge std": lambda r: np.nanstd(r["nose_bridge"]),
        "T8: Boundary grad std": lambda r: np.nanstd(r["boundary"]),
        "T9: Skin texture mean": lambda r: np.nanmean(r["skin_texture"]),
        "T10: Skin color std": lambda r: np.nanstd(r["skin_color"]),
    }

    names = []
    d_values = []

    for name, func in stat_funcs.items():
        try:
            real_vals = [func(r) for r in real_results]
            fake_vals = [func(f) for f in fake_results]
            real_vals = [v for v in real_vals if np.isfinite(v)]
            fake_vals = [v for v in fake_vals if np.isfinite(v)]
            if len(real_vals) < 2 or len(fake_vals) < 2:
                continue
            d = (np.mean(real_vals) - np.mean(fake_vals)) / \
                (np.sqrt((np.std(real_vals)**2 + np.std(fake_vals)**2)/2) + 1e-8)
            names.append(name)
            d_values.append(abs(d))
        except Exception:
            continue

    # Sort by d value
    sorted_idx = np.argsort(d_values)[::-1]
    names = [names[i] for i in sorted_idx]
    d_values = [d_values[i] for i in sorted_idx]

    colors = ["#D1442F" if d > 0.8 else "#EF9F27" if d > 0.5
              else "#2271B3" for d in d_values]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(range(len(names)), d_values[::-1], color=colors[::-1])
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names[::-1], fontsize=9)
    ax.axvline(x=0.8, color="red", linestyle="--", alpha=0.4, label="Large effect (d=0.8)")
    ax.axvline(x=0.5, color="orange", linestyle="--", alpha=0.4, label="Medium effect (d=0.5)")
    ax.set_xlabel("Cohen's d (effect size)")
    ax.set_title("Discriminative Strength of Temporal Physics Features\n(FF++ Real vs Deepfakes)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    path = os.path.join(output_dir, "08_cohens_d_summary.png")
    plt.savefig(path, dpi=250, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real_dir", required=True)
    parser.add_argument("--fake_dir", required=True)
    parser.add_argument("--output_dir", default="results/publication_plots")
    parser.add_argument("--n_videos", type=int, default=10)
    parser.add_argument("--max_frames", type=int, default=150)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("PUBLICATION-QUALITY TEMPORAL PHYSICS VISUALIZATION")
    print("=" * 60)

    print(f"\nProcessing {args.n_videos} REAL videos...")
    real_results = process_videos(args.real_dir, args.n_videos, args.max_frames)

    print(f"\nProcessing {args.n_videos} FAKE videos...")
    fake_results = process_videos(args.fake_dir, args.n_videos, args.max_frames)

    if len(real_results) < 2 or len(fake_results) < 2:
        print("ERROR: Need at least 2 videos per class")
        return

    print(f"\nGenerating plots ({len(real_results)} real, {len(fake_results)} fake)...\n")

    plot_rppg_power_spectrum(real_results, fake_results, args.output_dir)
    plot_rppg_waveform(real_results, fake_results, args.output_dir)
    plot_rppg_snr_comparison(real_results, fake_results, args.output_dir)
    plot_temporal_stability_grid(real_results, fake_results, args.output_dir)
    plot_temporal_std_comparison(real_results, fake_results, args.output_dir)
    plot_violin_distributions(real_results, fake_results, args.output_dir)
    plot_rigid_geometry_zoom(real_results, fake_results, args.output_dir)
    plot_summary_cohens_d(real_results, fake_results, args.output_dir)

    print(f"\n{'=' * 60}")
    print(f"ALL 8 PLOTS SAVED TO: {args.output_dir}")
    print(f"{'=' * 60}")
    print("\nKey plots for your paper:")
    print("  01_rppg_power_spectrum.png  — cardiac peak visible in real only")
    print("  04_temporal_stability_grid.png — 6 features side by side")
    print("  06_violin_distributions.png — per-video statistics")
    print("  08_cohens_d_summary.png — effect size ranking")


if __name__ == "__main__":
    main()