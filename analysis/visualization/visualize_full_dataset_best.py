#!/usr/bin/env python3
"""
PHANTOM LENS V2 — Full-Dataset Publication Visualization
==========================================================
Hybrid approach:
  - Statistical plots (Cohen's d, distributions, violin, box) → from FULL CSV
  - Temporal trace plots (frame-by-frame mean ± std) → from 50-video subsample
  
This gives statistically meaningful results on ALL extracted features.

Usage:
    python visualize_full_dataset_best.py \
        --real_csv features\ffpp_real.csv \
        --fake_csv features\ffpp_fake.csv \
        --real_dir "D:\PhantomLens\data\ffpp_official\original_sequences\youtube\c23\videos" \
        --fake_dir "D:\PhantomLens\data\ffpp_official\manipulated_sequences\Deepfakes\c23\videos" \
        --output_dir results\full_publication_plots \
        --trace_videos 50 \
        --max_frames 150
"""

import argparse
import os
import glob
import random
import warnings

import cv2
import numpy as np
import pandas as pd
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
# Plot Style
# ============================================================================
plt.rcParams.update({
    'font.family': 'serif', 'font.size': 11,
    'axes.titlesize': 13, 'axes.labelsize': 11,
    'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'legend.fontsize': 9, 'figure.dpi': 200,
    'axes.grid': True, 'grid.alpha': 0.15,
    'axes.spines.top': False, 'axes.spines.right': False,
})
REAL_COLOR = "#2271B3"
FAKE_COLOR = "#D1442F"

# ============================================================================
# MediaPipe + Landmark Setup (for trace extraction)
# ============================================================================
def init_face_mesh():
    import mediapipe as mp
    return mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False, max_num_faces=1,
        refine_landmarks=True, min_detection_confidence=0.5,
        min_tracking_confidence=0.5)

def get_landmarks(face_mesh, frame_rgb):
    results = face_mesh.process(frame_rgb)
    if not results.multi_face_landmarks:
        return None
    lm = results.multi_face_landmarks[0]
    h, w = frame_rgb.shape[:2]
    return np.array([(l.x * w, l.y * h) for l in lm.landmark], dtype=np.float32)

FACE_OVAL = [10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109]
FOREHEAD = [10,67,109,338,297,21,54,103,68,104]
LEFT_CHEEK = [116,117,118,119,120,121,187,207,206]
RIGHT_CHEEK = [345,346,347,348,349,350,411,427,426]
NOSE_BRIDGE = [168,6,197,195,5,4,1]

def get_face_mask(lm, shape):
    pts = lm[FACE_OVAL].astype(np.int32)
    hull = cv2.convexHull(pts)
    mask = np.zeros(shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 1)
    return mask

def get_face_size(lm):
    pts = lm[FACE_OVAL]
    return np.linalg.norm(pts.max(axis=0) - pts.min(axis=0))

def get_roi_mean_rgb(frame, lm, indices):
    pts = lm[indices].astype(np.int32)
    hull = cv2.convexHull(pts)
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 1)
    pix = frame[mask > 0]
    return pix.mean(axis=0) if len(pix) > 10 else np.array([0.0, 0.0, 0.0])


# ============================================================================
# PART 1: FULL CSV STATISTICAL ANALYSIS
# ============================================================================

def load_csv(paths):
    dfs = [pd.read_csv(p) for p in paths if os.path.exists(p)]
    df = pd.concat(dfs, ignore_index=True)
    feat_cols = sorted([c for c in df.columns if c.startswith('s_') or c.startswith('t_')])
    df[feat_cols] = df[feat_cols].replace([np.inf, -np.inf], np.nan)
    for c in feat_cols:
        df[c] = df[c].fillna(df[c].median())
    return df, feat_cols


def compute_cohens_d(real_vals, fake_vals):
    n1, n2 = len(real_vals), len(fake_vals)
    if n1 < 2 or n2 < 2:
        return 0.0
    m_diff = real_vals.mean() - fake_vals.mean()
    pooled = np.sqrt(((n1-1)*real_vals.std()**2 + (n2-1)*fake_vals.std()**2) / (n1+n2-2))
    return abs(m_diff) / (pooled + 1e-8)


def plot_full_cohens_d(real_df, fake_df, feat_cols, output_dir):
    """PLOT 1: Cohen's d for ALL 50 features on full dataset."""
    d_vals = {}
    for col in feat_cols:
        d = compute_cohens_d(real_df[col].values, fake_df[col].values)
        d_vals[col] = d

    sorted_feats = sorted(d_vals.items(), key=lambda x: x[1], reverse=True)
    names = [x[0] for x in sorted_feats]
    vals = [x[1] for x in sorted_feats]
    colors = ["#D1442F" if v > 0.8 else "#EF9F27" if v > 0.5
              else "#888888" if v > 0.2 else "#CCCCCC" for v in vals]

    fig, ax = plt.subplots(figsize=(10, 12))
    ax.barh(range(len(names)), vals[::-1], color=colors[::-1], height=0.7)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names[::-1], fontsize=8)
    ax.axvline(x=0.8, color="red", linestyle="--", alpha=0.4, label="Large (d=0.8)")
    ax.axvline(x=0.5, color="orange", linestyle="--", alpha=0.4, label="Medium (d=0.5)")
    ax.axvline(x=0.2, color="gray", linestyle="--", alpha=0.3, label="Small (d=0.2)")
    ax.set_xlabel("Cohen's d (effect size)")
    n_real = len(real_df)
    n_fake = len(fake_df)
    ax.set_title(f"Feature Discriminative Strength — Full Dataset\n"
                 f"(n={n_real} real + {n_fake} fake, {len(feat_cols)} features)")
    ax.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    path = os.path.join(output_dir, "01_cohens_d_full_dataset.png")
    plt.savefig(path, dpi=250, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

    # Print top features
    print(f"\n  {'Rank':>4} {'Feature':<40} {'d':>8} {'Effect'}")
    print("  " + "-" * 65)
    for rank, (name, d) in enumerate(sorted_feats[:20], 1):
        eff = "LARGE" if d > 0.8 else "Medium" if d > 0.5 else "Small" if d > 0.2 else "Weak"
        print(f"  {rank:4d} {name:<40} {d:8.4f} {eff}")

    return d_vals


def plot_violin_full(real_df, fake_df, feat_cols, d_vals, output_dir):
    """PLOT 2: Violin plots for top 12 features by Cohen's d."""
    sorted_feats = sorted(d_vals.items(), key=lambda x: x[1], reverse=True)
    top_feats = [x[0] for x in sorted_feats[:12]]

    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    fig.suptitle(f"Feature Distributions — Top 12 by Cohen's d\n"
                 f"(n={len(real_df)} real + {len(fake_df)} fake)",
                 fontsize=14, fontweight="bold")

    for idx, feat in enumerate(top_feats):
        ax = axes[idx // 4, idx % 4]
        r_vals = real_df[feat].dropna().values
        f_vals = fake_df[feat].dropna().values

        # Subsample for violin if too many
        if len(r_vals) > 500:
            r_sub = np.random.choice(r_vals, 500, replace=False)
        else:
            r_sub = r_vals
        if len(f_vals) > 500:
            f_sub = np.random.choice(f_vals, 500, replace=False)
        else:
            f_sub = f_vals

        parts = ax.violinplot([r_sub, f_sub], positions=[1, 2],
                               showmeans=True, showmedians=True)
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(REAL_COLOR if i == 0 else FAKE_COLOR)
            pc.set_alpha(0.4)

        d = d_vals[feat]
        t_stat, p_val = sp_stats.ttest_ind(r_vals[:500], f_vals[:500])
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"

        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Real", "Fake"], fontsize=8)
        short_name = feat.replace("t_", "T:").replace("s_", "S:")
        ax.set_title(f"{short_name}\nd={d:.3f} {sig}", fontsize=9)

    plt.tight_layout()
    path = os.path.join(output_dir, "02_violin_top12.png")
    plt.savefig(path, dpi=250, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_kde_full(real_df, fake_df, feat_cols, d_vals, output_dir):
    """PLOT 3: KDE distribution overlay for top 12 features."""
    sorted_feats = sorted(d_vals.items(), key=lambda x: x[1], reverse=True)
    top_feats = [x[0] for x in sorted_feats[:12]]

    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    fig.suptitle("PDF/KDE Distributions — Real vs Fake (Top 12 Features)",
                 fontsize=14, fontweight="bold")

    for idx, feat in enumerate(top_feats):
        ax = axes[idx // 4, idx % 4]
        r_vals = real_df[feat].dropna().values
        f_vals = fake_df[feat].dropna().values

        # Clip outliers for better visualization
        combined = np.concatenate([r_vals, f_vals])
        lo, hi = np.percentile(combined, [2, 98])
        r_clip = r_vals[(r_vals >= lo) & (r_vals <= hi)]
        f_clip = f_vals[(f_vals >= lo) & (f_vals <= hi)]

        if len(r_clip) > 10 and len(f_clip) > 10:
            bins = np.linspace(lo, hi, 60)
            ax.hist(r_clip, bins=bins, density=True, alpha=0.4, color=REAL_COLOR, label="Real")
            ax.hist(f_clip, bins=bins, density=True, alpha=0.4, color=FAKE_COLOR, label="Fake")

            # KDE overlay
            try:
                from scipy.stats import gaussian_kde
                x_range = np.linspace(lo, hi, 200)
                kde_r = gaussian_kde(r_clip)
                kde_f = gaussian_kde(f_clip)
                ax.plot(x_range, kde_r(x_range), color=REAL_COLOR, linewidth=2)
                ax.plot(x_range, kde_f(x_range), color=FAKE_COLOR, linewidth=2)
            except Exception:
                pass

        d = d_vals[feat]
        short_name = feat.replace("t_", "T:").replace("s_", "S:")
        ax.set_title(f"{short_name} (d={d:.3f})", fontsize=9)
        ax.legend(fontsize=7)
        ax.set_ylabel("Density", fontsize=7)

    plt.tight_layout()
    path = os.path.join(output_dir, "03_kde_distributions_top12.png")
    plt.savefig(path, dpi=250, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_pillar_analysis(real_df, fake_df, feat_cols, d_vals, output_dir):
    """PLOT 4: Per-pillar average and max Cohen's d."""
    pillars = {
        "P1: Noise": [f for f in feat_cols if "noise" in f and f.startswith("s_")],
        "P2: PRNU": [f for f in feat_cols if "prnu" in f and f.startswith("s_")],
        "P4: Shadow": [f for f in feat_cols if "shadow" in f or f == "s_face_bg_diff"],
        "P6: Compress": [f for f in feat_cols if any(k in f for k in ["benford","block","dbl"]) and f.startswith("s_")],
        "P8: Blur": [f for f in feat_cols if f == "s_blur_mag"],
        "P9: Flow": [f for f in feat_cols if "flow" in f and f.startswith("s_")],
        "T1: Noise\nstability": [f for f in feat_cols if "noise" in f and f.startswith("t_")],
        "T2: rPPG": [f for f in feat_cols if "rppg" in f],
        "T3: PRNU\ntemporal": [f for f in feat_cols if "prnu" in f and f.startswith("t_")],
        "T4: Face\nSSIM": [f for f in feat_cols if "ssim" in f],
        "T5: Codec\nresidual": [f for f in feat_cols if "residual" in f],
        "T6: Landmark": [f for f in feat_cols if any(k in f for k in ["landmark","jitter","accel","velocity","jaw"])],
        "T7: Rigid\ngeometry": [f for f in feat_cols if any(k in f for k in ["rigid","interpupillary","nose_bridge"])],
        "T8: Boundary": [f for f in feat_cols if "boundary" in f],
        "T9: Skin\ntexture": [f for f in feat_cols if "skin_texture" in f or "texture_warp" in f],
        "T10: Color\ntransfer": [f for f in feat_cols if "skin_color" in f or "skin_bg" in f],
        "T11: Specular": [f for f in feat_cols if "specular" in f],
        "T12: Blink": [f for f in feat_cols if "blink" in f],
        "T13: Motion\nblur": [f for f in feat_cols if "coupling" in f or "motion_blur" in f],
        "T14: DCT": [f for f in feat_cols if "dct" in f and f.startswith("t_")],
    }

    pillar_names = []
    avg_d = []
    max_d = []
    pillar_type = []

    for pname, feats in pillars.items():
        if not feats:
            continue
        ds = [d_vals.get(f, 0) for f in feats]
        pillar_names.append(pname)
        avg_d.append(np.mean(ds))
        max_d.append(np.max(ds))
        pillar_type.append("spatial" if pname.startswith("P") else "temporal")

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(pillar_names))
    colors_avg = [REAL_COLOR if t == "spatial" else "#1D9E75" for t in pillar_type]
    colors_max = ["#6BAED6" if t == "spatial" else "#74C476" for t in pillar_type]

    ax.bar(x - 0.15, avg_d, 0.3, label="Avg Cohen's d", color=colors_avg, alpha=0.8)
    ax.bar(x + 0.15, max_d, 0.3, label="Max Cohen's d", color=colors_max, alpha=0.8)
    ax.axhline(y=0.8, color="red", linestyle="--", alpha=0.4, label="Large effect")
    ax.axhline(y=0.5, color="orange", linestyle="--", alpha=0.3, label="Medium effect")
    ax.set_xticks(x)
    ax.set_xticklabels(pillar_names, fontsize=8)
    ax.set_ylabel("Cohen's d")
    n_real, n_fake = len(real_df), len(fake_df)
    ax.set_title(f"Per-Pillar Discriminative Strength\n"
                 f"(Blue = Spatial pillars, Green = Temporal pillars | "
                 f"n={n_real} real + {n_fake} fake)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    path = os.path.join(output_dir, "04_pillar_analysis_full.png")
    plt.savefig(path, dpi=250, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_box_grid(real_df, fake_df, d_vals, output_dir):
    """PLOT 5: Box plot grid — top 16 features with p-values."""
    sorted_feats = sorted(d_vals.items(), key=lambda x: x[1], reverse=True)
    top_feats = [x[0] for x in sorted_feats[:16]]

    fig, axes = plt.subplots(4, 4, figsize=(16, 12))
    fig.suptitle(f"Box Plots — Top 16 Features by Cohen's d\n"
                 f"(n={len(real_df)} real + {len(fake_df)} fake)",
                 fontsize=14, fontweight="bold")

    for idx, feat in enumerate(top_feats):
        ax = axes[idx // 4, idx % 4]
        r = real_df[feat].dropna().values
        f = fake_df[feat].dropna().values

        bp = ax.boxplot([r, f], labels=["Real", "Fake"], patch_artist=True, widths=0.5)
        bp['boxes'][0].set_facecolor(REAL_COLOR)
        bp['boxes'][0].set_alpha(0.4)
        bp['boxes'][1].set_facecolor(FAKE_COLOR)
        bp['boxes'][1].set_alpha(0.4)

        d = d_vals[feat]
        _, p = sp_stats.mannwhitneyu(r[:500], f[:500], alternative='two-sided')
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"

        short = feat.replace("t_", "T:").replace("s_", "S:")
        ax.set_title(f"{short}\nd={d:.3f} p={p:.1e} {sig}", fontsize=8)

    plt.tight_layout()
    path = os.path.join(output_dir, "05_box_plots_top16.png")
    plt.savefig(path, dpi=250, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_spatial_vs_temporal(d_vals, feat_cols, output_dir):
    """PLOT 6: Spatial vs temporal feature strength comparison."""
    spatial_d = [d_vals[f] for f in feat_cols if f.startswith("s_")]
    temporal_d = [d_vals[f] for f in feat_cols if f.startswith("t_")]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Box comparison
    bp = ax1.boxplot([spatial_d, temporal_d], labels=["Spatial (13)", "Temporal (37)"],
                      patch_artist=True, widths=0.5)
    bp['boxes'][0].set_facecolor(REAL_COLOR)
    bp['boxes'][0].set_alpha(0.4)
    bp['boxes'][1].set_facecolor("#1D9E75")
    bp['boxes'][1].set_alpha(0.4)
    ax1.scatter(np.ones(len(spatial_d)) + np.random.randn(len(spatial_d))*0.04,
                spatial_d, color=REAL_COLOR, alpha=0.6, s=30, zorder=3)
    ax1.scatter(np.ones(len(temporal_d))*2 + np.random.randn(len(temporal_d))*0.04,
                temporal_d, color="#1D9E75", alpha=0.6, s=30, zorder=3)
    ax1.set_ylabel("Cohen's d")
    ax1.set_title("Spatial vs Temporal Feature Strength")
    ax1.axhline(y=0.8, color="red", linestyle="--", alpha=0.3)

    # Count by effect size
    categories = ["Large\n(d>0.8)", "Medium\n(d>0.5)", "Small\n(d>0.2)", "Weak\n(d<0.2)"]
    s_counts = [sum(1 for d in spatial_d if d > 0.8), sum(1 for d in spatial_d if 0.5 < d <= 0.8),
                sum(1 for d in spatial_d if 0.2 < d <= 0.5), sum(1 for d in spatial_d if d <= 0.2)]
    t_counts = [sum(1 for d in temporal_d if d > 0.8), sum(1 for d in temporal_d if 0.5 < d <= 0.8),
                sum(1 for d in temporal_d if 0.2 < d <= 0.5), sum(1 for d in temporal_d if d <= 0.2)]

    x = np.arange(len(categories))
    ax2.bar(x - 0.17, s_counts, 0.3, label="Spatial", color=REAL_COLOR, alpha=0.8)
    ax2.bar(x + 0.17, t_counts, 0.3, label="Temporal", color="#1D9E75", alpha=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories, fontsize=9)
    ax2.set_ylabel("Number of features")
    ax2.set_title("Feature Count by Effect Size")
    ax2.legend()

    plt.tight_layout()
    path = os.path.join(output_dir, "06_spatial_vs_temporal.png")
    plt.savefig(path, dpi=250, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ============================================================================
# PART 2: TEMPORAL TRACE EXTRACTION (subsample of videos)
# ============================================================================

def load_video_data(video_path, max_frames=150):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0: fps = 30.0
    frames = []
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret: break
        frames.append(frame)
    cap.release()
    if len(frames) < 30: return None

    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    rgbs = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
    fm = init_face_mesh()
    lms = [get_landmarks(fm, r) for r in rgbs]
    fm.close()
    if sum(1 for l in lms if l is not None) / len(frames) < 0.5: return None
    return {"bgr": frames, "gray": grays, "rgb": rgbs, "landmarks": lms, "fps": fps, "n": len(frames)}


def extract_trace_noise_corr(data):
    corrs = []
    for i in range(1, data["n"]):
        lm, lm_p = data["landmarks"][i], data["landmarks"][i-1]
        if lm is None or lm_p is None: corrs.append(np.nan); continue
        mask = get_face_mask(lm, data["gray"][i].shape)
        r1 = (data["gray"][i-1].astype(float) - cv2.medianBlur(data["gray"][i-1],5).astype(float)) * mask
        r2 = (data["gray"][i].astype(float) - cv2.medianBlur(data["gray"][i],5).astype(float)) * mask
        v = (r1.flatten()!=0) & (r2.flatten()!=0)
        if v.sum() > 200:
            c = np.corrcoef(r1.flatten()[v], r2.flatten()[v])[0,1]
            corrs.append(c if np.isfinite(c) else np.nan)
        else: corrs.append(np.nan)
    return np.array(corrs)


def extract_trace_rppg_spectrum(data):
    """Returns (waveform, freqs, psd)."""
    rgb_sigs = []
    for f, lm in zip(data["rgb"], data["landmarks"]):
        if lm is None: rgb_sigs.append([0,0,0]); continue
        clean = cv2.bilateralFilter(f, d=7, sigmaColor=25, sigmaSpace=25)
        rgb_sigs.append(get_roi_mean_rgb(clean, lm, FOREHEAD).tolist())

    arr = np.array(rgb_sigs, dtype=np.float64)
    fps = data["fps"]
    nyq = fps / 2.0
    if nyq <= 4.0 or len(arr) < 60: return None, None, None

    win = max(20, min(int(1.6*fps), len(arr)))
    pulse = np.zeros(len(arr))
    for t in range(0, len(arr)-win+1):
        seg = arr[t:t+win]
        means = seg.mean(axis=0, keepdims=True)
        means[means<1] = 1
        Cn = seg / means
        S1 = Cn[:,1] - Cn[:,2]
        S2 = Cn[:,1] + Cn[:,2] - 2*Cn[:,0]
        std2 = S2.std()
        if std2 < 1e-10: continue
        h = S1 + (S1.std()/std2)*S2
        pulse[t:t+win] += h - h.mean()

    ov = np.minimum(np.arange(1,len(pulse)+1), np.minimum(win, np.arange(len(pulse),0,-1))).astype(float)
    ov[ov<1]=1
    pulse /= ov

    b, a = sp_signal.butter(3, [0.7/nyq, min(4.0/nyq, 0.99)], btype='band')
    try: filtered = sp_signal.filtfilt(b, a, pulse)
    except: return None, None, None

    nperseg = min(len(filtered), int(4*fps))
    if nperseg < 16: nperseg = len(filtered)
    try: freqs, psd = sp_signal.welch(filtered, fs=fps, nperseg=nperseg, noverlap=nperseg//2)
    except: return None, None, None

    return filtered, freqs, psd


def extract_trace_ipd(data):
    vals = []
    for lm in data["landmarks"]:
        if lm is None: vals.append(np.nan); continue
        fs = get_face_size(lm)
        if fs < 1: fs = 1
        le = lm[[33,133]].mean(axis=0)
        re = lm[[362,263]].mean(axis=0)
        vals.append(np.linalg.norm(le-re)/fs)
    return np.array(vals)


def extract_trace_skin_texture(data):
    corrs = []
    prev = None
    for g, lm in zip(data["gray"], data["landmarks"]):
        if lm is None: corrs.append(np.nan); prev=None; continue
        c = lm[RIGHT_CHEEK].mean(axis=0).astype(int)
        y,x = c[1],c[0]
        p = g[max(0,y-16):min(g.shape[0],y+16), max(0,x-16):min(g.shape[1],x+16)]
        if p.shape[0]<8 or p.shape[1]<8: corrs.append(np.nan); prev=None; continue
        p = cv2.resize(p,(32,32)).astype(np.float64)
        if prev is not None:
            cc = np.corrcoef(p.flatten(), prev.flatten())[0,1]
            corrs.append(cc if np.isfinite(cc) else np.nan)
        else: corrs.append(np.nan)
        prev = p
    return np.array(corrs)


def extract_trace_skin_color(data):
    vals = []
    for bgr, lm in zip(data["bgr"], data["landmarks"]):
        if lm is None: vals.append(np.nan); continue
        mask = get_face_mask(lm, bgr.shape)
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float64)
        pix = lab[mask>0]
        vals.append(pix[:,0].mean() if len(pix)>50 else np.nan)
    return np.array(vals)


def extract_trace_landmark_jitter(data):
    vals = []
    key = FACE_OVAL[:20] + NOSE_BRIDGE
    for i in range(1, data["n"]):
        lm, lp = data["landmarks"][i], data["landmarks"][i-1]
        if lm is None or lp is None: vals.append(np.nan); continue
        fs = get_face_size(lm)
        if fs < 1: fs = 1
        vals.append(np.linalg.norm(lm[key]-lp[key], axis=1).mean()/fs)
    return np.array(vals)


def process_trace_videos(video_dir, n_videos, max_frames):
    vids = sorted(glob.glob(os.path.join(video_dir, "*.mp4")))
    if not vids:
        vids = sorted(glob.glob(os.path.join(video_dir, "**", "*.mp4"), recursive=True))
    selected = random.sample(vids, min(n_videos, len(vids)))
    results = []
    for vp in selected:
        name = os.path.basename(vp)
        print(f"    {name}...", end=" ", flush=True)
        data = load_video_data(vp, max_frames)
        if data is None: print("SKIP"); continue
        r = {"fps": data["fps"]}
        r["noise_corr"] = extract_trace_noise_corr(data)
        r["rppg_wave"], r["rppg_freqs"], r["rppg_psd"] = extract_trace_rppg_spectrum(data)
        r["ipd"] = extract_trace_ipd(data)
        r["skin_texture"] = extract_trace_skin_texture(data)
        r["skin_color"] = extract_trace_skin_color(data)
        r["landmark_jitter"] = extract_trace_landmark_jitter(data)
        results.append(r)
        print("OK")
    return results


def plot_rppg_spectrum(real_traces, fake_traces, output_dir):
    """PLOT 7: rPPG power spectrum — THE key discriminative plot."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: individual spectra
    ax = axes[0]
    for r in real_traces:
        if r["rppg_freqs"] is not None:
            m = (r["rppg_freqs"]>=0.5) & (r["rppg_freqs"]<=4.5)
            p = r["rppg_psd"][m]
            p = p / (p.max()+1e-10)
            ax.plot(r["rppg_freqs"][m], p, color=REAL_COLOR, alpha=0.3, lw=1)
    for f in fake_traces:
        if f["rppg_freqs"] is not None:
            m = (f["rppg_freqs"]>=0.5) & (f["rppg_freqs"]<=4.5)
            p = f["rppg_psd"][m]
            p = p / (p.max()+1e-10)
            ax.plot(f["rppg_freqs"][m], p, color=FAKE_COLOR, alpha=0.3, lw=1)
    ax.axvspan(0.75, 3.0, alpha=0.08, color='green', label="Cardiac band")
    ax.plot([],[],color=REAL_COLOR,lw=2,label=f"Real (n={len(real_traces)})")
    ax.plot([],[],color=FAKE_COLOR,lw=2,label=f"Fake (n={len(fake_traces)})")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Normalized PSD")
    ax.set_title("Individual rPPG Spectra")
    ax.legend(fontsize=8)

    # Right: average ± std
    ax = axes[1]
    def avg_spec(traces):
        f_common = np.linspace(0.5, 4.5, 100)
        all_p = []
        for t in traces:
            if t["rppg_freqs"] is not None:
                m = (t["rppg_freqs"]>=0.5) & (t["rppg_freqs"]<=4.5)
                ff, pp = t["rppg_freqs"][m], t["rppg_psd"][m]
                if len(ff)>3:
                    pp = pp/(pp.max()+1e-10)
                    all_p.append(np.interp(f_common, ff, pp))
        if not all_p: return f_common, np.zeros(100), np.zeros(100)
        a = np.array(all_p)
        return f_common, a.mean(0), a.std(0)

    fr, mr, sr = avg_spec(real_traces)
    ff, mf, sf = avg_spec(fake_traces)
    ax.fill_between(fr, mr-sr, mr+sr, color=REAL_COLOR, alpha=0.15)
    ax.plot(fr, mr, color=REAL_COLOR, lw=2.5, label="Real (mean ± std)")
    ax.fill_between(ff, mf-sf, mf+sf, color=FAKE_COLOR, alpha=0.15)
    ax.plot(ff, mf, color=FAKE_COLOR, lw=2.5, label="Fake (mean ± std)")
    ax.axvspan(0.75, 3.0, alpha=0.08, color='green')
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Normalized PSD")
    ax.set_title("Average rPPG Spectrum (Real vs Fake)")
    ax.legend(fontsize=8)

    plt.tight_layout()
    path = os.path.join(output_dir, "07_rppg_power_spectrum.png")
    plt.savefig(path, dpi=250, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_temporal_traces_grid(real_traces, fake_traces, output_dir):
    """PLOT 8: 2x3 grid of temporal traces with mean ± std bands."""
    features = [
        ("noise_corr", "T1: Noise Fingerprint Correlation", "Pearson r"),
        ("skin_texture", "T9: Skin Texture Persistence", "Pearson r"),
        ("ipd", "T7: Interpupillary Distance (rigid)", "Normalized IPD"),
        ("skin_color", "T10: Skin Luminance Stability", "LAB L*"),
        ("landmark_jitter", "T6: Landmark Displacement", "Normalized disp."),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(13, 10))
    fig.suptitle(f"Temporal Physics Consistency — Mean ± Std Bands\n"
                 f"(n={len(real_traces)} real + {len(fake_traces)} fake)",
                 fontsize=14, fontweight="bold", y=1.01)

    fps = real_traces[0]["fps"] if real_traces else 30

    for idx, (key, title, ylabel) in enumerate(features):
        ax = axes[idx//2, idx%2]

        def band(traces):
            series = [t[key] for t in traces]
            ml = max(len(s) for s in series)
            pad = np.full((len(series), ml), np.nan)
            for i, s in enumerate(series): pad[i,:len(s)] = s
            return np.nanmean(pad, 0), np.nanstd(pad, 0)

        rm, rs = band(real_traces)
        fm, fs_v = band(fake_traces)
        tr = np.arange(len(rm))/fps
        tf = np.arange(len(fm))/fps

        ax.fill_between(tr, rm-rs, rm+rs, color=REAL_COLOR, alpha=0.15)
        ax.plot(tr, rm, color=REAL_COLOR, lw=2, label="Real")
        ax.fill_between(tf, fm-fs_v, fm+fs_v, color=FAKE_COLOR, alpha=0.15)
        ax.plot(tf, fm, color=FAKE_COLOR, lw=2, label="Fake")
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xlabel("Time (s)", fontsize=9)
        ax.legend(fontsize=8)

    # Last subplot: rPPG waveform (z-scored)
    ax = axes[2, 1]
    for r in real_traces[:5]:
        if r["rppg_wave"] is not None:
            w = (r["rppg_wave"] - r["rppg_wave"].mean()) / (r["rppg_wave"].std()+1e-10)
            t = np.arange(len(w))/fps
            ax.plot(t, w, color=REAL_COLOR, alpha=0.3, lw=0.8)
    for f in fake_traces[:5]:
        if f["rppg_wave"] is not None:
            w = (f["rppg_wave"] - f["rppg_wave"].mean()) / (f["rppg_wave"].std()+1e-10)
            t = np.arange(len(w))/fps
            ax.plot(t, w, color=FAKE_COLOR, alpha=0.3, lw=0.8)
    ax.plot([],[],color=REAL_COLOR,lw=2,label="Real")
    ax.plot([],[],color=FAKE_COLOR,lw=2,label="Fake")
    ax.set_title("T2: rPPG Waveform (z-scored)", fontsize=10)
    ax.set_ylabel("Amplitude (z)", fontsize=9)
    ax.set_xlabel("Time (s)", fontsize=9)
    ax.set_ylim(-4, 4)
    ax.legend(fontsize=8)

    plt.tight_layout()
    path = os.path.join(output_dir, "08_temporal_traces_grid.png")
    plt.savefig(path, dpi=250, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_rppg_snr_box(real_traces, fake_traces, output_dir):
    """PLOT 9: rPPG SNR box plot with statistical test."""
    def snr(t):
        if t["rppg_freqs"] is None: return 0.0
        m = (t["rppg_freqs"]>=0.7) & (t["rppg_freqs"]<=4.0)
        cp = t["rppg_psd"][m]
        return cp.max()/(cp.mean()+1e-10) if len(cp)>3 else 0.0

    rs = [snr(r) for r in real_traces]
    fs = [snr(f) for f in fake_traces]

    fig, ax = plt.subplots(figsize=(6, 5))
    bp = ax.boxplot([rs, fs], labels=["Real", "Fake"], patch_artist=True, widths=0.5)
    bp['boxes'][0].set_facecolor(REAL_COLOR); bp['boxes'][0].set_alpha(0.4)
    bp['boxes'][1].set_facecolor(FAKE_COLOR); bp['boxes'][1].set_alpha(0.4)
    ax.scatter(np.ones(len(rs))+np.random.randn(len(rs))*0.04, rs, color=REAL_COLOR, alpha=0.5, s=25, zorder=3)
    ax.scatter(np.ones(len(fs))*2+np.random.randn(len(fs))*0.04, fs, color=FAKE_COLOR, alpha=0.5, s=25, zorder=3)

    d = compute_cohens_d(np.array(rs), np.array(fs))
    _, p = sp_stats.mannwhitneyu(rs, fs, alternative='two-sided') if len(rs)>1 and len(fs)>1 else (0, 1)
    ax.set_title(f"rPPG Cardiac SNR\n(Cohen's d = {d:.3f}, p = {p:.4f})")
    ax.set_ylabel("Signal-to-Noise Ratio")
    plt.tight_layout()
    path = os.path.join(output_dir, "09_rppg_snr_boxplot.png")
    plt.savefig(path, dpi=250, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real_csv", required=True)
    parser.add_argument("--fake_csv", required=True)
    parser.add_argument("--real_dir", required=True)
    parser.add_argument("--fake_dir", required=True)
    parser.add_argument("--output_dir", default="results/full_publication_plots")
    parser.add_argument("--trace_videos", type=int, default=50)
    parser.add_argument("--max_frames", type=int, default=150)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ================================================================
    # PART 1: Full CSV statistical analysis
    # ================================================================
    print("=" * 60)
    print("PART 1: FULL CSV STATISTICAL ANALYSIS")
    print("=" * 60)

    real_df, feat_cols = load_csv([args.real_csv])
    fake_df, _ = load_csv([args.fake_csv])
    print(f"  Loaded: {len(real_df)} real + {len(fake_df)} fake, {len(feat_cols)} features\n")

    print("  Generating Cohen's d ranking...")
    d_vals = plot_full_cohens_d(real_df, fake_df, feat_cols, args.output_dir)

    print("\n  Generating violin plots...")
    plot_violin_full(real_df, fake_df, feat_cols, d_vals, args.output_dir)

    print("  Generating KDE distributions...")
    plot_kde_full(real_df, fake_df, feat_cols, d_vals, args.output_dir)

    print("  Generating pillar analysis...")
    plot_pillar_analysis(real_df, fake_df, feat_cols, d_vals, args.output_dir)

    print("  Generating box plots...")
    plot_box_grid(real_df, fake_df, d_vals, args.output_dir)

    print("  Generating spatial vs temporal comparison...")
    plot_spatial_vs_temporal(d_vals, feat_cols, args.output_dir)

    # ================================================================
    # PART 2: Temporal trace extraction (subsample)
    # ================================================================
    print(f"\n{'=' * 60}")
    print(f"PART 2: TEMPORAL TRACES ({args.trace_videos} videos per class)")
    print("=" * 60)

    print(f"\n  Processing REAL videos...")
    real_traces = process_trace_videos(args.real_dir, args.trace_videos, args.max_frames)

    print(f"\n  Processing FAKE videos...")
    fake_traces = process_trace_videos(args.fake_dir, args.trace_videos, args.max_frames)

    if len(real_traces) >= 2 and len(fake_traces) >= 2:
        print(f"\n  Generating rPPG spectrum plot...")
        plot_rppg_spectrum(real_traces, fake_traces, args.output_dir)

        print("  Generating temporal traces grid...")
        plot_temporal_traces_grid(real_traces, fake_traces, args.output_dir)

        print("  Generating rPPG SNR box plot...")
        plot_rppg_snr_box(real_traces, fake_traces, args.output_dir)
    else:
        print("  WARNING: Not enough trace videos processed")

    # ================================================================
    # Summary
    # ================================================================
    print(f"\n{'=' * 60}")
    print(f"ALL PLOTS SAVED TO: {args.output_dir}")
    print(f"{'=' * 60}")
    print(f"\n  Part 1 (full CSV, {len(real_df)+len(fake_df)} videos):")
    print(f"    01_cohens_d_full_dataset.png")
    print(f"    02_violin_top12.png")
    print(f"    03_kde_distributions_top12.png")
    print(f"    04_pillar_analysis_full.png")
    print(f"    05_box_plots_top16.png")
    print(f"    06_spatial_vs_temporal.png")
    print(f"\n  Part 2 (traces, {len(real_traces)}+{len(fake_traces)} videos):")
    print(f"    07_rppg_power_spectrum.png")
    print(f"    08_temporal_traces_grid.png")
    print(f"    09_rppg_snr_boxplot.png")


if __name__ == "__main__":
    main()