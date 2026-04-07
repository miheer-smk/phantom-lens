"""
rPPG Diagnostic — Celeb-DF v2 Quick Sanity Check
==================================================
Loads 20 real + 20 fake videos, extracts rPPG features,
and reports whether the extractor can distinguish them.

The key question: does pulse_snr behave as theory predicts
(real > fake), or is it inverted as previously observed?

Usage:
    python rppg_diagnostic.py
    python rppg_diagnostic.py --real_dir /path/to/real --fake_dir /path/to/fake
    python rppg_diagnostic.py --n_videos 50 --n_frames 128
"""

import os
import sys
import glob
import argparse
import numpy as np

try:
    import imageio.v3 as iio
    USE_V3 = True
except ImportError:
    import imageio
    USE_V3 = False

import cv2

# Import our extractor (assumes rppg_extractor.py is in same dir or on PYTHONPATH)
from rppg_extractor import compute_rppg_features


# ---------------------------------------------------------------------------
# Video loading
# ---------------------------------------------------------------------------
def get_video_fps(video_path):
    """
    Read actual fps from video metadata using imageio.
    Falls back to 25.0 if metadata is unavailable.
    """
    try:
        if USE_V3:
            meta = iio.immeta(video_path, plugin="pyav")
            fps = meta.get("fps", None)
            if fps and fps > 0:
                return float(fps)
        else:
            reader = imageio.get_reader(video_path)
            meta = reader.get_meta_data()
            reader.close()
            fps = meta.get("fps", None)
            if fps and fps > 0:
                return float(fps)
    except Exception as e:
        print(f"    [WARN] Could not read fps from {os.path.basename(video_path)}: {e}")
    return 25.0  # fallback


def load_frames(video_path, n_frames=64, target_size=(224, 224)):
    """
    Load n_frames evenly spaced frames from a video file.
    Returns list of BGR numpy arrays resized to target_size.
    """
    try:
        if USE_V3:
            # Read all frames first, then subsample
            all_frames = iio.imread(video_path, plugin="pyav")
        else:
            reader = imageio.get_reader(video_path)
            all_frames = []
            for frame in reader:
                all_frames.append(frame)
            reader.close()
            all_frames = np.array(all_frames)
    except Exception as e:
        print(f"    [ERROR] Could not read {os.path.basename(video_path)}: {e}")
        return []

    total = len(all_frames)
    if total == 0:
        return []

    # Evenly space n_frames across the video
    if total <= n_frames:
        indices = np.arange(total)
    else:
        indices = np.linspace(0, total - 1, n_frames, dtype=int)

    frames_bgr = []
    for idx in indices:
        frame_rgb = all_frames[idx]
        # imageio returns RGB, convert to BGR for our extractor
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        # Resize to target
        frame_bgr = cv2.resize(frame_bgr, target_size, interpolation=cv2.INTER_LINEAR)
        frames_bgr.append(frame_bgr)

    return frames_bgr


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def cohens_d(group_a, group_b):
    """
    Compute Cohen's d effect size between two groups.
    Positive d means group_a > group_b.
    """
    na, nb = len(group_a), len(group_b)
    if na < 2 or nb < 2:
        return 0.0
    mean_a, mean_b = np.mean(group_a), np.mean(group_b)
    var_a, var_b = np.var(group_a, ddof=1), np.var(group_b, ddof=1)
    # Pooled standard deviation
    pooled_std = np.sqrt(((na - 1) * var_a + (nb - 1) * var_b) / (na + nb - 2))
    if pooled_std < 1e-12:
        return 0.0
    return (mean_a - mean_b) / pooled_std


# ---------------------------------------------------------------------------
# Main diagnostic
# ---------------------------------------------------------------------------
def run_diagnostic(real_dir, fake_dir, n_videos=20, n_frames=64):
    # ---- Discover video files ----
    video_exts = ("*.mp4", "*.avi", "*.mkv", "*.mov", "*.webm")

    real_videos = []
    for ext in video_exts:
        real_videos.extend(glob.glob(os.path.join(real_dir, ext)))
    real_videos.sort()

    fake_videos = []
    for ext in video_exts:
        fake_videos.extend(glob.glob(os.path.join(fake_dir, ext)))
    fake_videos.sort()

    if len(real_videos) == 0:
        print(f"[FATAL] No video files found in {real_dir}")
        sys.exit(1)
    if len(fake_videos) == 0:
        print(f"[FATAL] No video files found in {fake_dir}")
        sys.exit(1)

    # Cap to n_videos
    real_videos = real_videos[:n_videos]
    fake_videos = fake_videos[:n_videos]

    print(f"Found {len(real_videos)} real, {len(fake_videos)} fake videos")
    print(f"Using {n_frames} frames per video")
    print()

    # ---- Extract features ----
    feature_names = ["pulse_snr", "spectral_entropy", "spatial_coherence"]

    real_features = []  # list of (3,) arrays
    fake_features = []

    print("Processing REAL videos:")
    for i, vpath in enumerate(real_videos):
        fname = os.path.basename(vpath)
        fps = get_video_fps(vpath)
        frames = load_frames(vpath, n_frames=n_frames)
        if len(frames) < 16:
            print(f"  [{i+1:2d}/{len(real_videos)}] {fname} — SKIPPED (only {len(frames)} frames)")
            continue
        feats = compute_rppg_features(frames, fps=fps)
        real_features.append(feats)
        print(f"  [{i+1:2d}/{len(real_videos)}] {fname}  fps={fps:.1f}  frames={len(frames):3d}  "
              f"snr={feats[0]:.4f}  ent={feats[1]:.4f}  coh={feats[2]:.4f}")

    print()
    print("Processing FAKE videos:")
    for i, vpath in enumerate(fake_videos):
        fname = os.path.basename(vpath)
        fps = get_video_fps(vpath)
        frames = load_frames(vpath, n_frames=n_frames)
        if len(frames) < 16:
            print(f"  [{i+1:2d}/{len(fake_videos)}] {fname} — SKIPPED (only {len(frames)} frames)")
            continue
        feats = compute_rppg_features(frames, fps=fps)
        fake_features.append(feats)
        print(f"  [{i+1:2d}/{len(fake_videos)}] {fname}  fps={fps:.1f}  frames={len(frames):3d}  "
              f"snr={feats[0]:.4f}  ent={feats[1]:.4f}  coh={feats[2]:.4f}")

    # ---- Compute statistics ----
    real_arr = np.array(real_features)  # shape (N_real, 3)
    fake_arr = np.array(fake_features)  # shape (N_fake, 3)

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"  Videos processed:  {len(real_arr)} real,  {len(fake_arr)} fake")
    print()

    if len(real_arr) == 0 or len(fake_arr) == 0:
        print("[FATAL] Not enough videos processed. Check paths and video files.")
        sys.exit(1)

    # Per-feature statistics
    print(f"  {'Feature':<22s}  {'Real Mean':>10s}  {'Fake Mean':>10s}  {'Diff':>8s}  {'Cohen d':>9s}  {'Direction':>12s}")
    print(f"  {'-'*22}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*9}  {'-'*12}")

    any_discriminative = False
    results = []

    for j, fname in enumerate(feature_names):
        r_mean = real_arr[:, j].mean()
        f_mean = fake_arr[:, j].mean()
        diff = r_mean - f_mean
        d = cohens_d(real_arr[:, j], fake_arr[:, j])

        # Direction check — does it match theory?
        # Theory: real > fake for all three features
        if diff > 0:
            direction = "REAL > FAKE"
            matches_theory = True
        else:
            direction = "FAKE > REAL"
            matches_theory = False

        is_disc = abs(d) > 0.3
        if is_disc:
            any_discriminative = True

        results.append({
            "name": fname,
            "r_mean": r_mean,
            "f_mean": f_mean,
            "diff": diff,
            "d": d,
            "direction": direction,
            "matches_theory": matches_theory,
            "discriminative": is_disc,
        })

        print(f"  {fname:<22s}  {r_mean:10.4f}  {f_mean:10.4f}  {diff:+8.4f}  {d:+9.4f}  {direction:>12s}")

    # ---- Verdicts ----
    print()
    print("-" * 70)

    # Overall discriminability
    if any_discriminative:
        print("  VERDICT:  DISCRIMINATIVE  (|Cohen's d| > 0.3 for at least one feature)")
    else:
        print("  VERDICT:  NOT DISCRIMINATIVE  (|Cohen's d| <= 0.3 for all features)")

    print()

    # Theory alignment check — this is the critical diagnostic
    print("  THEORY ALIGNMENT CHECK:")
    for r in results:
        symbol = "OK" if r["matches_theory"] else "INVERTED"
        print(f"    {r['name']:<22s}  {r['direction']:<14s}  [{symbol}]")

    inverted = [r for r in results if not r["matches_theory"] and r["discriminative"]]
    if inverted:
        print()
        print("  WARNING: The following features are INVERTED from theory:")
        for r in inverted:
            print(f"    - {r['name']} (d={r['d']:+.4f})")
        print()
        print("  POSSIBLE CAUSES OF INVERSION:")
        print("    1. Celeb-DF fakes are high-quality re-enactments that preserve")
        print("       some pulse signal from the source video (not pure generation)")
        print("    2. Celeb-DF reals are heavily compressed YouTube videos where")
        print("       the pulse signal is degraded by aggressive quantisation")
        print("    3. The face ROI (central 50%) may miss the actual face in some")
        print("       videos — use landmark-based ROI for production pipeline")
        print("    4. The SNR sigmoid calibration (threshold=5.0, slope=0.3) may")
        print("       need tuning — raw snr_db values may sit in a different range")
        print()
        print("  RECOMMENDED NEXT STEPS:")
        print("    1. Run this diagnostic on FF++ real/fake (same codec) to isolate")
        print("       whether inversion is dataset-specific or feature-level")
        print("    2. Print raw snr_db values before sigmoid to check calibration")
        print("    3. Try landmark-based face ROI instead of central crop")
        print("    4. If still inverted on FF++, the feature needs redesign — consider")
        print("       using CHROM or POS algorithm instead of raw green channel")
    else:
        all_match = all(r["matches_theory"] for r in results)
        if all_match:
            print()
            print("  All features align with theory (real > fake). Good to proceed.")

    print()
    print("=" * 70)

    # ---- Print raw arrays for further analysis ----
    print()
    print("RAW FEATURE ARRAYS (copy-paste for further analysis):")
    print(f"  real_snr       = {list(np.round(real_arr[:, 0], 4))}")
    print(f"  real_entropy   = {list(np.round(real_arr[:, 1], 4))}")
    print(f"  real_coherence = {list(np.round(real_arr[:, 2], 4))}")
    print(f"  fake_snr       = {list(np.round(fake_arr[:, 0], 4))}")
    print(f"  fake_entropy   = {list(np.round(fake_arr[:, 1], 4))}")
    print(f"  fake_coherence = {list(np.round(fake_arr[:, 2], 4))}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="rPPG Diagnostic on Celeb-DF v2")
    parser.add_argument("--real_dir", type=str, default="data/celebdf/real/",
                        help="Path to real videos directory")
    parser.add_argument("--fake_dir", type=str, default="data/celebdf/fake/",
                        help="Path to fake videos directory")
    parser.add_argument("--n_videos", type=int, default=20,
                        help="Number of videos to process per class")
    parser.add_argument("--n_frames", type=int, default=64,
                        help="Number of frames to extract per video")
    args = parser.parse_args()

    print("rPPG Diagnostic — Celeb-DF v2")
    print(f"  Real dir:   {args.real_dir}")
    print(f"  Fake dir:   {args.fake_dir}")
    print(f"  N videos:   {args.n_videos}")
    print(f"  N frames:   {args.n_frames}")
    print()

    run_diagnostic(args.real_dir, args.fake_dir, args.n_videos, args.n_frames)