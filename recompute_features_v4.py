"""
Phantom Lens V4 — Feature Extraction Pipeline
===============================================
Builds a new 20-dimensional feature vector per video by:
  1. Retaining strong V3 pillars (P1 Noise, P2 PRNU, P4 Shadow,
     P6 DCT, P7 Codec, P9 Flow) — 14 dims
  2. Adding new codec-invariant pillars:
     - Blending Boundary Gradient — 3 dims
     - Frequency Checkerboard — 3 dims
  Total: 20 dimensions per video

Processes all dataset folders, saves checkpoints after each,
and supports resume if interrupted.

Usage:
    python recompute_features_v4.py
    python recompute_features_v4.py --n_frames 32 --workers 4
    python recompute_features_v4.py --resume  # continue from last checkpoint
"""

import os
import sys
import glob
import time
import pickle
import argparse
import traceback
import numpy as np
from pathlib import Path

try:
    import imageio.v3 as iio
    USE_V3_IMAGEIO = True
except ImportError:
    import imageio
    USE_V3_IMAGEIO = False

import cv2

# ---------------------------------------------------------------------------
# Import existing V3 pillar functions
# ---------------------------------------------------------------------------
# These are the exact same functions used in precompute_features_v3.py
# We import them directly to ensure feature consistency
try:
    from precompute_features_v3 import extract_features_from_video
    V3_AVAILABLE = True
except ImportError:
    print("[WARN] Could not import precompute_features_v3.")
    print("       V3 pillar functions not available.")
    print("       Place precompute_features_v3.py in the same directory.")
    V3_AVAILABLE = False

# Import new V4 pillar functions
try:
    from blend_boundary import compute_blend_boundary
    BLEND_AVAILABLE = True
except ImportError:
    print("[WARN] Could not import blend_boundary.py")
    BLEND_AVAILABLE = False

try:
    from freq_checkerboard import compute_freq_checkerboard
    FREQ_AVAILABLE = True
except ImportError:
    print("[WARN] Could not import freq_checkerboard.py")
    FREQ_AVAILABLE = False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEFAULT_N_FRAMES = 16
DEFAULT_FRAME_SIZE = (224, 224)
OUTPUT_PKL = "data/precomputed_features_v4.pkl"
CHECKPOINT_DIR = "data/v4_checkpoints"
V4_FEATURE_DIM = 20  # total dimensions

# Dataset definitions: (path, label, dataset_source, generator_type)
DATASET_FOLDERS = [
    {
        "path": "data/ffpp_official/original_sequences/youtube/c23/videos/",
        "label": 0,
        "source": "ffpp_official",
        "generator": "real",
        "description": "FF++ Real (YouTube)",
    },
    {
        "path": "data/ffpp_official/manipulated_sequences/Deepfakes/c23/videos/",
        "label": 1,
        "source": "ffpp_official",
        "generator": "Deepfakes",
        "description": "FF++ Deepfakes",
    },
    {
        "path": "data/ffpp_official/manipulated_sequences/Face2Face/c23/videos/",
        "label": 1,
        "source": "ffpp_official",
        "generator": "Face2Face",
        "description": "FF++ Face2Face",
    },
    {
        "path": "data/ffpp_official/manipulated_sequences/FaceSwap/c23/videos/",
        "label": 1,
        "source": "ffpp_official",
        "generator": "FaceSwap",
        "description": "FF++ FaceSwap",
    },
    {
        "path": "data/ffpp_official/manipulated_sequences/NeuralTextures/c23/videos/",
        "label": 1,
        "source": "ffpp_official",
        "generator": "NeuralTextures",
        "description": "FF++ NeuralTextures",
    },
    {
        "path": "data/ffpp_official/manipulated_sequences/FaceShifter/c23/videos/",
        "label": 1,
        "source": "ffpp_official",
        "generator": "FaceShifter",
        "description": "FF++ FaceShifter",
    },
    {
        "path": "data/celebvhq/35666/",
        "label": 0,
        "source": "celebvhq",
        "generator": "real",
        "description": "CelebVHQ Real",
    },
    {
        "path": "data/celebdf/fake/",
        "label": 1,
        "source": "celebdf_fake",
        "generator": "CelebDF",
        "description": "Celeb-DF Fake",
    },
]

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mkv", ".mov", ".webm")


# ---------------------------------------------------------------------------
# Video I/O
# ---------------------------------------------------------------------------
def get_video_fps(video_path):
    """Read actual fps from video metadata via imageio. Fallback: 25.0."""
    try:
        if USE_V3_IMAGEIO:
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
    except Exception:
        pass
    return 25.0


def load_frames(video_path, n_frames=DEFAULT_N_FRAMES,
                target_size=DEFAULT_FRAME_SIZE):
    """
    Load n_frames evenly spaced frames from a video.
    Returns list of BGR numpy arrays resized to target_size,
    or empty list on failure.
    """
    try:
        if USE_V3_IMAGEIO:
            all_frames = iio.imread(video_path, plugin="pyav")
        else:
            reader = imageio.get_reader(video_path)
            all_frames = []
            for frame in reader:
                all_frames.append(frame)
            reader.close()
            all_frames = np.array(all_frames)
    except Exception:
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
        if frame_rgb.ndim != 3:
            continue
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        frame_bgr = cv2.resize(frame_bgr, target_size,
                               interpolation=cv2.INTER_LINEAR)
        frames_bgr.append(frame_bgr)

    return frames_bgr


# ---------------------------------------------------------------------------
# V3 pillar extraction — kept dimensions only
# ---------------------------------------------------------------------------
# V3 feature vector layout (30 dims):
#   P1 Noise:     dims 0, 1, 2
#   P2 PRNU:      dims 3, 4
#   P3 Bayer:     dims 5, 6           ← DROPPED
#   P4 Shadow:    dims 7, 8, 9
#   P5 Specular:  dims 10, 11         ← DROPPED
#   P6 DCT:       dims 12, 13, 14
#   P7 Codec:     dims 15, 16
#   P8 Blur:      dims 17, 18         ← DROPPED
#   P9 Flow:      dim  19
#   P10 Chromatic: dims 20, 21        ← DROPPED
#   P11 EyeSym:   dims 22, 23, 24     ← DROPPED
#   P12 Illum:    dims 25, 26         ← DROPPED

# Indices we KEEP from the V3 feature vector
V3_KEEP_INDICES = [
    0, 1, 2,       # P1 Noise (3 dims)
    3, 4,           # P2 PRNU (2 dims)
    7, 8, 9,        # P4 Shadow (3 dims)
    12, 13, 14,     # P6 DCT (3 dims)
    15, 16,         # P7 Codec (2 dims)
    19,             # P9 Flow (1 dim)
]
# Total kept from V3: 14 dims


def extract_v3_kept_features(frames_bgr, video_path):
    """
    Extract the full 30-dim V3 feature vector for one video by calling
    the original extract_features_from_video(), then select only the
    14 kept dimensions.

    extract_features_from_video returns a list of 30-dim vectors
    (one per frame). We average across frames to get a single 30-dim
    vector, then index with V3_KEEP_INDICES.

    Returns numpy array shape (14,) or None on failure.
    """
    if not V3_AVAILABLE:
        return None

    try:
        # Call the original V3 extraction function
        # Returns list of 30-dim vectors, one per frame
        frame_vectors = extract_features_from_video(video_path, n_frames=16)

        if frame_vectors is None or len(frame_vectors) == 0:
            return None

        # Convert to numpy array: shape (N_frames, 30)
        frame_arr = np.array(frame_vectors, dtype=np.float64)

        if frame_arr.ndim == 1:
            # Single vector returned instead of list — use directly
            v3_full = frame_arr
        else:
            # Average across frames to get one 30-dim vector
            v3_full = frame_arr.mean(axis=0)

        # Validate: need at least 20 dims to index up to dim 19
        if v3_full.shape[0] < 20:
            print(f"    [WARN] V3 vector has {v3_full.shape[0]} dims, expected 30")
            v3_full = np.pad(v3_full, (0, max(0, 30 - v3_full.shape[0])),
                             constant_values=0.5)

        # Select only the kept dimensions
        kept = v3_full[V3_KEEP_INDICES]  # (14,)

        return kept.astype(np.float64)

    except Exception as e:
        print(f"    [ERROR] V3 extraction failed: {e}")
        return None


def extract_v3_kept_from_full_vector(v3_full_vector):
    """
    Alternative: if you already have the full 30-dim V3 vector
    (e.g. from an existing pkl), extract the kept dimensions.

    This is useful if you want to recompute V4 from existing V3 pkl
    without re-running the V3 pillar functions.
    """
    v3 = np.asarray(v3_full_vector, dtype=np.float64)
    if v3.shape[0] < 20:
        # V3 vector too short — pad
        v3 = np.pad(v3, (0, max(0, 30 - v3.shape[0])), constant_values=0.5)
    return v3[V3_KEEP_INDICES]


# ---------------------------------------------------------------------------
# New V4 pillar extraction
# ---------------------------------------------------------------------------
def extract_new_pillar_features(frames_bgr):
    """
    Extract the two new V4 pillars across all frames and aggregate
    by averaging.

    Returns numpy array shape (6,):
      [0-2] blend_boundary features (averaged across frames)
      [3-5] freq_checkerboard features (averaged across frames)

    Returns array of 0.5 on failure.
    """
    n_new = 6  # 3 blend + 3 freq
    neutral = np.full(n_new, 0.5, dtype=np.float64)

    if len(frames_bgr) == 0:
        return neutral

    blend_feats = []
    freq_feats = []

    for frame in frames_bgr:
        # Blending boundary — per-frame
        if BLEND_AVAILABLE:
            try:
                bf = compute_blend_boundary(frame)
                blend_feats.append(bf)
            except Exception:
                blend_feats.append(np.full(3, 0.5))
        else:
            blend_feats.append(np.full(3, 0.5))

        # Frequency checkerboard — per-frame
        if FREQ_AVAILABLE:
            try:
                ff = compute_freq_checkerboard(frame)
                freq_feats.append(ff)
            except Exception:
                freq_feats.append(np.full(3, 0.5))
        else:
            freq_feats.append(np.full(3, 0.5))

    # Aggregate: mean across frames
    blend_mean = np.mean(blend_feats, axis=0)  # (3,)
    freq_mean = np.mean(freq_feats, axis=0)    # (3,)

    combined = np.concatenate([blend_mean, freq_mean])  # (6,)

    # Safety
    if not np.all(np.isfinite(combined)):
        combined = np.nan_to_num(combined, nan=0.5, posinf=0.5, neginf=0.5)

    return combined.astype(np.float64)


# ---------------------------------------------------------------------------
# Full V4 extraction for one video
# ---------------------------------------------------------------------------
def extract_v4_features(video_path, n_frames=DEFAULT_N_FRAMES):
    """
    Extract the complete 20-dim V4 feature vector for one video.

    Returns:
        features: numpy array shape (20,) or None on failure
        fps: detected fps (float)
        n_loaded: number of frames actually loaded
    """
    # Load frames
    frames_bgr = load_frames(video_path, n_frames=n_frames)
    if len(frames_bgr) < 2:
        return None, 0.0, 0

    fps = get_video_fps(video_path)
    n_loaded = len(frames_bgr)

    # --- Part A: Kept V3 pillars (14 dims) ---
    v3_kept = extract_v3_kept_features(frames_bgr, video_path)
    if v3_kept is None:
        v3_kept = np.full(14, 0.5, dtype=np.float64)

    # --- Part B: New V4 pillars (6 dims) ---
    new_feats = extract_new_pillar_features(frames_bgr)

    # --- Concatenate: 14 + 6 = 20 dims ---
    v4 = np.concatenate([v3_kept, new_feats])

    assert v4.shape[0] == V4_FEATURE_DIM, \
        f"V4 vector has {v4.shape[0]} dims, expected {V4_FEATURE_DIM}"

    return v4, fps, n_loaded


# ---------------------------------------------------------------------------
# V4 feature vector layout documentation
# ---------------------------------------------------------------------------
V4_FEATURE_NAMES = [
    # P1 Noise (kept from V3)
    "noise_variance",           # dim 0
    "noise_kurtosis",           # dim 1
    "noise_entropy",            # dim 2
    # P2 PRNU (kept from V3)
    "prnu_correlation",         # dim 3
    "prnu_energy",              # dim 4
    # P4 Shadow (kept from V3)
    "shadow_consistency",       # dim 5
    "shadow_direction",         # dim 6
    "shadow_intensity",         # dim 7
    # P6 DCT (kept from V3)
    "dct_energy_dist",          # dim 8
    "dct_block_artifacts",      # dim 9
    "dct_ratio",                # dim 10
    # P7 Codec (kept from V3)
    "codec_qp_estimate",        # dim 11
    "codec_bitrate_var",        # dim 12
    # P9 Flow (kept from V3)
    "flow_consistency",         # dim 13
    # Blend Boundary (NEW)
    "blend_boundary_grad_ratio",     # dim 14
    "blend_grad_dir_coherence",      # dim 15
    "blend_boundary_vs_bg_ratio",    # dim 16
    # Freq Checkerboard (NEW)
    "freq_peak_energy_ratio",        # dim 17
    "freq_peak_spacing_regularity",  # dim 18
    "freq_face_bg_spectral_kl",      # dim 19
]


# ---------------------------------------------------------------------------
# Checkpoint management
# ---------------------------------------------------------------------------
def save_checkpoint(folder_idx, results, checkpoint_dir=CHECKPOINT_DIR):
    """Save processed results after each dataset folder."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(checkpoint_dir, f"v4_ckpt_folder_{folder_idx:02d}.pkl")
    with open(ckpt_path, "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
    return ckpt_path


def load_checkpoint(folder_idx, checkpoint_dir=CHECKPOINT_DIR):
    """Load checkpoint for a specific folder. Returns None if not found."""
    ckpt_path = os.path.join(checkpoint_dir, f"v4_ckpt_folder_{folder_idx:02d}.pkl")
    if os.path.exists(ckpt_path):
        with open(ckpt_path, "rb") as f:
            return pickle.load(f)
    return None


def get_completed_folders(checkpoint_dir=CHECKPOINT_DIR):
    """Return set of folder indices that have completed checkpoints."""
    completed = set()
    if not os.path.exists(checkpoint_dir):
        return completed
    for fname in os.listdir(checkpoint_dir):
        if fname.startswith("v4_ckpt_folder_") and fname.endswith(".pkl"):
            try:
                idx = int(fname.replace("v4_ckpt_folder_", "").replace(".pkl", ""))
                completed.add(idx)
            except ValueError:
                pass
    return completed


# ---------------------------------------------------------------------------
# Discovery: find all videos in a directory
# ---------------------------------------------------------------------------
def discover_videos(directory):
    """Find all video files in a directory (non-recursive)."""
    videos = []
    if not os.path.isdir(directory):
        return videos
    for ext in VIDEO_EXTENSIONS:
        videos.extend(glob.glob(os.path.join(directory, f"*{ext}")))
    videos.sort()
    return videos


# ---------------------------------------------------------------------------
# Process one dataset folder
# ---------------------------------------------------------------------------
def process_folder(folder_cfg, n_frames=DEFAULT_N_FRAMES):
    """
    Process all videos in one dataset folder.

    Returns dict with keys:
        features: list of (20,) arrays
        labels: list of ints (0 or 1)
        video_ids: list of strings (filename without extension)
        dataset_sources: list of strings
        generator_types: list of strings
        errors: list of (video_id, error_msg) tuples
        stats: dict with timing and count info
    """
    path = folder_cfg["path"]
    label = folder_cfg["label"]
    source = folder_cfg["source"]
    generator = folder_cfg["generator"]
    desc = folder_cfg["description"]

    results = {
        "features": [],
        "labels": [],
        "video_ids": [],
        "dataset_sources": [],
        "generator_types": [],
        "errors": [],
        "stats": {},
    }

    videos = discover_videos(path)
    n_videos = len(videos)

    if n_videos == 0:
        print(f"  [SKIP] No videos found in {path}")
        results["stats"] = {
            "total": 0, "success": 0, "failed": 0,
            "time_sec": 0, "description": desc
        }
        return results

    print(f"  Processing {n_videos} videos from: {desc}")
    print(f"  Path: {path}")

    t_start = time.time()
    n_success = 0
    n_failed = 0

    for i, vpath in enumerate(videos):
        video_id = Path(vpath).stem  # filename without extension

        try:
            feats, fps, n_loaded = extract_v4_features(vpath, n_frames=n_frames)

            if feats is None:
                n_failed += 1
                results["errors"].append((video_id, "Failed to load frames"))
                continue

            results["features"].append(feats)
            results["labels"].append(label)
            results["video_ids"].append(video_id)
            results["dataset_sources"].append(source)
            results["generator_types"].append(generator)
            n_success += 1

        except Exception as e:
            n_failed += 1
            results["errors"].append((video_id, str(e)))
            continue

        # Progress every 50 videos
        if (i + 1) % 50 == 0 or (i + 1) == n_videos:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (n_videos - i - 1) / rate if rate > 0 else 0
            print(f"    [{i+1:5d}/{n_videos}]  "
                  f"ok={n_success}  fail={n_failed}  "
                  f"rate={rate:.1f} vid/s  ETA={eta:.0f}s")

    elapsed = time.time() - t_start
    results["stats"] = {
        "total": n_videos,
        "success": n_success,
        "failed": n_failed,
        "time_sec": elapsed,
        "description": desc,
    }

    print(f"  Done: {n_success}/{n_videos} videos in {elapsed:.1f}s "
          f"({n_failed} failed)")

    return results


# ---------------------------------------------------------------------------
# Merge all folder results into final pkl
# ---------------------------------------------------------------------------
def merge_results(all_results):
    """
    Merge results from all folders into a single dict suitable for pkl.

    Returns dict with:
        features:         numpy array (N, 20)
        labels:           numpy array (N,) int
        video_ids:        list of N strings
        dataset_sources:  list of N strings
        generator_types:  list of N strings
        feature_names:    list of 20 strings (column documentation)
        extraction_config: dict with metadata
    """
    all_features = []
    all_labels = []
    all_video_ids = []
    all_sources = []
    all_generators = []
    all_stats = []

    for r in all_results:
        all_features.extend(r["features"])
        all_labels.extend(r["labels"])
        all_video_ids.extend(r["video_ids"])
        all_sources.extend(r["dataset_sources"])
        all_generators.extend(r["generator_types"])
        all_stats.append(r["stats"])

    n_total = len(all_features)

    if n_total == 0:
        print("[FATAL] No features extracted from any folder.")
        return None

    features_arr = np.array(all_features, dtype=np.float64)  # (N, 20)
    labels_arr = np.array(all_labels, dtype=np.int64)         # (N,)

    # Sanity check
    assert features_arr.shape == (n_total, V4_FEATURE_DIM), \
        f"Feature matrix shape {features_arr.shape}, expected ({n_total}, {V4_FEATURE_DIM})"

    merged = {
        "features": features_arr,
        "labels": labels_arr,
        "video_ids": all_video_ids,
        "dataset_sources": all_sources,
        "generator_types": all_generators,
        "feature_names": V4_FEATURE_NAMES,
        "extraction_config": {
            "version": "V4",
            "feature_dim": V4_FEATURE_DIM,
            "kept_v3_pillars": ["P1_Noise", "P2_PRNU", "P4_Shadow",
                                "P6_DCT", "P7_Codec", "P9_Flow"],
            "new_v4_pillars": ["BlendBoundary", "FreqCheckerboard"],
            "dropped_v3_pillars": ["P3_Bayer", "P5_Specular", "P8_Blur",
                                   "P10_Chromatic", "P11_EyeSym", "P12_Illum"],
            "v3_kept_indices": V3_KEEP_INDICES,
        },
        "per_folder_stats": all_stats,
    }

    return merged


# ---------------------------------------------------------------------------
# Print summary report
# ---------------------------------------------------------------------------
def print_summary(merged):
    """Print a detailed summary of the extracted dataset."""
    features = merged["features"]
    labels = merged["labels"]
    sources = merged["dataset_sources"]
    generators = merged["generator_types"]
    n_total = len(labels)

    print()
    print("=" * 70)
    print("PHANTOM LENS V4 — EXTRACTION COMPLETE")
    print("=" * 70)
    print(f"  Total videos:    {n_total}")
    print(f"  Feature dims:    {features.shape[1]}")
    print(f"  Real (label=0):  {(labels == 0).sum()}")
    print(f"  Fake (label=1):  {(labels == 1).sum()}")
    print()

    # Per-source breakdown
    print("  Per-source breakdown:")
    unique_sources = sorted(set(sources))
    for src in unique_sources:
        mask = np.array([s == src for s in sources])
        n = mask.sum()
        n_real = ((labels == 0) & mask).sum()
        n_fake = ((labels == 1) & mask).sum()
        print(f"    {src:<20s}  total={n:6d}  real={n_real:6d}  fake={n_fake:6d}")

    # Per-generator breakdown
    print()
    print("  Per-generator breakdown:")
    unique_gens = sorted(set(generators))
    for gen in unique_gens:
        mask = np.array([g == gen for g in generators])
        n = mask.sum()
        print(f"    {gen:<20s}  total={n:6d}")

    # Feature statistics
    print()
    print("  Feature statistics (mean ± std):")
    print(f"    {'Feature':<35s}  {'Real':>14s}  {'Fake':>14s}")
    print(f"    {'-'*35}  {'-'*14}  {'-'*14}")
    real_mask = labels == 0
    fake_mask = labels == 1
    for j, fname in enumerate(V4_FEATURE_NAMES):
        r_mean = features[real_mask, j].mean() if real_mask.any() else 0
        r_std = features[real_mask, j].std() if real_mask.any() else 0
        f_mean = features[fake_mask, j].mean() if fake_mask.any() else 0
        f_std = features[fake_mask, j].std() if fake_mask.any() else 0
        print(f"    {fname:<35s}  {r_mean:6.4f}±{r_std:.4f}  "
              f"{f_mean:6.4f}±{f_std:.4f}")

    # NaN/Inf check
    n_nan = np.isnan(features).sum()
    n_inf = np.isinf(features).sum()
    if n_nan > 0 or n_inf > 0:
        print(f"\n  [WARN] Found {n_nan} NaN and {n_inf} Inf values in features!")
    else:
        print(f"\n  [OK] No NaN or Inf values in feature matrix.")

    # Per-folder timing
    print()
    print("  Per-folder timing:")
    for stat in merged["per_folder_stats"]:
        desc = stat.get("description", "?")
        total = stat.get("total", 0)
        t = stat.get("time_sec", 0)
        rate = total / t if t > 0 else 0
        print(f"    {desc:<30s}  {total:5d} videos  {t:7.1f}s  "
              f"({rate:.1f} vid/s)")

    print()
    print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Phantom Lens V4 Feature Extraction")
    parser.add_argument("--n_frames", type=int, default=DEFAULT_N_FRAMES,
                        help=f"Frames per video (default: {DEFAULT_N_FRAMES})")
    parser.add_argument("--output", type=str, default=OUTPUT_PKL,
                        help=f"Output pkl path (default: {OUTPUT_PKL})")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoints if available")
    parser.add_argument("--checkpoint_dir", type=str, default=CHECKPOINT_DIR,
                        help=f"Checkpoint directory (default: {CHECKPOINT_DIR})")
    parser.add_argument("--folders", type=str, default=None,
                        help="Comma-separated folder indices to process "
                             "(e.g., '0,1,5'). Default: all.")
    args = parser.parse_args()

    print("=" * 70)
    print("PHANTOM LENS V4 — Feature Extraction Pipeline")
    print("=" * 70)
    print(f"  Feature dimension:  {V4_FEATURE_DIM}")
    print(f"  Frames per video:   {args.n_frames}")
    print(f"  Output:             {args.output}")
    print(f"  Checkpoint dir:     {args.checkpoint_dir}")
    print(f"  Resume mode:        {args.resume}")
    print(f"  V3 pillars:         {'available' if V3_AVAILABLE else 'NOT FOUND'}")
    print(f"  Blend boundary:     {'available' if BLEND_AVAILABLE else 'NOT FOUND'}")
    print(f"  Freq checkerboard:  {'available' if FREQ_AVAILABLE else 'NOT FOUND'}")
    print()

    # Determine which folders to process
    if args.folders:
        folder_indices = [int(x.strip()) for x in args.folders.split(",")]
    else:
        folder_indices = list(range(len(DATASET_FOLDERS)))

    # Check which folders already have checkpoints
    completed = get_completed_folders(args.checkpoint_dir) if args.resume else set()
    if completed:
        print(f"  Found checkpoints for folders: {sorted(completed)}")

    # Output directory
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # Process each folder
    all_results = []
    total_time_start = time.time()

    for idx in folder_indices:
        if idx >= len(DATASET_FOLDERS):
            print(f"[WARN] Folder index {idx} out of range, skipping")
            continue

        folder_cfg = DATASET_FOLDERS[idx]
        print(f"\n{'─'*70}")
        print(f"Folder {idx}/{len(DATASET_FOLDERS)-1}: "
              f"{folder_cfg['description']}")
        print(f"{'─'*70}")

        # Check for existing checkpoint
        if args.resume and idx in completed:
            print(f"  [RESUME] Loading checkpoint for folder {idx}")
            ckpt = load_checkpoint(idx, args.checkpoint_dir)
            if ckpt is not None:
                all_results.append(ckpt)
                n = len(ckpt["features"])
                print(f"  [RESUME] Loaded {n} videos from checkpoint")
                continue
            else:
                print(f"  [RESUME] Checkpoint load failed, re-processing")

        # Process folder
        results = process_folder(folder_cfg, n_frames=args.n_frames)
        all_results.append(results)

        # Save checkpoint
        ckpt_path = save_checkpoint(idx, results, args.checkpoint_dir)
        print(f"  Checkpoint saved: {ckpt_path}")

    # Merge all results
    print(f"\n{'─'*70}")
    print("Merging all results...")
    merged = merge_results(all_results)

    if merged is None:
        print("[FATAL] No data to save. Exiting.")
        sys.exit(1)

    # Save final pkl
    with open(args.output, "wb") as f:
        pickle.dump(merged, f, protocol=pickle.HIGHEST_PROTOCOL)

    total_time = time.time() - total_time_start
    file_size_mb = os.path.getsize(args.output) / (1024 * 1024)

    print(f"Saved: {args.output} ({file_size_mb:.1f} MB)")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")

    # Print summary
    print_summary(merged)

    # Cleanup hint
    print(f"Checkpoints in {args.checkpoint_dir}/ can be deleted now.")
    print("Done.")


if __name__ == "__main__":
    main()