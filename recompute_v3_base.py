# Phantom Lens Research Framework
# Developer: Miheer Satish Kulkarni | IIIT Nagpur | 2026
# All Rights Reserved

"""
Phantom Lens V3 — Block 7: Overnight Feature Recomputation
===========================================================
Recomputes ALL existing video data with new 30-dim V3 extractor.
Creates: data/precomputed_features_v3_base.pkl

Processes:
  1. FF++ original real videos     (youtube/c23)
  2. FF++ Deepfakes                (manipulated/Deepfakes/c23)
  3. FF++ Face2Face                (manipulated/Face2Face/c23)
  4. FF++ FaceShifter              (manipulated/FaceShifter/c23)
  5. FF++ FaceSwap                 (manipulated/FaceSwap/c23)
  6. FF++ NeuralTextures           (manipulated/NeuralTextures/c23)
  7. CelebVHQ real videos          (data/celebvhq/35666)

DO NOT include Celeb-DF v2 — that is the unseen test set.
DO NOT include WildDeepfake here — added separately in Day 2.

Expected output:
  ~80,000 samples
  30-dim feature vectors
  Sources: ffpp_official, celebvhq

Runtime estimate: 4-8 hours on laptop
              OR: 1-2 hours on DGX Spark

Author: Miheer Satish Kulkarni, IIIT Nagpur
"""

import os
import sys
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from precompute_features_v3 import (
    extract_features_from_video,
    load_or_create_pkl,
    save_pkl,
    append_to_pkl
)

# ── CONFIG ────────────────────────────────────────────────────
OUTPUT_PKL  = "data/precomputed_features_v3_base.pkl"
N_FRAMES    = 16        # V3 uses 16 frames
SAVE_EVERY  = 200       # save checkpoint every N videos

# FF++ paths
FFPP_ROOT   = "data/ffpp_official"
FFPP_REAL   = "original_sequences/youtube/c23/videos"
FFPP_FAKES  = [
    ("manipulated_sequences/Deepfakes/c23/videos",      "deepfakes"),
    ("manipulated_sequences/Face2Face/c23/videos",      "face2face"),
    ("manipulated_sequences/FaceShifter/c23/videos",    "faceshifter"),
    ("manipulated_sequences/FaceSwap/c23/videos",       "faceswap"),
    ("manipulated_sequences/NeuralTextures/c23/videos", "neuraltextures"),
]
MAX_PER_FAKE = 8000     # max videos per fake type

# CelebVHQ path
CELEBVHQ_ROOT = "data/celebvhq/35666"
MAX_CELEBVHQ  = 5000    # max real videos from CelebVHQ


# ── PROCESS ONE FOLDER ────────────────────────────────────────
def process_folder(folder_path, label, source,
                   gen_type, max_videos, desc):
    """
    Process all videos in a folder.
    Returns features, labels, video_ids, sources, generators.
    """
    if not os.path.exists(folder_path):
        print(f"  SKIP — folder not found: {folder_path}")
        return [], [], [], [], []

    videos = sorted([
        f for f in os.listdir(folder_path)
        if f.lower().endswith(('.mp4', '.avi', '.mov'))
    ])[:max_videos]

    print(f"  Found {len(videos)} videos in {desc}")

    all_feats  = []
    all_labels = []
    all_vids   = []
    all_srcs   = []
    all_gens   = []
    failed     = 0

    for vf in tqdm(videos, desc=desc):
        vpath  = os.path.join(folder_path, vf)
        vid_id = f"{source}_{gen_type}_{os.path.splitext(vf)[0]}"

        feats = extract_features_from_video(
            vpath, n_frames=N_FRAMES)

        if len(feats) == 0:
            failed += 1
            continue

        for feat in feats:
            all_feats.append(feat)
            all_labels.append(label)
            all_vids.append(vid_id)
            all_srcs.append(source)
            all_gens.append(gen_type)

    print(f"  Done: {len(videos)-failed} ok | "
          f"{failed} failed | "
          f"{len(all_feats)} samples")
    return all_feats, all_labels, all_vids, all_srcs, all_gens


# ── MAIN ──────────────────────────────────────────────────────
def main():
    print("\n" + "="*60)
    print("  Phantom Lens V3 — Block 7 Overnight Recomputation")
    print("  Output: data/precomputed_features_v3_base.pkl")
    print("  Features: 30-dim (24 original + 6 new)")
    print("="*60 + "\n")

    # Start fresh — DO NOT load old V1 pkl
    # V1 pkl has 24-dim features, V3 needs 30-dim
    if os.path.exists(OUTPUT_PKL):
        print(f"WARNING: {OUTPUT_PKL} already exists.")
        print("Continuing from existing checkpoint...")
        data = load_or_create_pkl(OUTPUT_PKL)
    else:
        print("Starting fresh V3 pkl...")
        data = load_or_create_pkl(OUTPUT_PKL)

    print()

    # ── 1. FF++ REAL ──────────────────────────────────────────
    print("="*55)
    print("[1/7] FF++ Original Real Videos")
    print("="*55)
    real_folder = os.path.join(FFPP_ROOT, FFPP_REAL)
    f, l, v, s, g = process_folder(
        real_folder,
        label=0,
        source='ffpp_official',
        gen_type='real_youtube',
        max_videos=8000,
        desc='FF++ Real'
    )
    if len(f) > 0:
        data = append_to_pkl(data, f, l, v, s, g)
        save_pkl(data, OUTPUT_PKL)
        print(f"  Checkpoint saved. Total: {len(data['labels'])}\n")

    # ── 2-6. FF++ FAKE TYPES ──────────────────────────────────
    for i, (rel_path, gen_type) in enumerate(FFPP_FAKES):
        folder = os.path.join(FFPP_ROOT, rel_path)
        idx    = i + 2
        name   = gen_type.capitalize()
        print("="*55)
        print(f"[{idx}/7] FF++ Fake — {name}")
        print("="*55)
        f, l, v, s, g = process_folder(
            folder,
            label=1,
            source='ffpp_official',
            gen_type=gen_type,
            max_videos=MAX_PER_FAKE,
            desc=name
        )
        if len(f) > 0:
            data = append_to_pkl(data, f, l, v, s, g)
            save_pkl(data, OUTPUT_PKL)
            print(f"  Checkpoint saved. "
                  f"Total: {len(data['labels'])}\n")

    # ── 7. CELEBVHQ REAL ─────────────────────────────────────
    print("="*55)
    print("[7/7] CelebVHQ Real Videos")
    print("="*55)
    f, l, v, s, g = process_folder(
        CELEBVHQ_ROOT,
        label=0,
        source='celebvhq',
        gen_type='real',
        max_videos=MAX_CELEBVHQ,
        desc='CelebVHQ Real'
    )
    if len(f) > 0:
        data = append_to_pkl(data, f, l, v, s, g)
        save_pkl(data, OUTPUT_PKL)
        print(f"  Checkpoint saved. "
              f"Total: {len(data['labels'])}\n")

    # ── FINAL SUMMARY ─────────────────────────────────────────
    labels  = np.array(data['labels'])
    sources = np.array(data['dataset_sources'])
    total   = len(labels)
    real    = int((labels == 0).sum())
    fake    = int((labels == 1).sum())

    print("\n" + "="*60)
    print("  RECOMPUTATION COMPLETE")
    print("="*60)
    print(f"  Total samples  : {total}")
    print(f"  Real           : {real}")
    print(f"  Fake           : {fake}")
    print(f"  Feature dims   : {data['features'].shape[1]}")
    print(f"\n  Per dataset:")
    for src in sorted(set(sources)):
        sr = int(((sources==src) & (labels==0)).sum())
        sf = int(((sources==src) & (labels==1)).sum())
        print(f"    {src:<20} real={sr:6d}  fake={sf:6d}")
    print(f"\n  Saved to: {OUTPUT_PKL}")
    print("  Ready for Day 2 — merge with WildDeepfake")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()