# Phantom Lens Research Framework
# Developer: Miheer Satish Kulkarni | IIIT Nagpur | 2026
# All Rights Reserved



"""
Phantom Lens / PRISM — FF++ Official Dataset Preparation
Processes all 5 fake manipulation types + original real videos
Author: Miheer Satish Kulkarni, IIIT Nagpur

Run AFTER prepare_celebvhq.py and prepare_wilddeepfake.py
"""

import os
import sys
import numpy as np
from tqdm import tqdm
from precompute_features_v2 import (
    extract_features_from_video,
    load_or_create_pkl,
    save_pkl,
    append_to_pkl
)

# ─── CONFIG ──────────────────────────────────────────────────────────────────

PKL_PATH       = "data/precomputed_features.pkl"
FFPP_ROOT      = "data/ffpp_official"
MAX_PER_TYPE   = 4000   # max fake videos per manipulation type
MAX_REAL       = 8000  # max real videos from original
N_FRAMES       = 8      # frames sampled per video

# FF++ fake manipulation types
FAKE_TYPES = [
    ("Deepfakes",      "manipulated_sequences/Deepfakes/c23/videos"),
    ("Face2Face",      "manipulated_sequences/Face2Face/c23/videos"),
    ("FaceShifter",    "manipulated_sequences/FaceShifter/c23/videos"),
    ("FaceSwap",       "manipulated_sequences/FaceSwap/c23/videos"),
    ("NeuralTextures", "manipulated_sequences/NeuralTextures/c23/videos"),
]

# FF++ real original videos
REAL_PATH = "original_sequences/youtube/c23/videos"

# ─── MAIN ────────────────────────────────────────────────────────────────────

def process_folder(folder_path, label, dataset_source, generator_type,
                   max_videos, desc):
    """Process all videos in a folder and return features."""
    if not os.path.exists(folder_path):
        print(f"  Folder not found: {folder_path}")
        return [], [], [], [], []

    videos = [f for f in os.listdir(folder_path)
              if f.endswith('.mp4') or f.endswith('.avi')]
    videos = sorted(videos)[:max_videos]
    print(f"  Found {len(videos)} videos in {desc}")

    all_features, all_labels, all_vids, all_sources, all_gens = [], [], [], [], []
    failed = 0

    for video_file in tqdm(videos, desc=desc):
        video_path = os.path.join(folder_path, video_file)
        video_id = f"ffpp_{generator_type}_{os.path.splitext(video_file)[0]}"

        feats = extract_features_from_video(video_path, n_frames=N_FRAMES)

        if len(feats) == 0:
            failed += 1
            continue

        for feat in feats:
            all_features.append(feat)
            all_labels.append(label)
            all_vids.append(video_id)
            all_sources.append(dataset_source)
            all_gens.append(generator_type)

    print(f"  Processed: {len(videos) - failed} videos | "
          f"Failed: {failed} | "
          f"Samples: {len(all_features)}")

    return all_features, all_labels, all_vids, all_sources, all_gens


def main():
    print("=" * 60)
    print("FF++ Official Dataset Preparation — Phantom Lens V2")
    print("=" * 60)

    # Load existing pkl
    data = load_or_create_pkl(PKL_PATH)
    print()

    # ── Process FAKE types ────────────────────────────────────────────────────
    print("Processing FAKE manipulation types...")
    print("-" * 40)

    for gen_type, rel_path in FAKE_TYPES:
        folder_path = os.path.join(FFPP_ROOT, rel_path)
        print(f"\n[FAKE] {gen_type}")

        feats, labels, vids, sources, gens = process_folder(
            folder_path=folder_path,
            label=1,
            dataset_source="ffpp_official",
            generator_type=gen_type.lower(),
            max_videos=MAX_PER_TYPE,
            desc=gen_type
        )

        if len(feats) > 0:
            data = append_to_pkl(data, feats, labels, vids, sources, gens)
            save_pkl(data, PKL_PATH)
            print(f"  Saved checkpoint after {gen_type}")

    # ── Process REAL original videos ──────────────────────────────────────────
    print("\n" + "-" * 40)
    print("\n[REAL] Original YouTube Videos")

    real_folder = os.path.join(FFPP_ROOT, REAL_PATH)
    feats, labels, vids, sources, gens = process_folder(
        folder_path=real_folder,
        label=0,
        dataset_source="ffpp_official",
        generator_type="real_youtube",
        max_videos=MAX_REAL,
        desc="FF++ Original Real"
    )

    if len(feats) > 0:
        data = append_to_pkl(data, feats, labels, vids, sources, gens)
        save_pkl(data, PKL_PATH)

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("FF++ PREPARATION COMPLETE")
    print("=" * 60)

    labels_arr = np.array(data['labels'])
    real_count = np.sum(labels_arr == 0)
    fake_count = np.sum(labels_arr == 1)
    total = len(labels_arr)

    print(f"Total samples : {total}")
    print(f"Real samples  : {real_count}")
    print(f"Fake samples  : {fake_count}")
    print(f"Balance ratio : {real_count/fake_count:.2f}:1")

    # Per dataset breakdown
    print("\nPer dataset breakdown:")
    sources = data['dataset_sources']
    unique_sources = list(set(sources))
    for src in sorted(unique_sources):
        indices = [i for i, s in enumerate(sources) if s == src]
        src_labels = [data['labels'][i] for i in indices]
        src_real = sum(1 for l in src_labels if l == 0)
        src_fake = sum(1 for l in src_labels if l == 1)
        print(f"  {src:25s} real={src_real:6d} fake={src_fake:6d}")

    print("\nReady for training!")


if __name__ == "__main__":
    main()