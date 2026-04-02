# Phantom Lens Research Framework
# Developer: Miheer Satish Kulkarni | IIIT Nagpur | 2026
# All Rights Reserved


import os
import numpy as np
from tqdm import tqdm
from precompute_features_v2 import (
    extract_features_from_video,
    load_or_create_pkl,
    save_pkl,
    append_to_pkl
)

PKL_PATH   = "data/precomputed_features.pkl"
VIDEO_DIR  = "data/celebvhq/35666"
MAX_REAL   = 32000
N_FRAMES   = 8
SAVE_EVERY = 500

def main():
    print("=" * 60)
    print("CelebV-HQ Real Video Preparation — Phantom Lens V2")
    print(f"Reading from: {VIDEO_DIR}")
    print("=" * 60)

    data = load_or_create_pkl(PKL_PATH)

    existing_real = sum(1 for l in data['labels'] if l == 0)
    print(f"Existing real samples: {existing_real}")
    if existing_real >= MAX_REAL:
        print("Already have enough real samples. Skipping.")
        return

    if not os.path.exists(VIDEO_DIR):
        print(f"ERROR: Video directory not found: {VIDEO_DIR}")
        return

    videos = [f for f in os.listdir(VIDEO_DIR) if f.endswith('.mp4')]
    videos = sorted(videos)[:MAX_REAL]
    print(f"Found {len(videos)} videos")

    new_features, new_labels, new_vids, new_sources, new_gens = [], [], [], [], []
    processed = 0
    failed = 0
    total_samples = 0

    for idx, video_file in enumerate(tqdm(videos, desc="CelebV-HQ")):
        video_path = os.path.join(VIDEO_DIR, video_file)
        video_id = f"celebvhq_{idx:06d}"

        feats = extract_features_from_video(video_path, n_frames=N_FRAMES)

        if len(feats) == 0:
            failed += 1
            continue

        for feat in feats:
            new_features.append(feat)
            new_labels.append(0)
            new_vids.append(video_id)
            new_sources.append('celebvhq')
            new_gens.append('real')
            total_samples += 1

        processed += 1

        if processed % SAVE_EVERY == 0:
            data = append_to_pkl(data, new_features, new_labels,
                                 new_vids, new_sources, new_gens)
            save_pkl(data, PKL_PATH)
            new_features, new_labels = [], []
            new_vids, new_sources, new_gens = [], [], []
            print(f"  Checkpoint: {processed} videos | {total_samples} samples")

    if len(new_features) > 0:
        data = append_to_pkl(data, new_features, new_labels,
                             new_vids, new_sources, new_gens)
        save_pkl(data, PKL_PATH)

    print(f"\nCelebV-HQ complete:")
    print(f"  Processed: {processed} videos")
    print(f"  Failed: {failed}")
    print(f"  Samples added: {total_samples}")

if __name__ == "__main__":
    main()
