# Phantom Lens Research Framework
# Developer: Miheer Satish Kulkarni | IIIT Nagpur | 2026
# All Rights Reserved



import numpy as np
import cv2
import pickle
import os
from tqdm import tqdm
from datasets import load_dataset
from precompute_features_v2 import (
    extract_features_from_frame,
    load_or_create_pkl,
    save_pkl,
    append_to_pkl
)

PKL_PATH        = "data/precomputed_features.pkl"
MAX_FAKE        = 30000
SAVE_EVERY      = 2000

def main():
    print("=" * 60)
    print("WildDeepfake Fake Preparation — Phantom Lens V2")
    print("Streaming from HuggingFace...")
    print("=" * 60)

    data = load_or_create_pkl(PKL_PATH)

    existing_fake = sum(1 for l in data['labels'] if l == 1)
    print(f"Existing fake samples: {existing_fake}")
    print(f"Target: {MAX_FAKE} fake samples from WildDeepfake")

    print("Loading xingjunm/WildDeepfake from HuggingFace...")
    ds = load_dataset('xingjunm/WildDeepfake', split='train', streaming=True)

    new_features, new_labels, new_vids, new_sources, new_gens = [], [], [], [], []
    fake_count = 0
    failed = 0
    idx = 0

    print(f"Processing up to {MAX_FAKE} fake samples...")
    for sample in tqdm(ds):
        if fake_count >= MAX_FAKE:
            break

        try:
            key = sample['__key__']
            is_fake = 'fake' in key.lower()

            if not is_fake:
                continue

            # Convert PIL image to cv2 BGR format
            img_pil = sample['png'].convert('RGB')
            img_pil = img_pil.resize((224, 224))
            img_array = np.array(img_pil, dtype=np.uint8)
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

            feat = extract_features_from_frame(img_bgr)

            if feat is not None:
                video_id = f"wilddeepfake_{idx:06d}"
                new_features.append(feat)
                new_labels.append(1)
                new_vids.append(video_id)
                new_sources.append('wilddeepfake')
                new_gens.append('wilddeepfake_mixed')
                fake_count += 1

                if fake_count % SAVE_EVERY == 0:
                    data = append_to_pkl(data, new_features, new_labels,
                                        new_vids, new_sources, new_gens)
                    save_pkl(data, PKL_PATH)
                    new_features, new_labels = [], []
                    new_vids, new_sources, new_gens = [], [], []
                    print(f"  Checkpoint: {fake_count} fake samples saved")
            else:
                failed += 1

        except Exception:
            failed += 1

        idx += 1

    # Save remaining
    if len(new_features) > 0:
        data = append_to_pkl(data, new_features, new_labels,
                             new_vids, new_sources, new_gens)
        save_pkl(data, PKL_PATH)

    print(f"\nWildDeepfake complete:")
    print(f"  Fake samples added: {fake_count}")
    print(f"  Failed: {failed}")

if __name__ == "__main__":
    main()
