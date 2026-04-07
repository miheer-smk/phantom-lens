# Phantom Lens Research Framework
# Developer: Miheer Satish Kulkarni | IIIT Nagpur | 2026
# All Rights Reserved



import os
import io
import pickle
import numpy as np
from datasets import load_dataset
from PIL import Image
import cv2
from tqdm import tqdm

from src.pillars.pillar1_noise import compute_pillar1_score
from src.pillars.pillar2_light import compute_pillar2_score
from src.pillars.pillar3_compression import compute_pillar3_score

PKL_PATH = "data/precomputed_features.pkl"
MAX_FAKE_SAMPLES = 10000
MAX_REAL_SAMPLES = 0

def compute_features_from_image(img_array):
    try:
        p1 = compute_pillar1_score(img_array)
        p2 = compute_pillar2_score(img_array)
        p3 = compute_pillar3_score(img_array)
        f1_mean = p1['f1_raw']
        f1_std = 0.0
        f2_raw = p2['f2_raw']
        f2_var = 0.0
        f3_mean = p3['f3_raw']
        f3_benford = p3['benford_deviation']
        return np.array([f1_mean, f1_std, f2_raw, f2_var, f3_mean, f3_benford,
                         f1_mean * f2_raw, f1_mean * f3_mean], dtype=np.float32)
    except:
        return None

print("Loading existing pkl...")
with open(PKL_PATH, 'rb') as f:
    data = pickle.load(f)
existing_features = data['features']
existing_labels = data['labels']
print(f"Existing samples: {len(existing_labels)}")

print("Loading WildDeepfake train split (streaming)...")
ds = load_dataset('xingjunm/WildDeepfake', split='train', streaming=True)

new_features = []
new_labels = []
fake_count = 0
real_count = 0
failed = 0

print(f"Processing up to {MAX_FAKE_SAMPLES} fake + {MAX_REAL_SAMPLES} real samples...")

for sample in tqdm(ds):
    if fake_count >= MAX_FAKE_SAMPLES and real_count >= MAX_REAL_SAMPLES:
        break

    key = sample['__key__']
    is_fake = 'fake' in key.lower()
    is_real = 'real' in key.lower()

    if is_fake and fake_count >= MAX_FAKE_SAMPLES:
        continue
    if is_real and real_count >= MAX_REAL_SAMPLES:
        continue
    if not is_fake and not is_real:
        continue

    try:
        img = sample['png'].convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img, dtype=np.float32) / 255.0

        feat = compute_features_from_image(img_array)
        if feat is not None:
            new_features.append(feat)
            label = 1 if is_fake else 0
            new_labels.append(label)
            if is_fake:
                fake_count += 1
            else:
                real_count += 1
        else:
            failed += 1
    except:
        failed += 1

print(f"\nWildDeepfake — Fake: {fake_count}, Real: {real_count}, Failed: {failed}")

all_features = np.vstack([existing_features, np.array(new_features)])
all_labels = np.array(list(existing_labels) + new_labels)

print(f"Total samples after merge: {len(all_labels)}")

with open(PKL_PATH, 'wb') as f:
    pickle.dump({'features': all_features, 'labels': all_labels}, f, protocol=4)

print("Saved. Ready to train.")
