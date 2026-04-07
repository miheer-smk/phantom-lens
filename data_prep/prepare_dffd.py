# Phantom Lens Research Framework
# Developer: Miheer Satish Kulkarni | IIIT Nagpur | 2026
# All Rights Reserved


import os
import cv2
import pickle
import numpy as np
from tqdm import tqdm

from src.pillars.pillar1_noise import compute_pillar1_score
from src.pillars.pillar2_light import compute_pillar2_score
from src.pillars.pillar3_compression import compute_pillar3_score

PKL_PATH = "data/precomputed_features.pkl"
DFFD_ROOT = "data/dffd"

FAKE_FOLDERS = ["pggan_v1", "pggan_v2", "stargan", "stylegan_celeba", "stylegan_ffhq"]
REAL_FOLDERS = ["ffhq"]

MAX_FAKE_PER_FOLDER = 1600
MAX_REAL_PER_FOLDER = 8000

def load_image(path):
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.0

def compute_features(img):
    try:
        p1 = compute_pillar1_score(img)
        p2 = compute_pillar2_score(img)
        p3 = compute_pillar3_score(img)
        f1_mean = p1['f1_raw']
        f1_std = float(np.std(img))
        f2_raw = p2['f2_raw']
        f2_var = float(np.var(img))
        f3_mean = p3['f3_raw']
        f3_benford = p3['benford_deviation']
        return np.array([f1_mean, f1_std, f2_raw, f2_var, f3_mean, f3_benford,
                         f1_mean * f2_raw, f1_mean * f3_mean], dtype=np.float32)
    except:
        return None

# Load existing pkl
print("Loading existing pkl...")
with open(PKL_PATH, 'rb') as f:
    data = pickle.load(f)
existing_features = data['features']
existing_labels = data['labels']
print(f"Existing samples: {len(existing_labels)}")

new_features = []
new_labels = []

# Process FAKE folders
for folder_name in FAKE_FOLDERS:
    folder_path = os.path.join(DFFD_ROOT, folder_name, "train")
    if not os.path.exists(folder_path):
        print(f"Skipping {folder_name} — train folder not found yet")
        continue
    images = [f for f in os.listdir(folder_path) if f.endswith('.png') or f.endswith('.jpg')]
    images = images[:MAX_FAKE_PER_FOLDER]
    print(f"Processing {folder_name}: {len(images)} fake images")
    failed = 0
    for fname in tqdm(images, desc=folder_name):
        img = load_image(os.path.join(folder_path, fname))
        if img is None:
            failed += 1
            continue
        feat = compute_features(img)
        if feat is not None:
            new_features.append(feat)
            new_labels.append(1)
        else:
            failed += 1
    print(f"  Failed: {failed}")

# Process REAL folders
for folder_name in REAL_FOLDERS:
    folder_path = os.path.join(DFFD_ROOT, folder_name, "train")
    if not os.path.exists(folder_path):
        print(f"Skipping {folder_name} — train folder not found yet")
        continue
    images = [f for f in os.listdir(folder_path) if f.endswith('.png') or f.endswith('.jpg')]
    images = images[:MAX_REAL_PER_FOLDER]
    print(f"Processing {folder_name}: {len(images)} real images")
    failed = 0
    for fname in tqdm(images, desc=folder_name):
        img = load_image(os.path.join(folder_path, fname))
        if img is None:
            failed += 1
            continue
        feat = compute_features(img)
        if feat is not None:
            new_features.append(feat)
            new_labels.append(0)
        else:
            failed += 1
    print(f"  Failed: {failed}")

print(f"\nNew samples added: {len(new_features)}")

# Merge and save
all_features = np.vstack([existing_features, np.array(new_features)])
all_labels = np.array(list(existing_labels) + new_labels)
print(f"Total samples after merge: {len(all_labels)}")

with open(PKL_PATH, 'wb') as f:
    pickle.dump({'features': all_features, 'labels': all_labels}, f, protocol=4)

print("Saved. Ready to train.")