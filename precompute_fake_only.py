# Phantom Lens Research Framework
# Developer: Miheer Satish Kulkarni | IIIT Nagpur | 2026
# All Rights Reserved



import os
import pickle
import numpy as np
from tqdm import tqdm
import cv2

# Import pillars
from src.pillars.pillar1_noise import compute_pillar1_score
from src.pillars.pillar2_light import compute_pillar2_score
from src.pillars.pillar3_compression import compute_pillar3_score

FAKE_DIR = "data/fake"
PKL_PATH = "data/precomputed_features.pkl"

def load_frame(path):
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.0

def compute_features(folder_path):
    frames = []
    for f in sorted(os.listdir(folder_path)):
        if f.endswith('.jpg'):
            frame = load_frame(os.path.join(folder_path, f))
            if frame is not None:
                frames.append(frame)
    if len(frames) == 0:
        return None
    p1_scores = [compute_pillar1_score(f) for f in frames]
    p2_scores = [compute_pillar2_score(f) for f in frames]
    p3_scores = [compute_pillar3_score(f) for f in frames]
    f1_mean = np.mean([s['f1_raw'] for s in p1_scores])
    f1_std = np.std([s['f1_raw'] for s in p1_scores])
    f2_raw = np.mean([s['f2_raw'] for s in p2_scores])
    f2_var = np.var([s['f2_raw'] for s in p2_scores])
    f3_mean = np.mean([s['f3_raw'] for s in p3_scores])
    f3_benford = np.mean([s['benford_deviation'] for s in p3_scores])
    return np.array([f1_mean, f1_std, f2_raw, f2_var, f3_mean, f3_benford,
                     f1_mean * f2_raw, f1_mean * f3_mean], dtype=np.float32)

# Load existing pkl
print("Loading existing pkl...")
with open(PKL_PATH, 'rb') as f:
    data = pickle.load(f)

existing_features = data['features']
existing_labels = data['labels']
print(f"Existing samples: {len(existing_labels)}")

# Find only NEW fake folders not already in pkl
fake_folders = sorted([f for f in os.listdir(FAKE_DIR) 
                       if os.path.isdir(os.path.join(FAKE_DIR, f))
                       and f.startswith('deeper_fake_new')])

print(f"New fake folders to process: {len(fake_folders)}")

new_features = []
new_labels = []
failed = 0

for folder in tqdm(fake_folders):
    folder_path = os.path.join(FAKE_DIR, folder)
    feat = compute_features(folder_path)
    if feat is not None:
        new_features.append(feat)
        new_labels.append(1)
    else:
        failed += 1

print(f"New fake samples computed: {len(new_features)}, Failed: {failed}")

# Merge with existing
all_features = np.vstack([existing_features, np.array(new_features)])
all_labels = np.array(list(existing_labels) + new_labels)

print(f"Total samples after merge: {len(all_labels)}")

# Save back
with open(PKL_PATH, 'wb') as f:
    pickle.dump({'features': all_features, 'labels': all_labels}, f)

print("Saved merged pkl. Ready to train.")
