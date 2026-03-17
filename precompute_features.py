# Phantom Lens Research Framework
# Developer: Miheer Satish Kulkarni | IIIT Nagpur | 2026
# All Rights Reserved



import os
import numpy as np
import cv2
from tqdm import tqdm
import pickle
from src.pillars.pillar1_noise import compute_pillar1_score
from src.pillars.pillar2_light import compute_pillar2_score
from src.pillars.pillar3_compression import compute_pillar3_score

DATA_ROOT = "data/"
OUTPUT_FILE = "data/precomputed_features.pkl"

def extract_features(folder_path, frames_list, n_frames=8):
    indices = np.linspace(0, len(frames_list)-1, min(n_frames, len(frames_list)), dtype=int)
    selected = [frames_list[i] for i in indices]
    
    loaded = []
    for f in selected:
        path = os.path.join(folder_path, f)
        img = cv2.imread(path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = img.astype(np.float32) / 255.0
        loaded.append(img)
    
    if len(loaded) == 0:
        return None
    
    frames_np = np.stack(loaded, axis=0)
    T = frames_np.shape[0]
    
    # Pillar 1
    f1_scores = []
    for t in range(T):
        r = compute_pillar1_score(frames_np[t])
        f1_scores.append(r['f1_raw'])
    f1_mean = float(np.mean(f1_scores))
    f1_std = float(np.std(f1_scores))
    
    # Pillar 2
    frames_gray = []
    for t in range(T):
        gray = cv2.cvtColor((frames_np[t]*255).astype(np.uint8),
                            cv2.COLOR_RGB2GRAY).astype(np.float32)/255.0
        frames_gray.append(gray)
    r2 = compute_pillar2_score(frames_gray)
    f2_raw = r2['f2_raw']
    f2_var = r2['temporal_variance']
    
    # Pillar 3
    f3_scores = []
    for t in range(min(T, 4)):
        r3 = compute_pillar3_score(frames_np[t])
        f3_scores.append(r3['f3_raw'])
    f3_mean = float(np.mean(f3_scores))
    f3_benford = float(np.mean([compute_pillar3_score(frames_np[t])['benford_deviation']
                                for t in range(min(T, 4))]))
    
    features = np.array([
        f1_mean, f1_std,
        f2_raw, f2_var,
        f3_mean, f3_benford,
        np.clip(f1_mean * f2_raw, 0, 10),
        np.clip(f1_mean * f3_mean, 0, 10),
    ], dtype=np.float32)
    
    return features

# Process all samples
all_features = []
all_labels = []
failed = 0

for label, category in enumerate(['real', 'fake']):
    category_path = os.path.join(DATA_ROOT, category)
    if not os.path.exists(category_path):
        continue
    folders = sorted([f for f in os.listdir(category_path) 
                     if os.path.isdir(os.path.join(category_path, f))])
    
    print(f"\nProcessing {category} ({len(folders)} samples)...")
    
    for folder in tqdm(folders):
        folder_path = os.path.join(category_path, folder)
        frames = sorted([f for f in os.listdir(folder_path)
                        if f.endswith('.jpg') or f.endswith('.png')])
        if len(frames) == 0:
            failed += 1
            continue
        
        features = extract_features(folder_path, frames)
        if features is not None:
            all_features.append(features)
            all_labels.append(label)
        else:
            failed += 1

all_features = np.array(all_features)
all_labels = np.array(all_labels)

print(f"\nTotal samples: {len(all_features)}")
print(f"Failed: {failed}")
print(f"Feature shape: {all_features.shape}")

with open(OUTPUT_FILE, 'wb') as f:
    pickle.dump({'features': all_features, 'labels': all_labels}, f)

print(f"\nSaved to {OUTPUT_FILE}")
print("Precomputation complete. Now training will be fast.")
