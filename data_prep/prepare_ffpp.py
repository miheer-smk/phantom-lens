# Phantom Lens Research Framework
# Developer: Miheer Satish Kulkarni | IIIT Nagpur | 2026
# All Rights Reserved


import os
import cv2
import pickle
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from src.pillars.pillar1_noise import compute_pillar1_score
from src.pillars.pillar2_light import compute_pillar2_score
from src.pillars.pillar3_compression import compute_pillar3_score

PKL_PATH = "data/precomputed_features.pkl"
FF_ROOT = "data/ff_c23/FaceForensics++_C23"
FRAMES_PER_VIDEO = 8

FAKE_FOLDERS = ["Deepfakes", "Face2Face", "FaceShifter", "FaceSwap", "NeuralTextures"]

def load_frame(path):
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.0

def extract_frames(video_path, n_frames):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 1:
        cap.release()
        return []
    indices = [int(i * total / n_frames) for i in range(n_frames)]
    frames = []
    idx = 0
    next_t = 0
    while cap.isOpened() and next_t < len(indices):
        ret, frame = cap.read()
        if not ret:
            break
        if idx == indices[next_t]:
            frame = cv2.resize(frame, (224, 224))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame.astype(np.float32) / 255.0)
            next_t += 1
        idx += 1
    cap.release()
    return frames

def compute_features(frames):
    try:
        p1s = [compute_pillar1_score(f) for f in frames]
        p2s = [compute_pillar2_score(f) for f in frames]
        p3s = [compute_pillar3_score(f) for f in frames]
        f1_mean = np.mean([s['f1_raw'] for s in p1s])
        f1_std = np.std([s['f1_raw'] for s in p1s])
        f2_raw = np.mean([s['f2_raw'] for s in p2s])
        f2_var = np.var([s['f2_raw'] for s in p2s])
        f3_mean = np.mean([s['f3_raw'] for s in p3s])
        f3_benford = np.mean([s['benford_deviation'] for s in p3s])
        return np.array([f1_mean, f1_std, f2_raw, f2_var, f3_mean, f3_benford,
                         f1_mean * f2_raw, f1_mean * f3_mean], dtype=np.float32)
    except:
        return None

def process_video(video_path):
    frames = extract_frames(video_path, FRAMES_PER_VIDEO)
    if len(frames) == 0:
        return None
    return compute_features(frames)

# Load existing pkl
print("Loading existing pkl...")
with open(PKL_PATH, 'rb') as f:
    data = pickle.load(f)
existing_features = data['features']
existing_labels = data['labels']
print(f"Existing samples: {len(existing_labels)}")

# Find all fake videos
all_videos = []
for folder in FAKE_FOLDERS:
    folder_path = os.path.join(FF_ROOT, folder)
    if os.path.exists(folder_path):
        videos = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.mp4')]
        print(f"{folder}: {len(videos)} videos")
        all_videos.extend(videos)

print(f"Total fake videos: {len(all_videos)}")

new_features = []
new_labels = []
failed = 0

for video_path in tqdm(all_videos):
    feat = process_video(video_path)
    if feat is not None:
        new_features.append(feat)
        new_labels.append(1)
    else:
        failed += 1

print(f"Processed: {len(new_features)}, Failed: {failed}")

all_features = np.vstack([existing_features, np.array(new_features)])
all_labels = np.array(list(existing_labels) + new_labels)
print(f"Total samples after merge: {len(all_labels)}")

with open(PKL_PATH, 'wb') as f:
    pickle.dump({'features': all_features, 'labels': all_labels}, f, protocol=4)

print("Saved. Ready to train.")
