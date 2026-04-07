# Phantom Lens Research Framework
# Developer: Miheer Satish Kulkarni | IIIT Nagpur | 2026
# All Rights Reserved


import os
import cv2
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

VIDEO_DIR = "data/celebvhq/35666"
OUTPUT_REAL = "data/real"
FRAMES_PER_VIDEO = 8
NUM_WORKERS = 8

def extract_frames_from_video(args):
    video_path, output_folder, n_frames = args
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 1:
        cap.release()
        return 0
    indices = [int(i * total_frames / n_frames) for i in range(n_frames)]
    saved = 0
    frame_idx = 0
    next_target = 0
    while cap.isOpened() and next_target < len(indices):
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx == indices[next_target]:
            path = os.path.join(output_folder, f"frame_{saved:04d}.jpg")
            cv2.imwrite(path, frame)
            saved += 1
            next_target += 1
        frame_idx += 1
    cap.release()
    return saved

videos = [f for f in os.listdir(VIDEO_DIR) if f.endswith('.mp4')]
print(f"Found {len(videos)} real videos")

args = [(os.path.join(VIDEO_DIR, v), os.path.join(OUTPUT_REAL, f"celebvhq_{i:05d}"), FRAMES_PER_VIDEO)
        for i, v in enumerate(videos)]

print(f"Extracting frames with {NUM_WORKERS} workers...")
with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
    list(tqdm(executor.map(extract_frames_from_video, args), total=len(args)))

print("\nDone. All frames extracted.")
