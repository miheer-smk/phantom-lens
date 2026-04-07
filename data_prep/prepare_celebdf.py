# Phantom Lens Research Framework
# Developer: Miheer Satish Kulkarni | IIIT Nagpur | 2026
# All Rights Reserved



import os
import cv2
from tqdm import tqdm

REAL_DIR = "data/celebdf_v2/Celeb-real"
FAKE_DIR = "data/celebdf_v2/Celeb-synthesis"
OUTPUT_REAL = "data/real"
OUTPUT_FAKE = "data/fake"
FRAMES_PER_VIDEO = 8

def extract_frames_from_video(video_path, output_folder, n_frames=8):
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

# Process real videos
real_videos = [f for f in os.listdir(REAL_DIR) if f.endswith('.mp4')]
print(f"Processing {len(real_videos)} real videos...")
for i, video_file in enumerate(tqdm(real_videos)):
    video_path = os.path.join(REAL_DIR, video_file)
    folder_name = f"celebdf_real_{i:04d}"
    output_folder = os.path.join(OUTPUT_REAL, folder_name)
    extract_frames_from_video(video_path, output_folder, FRAMES_PER_VIDEO)

# Process fake videos
fake_videos = [f for f in os.listdir(FAKE_DIR) if f.endswith('.mp4')]
print(f"\nProcessing {len(fake_videos)} fake videos...")
for i, video_file in enumerate(tqdm(fake_videos)):
    video_path = os.path.join(FAKE_DIR, video_file)
    folder_name = f"celebdf_fake_{i:04d}"
    output_folder = os.path.join(OUTPUT_FAKE, folder_name)
    extract_frames_from_video(video_path, output_folder, FRAMES_PER_VIDEO)

print("\nDone. Frames extracted successfully.")
