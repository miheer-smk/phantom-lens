# Phantom Lens Research Framework
# Developer: Miheer Satish Kulkarni | IIIT Nagpur | 2026
# All Rights Reserved



import cv2
import numpy as np
import os

def load_frame(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.0

def extract_frames(video_path, output_folder, max_frames=64):
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved = 0
    while cap.isOpened() and saved < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % 3 == 0:
            path = os.path.join(output_folder, f"frame_{saved:04d}.png")
            cv2.imwrite(path, frame)
            saved += 1
        frame_count += 1
    cap.release()
    return saved