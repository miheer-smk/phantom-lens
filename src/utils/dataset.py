# Phantom Lens Research Framework
# Developer: Miheer Satish Kulkarni | IIIT Nagpur | 2026
# All Rights Reserved



import os
import numpy as np
import cv2
from torch.utils.data import Dataset
import torch

class PhantomLensDataset(Dataset):
    def __init__(self, data_root, split='train', n_frames=8):
        self.data_root = data_root
        self.n_frames = n_frames
        self.samples = []

        for label, category in enumerate(['real', 'fake']):
            category_path = os.path.join(data_root, category)
            if not os.path.exists(category_path):
                continue
            for video_folder in sorted(os.listdir(category_path)):
                folder_path = os.path.join(category_path, video_folder)
                if os.path.isdir(folder_path):
                    frames = sorted([f for f in os.listdir(folder_path)
                                   if f.endswith('.png') or f.endswith('.jpg')])
                    if len(frames) >= 1:
                        self.samples.append((folder_path, frames, label))

        n = len(self.samples)
        if split == 'train':
            self.samples = self.samples[:int(0.8*n)]
        else:
            self.samples = self.samples[int(0.8*n):]

        print(f"{split}: {len(self.samples)} samples loaded")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        folder_path, frames, label = self.samples[idx]
        indices = np.linspace(0, len(frames)-1, self.n_frames, dtype=int)
        selected_frames = [frames[i] for i in indices]
        loaded = []
        for frame_file in selected_frames:
            frame_path = os.path.join(folder_path, frame_file)
            img = cv2.imread(frame_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))
            img = img.astype(np.float32) / 255.0
            loaded.append(img)
        frames_array = np.stack(loaded, axis=0)
        frames_tensor = torch.from_numpy(frames_array).permute(0, 3, 1, 2)
        return frames_tensor, torch.tensor(label, dtype=torch.float32)