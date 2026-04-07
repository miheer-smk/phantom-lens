# Phantom Lens Research Framework
# Developer: Miheer Satish Kulkarni | IIIT Nagpur | 2026
# All Rights Reserved


import numpy as np
import cv2
import os

for i in range(10):
    os.makedirs(f'data/real/video_{i:03d}', exist_ok=True)
    img = np.random.poisson(128, (224,224,3)).clip(0,255).astype(np.uint8)
    cv2.imwrite(f'data/real/video_{i:03d}/frame_0000.jpg', img)

for i in range(10):
    os.makedirs(f'data/fake/video_{i:03d}', exist_ok=True)
    img = np.random.normal(128, 15, (224,224,3)).clip(0,255).astype(np.uint8)
    cv2.imwrite(f'data/fake/video_{i:03d}/frame_0000.jpg', img)

print('Created 10 real and 10 fake samples')
