# Phantom Lens Research Framework
# Developer: Miheer Satish Kulkarni | IIIT Nagpur | 2026
# All Rights Reserved



import numpy as np
import cv2
from src.utils.video_utils import load_frame
from src.pillars.pillar1_noise import compute_pillar1_score
from src.pillars.pillar2_light import compute_pillar2_score
from src.pillars.pillar3_compression import compute_pillar3_score

# Load real and fake images
real_frame = load_frame('data/real/test_real.jpg')
fake_frame = load_frame('data/fake/test_fake.jpg')

# Convert to grayscale for pillar 2
real_gray = cv2.cvtColor((real_frame*255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)/255.0
fake_gray = cv2.cvtColor((fake_frame*255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)/255.0

print("Testing Pillar 1 (Noise)...")
print("REAL score:", compute_pillar1_score(real_frame))
print("FAKE score:", compute_pillar1_score(fake_frame))

print("\nTesting Pillar 2 (Light)...")
print("REAL score:", compute_pillar2_score([real_gray]*8))
print("FAKE score:", compute_pillar2_score([fake_gray]*8))

print("\nTesting Pillar 3 (Compression)...")
print("REAL score:", compute_pillar3_score(real_frame))
print("FAKE score:", compute_pillar3_score(fake_frame))
