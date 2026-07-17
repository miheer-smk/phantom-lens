# Phantom Lens Research Framework
# Developer: Miheer Satish Kulkarni | IIIT Nagpur | 2026
# All Rights Reserved


import os
import shutil
from tqdm import tqdm

src_real = "data/real_fake_faces/real_and_fake_face_detection/real_and_fake_face/training_real"
src_fake = "data/real_fake_faces/real_and_fake_face_detection/real_and_fake_face/training_fake"

# Get all images
real_images = [f for f in os.listdir(src_real) if f.endswith('.jpg') or f.endswith('.png')]
fake_images = [f for f in os.listdir(src_fake) if f.endswith('.jpg') or f.endswith('.png')]

print(f"Real images found: {len(real_images)}")
print(f"Fake images found: {len(fake_images)}")

# Copy real images into individual video folders
print("Organising real images...")
for i, img_file in enumerate(tqdm(real_images)):
    folder = f"data/real/video_{i:04d}"
    os.makedirs(folder, exist_ok=True)
    shutil.copy(os.path.join(src_real, img_file), os.path.join(folder, "frame_0000.jpg"))

# Copy fake images into individual video folders
print("Organising fake images...")
for i, img_file in enumerate(tqdm(fake_images)):
    folder = f"data/fake/video_{i:04d}"
    os.makedirs(folder, exist_ok=True)
    shutil.copy(os.path.join(src_fake, img_file), os.path.join(folder, "frame_0000.jpg"))

print(f"Done. {len(real_images)} real and {len(fake_images)} fake samples ready.")
