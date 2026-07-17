"""
Extract CelebDF Real features — folder 8
Run this after main extraction completes.
"""
import sys
sys.path.insert(0, '.')

# Temporarily patch DATASET_FOLDERS to add CelebDF real as folder 8
import recompute_features_v4 as rv4

rv4.DATASET_FOLDERS.append({
    "path": "data/celebdf/real/",
    "label": 0,
    "source": "celebdf_real",
    "generator": "real",
    "description": "CelebDF Real (TEST ONLY)",
})

import os, pickle
import numpy as np

CHECKPOINT_DIR = "data/v4_checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

folder_idx = 8
folder_cfg = rv4.DATASET_FOLDERS[8]

print("=" * 60)
print("Extracting CelebDF Real features (folder 8 / test set)")
print("=" * 60)

results = rv4.process_folder(folder_cfg, n_frames=8)
ckpt_path = rv4.save_checkpoint(folder_idx, results, CHECKPOINT_DIR)
print(f"Checkpoint saved: {ckpt_path}")
print(f"Videos: {results['stats']['success']} ok, "
      f"{results['stats']['failed']} failed")