"""
Phantom Lens — Pre-V3 Safety Check
Verifies all V1 assets are saved before starting V3 work.
Run this FIRST before touching anything.
"""

import os
import pickle
import numpy as np

print("=" * 60)
print("  PHANTOM LENS — V1 SAFETY CHECK")
print("=" * 60)

all_good = True

# ── Check checkpoints ─────────────────────────────────────────
print("\n[1] Checking checkpoints...")
required_checkpoints = [
    "checkpoints/best_model_24dim_AUC8961.pt",
    "checkpoints/best_model.pt",
]
for ck in required_checkpoints:
    if os.path.exists(ck):
        size = os.path.getsize(ck)
        print(f"  OK  {ck}  ({size:,} bytes)")
    else:
        print(f"  MISSING  {ck}")
        all_good = False

# ── Check pkl ─────────────────────────────────────────────────
print("\n[2] Checking pkl...")
pkl_path = "data/precomputed_features.pkl"
if os.path.exists(pkl_path):
    size = os.path.getsize(pkl_path)
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    features = np.array(data['features'])
    labels   = np.array(data['labels'])
    real     = int((labels == 0).sum())
    fake     = int((labels == 1).sum())
    print(f"  OK  {pkl_path}  ({size:,} bytes)")
    print(f"      Shape: {features.shape}")
    print(f"      Real: {real}  Fake: {fake}  Total: {len(labels)}")
    sources = sorted(set(data['dataset_sources']))
    print(f"      Sources: {sources}")
else:
    print(f"  MISSING  {pkl_path}")
    all_good = False

# ── Check backup pkl ──────────────────────────────────────────
print("\n[3] Checking pkl backup...")
backup_path = "data/precomputed_features_backup.pkl"
if os.path.exists(backup_path):
    size = os.path.getsize(backup_path)
    print(f"  OK  {backup_path}  ({size:,} bytes)")
else:
    print(f"  MISSING — creating backup now...")
    import shutil
    shutil.copy(pkl_path, backup_path)
    size = os.path.getsize(backup_path)
    print(f"  CREATED  {backup_path}  ({size:,} bytes)")

# ── Check Celeb-DF v2 data ────────────────────────────────────
print("\n[4] Checking Celeb-DF v2 data...")
celebdf_dirs = [
    ("data/celebdf/real", "real videos"),
    ("data/celebdf/fake", "fake videos"),
]
for d, desc in celebdf_dirs:
    if os.path.exists(d):
        videos = [f for f in os.listdir(d)
                  if f.lower().endswith('.mp4')]
        print(f"  OK  {d}  ({len(videos)} {desc})")
    else:
        print(f"  MISSING  {d}")
        all_good = False

# ── Check scripts ─────────────────────────────────────────────
print("\n[5] Checking required scripts...")
required_scripts = [
    "precompute_features_v2.py",
    "train_v2.py",
]
for sc in required_scripts:
    if os.path.exists(sc):
        print(f"  OK  {sc}")
    else:
        print(f"  MISSING  {sc}")
        all_good = False

# ── Check ffmpeg ──────────────────────────────────────────────
print("\n[6] Checking ffmpeg...")
import subprocess
result = subprocess.run(
    ["ffmpeg", "-version"],
    capture_output=True, text=True
)
if result.returncode == 0:
    version_line = result.stdout.split('\n')[0]
    print(f"  OK  {version_line}")
else:
    print(f"  MISSING — run: set PATH=%PATH%;C:\\ffmpeg\\...\\bin")
    all_good = False

# ── Final verdict ─────────────────────────────────────────────
print("\n" + "=" * 60)
if all_good:
    print("  ALL CHECKS PASSED — SAFE TO START V3")
    print("  Your V1 work is secure.")
    print("  Proceed to next step.")
else:
    print("  SOME CHECKS FAILED — FIX BEFORE PROCEEDING")
    print("  Do NOT start V3 until all checks pass.")
print("=" * 60 + "\n")