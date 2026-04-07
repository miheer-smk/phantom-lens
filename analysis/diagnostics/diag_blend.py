"""
Blending Boundary Gradient Diagnostic — Celeb-DF v2
Tests whether boundary gradient features discriminate real vs fake.
"""
import os
import sys
import numpy as np
import imageio.v3 as iio
import cv2
sys.path.insert(0, '.')
from blend_boundary import compute_blend_boundary

REAL_DIR  = "data/celebdf/real"
FAKE_DIR  = "data/celebdf/fake"
N_VIDEOS  = 20
N_FRAMES  = 16

def get_frames(path, n=16):
    try:
        all_frames = iio.imread(path, plugin="pyav")
        total = len(all_frames)
        if total == 0:
            return []
        indices = np.linspace(0, total-1, min(n, total), dtype=int)
        frames = []
        for idx in indices:
            f = cv2.cvtColor(all_frames[idx], cv2.COLOR_RGB2BGR)
            f = cv2.resize(f, (224, 224))
            frames.append(f)
        return frames
    except:
        return []

def cohens_d(a, b):
    if len(a) < 2 or len(b) < 2:
        return 0.0
    pooled = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
    return (np.mean(a) - np.mean(b)) / (pooled + 1e-8)

real_feats = []
fake_feats = []

print("Processing REAL...")
for vf in sorted(os.listdir(REAL_DIR))[:N_VIDEOS]:
    if not vf.endswith('.mp4'): continue
    frames = get_frames(os.path.join(REAL_DIR, vf), N_FRAMES)
    if len(frames) < 4: continue
    feat = np.mean([compute_blend_boundary(f) for f in frames], axis=0)
    real_feats.append(feat)
    print(f"  {vf}  bgr={feat[0]:.4f}  coh={feat[1]:.4f}  bbg={feat[2]:.4f}")

print("\nProcessing FAKE...")
for vf in sorted(os.listdir(FAKE_DIR))[:N_VIDEOS]:
    if not vf.endswith('.mp4'): continue
    frames = get_frames(os.path.join(FAKE_DIR, vf), N_FRAMES)
    if len(frames) < 4: continue
    feat = np.mean([compute_blend_boundary(f) for f in frames], axis=0)
    fake_feats.append(feat)
    print(f"  {vf}  bgr={feat[0]:.4f}  coh={feat[1]:.4f}  bbg={feat[2]:.4f}")

real_arr = np.array(real_feats)
fake_arr = np.array(fake_feats)
names = ['boundary_grad_ratio', 'grad_direction_coh', 'boundary_vs_bg']

print(f"\n{'='*65}")
print("RESULTS")
print(f"{'='*65}")
print(f"  {'Feature':<22} {'Real':>8} {'Fake':>8} {'Diff':>8} {'d':>8} {'Signal?':>10}")
print(f"  {'-'*62}")

any_disc = False
for j, name in enumerate(names):
    rm = real_arr[:,j].mean()
    fm = fake_arr[:,j].mean()
    d  = cohens_d(real_arr[:,j], fake_arr[:,j])
    sig = "YES ***" if abs(d) > 0.3 else "mild" if abs(d) > 0.2 else "no"
    if abs(d) > 0.3: any_disc = True
    direction = "real>fake" if rm > fm else "fake>real"
    print(f"  {name:<22} {rm:>8.4f} {fm:>8.4f} {rm-fm:>+8.4f} {d:>+8.4f} {sig:>10}  {direction}")

print(f"\n  VERDICT: {'DISCRIMINATIVE' if any_disc else 'NOT DISCRIMINATIVE'}")
print(f"{'='*65}")