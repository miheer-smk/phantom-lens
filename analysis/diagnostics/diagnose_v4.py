"""
Diagnose V4 relative features — real vs fake comparison
"""
import sys
sys.path.insert(0, '.')
from precompute_features_v4 import *
import cv2
import imageio
import numpy as np

real_path = 'data/ffpp_official/original_sequences/youtube/c23/videos/000.mp4'
fake_path = 'data/ffpp_official/manipulated_sequences/Deepfakes/c23/videos/000_003.mp4'

def get_frames(path, n=8):
    reader = imageio.get_reader(path, format='ffmpeg')
    frames_bgr = []
    frames_gray = []
    for i in range(0, 80, 10):
        try:
            f = reader.get_data(i)
            f = cv2.resize(f, (224, 224))
            b = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
            g = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
            frames_bgr.append(b)
            frames_gray.append(g)
            if len(frames_bgr) >= n:
                break
        except:
            pass
    reader.close()
    return frames_bgr, frames_gray

print("Loading frames...")
rbgr, rgray = get_frames(real_path)
fbgr, fgray = get_frames(fake_path)
rrgb = cv2.cvtColor(rbgr[0], cv2.COLOR_BGR2RGB)
frgb = cv2.cvtColor(fbgr[0], cv2.COLOR_BGR2RGB)

print(f"\nFrames loaded: real={len(rgray)} fake={len(fgray)}")
print("\n" + "="*55)
print("PILLAR-BY-PILLAR REAL vs FAKE COMPARISON")
print("="*55)

pillars = [
    ("R1 Noise ratio",    compute_R1_noise_ratio,    rgray[0], fgray[0]),
    ("R2 PRNU ratio",     compute_R2_prnu_ratio,     rgray[0], fgray[0]),
    ("R3 Bayer ratio",    compute_R3_bayer_ratio,    rrgb,     frgb),
    ("R4 Lighting ratio", compute_R4_lighting_ratio, rrgb,     frgb),
    ("R10 Chromatic",     compute_R10_chromatic_ratio, rrgb,   frgb),
]

for name, fn, r_inp, f_inp in pillars:
    r_val = fn(r_inp)
    f_val = fn(f_inp)
    diff  = np.abs(r_val - f_val)
    print(f"\n{name}:")
    print(f"  Real: {np.round(r_val, 4)}")
    print(f"  Fake: {np.round(f_val, 4)}")
    print(f"  Diff: {np.round(diff, 4)}")

print("\nTemporal pillars:")
r5r = compute_R5_specular_ratio(rgray)
r5f = compute_R5_specular_ratio(fgray)
print(f"\nR5 Specular ratio:")
print(f"  Real: {np.round(r5r, 4)}")
print(f"  Fake: {np.round(r5f, 4)}")

r7r = compute_R7_temporal_ratio(rgray)
r7f = compute_R7_temporal_ratio(fgray)
print(f"\nR7 Temporal ratio:")
print(f"  Real: {np.round(r7r, 4)}")
print(f"  Fake: {np.round(r7f, 4)}")

r8r = compute_R8_blur_ratio(rgray)
r8f = compute_R8_blur_ratio(fgray)
print(f"\nR8 Blur ratio:")
print(f"  Real: {np.round(r8r, 4)}")
print(f"  Fake: {np.round(r8f, 4)}")

r9r = compute_R9_flow_ratio(rgray)
r9f = compute_R9_flow_ratio(fgray)
print(f"\nR9 Flow ratio:")
print(f"  Real: {np.round(r9r, 4)}")
print(f"  Fake: {np.round(r9f, 4)}")

print("\n" + "="*55)
print("SUMMARY — which features show meaningful difference?")
print("="*55)

all_real = np.concatenate([
    compute_R1_noise_ratio(rgray[0]),
    compute_R2_prnu_ratio(rgray[0]),
    compute_R3_bayer_ratio(rrgb),
    compute_R4_lighting_ratio(rrgb),
    r5r, r7r, r8r, r9r,
    compute_R10_chromatic_ratio(rrgb),
])
all_fake = np.concatenate([
    compute_R1_noise_ratio(fgray[0]),
    compute_R2_prnu_ratio(fgray[0]),
    compute_R3_bayer_ratio(frgb),
    compute_R4_lighting_ratio(frgb),
    r5f, r7f, r8f, r9f,
    compute_R10_chromatic_ratio(frgb),
])

dim_names = [
    'R1_noise','R1_vmr','R1_hf',
    'R2_prnu','R2_std',
    'R3_corr_diff','R3_corr_ratio',
    'R4_bright','R4_specular','R4_shadow',
    'R5_drift','R5_std',
    'R7_res','R7_var',
    'R8_blur','R8_var',
    'R9_flow','R9_consist',
    'R10_rg','R10_bg','R10_sat',
]

print(f"\n{'Dim':<20} {'Real':>8} {'Fake':>8} {'Diff':>8} {'Signal?':>10}")
print("-"*55)
for i, (name, r, f) in enumerate(zip(dim_names, all_real, all_fake)):
    d = abs(r - f)
    sig = "YES ***" if d > 0.1 else "mild" if d > 0.05 else "no"
    print(f"{name:<20} {r:>8.4f} {f:>8.4f} {d:>8.4f} {sig:>10}")