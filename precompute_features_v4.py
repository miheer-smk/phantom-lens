# Phantom Lens Research Framework
# Developer: Miheer Satish Kulkarni | IIIT Nagpur | 2026
# All Rights Reserved

"""
Phantom Lens / PRISM — Feature Extractor V4
RELATIVE intra-video physics features for domain invariance

KEY CHANGE FROM V3:
  V3 computed ABSOLUTE feature values — noise energy = 5.9
  V4 computes RELATIVE ratios — face_noise / background_noise = 1.8

  Why this matters:
    Absolute values depend on camera ISO, codec, resolution
    Ratios cancel out camera/codec because both face and background
    were recorded with the same camera in the same video
    Real face: face/background ratio ≈ 1.0 (same physics everywhere)
    Fake face: face/background ratio >> 1.0 (composited face has different physics)

  This makes features domain-invariant across FF++, CelebVHQ, Celeb-DF

Feature vector: 24-dim relative features
  R1:  Face/BG noise ratio              dims 0-2
  R2:  Face/BG PRNU ratio               dims 3-4
  R3:  Face/BG Bayer consistency ratio  dims 5-6
  R4:  Face/BG lighting ratio           dims 7-9
  R5:  Face/BG specular ratio           dims 10-11
  R6:  Face/BG DCT artifact ratio       dims 12-14
  R7:  Temporal consistency ratio       dims 15-16
  R8:  Face/BG blur ratio               dims 17-18
  R9:  Flow consistency ratio           dims 19-20
  R10: Chromatic aberration ratio       dims 21-23

Author: Miheer Satish Kulkarni, IIIT Nagpur
"""

import os
import cv2
import imageio
import pickle
import numpy as np
from scipy import fftpack


# ── FACE REGION EXTRACTION ────────────────────────────────────────────────────

def get_face_bg_masks(h, w, face_fraction=0.5):
    """
    Returns face mask and background mask.
    Face: central face_fraction of image
    Background: corners — guaranteed same camera, different scene content
    """
    cy, cx = h // 2, w // 2
    fh = int(h * face_fraction / 2)
    fw = int(w * face_fraction / 2)
    fh = max(fh, 16)
    fw = max(fw, 16)

    face_mask = np.zeros((h, w), dtype=bool)
    face_mask[cy-fh:cy+fh, cx-fw:cx+fw] = True

    bg_mask = np.zeros((h, w), dtype=bool)
    corner = max(h // 6, 8)
    bg_mask[:corner, :corner]   = True   # top-left
    bg_mask[:corner, -corner:]  = True   # top-right
    bg_mask[-corner:, :corner]  = True   # bottom-left
    bg_mask[-corner:, -corner:] = True   # bottom-right

    return face_mask, bg_mask


def safe_ratio(face_val, bg_val, clip_max=10.0):
    """Compute face/bg ratio safely. Returns 1.0 if bg is zero."""
    bg_val = max(float(bg_val), 1e-6)
    face_val = max(float(face_val), 0.0)
    return float(min(face_val / bg_val, clip_max))


# ── RELATIVE PILLAR FUNCTIONS ─────────────────────────────────────────────────

def compute_R1_noise_ratio(img_gray):
    """
    R1: Face noise / Background noise
    Real: ≈ 1.0 (same camera noise everywhere)
    Fake: > 1.0 (synthesised face has different noise texture)
    """
    img = img_gray.astype(np.float32)
    h, w = img.shape
    face_mask, bg_mask = get_face_bg_masks(h, w)

    blurred = cv2.medianBlur(img_gray, 3).astype(np.float32)
    residual = img - blurred

    face_noise_std  = float(residual[face_mask].std())
    bg_noise_std    = float(residual[bg_mask].std())
    noise_ratio     = safe_ratio(face_noise_std, bg_noise_std)

    def vmr_region(mask):
        region = img[mask]
        if len(region) < 16:
            return 0.0
        mu = region.mean()
        var = region.var()
        return float(var / (mu + 1e-6)) if mu > 5.0 else 0.0

    face_vmr = vmr_region(face_mask)
    bg_vmr   = vmr_region(bg_mask)
    vmr_ratio = safe_ratio(face_vmr, bg_vmr)

    lap = cv2.Laplacian(img_gray, cv2.CV_64F)
    face_hf = float(np.abs(lap[face_mask]).mean())
    bg_hf   = float(np.abs(lap[bg_mask]).mean())
    hf_ratio = safe_ratio(face_hf, bg_hf)

    return np.array([noise_ratio, vmr_ratio, hf_ratio], dtype=np.float32)


def compute_R2_prnu_ratio(img_gray):
    """
    R2: Face PRNU / Background PRNU
    Real: ≈ 1.0 (same sensor everywhere)
    Fake: != 1.0 (face from different source)
    """
    img = img_gray.astype(np.float32)
    h, w = img.shape
    face_mask, bg_mask = get_face_bg_masks(h, w)

    blur = cv2.GaussianBlur(img_gray, (5, 5), 1.0)
    prnu = img - blur.astype(np.float32)

    face_prnu_energy = float(np.abs(prnu[face_mask]).mean())
    bg_prnu_energy   = float(np.abs(prnu[bg_mask]).mean())
    prnu_ratio       = safe_ratio(face_prnu_energy, bg_prnu_energy)

    face_prnu_std = float(prnu[face_mask].std())
    bg_prnu_std   = float(prnu[bg_mask].std())
    std_ratio     = safe_ratio(face_prnu_std, bg_prnu_std)

    return np.array([prnu_ratio, std_ratio], dtype=np.float32)


def compute_R3_bayer_ratio(img_rgb):
    """
    R3: Face channel correlation / Background channel correlation
    Real: consistent Bayer pattern across whole image
    Fake: face has different colour channel correlations
    """
    img = img_rgb.astype(np.float32)
    h, w = img.shape[:2]
    face_mask, bg_mask = get_face_bg_masks(h, w)

    def channel_residual(ch):
        blur = cv2.GaussianBlur(ch.astype(np.uint8), (3,3), 0).astype(np.float32)
        return ch - blur

    r_res = channel_residual(img[:,:,0])
    g_res = channel_residual(img[:,:,1])

    def corr_region(mask):
        r = r_res[mask].flatten()
        g = g_res[mask].flatten()
        if r.std() < 1e-6 or g.std() < 1e-6:
            return 0.5
        return float(np.corrcoef(r, g)[0, 1])

    face_corr = corr_region(face_mask)
    bg_corr   = corr_region(bg_mask)
    corr_diff = float(abs(face_corr - bg_corr))
    corr_ratio = safe_ratio(abs(face_corr) + 1e-6, abs(bg_corr) + 1e-6)

    return np.array([corr_diff, corr_ratio], dtype=np.float32)


def compute_R4_lighting_ratio(img_rgb):
    """
    R4: Face lighting / Background lighting consistency
    Real: physically consistent illumination
    Fake: composited face may have wrong illumination
    """
    img = img_rgb.astype(np.float32) / 255.0
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    face_mask, bg_mask = get_face_bg_masks(h, w)

    face_brightness = float(gray[face_mask].mean())
    bg_brightness   = float(gray[bg_mask].mean())
    brightness_ratio = safe_ratio(face_brightness, bg_brightness)

    specular_threshold = 0.9
    face_specular = float((gray[face_mask] > specular_threshold).mean())
    bg_specular   = float((gray[bg_mask]   > specular_threshold).mean())
    specular_ratio = safe_ratio(face_specular + 1e-6, bg_specular + 1e-6)

    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    face_grad = float(grad_mag[face_mask].std())
    bg_grad   = float(grad_mag[bg_mask].std())
    shadow_ratio = safe_ratio(face_grad, bg_grad)

    return np.array([brightness_ratio, specular_ratio, shadow_ratio],
                    dtype=np.float32)


def compute_R5_specular_ratio(frames_gray):
    """
    R5: Face specular drift / Background specular drift
    Real: both move coherently with camera
    Fake: face specular moves inconsistently
    """
    if len(frames_gray) < 2:
        return np.array([1.0, 1.0], dtype=np.float32)

    face_drifts = []
    bg_drifts   = []

    for frame in frames_gray:
        img = frame.astype(np.float32)
        h, w = img.shape
        face_mask, bg_mask = get_face_bg_masks(h, w)
        threshold = np.percentile(img, 95)
        face_spec = (img > threshold) & face_mask
        bg_spec   = (img > threshold) & bg_mask
        if face_spec.sum() > 0:
            face_drifts.append(np.argwhere(face_spec).mean(axis=0))
        if bg_spec.sum() > 0:
            bg_drifts.append(np.argwhere(bg_spec).mean(axis=0))

    if len(face_drifts) < 2 or len(bg_drifts) < 2:
        return np.array([1.0, 1.0], dtype=np.float32)

    face_drift_mag = float(np.linalg.norm(np.diff(np.array(face_drifts), axis=0), axis=1).mean())
    bg_drift_mag   = float(np.linalg.norm(np.diff(np.array(bg_drifts),   axis=0), axis=1).mean())
    drift_ratio    = safe_ratio(face_drift_mag, bg_drift_mag)

    face_drift_std = float(np.linalg.norm(np.diff(np.array(face_drifts), axis=0), axis=1).std())
    bg_drift_std   = float(np.linalg.norm(np.diff(np.array(bg_drifts),   axis=0), axis=1).std())
    std_ratio      = safe_ratio(face_drift_std + 1e-6, bg_drift_std + 1e-6)

    return np.array([drift_ratio, std_ratio], dtype=np.float32)


def compute_R6_dct_ratio(img_gray):
    """
    R6: Face DCT artifact / Background DCT artifact
    Real: same compression applied to both
    Fake: face may have been compressed differently
    """
    img = img_gray.astype(np.float32)
    h, w = img.shape
    face_mask, bg_mask = get_face_bg_masks(h, w)

    def benford_dev_region(region_img):
        dct_coeffs = []
        rh, rw = region_img.shape
        for i in range(0, rh-8, 8):
            for j in range(0, rw-8, 8):
                block = region_img[i:i+8, j:j+8]
                dct_block = fftpack.dct(
                    fftpack.dct(block.T, norm='ortho').T, norm='ortho')
                dct_coeffs.extend(np.abs(dct_block.flatten()[1:]))
        dct_coeffs = np.array(dct_coeffs)
        dct_coeffs = dct_coeffs[dct_coeffs > 1.0]
        if len(dct_coeffs) < 50:
            return 0.0
        ld = np.floor(dct_coeffs / 10**np.floor(
            np.log10(dct_coeffs + 1e-10))).astype(int)
        ld = ld[(ld >= 1) & (ld <= 9)]
        if len(ld) < 20:
            return 0.0
        obs = np.bincount(ld, minlength=10)[1:] / (len(ld) + 1e-6)
        exp = np.array([np.log10(1 + 1/d) for d in range(1, 10)])
        return float(np.sum(np.abs(obs - exp)))

    cy, cx = h//2, w//2
    fh, fw = max(h//4, 16), max(w//4, 16)
    face_region = img[cy-fh:cy+fh, cx-fw:cx+fw]
    bg_region   = img[:h//6, :w//6]

    face_benford = benford_dev_region(face_region)
    bg_benford   = benford_dev_region(bg_region)
    benford_ratio = safe_ratio(face_benford + 1e-6, bg_benford + 1e-6)

    rows1 = img[7::8, :]; rows2 = img[8::8, :]
    min_r = min(rows1.shape[0], rows2.shape[0])
    if min_r > 0 and h > 16:
        block_all  = float(np.abs(rows1[:min_r]-rows2[:min_r]).mean())
        block_face = float(np.abs(
            rows1[:min_r, cx-fw:cx+fw] - rows2[:min_r, cx-fw:cx+fw]).mean())
        block_ratio = safe_ratio(block_face + 1e-6, block_all + 1e-6)
    else:
        block_ratio = 1.0

    return np.array([benford_ratio, block_ratio, 1.0], dtype=np.float32)


def compute_R7_temporal_ratio(frames_gray):
    """
    R7: Face temporal residual / Background temporal residual
    Real: consistent residuals across face and background
    Fake: face region has different temporal statistics
    """
    if len(frames_gray) < 2:
        return np.array([1.0, 1.0], dtype=np.float32)

    face_residuals = []
    bg_residuals   = []

    for i in range(1, len(frames_gray)):
        prev = frames_gray[i-1].astype(np.float32)
        curr = frames_gray[i].astype(np.float32)
        diff = np.abs(curr - prev)
        h, w = diff.shape
        face_mask, bg_mask = get_face_bg_masks(h, w)
        face_residuals.append(float(diff[face_mask].mean()))
        bg_residuals.append(float(diff[bg_mask].mean()))

    face_res_mean = float(np.mean(face_residuals))
    bg_res_mean   = float(np.mean(bg_residuals))
    res_ratio     = safe_ratio(face_res_mean, bg_res_mean)

    face_res_var  = float(np.var(face_residuals))
    bg_res_var    = float(np.var(bg_residuals))
    var_ratio     = safe_ratio(face_res_var + 1e-6, bg_res_var + 1e-6)

    return np.array([res_ratio, var_ratio], dtype=np.float32)


def compute_R8_blur_ratio(frames_gray):
    """
    R8: Face blur / Background blur
    Real: blur consistent across whole frame
    Fake: face region may have different sharpness
    """
    if len(frames_gray) == 0:
        return np.array([1.0, 1.0], dtype=np.float32)

    face_blurs = []
    bg_blurs   = []

    for frame in frames_gray:
        h, w = frame.shape
        face_mask, bg_mask = get_face_bg_masks(h, w)
        lap = cv2.Laplacian(frame, cv2.CV_64F)
        face_blurs.append(float(np.abs(lap[face_mask]).mean()))
        bg_blurs.append(float(np.abs(lap[bg_mask]).mean()))

    blur_ratio = safe_ratio(float(np.mean(face_blurs)),
                            float(np.mean(bg_blurs)))
    var_ratio  = safe_ratio(float(np.var(face_blurs)) + 1e-6,
                            float(np.var(bg_blurs))  + 1e-6)

    return np.array([blur_ratio, var_ratio], dtype=np.float32)


def compute_R9_flow_ratio(frames_gray):
    """
    R9: Face optical flow / Background optical flow
    Real: natural motion coherence between face and background
    Fake: face motion may be inconsistent with background
    """
    if len(frames_gray) < 2:
        return np.array([1.0, 1.0], dtype=np.float32)

    face_flows = []
    bg_flows   = []

    for i in range(1, min(len(frames_gray), 8)):
        try:
            prev = frames_gray[i-1]
            curr = frames_gray[i]
            flow = cv2.calcOpticalFlowFarneback(
                prev, curr, None,
                pyr_scale=0.5, levels=2, winsize=10,
                iterations=2, poly_n=5, poly_sigma=1.1, flags=0)
            mag = np.sqrt(flow[...,0]**2 + flow[...,1]**2)
            h, w = mag.shape
            face_mask, bg_mask = get_face_bg_masks(h, w)
            face_flows.append(float(mag[face_mask].mean()))
            bg_flows.append(float(mag[bg_mask].mean()))
        except Exception:
            continue

    if not face_flows:
        return np.array([1.0, 1.0], dtype=np.float32)

    flow_ratio = safe_ratio(float(np.mean(face_flows)),
                            float(np.mean(bg_flows)))
    consistency = safe_ratio(float(np.std(face_flows)) + 1e-6,
                             float(np.std(bg_flows))  + 1e-6)

    return np.array([flow_ratio, consistency], dtype=np.float32)


def compute_R10_chromatic_ratio(img_rgb):
    """
    R10: Face chromatic aberration / Background chromatic aberration
    Real: same lens distortion across whole frame
    Fake: face may have different chromatic properties
    """
    img = img_rgb.astype(np.float32)
    h, w = img.shape[:2]
    face_mask, bg_mask = get_face_bg_masks(h, w)

    r = img[:,:,0]
    g = img[:,:,1]
    b = img[:,:,2]

    rg_diff = np.abs(r - g)
    bg_diff = np.abs(b - g)

    face_rg = float(rg_diff[face_mask].mean())
    back_rg = float(rg_diff[bg_mask].mean())
    rg_ratio = safe_ratio(face_rg + 1e-6, back_rg + 1e-6)

    face_bg_ch = float(bg_diff[face_mask].mean())
    back_bg_ch = float(bg_diff[bg_mask].mean())
    bg_ratio   = safe_ratio(face_bg_ch + 1e-6, back_bg_ch + 1e-6)

    face_sat = float(np.std(img[face_mask], axis=0).mean())
    back_sat = float(np.std(img[bg_mask],   axis=0).mean())
    sat_ratio = safe_ratio(face_sat + 1e-6, back_sat + 1e-6)

    return np.array([rg_ratio, bg_ratio, sat_ratio], dtype=np.float32)


# ── MAIN EXTRACTION ───────────────────────────────────────────────────────────

def extract_features_from_video(video_path, n_frames=16):
    """
    Extract 24-dim relative physics features from video.
    All features are face/background ratios — domain invariant.
    Returns list of 24-dim feature vectors (one per frame).
    """
    try:
        reader = imageio.get_reader(video_path, format='ffmpeg')
        meta   = reader.get_meta_data()
        fps    = meta.get('fps', 25.0)
        duration = meta.get('duration', 0)
        total_frames = int(duration * fps)
        if total_frames < 2:
            total_frames = 100
        start = int(total_frames * 0.20)
        end   = int(total_frames * 0.80)
        if end - start < n_frames:
            start = 0
            end   = total_frames
        target_indices = set(
            [int(x) for x in np.linspace(start, end-1, n_frames)])

        frames_bgr  = []
        frames_gray = []
        for idx in sorted(target_indices):
            try:
                frame_rgb  = reader.get_data(idx)
                resized    = cv2.resize(frame_rgb, (224, 224))
                frame_bgr  = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
                frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                frames_bgr.append(frame_bgr)
                frames_gray.append(frame_gray)
            except Exception:
                continue
        reader.close()

        if len(frames_bgr) == 0:
            return []

        # Temporal features computed once per video
        r5 = compute_R5_specular_ratio(frames_gray)
        r7 = compute_R7_temporal_ratio(frames_gray)
        r8 = compute_R8_blur_ratio(frames_gray)
        r9 = compute_R9_flow_ratio(frames_gray)

        features = []
        for frame_bgr, frame_gray in zip(frames_bgr, frames_gray):
            try:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                r1  = compute_R1_noise_ratio(frame_gray)
                r2  = compute_R2_prnu_ratio(frame_gray)
                r3  = compute_R3_bayer_ratio(frame_rgb)
                r4  = compute_R4_lighting_ratio(frame_rgb)
                r6  = compute_R6_dct_ratio(frame_gray)
                r10 = compute_R10_chromatic_ratio(frame_rgb)

                feat = np.concatenate([
                    r1,   # dims 0-2   noise ratio
                    r2,   # dims 3-4   PRNU ratio
                    r3,   # dims 5-6   Bayer ratio
                    r4,   # dims 7-9   lighting ratio
                    r5,   # dims 10-11 specular ratio (temporal)
                    r6,   # dims 12-14 DCT ratio
                    r7,   # dims 15-16 temporal ratio
                    r8,   # dims 17-18 blur ratio
                    r9,   # dims 19-20 flow ratio
                    r10,  # dims 21-23 chromatic ratio
                ])

                assert feat.shape == (24,), \
                    f"Expected 24 dims, got {feat.shape}"

                feat = np.nan_to_num(
                    feat, nan=1.0, posinf=5.0, neginf=0.0)
                feat = np.clip(feat, 0.0, 10.0)

                features.append(feat.astype(np.float32))
            except Exception:
                continue

        return features

    except Exception:
        return []


# ── PKL UTILITIES ─────────────────────────────────────────────────────────────

def load_or_create_pkl(pkl_path):
    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        data['labels']          = list(data['labels'])
        data['video_ids']       = list(data['video_ids'])
        data['dataset_sources'] = list(data['dataset_sources'])
        data['generator_types'] = list(data['generator_types'])
        n    = len(data['labels'])
        real = sum(1 for l in data['labels'] if l == 0)
        fake = sum(1 for l in data['labels'] if l == 1)
        print(f"Loaded pkl: {n} samples | Real: {real} | Fake: {fake}")
        print(f"  Feature dims: {data['features'].shape[1]}")
    else:
        print("Creating fresh pkl (24-dim V4 relative features)")
        data = {
            'features':        np.empty((0, 24), dtype=np.float32),
            'labels':          [],
            'video_ids':       [],
            'dataset_sources': [],
            'generator_types': [],
        }
    return data


def save_pkl(data, pkl_path):
    os.makedirs(
        os.path.dirname(pkl_path) if os.path.dirname(pkl_path) else '.',
        exist_ok=True)
    n    = len(data['labels'])
    real = sum(1 for l in data['labels'] if l == 0)
    fake = sum(1 for l in data['labels'] if l == 1)
    with open(pkl_path, 'wb') as f:
        pickle.dump(data, f, protocol=4)
    print(f"Saved pkl: {n} total | {real} real | {fake} fake")


def append_to_pkl(data, new_features, new_labels,
                  new_video_ids, new_sources, new_generators):
    if len(new_features) == 0:
        return data
    new_feat_array = np.array(new_features, dtype=np.float32)
    if data['features'].shape[0] == 0:
        data['features'] = new_feat_array
    else:
        data['features'] = np.vstack([data['features'], new_feat_array])
    data['labels']          = list(data['labels'])          + list(new_labels)
    data['video_ids']       = list(data['video_ids'])       + list(new_video_ids)
    data['dataset_sources'] = list(data['dataset_sources']) + list(new_sources)
    data['generator_types'] = list(data['generator_types']) + list(new_generators)
    return data


# ── QUICK TEST ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        path = sys.argv[1]
        print(f"Testing on: {path}")
        feats = extract_features_from_video(path, n_frames=4)
        if feats:
            f = np.array(feats)
            print(f"Shape: {f.shape}")
            print(f"Mean:  {f.mean(axis=0).round(3)}")
            print(f"Std:   {f.std(axis=0).round(3)}")
            print(f"Min:   {f.min(axis=0).round(3)}")
            print(f"Max:   {f.max(axis=0).round(3)}")
            print("All values are ratios — real video should be near 1.0")
        else:
            print("Feature extraction failed")
    else:
        print("Usage: python precompute_features_v4.py <video_path>")
        print("Feature vector: 24-dim relative face/background ratios")