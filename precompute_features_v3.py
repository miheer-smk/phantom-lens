# Phantom Lens Research Framework
# Developer: Miheer Satish Kulkarni | IIIT Nagpur | 2026
# All Rights Reserved

"""
Phantom Lens / PRISM — Feature Extractor V3
10 original physics pillars + 2 new pillars = 30-dimensional feature vector

Pillars:
  P1:  Photon Shot Noise (unchanged)    dims 0-2
  P2:  PRNU Fingerprint (unchanged)     dims 3-4
  P3:  Bayer Demosaicing (unchanged)    dims 5-6
  P4:  Light Transport (unchanged)      dims 7-9
  P5:  Specular Coherence (unchanged)   dims 10-11
  P6:  DCT Compression (unchanged)      dims 12-14
  P7:  Codec Temporal Residual (unch.)  dims 15-16
  P8:  Motion Blur (unchanged)          dims 17-18
  P9:  Optical Flow (unchanged)         dims 19-20  (dim20=dead)
  P10: Chromatic Aberration (unchanged) dims 21-23
  P11: Eye Reflection Symmetry (NEW)    dims 24-26
  P12: Global Illumination Residue(NEW) dims 27-29

Changes from V2:
  1. Added import from pillar_new
  2. n_frames default changed 8 -> 16
  3. P11 and P12 computed and concatenated
  4. PKL empty shape 24 -> 30
  5. rPPG skipped (ratio=0.788, not discriminative)

Dead dim: P9_boundary dim 20 (var=0.000000) removed at training time
Live dims for training: 30 - 1 = 29

Author: Miheer Satish Kulkarni, IIIT Nagpur
"""

import os
import cv2
import imageio
import pickle
import numpy as np
from scipy import fftpack

from pillar_new import (
    compute_pillar11_eye_symmetry,
    compute_pillar12_illumination
)


def compute_pillar1(img_gray):
    img = img_gray.astype(np.float32)
    h, w = img.shape
    vmr_vals = []
    for i in range(0, h - 8, 8):
        for j in range(0, w - 8, 8):
            block = img[i:i+8, j:j+8]
            mu = block.mean()
            var = block.var()
            if mu > 5.0:
                vmr_vals.append(var / (mu + 1e-6))
    vmr = float(np.median(vmr_vals)) if vmr_vals else 0.0
    blurred = cv2.medianBlur(img_gray, 3).astype(np.float32)
    residual = img - blurred
    residual_std = float(residual.std())
    lap = cv2.Laplacian(img_gray, cv2.CV_64F)
    total_energy = float(np.abs(img).mean()) + 1e-6
    hf_ratio = float(np.abs(lap).mean()) / total_energy
    return np.array([vmr, residual_std, hf_ratio], dtype=np.float32)


def compute_pillar2(img_gray):
    img = img_gray.astype(np.float32)
    h, w = img.shape
    blur = cv2.GaussianBlur(img, (5, 5), 1.0)
    prnu_residual = img - blur
    cy, cx = h // 2, w // 2
    h4, w4 = h // 4, w // 4
    face_region = prnu_residual[cy-h4:cy+h4, cx-w4:cx+w4]
    face_energy = float(np.abs(face_region).mean())
    mask = np.ones((h, w), dtype=bool)
    mask[cy-h4:cy+h4, cx-w4:cx+w4] = False
    periphery_energy = float(np.abs(prnu_residual[mask]).mean())
    ratio = face_energy / (periphery_energy + 1e-6)
    return np.array([face_energy, ratio], dtype=np.float32)


def compute_pillar3(img_rgb):
    img = img_rgb.astype(np.float32)
    def channel_residual(ch):
        blur = cv2.GaussianBlur(ch.astype(np.uint8), (3, 3), 0).astype(np.float32)
        return ch - blur
    r_res = channel_residual(img[:, :, 0]).flatten()
    g_res = channel_residual(img[:, :, 1]).flatten()
    b_res = channel_residual(img[:, :, 2]).flatten()
    def safe_corr(a, b):
        if a.std() < 1e-6 or b.std() < 1e-6:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])
    return np.array([safe_corr(r_res, g_res), safe_corr(b_res, g_res)], dtype=np.float32)


def compute_pillar4(img_rgb):
    img = img_rgb.astype(np.float32) / 255.0
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    cy, cx = h // 2, w // 2
    h4, w4 = h // 4, w // 4
    face = gray[cy-h4:cy+h4, cx-w4:cx+w4]
    top = gray[:h4, :]
    left = gray[:, :w4]
    right = gray[:, -w4:]
    face_mean = float(face.mean())
    bg_mean = float(np.concatenate([top.flatten(), left.flatten(), right.flatten()]).mean())
    face_bg_diff = abs(face_mean - bg_mean)
    specular_ratio = float((face > 0.9).mean())
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    shadow_consistency = float(np.sqrt(grad_x**2 + grad_y**2).std())
    return np.array([face_bg_diff, specular_ratio, shadow_consistency], dtype=np.float32)


def compute_pillar5_single(img_gray):
    img = img_gray.astype(np.float32)
    threshold = np.percentile(img, 95)
    specular_mask = (img > threshold).astype(np.float32)
    specular_count = float(specular_mask.mean())
    if specular_mask.sum() > 0:
        coords = np.argwhere(specular_mask > 0)
        spread = float(coords.std())
    else:
        spread = 0.0
    return np.array([specular_count, spread], dtype=np.float32)


def compute_pillar5_temporal(frames_gray):
    if len(frames_gray) < 2:
        f = compute_pillar5_single(frames_gray[0])
        return np.array([f[0], f[1]], dtype=np.float32)
    specular_positions = []
    for frame in frames_gray:
        img = frame.astype(np.float32)
        threshold = np.percentile(img, 95)
        mask = (img > threshold)
        if mask.sum() > 0:
            coords = np.argwhere(mask)
            specular_positions.append(coords.mean(axis=0))
    if len(specular_positions) < 2:
        return np.array([0.0, 0.0], dtype=np.float32)
    positions = np.array(specular_positions)
    drifts = np.diff(positions, axis=0)
    drift_magnitudes = np.linalg.norm(drifts, axis=1)
    return np.array([float(drift_magnitudes.mean()), float(drift_magnitudes.std())], dtype=np.float32)


def compute_pillar6(img_gray):
    img = img_gray.astype(np.float32)
    h, w = img.shape
    dct_coeffs = []
    for i in range(0, h - 8, 8):
        for j in range(0, w - 8, 8):
            block = img[i:i+8, j:j+8]
            dct_block = fftpack.dct(fftpack.dct(block.T, norm='ortho').T, norm='ortho')
            dct_coeffs.extend(np.abs(dct_block.flatten()[1:]))
    dct_coeffs = np.array(dct_coeffs)
    dct_coeffs = dct_coeffs[dct_coeffs > 1.0]
    if len(dct_coeffs) > 100:
        leading_digits = np.floor(dct_coeffs / 10**np.floor(np.log10(dct_coeffs + 1e-10))).astype(int)
        leading_digits = leading_digits[(leading_digits >= 1) & (leading_digits <= 9)]
        if len(leading_digits) > 50:
            observed = np.bincount(leading_digits, minlength=10)[1:] / (len(leading_digits) + 1e-6)
            expected = np.array([np.log10(1 + 1/d) for d in range(1, 10)])
            benford_dev = float(np.sum(np.abs(observed - expected)))
        else:
            benford_dev = 0.0
    else:
        benford_dev = 0.0
    rows1 = img[7::8, :]
    rows2 = img[8::8, :]
    min_rows = min(rows1.shape[0], rows2.shape[0])
    h_diff = np.abs(rows1[:min_rows, :] - rows2[:min_rows, :]) if h > 16 else np.array([0.0])
    cols1 = img[:, 7::8]
    cols2 = img[:, 8::8]
    min_cols = min(cols1.shape[1], cols2.shape[1])
    v_diff = np.abs(cols1[:, :min_cols] - cols2[:, :min_cols]) if w > 16 else np.array([0.0])
    block_artifact = float(np.concatenate([h_diff.flatten(), v_diff.flatten()]).mean())
    hist, _ = np.histogram(dct_coeffs[:1000] if len(dct_coeffs) > 1000 else dct_coeffs, bins=50, range=(0, 100))
    hist_fft = np.abs(fftpack.fft(hist.astype(np.float32)))
    double_compress_score = float(hist_fft[2:8].max() / (hist_fft[1:].mean() + 1e-6))
    return np.array([benford_dev, block_artifact, double_compress_score], dtype=np.float32)


def compute_pillar7(frames_gray):
    if len(frames_gray) < 2:
        return np.array([0.0, 0.0], dtype=np.float32)
    residuals = []
    spatial_vars = []
    for i in range(1, len(frames_gray)):
        prev = cv2.resize(frames_gray[i-1], (224, 224)).astype(np.float32)
        curr = cv2.resize(frames_gray[i], (224, 224)).astype(np.float32)
        residual = np.abs(curr - prev)
        residuals.append(float(residual.mean()))
        h, w = residual.shape
        cy, cx = h // 2, w // 2
        h4, w4 = h // 4, w // 4
        spatial_vars.append(float(residual[cy-h4:cy+h4, cx-w4:cx+w4].var()))
    return np.array([float(np.mean(residuals)), float(np.mean(spatial_vars))], dtype=np.float32)


def compute_pillar8(frames_gray):
    if len(frames_gray) < 2:
        img = frames_gray[0].astype(np.float32)
        gx = cv2.Sobel(img, cv2.CV_64F, 1, 0)
        gy = cv2.Sobel(img, cv2.CV_64F, 0, 1)
        magnitude = float(np.sqrt(gx**2 + gy**2).mean())
        angles = np.arctan2(np.abs(gy) + 1e-6, np.abs(gx) + 1e-6)
        direction_consistency = float(1.0 / (angles.std() + 1e-6))
        return np.array([magnitude, min(direction_consistency, 10.0)], dtype=np.float32)
    magnitudes = []
    directions = []
    for i in range(1, len(frames_gray)):
        prev = frames_gray[i-1].astype(np.float32)
        curr = frames_gray[i].astype(np.float32)
        diff = curr - prev
        gx = cv2.Sobel(diff, cv2.CV_64F, 1, 0)
        gy = cv2.Sobel(diff, cv2.CV_64F, 0, 1)
        magnitudes.append(float(np.sqrt(gx**2 + gy**2).mean()))
        directions.append(float(np.arctan2(np.abs(gy).mean(), np.abs(gx).mean() + 1e-6)))
    blur_mag = float(np.mean(magnitudes))
    direction_consistency = float(1.0 / (np.std(directions) + 1e-6))
    return np.array([blur_mag, min(direction_consistency, 100.0)], dtype=np.float32)


def compute_pillar9(frames_gray):
    if len(frames_gray) < 2:
        return np.array([0.0, 0.0], dtype=np.float32)
    flow_mags = []
    boundary_discs = []
    for i in range(1, min(len(frames_gray), 4)):
        prev = frames_gray[i-1]
        curr = frames_gray[i]
        try:
            flow = cv2.calcOpticalFlowFarneback(
                prev, curr, None, pyr_scale=0.5, levels=2, winsize=10,
                iterations=2, poly_n=5, poly_sigma=1.1, flags=0)
            mag = float(np.sqrt(flow[..., 0]**2 + flow[..., 1]**2).mean())
            flow_mags.append(mag)
            flow_mag_map = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
            flow_grad = cv2.Laplacian(flow_mag_map.astype(np.float32), cv2.CV_64F)
            boundary_discs.append(float(np.abs(flow_grad).mean()))
        except Exception:
            flow_mags.append(0.0)
            boundary_discs.append(0.0)
    return np.array([float(np.mean(flow_mags)), float(np.mean(boundary_discs))], dtype=np.float32)


def compute_pillar10(img_rgb):
    img = img_rgb.astype(np.float32)
    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]
    h, w = g.shape
    def channel_shift(ch1, ch2):
        try:
            f1 = np.fft.fft2(ch1)
            f2 = np.fft.fft2(ch2)
            cross_power = f1 * np.conj(f2)
            norm = np.abs(cross_power) + 1e-10
            correlation = np.abs(np.fft.ifft2(cross_power / norm))
            peak = np.unravel_index(correlation.argmax(), correlation.shape)
            dy = peak[0] if peak[0] < h//2 else peak[0] - h
            dx = peak[1] if peak[1] < w//2 else peak[1] - w
            return float(np.sqrt(dy**2 + dx**2))
        except Exception:
            return 0.0
    r_g_shift = channel_shift(r, g)
    b_g_shift = channel_shift(b, g)
    edge_margin = max(h // 8, 1)
    center_r_g = np.abs(r[edge_margin:-edge_margin, edge_margin:-edge_margin] -
                        g[edge_margin:-edge_margin, edge_margin:-edge_margin]).mean()
    edge_r_g = np.abs(r - g).mean()
    edge_center_ratio = float(edge_r_g / (center_r_g + 1e-6))
    return np.array([r_g_shift, b_g_shift, edge_center_ratio], dtype=np.float32)


def extract_features_from_frame(frame_bgr):
    """Extract 30-dim feature vector from a single BGR frame."""
    try:
        frame_rgb  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        frame_rgb  = cv2.resize(frame_rgb,  (224, 224))
        frame_gray = cv2.resize(frame_gray, (224, 224))
        frame_bgr_r = cv2.resize(frame_bgr, (224, 224))
        p1  = compute_pillar1(frame_gray)
        p2  = compute_pillar2(frame_gray)
        p3  = compute_pillar3(frame_rgb)
        p4  = compute_pillar4(frame_rgb)
        p5  = compute_pillar5_single(frame_gray)
        p6  = compute_pillar6(frame_gray)
        p7  = np.array([0.0, 0.0], dtype=np.float32)
        p8  = compute_pillar8([frame_gray])
        p9  = np.array([0.0, 0.0], dtype=np.float32)
        p10 = compute_pillar10(frame_rgb)
        p11 = compute_pillar11_eye_symmetry([frame_bgr_r])
        p12 = compute_pillar12_illumination(frame_bgr_r)
        feat = np.concatenate([p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12])
        if not np.all(np.isfinite(feat)):
            feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
        return feat.astype(np.float32)
    except Exception:
        return None


def extract_features_from_video(video_path, n_frames=16):
    """
    Extract 30-dim features from video using imageio+ffmpeg.
    n_frames default changed 8->16 from V2.
    Returns list of 30-dim feature vectors.
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
        target_indices = set([int(x) for x in np.linspace(start, end - 1, n_frames)])
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
        p5_temporal  = compute_pillar5_temporal(frames_gray)
        p7_temporal  = compute_pillar7(frames_gray)
        p8_temporal  = compute_pillar8(frames_gray)
        p9_temporal  = compute_pillar9(frames_gray)
        p11_temporal = compute_pillar11_eye_symmetry(frames_bgr)  # NEW temporal
        features = []
        for frame_bgr, frame_gray in zip(frames_bgr, frames_gray):
            try:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                p1  = compute_pillar1(frame_gray)
                p2  = compute_pillar2(frame_gray)
                p3  = compute_pillar3(frame_rgb)
                p4  = compute_pillar4(frame_rgb)
                p6  = compute_pillar6(frame_gray)
                p10 = compute_pillar10(frame_rgb)
                p12 = compute_pillar12_illumination(frame_bgr)  # NEW per-frame
                feat = np.concatenate([
                    p1,           # dims 0-2
                    p2,           # dims 3-4
                    p3,           # dims 5-6
                    p4,           # dims 7-9
                    p5_temporal,  # dims 10-11
                    p6,           # dims 12-14
                    p7_temporal,  # dims 15-16
                    p8_temporal,  # dims 17-18
                    p9_temporal,  # dims 19-20
                    p10,          # dims 21-23
                    p11_temporal, # dims 24-26  NEW
                    p12,          # dims 27-29  NEW
                ])
                if not np.all(np.isfinite(feat)):
                    feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
                assert feat.shape == (30,), f"Expected 30 dims, got {feat.shape}"
                features.append(feat.astype(np.float32))
            except Exception:
                continue
        return features
    except Exception:
        return []


def load_or_create_pkl(pkl_path):
    """Load existing pkl or create fresh 30-dim one."""
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
        print(f"Loaded existing pkl: {n} samples | Real: {real} | Fake: {fake}")
        print(f"  Feature dims: {data['features'].shape[1]}")
    else:
        print("Creating fresh pkl (30-dim V3)")
        data = {
            'features':        np.empty((0, 30), dtype=np.float32),
            'labels':          [],
            'video_ids':       [],
            'dataset_sources': [],
            'generator_types': [],
        }
    return data


def save_pkl(data, pkl_path):
    os.makedirs(os.path.dirname(pkl_path) if os.path.dirname(pkl_path) else '.', exist_ok=True)
    n    = len(data['labels'])
    real = sum(1 for l in data['labels'] if l == 0)
    fake = sum(1 for l in data['labels'] if l == 1)
    with open(pkl_path, 'wb') as f:
        pickle.dump(data, f, protocol=4)
    print(f"Saved pkl: {n} total | {real} real | {fake} fake")


def append_to_pkl(data, new_features, new_labels, new_video_ids, new_sources, new_generators):
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