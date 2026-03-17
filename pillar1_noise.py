# Phantom Lens Research Framework
# Developer: Miheer Satish Kulkarni | IIIT Nagpur | 2026
# All Rights Reserved



import numpy as np
import cv2

def extract_noise_residual_fast(frame_float):
    """
    Fast noise estimation using Gaussian blur instead of BM3D.
    Much faster while still capturing the VMR signal.
    """
    frame_255 = (frame_float * 255).astype(np.float32)
    denoised = np.zeros_like(frame_255)
    for c in range(3):
        denoised[:, :, c] = cv2.GaussianBlur(frame_255[:, :, c], (5, 5), 0)
    residual = frame_255 - denoised
    return residual

def compute_pillar1_score(frame_float):
    noise = extract_noise_residual_fast(frame_float)
    luminance = 0.299*frame_float[:,:,0] + 0.587*frame_float[:,:,1] + 0.114*frame_float[:,:,2]
    luminance_flat = luminance.flatten()
    noise_flat = noise[:,:,1].flatten()
    n_bins = 10
    bin_edges = np.percentile(luminance_flat, np.linspace(0, 100, n_bins+1))
    vmr_values = []
    for i in range(n_bins):
        mask = (luminance_flat >= bin_edges[i]) & (luminance_flat < bin_edges[i+1])
        if mask.sum() < 50:
            continue
        bin_noise = noise_flat[mask]
        mean_noise = np.abs(np.mean(bin_noise))
        var_noise = np.var(bin_noise)
        if mean_noise > 1e-6:
            vmr = var_noise / mean_noise
            vmr_values.append(vmr)
    if len(vmr_values) < 3:
        return {'f1_raw': 0.5, 'vmr_std': 0.0}
    vmr_values = np.array(vmr_values)
    f1_score = float(np.mean(np.abs(vmr_values - 1.0)))
    return {
        'f1_raw': f1_score,
        'vmr_std': float(np.std(vmr_values))
    }