# Phantom Lens Research Framework
# Developer: Miheer Satish Kulkarni | IIIT Nagpur | 2026
# All Rights Reserved



import numpy as np
import cv2
from scipy.fft import dctn

def compute_block_dct(frame_gray_255):
    H, W = frame_gray_255.shape
    H = (H // 8) * 8
    W = (W // 8) * 8
    frame_trimmed = frame_gray_255[:H, :W].astype(np.float32)
    all_coeffs = []
    for i in range(0, H, 8):
        for j in range(0, W, 8):
            block = frame_trimmed[i:i+8, j:j+8]
            dct_block = dctn(block, norm='ortho')
            all_coeffs.extend(dct_block.flatten().tolist())
    return np.array(all_coeffs)

def benford_distribution():
    digits = np.arange(1, 10)
    return np.log10(1 + 1/digits)

def leading_digit(x):
    x = abs(x)
    if x < 1e-10:
        return None
    while x >= 10:
        x /= 10
    while x < 1:
        x *= 10
    return int(x)

def compute_pillar3_score(frame_float):
    frame_gray = cv2.cvtColor((frame_float*255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    dct_coeffs = compute_block_dct(frame_gray.astype(np.float32))
    digits = []
    for c in dct_coeffs:
        d = leading_digit(c)
        if d is not None and 1 <= d <= 9:
            digits.append(d)
    if len(digits) < 100:
        return {'f3_raw': 0.0, 'benford_deviation': 0.0}
    observed = np.zeros(9)
    for d in digits:
        observed[d-1] += 1
    observed = observed / observed.sum()
    expected = benford_distribution()
    chi2 = float(np.sum((observed - expected)**2 / (expected + 1e-10)))
    coeffs_mag = np.abs(dct_coeffs[np.abs(dct_coeffs) > 0.1])
    hist, _ = np.histogram(coeffs_mag, bins=256, range=(0, 100))
    hist_fft = np.abs(np.fft.rfft(hist.astype(np.float32)))
    dominant = np.max(hist_fft[2:64])
    baseline = np.median(hist_fft[2:])
    periodicity = float(dominant / (baseline + 1e-10))
    f3 = chi2 + (periodicity / 100.0)
    return {
        'f3_raw': f3,
        'benford_deviation': chi2
    }