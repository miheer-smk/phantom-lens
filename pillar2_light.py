# Phantom Lens Research Framework
# Developer: Miheer Satish Kulkarni | IIIT Nagpur | 2026
# All Rights Reserved



import numpy as np
import cv2

def estimate_light_from_shadows(face_region_gray):
    grad_x = cv2.Sobel(face_region_gray, cv2.CV_64F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(face_region_gray, cv2.CV_64F, 0, 1, ksize=5)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    magnitude_sum = magnitude.sum() + 1e-10
    mean_gx = (grad_x * magnitude).sum() / magnitude_sum
    mean_gy = (grad_y * magnitude).sum() / magnitude_sum
    azimuth = np.degrees(np.arctan2(mean_gy, mean_gx))
    return np.array([azimuth, 45.0])

def compute_pillar2_score(frames_gray):
    shadow_directions = []
    for frame in frames_gray:
        shadow_dir = estimate_light_from_shadows(frame)
        shadow_directions.append(shadow_dir)
    shadow_arr = np.array(shadow_directions)
    temporal_variance = float(np.var(shadow_arr[:, 0]))
    f2_score = temporal_variance / 1000.0
    return {
        'f2_raw': f2_score,
        'temporal_variance': temporal_variance
    }