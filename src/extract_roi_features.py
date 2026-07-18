#!/usr/bin/env python3
"""Track C — region-localized candidate features for Face2Face / NeuralTextures.
Reuses the frozen extractor's frame-loading + MediaPipe landmarks. Outputs one row per video:
video_path,label + the 8 ROI candidate features (roi_config.ROI_FEATURE_NAMES).
Deterministic; no RNG. Merge on video_path with the 50-feature CSVs for evaluation.
"""
import argparse, csv, os, sys, warnings
import numpy as np, cv2
from concurrent.futures import ProcessPoolExecutor, as_completed
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
import precompute_features_best as P   # frozen extractor: load_video_frames, init_face_mesh, get_landmarks, landmarks_to_mask
import roi_config as RC

def _roi_mask(landmarks, indices, shape):
    return P.landmarks_to_mask(landmarks, indices, shape)

def _dct_midband_energy(gray_roi):
    """Mid-frequency DCT energy (normalized) inside an ROI patch."""
    h, w = gray_roi.shape
    if h < 8 or w < 8: return 0.0
    g = cv2.resize(gray_roi.astype(np.float32), (32, 32))
    D = cv2.dct(g)
    A = np.abs(D); tot = A.sum() + 1e-8
    mid = A[4:16, 4:16].sum()          # mid band
    return float(mid / tot)

def _hf_residual_energy(gray_roi):
    if gray_roi.size < 64: return 0.0
    blur = cv2.GaussianBlur(gray_roi.astype(np.float32), (3, 3), 0)
    res = gray_roi.astype(np.float32) - blur
    return float(np.mean(res ** 2))

def process_video(video_path, label, max_frames=150):
    try:
        frames_bgr, fps = P.load_video_frames(video_path, max_frames=max_frames)
    except Exception:
        return None
    if frames_bgr is None or len(frames_bgr) < 10:
        return None
    fm = P.init_face_mesh()
    grays, lms = [], []
    for f in frames_bgr:
        rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        lm = P.get_landmarks(fm, rgb)
        grays.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
        lms.append(lm)
    fm.close()
    valid = [i for i, l in enumerate(lms) if l is not None]
    if len(valid) < 10:
        return None

    # ---- G1: mouth-interior temporal texture instability ----
    dct_series, hf_series, tex_patches = [], [], []
    for i in valid:
        g, lm = grays[i], lms[i]
        m = _roi_mask(lm, RC.MOUTH_REGION, g.shape)
        ys, xs = np.where(m > 0)
        if len(xs) < 30:
            continue
        y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
        patch = g[y0:y1+1, x0:x1+1]
        dct_series.append(_dct_midband_energy(patch))
        hf_series.append(_hf_residual_energy(patch))
        tex_patches.append(cv2.resize(patch.astype(np.float32), (24, 24)).ravel())
    if len(dct_series) < 5:
        return None
    g1_dct_std = float(np.std(dct_series))
    g1_hf = float(np.mean(hf_series))
    # texture flicker = 1 - mean frame-to-frame correlation of mouth patch
    corrs = []
    for a, b in zip(tex_patches[:-1], tex_patches[1:]):
        if a.std() > 1e-6 and b.std() > 1e-6:
            corrs.append(np.corrcoef(a, b)[0, 1])
    g1_flicker = float(1.0 - np.mean(corrs)) if corrs else 0.0

    # ---- G2: inner-vs-outer-face boundary discontinuity ----
    grad_disc, tex_mm, col_mm = [], [], []
    for i in valid:
        g, lm = grays[i], lms[i]
        inner = _roi_mask(lm, RC.INNER_FACE, g.shape)
        k = np.ones((9, 9), np.uint8)
        inner_d = cv2.dilate(inner, k); inner_e = cv2.erode(inner, k)
        ring = (inner_d > 0) & (inner_e == 0)          # boundary band
        just_in = (inner_e > 0); just_out = (inner_d == 0)
        gx = cv2.Sobel(g.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
        gm = np.sqrt(gx**2 + gy**2)
        if ring.sum() > 20:
            grad_disc.append(float(gm[ring].mean()))
        if just_in.sum() > 30 and just_out.sum() > 30:
            tex_mm.append(abs(float(g[just_in].std()) - float(g[just_out].std())))
            col_mm.append(abs(float(g[just_in].mean()) - float(g[just_out].mean())))
    g2_grad = float(np.mean(grad_disc)) if grad_disc else 0.0
    g2_tex = float(np.mean(tex_mm)) if tex_mm else 0.0
    g2_col = float(np.mean(col_mm)) if col_mm else 0.0

    # ---- G3: mouth landmark-motion <-> local-texture coupling ----
    mv, tv = [], []   # mouth landmark velocity, mouth texture-residual change
    prev_c = None; prev_patch = None
    for i in valid:
        g, lm = grays[i], lms[i]
        pts = lm[RC.MOUTH_REGION]
        c = pts.mean(0)
        m = _roi_mask(lm, RC.MOUTH_REGION, g.shape)
        ys, xs = np.where(m > 0)
        if len(xs) < 30:
            prev_c = c; prev_patch = None; continue
        patch = cv2.resize(g[ys.min():ys.max()+1, xs.min():xs.max()+1].astype(np.float32), (24, 24))
        if prev_c is not None and prev_patch is not None:
            mv.append(float(np.linalg.norm(c - prev_c)))
            tv.append(float(np.mean(np.abs(patch - prev_patch))))
        prev_c = c; prev_patch = patch
    if len(mv) >= 5 and np.std(mv) > 1e-6 and np.std(tv) > 1e-6:
        g3_corr = float(np.corrcoef(mv, tv)[0, 1])
        # lag-1 cross-correlation (coupling can be delayed)
        if len(mv) > 6:
            g3_lag = float(np.corrcoef(mv[:-1], tv[1:])[0, 1])
        else:
            g3_lag = g3_corr
    else:
        g3_corr = 0.0; g3_lag = 0.0

    vals = [g1_dct_std, g1_hf, g1_flicker, g2_grad, g2_tex, g2_col, g3_corr, g3_lag]
    vals = [0.0 if (v is None or np.isnan(v) or np.isinf(v)) else float(v) for v in vals]
    row = {"video_path": video_path, "label": label}
    row.update(dict(zip(RC.ROI_FEATURE_NAMES, vals)))
    return row

def _worker(a): return process_video(*a)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_dir", required=True); ap.add_argument("--output", required=True)
    ap.add_argument("--label", type=int, required=True); ap.add_argument("--max_frames", type=int, default=150)
    ap.add_argument("--workers", type=int, default=8); ap.add_argument("--append", action="store_true")
    a = ap.parse_args()
    vids = P.discover_videos(a.video_dir, a.label)
    print(f"Found {len(vids)} videos in {a.video_dir} (label={a.label})", flush=True)
    header = ["video_path", "label"] + RC.ROI_FEATURE_NAMES
    mode = "a" if a.append else "w"
    write_hdr = not a.append or not os.path.exists(a.output)
    out = open(a.output, mode, newline=""); w = csv.DictWriter(out, fieldnames=header)
    if write_hdr: w.writeheader()
    tasks = [(v, l, a.max_frames) for v, l in vids]; ok = fail = 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(_worker, t): t for t in tasks}
        for fut in as_completed(futs):
            r = fut.result()
            if r: w.writerow(r); out.flush(); ok += 1
            else: fail += 1
    out.close()
    print(f"Done. ok={ok} fail={fail} -> {a.output}", flush=True)

if __name__ == "__main__":
    main()
