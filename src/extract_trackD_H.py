#!/usr/bin/env python3
"""Track D — Group H: Gradient Structure Tensor features (per video).
Physical basis: real sensor noise is isotropic high-frequency -> characteristic luminance-gradient
anisotropy/coherence; GAN/rendered content is smoother / more locally coherent / more oriented.
Reuses the frozen extractor's frame-loading + MediaPipe landmarks (precompute_features_best).
Input: --manifest CSV (video_path,label). Output: one row/video = video_path,label + 10 H features.
Deterministic (no RNG). max_frames matches the pipeline default. Structure tensor computed on the
FACE_OVAL mask and on the background (outside a padded face box) for the domain-invariant ratio.
"""
import argparse, csv, os, sys, warnings
import numpy as np, cv2
from concurrent.futures import ProcessPoolExecutor, as_completed
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
import precompute_features_best as P

H_FEATURES = ["h_anisotropy", "h_coherence", "h_tensor_trace", "h_eig_ratio_log",
              "h_orientation_entropy", "h_face_bg_aniso_ratio",
              "h_anisotropy_tstd", "h_anisotropy_lag1",
              "h_orient_entropy_tstd", "h_orient_entropy_lag1"]
EPS = 1e-8

def _tensor_feats(gx, gy):
    """Structure-tensor eigen features over a set of gradient samples (gx,gy 1-D arrays)."""
    Jxx = float(np.mean(gx * gx)); Jyy = float(np.mean(gy * gy)); Jxy = float(np.mean(gx * gy))
    tr = Jxx + Jyy; det = Jxx * Jyy - Jxy * Jxy
    disc = np.sqrt(max(tr * tr - 4 * det, 0.0))
    l1 = (tr + disc) / 2.0; l2 = (tr - disc) / 2.0            # l1 >= l2 >= 0
    aniso = (l1 - l2) / (l1 + l2 + EPS)
    coher = l2 / (l1 + EPS)
    trace = l1 + l2
    eiglog = float(np.log((l1 + EPS) / (l2 + EPS)))
    return aniso, coher, trace, eiglog, l1, l2

def _orient_entropy(gx, gy, mag, nbins=18):
    theta = np.mod(np.arctan2(gy, gx), np.pi)                 # orientation in [0, pi)
    h, _ = np.histogram(theta, bins=nbins, range=(0, np.pi), weights=mag)
    p = h / (h.sum() + EPS)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())

def process_video(video_path, label, max_frames=150):
    try:
        frames_bgr, fps = P.load_video_frames(video_path, max_frames=max_frames)
    except Exception:
        return None
    if frames_bgr is None or len(frames_bgr) < 10:
        return None
    fm = P.init_face_mesh()
    aniso_s, coher_s, trace_s, eiglog_s, orient_s, ratio_s = [], [], [], [], [], []
    for f in frames_bgr:
        rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        lm = P.get_landmarks(fm, rgb)
        if lm is None:
            continue
        R, G, B = rgb[..., 0].astype(np.float32), rgb[..., 1].astype(np.float32), rgb[..., 2].astype(np.float32)
        L = 0.2126 * R + 0.7152 * G + 0.0722 * B
        Gx = cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3)
        Gy = cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3)
        Mag = np.sqrt(Gx * Gx + Gy * Gy)
        face = P.landmarks_to_mask(lm, P.FACE_OVAL, L.shape) > 0
        if face.sum() < 200:
            continue
        # background = outside a padded face bounding box
        ys, xs = np.where(face); pad = 20
        y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, L.shape[0] - 1)
        x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, L.shape[1] - 1)
        bg = np.ones_like(face); bg[y0:y1 + 1, x0:x1 + 1] = False
        a_f, c_f, t_f, e_f, l1, l2 = _tensor_feats(Gx[face], Gy[face])
        oe_f = _orient_entropy(Gx[face], Gy[face], Mag[face])
        aniso_s.append(a_f); coher_s.append(c_f); trace_s.append(t_f); eiglog_s.append(e_f); orient_s.append(oe_f)
        if bg.sum() > 500:
            a_bg, *_ = _tensor_feats(Gx[bg], Gy[bg])
            ratio_s.append(a_f / (a_bg + EPS))
    fm.close()
    if len(aniso_s) < 5:
        return None
    def lag1(x):
        x = np.asarray(x)
        if len(x) < 6 or x.std() < 1e-9:
            return 0.0
        return float(np.corrcoef(x[:-1], x[1:])[0, 1])
    vals = [float(np.mean(aniso_s)), float(np.mean(coher_s)), float(np.mean(trace_s)),
            float(np.mean(eiglog_s)), float(np.mean(orient_s)),
            float(np.mean(ratio_s)) if ratio_s else 1.0,
            float(np.std(aniso_s)), lag1(aniso_s),
            float(np.std(orient_s)), lag1(orient_s)]
    vals = [0.0 if (v is None or np.isnan(v) or np.isinf(v)) else float(v) for v in vals]
    row = {"video_path": video_path, "label": int(label)}
    row.update(dict(zip(H_FEATURES, vals)))
    return row

def _worker(a): return process_video(*a)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="CSV with columns video_path,label")
    ap.add_argument("--output", required=True)
    ap.add_argument("--max_frames", type=int, default=150)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    import pandas as pd
    man = pd.read_csv(a.manifest)
    tasks = [(r.video_path, int(r.label), a.max_frames) for r in man.itertuples()]
    print(f"Group H: {len(tasks)} videos -> {a.output}", flush=True)
    header = ["video_path", "label"] + H_FEATURES
    out = open(a.output, "w", newline=""); w = csv.DictWriter(out, fieldnames=header); w.writeheader()
    ok = fail = 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(_worker, t): t for t in tasks}
        for fut in as_completed(futs):
            r = fut.result()
            if r: w.writerow(r); out.flush(); ok += 1
            else: fail += 1
            if (ok + fail) % 200 == 0: print(f"  {ok+fail}/{len(tasks)} (ok={ok} fail={fail})", flush=True)
    out.close()
    print(f"Done. ok={ok} fail={fail} -> {a.output}", flush=True)

if __name__ == "__main__":
    main()
