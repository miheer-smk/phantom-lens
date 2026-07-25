#!/usr/bin/env python3
"""Track E — persist PER-FRAME spatial feature values (for E1 order-statistics + E2 windows).
Reuses precompute_features_best.extract_spatial_features_single_frame (the exact frozen per-frame
spatial extractor). Output: long CSV (video_path,label,frame + 13 spatial values). Deterministic.
One pass; E1 and E2 both derive from this — no re-extraction. Denser sampling (max_frames=60).
"""
import argparse, csv, os, sys, warnings
import numpy as np, cv2
from concurrent.futures import ProcessPoolExecutor, as_completed
warnings.filterwarnings("ignore"); sys.path.insert(0, os.path.dirname(__file__))
import precompute_features_best as P

# order returned by extract_spatial_features_single_frame
SPATIAL13=["s_noise_vmr","s_noise_res_std","s_noise_hf_ratio","s_prnu_energy","s_prnu_face_periph",
           "s_shadow_score","s_face_bg_diff","s_benford_dev","s_block_artifact","s_dbl_compress",
           "s_blur_mag","s_flow_mag","s_flow_dir_consist"]

def process_video(video_path, label, max_frames=60):
    try: frames,fps=P.load_video_frames(video_path,max_frames=max_frames)
    except Exception: return None
    if frames is None or len(frames)<10: return None
    fm=P.init_face_mesh(); rows=[]; prev=None
    for fi,f in enumerate(frames):
        rgb=cv2.cvtColor(f,cv2.COLOR_BGR2RGB); gray=cv2.cvtColor(f,cv2.COLOR_BGR2GRAY)
        lm=P.get_landmarks(fm,rgb)
        if lm is None: prev=gray; continue
        mask=P.landmarks_to_mask(lm,P.FACE_OVAL,gray.shape)
        if (mask>0).sum()<200: prev=gray; continue
        try:
            v=P.extract_spatial_features_single_frame(f,gray,prev,lm,mask)
        except Exception:
            prev=gray; continue
        v=[0.0 if (x is None or not np.isfinite(x)) else float(x) for x in v]
        rows.append([str(video_path),int(label),len(rows)]+v)   # FULL path (FF++ manips share basenames!)
        prev=gray
    fm.close()
    return rows if len(rows)>=5 else None

def _w(a): return process_video(*a)
if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--manifest",required=True); ap.add_argument("--output",required=True)
    ap.add_argument("--max_frames",type=int,default=60); ap.add_argument("--workers",type=int,default=16)
    a=ap.parse_args()
    import pandas as pd
    man=pd.read_csv(a.manifest); tasks=[(r.video_path,int(r.label),a.max_frames) for r in man.itertuples()]
    print(f"per-frame spatial: {len(tasks)} videos -> {a.output}",flush=True)
    out=open(a.output,"w",newline=""); w=csv.writer(out); w.writerow(["video_path","label","frame"]+SPATIAL13)
    ok=fail=nframes=0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs={ex.submit(_w,t):t for t in tasks}
        for fut in as_completed(futs):
            r=fut.result()
            if r: w.writerows(r); out.flush(); ok+=1; nframes+=len(r)
            else: fail+=1
            if (ok+fail)%200==0: print(f"  {ok+fail}/{len(tasks)} ({nframes} frames)",flush=True)
    out.close(); print(f"Done. ok={ok} fail={fail} frames={nframes} -> {a.output}",flush=True)
