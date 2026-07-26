#!/usr/bin/env python3
"""Track E4 — multi-scale Laplacian-of-Gaussian frequency descriptors (per video). DEV; sealed untouched.
Physical basis: blending seams / generator textures leave scale-localised frequency signatures. Build a
LoG pyramid (sigmas 1,2,4,8) over the FACE region; per scale: energy, entropy, kurtosis, and the
FACE/BACKGROUND energy ratio (dimensionless -> the transfer hope). Reuses precompute frame+landmark
helpers. Keyed by FULL video_path. max_frames=60 (consistent with the E3 extraction). Deterministic.
"""
import argparse, csv, os, sys, warnings
import numpy as np, cv2
from scipy.ndimage import gaussian_laplace
from scipy.stats import kurtosis
from concurrent.futures import ProcessPoolExecutor, as_completed
warnings.filterwarnings("ignore"); sys.path.insert(0, os.path.dirname(__file__))
import precompute_features_best as P
SIGMAS=[1,2,4,8]
E4_FEATURES=([f"e4_energy_s{s}" for s in SIGMAS]+[f"e4_entropy_s{s}" for s in SIGMAS]+
             [f"e4_kurt_s{s}" for s in SIGMAS]+[f"e4_facebg_ratio_s{s}" for s in SIGMAS]+
             ["e4_energy_slope","e4_hf_lf_ratio"])   # 4*4 + 2 = 18
EPS=1e-8
def _stats(vals):
    v=vals[np.isfinite(vals)]
    if len(v)<10: return 0.0,0.0,0.0
    energy=float(np.mean(v**2))
    h,_=np.histogram(v,bins=32,density=True); h=h[h>0]; ent=float(-(h*np.log(h)).sum()) if len(h) else 0.0
    kt=float(kurtosis(v)) if len(v)>3 else 0.0
    return energy,ent,kt

def process_video(video_path,label,max_frames=60):
    try: frames,fps=P.load_video_frames(video_path,max_frames=max_frames)
    except Exception: return None
    if frames is None or len(frames)<10: return None
    fm=P.init_face_mesh()
    acc={k:[] for k in E4_FEATURES}
    for f in frames:
        rgb=cv2.cvtColor(f,cv2.COLOR_BGR2RGB); lm=P.get_landmarks(fm,rgb)
        if lm is None: continue
        L=(0.2126*rgb[...,0]+0.7152*rgb[...,1]+0.0722*rgb[...,2]).astype(np.float32)
        face=P.landmarks_to_mask(lm,P.FACE_OVAL,L.shape)>0
        if face.sum()<200: continue
        ys,xs=np.where(face); pad=20
        y0,y1=max(ys.min()-pad,0),min(ys.max()+pad,L.shape[0]-1); x0,x1=max(xs.min()-pad,0),min(xs.max()+pad,L.shape[1]-1)
        bg=np.ones_like(face); bg[y0:y1+1,x0:x1+1]=False
        e_scales=[]
        for s in SIGMAS:
            log=gaussian_laplace(L,sigma=s)
            fe,fen,fk=_stats(log[face])
            acc[f"e4_energy_s{s}"].append(fe); acc[f"e4_entropy_s{s}"].append(fen); acc[f"e4_kurt_s{s}"].append(fk)
            be,_,_=_stats(log[bg]) if bg.sum()>500 else (fe,0,0)
            acc[f"e4_facebg_ratio_s{s}"].append(fe/(be+EPS))
            e_scales.append(fe)
        # cross-scale: energy slope (log-log) and HF/LF ratio (s=1 vs s=8)
        es=np.array(e_scales)+EPS
        acc["e4_energy_slope"].append(float(np.polyfit(np.log(SIGMAS),np.log(es),1)[0]))
        acc["e4_hf_lf_ratio"].append(float(es[0]/(es[-1]+EPS)))
    fm.close()
    if len(acc["e4_energy_s1"])<5: return None
    row={"video_path":str(video_path),"label":int(label)}
    for k in E4_FEATURES:
        v=np.array(acc[k],float); row[k]=float(np.mean(v[np.isfinite(v)])) if np.isfinite(v).any() else 0.0
    return row

def _w(a): return process_video(*a)
if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--manifest",required=True); ap.add_argument("--output",required=True)
    ap.add_argument("--max_frames",type=int,default=60); ap.add_argument("--workers",type=int,default=16)
    a=ap.parse_args()
    import pandas as pd
    man=pd.read_csv(a.manifest); tasks=[(r.video_path,int(r.label),a.max_frames) for r in man.itertuples()]
    print(f"E4 LoG: {len(tasks)} videos -> {a.output}",flush=True)
    out=open(a.output,"w",newline=""); w=csv.DictWriter(out,fieldnames=["video_path","label"]+E4_FEATURES); w.writeheader(); ok=fail=0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs={ex.submit(_w,t):t for t in tasks}
        for fut in as_completed(futs):
            r=fut.result()
            if r: w.writerow(r); out.flush(); ok+=1
            else: fail+=1
            if (ok+fail)%200==0: print(f"  {ok+fail}/{len(tasks)}",flush=True)
    out.close(); print(f"Done. ok={ok} fail={fail} -> {a.output}",flush=True)
