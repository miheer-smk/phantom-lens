#!/usr/bin/env python3
"""Track E3 — SBV feature extraction + PRE-FLIGHT sanity gate.
full_features(frames): compute the 196-D E1-expanded vector (13 spatial means + 37 temporal + 3 G1
+ 143 spatial order-statistics) on a frame list, reusing precompute_features_best. With sbvgen it
first self-blends each frame (sbv_generator). Keyed by FULL video_path (FF++ manips share basenames).

Pre-flight (`--preflight N`): on N real videos + their SBV, Cohen's d per feature real-vs-SBV; if the
boundary/blend features (T8) show |d|>0.5 the blend is visible to the physics descriptors and full
extraction is justified; else tune mask/strength/jitter first. Full mode extracts SBV features for a
manifest (label=1) at given --temporal_jitter.
"""
import argparse, csv, os, sys, warnings
import numpy as np, cv2
from scipy.stats import skew, kurtosis
warnings.filterwarnings("ignore"); sys.path.insert(0, os.path.dirname(__file__))
import precompute_features_best as P
import roi_config as RC
from sbv_generator import SBVGenerator
from extract_trackE_perframe import SPATIAL13
STATS=["mean","std","min","max","p10","p25","p75","p90","iqr","skew","kurt"]
AGG=[f"{f}__{s}" for f in SPATIAL13 for s in STATS]
G1=RC.CANDIDATE_GROUPS["G1_mouth_instability"]
TEMP=P.FEATURE_NAMES_TEMPORAL
FEATS=list(SPATIAL13)+list(TEMP)+list(G1)+AGG    # 13+37+3+143 = 196
BOUNDARY=["t_boundary_grad_temporal","t_boundary_color_disc","t_boundary_freq_leakage",
          "t_skin_bg_decorrelation","t_texture_warp_residual","t_face_ssim_mean","t_face_ssim_min",
          "s_face_bg_diff","s_prnu_face_periph","t_prnu_face_vs_bg"]   # boundary/blend-sensitive

def _ordstats(arr):  # arr: (F,13) -> 143 order stats
    out={}
    for j,f in enumerate(SPATIAL13):
        x=arr[:,j]
        out[f"{f}__mean"]=np.mean(x); out[f"{f}__std"]=np.std(x); out[f"{f}__min"]=np.min(x); out[f"{f}__max"]=np.max(x)
        q=np.percentile(x,[10,25,75,90]); out[f"{f}__p10"],out[f"{f}__p25"],out[f"{f}__p75"],out[f"{f}__p90"]=q
        out[f"{f}__iqr"]=q[2]-q[1]; out[f"{f}__skew"]=float(skew(x)) if len(x)>2 else 0.0; out[f"{f}__kurt"]=float(kurtosis(x)) if len(x)>3 else 0.0
    return out
def _g1(grays,lms,valid):
    def dctmid(p):
        h,w=p.shape
        if h<8 or w<8: return 0.0
        g=cv2.resize(p.astype(np.float32),(32,32)); D=np.abs(cv2.dct(g)); return float(D[4:16,4:16].sum()/(D.sum()+1e-8))
    def hfres(p):
        if p.size<64: return 0.0
        return float(np.mean((p.astype(np.float32)-cv2.GaussianBlur(p.astype(np.float32),(3,3),0))**2))
    dct=[];hf=[];tex=[]
    for i in valid:
        m=P.landmarks_to_mask(lms[i],RC.MOUTH_REGION,grays[i].shape); ys,xs=np.where(m>0)
        if len(xs)<30: continue
        patch=grays[i][ys.min():ys.max()+1,xs.min():xs.max()+1]
        dct.append(dctmid(patch)); hf.append(hfres(patch)); tex.append(cv2.resize(patch.astype(np.float32),(24,24)).ravel())
    if len(dct)<5: return {g:0.0 for g in G1}
    corrs=[np.corrcoef(a,b)[0,1] for a,b in zip(tex[:-1],tex[1:]) if a.std()>1e-6 and b.std()>1e-6]
    return {G1[0]:float(np.std(dct)),G1[1]:float(np.mean(hf)),G1[2]:float(1-np.mean(corrs)) if corrs else 0.0}

def full_features(frames_bgr, fps, sbvgen=None):
    if sbvgen is not None:
        fm0=P.init_face_mesh(); blended=[]
        for f in frames_bgr:
            lm=P.get_landmarks(fm0,cv2.cvtColor(f,cv2.COLOR_BGR2RGB)); b,_=sbvgen.frame(f,lm); blended.append(b)
        fm0.close(); frames_bgr=blended
    fm=P.init_face_mesh()
    rgb=[cv2.cvtColor(f,cv2.COLOR_BGR2RGB) for f in frames_bgr]; gray=[cv2.cvtColor(f,cv2.COLOR_BGR2GRAY) for f in frames_bgr]
    lms=[P.get_landmarks(fm,r) for r in rgb]; fm.close()
    valid=[i for i,l in enumerate(lms) if l is not None]
    if len(valid)<10: return None
    fmask=[(P.landmarks_to_mask(lms[i],P.FACE_OVAL,gray[i].shape) if lms[i] is not None else np.zeros_like(gray[i])) for i in range(len(gray))]
    bmask=[(1-m).astype(np.uint8) for m in fmask]
    # per-frame spatial
    pf=[]; prev=None
    for i in range(len(gray)):
        if lms[i] is None: prev=gray[i]; continue
        v=P.extract_spatial_features_single_frame(frames_bgr[i],gray[i],prev,lms[i],fmask[i]); prev=gray[i]
        pf.append([0.0 if (x is None or not np.isfinite(x)) else float(x) for x in v])
    if len(pf)<5: return None
    pf=np.array(pf); out={f:float(pf[:,j].mean()) for j,f in enumerate(SPATIAL13)}
    out.update(_ordstats(pf))
    # temporal t1..t14 exactly as the pipeline
    t=[P.extract_temporal_noise_stability(gray,fmask),P.extract_rppg(rgb,lms,fps),P.extract_temporal_prnu(gray,fmask,bmask),
       P.extract_face_structural_stability(gray,lms),P.extract_codec_temporal_residual(gray,fmask),P.extract_landmark_trajectory(lms,fps),
       P.extract_rigid_geometry(lms),P.extract_boundary_coherence(gray,frames_bgr,lms),P.extract_skin_texture(gray,lms),
       P.extract_color_transfer(frames_bgr,lms,fmask),P.extract_specular_temporal(gray,lms),P.extract_blink_dynamics(lms,fps),
       P.extract_motion_blur_coupling(gray,fmask),P.extract_dct_stability(gray,lms)]
    tv=np.nan_to_num(np.concatenate([np.atleast_1d(x) for x in t]),nan=0.0)
    out.update(dict(zip(TEMP,tv)))
    out.update(_g1(gray,lms,valid))
    return {k:(0.0 if (out.get(k) is None or not np.isfinite(out.get(k,0))) else float(out[k])) for k in FEATS}

def cohend(a,b):
    a=np.asarray(a); b=np.asarray(b); sp=np.sqrt(((len(a)-1)*a.var()+(len(b)-1)*b.var())/max(len(a)+len(b)-2,1))
    return float((b.mean()-a.mean())/(sp+1e-9))

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--preflight",type=int,default=0); ap.add_argument("--manifest"); ap.add_argument("--output")
    ap.add_argument("--temporal_jitter",type=float,default=1.0); ap.add_argument("--max_frames",type=int,default=60); ap.add_argument("--seed",type=int,default=42)
    a=ap.parse_args()
    import pandas as pd
    if a.preflight:
        man=pd.read_csv("features/trackD/manifest_ffpp_trainval.csv")
        reals=man[man.label==0].sample(a.preflight,random_state=a.seed)
        R={f:[] for f in FEATS}; S={f:[] for f in FEATS}; n=0
        for r in reals.itertuples():
            try: frames,fps=P.load_video_frames(r.video_path,max_frames=a.max_frames)
            except Exception: continue
            if len(frames)<20: continue
            fr=full_features(frames,fps,sbvgen=None)
            fs=full_features(frames,fps,sbvgen=SBVGenerator(seed=a.seed,temporal_jitter=a.temporal_jitter))
            if fr and fs:
                for f in FEATS: R[f].append(fr[f]); S[f].append(fs[f])
                n+=1
                if n%10==0: print(f"  preflight {n}/{a.preflight}",flush=True)
        ds={f:round(cohend(R[f],S[f]),3) for f in FEATS}
        import json; json.dump({"n":n,"temporal_jitter":a.temporal_jitter,"cohens_d":ds,
            "boundary_features":{f:ds[f] for f in BOUNDARY}}, open("results_clean/trackE_SBV_preflight.json","w"),indent=1)
        print("="*66); print(f"E3 SBV PRE-FLIGHT (n={n} real vs SBV, jitter={a.temporal_jitter})"); print("="*66)
        print("BOUNDARY / blend-sensitive features |Cohen's d| (gate: >0.5):")
        for f in BOUNDARY: print(f"   {f:28s} d={ds[f]:+.3f}  {'VISIBLE' if abs(ds[f])>0.5 else '(weak)'}")
        top=sorted(FEATS,key=lambda f:-abs(ds[f]))[:10]
        print("top-10 |d| overall:", [(f,ds[f]) for f in top])
        mx=max(abs(ds[f]) for f in BOUNDARY)
        print(f"\nGATE: max|d| on boundary feats = {mx:.3f} -> {'PASS (justify full extraction)' if mx>0.5 else 'FAIL (tune blend before full run)'}")
    else:  # full extraction of SBV features for a manifest (real videos -> SBV, label=1)
        man=pd.read_csv(a.manifest); man=man[man.label==0]
        out=open(a.output,"w",newline=""); w=csv.DictWriter(out,fieldnames=["video_path","label"]+FEATS); w.writeheader(); ok=fail=0
        for r in man.itertuples():
            try: frames,fps=P.load_video_frames(r.video_path,max_frames=a.max_frames)
            except Exception: fail+=1; continue
            fs=full_features(frames,fps,sbvgen=SBVGenerator(seed=a.seed+hash(r.video_path)%1000,temporal_jitter=a.temporal_jitter)) if len(frames)>=20 else None
            if fs: row={"video_path":str(r.video_path)+f"__sbv_j{a.temporal_jitter}","label":1}; row.update(fs); w.writerow(row); out.flush(); ok+=1
            else: fail+=1
            if (ok+fail)%100==0: print(f"  {ok+fail} (ok={ok})",flush=True)
        out.close(); print(f"Done. ok={ok} fail={fail} -> {a.output}",flush=True)
