#!/usr/bin/env python3
"""Test-Time Augmentation feature extraction for celebdf_dev. DEV target only; sealed untouched.
For each celebdf_dev video, produce N augmented versions (seeded, deterministic) and compute 196-D full_features
on each. Augmentations mimic capture/codec nuisance the domain gap partly rides on: JPEG re-encode (CRF proxy),
+-10% scale, brightness/contrast jitter, mild blur. At EVAL time the FF++-trained model scores each version and
we average predicted probabilities per video (TTA). No target labels used in training. Reuses full_features.
Usage: extract_trackE_TTA.py --manifest features/trackD/manifest_celebdf_dev.csv --output features/trackE/tta_celebdf_dev.csv --n_aug 3 [--max_frames 60] [--workers 12]
"""
import argparse, csv, os, sys, warnings
import numpy as np, cv2
from concurrent.futures import ProcessPoolExecutor, as_completed
warnings.filterwarnings("ignore"); sys.path.insert(0, os.path.dirname(__file__))
import precompute_features_best as P
from extract_trackE_SBV import full_features, FEATS
def augment(img, rng):
    out=img.copy()
    if rng.rand()<0.75:                                   # +-10% scale (down-up)
        s=rng.uniform(0.9,1.1); h,w=out.shape[:2]
        out=cv2.resize(out,(max(int(w*s),8),max(int(h*s),8))); out=cv2.resize(out,(w,h))
    if rng.rand()<0.75:                                   # brightness / contrast
        out=cv2.convertScaleAbs(out,alpha=rng.uniform(0.9,1.1),beta=rng.uniform(-12,12))
    if rng.rand()<0.75:                                   # JPEG re-encode (CRF proxy)
        q=int(rng.uniform(45,92)); ok,enc=cv2.imencode(".jpg",out,[int(cv2.IMWRITE_JPEG_QUALITY),q])
        if ok: out=cv2.imdecode(enc,cv2.IMREAD_COLOR)
    if rng.rand()<0.5:                                    # mild blur
        k=int(rng.choice([3,3,5])); out=cv2.GaussianBlur(out,(k,k),0)
    return out
def _one(args):
    vp,lab,aug_idx,mf=args
    try: frames,fps=P.load_video_frames(vp,max_frames=mf)
    except Exception: return None
    if frames is None or len(frames)<20: return None
    rng=np.random.RandomState((abs(hash(str(vp)))%100000)*7 + aug_idx*13 + 42)
    aug=[augment(f,rng) for f in frames]
    try: feat=full_features(aug,fps,sbvgen=None)
    except Exception: return None
    if not feat: return None
    return (str(vp),int(lab),int(aug_idx),feat)
if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--manifest",required=True); ap.add_argument("--output",required=True)
    ap.add_argument("--n_aug",type=int,default=3); ap.add_argument("--max_frames",type=int,default=60); ap.add_argument("--workers",type=int,default=12)
    a=ap.parse_args()
    import pandas as pd
    man=pd.read_csv(a.manifest)
    tasks=[(r.video_path,int(getattr(r,"label",1)),ai,a.max_frames) for r in man.itertuples() for ai in range(a.n_aug)]
    print(f"TTA: {len(man)} videos x {a.n_aug} aug = {len(tasks)} tasks -> {a.output}",flush=True)
    out=open(a.output,"w",newline=""); w=csv.DictWriter(out,fieldnames=["video_path","label","aug_idx"]+FEATS); w.writeheader(); ok=fail=0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs={ex.submit(_one,t):t for t in tasks}
        for fut in as_completed(futs):
            r=fut.result()
            if r:
                vp,lab,ai,feat=r; row={"video_path":vp,"label":lab,"aug_idx":ai}; row.update({k:feat.get(k,0.0) for k in FEATS})
                w.writerow(row); out.flush(); ok+=1
            else: fail+=1
            if (ok+fail)%300==0: print(f"  {ok+fail}/{len(tasks)}",flush=True)
    out.close(); print(f"Done. ok={ok} fail={fail} -> {a.output}",flush=True)
