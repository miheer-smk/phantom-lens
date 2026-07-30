#!/usr/bin/env python3
"""Extract 196-D full_features for WildDeepfake (SECOND cross-dataset validation target, winner's-curse fix).
WildDeepfake ships as flat face-crop PNGs in test/{real,fake}, named {videoid}_{frameidx}.png. Group by videoid,
sort by frameidx, load as a frame sequence, run the SAME full_features (196-D) as the video pipeline. fps nominal
30 (no source fps). This is an inductive held-out TARGET only: train on FF++, score AUC here — no training on it.
Usage: extract_wdf_196d.py --output features/trackE/wdf_196d.csv [--max_frames 60] [--workers 8]
"""
import argparse, csv, glob, os, sys, warnings
import numpy as np, cv2
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
warnings.filterwarnings("ignore"); sys.path.insert(0, os.path.dirname(__file__))
from extract_trackE_SBV import full_features, FEATS
WDF="/home/iiitn/Datasets/WildDeepfake/test"
def group_videos():
    vids=[]
    for cls,lab in (("real",0),("fake",1)):
        by=defaultdict(list)
        for f in glob.glob(f"{WDF}/{cls}/*.png"):
            b=os.path.basename(f); vid=b.rsplit("_",1)[0]
            try: idx=int(b.rsplit("_",1)[1].split(".")[0])
            except Exception: idx=0
            by[vid].append((idx,f))
        for vid,items in by.items():
            items.sort(); vids.append((f"wdf_{cls}_{vid}", lab, [f for _,f in items]))
    return vids
def _one(args):
    vid,lab,files,mf=args
    files=files[:mf]
    frames=[]
    for f in files:
        im=cv2.imread(f, cv2.IMREAD_COLOR)   # BGR
        if im is not None: frames.append(im)
    if len(frames)<10: return None
    try: feat=full_features(frames, 30.0, sbvgen=None)
    except Exception: return None
    if not feat: return None
    return (vid, int(lab), feat)
if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--output",required=True); ap.add_argument("--max_frames",type=int,default=60); ap.add_argument("--workers",type=int,default=8)
    a=ap.parse_args()
    vids=group_videos()
    print(f"WDF 196-D: {len(vids)} videos ({sum(1 for v in vids if v[1]==0)} real / {sum(1 for v in vids if v[1]==1)} fake) -> {a.output}",flush=True)
    tasks=[(vid,lab,files,a.max_frames) for vid,lab,files in vids]
    out=open(a.output,"w",newline=""); w=csv.DictWriter(out,fieldnames=["video_path","label"]+FEATS); w.writeheader(); ok=fail=0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs={ex.submit(_one,t):t for t in tasks}
        for fut in as_completed(futs):
            r=fut.result()
            if r:
                vid,lab,feat=r; row={"video_path":vid,"label":lab}; row.update({k:feat.get(k,0.0) for k in FEATS})
                w.writerow(row); out.flush(); ok+=1
            else: fail+=1
            if (ok+fail)%40==0: print(f"  {ok+fail}/{len(tasks)}",flush=True)
    out.close(); print(f"Done. ok={ok} fail={fail} -> {a.output}",flush=True)
