#!/usr/bin/env python3
"""WildDeepfake zero-shot feature extraction (53-D: original 50 + G1).
WildDeepfake test is pre-cropped face PNG frames grouped by sequence (<seq>_<frame>.png).
We reconstruct each sequence's frame list and inject it into the FROZEN extractor via a
loader shim, so the identical 50-feature + G1 pipeline runs unchanged. One row per sequence.
CAVEAT (documented): face-crops -> background-dependent spatial features degenerate; median
~30 frames/seq -> rPPG mostly returns defaults. Reported alongside results.
"""
import os,sys,csv,re,glob,warnings
import numpy as np, cv2
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
warnings.filterwarnings("ignore")
sys.path.insert(0,"src")
import precompute_features_best as P
import extract_roi_features as ROI
import roi_config as RC

ROOT="/home/iiitn/Datasets/WildDeepfake/test"
OUT="features/wilddeepfake_test_53d.csv"
ALLF = list(P.FEATURE_NAMES_SPATIAL)+list(P.FEATURE_NAMES_TEMPORAL)+RC.CANDIDATE_GROUPS["G1_mouth_instability"]

def sequences(cls):
    d=defaultdict(list)
    for p in glob.glob(f"{ROOT}/{cls}/*.png"):
        m=re.match(r"(\d+)_(\d+)\.png", os.path.basename(p))
        if m: d[m.group(1)].append((int(m.group(2)),p))
    return {s:[p for _,p in sorted(v)] for s,v in d.items()}

def _worker(args):
    seq_id, paths, label = args
    frames=[cv2.imread(p) for p in paths]
    frames=[f for f in frames if f is not None]
    if len(frames)<P.MIN_FRAMES_SPATIAL: return None
    # loader shim: inject this sequence's frames into the frozen extractor
    orig=P.load_video_frames
    P.load_video_frames=lambda path,max_frames=300,target_size=None: (frames[:max_frames], 30.0)
    try:
        row50=P.process_single_video(f"wdf::{seq_id}", label, max_frames=300)
        rowg1=ROI.process_video(f"wdf::{seq_id}", label, max_frames=150)
    finally:
        P.load_video_frames=orig
    if row50 is None: return None
    out={"video_path":f"{label}_{seq_id}","label":label}
    for f in P.FEATURE_NAMES_SPATIAL+P.FEATURE_NAMES_TEMPORAL: out[f]=row50.get(f,0.0)
    g1=RC.CANDIDATE_GROUPS["G1_mouth_instability"]
    for f in g1: out[f]=(rowg1.get(f,0.0) if rowg1 else 0.0)
    return out

def main():
    real=sequences("real"); fake=sequences("fake")
    tasks=[(s,paths,0) for s,paths in real.items()]+[(s,paths,1) for s,paths in fake.items()]
    print(f"sequences: real={len(real)} fake={len(fake)} total={len(tasks)}",flush=True)
    header=["video_path","label"]+P.FEATURE_NAMES_SPATIAL+P.FEATURE_NAMES_TEMPORAL+RC.CANDIDATE_GROUPS["G1_mouth_instability"]
    out=open(OUT,"w",newline=""); w=csv.DictWriter(out,fieldnames=header); w.writeheader()
    ok=fail=0
    with ProcessPoolExecutor(max_workers=10) as ex:
        futs={ex.submit(_worker,t):t for t in tasks}
        for fu in as_completed(futs):
            r=fu.result()
            if r: w.writerow(r); out.flush(); ok+=1
            else: fail+=1
    out.close()
    print(f"Done. ok={ok} fail={fail} -> {OUT}",flush=True)

if __name__=="__main__": main()
