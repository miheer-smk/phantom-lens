#!/usr/bin/env python3
"""Xception baseline — face-crop extraction (fair DL comparison, R5.2/R3.4).
Samples FRAMES_PER frames/video, MediaPipe face-crop -> 299x299, saved as JPG.
Writes a manifest with identity + split so training reuses the SAME identity-disjoint split as PRISM.
"""
import os,sys,csv,argparse,warnings
import numpy as np, cv2
from concurrent.futures import ProcessPoolExecutor, as_completed
warnings.filterwarnings("ignore"); sys.path.insert(0,"src")
import precompute_features_best as P
from protocol import load_id2split, clip_identities
FRAMES_PER=8; SZ=299
CROOT="data_xception/crops"

def _worker(args):
    vpath,label,dataset,split=args
    try:
        frames,_=P.load_video_frames(vpath,max_frames=64)
    except Exception: return []
    if not frames or len(frames)<4: return []
    idxs=np.linspace(0,len(frames)-1,min(FRAMES_PER,len(frames))).astype(int)
    fm=P.init_face_mesh(); rows=[]; base=os.path.splitext(os.path.basename(vpath))[0]
    outdir=f"{CROOT}/{dataset}/{label}"; os.makedirs(outdir,exist_ok=True)
    for j,i in enumerate(idxs):
        f=frames[i]; rgb=cv2.cvtColor(f,cv2.COLOR_BGR2RGB); lm=P.get_landmarks(fm,rgb)
        if lm is None: continue
        x0,y0,x1,y1=P.get_face_bbox(lm,f.shape,padding=0.2)
        if x1-x0<20 or y1-y0<20: continue
        crop=cv2.resize(f[y0:y1,x0:x1],(SZ,SZ))
        cp=f"{outdir}/{base}_{j}.jpg"; cv2.imwrite(cp,crop,[cv2.IMWRITE_JPEG_QUALITY,95])
        rows.append((cp,base,dataset,label,split))
    fm.close(); return rows

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--manifest",required=True)
    ap.add_argument("--set",required=True,choices=["ffpp","celebdf"]); ap.add_argument("--workers",type=int,default=10)
    a=ap.parse_args()
    id2split=load_id2split(); tasks=[]
    R=os.environ.get("DATASETS_ROOT","data")  # set DATASETS_ROOT to the parent dir of FaceForensics++/Celeb-DF-v2
    if a.set=="ffpp":
        specs=[("original_sequences/youtube/c23/videos",0,"real"),
               ("manipulated_sequences/Deepfakes/c23/videos",1,"deepfakes"),
               ("manipulated_sequences/Face2Face/c23/videos",1,"face2face"),
               ("manipulated_sequences/FaceSwap/c23/videos",1,"faceswap"),
               ("manipulated_sequences/NeuralTextures/c23/videos",1,"neuraltextures")]
        for sub,lab,ds in specs:
            d=f"{R}/FaceForensics++/{sub}"
            for fn in sorted(os.listdir(d)):
                if not fn.endswith(".mp4"): continue
                ids=clip_identities(fn); parts={id2split.get(i) for i in ids}; parts.discard(None)
                sp=parts.pop() if len(parts)==1 else None
                if sp: tasks.append((f"{d}/{fn}",lab,ds,sp))
    else:
        for sub,lab in [("Celeb-real",0),("YouTube-real",0),("Celeb-synthesis",1)]:
            d=f"{R}/Celeb-DF-v2/{sub}"
            for fn in sorted(os.listdir(d)):
                if fn.endswith(".mp4"): tasks.append((f"{d}/{fn}",lab,"celebdf","test"))
    print(f"{a.set}: {len(tasks)} videos -> crops",flush=True)
    hdr=["crop_path","video","dataset","label","split"]; new=not os.path.exists(a.manifest)
    out=open(a.manifest,"a",newline=""); w=csv.writer(out)
    if new: w.writerow(hdr)
    done=0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs={ex.submit(_worker,t):t for t in tasks}
        for fu in as_completed(futs):
            for r in fu.result(): w.writerow(r)
            out.flush(); done+=1
            if done%500==0: print(f"  {done}/{len(tasks)} videos",flush=True)
    out.close(); print(f"Done {a.set}. videos={done}",flush=True)

if __name__=="__main__": main()
