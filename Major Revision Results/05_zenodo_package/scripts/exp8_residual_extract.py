#!/usr/bin/env python3
"""EXP-8 PRNU-inspired residual comparison (R1) — extract residual descriptors per video.
3 residual methods (median[current], gaussian, wavelet). BM3D = NOT COMPUTED (no linux-aarch64 lib).
Per method: face residual energy, bg residual energy, face/bg ratio, face/bg correlation,
temporal consistency. One row/video. Reuses frozen extractor frame-loading + MediaPipe masks."""
import argparse,csv,os,sys,warnings
import numpy as np, cv2, pywt
from concurrent.futures import ProcessPoolExecutor, as_completed
warnings.filterwarnings("ignore"); sys.path.insert(0,"src")
import precompute_features_best as P
METHODS=["median","gaussian","wavelet"]
DESC=["face_energy","bg_energy","face_bg_ratio","face_bg_corr","temporal_consistency"]
COLS=[f"{m}_{d}" for m in METHODS for d in DESC]

def residual(gray,method):
    g=gray.astype(np.float64)
    if method=="median": den=cv2.medianBlur(gray,5).astype(np.float64)
    elif method=="gaussian": den=cv2.GaussianBlur(gray,(5,5),0).astype(np.float64)
    else:  # wavelet soft-threshold denoise (db4, level 2)
        c=pywt.wavedec2(g,'db4',level=2)
        sigma=np.median(np.abs(c[-1][-1]))/0.6745; thr=sigma*np.sqrt(2*np.log(g.size))
        c=[c[0]]+[tuple(pywt.threshold(d,thr,'soft') for d in lvl) for lvl in c[1:]]
        den=pywt.waverec2(c,'db4')[:g.shape[0],:g.shape[1]]
    return g-den

def _worker(args):
    vpath,label=args
    try: frames,_=P.load_video_frames(vpath,max_frames=64)
    except Exception: return None
    if not frames or len(frames)<8: return None
    fm=P.init_face_mesh()
    per={m:dict(fe=[],be=[],fbc=[],tc=[],prev=None) for m in METHODS}
    idxs=np.linspace(0,len(frames)-1,min(24,len(frames))).astype(int)
    for i in idxs:
        f=frames[i]; g=cv2.cvtColor(f,cv2.COLOR_BGR2GRAY); rgb=cv2.cvtColor(f,cv2.COLOR_BGR2RGB)
        lm=P.get_landmarks(fm,rgb)
        if lm is None: continue
        fmask=P.get_face_mask(lm,g.shape); bmask=(1-fmask).astype(np.uint8)
        for m in METHODS:
            r=residual(g,m); rf=r[fmask>0]; rb=r[bmask>0]
            if len(rf)<50 or len(rb)<50: continue
            fe=float(np.mean(rf**2)); be=float(np.mean(rb**2))
            per[m]["fe"].append(fe); per[m]["be"].append(be)
            # face/bg correlation: compare downsampled face vs bg residual histograms via corr of sorted samples
            n=min(len(rf),len(rb),2000)
            fa=np.sort(np.random.RandomState(42).choice(rf,n,replace=False)); bb=np.sort(np.random.RandomState(42).choice(rb,n,replace=False))
            if fa.std()>1e-6 and bb.std()>1e-6: per[m]["fbc"].append(float(np.corrcoef(fa,bb)[0,1]))
            # temporal consistency: corr of face-residual patch vs previous
            ys,xs=np.where(fmask>0); patch=cv2.resize(r[ys.min():ys.max()+1,xs.min():xs.max()+1],(32,32)).ravel()
            if per[m]["prev"] is not None and patch.std()>1e-6 and per[m]["prev"].std()>1e-6:
                per[m]["tc"].append(float(np.corrcoef(patch,per[m]["prev"])[0,1]))
            per[m]["prev"]=patch
    fm.close()
    row={"video_path":vpath,"label":label}
    for m in METHODS:
        fe=np.mean(per[m]["fe"]) if per[m]["fe"] else 0.0; be=np.mean(per[m]["be"]) if per[m]["be"] else 0.0
        row[f"{m}_face_energy"]=fe; row[f"{m}_bg_energy"]=be
        row[f"{m}_face_bg_ratio"]=fe/be if be>1e-9 else 0.0
        row[f"{m}_face_bg_corr"]=float(np.mean(per[m]["fbc"])) if per[m]["fbc"] else 0.0
        row[f"{m}_temporal_consistency"]=float(np.mean(per[m]["tc"])) if per[m]["tc"] else 0.0
    return row

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--video_dir",required=True); ap.add_argument("--output",required=True)
    ap.add_argument("--label",type=int,required=True); ap.add_argument("--workers",type=int,default=10)
    a=ap.parse_args(); vids=P.discover_videos(a.video_dir,a.label)
    print(f"{len(vids)} videos -> {a.output}",flush=True)
    hdr=["video_path","label"]+COLS; new=not os.path.exists(a.output)
    out=open(a.output,"a",newline=""); w=csv.DictWriter(out,fieldnames=hdr)
    if new: w.writeheader()
    ok=0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs={ex.submit(_worker,(v,l)):v for v,l in vids}
        for fu in as_completed(futs):
            r=fu.result()
            if r: w.writerow(r); out.flush(); ok+=1
    out.close(); print(f"done ok={ok}",flush=True)
if __name__=="__main__": main()
