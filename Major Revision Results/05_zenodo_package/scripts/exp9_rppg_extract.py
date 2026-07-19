#!/usr/bin/env python3
"""EXP-9 rPPG POS/CHROM comparison (R1) — extract pure-POS & pure-CHROM quality descriptors per video.
Current method (POS+CHROM dual) = existing 50-D t_rppg_* features. Here we add pure POS and pure CHROM.
Per method: snr, peak_prominence, interregion_corr, harmonic_ratio. Also brightness_var (illumination)
+ n_frames (sequence length) for stratification. Reuses forehead/cheek ROI traces from frozen extractor.
NOTE: rPPG here is a forensic temporal descriptor, not medical-grade pulse estimation."""
import argparse,csv,os,sys,warnings
import numpy as np, cv2
from scipy import signal as sps
from concurrent.futures import ProcessPoolExecutor, as_completed
warnings.filterwarnings("ignore"); sys.path.insert(0,"src")
import precompute_features_best as P
DESC=["snr","peak_prom","interreg_corr","harmonic"]
COLS=[f"pos_{d}" for d in DESC]+[f"chrom_{d}" for d in DESC]+["brightness_var","n_frames"]

def bandpass(x,fps,lo=0.7,hi=4.0):
    ny=fps/2; lo,hi=max(lo/ny,1e-3),min(hi/ny,0.99)
    if hi<=lo: return x
    b,a=sps.butter(3,[lo,hi],btype='band'); return sps.filtfilt(b,a,x)

def chrom(rgb,fps):  # de Haan & Jeanne 2013
    R,G,B=rgb[:,0],rgb[:,1],rgb[:,2]
    Rn,Gn,Bn=R/(R.mean()+1e-8),G/(G.mean()+1e-8),B/(B.mean()+1e-8)
    Xs=3*Rn-2*Gn; Ys=1.5*Rn+Gn-1.5*Bn
    Xf,Yf=bandpass(Xs,fps),bandpass(Ys,fps)
    al=Xf.std()/(Yf.std()+1e-8); return Xf-al*Yf

def pos(rgb,fps,wsec=1.6):  # Wang et al. 2017
    n=len(rgb); L=max(int(wsec*fps),6); H=np.zeros(n)
    for m in range(0,n-L):
        C=rgb[m:m+L]; mu=C.mean(0)+1e-8; Cn=C/mu
        S1=Cn[:,1]-Cn[:,2]; S2=Cn[:,1]+Cn[:,2]-2*Cn[:,0]
        h=S1+(S1.std()/(S2.std()+1e-8))*S2; h=h-h.mean()
        H[m:m+L]+=h
    return bandpass(H,fps)

def quality(sig,fps):
    if len(sig)<32 or np.std(sig)<1e-8: return 0.0,0.0,0.0
    f,pxx=sps.welch(sig,fps,nperseg=min(len(sig),256))
    band=(f>=0.7)&(f<=4.0)
    if band.sum()<2 or pxx[band].sum()<=0: return 0.0,0.0,0.0
    pk=np.argmax(pxx*band); fpk=f[pk]
    snr=float(pxx[pk]/(pxx[band].sum()+1e-12))
    prom=float(pxx[pk]/(np.median(pxx[band])+1e-12))
    h2=np.argmin(np.abs(f-2*fpk)); harm=float(pxx[h2]/(pxx[pk]+1e-12))
    return snr,prom,harm

def _worker(args):
    vpath,label=args
    try: frames,fps=P.load_video_frames(vpath,max_frames=150)
    except Exception: return None
    if not frames or len(frames)<P.MIN_FRAMES_RPPG: return None
    fm=P.init_face_mesh(); fh=[]; lc=[]; rc=[]; bri=[]
    for f in frames:
        rgb=cv2.cvtColor(f,cv2.COLOR_BGR2RGB); lm=P.get_landmarks(fm,rgb)
        bri.append(float(cv2.cvtColor(f,cv2.COLOR_BGR2GRAY).mean()))
        if lm is None: fh.append([0,0,0]); lc.append([0,0,0]); rc.append([0,0,0]); continue
        cf=cv2.bilateralFilter(rgb,7,25,25)
        fh.append(P.get_roi_mean_rgb(cf,lm,P.FOREHEAD)); lc.append(P.get_roi_mean_rgb(cf,lm,P.LEFT_CHEEK)); rc.append(P.get_roi_mean_rgb(cf,lm,P.RIGHT_CHEEK))
    fm.close()
    regions=[np.array(fh,float),np.array(lc,float),np.array(rc,float)]
    regions=[r for r in regions if (r.sum(1)>0).sum()>=P.MIN_FRAMES_RPPG]
    if len(regions)<2: return None
    row={"video_path":vpath,"label":label,"brightness_var":float(np.var(bri)),"n_frames":len(frames)}
    for name,algo in [("pos",pos),("chrom",chrom)]:
        sigs=[algo(r,fps) for r in regions]
        q=[quality(s,fps) for s in sigs]
        snr=np.mean([x[0] for x in q]); prom=np.mean([x[1] for x in q]); harm=np.mean([x[2] for x in q])
        # inter-region correlation of the pulse signals
        L=min(len(s) for s in sigs); cs=[]
        for i in range(len(sigs)):
            for j in range(i+1,len(sigs)):
                a,b=sigs[i][:L],sigs[j][:L]
                if a.std()>1e-8 and b.std()>1e-8: cs.append(np.corrcoef(a,b)[0,1])
        row[f"{name}_snr"]=snr; row[f"{name}_peak_prom"]=prom; row[f"{name}_harmonic"]=harm
        row[f"{name}_interreg_corr"]=float(np.mean(cs)) if cs else 0.0
    return row

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--video_dir",required=True); ap.add_argument("--output",required=True)
    ap.add_argument("--label",type=int,required=True); ap.add_argument("--workers",type=int,default=12)
    a=ap.parse_args(); vids=P.discover_videos(a.video_dir,a.label)
    print(f"{len(vids)} videos -> {a.output}",flush=True)
    hdr=["video_path","label"]+COLS; new=not os.path.exists(a.output)
    out=open(a.output,"a",newline=""); w=csv.DictWriter(out,fieldnames=hdr,extrasaction="ignore")
    if new: w.writeheader()
    ok=0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs={ex.submit(_worker,(v,l)):v for v,l in vids}
        for fu in as_completed(futs):
            r=fu.result()
            if r: w.writerow(r); out.flush(); ok+=1
    out.close(); print(f"done ok={ok}",flush=True)
if __name__=="__main__": main()
