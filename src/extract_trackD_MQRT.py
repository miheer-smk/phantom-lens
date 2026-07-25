#!/usr/bin/env python3
"""Track D Batch-2 — UNIFIED extractor: one MediaPipe pass per video -> families M,Q,R,T.
M cardiac cross-modal coherence (rPPG pulse vs ballistocardiographic head motion, 0.7-4 Hz).
Q muscle co-activation (cross-corr + lag between landmark-group displacement series).
R blink kinematics (close/open duration ratio, peak-velocity asymmetry, EAR skewness).
T rigid 3-D consistency (Tomasi-Kanade factorisation of rigid-landmark tracks).
All ratio/correlation/coherence -> dimensionless. Denser temporal sampling (max_frames=60).
Input: --manifest (video_path,label). Deterministic. Reuses precompute_features_best helpers.
"""
import argparse, csv, os, sys, warnings
import numpy as np, cv2
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.signal import butter, filtfilt, coherence, csd, welch, detrend
from scipy.stats import skew
warnings.filterwarnings("ignore"); sys.path.insert(0, os.path.dirname(__file__))
import precompute_features_best as P

LE=[33,160,158,133,153,144]; RE=[362,385,387,263,373,380]
FOREHEAD=[10,67,109,338,297,21,54,103,68,104]
LCHEEK=[116,117,118,119,120,121,187,207,206]; RCHEEK=[345,346,347,348,349,350,411,427,426]
NOSE=[168,6,197,195,5,4,1]; MOUTH=P.LIPS_OUTER if hasattr(P,"LIPS_OUTER") else [61,146,91,181,84,17,314,405,321,375,291,308,324,318,402,317,14,87,178,88,95]
RIGID=NOSE+[33,133,362,263,1,4]            # stable non-deforming points
M_F=["m_coh_at_f0","m_freq_agreement","m_max_inband_coh","m_phase_stability"]
Q_F=["q_mouth_eye_xcorr","q_mouth_eye_lag","q_cheek_mouth_xcorr","q_cheek_mouth_lag",
     "q_brow_eye_xcorr","q_brow_eye_lag","q_upper_lower_xcorr","q_upper_lower_lag"]
R_F=["r_close_open_ratio","r_peakvel_asym","r_ear_skew"]
T_F=["t3_reproj_err","t3_rank4_residual"]
FEATS=M_F+Q_F+R_F+T_F

def _bp(x,fs,lo=0.7,hi=4.0):
    x=np.asarray(x,float)
    if len(x)<9 or np.std(x)<1e-8: return x-np.mean(x)
    ny=fs/2.0; lo2,hi2=max(lo/ny,1e-3),min(hi/ny,0.99)
    if hi2<=lo2: return detrend(x)
    b,a=butter(2,[lo2,hi2],btype="band")
    try: return filtfilt(b,a,x)
    except Exception: return detrend(x)
def _xcorr(a,b,maxlag=8):
    a=(a-a.mean())/(a.std()+1e-8); b=(b-b.mean())/(b.std()+1e-8); n=len(a)
    best=0.0; bl=0
    for L in range(-maxlag,maxlag+1):
        if L<0: c=np.corrcoef(a[:n+L],b[-L:])[0,1] if n+L>3 else 0
        elif L>0: c=np.corrcoef(a[L:],b[:n-L])[0,1] if n-L>3 else 0
        else: c=np.corrcoef(a,b)[0,1]
        if np.isfinite(c) and abs(c)>abs(best): best=c; bl=L
    return float(best), int(bl)

def process_video(video_path,label,max_frames=60):
    try: frames,fps=P.load_video_frames(video_path,max_frames=max_frames)
    except Exception: return None
    if frames is None or len(frames)<20: return None
    # effective sampling rate given striding
    try:
        cap=cv2.VideoCapture(str(video_path)); total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()
        step=max(1,total//max_frames) if total>max_frames else 1
    except Exception: step=1
    fs=max(fps/step,4.0)
    fm=P.init_face_mesh()
    lms=[]; skin=[]
    for f in frames:
        rgb=cv2.cvtColor(f,cv2.COLOR_BGR2RGB); lm=P.get_landmarks(fm,rgb)
        lms.append(lm)
        if lm is not None:
            m=P.landmarks_to_mask(lm,FOREHEAD+LCHEEK+RCHEEK,rgb.shape)
            skin.append(rgb[m>0].reshape(-1,3).mean(0) if (m>0).sum()>20 else np.array([np.nan]*3))
        else: skin.append(np.array([np.nan]*3))
    fm.close()
    vi=[i for i,l in enumerate(lms) if l is not None]
    if len(vi)<20: return None
    def cen(idx): return np.array([lms[i][idx].mean(0) for i in vi])   # centroid track (pixels)
    mouth=cen(MOUTH); eye=cen(LE+RE); cheek=cen(LCHEEK+RCHEEK); brow=cen(FOREHEAD)
    rigid=np.array([lms[i][RIGID] for i in vi])                        # (F,P,2)
    def vel(c): return np.linalg.norm(np.diff(c,axis=0),axis=1)
    out={}

    # ---- M: rPPG (POS) vs head motion coherence ----
    S=np.array(skin)[vi]
    if np.isfinite(S).all() and len(S)>=20:
        Sn=S/ (S.mean(0)+1e-8)
        Xs=np.array([[0,1,-1],[-2,1,1]])@Sn.T
        pulse=Xs[0]+ (np.std(Xs[0])/(np.std(Xs[1])+1e-8))*Xs[1]
        pulse=_bp(pulse,fs)
        head=_bp(detrend(rigid.mean(1)[:,1]),fs)                       # vertical rigid centroid
        npg=min(len(pulse)//2, 32)
        if npg>=8:
            fco,Cxy=coherence(pulse,head,fs=fs,nperseg=npg)
            band=(fco>=0.7)&(fco<=4.0)
            fp,Pp=welch(pulse,fs=fs,nperseg=npg); fh,Ph=welch(head,fs=fs,nperseg=npg)
            bandp=(fp>=0.7)&(fp<=4.0)
            f0p=fp[bandp][np.argmax(Pp[bandp])] if bandp.any() else 0
            f0h=fh[bandp][np.argmax(Ph[bandp])] if bandp.any() else 0
            _,Pxy=csd(pulse,head,fs=fs,nperseg=npg); ph=np.angle(Pxy[band])
            out["m_coh_at_f0"]=float(Cxy[band][np.argmin(np.abs(fco[band]-f0p))]) if band.any() else 0.0
            out["m_freq_agreement"]=float(1.0/(1.0+abs(f0p-f0h)))
            out["m_max_inband_coh"]=float(Cxy[band].max()) if band.any() else 0.0
            out["m_phase_stability"]=float(1.0-min(np.std(ph)/np.pi,1.0)) if len(ph)>1 else 0.0
    for k in M_F: out.setdefault(k,0.0)

    # ---- Q: muscle co-activation ----
    vm,ve,vc,vb=vel(mouth),vel(eye),vel(cheek),vel(brow)
    upper=vel((eye+brow)/2); lower=vel((mouth+cheek)/2)
    if len(vm)>=6:
        out["q_mouth_eye_xcorr"],out["q_mouth_eye_lag"]=_xcorr(vm,ve)
        out["q_cheek_mouth_xcorr"],out["q_cheek_mouth_lag"]=_xcorr(vc,vm)
        out["q_brow_eye_xcorr"],out["q_brow_eye_lag"]=_xcorr(vb,ve)
        out["q_upper_lower_xcorr"],out["q_upper_lower_lag"]=_xcorr(upper,lower)
    for k in Q_F: out.setdefault(k,0.0)

    # ---- R: blink kinematics ----
    ear=np.array([ (P.compute_ear(lms[i],LE)+P.compute_ear(lms[i],RE))/2 for i in vi ])
    ear=ear[np.isfinite(ear)]
    if len(ear)>=12:
        out["r_ear_skew"]=float(skew(ear))
        thr=ear.mean()-0.6*ear.std(); dv=np.diff(ear)
        closes=[]; opens=[]; pv_c=[]; pv_o=[]
        i=1
        while i<len(ear)-1:
            if ear[i]<thr and ear[i]<=ear[i-1]:
                j=i
                while j>0 and ear[j-1]>ear[j]: j-=1        # closing start
                k=i
                while k<len(ear)-1 and ear[k+1]>ear[k]: k+=1  # opening end
                if i-j>=1 and k-i>=1:
                    closes.append(i-j); opens.append(k-i)
                    pv_c.append(abs(np.min(dv[max(j,0):i])) if i>j else 0)
                    pv_o.append(abs(np.max(dv[i:k])) if k>i else 0)
                i=k+1
            else: i+=1
        if closes:
            out["r_close_open_ratio"]=float(np.mean(closes)/(np.mean(opens)+1e-8))
            out["r_peakvel_asym"]=float((np.mean(pv_c)-np.mean(pv_o))/(np.mean(pv_c)+np.mean(pv_o)+1e-8))
    for k in R_F: out.setdefault(k,0.0)

    # ---- T: rigid 3-D (Tomasi-Kanade) ----
    W=np.vstack([rigid[:,:,0],rigid[:,:,1]]).astype(float)   # (2F, P)
    W=W-W.mean(1,keepdims=True)
    try:
        U,s,Vt=np.linalg.svd(W,full_matrices=False)
        rank=3
        Wr=(U[:,:rank]*s[:rank])@Vt[:rank]
        out["t3_reproj_err"]=float(np.linalg.norm(W-Wr)/(np.linalg.norm(W)+1e-8))
        out["t3_rank4_residual"]=float((s[4:]**2).sum()/((s**2).sum()+1e-8)) if len(s)>4 else 0.0
    except Exception:
        out["t3_reproj_err"]=0.0; out["t3_rank4_residual"]=0.0

    row={"video_path":video_path,"label":int(label)}
    row.update({k:(0.0 if (out[k] is None or not np.isfinite(out[k])) else float(out[k])) for k in FEATS})
    return row

def _w(a): return process_video(*a)
if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--manifest",required=True); ap.add_argument("--output",required=True)
    ap.add_argument("--max_frames",type=int,default=60); ap.add_argument("--workers",type=int,default=16)
    a=ap.parse_args()
    import pandas as pd
    man=pd.read_csv(a.manifest); tasks=[(r.video_path,int(r.label),a.max_frames) for r in man.itertuples()]
    print(f"MQRT: {len(tasks)} videos -> {a.output}",flush=True)
    hdr=["video_path","label"]+FEATS
    out=open(a.output,"w",newline=""); w=csv.DictWriter(out,fieldnames=hdr); w.writeheader(); ok=fail=0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs={ex.submit(_w,t):t for t in tasks}
        for fut in as_completed(futs):
            r=fut.result()
            if r: w.writerow(r); out.flush(); ok+=1
            else: fail+=1
            if (ok+fail)%200==0: print(f"  {ok+fail}/{len(tasks)}",flush=True)
    out.close(); print(f"Done. ok={ok} fail={fail} -> {a.output}",flush=True)
