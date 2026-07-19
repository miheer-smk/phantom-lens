#!/usr/bin/env python3
"""EXP-9 analysis (R1). rPPG AUC per method (current[POS+CHROM dual] / POS / CHROM) per condition:
low/high head motion (s_flow_mag), low/high illumination variance (brightness_var), c23/c40,
short/long sequences (n_frames). Real-vs-fake AUC from each method's 4 rPPG descriptors, identity-disjoint.
NOTE: rPPG = forensic temporal descriptor, NOT medical-grade pulse."""
import os,sys,json,subprocess,datetime
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore"); sys.path.insert(0,"src")
from protocol import make_splits
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import lightgbm as lgb, cv2
def true_len(vp):
    try:
        c=cv2.VideoCapture(vp); n=int(c.get(cv2.CAP_PROP_FRAME_COUNT)); c.release(); return n
    except: return 0
SEED=42; F="features"; OUT="results_clean"
MAN=["deepfakes","face2face","faceswap","neuraltextures"]
CUR=["t_rppg_snr","t_rppg_peak_prominence","t_rppg_interregion_corr","t_rppg_harmonic_ratio"]
POS=["pos_snr","pos_peak_prom","pos_interreg_corr","pos_harmonic"]
CHR=["chrom_snr","chrom_peak_prom","chrom_interreg_corr","chrom_harmonic"]
def base(p): return os.path.basename(str(p))
def load(name,comp):
    f50=pd.read_csv(f"{F}/ffpp_{'original' if name=='real' else name}_{comp}.csv")
    fr=pd.read_csv(f"{F}/rppg_{'original' if name=='real' else name}_{comp}.csv")
    f50["_b"]=f50.video_path.map(base); fr["_b"]=fr.video_path.map(base)
    m=f50.merge(fr[["_b"]+POS+CHR+["brightness_var","n_frames"]],on="_b",how="inner"); m["comp"]=comp
    return make_splits(m)
frames=[]
for comp in ("c23","c40"):
    frames.append(load("real",comp).assign(src="real"))
    for mm in MAN: frames.append(load(mm,comp).assign(src=mm))
D=pd.concat(frames,ignore_index=True)
allc=CUR+POS+CHR+["s_flow_mag","brightness_var","n_frames"]
for c in allc: D[c]=pd.to_numeric(D[c],errors="coerce").replace([np.inf,-np.inf],np.nan); D[c]=D[c].fillna(D[c].median())
def LGBM(): return lgb.LGBMClassifier(n_estimators=200,max_depth=6,learning_rate=0.05,num_leaves=31,min_child_samples=20,class_weight="balanced",random_state=SEED,verbose=-1,n_jobs=-1)
def commit():
    try: return subprocess.check_output(["git","rev-parse","--short","HEAD"],text=True).strip()
    except: return "nogit"
tr=D[D.partition.isin(["train","val"])]; te=D[D.partition=="test"].copy()
# stratum thresholds from TRAIN only
thr_motion=tr.s_flow_mag.median(); thr_illum=tr.brightness_var.median(); thr_len=tr.n_frames.median()
te["motion"]=np.where(te.s_flow_mag>=thr_motion,"high_motion","low_motion")
te["illum"]=np.where(te.brightness_var>=thr_illum,"high_illum","low_illum")
te=te.copy(); te["true_len"]=te.video_path.map(true_len)
thr_len=tr.assign(tl=tr.video_path.map(true_len)).tl.median()
te["length"]=np.where(te.true_len>=thr_len,"long_seq","short_seq")
def auc(cols,sub):
    if sub.label.nunique()<2 or len(sub)<20: return None
    return round(roc_auc_score(sub.label.values.astype(int),sub["_score_"].values),4)
rows=[]
for name,cols in [("current(POS+CHROM)",CUR),("POS",POS),("CHROM",CHR)]:
    sc=StandardScaler().fit(tr[cols].values); clf=LGBM(); clf.fit(sc.transform(tr[cols].values),tr['label'].values.astype(int))
    te["_score_"]=clf.predict_proba(sc.transform(te[cols].values))[:,1]
    r={"method":name,"overall":auc(cols,te)}
    for col,vals in [("comp",["c23","c40"]),("motion",["low_motion","high_motion"]),("illum",["low_illum","high_illum"]),("length",["short_seq","long_seq"])]:
        for v in vals: r[v]=auc(cols,te[te[col]==v])
    rows.append(r)
pd.DataFrame(rows).to_csv(f"{OUT}/rppg_comparison.csv",index=False)
json.dump(dict(provenance=dict(script="exp9_analyze.py",git_commit=commit(),seed=SEED,date=datetime.date.today().isoformat(),
    note="rPPG = forensic temporal descriptor, not medical-grade pulse; current=POS+CHROM dual (50-D t_rppg_*)",
    strata_thresholds=dict(motion=round(float(thr_motion),4),illum=round(float(thr_illum),2),length=int(thr_len))),
    n_test=int(len(te)),results=rows),open(f"{OUT}/rppg_comparison.json","w"),indent=2)
print("=== EXP-9 rPPG POS/CHROM COMPARISON (real-vs-fake AUC per condition) ===")
hdr=["method","overall","c23","c40","low_motion","high_motion","low_illum","high_illum","short_seq","long_seq"]
print("  "+" ".join(f"{h:>11s}" for h in hdr))
for r in rows: print("  "+f"{r['method']:>11s} "+" ".join(f"{str(r.get(h,'-')):>11s}" for h in hdr[1:]))
print(f"\nsaved {OUT}/rppg_comparison.csv, rppg_comparison.json (commit {commit()})")
