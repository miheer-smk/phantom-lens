#!/usr/bin/env python3
"""Track E2 — window-level MIL scoring. DEV only; sealed untouched. Reuses existing extraction.
Segment each video's per-frame spatial into windows (30 frames, stride 15) -> 143 order-stats/window
+ the video-level 37 temporal + 3 G1 (constant per window) = 183-D. Train real-vs-fake on WINDOWS,
score per window, aggregate to video by {mean,max,top3,p90,frac>0.5}; aggregator chosen by identity-
grouped 5-fold CV on celebdf_dev. Windows grouped by identity (never span train/val).
"""
import os, sys, json, subprocess, datetime, re
import numpy as np, pandas as pd, warnings
from scipy.stats import skew, kurtosis
warnings.filterwarnings("ignore"); sys.path.insert(0,"src")
from protocol import make_splits, clip_identities
from extract_trackE_perframe import SPATIAL13
from extract_trackE_SBV import FEATS
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
import lightgbm as lgb
SEED=42; F="features"; TE=f"{F}/trackE"; OUT="results_clean"; MAN=["deepfakes","face2face","faceswap","neuraltextures"]
WIN=30; STRIDE=15
STATS=["mean","std","min","max","p10","p25","p75","p90","iqr","skew","kurt"]
AGG=[f"{f}__{s}" for f in SPATIAL13 for s in STATS]
TEMP=[c for c in FEATS if c.startswith("t_")]; G1=[c for c in FEATS if c.startswith("roi_")]
WFEATS=AGG+TEMP+G1
def bn(p): return os.path.basename(str(p))
def commit():
    try: return subprocess.check_output(["git","rev-parse","--short","HEAD"],text=True).strip()
    except: return "nogit"
def LGBM(): return lgb.LGBMClassifier(n_estimators=300,max_depth=6,learning_rate=0.05,num_leaves=31,
    min_child_samples=20,class_weight="balanced",random_state=SEED,verbose=-1,n_jobs=1,deterministic=True,force_row_wise=True)
def winstats(a):  # a: (w,13) -> 143
    out=[]
    for j in range(13):
        x=a[:,j]; q=np.percentile(x,[10,25,75,90])
        out+=[x.mean(),x.std(),x.min(),x.max(),q[0],q[1],q[2],q[3],q[3]-q[1],
              float(skew(x)) if len(x)>2 else 0.0, float(kurtosis(x)) if len(x)>3 else 0.0]
    return out
# video-level temporal+G1 from plain_everyone; FF++ keyed by FULL path (basenames collide), celebdf by basename
DIR={"deepfakes":"Deepfakes","face2face":"Face2Face","faceswap":"FaceSwap","neuraltextures":"NeuralTextures"}
def method(p):
    for m,d in DIR.items():
        if f"/{d}/" in p: return m
    return "real" if "youtube" in p else ("celebdf" if "Celeb-DF" in p else "?")
ev=pd.read_csv(f"{TE}/plain_everyone_E3.csv"); ev["src"]=ev.video_path.map(method); ev["_b"]=ev.video_path.map(bn)
vt_ff=ev[ev.src.isin(["real"]+MAN)].drop_duplicates("video_path").set_index("video_path")[TEMP+G1].to_dict("index")
vt_cd=ev[ev.src=="celebdf"].drop_duplicates("_b").set_index("_b")[TEMP+G1].to_dict("index")
def build(perframe_csv,key,vtmap,is_path):
    d=pd.read_csv(perframe_csv); rows=[]
    for vid,g in d.groupby(key):
        g=g.sort_values("frame"); arr=g[SPATIAL13].values; lab=int(g.label.iloc[0])
        vtl=vtmap.get(vid) if is_path else vtmap.get(bn(vid))
        if vtl is None or len(arr)<WIN: continue
        for st in range(0,len(arr)-WIN+1,STRIDE):
            rows.append([vid,lab]+winstats(arr[st:st+WIN])+[vtl[c] for c in TEMP+G1])
    return pd.DataFrame(rows,columns=["vid","label"]+WFEATS)
ffw=build(f"{TE}/perframe_ffpp_trainval.csv","video_path",vt_ff,True)
cdw=build(f"{TE}/perframe_celebdf_dev.csv","video",vt_cd,False); cdw["_b"]=cdw.vid.map(bn)
ffw["video_path"]=ffw.vid; ffw=make_splits(ffw)
for c in WFEATS: ffw[c]=pd.to_numeric(ffw[c],errors="coerce").replace([np.inf,-np.inf],np.nan); cdw[c]=pd.to_numeric(cdw[c],errors="coerce").replace([np.inf,-np.inf],np.nan)
med=ffw[ffw.partition=="train"][WFEATS].median(); ffw[WFEATS]=ffw[WFEATS].fillna(med); cdw[WFEATS]=cdw[WFEATS].fillna(med)
print(f"windows: FF++ {len(ffw)} (from {ffw.vid.nunique()} vids), celebdf_dev {len(cdw)} (from {cdw._b.nunique()} vids)",flush=True)
tr=ffw[ffw.partition=="train"]
sc=StandardScaler().fit(tr[WFEATS].values); m=LGBM().fit(sc.transform(tr[WFEATS].values),tr.label.values.astype(int))
cdw["p"]=m.predict_proba(sc.transform(cdw[WFEATS].values))[:,1]
# aggregate windows -> video
def agg(gp,how):
    if how=="mean": return gp.mean()
    if how=="max": return gp.max()
    if how=="p90": return gp.quantile(0.9)
    if how=="top3": return gp.nlargest(3).mean()
    if how=="frac": return (gp>=0.5).mean()
cdvid=cdw.groupby("_b"); vids=list(cdvid.groups);
ylab=np.array([int(cdvid.get_group(v).label.iloc[0]) for v in vids])
ids=np.array([(re.findall(r"id(\d+)",v) or [v])[0] for v in vids])
def cv(scr):
    a=[roc_auc_score(ylab[i],scr[i]) for _,i in GroupKFold(5).split(scr,ylab,ids) if len(np.unique(ylab[i]))>1]
    return round(float(np.mean(a)),4),round(float(np.std(a)),4)
res={"provenance":dict(script="exp_trackE_E2.py",git_commit=commit(),seed=SEED,date=datetime.date.today().isoformat(),axis_dev_only=True,sealed_touched=False,window=WIN,stride=STRIDE),"aggregators":{}}
print("="*60);print("TRACK E2 — windowed MIL (celebdf_dev CV per aggregator)");print("="*60)
for how in ["mean","max","top3","p90","frac"]:
    scr=np.array([agg(cdvid.get_group(v).p,how) for v in vids]); mn,st=cv(scr)
    res["aggregators"][how]=dict(celebdf_dev_cv_mean=mn,celebdf_dev_cv_std=st)
    print(f"  {how:6s} celebdf_dev CV = {mn:.4f} ±{st:.3f}")
best=max(res["aggregators"],key=lambda k:res["aggregators"][k]["celebdf_dev_cv_mean"])
res["best_aggregator"]=best; res["vs_R0_0.6967"]=round(res["aggregators"][best]["celebdf_dev_cv_mean"]-0.6967,4)
json.dump(res,open(f"{OUT}/trackE_E2_dev.json","w"),indent=1)
print(f"\nbest aggregator: {best} -> {res['aggregators'][best]['celebdf_dev_cv_mean']} (Δ vs R0 0.6967: {res['vs_R0_0.6967']:+.4f})")
print(f"saved {OUT}/trackE_E2_dev.json (commit {commit()})")
