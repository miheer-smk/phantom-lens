#!/usr/bin/env python3
"""Track E — denser-sampling subset check. DEV; sealed=0. Does 100-frame extraction beat 60 on the SAME
subset? (The 0.63->0.70 gain was the representation; this tests whether frame-count adds on top.) Train
R0-RandomForest real-vs-manips on the FF++ subset, eval celebdf subset by identity-grouped 5-fold CV, at
60 frames (from plain_everyone_E3) vs 100 frames (new extraction). If clearly better, justify a full pass.
"""
import os, sys, json, subprocess, datetime, re
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore"); sys.path.insert(0,"src")
from protocol import make_splits
from extract_trackE_SBV import FEATS
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier
SEED=42; F="features"; TE=f"{F}/trackE"; OUT="results_clean"; MAN=["deepfakes","face2face","faceswap","neuraltextures"]
DIR={"deepfakes":"Deepfakes","face2face":"Face2Face","faceswap":"FaceSwap","neuraltextures":"NeuralTextures"}
def method(p):
    for m,d in DIR.items():
        if f"/{d}/" in p: return m
    return "real" if "youtube" in p else ("celebdf" if "Celeb-DF" in p else "?")
def commit():
    try: return subprocess.check_output(["git","rev-parse","--short","HEAD"],text=True).strip()
    except: return "nogit"
def RF(): return RandomForestClassifier(n_estimators=400,max_depth=8,min_samples_leaf=5,class_weight="balanced",random_state=SEED,n_jobs=-1)
sub=set(pd.read_csv("features/trackD/manifest_denser_subset.csv").video_path)
def evalset(df):
    df=df.copy(); df["src"]=df.video_path.map(method)
    for c in FEATS: df[c]=pd.to_numeric(df[c],errors="coerce").replace([np.inf,-np.inf],np.nan)
    ff=make_splits(df[df.src.isin(["real"]+MAN)].copy()); cd=df[df.src=="celebdf"].copy()
    med=ff[FEATS].median(); ff[FEATS]=ff[FEATS].fillna(med); cd[FEATS]=cd[FEATS].fillna(med)
    tr=pd.concat([ff[ff.src=="real"].assign(label=0)]+[ff[ff.src==m].assign(label=1) for m in MAN],ignore_index=True)
    sc=StandardScaler().fit(tr[FEATS].values); m=RF().fit(sc.transform(tr[FEATS].values),tr.label.values.astype(int))
    p=m.predict_proba(sc.transform(cd[FEATS].values))[:,1]; y=cd.label.values.astype(int)
    ids=cd.video_path.map(lambda x:(re.findall(r"id(\d+)",str(x)) or [os.path.basename(str(x))])[0]).values
    a=[roc_auc_score(y[i],p[i]) for _,i in GroupKFold(5).split(p,y,ids) if len(np.unique(y[i]))>1]
    return round(float(np.mean(a)),4),round(float(np.std(a)),4),int(len(cd)),int(len(tr))
ev60=pd.read_csv(f"{TE}/plain_everyone_E3.csv"); ev60=ev60[ev60.video_path.isin(sub)]
ev100=pd.read_csv(f"{TE}/denser_subset_100.csv")
r60=evalset(ev60); r100=evalset(ev100)
res=dict(provenance=dict(script="exp_trackE_denser_compare.py",git_commit=commit(),seed=SEED,date=datetime.date.today().isoformat(),axis_dev_only=True,sealed_touched=False,subset=len(sub)),
    frames_60=dict(celebdf_cv=r60[:2],n_cd=r60[2],n_train=r60[3]),
    frames_100=dict(celebdf_cv=r100[:2],n_cd=r100[2],n_train=r100[3]),
    delta_100_minus_60=round(r100[0]-r60[0],4))
json.dump(res,open(f"{OUT}/trackE_denser_dev.json","w"),indent=1)
print("="*60);print("TRACK E — DENSER SAMPLING (subset; celebdf CV)");print("="*60)
print(f"   60 frames: {r60[0]:.4f} ±{r60[1]:.3f}  (n_cd={r60[2]}, n_train={r60[3]})")
print(f"  100 frames: {r100[0]:.4f} ±{r100[1]:.3f}  (n_cd={r100[2]}, n_train={r100[3]})")
print(f"  Δ(100-60): {r100[0]-r60[0]:+.4f}  -> {'trend continues, justify full pass' if r100[0]-r60[0]>0.01 else 'no meaningful gain, do NOT re-extract'}")
print(f"saved {OUT}/trackE_denser_dev.json (commit {commit()})")
