#!/usr/bin/env python3
"""POST-FREEZE DESCRIPTIVE — frozen 196-D ensemble on FF++ TEST (in-distribution number). No selection/tuning.
Frozen RF+ExtraTrees+LGBM rank ensemble trained on FF++ train, scored on FF++ test (700 videos, official split).
Reports AUC + bootstrap 95% CI + real/fake recall (prob-avg @0.5)."""
import os, sys, json, subprocess, datetime, re
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore"); sys.path.insert(0, "src")
from protocol import make_splits
from extract_trackE_SBV import FEATS
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from scipy.stats import rankdata
import lightgbm as lgb
SEED=42; TE="features/trackE"; OUT="results_clean"; MAN=["deepfakes","face2face","faceswap","neuraltextures"]
DIR={"deepfakes":"Deepfakes","face2face":"Face2Face","faceswap":"FaceSwap","neuraltextures":"NeuralTextures"}
def method(p):
    for m,d in DIR.items():
        if f"/{d}/" in p: return m
    return "real" if "youtube" in p else ("celebdf" if "Celeb-DF" in p else "?")
def commit():
    try: return subprocess.check_output(["git","rev-parse","--short","HEAD"],text=True).strip()
    except: return "nogit"
ev=pd.read_csv(f"{TE}/plain_everyone_E3.csv"); ev["src"]=ev.video_path.map(method)
for c in FEATS: ev[c]=pd.to_numeric(ev[c],errors="coerce").replace([np.inf,-np.inf],np.nan)
ff=make_splits(ev[ev.src.isin(["real"]+MAN)].copy()); med=ff[ff.partition=="train"][FEATS].median()
tr=pd.concat([ff[(ff.src=="real")&(ff.partition=="train")].assign(label=0)]+
             [ff[(ff.src==m)&(ff.partition=="train")].assign(label=1) for m in MAN],ignore_index=True)
test=pd.read_csv(f"{TE}/plain_ffpp_test.csv")
for c in FEATS: test[c]=pd.to_numeric(test[c],errors="coerce")
yft=test.label.values.astype(int)
ids=test.video_path.map(lambda p: os.path.basename(str(p)).split("_")[0]).values
def L(): return lgb.LGBMClassifier(n_estimators=300,learning_rate=0.05,num_leaves=31,min_child_samples=20,max_depth=6,class_weight="balanced",random_state=SEED,verbose=-1,n_jobs=-1,deterministic=True,force_row_wise=True)
sc=StandardScaler().fit(tr[FEATS].fillna(med).values); Xtr=sc.transform(tr[FEATS].fillna(med).values); ytr=tr.label.values.astype(int)
Xte=sc.transform(test[FEATS].fillna(med).values)
models={"RF":RandomForestClassifier(n_estimators=400,max_depth=8,min_samples_leaf=5,class_weight="balanced",random_state=SEED,n_jobs=-1),
        "ET":ExtraTreesClassifier(n_estimators=600,max_depth=10,min_samples_leaf=4,class_weight="balanced",random_state=SEED,n_jobs=-1),
        "LGBM":L()}
P=[]
for m in models.values(): m.fit(Xtr,ytr); P.append(m.predict_proba(Xte)[:,1])
P=np.array(P); rank=np.mean([rankdata(p) for p in P],axis=0); prob=P.mean(axis=0)
auc=round(roc_auc_score(yft,rank),4)
rng=np.random.RandomState(SEED); uids=np.unique(ids); a=[]
for _ in range(2000):
    s=rng.choice(uids,len(uids),replace=True); mk=np.isin(ids,s)
    if len(np.unique(yft[mk]))>1: a.append(roc_auc_score(yft[mk],rank[mk]))
lo,hi=round(float(np.percentile(a,2.5)),4),round(float(np.percentile(a,97.5)),4)
pr=(prob>=0.5).astype(int); rr=round(float((pr[yft==0]==0).mean()),3); fr=round(float((pr[yft==1]==1).mean()),3)
res=dict(provenance=dict(script="exp_trackE_ffpptest_eval.py",git_commit=commit(),seed=SEED,date=datetime.date.today().isoformat(),
    kind="POST-FREEZE DESCRIPTIVE in-distribution",classifier="RF+ExtraTrees+LGBM rank ensemble",rep="196-D"),
    ffpp_test=dict(auc=auc,ci95=[lo,hi],n=int(len(test)),reals=int((yft==0).sum()),fakes=int((yft==1).sum()),real_recall=rr,fake_recall=fr))
os.makedirs(OUT,exist_ok=True); json.dump(res,open(f"{OUT}/POSTFREEZE_ffpptest.json","w"),indent=1)
print("="*60);print("POST-FREEZE — frozen 196-D ensemble on FF++ TEST (in-dist)");print("="*60)
print(f"  FF++ TEST AUC = {auc}  95% CI [{lo}, {hi}]  (n={len(test)}, {int((yft==0).sum())}r/{int((yft==1).sum())}f)")
print(f"  real recall {rr} | fake recall {fr}")
print(f"saved {OUT}/POSTFREEZE_ffpptest.json")
