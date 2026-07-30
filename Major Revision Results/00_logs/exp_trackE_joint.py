#!/usr/bin/env python3
"""Track E — JOINT model selection across TWO cross-dataset targets (winner's-curse fix). DEV; sealed=0.
After 55 celebdf_dev evals the argmax is inflated. Score every candidate on BOTH celebdf_dev CV (identity-grouped
5-fold) AND WildDeepfake AUC (independent second target, FF++-trained, inductive). Prefer configs strong on BOTH
and the rank ensemble. 196-D R0. Reports both columns for every candidate; ranks by mean of the two.
"""
import os, sys, json, subprocess, datetime, re
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore"); sys.path.insert(0, "src")
from protocol import make_splits
from extract_trackE_SBV import FEATS
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from scipy.stats import rankdata
import lightgbm as lgb
SEED=42; F="features"; TE=f"{F}/trackE"; OUT="results_clean"; MAN=["deepfakes","face2face","faceswap","neuraltextures"]
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
ff=make_splits(ev[ev.src.isin(["real"]+MAN)].copy()); cd=ev[ev.src=="celebdf"].copy()
med=ff[ff.partition=="train"][FEATS].median(); ff[FEATS]=ff[FEATS].fillna(med); cd[FEATS]=cd[FEATS].fillna(med)
yc=cd.label.values.astype(int); cd_ids=cd.video_path.map(lambda p:(re.findall(r"id(\d+)",str(p)) or [os.path.basename(str(p))])[0]).values
tr=pd.concat([ff[(ff.src=="real")&(ff.partition=="train")].assign(label=0)]+[ff[(ff.src==m)&(ff.partition=="train")].assign(label=1) for m in MAN],ignore_index=True)
sc=StandardScaler().fit(tr[FEATS].values); Xtr=sc.transform(tr[FEATS].values); ytr=tr.label.values.astype(int); Xcd=sc.transform(cd[FEATS].values)
wdf=pd.read_csv(f"{TE}/wdf_196d.csv")
for c in FEATS: wdf[c]=pd.to_numeric(wdf[c],errors="coerce")
wdf[FEATS]=wdf[FEATS].fillna(med); Xwdf=sc.transform(wdf[FEATS].values); ywdf=wdf.label.values.astype(int)
def cvauc(p):
    a=[roc_auc_score(yc[i],p[i]) for _,i in GroupKFold(5).split(p,yc,cd_ids) if len(np.unique(yc[i]))>1]
    return round(float(np.mean(a)),4)
def L(): return lgb.LGBMClassifier(n_estimators=300,learning_rate=0.05,num_leaves=31,min_child_samples=20,max_depth=6,class_weight="balanced",random_state=SEED,verbose=-1,n_jobs=-1,deterministic=True,force_row_wise=True)
models={"RF_d8":RandomForestClassifier(n_estimators=400,max_depth=8,min_samples_leaf=5,class_weight="balanced",random_state=SEED,n_jobs=-1),
        "ExtraTrees":ExtraTreesClassifier(n_estimators=600,max_depth=10,min_samples_leaf=4,class_weight="balanced",random_state=SEED,n_jobs=-1),
        "LGBM_d6":L()}
Pcd={}; Pwdf={}
for k,m in models.items():
    m.fit(Xtr,ytr); Pcd[k]=m.predict_proba(Xcd)[:,1]; Pwdf[k]=m.predict_proba(Xwdf)[:,1]
def rank(keys,P): return np.mean([rankdata(P[k]) for k in keys],axis=0)
res={"provenance":dict(script="exp_trackE_joint.py",git_commit=commit(),seed=SEED,date=datetime.date.today().isoformat(),axis_dev_only=True,sealed_touched=False,targets=["celebdf_dev_cv","wilddeepfake_auc"],n_wdf=int(len(wdf))),"candidates":{}}
print("="*70);print("TRACK E — JOINT SELECTION: celebdf_dev CV + WildDeepfake AUC");print("="*70)
print(f"  {'candidate':22s} {'celebdf_cv':>11s} {'wdf_auc':>9s} {'mean':>7s}")
def add(tag,pcd,pwdf):
    c=cvauc(pcd); w=round(roc_auc_score(ywdf,pwdf),4); res["candidates"][tag]=dict(celebdf_dev_cv=c,wilddeepfake_auc=w,mean=round((c+w)/2,4))
    print(f"  {tag:22s} {c:11.4f} {w:9.4f} {(c+w)/2:7.4f}")
for k in models: add(k,Pcd[k],Pwdf[k])
add("RF+ET_rank",rank(["RF_d8","ExtraTrees"],Pcd),rank(["RF_d8","ExtraTrees"],Pwdf))
add("RF+ET+LGBM_rank",rank(["RF_d8","ExtraTrees","LGBM_d6"],Pcd),rank(["RF_d8","ExtraTrees","LGBM_d6"],Pwdf))
best=max(res["candidates"],key=lambda k:res["candidates"][k]["mean"])
res["best_by_joint_mean"]=best
os.makedirs(OUT,exist_ok=True); json.dump(res,open(f"{OUT}/trackE_joint_dev.json","w"),indent=1)
print(f"\n  BEST by joint mean: {best} -> celebdf {res['candidates'][best]['celebdf_dev_cv']} | wdf {res['candidates'][best]['wilddeepfake_auc']}")
print(f"saved {OUT}/trackE_joint_dev.json (commit {commit()})")
