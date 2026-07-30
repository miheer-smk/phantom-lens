#!/usr/bin/env python3
"""Track E — Test-Time Augmentation eval. DEV only; sealed=0. Train FF++ (RF and rank ensemble), score each
celebdf_dev video as (a) base = original 196-D, (b) TTA-N = mean predicted prob over {original + N augmented}.
Identity-grouped 5-fold celebdf_dev CV. Reports base vs TTA-2 vs TTA-3 for RF and the RF+ET+LGBM rank ensemble.
Usage: exp_trackE_TTA_eval.py <tta_csv> [everyone_csv]
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
TTA_CSV=sys.argv[1] if len(sys.argv)>1 else f"{TE}/tta_celebdf_dev.csv"
EVERYONE=sys.argv[2] if len(sys.argv)>2 else f"{TE}/plain_everyone_E3.csv"
def method(p):
    for m,d in DIR.items():
        if f"/{d}/" in p: return m
    return "real" if "youtube" in p else ("celebdf" if "Celeb-DF" in p else "?")
def commit():
    try: return subprocess.check_output(["git","rev-parse","--short","HEAD"],text=True).strip()
    except: return "nogit"
ev=pd.read_csv(EVERYONE); ev["src"]=ev.video_path.map(method)
for c in FEATS: ev[c]=pd.to_numeric(ev[c],errors="coerce").replace([np.inf,-np.inf],np.nan)
ff=make_splits(ev[ev.src.isin(["real"]+MAN)].copy()); cd=ev[ev.src=="celebdf"].copy()
med=ff[ff.partition=="train"][FEATS].median(); ff[FEATS]=ff[FEATS].fillna(med); cd[FEATS]=cd[FEATS].fillna(med)
tr=pd.concat([ff[(ff.src=="real")&(ff.partition=="train")].assign(label=0)]+[ff[(ff.src==m)&(ff.partition=="train")].assign(label=1) for m in MAN],ignore_index=True)
sc=StandardScaler().fit(tr[FEATS].values); Xtr=sc.transform(tr[FEATS].values); ytr=tr.label.values.astype(int)
# models
def L(): return lgb.LGBMClassifier(n_estimators=300,learning_rate=0.05,num_leaves=31,min_child_samples=20,max_depth=6,class_weight="balanced",random_state=SEED,verbose=-1,n_jobs=-1,deterministic=True,force_row_wise=True)
models={"RF":RandomForestClassifier(n_estimators=400,max_depth=8,min_samples_leaf=5,class_weight="balanced",random_state=SEED,n_jobs=-1),
        "ET":ExtraTreesClassifier(n_estimators=600,max_depth=10,min_samples_leaf=4,class_weight="balanced",random_state=SEED,n_jobs=-1),
        "LGBM":L()}
for m in models.values(): m.fit(Xtr,ytr)
def prob(X):  # per-model probs on a feature matrix
    return {k:m.predict_proba(X)[:,1] for k,m in models.items()}
# base: original celebdf features (aligned order)
cd=cd.reset_index(drop=True); yc=cd.label.values.astype(int)
cd_ids=cd.video_path.map(lambda p:(re.findall(r"id(\d+)",str(p)) or [os.path.basename(str(p))])[0]).values
Xbase=sc.transform(cd[FEATS].fillna(med).values); pbase=prob(Xbase)
# TTA augmented versions
tta=pd.read_csv(TTA_CSV)
for c in FEATS: tta[c]=pd.to_numeric(tta[c],errors="coerce")
tta[FEATS]=tta[FEATS].fillna(med)
# per-model prob for each augmented row, indexed by (video_path, aug_idx)
Xaug=sc.transform(tta[FEATS].values); paug=prob(Xaug)
tta_idx=tta[["video_path","aug_idx"]].copy()
for k in models: tta_idx[f"p_{k}"]=paug[k]
def cv(p):
    a=[roc_auc_score(yc[i],p[i]) for _,i in GroupKFold(5).split(p,yc,cd_ids) if len(np.unique(yc[i]))>1]
    return round(float(np.mean(a)),4),round(float(np.std(a)),4)
def rankens(pd_):  # rank-avg RF+ET+LGBM
    return np.mean([rankdata(pd_["RF"]),rankdata(pd_["ET"]),rankdata(pd_["LGBM"])],axis=0)
def tta_prob(model_key, n):
    # mean over {base + first n augmented} per video, aligned to cd order
    scores=[]
    for i,vp in enumerate(cd.video_path.values):
        aug=tta_idx[(tta_idx.video_path==vp)&(tta_idx.aug_idx<n)][f"p_{model_key}"].values
        vals=np.concatenate([[pbase[model_key][i]],aug]) if len(aug) else np.array([pbase[model_key][i]])
        scores.append(vals.mean())
    return np.array(scores)
res={"provenance":dict(script="exp_trackE_TTA_eval.py",git_commit=commit(),seed=SEED,date=datetime.date.today().isoformat(),axis_dev_only=True,sealed_touched=False,tta=TTA_CSV),"results":{}}
print("="*66);print("TRACK E — TEST-TIME AUGMENTATION (celebdf_dev CV)");print("="*66)
n_aug_avail=int(tta.aug_idx.max())+1
for tag in ("RF","ensemble"):
    if tag=="RF":
        b=cv(pbase["RF"]); rows={"base":b}
        for n in (2,3):
            if n<=n_aug_avail: rows[f"TTA{n}"]=cv(tta_prob("RF",n))
    else:
        b=cv(rankens(pbase)); rows={"base":b}
        for n in (2,3):
            if n<=n_aug_avail:
                # rank-ensemble of TTA-averaged per-model probs
                pe=np.mean([rankdata(tta_prob("RF",n)),rankdata(tta_prob("ET",n)),rankdata(tta_prob("LGBM",n))],axis=0)
                rows[f"TTA{n}"]=cv(pe)
    res["results"][tag]=rows
    line=" | ".join(f"{k} {v[0]:.4f}±{v[1]:.3f}" for k,v in rows.items())
    print(f"  {tag:9s}: {line}")
os.makedirs(OUT,exist_ok=True); json.dump(res,open(f"{OUT}/trackE_TTA_dev.json","w"),indent=1)
print(f"saved {OUT}/trackE_TTA_dev.json (commit {commit()})")
