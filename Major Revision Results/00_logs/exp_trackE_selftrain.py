#!/usr/bin/env python3
"""Track E — pseudo-label SELF-TRAINING (transductive UDA). DEV only; sealed=0. Zero extraction.
We only tested feature-ALIGNMENT UDA (CORAL/subspace/quantile/std — all negative). Self-training adapts the
DECISION BOUNDARY instead: train on FF++ (R0, 196-D, RandomForest), pseudo-label the most-confident target
points, retrain including them, iterate. Uses NO target labels (labels only score AUC).
HONEST CV: identity-grouped 5-fold; within each fold, pseudo-label only the TRAINING folds' celebdf and
evaluate the held-out fold (the eval fold never trains on its own pseudo-labels). Sweep top-k% ∈ {10,20,30}%,
rounds=3. Reported as unsupervised TRANSDUCTIVE DA — distinct from the inductive zero-shot number.
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
ev=pd.read_csv(f"{TE}/plain_everyone_E3.csv"); ev["src"]=ev.video_path.map(method)
for c in FEATS: ev[c]=pd.to_numeric(ev[c],errors="coerce").replace([np.inf,-np.inf],np.nan)
ff=make_splits(ev[ev.src.isin(["real"]+MAN)].copy()); cd=ev[ev.src=="celebdf"].copy()
med=ff[ff.partition=="train"][FEATS].median(); ff[FEATS]=ff[FEATS].fillna(med); cd[FEATS]=cd[FEATS].fillna(med)
yc=cd.label.values.astype(int)
cd_ids=cd.video_path.map(lambda p:(re.findall(r"id(\d+)",str(p)) or [os.path.basename(str(p))])[0]).values
real_tr=ff[(ff.src=="real")&(ff.partition=="train")]
srcX=pd.concat([real_tr.assign(label=0)]+[ff[(ff.src==m)&(ff.partition=="train")].assign(label=1) for m in MAN],ignore_index=True)
sc=StandardScaler().fit(srcX[FEATS].values); Xs=sc.transform(srcX[FEATS].values); ys=srcX.label.values.astype(int)
Xcd_all=sc.transform(cd[FEATS].values)

def selftrain(pool_X, eval_X, topk, rounds=3):
    m=RF().fit(Xs,ys)
    for _ in range(rounds):
        p=m.predict_proba(pool_X)[:,1]; n=len(p); k=max(int(topk*n),1)
        order=np.argsort(p)
        pl_idx=np.r_[order[:k],order[-k:]]; pl_y=np.r_[np.zeros(k),np.ones(k)]   # most-confident real / fake
        Xt=np.vstack([Xs,pool_X[pl_idx]]); yt=np.r_[ys,pl_y]
        m=RF().fit(Xt,yt)
    return m.predict_proba(eval_X)[:,1]

# baseline R0 (no self-training) CV for reference
def cv_pred(predfn):
    aucs=[]
    for tri,tei in GroupKFold(5).split(Xcd_all,yc,cd_ids):
        pe=predfn(Xcd_all[tri],Xcd_all[tei])
        if len(np.unique(yc[tei]))>1: aucs.append(roc_auc_score(yc[tei],pe))
    return round(float(np.mean(aucs)),4),round(float(np.std(aucs)),4)
m0=RF().fit(Xs,ys); base=cv_pred(lambda pool,ev: m0.predict_proba(ev)[:,1])
res=dict(provenance=dict(script="exp_trackE_selftrain.py",git_commit=commit(),seed=SEED,date=datetime.date.today().isoformat(),
    axis_dev_only=True,sealed_touched=False,type="unsupervised transductive DA (self-training)",classifier="RandomForest_d8"),
    R0_no_selftrain_cv=list(base), sweep={})
print("="*66);print("TRACK E — PSEUDO-LABEL SELF-TRAINING (transductive; celebdf_dev CV)");print("="*66)
print(f"  R0 (no self-train):     {base[0]:.4f} ±{base[1]:.3f}")
for topk in (0.10,0.20,0.30):
    mn,st=cv_pred(lambda pool,ev,tk=topk: selftrain(pool,ev,tk))
    res["sweep"][f"top{int(topk*100)}pct"]=dict(celebdf_dev_cv_mean=mn,celebdf_dev_cv_std=st,delta_vs_R0=round(mn-base[0],4))
    print(f"  self-train top-{int(topk*100)}%:    {mn:.4f} ±{st:.3f}   Δvs R0 {mn-base[0]:+.4f}")
best=max(res["sweep"],key=lambda k:res["sweep"][k]["celebdf_dev_cv_mean"])
res["best"]=best; res["best_delta_vs_R0"]=res["sweep"][best]["delta_vs_R0"]
json.dump(res,open(f"{OUT}/trackE_selftrain_dev.json","w"),indent=1)
print(f"\nbest: {best} -> {res['sweep'][best]['celebdf_dev_cv_mean']} (Δ {res['best_delta_vs_R0']:+.4f}; threshold +0.03)")
print("NOTE: transductive (uses unlabeled target features) — report SEPARATELY from inductive zero-shot.")
print(f"saved {OUT}/trackE_selftrain_dev.json (commit {commit()})")
