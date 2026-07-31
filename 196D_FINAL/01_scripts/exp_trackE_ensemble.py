#!/usr/bin/env python3
"""Track E — per-manipulation ensemble. DEV only; sealed untouched. Zero extraction (reuses 196-D).
Train real-vs-{DF,F2F,FS,NT} SEPARATELY, average the 4 scores -> ensemble. Compare to R0 (real-vs-all).
Selection metric = identity-grouped 5-fold CV within celebdf_dev. Reports real/fake recall.
"""
import os, sys, json, subprocess, datetime, re
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore"); sys.path.insert(0,"src")
from protocol import make_splits
from extract_trackE_SBV import FEATS
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
import lightgbm as lgb
SEED=42; F="features"; TE=f"{F}/trackE"; OUT="results_clean"; MAN=["deepfakes","face2face","faceswap","neuraltextures"]
DIRMAP={"deepfakes":"Deepfakes","face2face":"Face2Face","faceswap":"FaceSwap","neuraltextures":"NeuralTextures"}
def method(p):
    for m,d in DIRMAP.items():
        if f"/{d}/" in p: return m
    if "original_sequences" in p or "/youtube/" in p: return "real"
    return "celebdf" if "Celeb-DF" in p else "?"
def commit():
    try: return subprocess.check_output(["git","rev-parse","--short","HEAD"],text=True).strip()
    except: return "nogit"
def LGBM(): return lgb.LGBMClassifier(n_estimators=300,max_depth=6,learning_rate=0.05,num_leaves=31,
    min_child_samples=20,class_weight="balanced",random_state=SEED,verbose=-1,n_jobs=1,deterministic=True,force_row_wise=True)
ev=pd.read_csv(f"{TE}/plain_everyone_E3.csv"); ev["src"]=ev.video_path.map(method)
for c in FEATS: ev[c]=pd.to_numeric(ev[c],errors="coerce").replace([np.inf,-np.inf],np.nan)
ff=make_splits(ev[ev.src.isin(["real"]+MAN)].copy()); cd=ev[ev.src=="celebdf"].copy()
tr_med=ff[ff.partition=="train"][FEATS].median()
ff[FEATS]=ff[FEATS].fillna(tr_med); cd[FEATS]=cd[FEATS].fillna(tr_med)
yc=cd.label.values.astype(int)
cd_ids=cd.video_path.map(lambda p:(re.findall(r"id(\d+)",str(p)) or [os.path.basename(str(p))])[0]).values
def cv(p):
    a=[roc_auc_score(yc[i],p[i]) for _,i in GroupKFold(5).split(p,yc,cd_ids) if len(np.unique(yc[i]))>1]
    return round(float(np.mean(a)),4),round(float(np.std(a)),4)
def rec(p,t=0.5):
    pr=(p>=t).astype(int); return round(float((pr[yc==0]==0).mean()),3),round(float((pr[yc==1]==1).mean()),3)
real_tr=ff[(ff.src=="real")&(ff.partition=="train")]
def fitpred(train_df,te):
    sc=StandardScaler().fit(train_df[FEATS].values); m=LGBM().fit(sc.transform(train_df[FEATS].values),train_df.label.values.astype(int))
    return m.predict_proba(sc.transform(te[FEATS].values))[:,1]

# R0 reference (real vs all manips)
r0_tr=pd.concat([real_tr.assign(label=0)]+[ff[(ff.src==m)&(ff.partition=="train")].assign(label=1) for m in MAN],ignore_index=True)
p_r0=fitpred(r0_tr,cd); r0m,r0s=cv(p_r0)
# per-manip models -> average on celebdf
per={}; scores=[]
for m in MAN:
    tr=pd.concat([real_tr.assign(label=0), ff[(ff.src==m)&(ff.partition=="train")].assign(label=1)],ignore_index=True)
    p=fitpred(tr,cd); per[m]=round(cv(p)[0],4); scores.append(p)
ens=np.mean(scores,axis=0); em,es=cv(ens); err,efr=rec(ens)
res=dict(provenance=dict(script="exp_trackE_ensemble.py",git_commit=commit(),seed=SEED,date=datetime.date.today().isoformat(),axis_dev_only=True,sealed_touched=False),
    R0_real_vs_all=dict(celebdf_dev_cv=[r0m,r0s]),
    per_manip_celebdf_dev_cv=per,
    ensemble_avg=dict(celebdf_dev_cv_mean=em,celebdf_dev_cv_std=es,real_recall=err,fake_recall=efr,delta_vs_R0=round(em-r0m,4)))
json.dump(res,open(f"{OUT}/trackE_ensemble_dev.json","w"),indent=1)
print("="*66);print("TRACK E — PER-MANIPULATION ENSEMBLE (celebdf_dev CV)");print("="*66)
print(f"  R0 real-vs-all:        {r0m:.4f} ±{r0s:.3f}")
print(f"  per-manip CV: "+"  ".join(f"{m}={per[m]:.4f}" for m in MAN))
print(f"  ENSEMBLE (avg of 4):   {em:.4f} ±{es:.3f}  Δvs R0 {em-r0m:+.4f}  | realRec {err} fakeRec {efr}")
print(f"saved {OUT}/trackE_ensemble_dev.json (commit {commit()})")
