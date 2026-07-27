#!/usr/bin/env python3
"""Track E — STRONG-MEMBER ensemble + ExtraTrees. DEV only; sealed=0. Zero extraction (196-D R0).
The clfsweep naive avg ensemble FAILED (0.684) because weak LogReg/SVM dragged it down. Retest with only
strong tree members, and add ExtraTrees (more randomized splits than RF -> often better cross-dataset variance),
and RANK-averaging (more robust than prob-averaging for AUC). n_jobs limited (4) to not starve the 100-frame pass.
Select by identity-grouped 5-fold celebdf_dev CV; report real/fake recall. Bar cross +0.03 (Holm at freeze).
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
NJ=4
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
yc=cd.label.values.astype(int)
cd_ids=cd.video_path.map(lambda p:(re.findall(r"id(\d+)",str(p)) or [os.path.basename(str(p))])[0]).values
tr=pd.concat([ff[(ff.src=="real")&(ff.partition=="train")].assign(label=0)]+[ff[(ff.src==m)&(ff.partition=="train")].assign(label=1) for m in MAN],ignore_index=True)
sc=StandardScaler().fit(tr[FEATS].values); Xtr=sc.transform(tr[FEATS].values); ytr=tr.label.values.astype(int); Xcd=sc.transform(cd[FEATS].values)
def cv(p):
    a=[roc_auc_score(yc[i],p[i]) for _,i in GroupKFold(5).split(p,yc,cd_ids) if len(np.unique(yc[i]))>1]
    return round(float(np.mean(a)),4),round(float(np.std(a)),4)
def rec(p,t=0.5):
    pr=(p>=t).astype(int); return round(float((pr[yc==0]==0).mean()),3),round(float((pr[yc==1]==1).mean()),3)
def L(**k):
    d=dict(n_estimators=300,learning_rate=0.05,num_leaves=31,min_child_samples=20,class_weight="balanced",random_state=SEED,verbose=-1,n_jobs=NJ,deterministic=True,force_row_wise=True,max_depth=6); d.update(k); return lgb.LGBMClassifier(**d)
members={
 "RF_d8": RandomForestClassifier(n_estimators=400,max_depth=8,min_samples_leaf=5,class_weight="balanced",random_state=SEED,n_jobs=NJ),
 "ExtraTrees": ExtraTreesClassifier(n_estimators=600,max_depth=10,min_samples_leaf=4,class_weight="balanced",random_state=SEED,n_jobs=NJ),
 "LGBM_d6": L(),
}
P={}  # celebdf probs per member
res={"provenance":dict(script="exp_trackE_ens2.py",git_commit=commit(),seed=SEED,date=datetime.date.today().isoformat(),axis_dev_only=True,sealed_touched=False,rep="196-D R0",n_jobs=NJ),"members":{},"ensembles":{}}
print("="*70);print("TRACK E — STRONG-MEMBER ENSEMBLE + ExtraTrees (196-D; celebdf_dev CV)");print("="*70)
for name,m in members.items():
    m.fit(Xtr,ytr); p=m.predict_proba(Xcd)[:,1]; P[name]=p; cm,cs=cv(p); rr,fr=rec(p)
    res["members"][name]=dict(celebdf_dev_cv_mean=cm,celebdf_dev_cv_std=cs,real_recall=rr,fake_recall=fr)
    print(f"  {name:16s} cross {cm:.4f} ±{cs:.3f} | realRec {rr} fakeRec {fr}")
def rankavg(keys): return np.mean([rankdata(P[k]) for k in keys],axis=0)
def probavg(keys): return np.mean([P[k] for k in keys],axis=0)
combos=[("RF+ET",["RF_d8","ExtraTrees"]),("RF+ET+LGBM",["RF_d8","ExtraTrees","LGBM_d6"])]
print("  --- ensembles ---")
for tag,keys in combos:
    for kind,fn in (("prob",probavg),("rank",rankavg)):
        pe=fn(keys); cm,cs=cv(pe); rr,fr=rec(pe)
        res["ensembles"][f"{tag}_{kind}"]=dict(celebdf_dev_cv_mean=cm,celebdf_dev_cv_std=cs,real_recall=rr,fake_recall=fr)
        print(f"  {tag+'_'+kind:16s} cross {cm:.4f} ±{cs:.3f} | realRec {rr} fakeRec {fr}")
allc={**{k:v['celebdf_dev_cv_mean'] for k,v in res['members'].items()},**{k:v['celebdf_dev_cv_mean'] for k,v in res['ensembles'].items()}}
best=max(allc,key=allc.get); res["best"]=best; res["best_cv"]=allc[best]; res["delta_vs_RF_0.7018"]=round(allc[best]-0.7018,4)
os.makedirs(OUT,exist_ok=True); json.dump(res,open(f"{OUT}/trackE_ens2_dev.json","w"),indent=1)
print(f"\n  BEST: {best} -> {allc[best]:.4f} (Δ vs RF_d8 0.7018: {res['delta_vs_RF_0.7018']:+.4f}; bar +0.03)")
print(f"saved {OUT}/trackE_ens2_dev.json (commit {commit()})")
