#!/usr/bin/env python3
"""Track E3 — Self-Blended-Video training regimes. DEV only; sealed untouched.
All features are 196-D E1-expanded from the SAME full_features pipeline (consistent, no sampling confound):
  everyone (real+manips+celebdf_dev): plain_everyone_E3.csv ; SBV: SBV_ffpp_train_j*.csv (label 1).
Regimes (train on FF++ TRAIN identities; test FF++ val per-manip + celebdf_dev):
  R0 real vs FF++ manips (current) | R1 real vs SBV (no real fakes) | R2 real vs SBV+manips (hybrid).
Winning regime chosen on celebdf_dev per the frozen rule. Bootstrap ΔAUC vs R0; seed 42, locked LGBM.
"""
import os, sys, json, subprocess, datetime, glob, re
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore"); sys.path.insert(0, "src")
from protocol import make_splits
from extract_trackE_SBV import FEATS
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
SEED=42; F="features"; TE=f"{F}/trackE"; OUT="results_clean"; MAN=["deepfakes","face2face","faceswap","neuraltextures"]
def commit():
    try: return subprocess.check_output(["git","rev-parse","--short","HEAD"],text=True).strip()
    except: return "nogit"
def LGBM(): return lgb.LGBMClassifier(n_estimators=300,max_depth=6,learning_rate=0.05,num_leaves=31,
    min_child_samples=20,class_weight="balanced",random_state=SEED,verbose=-1,n_jobs=1,deterministic=True,force_row_wise=True)
DIRMAP={"deepfakes":"Deepfakes","face2face":"Face2Face","faceswap":"FaceSwap","neuraltextures":"NeuralTextures"}
def method(p):  # FF++ dir names are mixed-case (Face2Face etc.) -> explicit map, NOT .capitalize()
    for m,d in DIRMAP.items():
        if f"/{d}/" in p: return m
    if "original_sequences" in p or "/youtube/" in p: return "real"
    return "celebdf" if "Celeb-DF" in p else "?"

ev=pd.read_csv(f"{TE}/plain_everyone_E3.csv")
for c in FEATS: ev[c]=pd.to_numeric(ev[c],errors="coerce").replace([np.inf,-np.inf],np.nan)
ev["src"]=ev.video_path.map(method)
ff=ev[ev.src.isin(["real"]+MAN)].copy(); ff=make_splits(ff)
cd=ev[ev.src=="celebdf"].copy()
sbv=pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{TE}/SBV_ffpp_train_j*.csv"))],ignore_index=True)
for c in FEATS: sbv[c]=pd.to_numeric(sbv[c],errors="coerce").replace([np.inf,-np.inf],np.nan)
# train-only imputation medians (FF++ train)
tr_med=ff[ff.partition=="train"][FEATS].median()
for df in (ff,cd,sbv):
    df[FEATS]=df[FEATS].fillna(tr_med)
print(f"loaded: FF++ {len(ff)} (real {int((ff.src=='real').sum())}), celebdf_dev {len(cd)}, SBV {len(sbv)}",flush=True)

real_tr=ff[(ff.src=="real")&(ff.partition=="train")]; real_va=ff[(ff.src=="real")&(ff.partition=="val")]
man_tr={m:ff[(ff.src==m)&(ff.partition=="train")] for m in MAN}; man_va={m:ff[(ff.src==m)&(ff.partition=="val")] for m in MAN}
yc=cd.label.values.astype(int)
def fit(train_df):
    sc=StandardScaler().fit(train_df[FEATS].values); m=LGBM().fit(sc.transform(train_df[FEATS].values),train_df.label.values.astype(int))
    return sc,m
def score(sc,m,df): return m.predict_proba(sc.transform(df[FEATS].values))[:,1]
def indist(sc,m):
    ys=[];ps=[]
    for mm in MAN:
        va=pd.concat([real_va,man_va[mm]],ignore_index=True); ps.append(score(sc,m,va)); ys.append(va.label.values.astype(int))
    return roc_auc_score(np.concatenate(ys),np.concatenate(ps))
def boot(y,pa,pb,n=2000,s=SEED):
    rng=np.random.RandomState(s); d=[]
    for _ in range(n):
        i=rng.randint(0,len(y),len(y))
        if len(np.unique(y[i]))<2: continue
        d.append(roc_auc_score(y[i],pa[i])-roc_auc_score(y[i],pb[i]))
    d=np.array(d); return round(float(np.percentile(d,2.5)),4),round(float(np.percentile(d,97.5)),4),float(max(2*min((d<=0).mean(),(d>=0).mean()),1e-4))

regimes={
 "R0_real_vs_ffpp": pd.concat([real_tr.assign(label=0)]+[man_tr[m].assign(label=1) for m in MAN],ignore_index=True),
 "R1_real_vs_SBV":  pd.concat([real_tr.assign(label=0), sbv.assign(label=1)],ignore_index=True),
 "R2_hybrid":       pd.concat([real_tr.assign(label=0), sbv.assign(label=1)]+[man_tr[m].assign(label=1) for m in MAN],ignore_index=True),
}
res={"provenance":dict(script="exp_trackE_E3_eval.py",git_commit=commit(),seed=SEED,date=datetime.date.today().isoformat(),
     axis_dev_only=True,sealed_touched=False,feature_set="196-D E1-expanded full_features (consistent)",n_SBV=int(len(sbv))),
     "regimes":{}}
# baseline R0 predictions for bootstrap reference
sc0,m0=fit(regimes["R0_real_vs_ffpp"]); pc0=score(sc0,m0,cd); r0_in=indist(sc0,m0); r0_cr=roc_auc_score(yc,pc0)
for name,tr in regimes.items():
    sc,m=fit(tr); ind=indist(sc,m); pc=score(sc,m,cd); cr=roc_auc_score(yc,pc)
    lo,hi,p=boot(yc,pc,pc0) if name!="R0_real_vs_ffpp" else (0.0,0.0,1.0)
    res["regimes"][name]=dict(n_train=int(len(tr)),indist_auc=round(ind,4),celebdf_dev_auc=round(cr,4),
        cross_delta_vs_R0=round(cr-r0_cr,4),cross_ci_vs_R0=[lo,hi],cross_p_vs_R0=round(p,4))
best=max(res["regimes"],key=lambda k: res["regimes"][k]["celebdf_dev_auc"])
res["winning_regime_by_celebdf_dev"]=best
json.dump(res,open(f"{OUT}/trackE_E3_dev.json","w"),indent=1)
print("="*74);print("TRACK E3 — SBV training regimes (DEV; feature set 196-D consistent)");print("="*74)
print(f"{'regime':20s} {'in-dist':>8s} {'celebdf_dev':>12s} {'Δcross vs R0':>13s}")
for name,r in res["regimes"].items():
    print(f"  {name:20s} {r['indist_auc']:8.4f} {r['celebdf_dev_auc']:12.4f} {r['cross_delta_vs_R0']:+13.4f} (p{r['cross_p_vs_R0']})")
print(f"\nwinning regime (celebdf_dev): {best} -> {res['regimes'][best]['celebdf_dev_auc']}")
print("NOTE: all celebdf_dev numbers are DEV, not sealed (sealed budget 1, unspent).")
print(f"saved {OUT}/trackE_E3_dev.json (commit {commit()})")
