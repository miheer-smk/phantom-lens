#!/usr/bin/env python3
"""Pillar-ONLY analysis (complement to remove-one-out) — standalone power of each domain.
For each pillar, train on ONLY that pillar's features; evaluate per dataset. AUC + bootstrap CI.
Together with pillar_ablation.csv this is the complete ablation for Reviewers 1/3/5.
Identity-disjoint; column-selection only; 50-D vector unchanged.
"""
import os,sys,json,subprocess,datetime
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore"); sys.path.insert(0,"src")
from protocol import make_splits
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
SEED=42; F="features"; OUT="results_clean"
PILLARS=json.load(open("splits/pillar_map.json"))
MAN={"Deepfakes":"ffpp_deepfakes_c23.csv","Face2Face":"ffpp_face2face_c23.csv",
     "FaceSwap":"ffpp_faceswap_c23.csv","NeuralTextures":"ffpp_neuraltextures_c23.csv"}
real=pd.read_csv(f"{F}/ffpp_original_c23.csv"); cd=pd.read_csv(f"{F}/celebdf_features.csv")
FC=sorted([c for c in real.columns if c[:2] in ("s_","t_")])
def clean(df):
    d=df.copy(); d[FC]=d[FC].replace([np.inf,-np.inf],np.nan)
    for c in FC: d[c]=d[c].fillna(d[c].median())
    return d
real=make_splits(clean(real)); MANd={k:make_splits(clean(pd.read_csv(f"{F}/{v}"))) for k,v in MAN.items()}; cd=clean(cd)
def LGBM(): return lgb.LGBMClassifier(n_estimators=200,max_depth=6,learning_rate=0.05,num_leaves=31,
    min_child_samples=20,class_weight="balanced",random_state=SEED,verbose=-1,n_jobs=4)
def boot(y,p,n=1500,s=SEED):
    rng=np.random.RandomState(s); b=[]
    for _ in range(n):
        i=rng.randint(0,len(y),len(y))
        if len(np.unique(y[i]))>1: b.append(roc_auc_score(y[i],p[i]))
    return float(np.percentile(b,2.5)),float(np.percentile(b,97.5))
def commit():
    try: return subprocess.check_output(["git","rev-parse","--short","HEAD"],text=True).strip()
    except: return "nogit"
def fit_auc(cols,tr,te):
    Xtr=tr[cols].values.astype(float); ytr=tr['label'].values.astype(int)
    Xte=te[cols].values.astype(float); yte=te['label'].values.astype(int)
    sc=StandardScaler().fit(Xtr); clf=LGBM(); clf.fit(sc.transform(Xtr),ytr)
    p=clf.predict_proba(sc.transform(Xte))[:,1]
    return roc_auc_score(yte,p),boot(yte,p),yte

rows=[]
def do(name,tr,te):
    full,_,_=fit_auc(FC,tr,te)
    for pil,feats in PILLARS.items():
        a,(lo,hi),_=fit_auc(feats,tr,te)
        rows.append(dict(dataset=name,pillar=pil,n_feats=len(feats),pillar_only_auc=round(a,4),
            ci_lo=round(lo,4),ci_hi=round(hi,4),full50_auc=round(full,4)))
        print(f"  {name:14s} {pil:22s} only={a:.4f} CI[{lo:.3f},{hi:.3f}] (full50={full:.3f})",flush=True)
print("="*72+"\nPILLAR-ONLY (standalone AUC per domain, identity-disjoint)\n"+"="*72)
for m,mdf in MANd.items():
    do(m,pd.concat([real[real.partition=="train"],mdf[mdf.partition=="train"]],ignore_index=True),
         pd.concat([real[real.partition=="test"], mdf[mdf.partition=="test"]], ignore_index=True))
trf=pd.concat([real[real.partition=="train"]]+[MANd[m][MANd[m].partition=="train"] for m in MAN],ignore_index=True)
do("CelebDF",trf,cd)
df=pd.DataFrame(rows); df.to_csv(f"{OUT}/pillar_only.csv",index=False)
json.dump(dict(provenance=dict(script="Major Revision Results/00_logs/pillar_only.py",git_commit=commit(),
    seed=SEED,date=datetime.date.today().isoformat(),protocol="identity-disjoint; train on single pillar only"),
    rows=rows),open(f"{OUT}/pillar_only.json","w"),indent=1)
print(f"\nWrote {OUT}/pillar_only.csv ({len(rows)} rows) commit {commit()}")
