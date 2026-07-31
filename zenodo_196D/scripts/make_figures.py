#!/usr/bin/env python3
"""Generate publication figures (descriptive; no model changes). Reproduces frozen-ensemble scores on the
already-extracted sealed celebdf_test for ROC curves, plus a trade-off plot and a negative-results bar chart.
Run from repo root: .venv/bin/python 196D_FINAL/01_scripts/make_figures.py"""
import os, sys, re
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore"); sys.path.insert(0,"src")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from protocol import make_splits
from extract_trackE_SBV import FEATS
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from scipy.stats import rankdata
import lightgbm as lgb
SEED=42; TE="features/trackE"; OUTD="196D_FINAL/05_figures"; MAN=["deepfakes","face2face","faceswap","neuraltextures"]
DIR={"deepfakes":"Deepfakes","face2face":"Face2Face","faceswap":"FaceSwap","neuraltextures":"NeuralTextures"}
os.makedirs(OUTD,exist_ok=True)
def method(p):
    for m,d in DIR.items():
        if f"/{d}/" in p: return m
    return "real" if "youtube" in p else ("celebdf" if "Celeb-DF" in p else "?")
ev=pd.read_csv(f"{TE}/plain_everyone_E3.csv"); ev["src"]=ev.video_path.map(method)
for c in FEATS: ev[c]=pd.to_numeric(ev[c],errors="coerce").replace([np.inf,-np.inf],np.nan)
ff=make_splits(ev[ev.src.isin(["real"]+MAN)].copy()); med=ff[ff.partition=="train"][FEATS].median()
tr=pd.concat([ff[(ff.src=="real")&(ff.partition=="train")].assign(label=0)]+[ff[(ff.src==m)&(ff.partition=="train")].assign(label=1) for m in MAN],ignore_index=True)
test=pd.read_csv(f"{TE}/plain_celebdf_test.csv"); test["label"]=test.get("label",1); yct=test.label.values.astype(int)
def L(): return lgb.LGBMClassifier(n_estimators=300,learning_rate=0.05,num_leaves=31,min_child_samples=20,max_depth=6,class_weight="balanced",random_state=SEED,verbose=-1,n_jobs=-1,deterministic=True,force_row_wise=True)
def score(cols):
    sc=StandardScaler().fit(tr[cols].fillna(med[cols]).values); ytr=tr.label.values.astype(int)
    Xtr=sc.transform(tr[cols].fillna(med[cols]).values); Xte=sc.transform(test[cols].fillna(med[cols]).values)
    P=[]
    for m in [RandomForestClassifier(n_estimators=400,max_depth=8,min_samples_leaf=5,class_weight="balanced",random_state=SEED,n_jobs=-1),
              ExtraTreesClassifier(n_estimators=600,max_depth=10,min_samples_leaf=4,class_weight="balanced",random_state=SEED,n_jobs=-1),L()]:
        m.fit(Xtr,ytr); P.append(m.predict_proba(Xte)[:,1])
    return np.mean([rankdata(p) for p in P],axis=0)
# ---- Fig 1: ROC on sealed celebdf_test ----
plt.figure(figsize=(5,5))
for cols,lab,c in [(FEATS[:50],"50-D",'#888'),(FEATS[:53],"53-D",'#3b7'),(FEATS,"196-D",'#e45')]:
    s=score(cols); fpr,tpr,_=roc_curve(yct,s); auc=roc_auc_score(yct,s)
    plt.plot(fpr,tpr,color=c,lw=2,label=f"{lab} (AUC {auc:.3f})")
plt.plot([0,1],[0,1],'k--',lw=1,alpha=.5); plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
plt.title("Sealed Celeb-DF-v2 test — ROC by representation"); plt.legend(loc="lower right"); plt.tight_layout()
plt.savefig(f"{OUTD}/fig1_roc_celebdf_test.png",dpi=200); plt.close()
# ---- Fig 2: in-dist vs cross trade-off ----
plt.figure(figsize=(5,5))
pts={"53-D":(0.8358,0.6830),"196-D":(0.8420,0.7133)}
for k,(x,y) in pts.items(): plt.scatter(x,y,s=80); plt.annotate(k,(x,y),textcoords="offset points",xytext=(6,6))
plt.plot([pts["53-D"][0],pts["196-D"][0]],[pts["53-D"][1],pts["196-D"][1]],'--',alpha=.4)
plt.xlabel("FF++ test (in-distribution, mean-of-4 AUC)"); plt.ylabel("Celeb-DF-v2 sealed test (cross AUC)")
plt.title("In-distribution vs cross-dataset"); plt.grid(alpha=.3); plt.tight_layout()
plt.savefig(f"{OUTD}/fig2_tradeoff.png",dpi=200); plt.close()
# ---- Fig 3: negative-results bar (Δcross by lever) ----
levers=[("E1 order-stats",0.0326,"win"),("rank ensemble",0.0107,"win"),("E4 LoG",0.0055,"neg"),("X2 drop-rPPG",0.0055,"neg"),
        ("random-subspace",0.0013,"neg"),("per-manip ens",0.0008,"neg"),("Q co-activation",0.0001,"neg"),("E5 temp-diff",-0.0017,"neg"),
        ("denser 100f",-0.003,"neg"),("TTA",-0.004,"neg"),("X4 fakes",-0.0098,"neg"),("Group H",-0.0137,"neg"),("M cardiac",-0.0141,"neg"),
        ("per-dom quantile",-0.0151,"neg"),("X4 reals",-0.020,"neg"),("self-train k10",-0.031,"neg"),("CORAL",-0.0396,"neg"),("subspace d20",-0.090,"neg")]
levers.sort(key=lambda x:x[1])
plt.figure(figsize=(7,6)); names=[l[0] for l in levers]; vals=[l[1] for l in levers]
cols=['#e45' if l[2]=="win" else ('#4a8' if l[1]>0 else '#c66') for l in levers]
plt.barh(names,vals,color=cols); plt.axvline(0,color='k',lw=.8); plt.axvline(0.03,color='#999',ls='--',lw=1)
plt.xlabel("Δ cross-dataset AUC (celebdf_dev) vs base"); plt.title("Track D/E levers (dashed = +0.03 inclusion bar)")
plt.tight_layout(); plt.savefig(f"{OUTD}/fig3_negative_results.png",dpi=200); plt.close()
print("saved:",os.listdir(OUTD))
