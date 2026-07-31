#!/usr/bin/env python3
"""Per-video predictions of the FROZEN 196-D ensemble (descriptive; for the Zenodo deposit). No model changes.
Trains the frozen RF+ET+LGBM rank ensemble on FF++ train, emits per-video P(fake) for sealed celebdf_test and
FF++ test. rank_pfake = per-model-prob rank-averaged then min-max scaled to [0,1] (monotone; AUC-equivalent);
prob_pfake = mean predicted probability. Output: 196D_FINAL/03_results/per_video_predictions.csv"""
import os, sys, re
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore"); sys.path.insert(0,"src")
from protocol import make_splits
from extract_trackE_SBV import FEATS
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from scipy.stats import rankdata
import lightgbm as lgb
SEED=42; TE="features/trackE"; MAN=["deepfakes","face2face","faceswap","neuraltextures"]
DIR={"deepfakes":"Deepfakes","face2face":"Face2Face","faceswap":"FaceSwap","neuraltextures":"NeuralTextures"}
def method(p):
    for m,d in DIR.items():
        if f"/{d}/" in p: return m
    return "real" if "youtube" in p else ("celebdf" if "Celeb-DF" in p else "?")
ev=pd.read_csv(f"{TE}/plain_everyone_E3.csv"); ev["src"]=ev.video_path.map(method)
for c in FEATS: ev[c]=pd.to_numeric(ev[c],errors="coerce").replace([np.inf,-np.inf],np.nan)
ff=make_splits(ev[ev.src.isin(["real"]+MAN)].copy()); med=ff[ff.partition=="train"][FEATS].median()
tr=pd.concat([ff[(ff.src=="real")&(ff.partition=="train")].assign(label=0)]+[ff[(ff.src==m)&(ff.partition=="train")].assign(label=1) for m in MAN],ignore_index=True)
sc=StandardScaler().fit(tr[FEATS].fillna(med).values); ytr=tr.label.values.astype(int); Xtr=sc.transform(tr[FEATS].fillna(med).values)
def L(): return lgb.LGBMClassifier(n_estimators=300,learning_rate=0.05,num_leaves=31,min_child_samples=20,max_depth=6,class_weight="balanced",random_state=SEED,verbose=-1,n_jobs=-1,deterministic=True,force_row_wise=True)
models=[RandomForestClassifier(n_estimators=400,max_depth=8,min_samples_leaf=5,class_weight="balanced",random_state=SEED,n_jobs=-1),
        ExtraTreesClassifier(n_estimators=600,max_depth=10,min_samples_leaf=4,class_weight="balanced",random_state=SEED,n_jobs=-1),L()]
for m in models: m.fit(Xtr,ytr)
def preds(df,split):
    X=sc.transform(df[FEATS].fillna(med).values); P=np.array([m.predict_proba(X)[:,1] for m in models])
    r=np.mean([rankdata(p) for p in P],axis=0); r=(r-r.min())/(r.max()-r.min()+1e-9)
    return pd.DataFrame({"video_path":df.video_path.values,"split":split,"true_label":df.label.values.astype(int),
                         "rank_pfake":np.round(r,6),"prob_pfake":np.round(P.mean(axis=0),6)})
test=pd.read_csv(f"{TE}/plain_celebdf_test.csv"); test["label"]=test.get("label",1)
for c in FEATS: test[c]=pd.to_numeric(test[c],errors="coerce")
ftest=pd.read_csv(f"{TE}/plain_ffpp_test.csv"); ftest["src"]=ftest.video_path.map(method); ftest["label"]=(ftest.src!="real").astype(int)
for c in FEATS: ftest[c]=pd.to_numeric(ftest[c],errors="coerce")
out=pd.concat([preds(test,"celebdf_test_SEALED"),preds(ftest,"ffpp_test")],ignore_index=True)
os.makedirs("196D_FINAL/03_results",exist_ok=True); out.to_csv("196D_FINAL/03_results/per_video_predictions.csv",index=False)
print(f"  per_video_predictions.csv: {len(out)} rows ({(out.split=='celebdf_test_SEALED').sum()} celebdf_test, {(out.split=='ffpp_test').sum()} ffpp_test)")
