#!/usr/bin/env python3
"""STEP 2 — POST-FREEZE DESCRIPTIVE per-manipulation FF++ test (comparability with the paper's mean-of-4).
No tuning/selection/model changes. Frozen RF+ET+LGBM rank ensemble, FF++-train only. On plain_ffpp_test.csv:
 - 196-D and 53-D: per-manip AUC (real vs each of DF/F2F/FS/NT) + mean-of-4 + bootstrap CIs, and POOLED AUC.
Emits the 2x2 comparability table (pooled vs mean-of-per-manip, for both models). celebdf untouched."""
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
LAB={"deepfakes":"DF","face2face":"F2F","faceswap":"FS","neuraltextures":"NT"}
F53=FEATS[:53]; F196=FEATS
def method(p):
    for m,d in DIR.items():
        if f"/{d}/" in p: return m
    return "real" if "youtube" in p else "?"
def commit():
    try: return subprocess.check_output(["git","rev-parse","--short","HEAD"],text=True).strip()
    except: return "nogit"
ev=pd.read_csv(f"{TE}/plain_everyone_E3.csv"); ev["src"]=ev.video_path.map(method)
for c in F196: ev[c]=pd.to_numeric(ev[c],errors="coerce").replace([np.inf,-np.inf],np.nan)
ff=make_splits(ev[ev.src.isin(["real"]+MAN)].copy()); med=ff[ff.partition=="train"][F196].median()
tr=pd.concat([ff[(ff.src=="real")&(ff.partition=="train")].assign(label=0)]+
             [ff[(ff.src==m)&(ff.partition=="train")].assign(label=1) for m in MAN],ignore_index=True)
test=pd.read_csv(f"{TE}/plain_ffpp_test.csv"); test["src"]=test.video_path.map(method)
for c in F196: test[c]=pd.to_numeric(test[c],errors="coerce")
def L(): return lgb.LGBMClassifier(n_estimators=300,learning_rate=0.05,num_leaves=31,min_child_samples=20,max_depth=6,class_weight="balanced",random_state=SEED,verbose=-1,n_jobs=-1,deterministic=True,force_row_wise=True)
def frozen_scores(cols):
    sc=StandardScaler().fit(tr[cols].fillna(med[cols]).values); ytr=tr.label.values.astype(int)
    Xtr=sc.transform(tr[cols].fillna(med[cols]).values); Xte=sc.transform(test[cols].fillna(med[cols]).values)
    models=[RandomForestClassifier(n_estimators=400,max_depth=8,min_samples_leaf=5,class_weight="balanced",random_state=SEED,n_jobs=-1),
            ExtraTreesClassifier(n_estimators=600,max_depth=10,min_samples_leaf=4,class_weight="balanced",random_state=SEED,n_jobs=-1),L()]
    P=[]
    for m in models: m.fit(Xtr,ytr); P.append(m.predict_proba(Xte)[:,1])
    return np.mean([rankdata(p) for p in P],axis=0)   # rank ensemble
def boot(y,s,n=2000):
    rng=np.random.RandomState(SEED); a=[]
    for _ in range(n):
        idx=rng.randint(0,len(y),len(y))
        if len(np.unique(y[idx]))>1: a.append(roc_auc_score(y[idx],s[idx]))
    return round(float(np.percentile(a,2.5)),4),round(float(np.percentile(a,97.5)),4)
reals=(test.src=="real").values
def per_manip(cols):
    s=frozen_scores(cols); out={}; pm=[]
    for m in MAN:
        mask=reals|(test.src==m).values; y=(test.src[mask]==m).astype(int).values
        auc=roc_auc_score(y,s[mask]); lo,hi=boot(y,s[mask]); out[LAB[m]]=dict(auc=round(auc,4),ci95=[lo,hi]); pm.append(auc)
    yp=(test.src!="real").astype(int).values; pooled=round(roc_auc_score(yp,s),4); plo,phi=boot(yp,s)
    return dict(per_manip=out, mean_of_4=round(float(np.mean(pm)),4), pooled=dict(auc=pooled,ci95=[plo,phi]), n=int(len(test)))
r196=per_manip(F196); r53=per_manip(F53)
res=dict(provenance=dict(script="exp_trackE_permanip.py",git_commit=commit(),seed=SEED,date=datetime.date.today().isoformat(),
    kind="POST-FREEZE DESCRIPTIVE per-manip FF++ test",classifier="RF+ET+LGBM rank ensemble"),
    frozen_196D=r196, model_53D=r53)
os.makedirs(OUT,exist_ok=True); json.dump(res,open(f"{OUT}/POSTFREEZE_permanip.json","w"),indent=1)
print("="*72);print("STEP 2 — FF++ TEST per-manip vs pooled (frozen ensemble; post-freeze descriptive)");print("="*72)
for tag,r in [("196-D",r196),("53-D",r53)]:
    pm=" ".join(f"{k} {v['auc']:.3f}" for k,v in r["per_manip"].items())
    print(f"  {tag}: per-manip [{pm}] | mean-of-4 {r['mean_of_4']:.4f} | pooled {r['pooled']['auc']:.4f} {r['pooled']['ci95']}")
print("\n  2x2 COMPARABILITY TABLE (AUC):")
print(f"  {'model':8s} {'pooled':>10s} {'mean-of-4':>10s}")
print(f"  {'196-D':8s} {r196['pooled']['auc']:10.4f} {r196['mean_of_4']:10.4f}")
print(f"  {'53-D':8s} {r53['pooled']['auc']:10.4f} {r53['mean_of_4']:10.4f}")
print(f"saved {OUT}/POSTFREEZE_permanip.json (commit {commit()})")
