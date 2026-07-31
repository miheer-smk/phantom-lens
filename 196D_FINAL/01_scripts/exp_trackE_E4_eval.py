#!/usr/bin/env python3
"""Track E4 eval — LoG frequency features stacked on the 196-D R0 rep, RandomForest classifier (new best).
DEV only; sealed=0. 196-D vs 196+18-D. Select by identity-grouped 5-fold CV on celebdf_dev; report both axes
+ real/fake recall. Threshold: +0.005 in-dist / +0.03 cross (Holm across the full ledger, applied at freeze).
"""
import os, sys, json, subprocess, datetime, re
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore"); sys.path.insert(0,"src")
from protocol import make_splits
from extract_trackE_SBV import FEATS
from extract_trackE_E4 import E4_FEATURES
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
def bn(p): return os.path.basename(str(p))
ev=pd.read_csv(f"{TE}/plain_everyone_E3.csv"); ev["src"]=ev.video_path.map(method)
e4=pd.read_csv(f"{TE}/E4_everyone.csv")
ev=ev.merge(e4[["video_path"]+E4_FEATURES],on="video_path",how="inner")
ALL=FEATS+E4_FEATURES
for c in ALL: ev[c]=pd.to_numeric(ev[c],errors="coerce").replace([np.inf,-np.inf],np.nan)
ff=make_splits(ev[ev.src.isin(["real"]+MAN)].copy()); cd=ev[ev.src=="celebdf"].copy()
med=ff[ff.partition=="train"][ALL].median(); ff[ALL]=ff[ALL].fillna(med); cd[ALL]=cd[ALL].fillna(med)
yc=cd.label.values.astype(int)
cd_ids=cd.video_path.map(lambda p:(re.findall(r"id(\d+)",str(p)) or [bn(p)])[0]).values
real_tr=ff[(ff.src=="real")&(ff.partition=="train")]
tr=pd.concat([real_tr.assign(label=0)]+[ff[(ff.src==m)&(ff.partition=="train")].assign(label=1) for m in MAN],ignore_index=True)
val={m:pd.concat([ff[(ff.src=='real')&(ff.partition=='val')],ff[(ff.src==m)&(ff.partition=='val')]],ignore_index=True) for m in MAN}
def cv(p):
    a=[roc_auc_score(yc[i],p[i]) for _,i in GroupKFold(5).split(p,yc,cd_ids) if len(np.unique(yc[i]))>1]
    return round(float(np.mean(a)),4),round(float(np.std(a)),4)
def rec(p,t=0.5):
    pr=(p>=t).astype(int); return round(float((pr[yc==0]==0).mean()),3),round(float((pr[yc==1]==1).mean()),3)
def run(cols):
    sc=StandardScaler().fit(tr[cols].values); m=RF().fit(sc.transform(tr[cols].values),tr.label.values.astype(int))
    ys=[];ps=[]
    for mm in MAN: ys.append(val[mm].label.values.astype(int)); ps.append(m.predict_proba(sc.transform(val[mm][cols].values))[:,1])
    ind=round(roc_auc_score(np.concatenate(ys),np.concatenate(ps)),4)
    pc=m.predict_proba(sc.transform(cd[cols].values))[:,1]; cm,cs=cv(pc); rr,fr=rec(pc)
    return dict(indist_auc=ind,celebdf_dev_cv_mean=cm,celebdf_dev_cv_std=cs,real_recall=rr,fake_recall=fr)
base=run(FEATS); stk=run(FEATS+E4_FEATURES)
res=dict(provenance=dict(script="exp_trackE_E4_eval.py",git_commit=commit(),seed=SEED,date=datetime.date.today().isoformat(),
    axis_dev_only=True,sealed_touched=False,classifier="RandomForest_d8",n_E4=len(E4_FEATURES)),
    base_196D=base, plus_E4_214D=stk,
    delta_indist=round(stk["indist_auc"]-base["indist_auc"],4), delta_cross=round(stk["celebdf_dev_cv_mean"]-base["celebdf_dev_cv_mean"],4))
json.dump(res,open(f"{OUT}/trackE_E4_dev.json","w"),indent=1)
print("="*70);print("TRACK E4 — LoG frequency stacked on 196-D (RandomForest; celebdf_dev CV)");print("="*70)
print(f"  196-D base:   in-dist {base['indist_auc']:.4f} | cross CV {base['celebdf_dev_cv_mean']:.4f} ±{base['celebdf_dev_cv_std']:.3f} | realRec {base['real_recall']} fakeRec {base['fake_recall']}")
print(f"  +E4 (214-D):  in-dist {stk['indist_auc']:.4f} | cross CV {stk['celebdf_dev_cv_mean']:.4f} ±{stk['celebdf_dev_cv_std']:.3f} | realRec {stk['real_recall']} fakeRec {stk['fake_recall']}")
print(f"  Δ: in-dist {res['delta_indist']:+.4f} | cross {res['delta_cross']:+.4f}  (thresholds +0.005 / +0.03)")
print(f"saved {OUT}/trackE_E4_dev.json (commit {commit()})")
