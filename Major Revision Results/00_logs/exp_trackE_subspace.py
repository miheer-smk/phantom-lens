#!/usr/bin/env python3
"""Track E — RANDOM-SUBSPACE (feature-bagging) ensemble. DEV only; sealed=0. Zero extraction (196-D R0).
Hypothesis: feature bagging helps under domain shift by preventing reliance on any single domain-sensitive
feature — each member sees a random 50% of the 196-D, so no member can lean on one feature that shifts
celebdf<-FF++. Average RANKS across M members. Sweep M, base learner {RF, ExtraTrees}, and combine with the
model-diversity ensemble. Select by identity-grouped 5-fold celebdf_dev CV; report real/fake recall.
n_jobs limited (3) to be polite to a co-running extraction. Bar cross +0.03 (Holm at freeze).
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
SEED=42; F="features"; TE=f"{F}/trackE"; OUT="results_clean"; MAN=["deepfakes","face2face","faceswap","neuraltextures"]; NJ=3
DIR={"deepfakes":"Deepfakes","face2face":"Face2Face","faceswap":"FaceSwap","neuraltextures":"NeuralTextures"}
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
yc=cd.label.values.astype(int); cd_ids=cd.video_path.map(lambda p:(re.findall(r"id(\d+)",str(p)) or [os.path.basename(str(p))])[0]).values
tr=pd.concat([ff[(ff.src=="real")&(ff.partition=="train")].assign(label=0)]+[ff[(ff.src==m)&(ff.partition=="train")].assign(label=1) for m in MAN],ignore_index=True)
sc=StandardScaler().fit(tr[FEATS].values); Xtr=sc.transform(tr[FEATS].values); ytr=tr.label.values.astype(int); Xcd=sc.transform(cd[FEATS].values)
def cv(p):
    a=[roc_auc_score(yc[i],p[i]) for _,i in GroupKFold(5).split(p,yc,cd_ids) if len(np.unique(yc[i]))>1]
    return round(float(np.mean(a)),4),round(float(np.std(a)),4)
def rec(p,t=0.5):
    pr=(p>=t).astype(int); return round(float((pr[yc==0]==0).mean()),3),round(float((pr[yc==1]==1).mean()),3)
def base_model(kind):
    if kind=="RF": return RandomForestClassifier(n_estimators=200,max_depth=8,min_samples_leaf=5,class_weight="balanced",random_state=SEED,n_jobs=NJ)
    return ExtraTreesClassifier(n_estimators=300,max_depth=10,min_samples_leaf=4,class_weight="balanced",random_state=SEED,n_jobs=NJ)
def subspace(kind, M, frac=0.5):
    rng=np.random.RandomState(SEED); D=Xtr.shape[1]; k=int(frac*D); ranks=np.zeros(len(Xcd))
    for i in range(M):
        cols=rng.choice(D,k,replace=False)
        m=base_model(kind); m.set_params(random_state=SEED+i); m.fit(Xtr[:,cols],ytr)
        ranks+=rankdata(m.predict_proba(Xcd[:,cols])[:,1])
    return ranks/M
res={"provenance":dict(script="exp_trackE_subspace.py",git_commit=commit(),seed=SEED,date=datetime.date.today().isoformat(),axis_dev_only=True,sealed_touched=False,rep="196-D R0",n_jobs=NJ),"configs":{}}
print("="*72);print("TRACK E — RANDOM-SUBSPACE (feature-bagging) ENSEMBLE (celebdf_dev CV)");print("="*72)
print(f"{'config':24s} {'cross_cv':>9s} {'±std':>6s} {'realRec':>8s} {'fakeRec':>8s}")
best=("",-1)
for kind in ("RF","ExtraTrees"):
    for M in (15,30):
        p=subspace(kind,M); cm,cs=cv(p); rr,fr=rec(p)
        tag=f"{kind}_M{M}_50pct"; res["configs"][tag]=dict(celebdf_dev_cv_mean=cm,celebdf_dev_cv_std=cs,real_recall=rr,fake_recall=fr)
        print(f"  {tag:24s} {cm:9.4f} {cs:6.3f} {rr:8.3f} {fr:8.3f}")
        if cm>best[1]: best=(tag,cm)
res["best"]=best[0]; res["best_cv"]=best[1]; res["delta_vs_RF_0.7018"]=round(best[1]-0.7018,4); res["delta_vs_ens_0.7125"]=round(best[1]-0.7125,4)
os.makedirs(OUT,exist_ok=True); json.dump(res,open(f"{OUT}/trackE_subspace_dev.json","w"),indent=1)
print(f"\n  BEST: {best[0]} -> {best[1]:.4f} (Δ vs RF 0.7018: {res['delta_vs_RF_0.7018']:+.4f} | vs ensemble 0.7125: {res['delta_vs_ens_0.7125']:+.4f})")
print(f"saved {OUT}/trackE_subspace_dev.json (commit {commit()})")
