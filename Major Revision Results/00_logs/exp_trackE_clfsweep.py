#!/usr/bin/env python3
"""Track E — classifier / regularization sweep on the 196-D R0 representation. DEV only; sealed=0.
Hypothesis: lower-capacity / smoother models transfer cross-dataset better than the locked LightGBM
(GBDT splits on absolute thresholds, which shift under domain change; pre-revision RF>LGBM cross).
Sweep: LightGBM depth {2,3,4,6} + strong-reg; RandomForest (depth-capped); L2 LogReg (standardized);
RBF-SVM; calibrated avg-ensemble LR+RF+LGBM. Select by identity-grouped 5-fold CV on celebdf_dev.
Zero extraction (reuses plain_everyone_E3 196-D). Reports in-dist + celebdf_dev CV + real/fake recall.
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
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import lightgbm as lgb
SEED=42; F="features"; TE=f"{F}/trackE"; OUT="results_clean"; MAN=["deepfakes","face2face","faceswap","neuraltextures"]
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
yc=cd.label.values.astype(int)
cd_ids=cd.video_path.map(lambda p:(re.findall(r"id(\d+)",str(p)) or [os.path.basename(str(p))])[0]).values
real_tr=ff[(ff.src=="real")&(ff.partition=="train")]
tr=pd.concat([real_tr.assign(label=0)]+[ff[(ff.src==m)&(ff.partition=="train")].assign(label=1) for m in MAN],ignore_index=True)
sc=StandardScaler().fit(tr[FEATS].values)
Xtr=sc.transform(tr[FEATS].values); ytr=tr.label.values.astype(int)
Xcd=sc.transform(cd[FEATS].values)
val={m:pd.concat([ff[(ff.src=='real')&(ff.partition=='val')],ff[(ff.src==m)&(ff.partition=='val')]],ignore_index=True) for m in MAN}
def cv(p):
    a=[roc_auc_score(yc[i],p[i]) for _,i in GroupKFold(5).split(p,yc,cd_ids) if len(np.unique(yc[i]))>1]
    return round(float(np.mean(a)),4),round(float(np.std(a)),4)
def rec(p,t=0.5):
    pr=(p>=t).astype(int); return round(float((pr[yc==0]==0).mean()),3),round(float((pr[yc==1]==1).mean()),3)
def indist(pf):
    ys=[];ps=[]
    for m in MAN: ys.append(val[m].label.values.astype(int)); ps.append(pf(sc.transform(val[m][FEATS].values)))
    return round(roc_auc_score(np.concatenate(ys),np.concatenate(ps)),4)

def L(**k):
    d=dict(n_estimators=300,learning_rate=0.05,num_leaves=31,min_child_samples=20,class_weight="balanced",
           random_state=SEED,verbose=-1,n_jobs=1,deterministic=True,force_row_wise=True)
    d.update(k); return lgb.LGBMClassifier(**d)
models={}
for d in (2,3,4,6): models[f"LGBM_d{d}"]=L(max_depth=d)
models["LGBM_strongreg"]=L(max_depth=3,reg_alpha=5.0,reg_lambda=5.0,min_child_samples=50,num_leaves=15)
models["RandomForest_d8"]=RandomForestClassifier(n_estimators=400,max_depth=8,min_samples_leaf=5,class_weight="balanced",random_state=SEED,n_jobs=-1)
models["LogReg_L2"]=LogisticRegression(C=1.0,max_iter=2000,class_weight="balanced")
models["RBF_SVM"]=SVC(kernel="rbf",C=1.0,gamma="scale",probability=True,class_weight="balanced",random_state=SEED)

res={"provenance":dict(script="exp_trackE_clfsweep.py",git_commit=commit(),seed=SEED,date=datetime.date.today().isoformat(),axis_dev_only=True,sealed_touched=False,rep="196-D R0"),"models":{}}
fitted={}
print("="*74);print("TRACK E — CLASSIFIER SWEEP (196-D R0; select by celebdf_dev CV)");print("="*74)
print(f"{'model':18s} {'in-dist':>8s} {'cv_cross':>9s} {'±std':>6s} {'realRec':>8s} {'fakeRec':>8s}")
for name,m in models.items():
    m.fit(Xtr,ytr); fitted[name]=m
    pf=lambda X,mm=m: mm.predict_proba(X)[:,1]
    pcd=m.predict_proba(Xcd)[:,1]; cm,cs=cv(pcd); rr,fr=rec(pcd); ind=indist(pf)
    res["models"][name]=dict(indist_auc=ind,celebdf_dev_cv_mean=cm,celebdf_dev_cv_std=cs,real_recall=rr,fake_recall=fr)
    print(f"  {name:18s} {ind:8.4f} {cm:9.4f} {cs:6.3f} {rr:8.3f} {fr:8.3f}")
# calibrated avg-ensemble LR+RF+LGBM_d4
pe=np.mean([fitted["LogReg_L2"].predict_proba(Xcd)[:,1],fitted["RandomForest_d8"].predict_proba(Xcd)[:,1],fitted["LGBM_d4"].predict_proba(Xcd)[:,1]],axis=0)
cm,cs=cv(pe); rr,fr=rec(pe)
ind_e=indist(lambda X: np.mean([fitted["LogReg_L2"].predict_proba(X)[:,1],fitted["RandomForest_d8"].predict_proba(X)[:,1],fitted["LGBM_d4"].predict_proba(X)[:,1]],axis=0))
res["models"]["ensemble_LR_RF_LGBM"]=dict(indist_auc=ind_e,celebdf_dev_cv_mean=cm,celebdf_dev_cv_std=cs,real_recall=rr,fake_recall=fr)
print(f"  {'ens_LR+RF+LGBM':18s} {ind_e:8.4f} {cm:9.4f} {cs:6.3f} {rr:8.3f} {fr:8.3f}")
best=max(res["models"],key=lambda k:res["models"][k]["celebdf_dev_cv_mean"])
res["best_by_celebdf_dev_cv"]=best; res["vs_LGBM_d6_R0_0.6967"]=round(res["models"][best]["celebdf_dev_cv_mean"]-0.6967,4)
json.dump(res,open(f"{OUT}/trackE_clfsweep_dev.json","w"),indent=1)
print(f"\nBEST: {best} -> celebdf_dev CV {res['models'][best]['celebdf_dev_cv_mean']} (Δ vs LGBM_d6 R0 0.6967: {res['vs_LGBM_d6_R0_0.6967']:+.4f})")
print(f"saved {OUT}/trackE_clfsweep_dev.json (commit {commit()})")
