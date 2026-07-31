#!/usr/bin/env python3
"""POST-FREEZE DESCRIPTIVE comparison on the ALREADY-UNSEALED celebdf_test half. No selection, no tuning, no
model changes — the frozen model is fixed; this only DESCRIBES how the locked 50-D / 53-D baselines and the
frozen 196-D E1-expanded rep compare on identical sealed-test data, with paired DeLong significance.
Classifier held CONSTANT (frozen RF+ExtraTrees+LGBM rank ensemble) across all three reps so the only difference
is the representation. celebdf_test is already unsealed (budget already spent); this is descriptive re-analysis.
Reports: AUC + identity-grouped bootstrap 95% CI + real/fake recall (prob-avg @0.5) for 50-D/53-D/196-D, and
DeLong z,p for 196-D vs 50-D and 196-D vs 53-D on the shared test set.
"""
import os, sys, json, subprocess, datetime, re
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore"); sys.path.insert(0, "src")
from protocol import make_splits
from extract_trackE_SBV import FEATS
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from scipy.stats import rankdata, norm
import lightgbm as lgb
SEED=42; TE="features/trackE"; OUT="results_clean"; MAN=["deepfakes","face2face","faceswap","neuraltextures"]
DIR={"deepfakes":"Deepfakes","face2face":"Face2Face","faceswap":"FaceSwap","neuraltextures":"NeuralTextures"}
F50=FEATS[:50]; F53=FEATS[:53]; F196=FEATS
def method(p):
    for m,d in DIR.items():
        if f"/{d}/" in p: return m
    return "real" if "youtube" in p else ("celebdf" if "Celeb-DF" in p else "?")
def commit():
    try: return subprocess.check_output(["git","rev-parse","--short","HEAD"],text=True).strip()
    except: return "nogit"
# ---- DeLong (fast, Sun & Xu 2014) ----
def _midrank(x):
    J=np.argsort(x); Z=x[J]; N=len(x); T=np.zeros(N); i=0
    while i<N:
        j=i
        while j<N and Z[j]==Z[i]: j+=1
        T[i:j]=0.5*(i+j-1)+1; i=j
    T2=np.empty(N); T2[J]=T; return T2
def delong(y,s1,s2):
    order=np.argsort(-y,kind="stable"); m=int(y.sum()); n=len(y)-m
    preds=np.vstack((s1,s2))[:,order]; k=2
    tx=np.empty([k,m]);ty=np.empty([k,n]);tz=np.empty([k,m+n])
    for r in range(k):
        tx[r]=_midrank(preds[r,:m]); ty[r]=_midrank(preds[r,m:]); tz[r]=_midrank(preds[r])
    aucs=tz[:,:m].sum(axis=1)/m/n-(m+1.0)/2.0/n
    v01=(tz[:,:m]-tx)/n; v10=1.0-(tz[:,m:]-ty)/m
    cov=np.cov(v01)/m+np.cov(v10)/n
    var=cov[0,0]+cov[1,1]-2*cov[0,1]
    z=(aucs[0]-aucs[1])/np.sqrt(var+1e-15); p=2*(1-norm.cdf(abs(z)))
    return float(aucs[0]),float(aucs[1]),float(z),float(p)
# ---- data ----
ev=pd.read_csv(f"{TE}/plain_everyone_E3.csv"); ev["src"]=ev.video_path.map(method)
for c in F196: ev[c]=pd.to_numeric(ev[c],errors="coerce").replace([np.inf,-np.inf],np.nan)
ff=make_splits(ev[ev.src.isin(["real"]+MAN)].copy())
med=ff[ff.partition=="train"][F196].median()
tr=pd.concat([ff[(ff.src=="real")&(ff.partition=="train")].assign(label=0)]+
             [ff[(ff.src==m)&(ff.partition=="train")].assign(label=1) for m in MAN],ignore_index=True)
test=pd.read_csv(f"{TE}/plain_celebdf_test.csv"); test["label"]=test.get("label",1)
for c in F196: test[c]=pd.to_numeric(test[c],errors="coerce")
yct=test.label.values.astype(int)
ids=test.video_path.map(lambda p:(re.findall(r"id(\d+)",str(p)) or [os.path.basename(str(p))])[0]).values
def L(): return lgb.LGBMClassifier(n_estimators=300,learning_rate=0.05,num_leaves=31,min_child_samples=20,max_depth=6,class_weight="balanced",random_state=SEED,verbose=-1,n_jobs=-1,deterministic=True,force_row_wise=True)
def fit_score(cols):
    sc=StandardScaler().fit(tr[cols].fillna(med[cols]).values)
    Xtr=sc.transform(tr[cols].fillna(med[cols]).values); ytr=tr.label.values.astype(int)
    Xte=sc.transform(test[cols].fillna(med[cols]).values)
    models={"RF":RandomForestClassifier(n_estimators=400,max_depth=8,min_samples_leaf=5,class_weight="balanced",random_state=SEED,n_jobs=-1),
            "ET":ExtraTreesClassifier(n_estimators=600,max_depth=10,min_samples_leaf=4,class_weight="balanced",random_state=SEED,n_jobs=-1),
            "LGBM":L()}
    P=[]
    for m in models.values(): m.fit(Xtr,ytr); P.append(m.predict_proba(Xte)[:,1])
    P=np.array(P)
    rank=np.mean([rankdata(p) for p in P],axis=0)   # frozen model = rank-avg (for AUC)
    prob=P.mean(axis=0)                              # prob-avg (for recall @0.5)
    return rank, prob
def boot_ci(y,score,ids,n=2000):
    uids=np.unique(ids); rng=np.random.RandomState(SEED); a=[]
    for _ in range(n):
        s=rng.choice(uids,len(uids),replace=True); mk=np.isin(ids,s)
        if len(np.unique(y[mk]))>1: a.append(roc_auc_score(y[mk],score[mk]))
    return round(float(np.percentile(a,2.5)),4),round(float(np.percentile(a,97.5)),4)
def recall(prob,t=0.5):
    pr=(prob>=t).astype(int); return round(float((pr[yct==0]==0).mean()),3),round(float((pr[yct==1]==1).mean()),3)
res={"provenance":dict(script="exp_trackE_postfreeze_compare.py",git_commit=commit(),seed=SEED,date=datetime.date.today().isoformat(),
    kind="POST-FREEZE DESCRIPTIVE (no selection after unseal; classifier held constant across reps)",
    classifier="RF+ExtraTrees+LGBM_d6 rank ensemble",test="celebdf_test (2273)"),"reps":{}}
scores={}
print("="*74);print("POST-FREEZE DESCRIPTIVE — 50-D vs 53-D vs 196-D on celebdf_test (frozen ensemble)");print("="*74)
print(f"  {'rep':8s} {'AUC':>7s} {'95% CI':>16s} {'realRec':>8s} {'fakeRec':>8s}")
for tag,cols in [("50-D",F50),("53-D",F53),("196-D",F196)]:
    rank,prob=fit_score(cols); scores[tag]=rank
    auc=round(roc_auc_score(yct,rank),4); lo,hi=boot_ci(yct,rank,ids); rr,fr=recall(prob)
    res["reps"][tag]=dict(auc=auc,ci95=[lo,hi],real_recall=rr,fake_recall=fr,n_features=len(cols))
    print(f"  {tag:8s} {auc:7.4f}   [{lo:.3f},{hi:.3f}]  {rr:8.3f} {fr:8.3f}")
print("  --- paired DeLong (shared celebdf_test) ---")
res["delong"]={}
for base in ("50-D","53-D"):
    a196,ab,z,p=delong(yct,scores["196-D"],scores[base])
    res["delong"][f"196D_vs_{base}"]=dict(auc_196=round(a196,4),auc_base=round(ab,4),delta=round(a196-ab,4),z=round(z,3),p_value=float(f"{p:.3g}"))
    print(f"  196-D vs {base}: Δ{a196-ab:+.4f}  z={z:.2f}  p={p:.3g}  {'(sig)' if p<0.05 else '(n.s.)'}")
os.makedirs(OUT,exist_ok=True); json.dump(res,open(f"{OUT}/POSTFREEZE_compare.json","w"),indent=1)
print(f"saved {OUT}/POSTFREEZE_compare.json (commit {commit()})")
