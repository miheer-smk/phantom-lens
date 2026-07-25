#!/usr/bin/env python3
"""Track D Batch-2 — M/Q/R/T DEV evaluation (FF++ train/val + celebdf_dev ONLY; sealed untouched).
Each family (M,Q,R,T) and ALL-combined vs 53-D baseline. In-dist (per-manip + pooled) & celebdf_dev.
Bootstrap ΔAUC p-values -> Holm across all family×axis tests. Thresholds: in-dist +0.005, cross +0.03.
Per-feature Cohen's d. Train-only imputer, seed 42, locked LightGBM. Logs dev-eval ledger additions.
"""
import os, sys, json, subprocess, datetime
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore"); sys.path.insert(0, "src")
from protocol import make_splits
import roi_config as RC
from extract_trackD_MQRT import M_F, Q_F, R_F, T_F
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
SEED=42; F="features"; TD=f"{F}/trackD"; OUT="results_clean"
G1=RC.CANDIDATE_GROUPS["G1_mouth_instability"]; MAN=["deepfakes","face2face","faceswap","neuraltextures"]
FAMS={"M":M_F,"Q":Q_F,"R":R_F,"T":T_F,"ALL":M_F+Q_F+R_F+T_F}
def bn(p): return os.path.basename(str(p))
def commit():
    try: return subprocess.check_output(["git","rev-parse","--short","HEAD"],text=True).strip()
    except: return "nogit"
def LGBM(): return lgb.LGBMClassifier(n_estimators=200,max_depth=6,learning_rate=0.05,num_leaves=31,
    min_child_samples=20,class_weight="balanced",random_state=SEED,verbose=-1,n_jobs=1,deterministic=True,force_row_wise=True)
def holm(pv):
    idx=np.argsort(pv); out=np.empty(len(pv)); m=len(pv); prev=0
    for r,i in enumerate(idx): prev=max(prev,(m-r)*pv[i]); out[i]=min(prev,1.0)
    return out

allF=M_F+Q_F+R_F+T_F
mq_ff=pd.read_csv(f"{TD}/MQRT_ffpp_trainval.csv"); mq_ff["_b"]=mq_ff.video_path.map(bn)
mq_cd=pd.read_csv(f"{TD}/MQRT_celebdf_dev.csv"); mq_cd["_b"]=mq_cd.video_path.map(bn)
def ff_set(name):
    o=pd.read_csv(f"{F}/ffpp_{'original' if name=='real' else name}_c23.csv")
    r=pd.read_csv(f"{F}/roi_{'original' if name=='real' else name}_c23.csv")
    o["_b"]=o.video_path.map(bn); r["_b"]=r.video_path.map(bn)
    m=o.merge(r[["_b"]+G1],on="_b",how="inner").merge(mq_ff[["_b"]+allF],on="_b",how="inner")
    return make_splits(m)
FF={k:ff_set(k) for k in ["real"]+MAN}
FC=sorted([c for c in FF["real"].columns if c[:2] in ("s_","t_")]); C53=FC+G1
def imp(df,cols):
    d=df.copy()
    for c in cols: d[c]=pd.to_numeric(d[c],errors="coerce").replace([np.inf,-np.inf],np.nan)
    d[cols]=d[cols].fillna(d.loc[d.partition=="train",cols].median()); return d
FF={k:imp(v,C53+allF) for k,v in FF.items()}
cd=pd.read_csv(f"{F}/celebdf_features.csv"); cd["_b"]=cd.video_path.map(bn)
g1=pd.read_csv(f"{TD}/G1_celebdf_dev.csv"); g1["_b"]=g1.video_path.map(bn)
CD=cd.merge(g1[["_b"]+G1],on="_b",how="inner").merge(mq_cd[["_b"]+allF],on="_b",how="inner")
trall=pd.concat([FF["real"][FF["real"].partition=="train"]]+[FF[m][FF[m].partition=="train"] for m in MAN],ignore_index=True)
med=trall[C53+allF].median(); CDi=CD.copy()
for c in C53+allF: CDi[c]=pd.to_numeric(CDi[c],errors="coerce").replace([np.inf,-np.inf],np.nan).fillna(med[c])
print(f"celebdf_dev merged: {len(CDi)} (real {int((CDi.label==0).sum())}, fake {int((CDi.label==1).sum())})",flush=True)

def fit_pred(tr,te,cols):
    sc=StandardScaler().fit(tr[cols].values); m=LGBM().fit(sc.transform(tr[cols].values),tr.label.values.astype(int))
    return m.predict_proba(sc.transform(te[cols].values))[:,1]
def boot_p_ci(y,pa,pb,n=2000,s=SEED):
    rng=np.random.RandomState(s); d=[]
    for _ in range(n):
        i=rng.randint(0,len(y),len(y))
        if len(np.unique(y[i]))<2: continue
        d.append(roc_auc_score(y[i],pa[i])-roc_auc_score(y[i],pb[i]))
    d=np.array(d); p=2*min((d<=0).mean(),(d>=0).mean())
    return round(float(np.percentile(d,2.5)),4),round(float(np.percentile(d,97.5)),4),float(max(p,1e-4))

# pooled in-dist predictions (per-manip models concatenated) for 53 and each family
def indist_preds(cols):
    ys=[]; ps=[]
    for m in MAN:
        tr=pd.concat([FF["real"][FF["real"].partition=="train"],FF[m][FF[m].partition=="train"]],ignore_index=True)
        va=pd.concat([FF["real"][FF["real"].partition=="val"],  FF[m][FF[m].partition=="val"]],  ignore_index=True)
        ps.append(fit_pred(tr,va,cols)); ys.append(va.label.values.astype(int))
    return np.concatenate(ys),np.concatenate(ps)
yI,p53I=indist_preds(C53)
yc=CDi.label.values.astype(int); p53C=fit_pred(trall,CDi,C53)
auc53I=roc_auc_score(yI,p53I); auc53C=roc_auc_score(yc,p53C)

rows=[]; pvals=[]; keys=[]
for fam,feats in FAMS.items():
    cols=C53+feats
    _,pI=indist_preds(cols); aI=roc_auc_score(yI,pI); loI,hiI,ppI=boot_p_ci(yI,pI,p53I)
    pC=fit_pred(trall,CDi,cols); aC=roc_auc_score(yc,pC); loC,hiC,ppC=boot_p_ci(yc,pC,p53C)
    rows.append(dict(family=fam,n_feats=len(feats),
        indist_auc=round(aI,4),indist_delta=round(aI-auc53I,4),indist_ci=[loI,hiI],indist_p=ppI,
        cross_auc=round(aC,4),cross_delta=round(aC-auc53C,4),cross_ci=[loC,hiC],cross_p=ppC))
    pvals+= [ppI,ppC]; keys+=[(fam,"indist"),(fam,"cross")]
ph=holm(np.array(pvals))
for k,p in zip(keys,ph):
    fam,ax=k
    for r in rows:
        if r["family"]==fam: r[f"{ax}_p_holm"]=round(float(p),4)

def cohen(df,f):
    a=df[df.label==0][f].astype(float); b=df[df.label==1][f].astype(float)
    sp=np.sqrt(((len(a)-1)*a.var()+(len(b)-1)*b.var())/max(len(a)+len(b)-2,1)); return round(float((b.mean()-a.mean())/(sp+1e-9)),3)
valpool=pd.concat([FF["real"][FF["real"].partition=="val"]]+[FF[m][FF[m].partition=="val"] for m in MAN],ignore_index=True)
dcoh={f:dict(val_d=cohen(valpool,f),celebdf_dev_d=cohen(CDi,f)) for f in allF}

res=dict(provenance=dict(script="exp_trackD_MQRT_eval.py",git_commit=commit(),seed=SEED,date=datetime.date.today().isoformat(),
    axis_dev_only=True,sealed_touched=False,thresholds=dict(indist=0.005,cross=0.03),correction="Holm across 10 family×axis tests"),
    baseline_53D=dict(indist_pooled=round(auc53I,4),celebdf_dev=round(auc53C,4)),families=rows,per_feature_cohens_d=dcoh)
json.dump(res,open(f"{OUT}/trackD_MQRT_dev.json","w"),indent=1)
print("="*82);print("TRACK D — M/Q/R/T DEV (53-D baseline: in-dist %.4f, celebdf_dev %.4f)"%(auc53I,auc53C));print("="*82)
print(f"{'fam':4s} {'in-dist Δ':>10s} {'p_holm':>7s} | {'cross Δ':>9s} {'p_holm':>7s}   verdict")
for r in rows:
    inc_i=r['indist_delta']>=0.005 and r.get('indist_p_holm',1)<0.05
    inc_c=r['cross_delta']>=0.03 and r.get('cross_p_holm',1)<0.05
    v="INCLUDE" if (inc_i or inc_c) else "reject"
    print(f"{r['family']:4s} {r['indist_delta']:+10.4f} {r.get('indist_p_holm',1):7.3f} | {r['cross_delta']:+9.4f} {r.get('cross_p_holm',1):7.3f}   {v}")
print(f"saved {OUT}/trackD_MQRT_dev.json (commit {commit()})")
