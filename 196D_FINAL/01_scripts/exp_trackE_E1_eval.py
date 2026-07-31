#!/usr/bin/env python3
"""Track E1 — richer temporal aggregation. DEV only; sealed untouched.
From persisted per-frame spatial values, replace each spatial feature's single mean with 11 order
statistics (mean,std,min,max,p10,p25,p75,p90,IQR,skew,kurtosis) -> 13x11 = 143 spatial-stat features.
Models: baseline 53-D; E1-replace (143 stats + 37 temporal + 3 G1); E1-additive (53-D + 143 stats).
Measure FF++ val per-manip + celebdf_dev. Bootstrap ΔAUC + Holm; thresholds +0.005 in-dist / +0.03 cross.
Train-only imputer, seed 42, locked LightGBM. Saves E1_aggstats_*.csv (committed).
"""
import os, sys, json, subprocess, datetime
import numpy as np, pandas as pd, warnings
from scipy.stats import skew, kurtosis
warnings.filterwarnings("ignore"); sys.path.insert(0, "src")
from protocol import make_splits
from sealed import celebdf_partition
import roi_config as RC
from extract_trackE_perframe import SPATIAL13
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
SEED=42; F="features"; TE=f"{F}/trackE"; TD=f"{F}/trackD"; OUT="results_clean"
G1=RC.CANDIDATE_GROUPS["G1_mouth_instability"]; MAN=["deepfakes","face2face","faceswap","neuraltextures"]
STATS=["mean","std","min","max","p10","p25","p75","p90","iqr","skew","kurt"]
def commit():
    try: return subprocess.check_output(["git","rev-parse","--short","HEAD"],text=True).strip()
    except: return "nogit"
def LGBM(): return lgb.LGBMClassifier(n_estimators=300,max_depth=6,learning_rate=0.05,num_leaves=31,
    min_child_samples=20,class_weight="balanced",random_state=SEED,verbose=-1,n_jobs=1,deterministic=True,force_row_wise=True)
def bn(p): return os.path.basename(str(p))

def aggregate(perframe_csv):
    d=pd.read_csv(perframe_csv)
    key="video_path" if "video_path" in d.columns else "video"   # FF++ uses full path (basename collisions); celebdf basename
    g=d.groupby(key)
    rows={}
    for feat in SPATIAL13:
        s=g[feat]
        q=s.quantile([.10,.25,.75,.90]).unstack()
        rows[f"{feat}__mean"]=s.mean(); rows[f"{feat}__std"]=s.std()
        rows[f"{feat}__min"]=s.min(); rows[f"{feat}__max"]=s.max()
        rows[f"{feat}__p10"]=q[.10]; rows[f"{feat}__p25"]=q[.25]; rows[f"{feat}__p75"]=q[.75]; rows[f"{feat}__p90"]=q[.90]
        rows[f"{feat}__iqr"]=q[.75]-q[.25]
        rows[f"{feat}__skew"]=s.apply(lambda x: float(skew(x)) if len(x)>2 else 0.0)
        rows[f"{feat}__kurt"]=s.apply(lambda x: float(kurtosis(x)) if len(x)>3 else 0.0)
    out=pd.DataFrame(rows); out.insert(0,"key",out.index); out=out.reset_index(drop=True)
    out["label"]=g["label"].first().values
    return out
AGG=[f"{f}__{s}" for f in SPATIAL13 for s in STATS]   # 143

os.makedirs(TE,exist_ok=True)
agg_ff=aggregate(f"{TE}/perframe_ffpp_trainval.csv"); agg_ff.to_csv(f"{TE}/E1_aggstats_ffpp_trainval.csv",index=False)
agg_cd=aggregate(f"{TE}/perframe_celebdf_dev.csv");   agg_cd.to_csv(f"{TE}/E1_aggstats_celebdf_dev.csv",index=False)
print(f"aggregated: FF++ {len(agg_ff)} videos, celebdf_dev {len(agg_cd)} videos, {len(AGG)} stat features",flush=True)

# assemble feature frames (existing 50-D provides the 37 temporal; roi provides G1)
def ff_set(name):
    o=pd.read_csv(f"{F}/ffpp_{'original' if name=='real' else name}_c23.csv")
    r=pd.read_csv(f"{F}/roi_{'original' if name=='real' else name}_c23.csv")
    o["_b"]=o.video_path.map(bn); r["_b"]=r.video_path.map(bn)
    # FF++ agg is keyed by FULL video_path (basenames collide across manips) -> merge on video_path
    m=o.merge(r[["_b"]+G1],on="_b",how="inner").merge(agg_ff.rename(columns={"key":"video_path"}),on="video_path",how="inner",suffixes=("","_agg"))
    return make_splits(m)
FF={k:ff_set(k) for k in ["real"]+MAN}
S13=[c for c in FF["real"].columns if c in SPATIAL13]; T37=sorted([c for c in FF["real"].columns if c.startswith("t_")])
C53=S13+T37+G1; C_REPL=AGG+T37+G1; C_ADD=S13+T37+G1+AGG
def imp(df,cols):
    d=df.copy()
    for c in cols: d[c]=pd.to_numeric(d[c],errors="coerce").replace([np.inf,-np.inf],np.nan)
    d[cols]=d[cols].fillna(d.loc[d.partition=="train",cols].median()); return d
allc=sorted(set(C53+C_REPL+C_ADD)); FF={k:imp(v,allc) for k,v in FF.items()}
cd=pd.read_csv(f"{F}/celebdf_features.csv"); cd["_b"]=cd.video_path.map(bn)
g1=pd.read_csv(f"{TD}/G1_celebdf_dev.csv"); g1["_b"]=g1.video_path.map(bn)
CD=cd.merge(g1[["_b"]+G1],on="_b",how="inner").merge(agg_cd.rename(columns={"key":"_b"}),on="_b",how="inner",suffixes=("","_agg"))  # celebdf basename unique
trall=pd.concat([FF["real"][FF["real"].partition=="train"]]+[FF[m][FF[m].partition=="train"] for m in MAN],ignore_index=True)
med=trall[allc].median(); CDi=CD.copy()
for c in allc: CDi[c]=pd.to_numeric(CDi[c],errors="coerce").replace([np.inf,-np.inf],np.nan).fillna(med[c])
yc=CDi.label.values.astype(int)
print(f"celebdf_dev merged: {len(CDi)} (real {int((yc==0).sum())}, fake {int((yc==1).sum())})",flush=True)

def fit_pred(tr,te,cols):
    sc=StandardScaler().fit(tr[cols].values); m=LGBM().fit(sc.transform(tr[cols].values),tr.label.values.astype(int))
    return m.predict_proba(sc.transform(te[cols].values))[:,1]
def boot(y,pa,pb,n=2000,s=SEED):
    rng=np.random.RandomState(s); d=[]
    for _ in range(n):
        i=rng.randint(0,len(y),len(y))
        if len(np.unique(y[i]))<2: continue
        d.append(roc_auc_score(y[i],pa[i])-roc_auc_score(y[i],pb[i]))
    d=np.array(d); return round(float(np.percentile(d,2.5)),4),round(float(np.percentile(d,97.5)),4),float(max(2*min((d<=0).mean(),(d>=0).mean()),1e-4))
def holm(pv):
    idx=np.argsort(pv); o=np.empty(len(pv)); prev=0; m=len(pv)
    for r,i in enumerate(idx): prev=max(prev,(m-r)*pv[i]); o[i]=min(prev,1.0)
    return o

def indist(cols):
    ys=[];ps=[];per={}
    for mn in MAN:
        tr=pd.concat([FF["real"][FF["real"].partition=="train"],FF[mn][FF[mn].partition=="train"]],ignore_index=True)
        va=pd.concat([FF["real"][FF["real"].partition=="val"],  FF[mn][FF[mn].partition=="val"]],  ignore_index=True)
        p=fit_pred(tr,va,cols); y=va.label.values.astype(int); ps.append(p); ys.append(y); per[mn]=(y,p)
    return np.concatenate(ys),np.concatenate(ps),per

yI,p53I,per53=indist(C53); auc53I=roc_auc_score(yI,p53I); p53C=fit_pred(trall,CDi,C53); auc53C=roc_auc_score(yc,p53C)
res={"provenance":dict(script="exp_trackE_E1_eval.py",git_commit=commit(),seed=SEED,date=datetime.date.today().isoformat(),
     axis_dev_only=True,sealed_touched=False,n_stat_features=len(AGG),thresholds=dict(indist=0.005,cross=0.03)),
     "baseline_53D":dict(indist_pooled=round(auc53I,4),celebdf_dev=round(auc53C,4)),"variants":[]}
tests=[]; keys=[]
for name,cols in [("E1_replace",C_REPL),("E1_additive",C_ADD)]:
    yv,pv,perv=indist(cols); aI=roc_auc_score(yv,pv); loI,hiI,pI=boot(yI,pv,p53I)
    pc=fit_pred(trall,CDi,cols); aC=roc_auc_score(yc,pc); loC,hiC,pC=boot(yc,pc,p53C)
    permanip={mn:round(roc_auc_score(*perv[mn]),4) for mn in MAN}
    permanip_d={mn:round(roc_auc_score(*perv[mn])-roc_auc_score(*per53[mn]),4) for mn in MAN}
    res["variants"].append(dict(model=name,indist_auc=round(aI,4),indist_delta=round(aI-auc53I,4),indist_ci=[loI,hiI],
        cross_auc=round(aC,4),cross_delta=round(aC-auc53C,4),cross_ci=[loC,hiC],per_manip_auc=permanip,per_manip_delta=permanip_d))
    tests+=[pI,pC]; keys+=[(name,"indist"),(name,"cross")]
for (nm,ax),p in zip(keys,holm(np.array(tests))):
    for v in res["variants"]:
        if v["model"]==nm: v[f"{ax}_p_holm"]=round(float(p),4)
json.dump(res,open(f"{OUT}/trackE_E1_dev.json","w"),indent=1)
print("="*80);print("TRACK E1 — richer temporal aggregation. 53-D base: in-dist %.4f, celebdf_dev %.4f"%(auc53I,auc53C));print("="*80)
for v in res["variants"]:
    print(f"  {v['model']:12s} in-dist {v['indist_delta']:+.4f} (p_holm {v.get('indist_p_holm',1):.3f}) | cross {v['cross_delta']:+.4f} (p_holm {v.get('cross_p_holm',1):.3f})")
    print(f"     per-manip Δ: "+"  ".join(f"{m}={v['per_manip_delta'][m]:+.4f}" for m in MAN))
print(f"saved {OUT}/trackE_E1_dev.json (commit {commit()})")
