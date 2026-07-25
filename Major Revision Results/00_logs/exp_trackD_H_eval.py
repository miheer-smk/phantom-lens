#!/usr/bin/env python3
"""Track D — Group H DEV evaluation (FF++ train/val + celebdf_dev ONLY; sealed sets untouched).
53-D baseline vs 53-D+H:
  IN-DISTRIBUTION: per-manipulation, train real+manip TRAIN, eval real+manip VAL (identity-disjoint).
  CROSS-DATASET  : train FF++ TRAIN (real+all manips), eval celebdf_dev.
Reports incremental ΔAUC (paired bootstrap 95% CI) on both axes + per-feature Cohen's d (val & dev).
Train-only imputer/scaler (leakfree), seed 42, locked LightGBM. No sealed access.
"""
import os, sys, json, subprocess, datetime
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore"); sys.path.insert(0, "src")
from protocol import make_splits
from leakfree import split_impute, impute_with, pooled_train_median
import roi_config as RC
from extract_trackD_H import H_FEATURES
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

SEED=42; F="features"; TD=f"{F}/trackD"; OUT="results_clean"
G1=RC.CANDIDATE_GROUPS["G1_mouth_instability"]
MAN=["deepfakes","face2face","faceswap","neuraltextures"]
def bn(p): return os.path.basename(str(p))
def commit():
    try: return subprocess.check_output(["git","rev-parse","--short","HEAD"],text=True).strip()
    except: return "nogit"
def LGBM(): return lgb.LGBMClassifier(n_estimators=200,max_depth=6,learning_rate=0.05,num_leaves=31,
    min_child_samples=20,class_weight="balanced",random_state=SEED,verbose=-1,n_jobs=1,deterministic=True,force_row_wise=True)

# ---- load 50-D + G1 + H, merged per set (FF++) ----
H_ff=pd.read_csv(f"{TD}/H_ffpp_trainval.csv"); H_ff["_b"]=H_ff.video_path.map(bn)
def ff_set(name):
    o=pd.read_csv(f"{F}/ffpp_{'original' if name=='real' else name}_c23.csv")
    r=pd.read_csv(f"{F}/roi_{'original' if name=='real' else name}_c23.csv")
    o["_b"]=o.video_path.map(bn); r["_b"]=r.video_path.map(bn)
    m=o.merge(r[["_b"]+G1],on="_b",how="inner").merge(H_ff[["_b"]+H_FEATURES],on="_b",how="inner")
    return make_splits(m)
FF={k:ff_set(k) for k in ["real"]+MAN}
FC=sorted([c for c in FF["real"].columns if c[:2] in ("s_","t_")])
C53=FC+G1; C53H=FC+G1+H_FEATURES

# ---- celebdf dev: 50-D + G1 + H ----
cd=pd.read_csv(f"{F}/celebdf_features.csv"); cd["_b"]=cd.video_path.map(bn)
g1cd=pd.read_csv(f"{TD}/G1_celebdf_dev.csv"); g1cd["_b"]=g1cd.video_path.map(bn)
hcd=pd.read_csv(f"{TD}/H_celebdf_dev.csv"); hcd["_b"]=hcd.video_path.map(bn)
CD=cd.merge(g1cd[["_b"]+G1],on="_b",how="inner").merge(hcd[["_b"]+H_FEATURES],on="_b",how="inner")
print(f"celebdf_dev merged: {len(CD)} videos (real {int((CD.label==0).sum())}, fake {int((CD.label==1).sum())})",flush=True)

def imp(df,cols):  # train-only imputer via partition if present else pooled
    d=df.copy()
    for c in cols: d[c]=pd.to_numeric(d[c],errors="coerce").replace([np.inf,-np.inf],np.nan)
    if "partition" in d: d[cols]=d[cols].fillna(d.loc[d.partition=="train",cols].median())
    return d
FF={k:imp(v,C53H) for k,v in FF.items()}

def boot_ci(y,pa,pb,n=1500,s=SEED):
    rng=np.random.RandomState(s); d=[]
    for _ in range(n):
        i=rng.randint(0,len(y),len(y))
        if len(np.unique(y[i]))<2: continue
        d.append(roc_auc_score(y[i],pa[i])-roc_auc_score(y[i],pb[i]))
    return round(float(np.percentile(d,2.5)),4),round(float(np.percentile(d,97.5)),4)
def fit_pred(tr,te,cols):
    sc=StandardScaler().fit(tr[cols].values); m=LGBM().fit(sc.transform(tr[cols].values),tr.label.values.astype(int))
    return m.predict_proba(sc.transform(te[cols].values))[:,1]

# ---- IN-DISTRIBUTION per manip (val) ----
indist=[]
for m in MAN:
    tr=pd.concat([FF["real"][FF["real"].partition=="train"],FF[m][FF[m].partition=="train"]],ignore_index=True)
    va=pd.concat([FF["real"][FF["real"].partition=="val"],  FF[m][FF[m].partition=="val"]],  ignore_index=True)
    y=va.label.values.astype(int)
    p53=fit_pred(tr,va,C53); p53h=fit_pred(tr,va,C53H)
    a53,a53h=roc_auc_score(y,p53),roc_auc_score(y,p53h)
    lo,hi=boot_ci(y,p53h,p53)
    indist.append(dict(manip=m,auc_53=round(a53,4),auc_53H=round(a53h,4),delta=round(a53h-a53,4),ci=[lo,hi],n_val=int(len(y))))
mean53=np.mean([r["auc_53"] for r in indist]); mean53h=np.mean([r["auc_53H"] for r in indist])

# ---- CROSS-DATASET (celebdf_dev) ----
trall=pd.concat([FF["real"][FF["real"].partition=="train"]]+[FF[m][FF[m].partition=="train"] for m in MAN],ignore_index=True)
ff_med=trall[C53H].median()
CDi=impute_with(CD,C53H,ff_med) if False else CD.copy()
for c in C53H: CDi[c]=pd.to_numeric(CDi[c],errors="coerce").replace([np.inf,-np.inf],np.nan).fillna(ff_med[c])
yc=CDi.label.values.astype(int)
pc53=fit_pred(trall,CDi,C53); pc53h=fit_pred(trall,CDi,C53H)
c53,c53h=roc_auc_score(yc,pc53),roc_auc_score(yc,pc53h); clo,chi=boot_ci(yc,pc53h,pc53)

# ---- per-feature Cohen's d (val pooled, and celebdf_dev) ----
def cohen(df,feat):
    a=df[df.label==0][feat].astype(float); b=df[df.label==1][feat].astype(float)
    sp=np.sqrt(((len(a)-1)*a.var()+(len(b)-1)*b.var())/max(len(a)+len(b)-2,1))
    return round(float((b.mean()-a.mean())/(sp+1e-9)),3)
valpool=pd.concat([FF["real"][FF["real"].partition=="val"]]+[FF[m][FF[m].partition=="val"] for m in MAN],ignore_index=True)
dcoh={h:dict(val_d=cohen(valpool,h),celebdf_dev_d=cohen(CDi,h)) for h in H_FEATURES}

res=dict(provenance=dict(script="exp_trackD_H_eval.py",git_commit=commit(),seed=SEED,date=datetime.date.today().isoformat(),
    axis_dev_only=True,sealed_touched=False,n_H_features=len(H_FEATURES)),
    in_distribution=dict(per_manip=indist,mean_53=round(mean53,4),mean_53H=round(mean53h,4),mean_delta=round(mean53h-mean53,4)),
    cross_dataset_celebdf_dev=dict(n=int(len(CDi)),auc_53=round(c53,4),auc_53H=round(c53h,4),delta=round(c53h-c53,4),ci=[clo,chi]),
    per_feature_cohens_d=dcoh)
json.dump(res,open(f"{OUT}/trackD_H_dev.json","w"),indent=1)
print("="*70);print("TRACK D — GROUP H (gradient structure tensor) — DEV RESULTS");print("="*70)
print("IN-DISTRIBUTION (FF++ val), 53-D vs 53-D+H:")
for r in indist: print(f"  {r['manip']:15s} 53={r['auc_53']:.4f} 53+H={r['auc_53H']:.4f} Δ={r['delta']:+.4f} CI{r['ci']}")
print(f"  MEAN            53={mean53:.4f} 53+H={mean53h:.4f} Δ={mean53h-mean53:+.4f}")
print(f"CROSS-DATASET (celebdf_dev, n={len(CDi)}): 53={c53:.4f} 53+H={c53h:.4f} Δ={c53h-c53:+.4f} CI[{clo},{chi}]")
print("per-feature Cohen's d (val | celebdf_dev):")
for h,dd in dcoh.items(): print(f"  {h:24s} {dd['val_d']:+.3f} | {dd['celebdf_dev_d']:+.3f}")
print(f"saved {OUT}/trackD_H_dev.json (commit {commit()})")
