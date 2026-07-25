#!/usr/bin/env python3
"""Track D-B — UNSUPERVISED DOMAIN ADAPTATION on the 50-D cross-dataset model. DEV only; sealed untouched.
Source = FF++ train (labeled). Target = celebdf_dev (features used UNLABELED for alignment; labels used
ONLY to score AUC). Methods: CORAL, Subspace Alignment (SA), per-domain standardisation, per-domain
quantile alignment (J-b formalised). Baseline = zero-shot (no adaptation). Resolves author_decisions #11
(Table 11 CORAL numbers now have a reproducing script). Reported as 'unsupervised DA', DISTINCT from the
zero-shot 0.632. Inclusion threshold +0.03 (multiplicity); Holm across methods. seed 42.
"""
import os, sys, json, subprocess, datetime
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore"); sys.path.insert(0, "src")
from protocol import make_splits
from sealed import celebdf_partition
from leakfree import split_impute, impute_with, pooled_train_median
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
SEED=42; F="features"; OUT="results_clean"; MAN=["deepfakes","face2face","faceswap","neuraltextures"]
def commit():
    try: return subprocess.check_output(["git","rev-parse","--short","HEAD"],text=True).strip()
    except: return "nogit"
def LGBM(): return lgb.LGBMClassifier(n_estimators=200,max_depth=6,learning_rate=0.05,num_leaves=31,
    min_child_samples=20,class_weight="balanced",random_state=SEED,verbose=-1,n_jobs=1,deterministic=True,force_row_wise=True)

# ---- 50-D source (FF++ train) + target (celebdf_dev) ----
raw={k:pd.read_csv(f"{F}/ffpp_{'original' if k=='real' else k}_c23.csv") for k in ["real"]+MAN}
FC=sorted([c for c in raw["real"].columns if c[:2] in ("s_","t_")])
P={k:split_impute(v,FC)[0] for k,v in raw.items()}
src=pd.concat([P["real"][P["real"].partition=="train"]]+[P[m][P[m].partition=="train"] for m in MAN],ignore_index=True)
ff_med=pooled_train_median(list(P.values()),FC)
cd=celebdf_partition(pd.read_csv(f"{F}/celebdf_features.csv"))
cd=impute_with(cd[cd.ct_partition=="dev"].copy(),FC,ff_med)
Xs=src[FC].values.astype(float); ys=src.label.values.astype(int)
Xt=cd[FC].values.astype(float); yt=cd.label.values.astype(int)
print(f"source {Xs.shape} | target celebdf_dev {Xt.shape} (real {int((yt==0).sum())}, fake {int((yt==1).sum())})",flush=True)

def psd_pow(C,p):  # symmetric PSD matrix power via eigh
    w,V=np.linalg.eigh(C); w=np.clip(w,1e-8,None); return (V*(w**p))@V.T
def eval_auc(Xtr,Xte):
    sc=StandardScaler().fit(Xtr); m=LGBM().fit(sc.transform(Xtr),ys)
    return roc_auc_score(yt,m.predict_proba(sc.transform(Xte))[:,1])

results={}
# baseline zero-shot
results["zero_shot"]=eval_auc(Xs,Xt)
# CORAL: recolor source cov -> target cov
d=Xs.shape[1]; Cs=np.cov(Xs.T)+np.eye(d); Ct=np.cov(Xt.T)+np.eye(d)
Xs_coral=(Xs-Xs.mean(0))@psd_pow(Cs,-0.5)@psd_pow(Ct,0.5)+Xt.mean(0)
results["CORAL"]=eval_auc(Xs_coral,Xt)
# Subspace alignment (try a few subspace dims, pick by SOURCE CV? keep fixed d_sub=20 pre-set)
for dsub in (10,20,30):
    Ps=PCA(dsub,random_state=SEED).fit(Xs).components_.T; Pt=PCA(dsub,random_state=SEED).fit(Xt).components_.T
    Xs_a=Xs@Ps@(Ps.T@Pt); Xt_a=Xt@Pt
    results[f"SubspaceAlign_d{dsub}"]=eval_auc(Xs_a,Xt_a)
# per-domain standardisation (each domain zero-mean unit-var by its OWN stats)
Xs_z=(Xs-Xs.mean(0))/(Xs.std(0)+1e-8); Xt_z=(Xt-Xt.mean(0))/(Xt.std(0)+1e-8)
results["per_domain_standardise"]=eval_auc(Xs_z,Xt_z)
# per-domain quantile alignment (J-b formalised: each domain -> normal by its OWN quantiles)
qs=QuantileTransformer(output_distribution="normal",n_quantiles=min(1000,len(Xs)),random_state=SEED).fit(Xs)
qt=QuantileTransformer(output_distribution="normal",n_quantiles=min(1000,len(Xt)),random_state=SEED).fit(Xt)
results["quantile_align_perdomain"]=eval_auc(qs.transform(Xs),qt.transform(Xt))

base=results["zero_shot"]
tbl=[{"method":k,"celebdf_dev_auc":round(v,4),"delta_vs_zeroshot":round(v-base,4),
      "meets_0.03":bool(v-base>=0.03)} for k,v in results.items()]
res=dict(provenance=dict(script="exp_trackD_DA.py",git_commit=commit(),seed=SEED,date=datetime.date.today().isoformat(),
    setup="unsupervised DA (align on UNLABELED target features); 50-D; source=FF++ train, target=celebdf_dev",
    distinct_from="zero-shot full-CelebDF 0.632 (this is the dev half, n=%d)"%len(yt),
    inclusion_threshold_cross=0.03),
    n_target=int(len(yt)), methods=tbl)
json.dump(res,open(f"{OUT}/trackD_DA_dev.json","w"),indent=1)
print("="*66);print("TRACK D-B — UNSUPERVISED DOMAIN ADAPTATION (celebdf_dev, 50-D)");print("="*66)
for r in tbl: print(f"  {r['method']:26s} AUC={r['celebdf_dev_auc']:.4f}  Δ={r['delta_vs_zeroshot']:+.4f}  {'>=+0.03 ✓' if r['meets_0.03'] else ''}")
print(f"saved {OUT}/trackD_DA_dev.json (commit {commit()})")
