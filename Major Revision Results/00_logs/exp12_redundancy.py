#!/usr/bin/env python3
"""EXP-12 Feature redundancy (R3.12) — 50-D locked features, identity-disjoint.
Pearson+Spearman corr, VIF, hierarchical clustering, near-zero-variance flags, |r|>0.90 pairs,
and AUC after dropping one feature of each highly-correlated pair (which-to-drop chosen on
train+val only, via pillar-only/importance; evaluated once on test). Pairs with pillar-only table.
"""
import os,sys,json,subprocess,datetime
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore"); sys.path.insert(0,"src")
from protocol import make_splits, assert_no_identity_overlap
from scipy.stats import spearmanr
from scipy.cluster.hierarchy import linkage, fcluster
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
SEED=42; F="features"; OUT="results_clean"; FIG="Major Revision Results/03_figures/exp12_feature_redundancy"
os.makedirs(FIG,exist_ok=True)
MAN=["deepfakes","face2face","faceswap","neuraltextures"]
def load(name):
    o=pd.read_csv(f"{F}/ffpp_{name}_c23.csv") if name!="real" else pd.read_csv(f"{F}/ffpp_original_c23.csv")
    return make_splits(o)
real=load("real"); MANd={m:load(m) for m in MAN}
FC=sorted([c for c in real.columns if c[:2] in ("s_","t_")])
def clean(df):
    d=df.copy()
    for c in FC: d[c]=pd.to_numeric(d[c],errors="coerce").replace([np.inf,-np.inf],np.nan); d[c]=d[c].fillna(d[c].median())
    return d
real=clean(real); MANd={m:clean(v) for m,v in MANd.items()}
assert_no_identity_overlap([(real[real.partition==p],p) for p in ("train","val","test")]
    +[(MANd[m][MANd[m].partition==p],p) for m in MAN for p in ("train","val","test")])
print("identity-disjoint assertion PASSED",flush=True)
def cat(parts): return pd.concat([real[real.partition.isin(parts)]]+[MANd[m][MANd[m].partition.isin(parts)] for m in MAN],ignore_index=True)
trv=cat(["train","val"]); te=cat(["test"])
X=trv[FC].values.astype(float); Xs=StandardScaler().fit_transform(X)

# ---- correlations ----
P=np.corrcoef(Xs.T); S=spearmanr(Xs).correlation
# ---- near-zero-variance ----
var=Xs.var(0); nzv=[FC[i] for i in range(len(FC)) if var[i]<1e-3]
# ---- VIF (on standardized, guard singular) ----
vif={}
for i,f in enumerate(FC):
    try:
        v=variance_inflation_factor(Xs,i); vif[f]=round(float(v),2) if np.isfinite(v) else None
    except Exception: vif[f]=None
# ---- hierarchical clustering ----
d=1-np.abs(P); np.fill_diagonal(d,0)
from scipy.spatial.distance import squareform
Z=linkage(squareform(d,checks=False),method="average")
clusters=fcluster(Z,t=0.1,criterion="distance")  # |r|>0.9 groups
plt.figure(figsize=(14,5)); dendrogram(Z,labels=FC,leaf_font_size=6); plt.title("Feature hierarchical clustering (1-|Pearson r|)")
plt.tight_layout(); plt.savefig(f"{FIG}/dendrogram.png",dpi=130); plt.close()
# ---- correlation heatmap ----
plt.figure(figsize=(11,9)); im=plt.imshow(P,cmap="RdBu_r",vmin=-1,vmax=1)
plt.colorbar(im,fraction=0.046); plt.xticks(range(len(FC)),FC,rotation=90,fontsize=4); plt.yticks(range(len(FC)),FC,fontsize=4)
plt.title("Pearson correlation (50 features)"); plt.tight_layout(); plt.savefig(f"{FIG}/corr_heatmap.png",dpi=140); plt.close()

# ---- |r|>0.90 pairs; drop-decision on train+val importance only ----
imp=lgb.LGBMClassifier(n_estimators=200,max_depth=6,learning_rate=0.05,num_leaves=31,min_child_samples=20,class_weight="balanced",random_state=SEED,verbose=-1,n_jobs=-1)
imp.fit(Xs,trv['label'].values.astype(int)); importance=dict(zip(FC,imp.feature_importances_))
pairs=[]; to_drop=set()
for i in range(len(FC)):
    for j in range(i+1,len(FC)):
        if abs(P[i,j])>0.90:
            a,b=FC[i],FC[j]; drop=a if importance[a]<=importance[b] else b  # drop lower-importance (train+val)
            pairs.append(dict(feature_1=a,feature_2=b,pearson=round(float(P[i,j]),4),spearman=round(float(S[i,j]),4),
                              dropped=drop,action="Removed one (lower train+val importance)"))
            to_drop.add(drop)
kept=[f for f in FC if f not in to_drop]

# ---- AUC after dedup (evaluate ONCE on test, per manip + overall) ----
def LGBM(): return lgb.LGBMClassifier(n_estimators=200,max_depth=6,learning_rate=0.05,num_leaves=31,min_child_samples=20,class_weight="balanced",random_state=SEED,verbose=-1,n_jobs=-1)
def auc_on(cols):
    scaler=StandardScaler().fit(trv[cols].values); m=LGBM(); m.fit(scaler.transform(trv[cols].values),trv['label'].values.astype(int))
    res={}
    for mm in MAN:
        sub=te[(te.dataset==mm)|(te.dataset=="real")] if 'dataset' in te else None
    # per-manip via test partition of each manip + real test
    for mm,md in MANd.items():
        tt=pd.concat([real[real.partition=="test"],md[md.partition=="test"]],ignore_index=True)
        p=m.predict_proba(scaler.transform(tt[cols].values))[:,1]; res[mm]=round(roc_auc_score(tt['label'].values.astype(int),p),4)
    return res
full_auc=auc_on(FC); dedup_auc=auc_on(kept)

def commit():
    try: return subprocess.check_output(["git","rev-parse","--short","HEAD"],text=True).strip()
    except: return "nogit"
pd.DataFrame(pairs).to_csv(f"{OUT}/redundancy_pairs.csv",index=False)
perf=pd.DataFrame({"manip":list(full_auc),"full_50_auc":[full_auc[k] for k in full_auc],
    "dedup_auc":[dedup_auc[k] for k in full_auc],"n_features_dedup":[len(kept)]*len(full_auc)})
perf["delta"]=perf.dedup_auc-perf.full_50_auc; perf.to_csv(f"{OUT}/redundancy_performance.csv",index=False)
out=dict(provenance=dict(script="exp12_redundancy.py",git_commit=commit(),seed=SEED,date=datetime.date.today().isoformat(),
    protocol="identity-disjoint; drop-decision on train+val importance only; test eval once"),
    n_highly_correlated_pairs=len(pairs),n_features_dropped=len(to_drop),n_kept=len(kept),
    near_zero_variance=nzv,vif_max=max([v for v in vif.values() if v],default=None),
    pairs=pairs,full_auc=full_auc,dedup_auc=dedup_auc)
json.dump(out,open(f"{OUT}/redundancy.json","w"),indent=2)
print("\n=== EXP-12 FEATURE REDUNDANCY (50-D) ===")
print(f"|r|>0.90 pairs: {len(pairs)} | features dropped: {len(to_drop)} | kept: {len(kept)}/50")
print(f"near-zero-variance: {nzv or 'none'}")
print(f"VIF: max={out['vif_max']} | n features VIF>10: {sum(1 for v in vif.values() if v and v>10)}")
for p in pairs: print(f"   {p['feature_1']} ~ {p['feature_2']}  r={p['pearson']}  drop {p['dropped']}")
print(f"\n{'manip':16s} full-50  dedup({len(kept)})   Δ")
for k in full_auc: print(f"  {k:16s} {full_auc[k]:.4f}  {dedup_auc[k]:.4f}  {dedup_auc[k]-full_auc[k]:+.4f}")
print(f"saved {OUT}/redundancy_pairs.csv, redundancy_performance.csv, redundancy.json; figs in {FIG} (commit {commit()})")
