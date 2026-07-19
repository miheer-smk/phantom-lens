#!/usr/bin/env python3
"""SHAP ranking stability (Reviewer 5 / Experiment 2), identity-disjoint.
(A) Per-manipulation SHAP rankings (DF/F2F/FS/NT) -> Spearman, top-5/top-10 overlap, mean rank, rank SD.
(B) Across identity-grouped CV folds (GroupKFold by source identity) -> per-fold rankings + Spearman matrix.
(C) Explicit test of the manuscript claim about NeuralTextures top-3 dominance.
Uses the 50-D locked features (the set the manuscript's ranking discusses). shap.TreeExplainer on LightGBM."""
import os,sys,json,subprocess,datetime,itertools
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore"); sys.path.insert(0,"src")
from protocol import make_splits, clip_identities
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
import lightgbm as lgb, shap
SEED=42; F="features"; OUT="results_clean"
MAN={"Deepfakes":"deepfakes","Face2Face":"face2face","FaceSwap":"faceswap","NeuralTextures":"neuraltextures"}
real=pd.read_csv(f"{F}/ffpp_original_c23.csv"); FC=sorted([c for c in real.columns if c[:2] in ("s_","t_")])
def clean(df):
    d=df.copy()
    for c in FC: d[c]=pd.to_numeric(d[c],errors="coerce").replace([np.inf,-np.inf],np.nan); d[c]=d[c].fillna(d[c].median())
    return make_splits(d)
realc=clean(real); MANc={k:clean(pd.read_csv(f"{F}/ffpp_{v}_c23.csv")) for k,v in MAN.items()}
def LGBM(): return lgb.LGBMClassifier(n_estimators=200,max_depth=6,learning_rate=0.05,num_leaves=31,min_child_samples=20,class_weight="balanced",random_state=SEED,verbose=-1,n_jobs=-1)
def shap_rank(Xtr,ytr,Xexpl):
    sc=StandardScaler().fit(Xtr); clf=LGBM(); clf.fit(sc.transform(Xtr),ytr)
    ex=shap.TreeExplainer(clf); sv=ex.shap_values(sc.transform(Xexpl))
    if isinstance(sv,list): sv=sv[1]
    imp=np.abs(sv).mean(0)
    return imp  # importance per feature (order of FC)
def commit():
    try: return subprocess.check_output(["git","rev-parse","--short","HEAD"],text=True).strip()
    except: return "nogit"

# ---- (A) per-manipulation rankings (train on that manip's train ids; explain on its val ids) ----
imp_by_manip={}
for m,mdf in MANc.items():
    tr=pd.concat([realc[realc.partition=="train"],mdf[mdf.partition=="train"]],ignore_index=True)
    ex=pd.concat([realc[realc.partition=="val"],  mdf[mdf.partition=="val"]],  ignore_index=True)
    imp=shap_rank(tr[FC].values.astype(float),tr['label'].values.astype(int),ex[FC].values.astype(float))
    imp_by_manip[m]=imp
    print(f"  SHAP done: {m}",flush=True)
rankdf=pd.DataFrame({m:pd.Series(imp,index=FC).rank(ascending=False) for m,imp in imp_by_manip.items()})
rankdf["mean_rank"]=rankdf.mean(1); rankdf["rank_sd"]=rankdf[list(MAN)].std(1)
rankdf=rankdf.sort_values("mean_rank")
# cross-manip Spearman + top-k overlap
mans=list(MAN); sp={}; ov5={}; ov10={}
for a,b in itertools.combinations(mans,2):
    sp[f"{a}~{b}"]=round(spearmanr(imp_by_manip[a],imp_by_manip[b]).correlation,3)
    ta=set(pd.Series(imp_by_manip[a],index=FC).nlargest(5).index); tb=set(pd.Series(imp_by_manip[b],index=FC).nlargest(5).index)
    ov5[f"{a}~{b}"]=len(ta&tb)
    ta=set(pd.Series(imp_by_manip[a],index=FC).nlargest(10).index); tb=set(pd.Series(imp_by_manip[b],index=FC).nlargest(10).index)
    ov10[f"{a}~{b}"]=len(ta&tb)

# ---- (B) across identity-grouped CV folds (multi-manip training set) ----
allc=pd.concat([realc[realc.partition!="test"]]+[MANc[m][MANc[m].partition!="test"] for m in MAN],ignore_index=True)
groups=allc['video_path'].map(lambda p: sorted(clip_identities(p))[0])
gkf=GroupKFold(n_splits=10); fold_imps=[]
for fi,(tri,_) in enumerate(gkf.split(allc[FC].values,allc['label'].values,groups)):
    sub=allc.iloc[tri]
    imp=shap_rank(sub[FC].values.astype(float),sub['label'].values.astype(int),sub[FC].values.astype(float))
    fold_imps.append(imp)
fold_ranks=np.array([pd.Series(im,index=FC).rank(ascending=False).values for im in fold_imps])
foldSp=np.zeros((10,10))
for i in range(10):
    for j in range(10): foldSp[i,j]=spearmanr(fold_imps[i],fold_imps[j]).correlation
mean_fold_sp=round((foldSp[np.triu_indices(10,1)]).mean(),3)
fold_rank_sd=pd.Series(fold_ranks.std(0),index=FC)

# ---- (C) NeuralTextures top-3 claim ----
nt_top3=list(pd.Series(imp_by_manip["NeuralTextures"],index=FC).nlargest(3).index)
# how stable are these across manips & folds?
nt_top3_stability={f:{"mean_rank":round(float(rankdf.loc[f,"mean_rank"]),1),
                      "rank_sd_manip":round(float(rankdf.loc[f,"rank_sd"]),1),
                      "fold_rank_sd":round(float(fold_rank_sd[f]),1)} for f in nt_top3}

out=dict(provenance=dict(script="shap_stability.py",git_commit=commit(),seed=SEED,date=datetime.date.today().isoformat(),
    method="shap.TreeExplainer on LightGBM, 50-D, identity-disjoint; GroupKFold by source identity"),
    per_manip_spearman=sp, top5_overlap=ov5, top10_overlap=ov10,
    mean_cross_fold_spearman=mean_fold_sp,
    nt_top3=nt_top3, nt_top3_stability=nt_top3_stability)
rankdf.round(2).to_csv(f"{OUT}/shap_ranking_by_manip.csv")
json.dump(out,open(f"{OUT}/shap_stability.json","w"),indent=1)
print("\n=== SHAP STABILITY ===")
print("cross-manip Spearman:",sp)
print("top-5 overlap:",ov5,"| top-10 overlap:",ov10)
print("mean cross-fold Spearman (10 folds):",mean_fold_sp)
print("NeuralTextures top-3:",nt_top3)
for f,s in nt_top3_stability.items(): print(f"   {f}: {s}")
print(f"saved {OUT}/shap_stability.json, shap_ranking_by_manip.csv (commit {commit()})")
