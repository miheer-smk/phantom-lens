#!/usr/bin/env python3
"""Track E — X1 (KS-stability feature selection) + X2 (rPPG drop), on the E1-expanded 196-D set.
DEV only; sealed untouched. Compose with E1. No extraction (reuses E1 aggstats + 50-D + G1).
X1: rank features by KS distance between FF++-train and UNLABELED celebdf_dev marginals; train top-k
    most domain-stable subsets k in {20,30,50,80,120}; report in-dist + celebdf_dev.
X2: drop rPPG temporal features entirely; report both axes vs full E1-expanded set.
Train-only imputer, seed 42, locked LightGBM; thresholds +0.005 in-dist / +0.03 cross.
"""
import os, sys, json, subprocess, datetime
import numpy as np, pandas as pd, warnings
from scipy.stats import ks_2samp
warnings.filterwarnings("ignore"); sys.path.insert(0, "src")
from protocol import make_splits
import roi_config as RC
from extract_trackE_perframe import SPATIAL13
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
SEED=42; F="features"; TE=f"{F}/trackE"; TD=f"{F}/trackD"; OUT="results_clean"
G1=RC.CANDIDATE_GROUPS["G1_mouth_instability"]; MAN=["deepfakes","face2face","faceswap","neuraltextures"]
STATS=["mean","std","min","max","p10","p25","p75","p90","iqr","skew","kurt"]
AGG=[f"{f}__{s}" for f in SPATIAL13 for s in STATS]
RPPG=["t_rppg_snr","t_rppg_peak_prominence","t_rppg_interregion_corr","t_rppg_harmonic_ratio"]
def bn(p): return os.path.basename(str(p))
def commit():
    try: return subprocess.check_output(["git","rev-parse","--short","HEAD"],text=True).strip()
    except: return "nogit"
def LGBM(): return lgb.LGBMClassifier(n_estimators=300,max_depth=6,learning_rate=0.05,num_leaves=31,
    min_child_samples=20,class_weight="balanced",random_state=SEED,verbose=-1,n_jobs=1,deterministic=True,force_row_wise=True)

agg_ff=pd.read_csv(f"{TE}/E1_aggstats_ffpp_trainval.csv"); agg_cd=pd.read_csv(f"{TE}/E1_aggstats_celebdf_dev.csv")
def ff_set(name):
    o=pd.read_csv(f"{F}/ffpp_{'original' if name=='real' else name}_c23.csv")
    r=pd.read_csv(f"{F}/roi_{'original' if name=='real' else name}_c23.csv")
    o["_b"]=o.video_path.map(bn); r["_b"]=r.video_path.map(bn)
    m=o.merge(r[["_b"]+G1],on="_b",how="inner").merge(agg_ff.rename(columns={"key":"video_path"}),on="video_path",how="inner",suffixes=("","_a"))
    return make_splits(m)
FF={k:ff_set(k) for k in ["real"]+MAN}
S13=[c for c in FF["real"].columns if c in SPATIAL13]; T37=sorted([c for c in FF["real"].columns if c.startswith("t_")])
FULL=S13+T37+G1+AGG                       # E1-expanded 196-D
def imp(df,cols):
    d=df.copy()
    for c in cols: d[c]=pd.to_numeric(d[c],errors="coerce").replace([np.inf,-np.inf],np.nan)
    d[cols]=d[cols].fillna(d.loc[d.partition=="train",cols].median()); return d
FF={k:imp(v,FULL) for k,v in FF.items()}
cd=pd.read_csv(f"{F}/celebdf_features.csv"); cd["_b"]=cd.video_path.map(bn)
g1=pd.read_csv(f"{TD}/G1_celebdf_dev.csv"); g1["_b"]=g1.video_path.map(bn)
CD=cd.merge(g1[["_b"]+G1],on="_b",how="inner").merge(agg_cd.rename(columns={"key":"_b"}),on="_b",how="inner",suffixes=("","_a"))
trall=pd.concat([FF["real"][FF["real"].partition=="train"]]+[FF[m][FF[m].partition=="train"] for m in MAN],ignore_index=True)
med=trall[FULL].median(); CDi=CD.copy()
for c in FULL: CDi[c]=pd.to_numeric(CDi[c],errors="coerce").replace([np.inf,-np.inf],np.nan).fillna(med[c])
yc=CDi.label.values.astype(int)

def fit_pred(tr,te,cols):
    sc=StandardScaler().fit(tr[cols].values); m=LGBM().fit(sc.transform(tr[cols].values),tr.label.values.astype(int))
    return m.predict_proba(sc.transform(te[cols].values))[:,1]
def indist(cols):
    ys=[];ps=[]
    for mn in MAN:
        tr=pd.concat([FF["real"][FF["real"].partition=="train"],FF[mn][FF[mn].partition=="train"]],ignore_index=True)
        va=pd.concat([FF["real"][FF["real"].partition=="val"],  FF[mn][FF[mn].partition=="val"]],  ignore_index=True)
        ps.append(fit_pred(tr,va,cols)); ys.append(va.label.values.astype(int))
    return roc_auc_score(np.concatenate(ys),np.concatenate(ps))

# reference: full E1-expanded set
full_in=indist(FULL); full_cr=roc_auc_score(yc,fit_pred(trall,CDi,FULL))

# X1 — KS distance FF++ train vs celebdf_dev (unlabeled), rank stable
ks={c:ks_2samp(trall[c].values, CDi[c].values).statistic for c in FULL}
order=sorted(FULL,key=lambda c: ks[c])   # ascending KS = most domain-stable first
x1=[]
for k in (20,30,50,80,120,len(FULL)):
    cols=order[:k]
    x1.append(dict(k=k,indist_auc=round(indist(cols),4),cross_auc=round(roc_auc_score(yc,fit_pred(trall,CDi,cols)),4)))

# X2 — drop rPPG
noR=[c for c in FULL if c not in RPPG]
x2=dict(indist_auc=round(indist(noR),4),cross_auc=round(roc_auc_score(yc,fit_pred(trall,CDi,noR)),4),n_dropped=len([c for c in FULL if c in RPPG]))

res=dict(provenance=dict(script="exp_trackE_X1X2.py",git_commit=commit(),seed=SEED,date=datetime.date.today().isoformat(),
    axis_dev_only=True,sealed_touched=False,base="E1-expanded 196-D",full_indist=round(full_in,4),full_cross=round(full_cr,4)),
    X1_ks_stable=x1, X1_most_stable_features=order[:15], X1_least_stable=order[-10:],
    X2_drop_rppg=x2)
json.dump(res,open(f"{OUT}/trackE_X1X2_dev.json","w"),indent=1)
print("="*70);print("TRACK E X1/X2 (on E1-expanded 196-D). full: in-dist %.4f cross %.4f"%(full_in,full_cr));print("="*70)
print("X1 KS-stable top-k:")
for r in x1: print(f"   k={r['k']:3d}  in-dist {r['indist_auc']:.4f} (Δ{r['indist_auc']-full_in:+.4f}) | cross {r['cross_auc']:.4f} (Δ{r['cross_auc']-full_cr:+.4f})")
print(f"X2 drop rPPG ({x2['n_dropped']} feats): in-dist {x2['indist_auc']:.4f} (Δ{x2['indist_auc']-full_in:+.4f}) | cross {x2['cross_auc']:.4f} (Δ{x2['cross_auc']-full_cr:+.4f})")
print("most domain-stable feats:",order[:6])
print(f"saved {OUT}/trackE_X1X2_dev.json (commit {commit()})")
