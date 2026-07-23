#!/usr/bin/env python3
"""EXP-3 Compression c23/c40 across ALL 4 manipulations (R5.4). 50-D, identity-disjoint.
Per manip {DF,F2F,FS,NT}: train/test c23; train/test c40; cross train-c23->test-c40.
Report AUC (bootstrap CI) + MCC per compression + ΔAUC, and which feature GROUPS degrade most
under c40 (pillar-only AUC c23 vs c40)."""
import os,sys,json,subprocess,datetime
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore"); sys.path.insert(0,"src")
from protocol import make_splits, assert_no_identity_overlap
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, matthews_corrcoef
import lightgbm as lgb
SEED=42; F="features"; OUT="results_clean"
PILLARS=json.load(open("splits/pillar_map.json"))
MAN=["deepfakes","face2face","faceswap","neuraltextures"]
def load(name,comp):
    fn=f"ffpp_{'original' if name=='real' else name}_{comp}.csv"
    return make_splits(pd.read_csv(f"{F}/{fn}"))
real={c:None for c in ("c23","c40")}
FCref=None
def clean(df,FC):  # M1 fix: TRAIN-partition medians only (df already has 'partition' from load->make_splits)
    d=df.copy()
    for c in FC: d[c]=pd.to_numeric(d[c],errors="coerce").replace([np.inf,-np.inf],np.nan)
    d[FC]=d[FC].fillna(d.loc[d.partition=="train",FC].median())
    return d
# load all
data={}
for comp in ("c23","c40"):
    data[("real",comp)]=load("real",comp)
    for m in MAN: data[(m,comp)]=load(m,comp)
FC=sorted([c for c in data[("real","c23")].columns if c[:2] in ("s_","t_")])
data={k:clean(v,FC) for k,v in data.items()}
def LGBM(): return lgb.LGBMClassifier(n_estimators=200,max_depth=6,learning_rate=0.05,num_leaves=31,min_child_samples=20,class_weight="balanced",random_state=SEED,verbose=-1,n_jobs=-1)
def boot(y,p,n=2000,s=SEED):
    rng=np.random.RandomState(s); b=[]
    for _ in range(n):
        i=rng.randint(0,len(y),len(y))
        if len(np.unique(y[i]))>1: b.append(roc_auc_score(y[i],p[i]))
    return round(np.percentile(b,2.5),4),round(np.percentile(b,97.5),4)
def fit_pred(train_comp,test_comp,manip,cols=None):
    cols=cols or FC
    rtr=data[("real",train_comp)]; mtr=data[(manip,train_comp)]
    rte=data[("real",test_comp)]; mte=data[(manip,test_comp)]
    tr=pd.concat([rtr[rtr.partition.isin(["train","val"])],mtr[mtr.partition.isin(["train","val"])]],ignore_index=True)
    te=pd.concat([rte[rte.partition=="test"],mte[mte.partition=="test"]],ignore_index=True)
    assert_no_identity_overlap([(tr,"train"),(te,"test")])
    sc=StandardScaler().fit(tr[cols].values); clf=LGBM(); clf.fit(sc.transform(tr[cols].values),tr['label'].values.astype(int))
    p=clf.predict_proba(sc.transform(te[cols].values))[:,1]; y=te['label'].values.astype(int)
    return roc_auc_score(y,p),boot(y,p),matthews_corrcoef(y,(p>=.5).astype(int))
def commit():
    try: return subprocess.check_output(["git","rev-parse","--short","HEAD"],text=True).strip()
    except: return "nogit"

rows=[]
print("=== EXP-3 COMPRESSION (50-D, identity-disjoint) ===")
print(f"{'manip':15s} {'c23_AUC':>8s} {'c40_AUC':>8s} {'ΔAUC':>7s} {'c23_MCC':>8s} {'c40_MCC':>8s} {'c23→c40':>8s}")
for m in MAN:
    a23,ci23,mc23=fit_pred("c23","c23",m)
    a40,ci40,mc40=fit_pred("c40","c40",m)
    ax,cix,mcx=fit_pred("c23","c40",m)  # train c23 -> test c40
    rows.append(dict(manipulation=m,c23_auc=round(a23,4),c23_ci=list(ci23),c40_auc=round(a40,4),c40_ci=list(ci40),
        delta_auc=round(a40-a23,4),c23_mcc=round(mc23,4),c40_mcc=round(mc40,4),
        cross_c23train_c40test_auc=round(ax,4)))
    print(f"{m:15s} {a23:8.4f} {a40:8.4f} {a40-a23:+7.4f} {mc23:8.4f} {mc40:8.4f} {ax:8.4f}")

# ---- feature-group degradation: pillar-only AUC c23 vs c40 (avg over manips) ----
print("\n=== feature-group degradation under c40 (pillar-only AUC, avg over 4 manips) ===")
grp=[]
for pil,feats in PILLARS.items():
    a23s=[]; a40s=[]
    for m in MAN:
        a23s.append(fit_pred("c23","c23",m,feats)[0]); a40s.append(fit_pred("c40","c40",m,feats)[0])
    ma23=float(np.mean(a23s)); ma40=float(np.mean(a40s))
    grp.append(dict(pillar=pil,c23_auc=round(ma23,4),c40_auc=round(ma40,4),degradation=round(ma23-ma40,4)))
grp=sorted(grp,key=lambda x:-x["degradation"])
for g in grp[:6]: print(f"  {g['pillar']:22s} c23={g['c23_auc']:.3f} c40={g['c40_auc']:.3f} drop={g['degradation']:+.4f}")

pd.DataFrame(rows).to_csv(f"{OUT}/compression_all_manips.csv",index=False)
pd.DataFrame(grp).to_csv(f"{OUT}/compression_group_degradation.csv",index=False)
json.dump(dict(provenance=dict(script="exp3_compression.py",git_commit=commit(),seed=SEED,date=datetime.date.today().isoformat(),
    protocol="identity-disjoint; per-manip train(train+val ids)/test(test ids); 50-D"),
    per_manip=rows,group_degradation=grp),open(f"{OUT}/compression.json","w"),indent=2)
print(f"\nsaved {OUT}/compression_all_manips.csv, compression_group_degradation.csv, compression.json (commit {commit()})")
