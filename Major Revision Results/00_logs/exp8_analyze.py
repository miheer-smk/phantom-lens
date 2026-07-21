#!/usr/bin/env python3
"""EXP-8 analysis (R1). Per residual method (median/gaussian/wavelet):
 descriptor table (face/bg energy, ratio, corr, temporal consistency; real vs fake) +
 classification AUC = [50-D minus 4 PRNU-residual feats] + [method's 5 descriptors], per manip.
BM3D = NOT COMPUTED (no linux-aarch64 lib). Identity-disjoint."""
import os,sys,json,subprocess,datetime
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore"); sys.path.insert(0,"src")
from protocol import make_splits
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
SEED=42; F="features"; OUT="results_clean"
METHODS=["median","gaussian","wavelet"]; DESC=["face_energy","bg_energy","face_bg_ratio","face_bg_corr","temporal_consistency"]
PRNU_RESID=["s_prnu_energy","s_prnu_face_periph","t_prnu_temporal_stability","t_prnu_face_vs_bg"]
MAN=["deepfakes","face2face","faceswap","neuraltextures"]
def base(p): return os.path.basename(str(p))
def load(name):
    o=pd.read_csv(f"{F}/ffpp_{'original' if name=='real' else name}_c23.csv")
    r=pd.read_csv(f"{F}/residual_{'original' if name=='real' else name}_c23.csv")
    o["_b"]=o.video_path.map(base); r["_b"]=r.video_path.map(base)
    return make_splits(o.merge(r[["_b"]+[f"{m}_{d}" for m in METHODS for d in DESC]],on="_b",how="inner"))
real=load("real"); MANd={m:load(m) for m in MAN}
FC=sorted([c for c in real.columns if c[:2] in ("s_","t_")]); base46=[c for c in FC if c not in PRNU_RESID]
allcols=base46+[f"{m}_{d}" for m in METHODS for d in DESC]
def clean(df):  # M1 fix: TRAIN-partition medians only (df already has 'partition' from load->make_splits)
    d=df.copy()
    for c in allcols: d[c]=pd.to_numeric(d[c],errors="coerce").replace([np.inf,-np.inf],np.nan)
    d[allcols]=d[allcols].fillna(d.loc[d.partition=="train",allcols].median())
    return d
real=clean(real); MANd={m:clean(v) for m,v in MANd.items()}
def LGBM(): return lgb.LGBMClassifier(n_estimators=200,max_depth=6,learning_rate=0.05,num_leaves=31,min_child_samples=20,class_weight="balanced",random_state=SEED,verbose=-1,n_jobs=-1)
def commit():
    try: return subprocess.check_output(["git","rev-parse","--short","HEAD"],text=True).strip()
    except: return "nogit"
def perm_auc(cols):
    res={}
    for m,md in MANd.items():
        tr=pd.concat([real[real.partition.isin(["train","val"])],md[md.partition.isin(["train","val"])]],ignore_index=True)
        te=pd.concat([real[real.partition=="test"],md[md.partition=="test"]],ignore_index=True)
        sc=StandardScaler().fit(tr[cols].values); clf=LGBM(); clf.fit(sc.transform(tr[cols].values),tr['label'].values.astype(int))
        p=clf.predict_proba(sc.transform(te[cols].values))[:,1]; res[m]=round(roc_auc_score(te['label'].values.astype(int),p),4)
    return res

# ---- descriptor table (real vs fake, per method) ----
allfake=pd.concat([MANd[m] for m in MAN],ignore_index=True)
desc_rows=[]
for m in METHODS:
    for d in DESC:
        col=f"{m}_{d}"
        desc_rows.append(dict(method=m,descriptor=d,real_mean=round(float(real[col].mean()),4),fake_mean=round(float(allfake[col].mean()),4)))
# ---- classification AUC per method ----
auc_rows=[]
for m in METHODS:
    cols=base46+[f"{m}_{d}" for d in DESC]
    a=perm_auc(cols); auc_rows.append(dict(residual_method=m,**{f"auc_{k}":v for k,v in a.items()},auc_mean=round(np.mean(list(a.values())),4)))
# reference: current 50-D (median-based, includes PRNU features)
cur=perm_auc(FC); auc_rows.append(dict(residual_method="current_50D_ref",**{f"auc_{k}":v for k,v in cur.items()},auc_mean=round(np.mean(list(cur.values())),4)))
# BM3D not computed
auc_rows.append(dict(residual_method="bm3d",auc_deepfakes="NOT COMPUTED — BM3D library unavailable (no linux-aarch64 binary)"))

pd.DataFrame(desc_rows).to_csv(f"{OUT}/prnu_descriptors.csv",index=False)
pd.DataFrame(auc_rows).to_csv(f"{OUT}/prnu_comparison.csv",index=False)
json.dump(dict(provenance=dict(script="exp8_analyze.py",git_commit=commit(),seed=SEED,date=datetime.date.today().isoformat(),
    bm3d="NOT COMPUTED — no linux-aarch64 native library",terminology_note="descriptors are PRNU-INSPIRED residual-energy, not camera PRNU fingerprints"),
    descriptors=desc_rows,classification=auc_rows),open(f"{OUT}/prnu_comparison.json","w"),indent=2)
print("=== EXP-8 PRNU-INSPIRED RESIDUAL COMPARISON (R1) ===")
print("descriptor means (real vs fake):")
for r in desc_rows: print(f"  {r['method']:9s} {r['descriptor']:20s} real={r['real_mean']:.4f} fake={r['fake_mean']:.4f}")
print("\nclassification AUC per manip (base46 + method's 5 descriptors):")
for r in auc_rows:
    if 'auc_mean' in r: print(f"  {r['residual_method']:16s} DF={r.get('auc_deepfakes')} F2F={r.get('auc_face2face')} FS={r.get('auc_faceswap')} NT={r.get('auc_neuraltextures')} mean={r['auc_mean']}")
    else: print(f"  {r['residual_method']:16s} {r.get('auc_deepfakes')}")
print(f"saved {OUT}/prnu_descriptors.csv, prnu_comparison.csv, prnu_comparison.json (commit {commit()})")
