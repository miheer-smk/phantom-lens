#!/usr/bin/env python3
"""Per-pillar ablation (Reviewers 1/3/5) — remove-one-pillar-at-a-time.
20 pillars (6 spatial P1/P2/P4/P6/P8/P9 + 14 temporal T1-T14) x {DF,F2F,FS,NT,CelebDF}.
For each cell: full-50 AUC, leave-one-pillar-out AUC, ΔAUC (=full-ablated; +ve => pillar helps),
with PAIRED bootstrap 95% CI on ΔAUC. Identity-disjoint. Column-masking only — the 50-D vector,
CSVs, and hashes are unchanged (removing a pillar = selecting the other columns for one run).
"""
import os,sys,json,hashlib,subprocess,datetime
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore"); sys.path.insert(0,"src")
from protocol import make_splits
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
SEED=42; F="features"; OUT="results_clean"; os.makedirs(OUT,exist_ok=True)
PILLARS=json.load(open("splits/pillar_map.json"))
MAN={"Deepfakes":"ffpp_deepfakes_c23.csv","Face2Face":"ffpp_face2face_c23.csv",
     "FaceSwap":"ffpp_faceswap_c23.csv","NeuralTextures":"ffpp_neuraltextures_c23.csv"}
real=pd.read_csv(f"{F}/ffpp_original_c23.csv"); cd=pd.read_csv(f"{F}/celebdf_features.csv")
FC=sorted([c for c in real.columns if c[:2] in ("s_","t_")])
def clean(df):
    d=df.copy(); d[FC]=d[FC].replace([np.inf,-np.inf],np.nan)
    for c in FC: d[c]=d[c].fillna(d[c].median())
    return d
real=make_splits(clean(real)); MANd={k:make_splits(clean(pd.read_csv(f"{F}/{v}"))) for k,v in MAN.items()}; cd=clean(cd)
def LGBM(): return lgb.LGBMClassifier(n_estimators=200,max_depth=6,learning_rate=0.05,num_leaves=31,
    min_child_samples=20,class_weight="balanced",random_state=SEED,verbose=-1,n_jobs=4)
def commit():
    try: return subprocess.check_output(["git","rev-parse","--short","HEAD"],text=True).strip()
    except: return "nogit"

def preds(cols, Xtr,ytr,Xte):
    sc=StandardScaler().fit(Xtr[:,cols]); clf=LGBM(); clf.fit(sc.transform(Xtr[:,cols]),ytr)
    return clf.predict_proba(sc.transform(Xte[:,cols]))[:,1]

def paired_delta_ci(y,pf,pa,n=1500,s=SEED):
    rng=np.random.RandomState(s); d=[]
    for _ in range(n):
        i=rng.randint(0,len(y),len(y))
        if len(np.unique(y[i]))<2: continue
        d.append(roc_auc_score(y[i],pf[i])-roc_auc_score(y[i],pa[i]))
    return float(np.percentile(d,2.5)), float(np.percentile(d,97.5))

def eval_dataset(name, train_frames, test_frames):
    tr=pd.concat(train_frames,ignore_index=True); te=pd.concat(test_frames,ignore_index=True)
    Xtr=tr[FC].values.astype(float); ytr=tr['label'].values.astype(int)
    Xte=te[FC].values.astype(float); yte=te['label'].values.astype(int)
    idx={f:i for i,f in enumerate(FC)}
    pf=preds(list(range(len(FC))),Xtr,ytr,Xte); full=roc_auc_score(yte,pf)
    rows=[]
    for pil,feats in PILLARS.items():
        keep=[i for f,i in idx.items() if f not in feats]
        pa=preds(keep,Xtr,ytr,Xte); ab=roc_auc_score(yte,pa)
        lo,hi=paired_delta_ci(yte,pf,pa)
        rows.append(dict(dataset=name,pillar=pil,n_pillar_feats=len(feats),full_auc=round(full,4),
            loGo_auc=round(ab,4),delta_auc=round(full-ab,4),delta_ci_lo=round(lo,4),delta_ci_hi=round(hi,4),
            n_test=int(len(yte))))
        print(f"  {name:14s} {pil:22s} full={full:.4f} loGo={ab:.4f} Δ={full-ab:+.4f} CI[{lo:+.4f},{hi:+.4f}]",flush=True)
    return rows

allrows=[]
print("="*80+"\nPER-PILLAR ABLATION (identity-disjoint, ΔAUC=full-leaveOneOut, paired bootstrap CI)\n"+"="*80)
for m,mdf in MANd.items():
    allrows+=eval_dataset(m,[real[real.partition=="train"],mdf[mdf.partition=="train"]],
                            [real[real.partition=="test"], mdf[mdf.partition=="test"]])
# CelebDF zero-shot: train FF++ (train ids, real+all manips) -> test CelebDF
trf=[real[real.partition=="train"]]+[MANd[m][MANd[m].partition=="train"] for m in MAN]
allrows+=eval_dataset("CelebDF",trf,[cd])
df=pd.DataFrame(allrows); df.to_csv(f"{OUT}/pillar_ablation.csv",index=False)
prov=dict(script="Major Revision Results/00_logs/pillar_ablation.py",git_commit=commit(),seed=SEED,
    date=datetime.date.today().isoformat(),protocol="identity-disjoint; column-mask only; 50-D unchanged",
    n_pillars=len(PILLARS),classifier="LightGBM(200,6,0.05,31,20,balanced)")
json.dump(dict(provenance=prov,rows=allrows),open(f"{OUT}/pillar_ablation.json","w"),indent=1)
print(f"\nWrote {OUT}/pillar_ablation.csv ({len(allrows)} rows) commit {prov['git_commit']}")
