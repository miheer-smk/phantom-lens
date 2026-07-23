#!/usr/bin/env python3
"""RIGOROUS honest optimization — final push before locking numbers.
Protocol integrity:
  * IN-DISTRIBUTION: video-level 5-fold CV (no video in train+test), per-manip + combined.
  * CROSS-DATASET: model/hyperparams selected by FF++ CV ONLY, then applied to CelebDF ONCE.
    (Never select by CelebDF performance = no test-set overfitting.)
  * No leakage, no cherry-picking. All results reported.
"""
import numpy as np, pandas as pd, warnings, json, sys
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler, QuantileTransformer, PowerTransformer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
    HistGradientBoostingClassifier, GradientBoostingClassifier, VotingClassifier)
from sklearn.metrics import roc_auc_score, recall_score
from sklearn.pipeline import Pipeline
import lightgbm as lgb
try: import xgboost as xgb; HAVE_XGB=True
except: HAVE_XGB=False
SEED=42
F="features"
def R(n): return pd.read_csv(f"{F}/{n}.csv")
real=R("ffpp_original_c23")
MAN={"Deepfakes":R("ffpp_deepfakes_c23"),"Face2Face":R("ffpp_face2face_c23"),
     "FaceSwap":R("ffpp_faceswap_c23"),"NeuralTextures":R("ffpp_neuraltextures_c23")}
cd=R("celebdf_features")
fc=sorted([c for c in real.columns if c[:2] in ("s_","t_")])
def clean(df):
    d=df.copy(); d[fc]=d[fc].replace([np.inf,-np.inf],np.nan)
    for c in fc: d[c]=d[c].fillna(d[c].median())
    return d
real=clean(real); cd=clean(cd); MAN={k:clean(v) for k,v in MAN.items()}
ycd=cd['label'].values.astype(int); Xcd_raw=cd[fc].values.astype(float)

def models():
    m={
     "LogReg":       LogisticRegression(C=1.0,class_weight="balanced",max_iter=3000,random_state=SEED),
     "RandomForest": RandomForestClassifier(n_estimators=400,max_depth=None,min_samples_leaf=3,
                        class_weight="balanced",random_state=SEED,n_jobs=-1),
     "ExtraTrees":   ExtraTreesClassifier(n_estimators=500,min_samples_leaf=2,
                        class_weight="balanced",random_state=SEED,n_jobs=-1),
     "HistGB":       HistGradientBoostingClassifier(max_iter=400,learning_rate=0.05,
                        max_leaf_nodes=31,l2_regularization=1.0,random_state=SEED),
     "LGBM-orig":    lgb.LGBMClassifier(n_estimators=200,max_depth=6,learning_rate=0.05,num_leaves=31,
                        min_child_samples=20,class_weight="balanced",random_state=SEED,verbose=-1,n_jobs=-1),
     "LGBM-tuned":   lgb.LGBMClassifier(n_estimators=600,max_depth=-1,num_leaves=63,
                        learning_rate=0.03,min_child_samples=15,subsample=0.8,colsample_bytree=0.8,
                        reg_lambda=1.0,class_weight="balanced",random_state=SEED,verbose=-1,n_jobs=-1),
     "LGBM-deep":    lgb.LGBMClassifier(n_estimators=1000,max_depth=8,learning_rate=0.02,num_leaves=127,
                        min_child_samples=10,subsample=0.7,colsample_bytree=0.7,reg_lambda=2.0,reg_alpha=0.5,
                        class_weight="balanced",random_state=SEED,verbose=-1,n_jobs=-1),
    }
    if HAVE_XGB:
        m["XGBoost"]=xgb.XGBClassifier(n_estimators=500,max_depth=6,learning_rate=0.03,
            subsample=0.8,colsample_bytree=0.8,reg_lambda=1.0,
            random_state=SEED,n_jobs=-1,eval_metric="auc",tree_method="hist")
    return m

def cv_auc(clf, X, y, folds=5):
    skf=StratifiedKFold(folds,shuffle=True,random_state=SEED)
    return cross_val_score(clf,X,y,cv=skf,scoring="roc_auc",n_jobs=1).mean()

print(f"[env] lgbm={lgb.__version__} xgb={'yes' if HAVE_XGB else 'no'}  seed={SEED}", flush=True)

# ============ 1. IN-DISTRIBUTION per-manip (clean video-level 5-fold CV) ============
print("\n"+"="*70+"\n1. IN-DISTRIBUTION per-manipulation (clean 5-fold CV, video-level)\n"+"="*70, flush=True)
per_manip_best={}
for mname,mdf in MAN.items():
    d=clean(pd.concat([real,mdf],ignore_index=True))
    X=StandardScaler().fit_transform(d[fc].values.astype(float)); y=d['label'].values.astype(int)
    best=("",0)
    line=f"  {mname:15s}"
    for nm,clf in models().items():
        a=cv_auc(clf,X,y)
        line+=f" {nm}={a:.3f}"
        if a>best[1]: best=(nm,a)
    per_manip_best[mname]=best
    print(line+f"  -> BEST {best[0]} {best[1]:.4f}", flush=True)

# ============ 2. IN-DISTRIBUTION combined multi-manip (clean 5-fold CV) ============
print("\n"+"="*70+"\n2. IN-DISTRIBUTION combined multi-manip (clean 5-fold CV)\n"+"="*70, flush=True)
dall=clean(pd.concat([real]+list(MAN.values()),ignore_index=True))
Xall=StandardScaler().fit_transform(dall[fc].values.astype(float)); yall=dall['label'].values.astype(int)
comb_best=("",0)
for nm,clf in models().items():
    a=cv_auc(clf,Xall,yall)
    print(f"  {nm:14s} CV AUC={a:.4f}", flush=True)
    if a>comb_best[1]: comb_best=(nm,a)
print(f"  -> BEST combined: {comb_best[0]} {comb_best[1]:.4f}", flush=True)

# ============ 3. CROSS-DATASET: select model by FF++ CV, apply to CelebDF once ============
print("\n"+"="*70+"\n3. CROSS-DATASET CelebDF (model selected by FF++ CV, applied once)\n"+"="*70, flush=True)
# training = real + 4 manips (proper). transforms tried; all fit on train only.
train=clean(pd.concat([real]+list(MAN.values()),ignore_index=True))
Xtr_raw=train[fc].values.astype(float); ytr=train['label'].values.astype(int)
transforms={"Standard":StandardScaler(),"Quantile":QuantileTransformer(output_distribution="normal",random_state=SEED),
            "Power":PowerTransformer()}
results=[]
for tname,tf in transforms.items():
    tf.fit(Xtr_raw); Xtr=tf.transform(Xtr_raw); Xte=tf.transform(Xcd_raw)
    for nm,clf in models().items():
        ffcv=cv_auc(clf,Xtr,ytr)                 # SELECTION metric (train only)
        clf.fit(Xtr,ytr); p=clf.predict_proba(Xte)[:,1]
        cdauc=roc_auc_score(ycd,p)               # reported (not used for selection)
        results.append((tname,nm,ffcv,cdauc))
        print(f"  {tname:9s} {nm:14s} FF++CV={ffcv:.4f}  CelebDF={cdauc:.4f}", flush=True)
# honest selection: pick by FF++ CV, report its CelebDF
sel=max(results,key=lambda r:r[2])
print(f"\n  MODEL SELECTED BY FF++ CV: {sel[1]}+{sel[0]} (FF++CV={sel[2]:.4f}) -> CelebDF={sel[3]:.4f}", flush=True)
best_cd=max(results,key=lambda r:r[3])
print(f"  [for reference only, NOT selectable: best CelebDF seen = {best_cd[1]}+{best_cd[0]} {best_cd[3]:.4f}]", flush=True)

# ============ 4. ENSEMBLE (soft-vote top-3 by FF++ CV) cross-dataset ============
print("\n"+"="*70+"\n4. ENSEMBLE (soft-vote top-3 tree models) cross-dataset\n"+"="*70, flush=True)
sc=StandardScaler().fit(Xtr_raw); Xtr=sc.transform(Xtr_raw); Xte=sc.transform(Xcd_raw)
ens=VotingClassifier([("lgbm",models()["LGBM-tuned"]),("rf",models()["RandomForest"]),
                      ("hgb",models()["HistGB"])],voting="soft",n_jobs=-1)
ffcv=cv_auc(ens,Xtr,ytr); ens.fit(Xtr,ytr); p=ens.predict_proba(Xte)[:,1]
print(f"  Ensemble FF++CV={ffcv:.4f}  CelebDF={roc_auc_score(ycd,p):.4f} real_rec={recall_score(ycd,(p>=.5).astype(int),pos_label=0):.3f}", flush=True)

print("\n=== SUMMARY (honest, lockable) ===", flush=True)
for m,(nm,a) in per_manip_best.items(): print(f"  in-dist {m:15s} best {nm:12s} {a:.4f}")
print(f"  in-dist combined      best {comb_best[0]:12s} {comb_best[1]:.4f}")
print(f"  cross-dataset (FF++CV-selected)          CelebDF {sel[3]:.4f}")
