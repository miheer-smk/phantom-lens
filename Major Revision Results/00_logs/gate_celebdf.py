#!/usr/bin/env python3
"""CelebDF cross-dataset GATE — reproduce the 0.6989 zero-shot number.
This is CLEAN by construction: CelebDF is NEVER in training (zero-shot), so the
train=test leakage that inflates the per-manip in-distribution numbers does NOT apply here.
Train: FF++ real_train + all 4 manip fakes (same as run_celebdf_eval.py). Test: CelebDF (frozen).
"""
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, matthews_corrcoef, recall_score, average_precision_score
import lightgbm as lgb
FEAT="features"
real=pd.read_csv(f"{FEAT}/ffpp_original_c23.csv")
rtr,_=train_test_split(real,test_size=0.2,random_state=42,shuffle=True)
manips=["ffpp_deepfakes_c23.csv","ffpp_face2face_c23.csv","ffpp_faceswap_c23.csv","ffpp_neuraltextures_c23.csv"]
train=pd.concat([rtr]+[pd.read_csv(f"{FEAT}/{m}") for m in manips], ignore_index=True)
cd=pd.read_csv(f"{FEAT}/celebdf_features.csv")
fc=sorted([c for c in train.columns if c[:2] in ("s_","t_")])
def clean(df):
    d=df.copy(); d[fc]=d[fc].replace([np.inf,-np.inf],np.nan)
    for c in fc: d[c]=d[c].fillna(d[c].median())
    return d
train=clean(train); cd=clean(cd)
Xtr=train[fc].values.astype(float); ytr=train['label'].values.astype(int)
sc=StandardScaler(); Xtr=sc.fit_transform(Xtr)
clf=lgb.LGBMClassifier(n_estimators=200,max_depth=6,learning_rate=0.05,num_leaves=31,
                       min_child_samples=20,class_weight="balanced",random_state=42,verbose=-1)
clf.fit(Xtr,ytr)
Xcd=sc.transform(cd[fc].values.astype(float)); ycd=cd['label'].values.astype(int)
p=clf.predict_proba(Xcd)[:,1]; pred=(p>=0.5).astype(int)
auc=roc_auc_score(ycd,p)
print(f"[env] lightgbm={lgb.__version__}")
print(f"CelebDF zero-shot (n={len(ycd)}: real={int((ycd==0).sum())} fake={int((ycd==1).sum())})")
print(f"  AUC          = {auc:.4f}   (published 0.6989)   Δ={auc-0.6989:+.4f}")
print(f"  AvgPrecision = {average_precision_score(ycd,p):.4f}   (published 0.9243)")
print(f"  macro-F1     = {f1_score(ycd,pred,average='macro'):.4f}   (published 0.6252)")
print(f"  real recall  = {recall_score(ycd,pred,pos_label=0):.4f}   (published 0.4020)")
print(f"  fake recall  = {recall_score(ycd,pred,pos_label=1):.4f}   (published 0.8745)")
print(f"  MCC          = {matthews_corrcoef(ycd,pred):.4f}   (published 0.2537)")
print(f"\n[note] This number is CLEAN (zero-shot, no train=test leakage). It is the honest")
print(f"       cross-dataset generalization result and is NOT affected by the exp1 leak.")
