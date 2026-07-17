#!/usr/bin/env python3
"""Reproduction GATE (exp1 per-manipulation), faithful to results/exp1/run_exp1.py.
Reals split from ffpp_original_c23.csv via official FF++ JSON split.
Targets (document_pdf.pdf p2): Deepfakes 0.9709 / Face2Face 0.8818 / FaceSwap 0.9999 / NeuralTextures 0.9991.
"""
import json, os, numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, matthews_corrcoef
import lightgbm as lgb
import sklearn

BASE="/home/iiitn/Downloads/phantom-lens-main"
FEAT=f"{BASE}/features"
SPL=f"{BASE}/Major Revision Results/01_splits/ffpp_official"
print(f"[env] lightgbm={lgb.__version__} sklearn={sklearn.__version__} numpy={np.__version__}")
print(f"[seed] 42 everywhere\n")

# SPLIT: reproduce results/exp2/run_exp2.py:176 EXACTLY —
#   r_train, r_test = train_test_split(real_df, test_size=0.2, random_state=42, shuffle=True)
# (NOT the official JSON split — the original code used random 80/20 seed 42)
from sklearn.model_selection import train_test_split
_real_df = pd.read_csv(f"{FEAT}/ffpp_original_c23.csv")
_rtr, _rte = train_test_split(_real_df, test_size=0.2, random_state=42, shuffle=True)
def real_split(which):
    return (_rtr if which=="train" else _rte).copy()

def load_concat(frames):
    df=pd.concat(frames, ignore_index=True)
    fc=sorted([c for c in df.columns if c.startswith("s_") or c.startswith("t_")])
    df[fc]=df[fc].replace([np.inf,-np.inf], np.nan)
    for c in fc: df[c]=df[c].fillna(df[c].median())
    return df, fc

MANIP={"Deepfakes":"ffpp_deepfakes_c23.csv","Face2Face":"ffpp_face2face_c23.csv",
       "FaceSwap":"ffpp_faceswap_c23.csv","NeuralTextures":"ffpp_neuraltextures_c23.csv"}
TARGET={"Deepfakes":0.9709,"Face2Face":0.8818,"FaceSwap":0.9999,"NeuralTextures":0.9991}

rtrain=real_split("train"); rtest=real_split("test")
manip_df={k:pd.read_csv(f"{FEAT}/{v}") for k,v in MANIP.items()}

# TRAIN = real_train + all 4 manip fakes (exactly like run_exp1.py TRAIN_FILES)
train_df, fc = load_concat([rtrain]+[manip_df[k] for k in MANIP])
Xtr=train_df[fc].values.astype(np.float64); ytr=train_df['label'].values.astype(int)
sc=StandardScaler(); Xtr_s=sc.fit_transform(Xtr)
print(f"[train] {len(ytr)} samples (real={rtrain.shape[0]}, fake={sum(len(manip_df[k]) for k in MANIP)}), feats={len(fc)}")
print(f"[reals] official train={len(rtrain)} test={len(rtest)}\n")

clf=lgb.LGBMClassifier(n_estimators=200,max_depth=6,learning_rate=0.05,num_leaves=31,
                       min_child_samples=20,class_weight="balanced",random_state=42,verbose=-1)
clf.fit(Xtr_s, ytr)

print(f"{'manip':16s} {'AUC':>8s} {'target':>8s} {'Δ':>8s}  {'F1':>7s} {'MCC':>7s}")
res={}
for k,v in MANIP.items():
    te,_=load_concat([rtest, manip_df[k]])
    Xte=sc.transform(te[fc].values.astype(np.float64)); yte=te['label'].values.astype(int)
    p=clf.predict_proba(Xte)[:,1]
    auc=roc_auc_score(yte,p); pred=(p>=0.5).astype(int)
    f1=f1_score(yte,pred,average='macro'); mcc=matthews_corrcoef(yte,pred)
    d=auc-TARGET[k]
    flag = "OK" if abs(d)<0.01 else ("~"if abs(d)<0.03 else "!!")
    print(f"{k:16s} {auc:8.4f} {TARGET[k]:8.4f} {d:+8.4f}  {f1:7.4f} {mcc:7.4f}  {flag}")
    res[k]=auc
print(f"\n[interpretation] |Δ|<0.01 OK · <0.03 close · else investigate.")
print(f"[note] exp3 All-50 0.9939 needs FaceShifter (train)+NeuralTextures(test) — not gated here.")
