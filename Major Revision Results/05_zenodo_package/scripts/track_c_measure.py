#!/usr/bin/env python3
"""Track C measurement — honest incremental value of ROI candidate feature groups.
Protocol: fit on TRAIN identities, evaluate on VALIDATION identities (test + CelebDF untouched).
Reports, per target manipulation (Face2Face, NeuralTextures):
  * baseline val-AUC on the LOCKED original 50 features
  * val-AUC of 50 + each group G_i separately (incremental delta)  [user requirement 2a]
  * val-AUC of 50 + all groups
  * Cohen's d (real vs fake, val split) for every new ROI feature   [Reviewer 4 / requirement 2b]
The original 50-D vector is NOT modified; ROI features are an additive extended set only.
"""
import os, sys, json, hashlib, subprocess, datetime
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "src")
from protocol import make_splits, load_id2split
import roi_config as RC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
SEED=42
F="features"
def base(p): return os.path.basename(str(p))
def sha(p):
    h=hashlib.sha256();
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()[:16]
def commit():
    try: return subprocess.check_output(["git","rev-parse","--short","HEAD"],text=True).strip()
    except: return "nogit"

id2split=load_id2split()
orig={"real":"ffpp_original_c23.csv","Face2Face":"ffpp_face2face_c23.csv","NeuralTextures":"ffpp_neuraltextures_c23.csv"}
roi ={"real":"roi_original_c23.csv","Face2Face":"roi_face2face_c23.csv","NeuralTextures":"roi_neuraltextures_c23.csv"}
for d in list(orig.values())+list(roi.values()):
    if not os.path.exists(f"{F}/{d}"): sys.exit(f"missing {d} (ROI extraction not finished)")

O={k:pd.read_csv(f"{F}/{v}") for k,v in orig.items()}
R={k:pd.read_csv(f"{F}/{v}") for k,v in roi.items()}
FC=sorted([c for c in O["real"].columns if c[:2] in ("s_","t_")])
ROI_FEATS=RC.ROI_FEATURE_NAMES
def merged(k):
    o=O[k].copy(); r=R[k].copy()
    o["_b"]=o["video_path"].map(base); r["_b"]=r["video_path"].map(base)
    m=o.merge(r[["_b"]+ROI_FEATS], on="_b", how="inner")
    m=make_splits(m)  # partition by identity (uses video_path from original)
    for c in FC+ROI_FEATS:
        m[c]=pd.to_numeric(m[c],errors="coerce").replace([np.inf,-np.inf],np.nan)
        m[c]=m[c].fillna(m[c].median())
    return m
M={k:merged(k) for k in orig}

def LGBM(): return lgb.LGBMClassifier(n_estimators=200,max_depth=6,learning_rate=0.05,num_leaves=31,
    min_child_samples=20,class_weight="balanced",random_state=SEED,verbose=-1,n_jobs=-1)

def val_auc(manip, cols):
    real=M["real"]; man=M[manip]
    tr=pd.concat([real[real.partition=="train"], man[man.partition=="train"]],ignore_index=True)
    va=pd.concat([real[real.partition=="val"],   man[man.partition=="val"]],  ignore_index=True)
    ytr=np.r_[np.zeros((real.partition=="train").sum()), np.ones((man.partition=="train").sum())]
    yva=np.r_[np.zeros((real.partition=="val").sum()),   np.ones((man.partition=="val").sum())]
    sc=StandardScaler().fit(tr[cols].values); clf=LGBM(); clf.fit(sc.transform(tr[cols].values),ytr)
    p=clf.predict_proba(sc.transform(va[cols].values))[:,1]
    return roc_auc_score(yva,p)

def cohens_d(manip, feat):
    real=M["real"]; man=M[manip]
    a=man[man.partition=="val"][feat].values.astype(float)  # fake
    b=real[real.partition=="val"][feat].values.astype(float) # real
    na,nb=len(a),len(b)
    sp=np.sqrt(((na-1)*a.std(ddof=1)**2+(nb-1)*b.std(ddof=1)**2)/(na+nb-2)+1e-12)
    return float((a.mean()-b.mean())/sp)

out={"provenance":{"script":"Major Revision Results/00_logs/track_c_measure.py","git_commit":commit(),
     "seed":SEED,"date":datetime.date.today().isoformat(),"protocol":"fit train / eval val (identity-disjoint); test+CelebDF untouched",
     "roi_csv_sha256":{v:sha(f"{F}/{v}") for v in roi.values()},"note":"original 50-D locked & unmodified; ROI additive only"},
     "results":{}}
print("="*72+"\nTRACK C — incremental val-AUC of ROI groups + Cohen's d (val split)\n"+"="*72)
for manip in ["Face2Face","NeuralTextures"]:
    b50=val_auc(manip, FC)
    row={"baseline_50_valAUC":round(b50,4),"groups":{},"all_roi_valAUC":round(val_auc(manip, FC+ROI_FEATS),4),"cohens_d":{}}
    print(f"\n### {manip}  (baseline 50-D val-AUC = {b50:.4f})")
    for gname,gfeats in RC.CANDIDATE_GROUPS.items():
        a=val_auc(manip, FC+gfeats)
        row["groups"][gname]={"valAUC":round(a,4),"delta":round(a-b50,4),"features":gfeats}
        print(f"  +{gname:24s} val-AUC={a:.4f}  Δ={a-b50:+.4f}")
    print(f"  +ALL ROI ({len(ROI_FEATS)})           val-AUC={row['all_roi_valAUC']:.4f}  Δ={row['all_roi_valAUC']-b50:+.4f}")
    print(f"  Cohen's d (val, real vs fake):")
    for f in ROI_FEATS:
        d=cohens_d(manip,f); row["cohens_d"][f]=round(d,3)
        mag="large" if abs(d)>=0.8 else "medium" if abs(d)>=0.5 else "small" if abs(d)>=0.2 else "negligible"
        print(f"     {f:34s} d={d:+.3f} ({mag})")
    out["results"][manip]=row
os.makedirs("results_clean",exist_ok=True)
json.dump(out, open("results_clean/track_c.json","w"), indent=1)
print(f"\nWrote results_clean/track_c.json (commit {out['provenance']['git_commit']})")
