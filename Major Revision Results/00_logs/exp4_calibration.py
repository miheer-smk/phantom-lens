#!/usr/bin/env python3
"""EXP-4 Threshold calibration (R1/R5.3). Answers the real-recall-collapse criticism.
ALL thresholds/calibrators derived on FF++ VALIDATION split only, then applied UNCHANGED to
CelebDF (and WildDeepfake). Never selected on test labels (§0 rule 2). 53-D model.
Configs: 0.50, Youden-J, val-macroF1-max, val-balacc-max, Platt, isotonic."""
import os,sys,json,hashlib,subprocess,datetime
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore"); sys.path.insert(0,"src")
from protocol import make_splits, assert_no_identity_overlap
import roi_config as RC
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, f1_score, recall_score, matthews_corrcoef, roc_curve, balanced_accuracy_score
import lightgbm as lgb
SEED=42; F="features"; OUT="results_clean"
def base(p): return os.path.basename(str(p))
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()[:16]
def commit():
    try: return subprocess.check_output(["git","rev-parse","--short","HEAD"],text=True).strip()
    except: return "nogit"
G1=RC.CANDIDATE_GROUPS["G1_mouth_instability"]
MAN=["deepfakes","face2face","faceswap","neuraltextures"]
def with_g1(name):
    o=pd.read_csv(f"{F}/ffpp_{name}_c23.csv") if name!="real" else pd.read_csv(f"{F}/ffpp_original_c23.csv")
    r=pd.read_csv(f"{F}/roi_{'original' if name=='real' else name}_c23.csv")
    o["_b"]=o.video_path.map(base); r["_b"]=r.video_path.map(base)
    return make_splits(o.merge(r[["_b"]+G1],on="_b",how="inner"))
real=with_g1("real"); MANd={m:with_g1(m) for m in MAN}
FC=sorted([c for c in real.columns if c[:2] in ("s_","t_")]); COLS=FC+G1
def clean(df):
    d=df.copy()
    for c in COLS: d[c]=pd.to_numeric(d[c],errors="coerce").replace([np.inf,-np.inf],np.nan); d[c]=d[c].fillna(d[c].median())
    return d
real=clean(real); MANd={m:clean(v) for m,v in MANd.items()}

# identity-disjoint assertion (train/val/test)
assert_no_identity_overlap([(real[real.partition=="train"],"train"),(real[real.partition=="val"],"val"),(real[real.partition=="test"],"test")]
    +[(MANd[m][MANd[m].partition==p],p) for m in MAN for p in ("train","val","test")])
print("identity-disjoint assertion PASSED",flush=True)

# train 53-D on TRAIN identities (real+all manip)
tr=pd.concat([real[real.partition=="train"]]+[MANd[m][MANd[m].partition=="train"] for m in MAN],ignore_index=True)
va=pd.concat([real[real.partition=="val"]]+[MANd[m][MANd[m].partition=="val"] for m in MAN],ignore_index=True)
sc=StandardScaler().fit(tr[COLS].values)
clf=lgb.LGBMClassifier(n_estimators=200,max_depth=6,learning_rate=0.05,num_leaves=31,min_child_samples=20,class_weight="balanced",random_state=SEED,verbose=-1,n_jobs=-1)
clf.fit(sc.transform(tr[COLS].values),tr['label'].values.astype(int))
pv=clf.predict_proba(sc.transform(va[COLS].values))[:,1]; yv=va['label'].values.astype(int)  # VAL scores (for deriving thresholds)

# CelebDF test scores (labels NOT used for any threshold selection)
cd=clean(pd.read_csv(f"{F}/celebdf_features.csv")); pc=clf.predict_proba(sc.transform(cd[COLS].values))[:,1]; yc=cd['label'].values.astype(int)

# ---- derive thresholds/calibrators on VAL only ----
fpr,tpr,thr=roc_curve(yv,pv); youden=thr[np.argmax(tpr-fpr)]
grid=np.linspace(0.01,0.99,99)
f1_th=grid[np.argmax([f1_score(yv,(pv>=t).astype(int),average='macro') for t in grid])]
ba_th=grid[np.argmax([balanced_accuracy_score(yv,(pv>=t).astype(int)) for t in grid])]
platt=LogisticRegression().fit(pv.reshape(-1,1),yv)          # Platt on val
iso=IsotonicRegression(out_of_bounds='clip').fit(pv,yv)      # isotonic on val

def metrics(y,p,thresh=0.5,calibrated_p=None):
    pp = calibrated_p if calibrated_p is not None else p
    pred=(pp>=thresh).astype(int)
    return dict(auc=round(roc_auc_score(y,p),4),macro_f1=round(f1_score(y,pred,average='macro'),4),
        real_recall=round(recall_score(y,pred,pos_label=0),4),fake_recall=round(recall_score(y,pred,pos_label=1),4),
        mcc=round(matthews_corrcoef(y,pred),4))

configs=[]
configs.append(("theta_0.50", metrics(yc,pc,0.5)))
configs.append(("Youden_J(val)", metrics(yc,pc,float(youden))))
configs.append(("val_macroF1_max", metrics(yc,pc,float(f1_th))))
configs.append(("val_balacc_max", metrics(yc,pc,float(ba_th))))
configs.append(("Platt(val)", metrics(yc,pc,0.5,platt.predict_proba(pc.reshape(-1,1))[:,1])))
configs.append(("isotonic(val)", metrics(yc,pc,0.5,iso.predict(pc))))

rows=[dict(config=n,threshold_source="FF++ val only",**m) for n,m in configs]
pd.DataFrame(rows).to_csv(f"{OUT}/calibration.csv",index=False)
prov=dict(script="exp4_calibration.py",git_commit=commit(),seed=SEED,date=datetime.date.today().isoformat(),
    rule="ALL thresholds/calibrators derived on FF++ VAL only, applied unchanged to CelebDF; test labels NEVER used for selection",
    thresholds=dict(youden=round(float(youden),4),val_macroF1=round(float(f1_th),4),val_balacc=round(float(ba_th),4)),
    feature_csv_sha256={"celebdf_features.csv":sha(f"{F}/celebdf_features.csv")})
json.dump(dict(provenance=prov,celebdf=rows),open(f"{OUT}/calibration.json","w"),indent=2)
print("\n=== EXP-4 CALIBRATION — CelebDF (thresholds from FF++ val only) ===")
print(f"{'config':18s} {'AUC':>7s} {'macroF1':>8s} {'real_rec':>9s} {'fake_rec':>9s} {'MCC':>7s}")
for n,m in configs:
    print(f"{n:18s} {m['auc']:7.4f} {m['macro_f1']:8.4f} {m['real_recall']:9.4f} {m['fake_recall']:9.4f} {m['mcc']:7.4f}")
print("\nNOTE: AUC identical across rows (threshold changes ranking-independent classification only).")
print(f"saved {OUT}/calibration.csv, calibration.json (commit {commit()})")
