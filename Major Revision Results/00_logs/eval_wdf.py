#!/usr/bin/env python3
"""WildDeepfake zero-shot evaluation. FF++-trained 53-D model -> WildDeepfake test (held out).
Reports AUC + per-class recall (real & fake separately) to test whether the CelebDF
real-class-mismatch (low real recall) generalizes. No tuning on WildDeepfake (§0)."""
import os,sys,json,hashlib,subprocess,datetime
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore"); sys.path.insert(0,"src")
from protocol import make_splits; import roi_config as RC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, recall_score, f1_score, matthews_corrcoef, average_precision_score
import lightgbm as lgb
SEED=42; F="features"
def base(p): return os.path.basename(str(p))
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()[:16]
G1=RC.CANDIDATE_GROUPS["G1_mouth_instability"]
MAN=["deepfakes","face2face","faceswap","neuraltextures"]
def with_g1(name):
    o=pd.read_csv(f"{F}/ffpp_{name}_c23.csv") if name!="real" else pd.read_csv(f"{F}/ffpp_original_c23.csv")
    r=pd.read_csv(f"{F}/roi_{'original' if name=='real' else name}_c23.csv")
    o["_b"]=o.video_path.map(base); r["_b"]=r.video_path.map(base)
    return make_splits(o.merge(r[["_b"]+G1],on="_b",how="inner"))
real=with_g1("real"); MANd={m:with_g1(m) for m in MAN}
FC=sorted([c for c in real.columns if c[:2] in ("s_","t_")]); COLS=FC+G1
wdf=pd.read_csv(f"{F}/wilddeepfake_test_53d.csv")
def clean(df):
    d=df.copy()
    for c in COLS: d[c]=pd.to_numeric(d[c],errors="coerce").replace([np.inf,-np.inf],np.nan); d[c]=d[c].fillna(d[c].median())
    return d
# train FF++ 53-D on TRAIN identities (real + all manips), zero-shot -> WDF
tr=pd.concat([clean(real)[clean(real).partition=="train"]]+[clean(MANd[m])[clean(MANd[m]).partition=="train"] for m in MAN],ignore_index=True)
Xtr=tr[COLS].values.astype(float); ytr=tr['label'].values.astype(int)
sc=StandardScaler().fit(Xtr)
clf=lgb.LGBMClassifier(n_estimators=200,max_depth=6,learning_rate=0.05,num_leaves=31,min_child_samples=20,class_weight="balanced",random_state=SEED,verbose=-1,n_jobs=-1)
clf.fit(sc.transform(Xtr),ytr)
w=clean(wdf); Xw=sc.transform(w[COLS].values.astype(float)); yw=w['label'].values.astype(int)
p=clf.predict_proba(Xw)[:,1]; pred=(p>=0.5).astype(int)
# degeneracy: fraction of zero features per row (face-crop caveat)
zero_frac=float((w[COLS].values==0).mean())
def boot(y,pp,n=2000,s=SEED):
    rng=np.random.RandomState(s); b=[]
    for _ in range(n):
        i=rng.randint(0,len(y),len(y))
        if len(np.unique(y[i]))>1: b.append(roc_auc_score(y[i],pp[i]))
    return round(np.percentile(b,2.5),4),round(np.percentile(b,97.5),4)
res=dict(dataset="WildDeepfake_test",n=int(len(yw)),n_real=int((yw==0).sum()),n_fake=int((yw==1).sum()),
    auc=round(roc_auc_score(yw,p),4),auc_ci95=list(boot(yw,p)),ap=round(average_precision_score(yw,p),4),
    real_recall=round(recall_score(yw,pred,pos_label=0),4),fake_recall=round(recall_score(yw,pred,pos_label=1),4),
    macro_f1=round(f1_score(yw,pred,average='macro'),4),mcc=round(matthews_corrcoef(yw,pred),4),
    zero_feature_fraction=round(zero_frac,3))
def commit():
    try: return subprocess.check_output(["git","rev-parse","--short","HEAD"],text=True).strip()
    except: return "nogit"
res["provenance"]=dict(script="eval_wdf.py",git_commit=commit(),seed=SEED,date=datetime.date.today().isoformat(),
    train="FF++ 53-D (train identities, real+4manip+G1)",test="WildDeepfake test (zero-shot, no tuning)",
    wdf_csv_sha256=sha(f"{F}/wilddeepfake_test_53d.csv"))
os.makedirs("results_clean",exist_ok=True); json.dump(res,open("results_clean/zeroshot_wilddeepfake.json","w"),indent=1)
print("=== WildDeepfake ZERO-SHOT (FF++ 53-D -> WildDeepfake test) ===")
print(f"  n={res['n']} (real={res['n_real']}, fake={res['n_fake']})  [zero-feature fraction={res['zero_feature_fraction']} — face-crop caveat]")
print(f"  AUC        = {res['auc']}  CI{res['auc_ci95']}   AP={res['ap']}")
print(f"  real recall= {res['real_recall']}   fake recall= {res['fake_recall']}")
print(f"  macro-F1   = {res['macro_f1']}   MCC={res['mcc']}")
print(f"\n  vs CelebDF (real_rec 0.40, fake_rec 0.87): does the real-class mismatch generalize?")
print("saved results_clean/zeroshot_wilddeepfake.json")
