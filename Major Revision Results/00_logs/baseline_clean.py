#!/usr/bin/env python3
"""LOCKED CLEAN BASELINE — leakage-free, identity-disjoint (Track A checkpoint).
Regimes: (1) in-distribution per manipulation, (2) leave-one-manipulation-out cross-manip,
(3) zero-shot Celeb-DF v2. Classifier = LightGBM (PRISM's documented classifier, fixed params;
no test-set model selection). Emits protocol_matrix.csv, results_clean/baseline.json, provenance.
"""
import json, os, sys, hashlib, subprocess, datetime
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, "src")
from protocol import make_splits, assert_no_identity_overlap, load_id2split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, matthews_corrcoef
import lightgbm as lgb

SEED=42; np.random.seed(SEED)
FEAT="features"; OUT="results_clean"; os.makedirs(OUT, exist_ok=True)
FILES={"real":"ffpp_original_c23.csv","Deepfakes":"ffpp_deepfakes_c23.csv",
       "Face2Face":"ffpp_face2face_c23.csv","FaceSwap":"ffpp_faceswap_c23.csv",
       "NeuralTextures":"ffpp_neuraltextures_c23.csv","CelebDF":"celebdf_features.csv"}
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()[:16]
def git_commit():
    try: return subprocess.check_output(["git","rev-parse","--short","HEAD"],text=True).strip()
    except: return "nogit"

id2split=load_id2split()
raw={k:pd.read_csv(f"{FEAT}/{v}") for k,v in FILES.items()}
hashes={v:sha(f"{FEAT}/{v}") for v in FILES.values()}
fc=sorted([c for c in raw["real"].columns if c[:2] in ("s_","t_")])
def clean(df):
    d=df.copy(); d[fc]=d[fc].replace([np.inf,-np.inf],np.nan)
    for c in fc: d[c]=d[c].fillna(d[c].median())
    return d
# partition FF++ frames by identity
P={k:make_splits(clean(v)) for k,v in raw.items() if k!="CelebDF"}
CD=clean(raw["CelebDF"])
def part(k,p): d=P[k]; return d[d.partition==p]
def LGBM(): return lgb.LGBMClassifier(n_estimators=200,max_depth=6,learning_rate=0.05,
    num_leaves=31,min_child_samples=20,class_weight="balanced",random_state=SEED,verbose=-1,n_jobs=-1)

def boot_ci(y,p,n=2000,seed=SEED):
    rng=np.random.RandomState(seed); b=[]
    for _ in range(n):
        idx=rng.randint(0,len(y),len(y))
        if len(np.unique(y[idx]))<2: continue
        b.append(roc_auc_score(y[idx],p[idx]))
    return float(np.percentile(b,2.5)), float(np.percentile(b,97.5))

def metrics(y,p):
    pred=(p>=0.5).astype(int); lo,hi=boot_ci(y,p)
    return dict(auc=round(roc_auc_score(y,p),4),ci95=[round(lo,4),round(hi,4)],
        f1=round(f1_score(y,pred,average='macro'),4),precision=round(precision_score(y,pred,zero_division=0),4),
        recall=round(recall_score(y,pred,zero_division=0),4),mcc=round(matthews_corrcoef(y,pred),4),
        n_test=int(len(y)),n_test_real=int((y==0).sum()),n_test_fake=int((y==1).sum()))

def fit_eval(train_frames, test_frames, tag):
    assert_no_identity_overlap([(d,'train') for d,_ in train_frames]+[(d,'test') for d,_ in test_frames])
    tr=pd.concat([d for d,_ in train_frames],ignore_index=True); te=pd.concat([d for d,_ in test_frames],ignore_index=True)
    Xtr=tr[fc].values.astype(float); ytr=tr['label'].values.astype(int)
    Xte=te[fc].values.astype(float); yte=te['label'].values.astype(int)
    sc=StandardScaler().fit(Xtr); clf=LGBM(); clf.fit(sc.transform(Xtr),ytr)
    p=clf.predict_proba(sc.transform(Xte))[:,1]
    m=metrics(yte,p); m["n_train"]=int(len(ytr)); m["tag"]=tag
    return m

results={"regime1_in_distribution":{},"regime2_cross_manip_LOMO":{},"regime3_zero_shot_celebdf":{}}
proto_rows=[]
MAN=["Deepfakes","Face2Face","FaceSwap","NeuralTextures"]

print("="*66+"\nREGIME 1 — IN-DISTRIBUTION per manipulation (identity-disjoint)\n"+"="*66)
for m in MAN:
    r=fit_eval([(part("real","train"),0),(part(m,"train"),1)],
               [(part("real","test"),0),(part(m,"test"),1)], f"indist_{m}")
    results["regime1_in_distribution"][m]=r
    proto_rows.append(dict(regime="in_dist",manip=m,n_train=r["n_train"],n_test=r["n_test"],
        n_test_real=r["n_test_real"],n_test_fake=r["n_test_fake"],identity_overlap=0))
    print(f"  {m:15s} AUC={r['auc']:.4f} CI{r['ci95']} F1={r['f1']:.3f} MCC={r['mcc']:.3f} (ntr={r['n_train']},nte={r['n_test']})")

print("\n"+"="*66+"\nREGIME 2 — CROSS-MANIP leave-one-manipulation-out\n"+"="*66)
for held in MAN:
    others=[x for x in MAN if x!=held]
    tr=[(part("real","train"),0)]+[(part(o,"train"),1) for o in others]
    te=[(part("real","test"),0),(part(held,"test"),1)]
    r=fit_eval(tr,te,f"LOMO_{held}")
    results["regime2_cross_manip_LOMO"][held]=r
    proto_rows.append(dict(regime="cross_manip_LOMO",manip=held,n_train=r["n_train"],n_test=r["n_test"],
        n_test_real=r["n_test_real"],n_test_fake=r["n_test_fake"],identity_overlap=0))
    print(f"  held-out {held:15s} AUC={r['auc']:.4f} CI{r['ci95']} F1={r['f1']:.3f} MCC={r['mcc']:.3f}")

print("\n"+"="*66+"\nREGIME 3 — ZERO-SHOT Celeb-DF v2 (FF++ train identities only)\n"+"="*66)
tr=[(part("real","train"),0)]+[(part(m,"train"),1) for m in MAN]
# CelebDF has no FF++ identities -> assertion covers FF++ side; CelebDF is disjoint dataset
trc=pd.concat([d for d,_ in tr],ignore_index=True)
Xtr=trc[fc].values.astype(float); ytr=trc['label'].values.astype(int)
sc=StandardScaler().fit(Xtr); clf=LGBM(); clf.fit(sc.transform(Xtr),ytr)
pc=clf.predict_proba(sc.transform(CD[fc].values.astype(float)))[:,1]; yc=CD['label'].values.astype(int)
r=metrics(yc,pc); r["n_train"]=int(len(ytr)); r["tag"]="zeroshot_celebdf"
results["regime3_zero_shot_celebdf"]["CelebDF_v2"]=r
proto_rows.append(dict(regime="zero_shot",manip="CelebDF_v2",n_train=r["n_train"],n_test=r["n_test"],
    n_test_real=r["n_test_real"],n_test_fake=r["n_test_fake"],identity_overlap=0))
print(f"  Celeb-DF v2 zero-shot  AUC={r['auc']:.4f} CI{r['ci95']} F1={r['f1']:.3f} real_rec={r['recall']:.3f} MCC={r['mcc']:.3f} (ntr={r['n_train']},nte={r['n_test']})")

prov={"script":"Major Revision Results/00_logs/baseline_clean.py","git_commit":git_commit(),
      "seed":SEED,"date":datetime.date.today().isoformat(),"classifier":"LightGBM(n_est=200,depth=6,lr=0.05,leaves=31,min_child=20,balanced)",
      "feature_csv_sha256":hashes,"n_features":len(fc),"protocol":"identity-disjoint official FF++ 720/140/140, seed42"}
json.dump({"provenance":prov,"results":results}, open(f"{OUT}/baseline.json","w"), indent=1)
pd.DataFrame(proto_rows).to_csv(f"{OUT}/protocol_matrix.csv", index=False)
print(f"\nWrote {OUT}/baseline.json and {OUT}/protocol_matrix.csv  (commit {prov['git_commit']})")
