#!/usr/bin/env python3
"""PHASE-4 SINGLE SEALED EVALUATION — frozen model on Celeb-DF-v2 TEST + FF++ TEST. Budget 1.
FROZEN (see trackE_FREEZE.md): 196-D E1-expanded rep + RF+ExtraTrees+LGBM_d6 RANK ensemble. Trained on FF++
train ONLY. This script spends the single sealed evaluation. It is gated: without --unseal it DRY-RUNS
(verifies data + prints counts, computes NOTHING on sealed labels). With --unseal it scores celebdf_test and
FF++ test, reports AUC + identity-grouped bootstrap 95% CI + predicted-vs-actual, and logs the spend.

PREREQUISITE (label-agnostic, does NOT spend budget): extract 196-D on the sealed test videos ->
  features/trackE/plain_celebdf_test.csv   (build manifest via --make_test_manifest, then extract_trackE_SBV --plain)

Usage:
  # 1) build sealed-test manifest (label-agnostic)
  python .../exp_trackE_SEALED_eval.py --make_test_manifest
  # 2) extract (label-agnostic, ~3-4h):
  .venv/bin/python src/extract_trackE_SBV.py --plain --manifest features/trackD/manifest_celebdf_test.csv \
        --output features/trackE/plain_celebdf_test.csv --max_frames 60
  # 3) dry run (no budget spent):
  python .../exp_trackE_SEALED_eval.py
  # 4) THE sealed evaluation (spends budget 1):
  python .../exp_trackE_SEALED_eval.py --unseal
"""
import os, sys, json, subprocess, datetime, re, argparse, glob
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore"); sys.path.insert(0, "src")
from protocol import make_splits
from extract_trackE_SBV import FEATS
from sealed import celebdf_partition, unseal, sealed_eval_count
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from scipy.stats import rankdata
import lightgbm as lgb
SEED=42; TE="features/trackE"; OUT="results_clean"; MAN=["deepfakes","face2face","faceswap","neuraltextures"]
DIR={"deepfakes":"Deepfakes","face2face":"Face2Face","faceswap":"FaceSwap","neuraltextures":"NeuralTextures"}
PRED_POINT=0.68; PRED_LO=0.65; PRED_HI=0.71                      # pre-registered (trackE_preregistration.md)
CELEBDF_ROOT="/home/iiitn/Datasets/Celeb-DF-v2"
def method(p):
    for m,d in DIR.items():
        if f"/{d}/" in p: return m
    return "real" if "youtube" in p else ("celebdf" if "Celeb-DF" in p else "?")
def commit():
    try: return subprocess.check_output(["git","rev-parse","--short","HEAD"],text=True).strip()
    except: return "nogit"

def make_test_manifest():
    # AUTHORITATIVE source = the split's declared feature file, NOT a dataset glob (which over-counts).
    src=pd.read_csv("features/celebdf_features.csv")
    df=celebdf_partition(src); df=df[df.ct_partition=="test"].copy()
    assert len(df)==2273, f"expected 2273 sealed-test videos, got {len(df)} — split mismatch, ABORT"
    df[["video_path","label"]].to_csv("features/trackD/manifest_celebdf_test.csv",index=False)
    print(f"  manifest_celebdf_test.csv: {len(df)} SEALED-test videos ({(df.label==0).sum()} real/{(df.label==1).sum()} fake)")
    return

def build_model_and_train():
    ev=pd.read_csv(f"{TE}/plain_everyone_E3.csv"); ev["src"]=ev.video_path.map(method)
    for c in FEATS: ev[c]=pd.to_numeric(ev[c],errors="coerce").replace([np.inf,-np.inf],np.nan)
    ff=make_splits(ev[ev.src.isin(["real"]+MAN)].copy())
    med=ff[ff.partition=="train"][FEATS].median(); ff[FEATS]=ff[FEATS].fillna(med)
    tr=pd.concat([ff[(ff.src=="real")&(ff.partition=="train")].assign(label=0)]+
                 [ff[(ff.src==m)&(ff.partition=="train")].assign(label=1) for m in MAN],ignore_index=True)
    sc=StandardScaler().fit(tr[FEATS].values); Xtr=sc.transform(tr[FEATS].values); ytr=tr.label.values.astype(int)
    def L(): return lgb.LGBMClassifier(n_estimators=300,learning_rate=0.05,num_leaves=31,min_child_samples=20,max_depth=6,class_weight="balanced",random_state=SEED,verbose=-1,n_jobs=-1,deterministic=True,force_row_wise=True)
    models={"RF":RandomForestClassifier(n_estimators=400,max_depth=8,min_samples_leaf=5,class_weight="balanced",random_state=SEED,n_jobs=-1),
            "ET":ExtraTreesClassifier(n_estimators=600,max_depth=10,min_samples_leaf=4,class_weight="balanced",random_state=SEED,n_jobs=-1),
            "LGBM":L()}
    for m in models.values(): m.fit(Xtr,ytr)
    return sc,med,models,ff
def rank_ens(models,sc,med,X):
    Xs=sc.transform(pd.DataFrame(X,columns=FEATS).fillna(med).values)
    return np.mean([rankdata(m.predict_proba(Xs)[:,1]) for m in models.values()],axis=0), \
           {k:m.predict_proba(Xs)[:,1] for k,m in models.items()}
def boot_ci(y,score,ids,n=2000):
    uids=np.unique(ids); rng=np.random.RandomState(SEED); aucs=[]
    for _ in range(n):
        samp=rng.choice(uids,len(uids),replace=True); mask=np.isin(ids,samp)
        if len(np.unique(y[mask]))>1: aucs.append(roc_auc_score(y[mask],score[mask]))
    return round(float(np.percentile(aucs,2.5)),4),round(float(np.percentile(aucs,97.5)),4)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--make_test_manifest",action="store_true")
    ap.add_argument("--unseal",action="store_true")
    a=ap.parse_args()
    if a.make_test_manifest: make_test_manifest(); sys.exit(0)
    TEST_CSV=f"{TE}/plain_celebdf_test.csv"
    print(f"sealed evals already spent: {sealed_eval_count()}")
    if not os.path.exists(TEST_CSV):
        print(f"  PREREQUISITE MISSING: {TEST_CSV} not found. Build manifest (--make_test_manifest) then extract (see header). DRY-RUN cannot proceed."); sys.exit(1)
    sc,med,models,ff=build_model_and_train()
    test=pd.read_csv(TEST_CSV); test["label"]=test.get("label",1)
    ff_test=ff[ff.partition=="test"].copy()
    print(f"  celebdf_test videos: {len(test)} | FF++ test videos: {len(ff_test)}")
    if not a.unseal:
        print("  DRY-RUN OK (data present, model trains). Re-run with --unseal to spend the single sealed evaluation."); sys.exit(0)
    unseal("celebdf_test", allow_sealed=True)     # SPENDS THE BUDGET (logged)
    # celebdf_test
    yct=test.label.values.astype(int)
    ids=test.video_path.map(lambda p:(re.findall(r"id(\d+)",str(p)) or [os.path.basename(str(p))])[0]).values
    ens,per=rank_ens(models,sc,med,test[FEATS].values)
    auc_ct=round(roc_auc_score(yct,ens),4); lo,hi=boot_ci(yct,ens,ids)
    # FF++ test (rank ensemble)
    yft=ff_test.label.values.astype(int) if "label" in ff_test else (ff_test.src.isin(MAN)).astype(int).values
    ensf,_=rank_ens(models,sc,med,ff_test[FEATS].values); auc_ft=round(roc_auc_score(yft,ensf),4)
    # reference single models on celebdf_test (robustness check — not selection)
    ref={k:round(roc_auc_score(yct,per[k]),4) for k in per}
    res=dict(provenance=dict(script="exp_trackE_SEALED_eval.py",git_commit=commit(),seed=SEED,date=datetime.date.today().isoformat(),
        frozen_model="RF+ExtraTrees+LGBM_d6 rank ensemble",frozen_rep="196-D E1-expanded",sealed=True,dev_eval_count=56),
        predicted=dict(point=PRED_POINT,interval=[PRED_LO,PRED_HI]),
        celebdf_test=dict(auc=auc_ct,ci95=[lo,hi],n=len(test),reals=int((yct==0).sum()),fakes=int((yct==1).sum()),single_model_ref=ref),
        ffpp_test=dict(auc=auc_ft,n=len(ff_test)),
        predicted_vs_actual=dict(point=PRED_POINT,actual=auc_ct,within_interval=bool(PRED_LO<=auc_ct<=PRED_HI)))
    os.makedirs(OUT,exist_ok=True); json.dump(res,open(f"{OUT}/SEALED_final.json","w"),indent=1)
    print("="*66);print("PHASE-4 SEALED RESULT (budget spent)");print("="*66)
    print(f"  Celeb-DF-v2 TEST AUC = {auc_ct}  95% CI [{lo}, {hi}]  (n={len(test)})")
    print(f"  FF++ TEST AUC        = {auc_ft}")
    print(f"  single-model ref (celebdf_test): {ref}")
    print(f"  PREDICTED {PRED_POINT} [{PRED_LO},{PRED_HI}] -> ACTUAL {auc_ct} -> within interval: {PRED_LO<=auc_ct<=PRED_HI}")
    print(f"  sealed evals now spent: {sealed_eval_count()}")
    print(f"saved {OUT}/SEALED_final.json")
