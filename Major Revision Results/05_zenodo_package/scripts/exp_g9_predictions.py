#!/usr/bin/env python3
"""G9 — Persist PER-VIDEO predictions for every PRISM regime, then re-run DeLong FROM the
persisted probs (auditable). Schema (guide #40):
  video_path, source_id, dataset, manipulation, compression, true_label, pred_prob, pred_label,
  split, model, seed
Regimes persisted (identity-disjoint, seed 42, M1 train-only imputer, locked LightGBM):
  PRISM_50D_indist   (DF/F2F/FS/NT, test)      PRISM_53D_indist (DF/F2F/FS/NT, test)
  PRISM_50D_LOMO     (cross-manip, test)        PRISM_50D_zeroshot (Celeb-DF)
  PRISM_53D_zeroshot (WildDeepfake)
Xception per-video is persisted by exp_g9_xception_predictions.py (GPU). Output:
  results_clean/predictions_per_video.csv  (+ append=Xception rows)
Then recompute DeLong 53-vs-50 per manip from the persisted probs and compare to locked delong_53vs50.csv.
"""
import os, sys, json, subprocess, datetime
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore"); sys.path.insert(0, "src")
from protocol import make_splits, clip_identities
from leakfree import split_impute, impute_with, pooled_train_median
import roi_config as RC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from delong import delong_roc_test
import lightgbm as lgb

SEED=42; np.random.seed(SEED); F="features"; OUT="results_clean"
G1=RC.CANDIDATE_GROUPS["G1_mouth_instability"]; ROI=RC.ROI_FEATURE_NAMES
MAN=["Deepfakes","Face2Face","FaceSwap","NeuralTextures"]
CODE={"Deepfakes":"deepfakes","Face2Face":"face2face","FaceSwap":"faceswap","NeuralTextures":"neuraltextures"}
def basen(p): return os.path.basename(str(p))
def sid(p):
    ids=sorted(clip_identities(p)); return ids[0] if ids else basen(p)
def commit():
    try: return subprocess.check_output(["git","rev-parse","--short","HEAD"],text=True).strip()
    except: return "nogit"
def LGBM(): return lgb.LGBMClassifier(n_estimators=200,max_depth=6,learning_rate=0.05,num_leaves=31,
    min_child_samples=20,class_weight="balanced",random_state=SEED,verbose=-1,n_jobs=-1)

# ---- load 50-D and 53-D (ROI-merged), M1 train-only imputer ----
raw={k:pd.read_csv(f"{F}/ffpp_{'original' if k=='real' else CODE.get(k,k)}_c23.csv") for k in ["real"]+MAN}
FC=sorted([c for c in raw["real"].columns if c[:2] in ("s_","t_")])
P50={k:split_impute(v,FC)[0] for k,v in raw.items()}
ff_med50=pooled_train_median(list(P50.values()),FC)
# 53-D merge (50-D + ROI G1) per set
def merged53(k):
    o=raw[k].copy(); r=pd.read_csv(f"{F}/roi_{'original' if k=='real' else CODE[k]}_c23.csv").copy()
    o["_b"]=o.video_path.map(basen); r["_b"]=r.video_path.map(basen)
    m=o.merge(r[["_b"]+ROI],on="_b",how="inner")
    m=make_splits(m); cols=FC+ROI
    for c in cols: m[c]=pd.to_numeric(m[c],errors="coerce").replace([np.inf,-np.inf],np.nan)
    m[cols]=m[cols].fillna(m.loc[m.partition=="train",cols].median())
    return m
P53={k:merged53(k) for k in ["real"]+MAN}
COLS53=FC+G1

rows=[]   # long-format prediction rows
def emit(df, prob, model, dataset, manip, comp, split):
    for i,(_,r) in enumerate(df.reset_index(drop=True).iterrows()):
        p=float(prob[i])
        rows.append(dict(video_path=basen(r["video_path"]), source_id=sid(r["video_path"]),
            dataset=dataset, manipulation=manip, compression=comp, true_label=int(r["label"]),
            pred_prob=round(p,6), pred_label=int(p>=0.5), split=split, model=model, seed=SEED))

probs={}  # (model, key) -> (video_ids, y, p) for DeLong recompute
def fit_predict(train_frames, test_df, cols):
    tr=pd.concat(train_frames,ignore_index=True)
    sc=StandardScaler().fit(tr[cols].values); clf=LGBM(); clf.fit(sc.transform(tr[cols].values),tr['label'].values.astype(int))
    return clf.predict_proba(sc.transform(test_df[cols].values))[:,1]

# ---- (1) in-distribution 50-D and 53-D per manip ----
for m in MAN:
    te50=P50[m][P50[m].partition=="test"]; re50=P50["real"][P50["real"].partition=="test"]
    test50=pd.concat([re50,te50],ignore_index=True)
    p50=fit_predict([P50["real"][P50["real"].partition=="train"],P50[m][P50[m].partition=="train"]],test50,FC)
    emit(test50,p50,"PRISM_50D_indist","FFpp",CODE[m],"c23","test")
    probs[("50D",m)]=(test50["video_path"].map(basen).values, test50["label"].values.astype(int), p50)
    te53=P53[m][P53[m].partition=="test"]; re53=P53["real"][P53["real"].partition=="test"]
    test53=pd.concat([re53,te53],ignore_index=True)
    p53=fit_predict([P53["real"][P53["real"].partition=="train"],P53[m][P53[m].partition=="train"]],test53,COLS53)
    emit(test53,p53,"PRISM_53D_indist","FFpp",CODE[m],"c23","test")
    probs[("53D",m)]=(test53["video_path"].map(basen).values, test53["label"].values.astype(int), p53)

# ---- (2) cross-manip LOMO 50-D ----
for held in MAN:
    others=[x for x in MAN if x!=held]
    tr=[P50["real"][P50["real"].partition=="train"]]+[P50[o][P50[o].partition=="train"] for o in others]
    te=pd.concat([P50["real"][P50["real"].partition=="test"],P50[held][P50[held].partition=="test"]],ignore_index=True)
    p=fit_predict(tr,te,FC); emit(te,p,"PRISM_50D_LOMO","FFpp",CODE[held],"c23","test")

# ---- (3) zero-shot Celeb-DF 50-D ----
cd=impute_with(pd.read_csv(f"{F}/celebdf_features.csv"),FC,ff_med50)
trc=[P50["real"][P50["real"].partition=="train"]]+[P50[m][P50[m].partition=="train"] for m in MAN]
pcd=fit_predict(trc,cd,FC); emit(cd,pcd,"PRISM_50D_zeroshot","CelebDF","NA","c23","zero_shot")

# ---- (4) zero-shot WildDeepfake 53-D ----
wdf=pd.read_csv(f"{F}/wilddeepfake_test_53d.csv")
for c in COLS53: wdf[c]=pd.to_numeric(wdf[c],errors="coerce").replace([np.inf,-np.inf],np.nan)
trw=pd.concat([P53["real"][P53["real"].partition=="train"]]+[P53[m][P53[m].partition=="train"] for m in MAN],ignore_index=True)
wdf[COLS53]=wdf[COLS53].fillna(trw[COLS53].median())
pw=fit_predict([trw],wdf,COLS53); emit(wdf,pw,"PRISM_53D_zeroshot","WildDeepfake","NA","NA","zero_shot")

pred=pd.DataFrame(rows)
pred.to_csv(f"{OUT}/predictions_per_video.csv",index=False)

# ---- recompute DeLong 53 vs 50 per manip FROM persisted probs; compare to locked ----
recompute=[]
for m in MAN:
    idb,y50,p50=probs[("50D",m)]; idb2,y53,p53=probs[("53D",m)]
    # align by video basename (53-D may drop a few via ROI merge)
    d50=pd.DataFrame({"b":idb,"y":y50,"p50":p50}); d53=pd.DataFrame({"b":idb2,"y":y53,"p53":p53})
    mg=d50.merge(d53[["b","p53"]],on="b")
    a50,a53,z,pv=delong_roc_test(mg.y.values,mg.p50.values,mg.p53.values)
    recompute.append(dict(manip=CODE[m],n=len(mg),auc50=round(a50,4),auc53=round(a53,4),
        delta=round(a53-a50,4),z=round(z,3),p=float(pv)))
rc=pd.DataFrame(recompute); rc.to_csv(f"{OUT}/delong_53vs50_from_predictions.csv",index=False)

prov=dict(script="exp_g9_predictions.py",git_commit=commit(),seed=SEED,date=datetime.date.today().isoformat(),
    schema=list(pred.columns), n_rows=int(len(pred)),
    models=sorted(pred.model.unique().tolist()), note="per-video predictions; DeLong recomputed from persisted probs")
json.dump(prov,open(f"{OUT}/predictions_manifest.json","w"),indent=1)

print("="*66); print("G9 — PER-VIDEO PREDICTIONS PERSISTED"); print("="*66)
print(pred.groupby(["model","dataset"]).size().to_string())
print(f"\ntotal rows: {len(pred)}  -> {OUT}/predictions_per_video.csv")
print("\nDeLong 53-vs-50 recomputed FROM persisted probs:")
print(rc.to_string(index=False))
print(f"\nsaved predictions_per_video.csv, delong_53vs50_from_predictions.csv (commit {commit()})")
