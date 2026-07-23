#!/usr/bin/env python3
"""M4 — Missingness-as-signal audit (guide #8). Two questions:
 (1) Per-dataset / per-class EXTRACTION success/failure rates + per-feature missingness.
 (2) MISSINGNESS-ONLY classifiers (binary validity indicators as the ONLY features):
     (a) real-vs-fake  (within FF++, identity-disjoint)
     (b) dataset identity (FF++ vs Celeb-DF, shared 50-D representation)
 If missingness alone predicts label/dataset above chance -> potential confound to disclose.

Design notes (honesty):
 - Rows are VIDEO-level. FF++ residual_*_c23.csv carries the full 1000-video attempted list per set
   (residual extraction ran on all 1000), used as the extraction denominator.
 - Validity indicators are cross-feature-family extraction flags per video: valid_50d / valid_roi /
   valid_rppg (each = did that family produce a row for this video). Within extracted rows the 50-D/
   ROI/rPPG matrices have ~0 NaN, so the informative missingness is at the video/extraction level.
 - Celeb-DF had only the 50-D family extracted, so the FF++-vs-CelebDF check uses ONLY the shared
   valid_50d indicator (using valid_roi/valid_rppg there would be a pipeline artifact, not a confound).
"""
import os, sys, json, subprocess, datetime, glob
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore"); sys.path.insert(0, "src")
from protocol import make_splits, assert_no_identity_overlap
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

SEED = 42; np.random.seed(SEED); F = "features"; OUT = "results_clean"
def base(p): return os.path.splitext(os.path.basename(str(p)))[0]
def commit():
    try: return subprocess.check_output(["git","rev-parse","--short","HEAD"], text=True).strip()
    except: return "nogit"
def bootci(y, s, n=2000, seed=SEED):
    rng = np.random.RandomState(seed); b = []
    for _ in range(n):
        i = rng.randint(0, len(y), len(y))
        if len(np.unique(y[i])) < 2: continue
        b.append(roc_auc_score(y[i], s[i]))
    return round(float(np.percentile(b,2.5)),4), round(float(np.percentile(b,97.5)),4)

SETS = {"original":0, "deepfakes":1, "face2face":1, "faceswap":1, "neuraltextures":1}  # 0=real 1=fake

def bset(path):  # basenames present in a CSV
    return set(base(v) for v in pd.read_csv(path)["video_path"]) if os.path.exists(path) else set()

# ---------- family membership sets per FF++ set ----------
fam = {}
for s in SETS:
    o = "original" if s == "original" else s
    fam[s] = dict(
        attempted = bset(f"{F}/residual_{o}_c23.csv"),                        # full 1000 list
        v50  = bset(f"{F}/ffpp_{'original' if s=='original' else s}_c23.csv"),
        vroi = bset(f"{F}/roi_{o}_c23.csv"),
        vrppg= bset(f"{F}/rppg_{o}_c23.csv"),
        vresid = bset(f"{F}/residual_{o}_c23.csv"),
    )

# ---------- (1) extraction success table ----------
succ_rows = []
for s, lab in SETS.items():
    a = fam[s]["attempted"];  N = len(a)
    row = dict(dataset="FFpp", set=s, klass=("real" if lab==0 else "fake"), attempted=N)
    for k in ("v50","vroi","vrppg","vresid"):
        got = len(a & fam[s][k])
        row[k+"_ok"] = got; row[k+"_rate"] = round(got/N, 4)
    succ_rows.append(row)
# Celeb-DF from manifest (attempted) vs 50-D CSV (extracted), per class
mani = pd.read_csv("data_xception/manifest_celebdf.csv").drop_duplicates("video")
cd50 = pd.read_csv(f"{F}/celebdf_features.csv"); cd50["b"] = cd50["video_path"].map(base)
mani["b"] = mani["video"].map(base)
cd_ext = set(cd50["b"])
for lab, kl in [(0,"real"),(1,"fake")]:
    att = set(mani[mani.label==lab]["b"]); N = len(att)
    got = len(att & cd_ext)
    succ_rows.append(dict(dataset="CelebDF", set="celebdf", klass=kl, attempted=N,
        v50_ok=got, v50_rate=round(got/N,4), vroi_ok=None, vroi_rate=None,
        vrppg_ok=None, vrppg_rate=None, vresid_ok=None, vresid_rate=None))
succ = pd.DataFrame(succ_rows)
succ.to_csv(f"{OUT}/missingness_success_rates.csv", index=False)

# ---------- per-feature (cell-level) missingness within extracted rows ----------
permiss = {}
for name, patt in [("50D","ffpp_{o}_c23"),("ROI","roi_{o}_c23"),("rPPG","rppg_{o}_c23"),("residual","residual_{o}_c23")]:
    tot_na = tot_cell = 0
    for s in SETS:
        o = "original" if s=="original" else s
        p = f"{F}/{patt.format(o=o)}.csv"
        if not os.path.exists(p): continue
        d = pd.read_csv(p); num = d.select_dtypes(include=[np.number]).columns
        m = d[num].replace([np.inf,-np.inf], np.nan)
        tot_na += int(m.isna().sum().sum()); tot_cell += int(m.size)
    permiss[name] = dict(cell_missing=tot_na, cells=tot_cell,
                         pct=round(100*tot_na/max(tot_cell,1), 4))

# ---------- (2a) missingness-only classifier: real vs fake (FF++, identity-disjoint) ----------
rows = []
for s, lab in SETS.items():
    for b in sorted(fam[s]["attempted"]):   # sort: set iteration order is otherwise nondeterministic
        rows.append(dict(video_path=b, label=lab,
            v50=int(b in fam[s]["v50"]), vroi=int(b in fam[s]["vroi"]),
            vrppg=int(b in fam[s]["vrppg"])))
mm = pd.DataFrame(rows)
mm = make_splits(mm, path_col="video_path")
mm = mm[mm.partition.isin(["train","test"])].copy()
tr, te = mm[mm.partition=="train"], mm[mm.partition=="test"]
assert_no_identity_overlap([(tr,"train"),(te,"test")], path_col="video_path")
IND = ["v50","vroi","vrppg"]
def fit_auc(model, tr, te, cols):
    m = model.fit(tr[cols].values, tr["label"].values)
    s = m.predict_proba(te[cols].values)[:,1]
    return roc_auc_score(te["label"].values, s), s
# Deterministic LogisticRegression on the binary validity indicators. (LightGBM is not reproducible
# on these degenerate few-binary-feature problems even with deterministic=True/n_jobs=1; a linear model
# is the appropriate and fully reproducible choice, and the conclusion is unchanged.)
auc_rf, s_rf = fit_auc(LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs"), tr, te, IND)
ci_rf = bootci(te["label"].values, s_rf)

# ---------- (2c) missingness-only classifier: real vs fake WITHIN Celeb-DF (v50 extraction flag) ----------
# CelebDF reals fail 50-D extraction more than fakes -> class-dependent selection bias in the zero-shot test.
cdm = mani[["b","label"]].copy(); cdm["v50"] = cdm["b"].isin(cd_ext).astype(int)
rng3 = np.random.RandomState(SEED); ix = rng3.permutation(len(cdm)); c3 = int(0.7*len(cdm))
cdtr, cdte = cdm.iloc[ix[:c3]], cdm.iloc[ix[c3:]]
m3 = LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs").fit(cdtr[["v50"]].values, cdtr["label"].values)
s_cd = m3.predict_proba(cdte[["v50"]].values)[:,1]
auc_cdrf = roc_auc_score(cdte["label"].values, s_cd); ci_cdrf = bootci(cdte["label"].values, s_cd)

# ---------- (2b) missingness-only classifier: dataset identity FF++ vs CelebDF (shared 50-D) ----------
ff = pd.DataFrame([dict(v50=int(b in fam[s]["v50"]), ds=0)
                   for s in SETS for b in sorted(fam[s]["attempted"])])  # sort: deterministic order
cd = pd.DataFrame([dict(v50=int(b in cd_ext), ds=1) for b in mani["b"]])
di = pd.concat([ff, cd], ignore_index=True)
rng = np.random.RandomState(SEED); idx = rng.permutation(len(di)); cut = int(0.7*len(di))
ditr, dite = di.iloc[idx[:cut]], di.iloc[idx[cut:]]
m2 = LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs").fit(ditr[["v50"]].values, ditr["ds"].values)
s_di = m2.predict_proba(dite[["v50"]].values)[:,1]
auc_di = roc_auc_score(dite["ds"].values, s_di); ci_di = bootci(dite["ds"].values, s_di)

# ---------- assemble ----------
res = dict(
    provenance=dict(script="Major Revision Results/00_logs/exp_m4_missingness.py",
        git_commit=commit(), seed=SEED, date=datetime.date.today().isoformat(),
        note="video-level validity indicators; identity-disjoint for real-vs-fake; chance AUC=0.5"),
    extraction_success=succ_rows,
    per_feature_cell_missingness=permiss,
    missingness_only_classifier=dict(
        model="LogisticRegression (deterministic; binary validity indicators)",
        real_vs_fake_FFpp=dict(features=IND, n_train=int(len(tr)), n_test=int(len(te)),
            auc=round(auc_rf,4), auc_ci95=list(ci_rf), chance=0.5),
        real_vs_fake_CelebDF=dict(features=["v50"], n_train=int(len(cdtr)), n_test=int(len(cdte)),
            auc=round(auc_cdrf,4), auc_ci95=list(ci_cdrf), chance=0.5,
            note="CelebDF real 50-D extraction 0.912 vs fake 0.948 -> mild class-dependent selection bias"),
        dataset_identity_FFpp_vs_CelebDF=dict(features=["v50"], n_train=int(len(ditr)),
            n_test=int(len(dite)), auc=round(auc_di,4), auc_ci95=list(ci_di), chance=0.5,
            caveat="shared 50-D validity only (ROI/rPPG/residual not extracted for CelebDF)")))
json.dump(res, open(f"{OUT}/missingness_audit.json","w"), indent=1)

# ---------- print ----------
print("="*74); print("M4 — MISSINGNESS AUDIT"); print("="*74)
print("\n[1] EXTRACTION SUCCESS RATE (extracted / attempted), by set & class")
print(succ.to_string(index=False))
print("\n[2] PER-FEATURE CELL MISSINGNESS within extracted rows")
for k,v in permiss.items(): print(f"   {k:9s} {v['cell_missing']}/{v['cells']} cells = {v['pct']}%")
print("\n[3] *** MISSINGNESS-ONLY CLASSIFIER AUCs — LogisticRegression, deterministic (chance = 0.50) ***")
print(f"   (a) real-vs-fake  (FF++, identity-disjoint, {IND}):")
print(f"         AUC = {auc_rf:.4f}  CI{ci_rf}")
print(f"   (b) dataset identity (FF++ vs Celeb-DF, [v50] only):")
print(f"         AUC = {auc_di:.4f}  CI{ci_di}")
print(f"   (c) real-vs-fake WITHIN Celeb-DF ([v50] only):")
print(f"         AUC = {auc_cdrf:.4f}  CI{ci_cdrf}   (real extract 0.912 vs fake 0.948)")
print(f"\nsaved {OUT}/missingness_audit.json, missingness_success_rates.csv (commit {commit()})")
