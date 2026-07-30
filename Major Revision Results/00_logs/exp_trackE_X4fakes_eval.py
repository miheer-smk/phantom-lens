#!/usr/bin/env python3
"""Track E — X4-FAKES: DIVERSE high-quality FAKE augmentation. DEV only; sealed=0.
Mirror of X4 (which added diverse REALS and failed for ranking). Here we widen the FAKE class instead: our
training fakes are ONLY FF++'s 4 manip families (lower quality, stylistically unlike Celeb-DF). Add real,
high-quality deepfakes from a DIFFERENT source (DFD / Google DeepFakeDetection, same 28 actors) to extend the
fake region of feature space toward Celeb-DF's generation quality. NOT synthetic SBV blends (those failed) —
real diverse-source fakes. Celeb-DF stays SEALED. 196-D R0 (60-frame), RandomForest, identity-grouped celebdf_dev CV.
Sweep amount added {0(base), half, all}. Report real/fake recall separately. Bar cross +0.03 (Holm at freeze).
PRE-REGISTERED prediction: cross-AUC UP (fake class covers Celeb-DF fake style -> sharper boundary); real recall
maintained or up; dose-response. If it also fails, that closes diverse-data augmentation on BOTH classes.
Usage: exp_trackE_X4fakes_eval.py <dfd_fakes_csv> [everyone_csv]
"""
import os, sys, json, subprocess, datetime, re
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore"); sys.path.insert(0, "src")
from protocol import make_splits
from extract_trackE_SBV import FEATS
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier
SEED=42; F="features"; TE=f"{F}/trackE"; OUT="results_clean"; MAN=["deepfakes","face2face","faceswap","neuraltextures"]
DIR={"deepfakes":"Deepfakes","face2face":"Face2Face","faceswap":"FaceSwap","neuraltextures":"NeuralTextures"}
FAKES_CSV=sys.argv[1] if len(sys.argv)>1 else f"{TE}/plain_dfd_fakes.csv"
EVERYONE=sys.argv[2] if len(sys.argv)>2 else f"{TE}/plain_everyone_E3.csv"
def method(p):
    for m,d in DIR.items():
        if f"/{d}/" in p: return m
    return "real" if "youtube" in p else ("celebdf" if "Celeb-DF" in p else "?")
def commit():
    try: return subprocess.check_output(["git","rev-parse","--short","HEAD"],text=True).strip()
    except: return "nogit"
def RF(): return RandomForestClassifier(n_estimators=400,max_depth=8,min_samples_leaf=5,class_weight="balanced",random_state=SEED,n_jobs=-1)
ev=pd.read_csv(EVERYONE); ev["src"]=ev.video_path.map(method)
for c in FEATS: ev[c]=pd.to_numeric(ev[c],errors="coerce").replace([np.inf,-np.inf],np.nan)
ff=make_splits(ev[ev.src.isin(["real"]+MAN)].copy()); cd=ev[ev.src=="celebdf"].copy()
yc=cd.label.values.astype(int)
cd_ids=cd.video_path.map(lambda p:(re.findall(r"id(\d+)",str(p)) or [os.path.basename(str(p))])[0]).values
dfd=pd.read_csv(FAKES_CSV)
for c in FEATS: dfd[c]=pd.to_numeric(dfd[c],errors="coerce").replace([np.inf,-np.inf],np.nan)
dfd["label"]=1                                                   # DFD fakes -> FAKE class
real_tr=ff[(ff.src=="real")&(ff.partition=="train")].assign(label=0)
manip_tr=pd.concat([ff[(ff.src==m)&(ff.partition=="train")].assign(label=1) for m in MAN],ignore_index=True)
val={m:pd.concat([ff[(ff.src=='real')&(ff.partition=='val')],ff[(ff.src==m)&(ff.partition=='val')]],ignore_index=True) for m in MAN}
def cv(p):
    a=[roc_auc_score(yc[i],p[i]) for _,i in GroupKFold(5).split(p,yc,cd_ids) if len(np.unique(yc[i]))>1]
    return round(float(np.mean(a)),4),round(float(np.std(a)),4)
def rec(p,t=0.5):
    pr=(p>=t).astype(int); return round(float((pr[yc==0]==0).mean()),3),round(float((pr[yc==1]==1).mean()),3)
def run(n_add):
    add=dfd.sample(n=n_add,random_state=SEED)[FEATS+["label"]] if n_add>0 else dfd.iloc[0:0][FEATS+["label"]]
    tr=pd.concat([real_tr[FEATS+["label"]], manip_tr[FEATS+["label"]], add], ignore_index=True)
    med=tr[FEATS].median(); trX=tr[FEATS].fillna(med).values; cdX=cd[FEATS].fillna(med).values
    sc=StandardScaler().fit(trX); m=RF().fit(sc.transform(trX),tr.label.values.astype(int))
    ys=[];ps=[]
    for mm in MAN: ys.append(val[mm].label.values.astype(int)); ps.append(m.predict_proba(sc.transform(val[mm][FEATS].fillna(med).values))[:,1])
    ind=round(roc_auc_score(np.concatenate(ys),np.concatenate(ps)),4)
    pc=m.predict_proba(sc.transform(cdX))[:,1]; cm,cs=cv(pc); rr,fr=rec(pc)
    return dict(n_fakes_added=int(n_add),n_fake_train=int(len(manip_tr)+n_add),indist_auc=ind,celebdf_dev_cv_mean=cm,celebdf_dev_cv_std=cs,real_recall=rr,fake_recall=fr)
N=len(dfd); base=run(0); half=run(N//2); full=run(N)
res=dict(provenance=dict(script="exp_trackE_X4fakes_eval.py",git_commit=commit(),seed=SEED,date=datetime.date.today().isoformat(),
    axis_dev_only=True,sealed_touched=False,classifier="RandomForest_d8",rep="196-D R0 (60-frame)",dfd_fakes=FAKES_CSV,n_available=int(N)),
    base_no_dfd=base, plus_half=half, plus_all=full,
    delta_cross_full=round(full["celebdf_dev_cv_mean"]-base["celebdf_dev_cv_mean"],4),
    delta_indist_full=round(full["indist_auc"]-base["indist_auc"],4))
os.makedirs(OUT,exist_ok=True); json.dump(res,open(f"{OUT}/trackE_X4fakes_dev.json","w"),indent=1)
print("="*78);print("TRACK E — X4-FAKES DIVERSE FAKE AUGMENTATION (DFD fakes -> fake class; celebdf_dev CV)");print("="*78)
print(f"{'variant':16s} {'nFakeTr':>8s} {'in-dist':>8s} {'cross_cv':>9s} {'±std':>6s} {'realRec':>8s} {'fakeRec':>8s}")
for tag,r in [("base(0)",base),("half(+%d)"%(N//2),half),("all(+%d)"%N,full)]:
    print(f"  {tag:16s} {r['n_fake_train']:8d} {r['indist_auc']:8.4f} {r['celebdf_dev_cv_mean']:9.4f} {r['celebdf_dev_cv_std']:6.3f} {r['real_recall']:8.3f} {r['fake_recall']:8.3f}")
print(f"\n  Δ cross (all vs base): {res['delta_cross_full']:+.4f} | Δ in-dist: {res['delta_indist_full']:+.4f}  (bar cross +0.03)")
print(f"saved {OUT}/trackE_X4fakes_dev.json (commit {commit()})")
