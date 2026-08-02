#!/usr/bin/env python3
"""McNemar + Wilcoxon (reviewer request). POST-FREEZE DESCRIPTIVE — no tuning, no model changes.
PRISM = frozen 196-D rank/prob ensemble (prob-avg for labels). Xception = frozen xception_best.pt re-scored by
INFERENCE on the saved crops (the persisted CelebDF Xception predictions are keyed by an unmappable sequential
index, so we re-score the frozen weight to obtain basename-keyed per-video probabilities; this is inference, not
retraining). Thresholds: theta=0.50 and each model's F1-optimal threshold derived on the FF++ VALIDATION partition
only. Emits mcnemar_wilcoxon_results.json, a markdown table, and prism_vs_xception_predictions.csv.
"""
import os, sys, json, subprocess, datetime, re
import numpy as np, pandas as pd, warnings, cv2
warnings.filterwarnings("ignore"); sys.path.insert(0,"src")
from protocol import make_splits
from extract_trackE_SBV import FEATS
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from scipy.stats import rankdata, wilcoxon, binomtest, chi2 as chi2dist
import lightgbm as lgb, torch, timm
from torch.utils.data import Dataset, DataLoader
SEED=42; F="features"; TE="features/trackE"; OUT="results_clean"; MAN=["deepfakes","face2face","faceswap","neuraltextures"]
DIR={"deepfakes":"Deepfakes","face2face":"Face2Face","faceswap":"FaceSwap","neuraltextures":"NeuralTextures"}
DEV='cuda' if torch.cuda.is_available() else 'cpu'
MEAN=np.array([0.485,0.456,0.406],np.float32); STD=np.array([0.229,0.224,0.225],np.float32)
def base(p): return os.path.splitext(os.path.basename(str(p)))[0]
def method(p):
    for m,d in DIR.items():
        if f"/{d}/" in p: return m
    return "real" if ("youtube" in p or "original_sequences" in p) else ("celebdf" if "Celeb-DF" in p else "?")
def commit():
    try: return subprocess.check_output(["git","rev-parse","--short","HEAD"],text=True).strip()
    except: return "nogit"

# ---------- Xception: re-score frozen weight on crops (inference) ----------
class DS(Dataset):
    def __init__(self,df): self.df=df.reset_index(drop=True)
    def __len__(self): return len(self.df)
    def __getitem__(self,i):
        r=self.df.iloc[i]; im=cv2.imread(r.crop_path)
        if im is None: im=np.zeros((299,299,3),np.uint8)
        im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB).astype(np.float32)/255.; im=(im-MEAN)/STD
        return torch.from_numpy(im.transpose(2,0,1)), i
def xcep_score(manifest):
    m=timm.create_model('legacy_xception',num_classes=1); m.load_state_dict(torch.load("data_xception/xception_best.pt",map_location=DEV)); m=m.to(DEV).eval()
    ps=np.zeros(len(manifest))
    with torch.no_grad():
        for x,idx in DataLoader(DS(manifest),batch_size=128,num_workers=8):
            ps[idx.numpy()]=torch.sigmoid(m(x.to(DEV))).cpu().numpy().ravel()
    manifest=manifest.copy(); manifest["p"]=ps; return manifest
print("re-scoring frozen Xception on crops (inference) ...",flush=True)
cdm=pd.read_csv("data_xception/manifest_celebdf.csv").drop_duplicates("crop_path")
cdm=xcep_score(cdm); xcd=cdm.groupby(["video","label"])["p"].mean().reset_index(); xcd["vid"]=xcd.video.map(base)
ffm=pd.read_csv("data_xception/manifest_ffpp.csv").drop_duplicates("crop_path")
ffm["manip"]=ffm.crop_path.map(lambda p: p.split("/crops/")[1].split("/")[0])   # crops/<manip>/...
ffm=xcep_score(ffm); xff=ffm.groupby(["video","manip","split","label"])["p"].mean().reset_index(); xff["vid"]=xff.video.map(base)
print(f"  Xception: celebdf {len(xcd)} vids, ffpp {len(xff)} (video,manip) rows",flush=True)

# ---------- PRISM: frozen ensemble, per-video probs (prob-avg) ----------
ev=pd.read_csv(f"{TE}/plain_everyone_E3.csv"); ev["src"]=ev.video_path.map(method)
for c in FEATS: ev[c]=pd.to_numeric(ev[c],errors="coerce").replace([np.inf,-np.inf],np.nan)
ff=make_splits(ev[ev.src.isin(["real"]+MAN)].copy()); med=ff[ff.partition=="train"][FEATS].median()
tr=pd.concat([ff[(ff.src=="real")&(ff.partition=="train")].assign(label=0)]+[ff[(ff.src==m)&(ff.partition=="train")].assign(label=1) for m in MAN],ignore_index=True)
ffval=pd.concat([ff[(ff.src=="real")&(ff.partition=="val")].assign(label=0)]+[ff[(ff.src==m)&(ff.partition=="val")].assign(label=1) for m in MAN],ignore_index=True)
cdtest=pd.read_csv(f"{TE}/plain_celebdf_test.csv"); cdtest["label"]=cdtest.get("label",1)
fftest=pd.read_csv(f"{TE}/plain_ffpp_test.csv"); fftest["src"]=fftest.video_path.map(method); fftest["label"]=(fftest.src!="real").astype(int)
for df in (cdtest,fftest):
    for c in FEATS: df[c]=pd.to_numeric(df[c],errors="coerce")
def L(): return lgb.LGBMClassifier(n_estimators=300,learning_rate=0.05,num_leaves=31,min_child_samples=20,max_depth=6,class_weight="balanced",random_state=SEED,verbose=-1,n_jobs=-1,deterministic=True,force_row_wise=True)
def prism_prob(cols, frames):
    sc=StandardScaler().fit(tr[cols].fillna(med[cols]).values); ytr=tr.label.values.astype(int)
    Xtr=sc.transform(tr[cols].fillna(med[cols]).values)
    models=[RandomForestClassifier(n_estimators=400,max_depth=8,min_samples_leaf=5,class_weight="balanced",random_state=SEED,n_jobs=-1),
            ExtraTreesClassifier(n_estimators=600,max_depth=10,min_samples_leaf=4,class_weight="balanced",random_state=SEED,n_jobs=-1),L()]
    for m in models: m.fit(Xtr,ytr)
    return [np.mean([m.predict_proba(sc.transform(fr[cols].fillna(med[cols]).values))[:,1] for m in models],axis=0) for fr in frames]
COLS={"196D":FEATS,"53D":FEATS[:53],"50D":FEATS[:50]}
P={}
for tag,cols in COLS.items():
    pv,pcd,pft=prism_prob(cols,[ffval,cdtest,fftest]); P[tag]=dict(val=pv,cd=pcd,ff=pft)
# F1-optimal threshold on FF++ val (per model)
def f1opt(prob,y):
    ts=np.unique(np.round(prob,3)); best=(0.5,-1)
    for t in ts:
        f=f1_score(y,(prob>=t).astype(int),zero_division=0)
        if f>best[1]: best=(float(t),f)
    return best[0]
yval=ffval.label.values.astype(int)
THR={tag:{"theta":0.50,"f1opt_ffval":round(f1opt(P[tag]["val"],yval),3)} for tag in COLS}
# Xception F1-opt threshold on FF++ val
xval=xff[xff.split=="val"]; yxv=xval.label.values.astype(int)
THR["Xception"]={"theta":0.50,"f1opt_ffval":round(f1opt(xval.p.values,yxv),3)}
print("  thresholds:",{k:v["f1opt_ffval"] for k,v in THR.items()},flush=True)

# ---------- assemble aligned per-video frames ----------
cdtest["vid"]=cdtest.video_path.map(base); cdtest["id"]=cdtest.video_path.map(lambda p:(re.findall(r"id(\d+)",str(p)) or [base(p)])[0])
fftest["vid"]=fftest.video_path.map(base); fftest["manip"]=fftest.src
prism_cd=cdtest[["vid","id","label"]].copy()
for tag in COLS: prism_cd[f"p_{tag}"]=P[tag]["cd"]
prism_ff=fftest[["vid","manip","label"]].copy(); prism_ff["p_196D"]=P["196D"]["ff"]
# join Xception
cd=prism_cd.merge(xcd[["vid","p"]].rename(columns={"p":"p_xcep"}),on="vid",how="inner")
ff=prism_ff.merge(xff[xff.split=="test"][["vid","manip","p"]].rename(columns={"p":"p_xcep"}),on=["vid","manip"],how="inner")
print(f"  joined: celebdf {len(cd)} (of {len(prism_cd)} sealed-test), ffpp {len(ff)} (of {len(prism_ff)})",flush=True)

# ---------- McNemar ----------
def correct(prob,y,t): return ((prob>=t).astype(int)==y)
def mcnemar(pc, oc):   # prism-correct, other-correct (bool arrays)
    b=int(np.sum(pc&~oc)); c=int(np.sum(~pc&oc)); a=int(np.sum(pc&oc)); d=int(np.sum(~pc&~oc)); n=b+c
    if n==0: return dict(a=a,b=b,c=c,d=d,n_discordant=0,test="none",chi2=None,p_value=1.0,direction="tie")
    if n<25:
        p=binomtest(min(b,c),n,0.5,alternative="two-sided").pvalue; test="exact_binomial"; chi2=None
    else:
        chi2=(abs(b-c)-1)**2/n; p=float(1-chi2dist.cdf(chi2,1)); test="mcnemar_cc"; chi2=round(chi2,4)
    return dict(a=a,b=b,c=c,d=d,n_discordant=n,test=test,chi2=chi2,p_value=float(p),
                direction=("PRISM better" if b>c else "baseline better" if c>b else "tie"))
comparisons={}
def run_cmp(name, prism_prob_arr, other_prob_arr, y, t_prism, t_other, prism_tag, other_tag):
    pc=correct(prism_prob_arr,y,t_prism); oc=correct(other_prob_arr,y,t_other)
    r=mcnemar(pc,oc); r.update(n=len(y),prism=prism_tag,baseline=other_tag,t_prism=t_prism,t_baseline=t_other,
        prism_acc=round(float(pc.mean()),4),baseline_acc=round(float(oc.mean()),4)); return r
ycd=cd.label.values.astype(int); yff=ff.label.values.astype(int)
for thr_name in ("theta","f1opt_ffval"):
    tag=thr_name
    # (a) 196 vs Xception celebdf
    comparisons[f"196D_vs_Xception__CelebDFtest__{tag}"]=run_cmp("cdX",cd.p_196D.values,cd.p_xcep.values,ycd,THR["196D"][thr_name],THR["Xception"][thr_name],"PRISM_196D","Xception")
    # (b) 196 vs Xception ffpp
    comparisons[f"196D_vs_Xception__FFpptest__{tag}"]=run_cmp("ffX",ff.p_196D.values,ff.p_xcep.values,yff,THR["196D"][thr_name],THR["Xception"][thr_name],"PRISM_196D","Xception")
    # (c) 196 vs 50 celebdf ; (d) 196 vs 53 celebdf
    comparisons[f"196D_vs_50D__CelebDFtest__{tag}"]=run_cmp("c50",cd.p_196D.values,cd.p_50D.values,ycd,THR["196D"][thr_name],THR["50D"][thr_name],"PRISM_196D","PRISM_50D")
    comparisons[f"196D_vs_53D__CelebDFtest__{tag}"]=run_cmp("c53",cd.p_196D.values,cd.p_53D.values,ycd,THR["196D"][thr_name],THR["53D"][thr_name],"PRISM_196D","PRISM_53D")

# ---------- Wilcoxon: PRISM-196 vs Xception AUC per identity-grouped CV fold (celebdf) ----------
ids=cd.id.values; gk=GroupKFold(5); aP=[];aX=[]
for _,te in gk.split(cd,ycd,ids):
    if len(np.unique(ycd[te]))>1:
        aP.append(roc_auc_score(ycd[te],cd.p_196D.values[te])); aX.append(roc_auc_score(ycd[te],cd.p_xcep.values[te]))
nfold=len(aP); wstat,wp=(wilcoxon(aP,aX) if nfold>=1 and len(set(np.round(np.array(aP)-np.array(aX),6)))>1 else (float('nan'),float('nan')))
wil=dict(test="wilcoxon_signed_rank",target="CelebDF sealed test, identity-grouped CV",n_folds=int(nfold),
    prism196_fold_auc=[round(x,4) for x in aP],xception_fold_auc=[round(x,4) for x in aX],
    mean_prism196=round(float(np.mean(aP)),4),mean_xception=round(float(np.mean(aX)),4),
    statistic=(None if np.isnan(wstat) else float(wstat)),p_value=(None if np.isnan(wp) else float(wp)),
    min_achievable_p_note=f"With {nfold} folds the minimum two-sided Wilcoxon p is {2**-(nfold-1) if nfold>0 else 1:.4f}; it CANNOT reach p<0.05, so this test cannot establish significance regardless of the effect.")

# ---------- deliverable predictions CSV (PRISM=196D) ----------
def pred_rows(frame,y,prob_p,prob_x,dataset,t_p,t_x):
    return pd.DataFrame(dict(video_name=frame.vid.values,dataset=dataset,ground_truth=y,
        prism_pred_label=(prob_p>=t_p).astype(int),prism_prob=np.round(prob_p,6),
        xception_pred_label=(prob_x>=t_x).astype(int),xception_prob=np.round(prob_x,6)))
predcsv=pd.concat([
    pred_rows(cd,ycd,cd.p_196D.values,cd.p_xcep.values,"CelebDF_test_SEALED",THR["196D"]["theta"],THR["Xception"]["theta"]),
    pred_rows(ff,yff,ff.p_196D.values,ff.p_xcep.values,"FFpp_test",THR["196D"]["theta"],THR["Xception"]["theta"])],ignore_index=True)
os.makedirs("196D_FINAL/03_results",exist_ok=True)
predcsv.to_csv("196D_FINAL/03_results/prism_vs_xception_predictions.csv",index=False)

res=dict(provenance=dict(script="exp_mcnemar_wilcoxon.py",git_commit=commit(),seed=SEED,date=datetime.date.today().isoformat(),
    kind="POST-FREEZE DESCRIPTIVE (no tuning/model changes; Xception re-scored by inference from xception_best.pt)",
    n_celebdf_joined=int(len(cd)),n_ffpp_joined=int(len(ff)),thresholds=THR),
    mcnemar=comparisons, wilcoxon=wil)
json.dump(res,open(f"{OUT}/mcnemar_wilcoxon_results.json","w"),indent=1)
print("\n=== McNemar (theta=0.50) ===")
for k,v in comparisons.items():
    if k.endswith("theta"): print(f"  {k}: b={v['b']} c={v['c']} n={v['n']} {v['test']} p={v['p_value']:.3g} -> {v['direction']} (prismAcc {v['prism_acc']} vs {v['baseline_acc']})")
print("=== Wilcoxon ===")
print(f"  folds={wil['n_folds']} PRISM196 mean AUC {wil['mean_prism196']} vs Xception {wil['mean_xception']} | p={wil['p_value']}")
print(f"  {wil['min_achievable_p_note']}")
print(f"saved {OUT}/mcnemar_wilcoxon_results.json + 196D_FINAL/03_results/prism_vs_xception_predictions.csv (commit {commit()})")
