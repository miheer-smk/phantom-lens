#!/usr/bin/env python3
"""Track D — Group J (domain-invariant reformulations), CHEAP subset. DEV only; sealed untouched.
J-a: additive dimensionless RATIOS among existing 50-D magnitude features (tree-usable, scale-canceling).
J-b: train-fitted QUANTILE ALIGNMENT of the 53-D representation (distribution-shape-not-scale; for a tree
     in-distribution AUC is ~unchanged and only the cross-dataset mapping changes).
Baseline = 53-D (50-D + G1). Measured on FF++ val (in-dist) + celebdf_dev (cross). ΔAUC + Cohen's d.
Train-only imputer/scaler, seed 42, locked LightGBM. (J-extract face-bg / within-video contrast deferred.)
"""
import os, sys, json, subprocess, datetime
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore"); sys.path.insert(0, "src")
from protocol import make_splits
import roi_config as RC
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

SEED=42; F="features"; TD=f"{F}/trackD"; OUT="results_clean"; EPS=1e-6
G1=RC.CANDIDATE_GROUPS["G1_mouth_instability"]; MAN=["deepfakes","face2face","faceswap","neuraltextures"]
def bn(p): return os.path.basename(str(p))
def commit():
    try: return subprocess.check_output(["git","rev-parse","--short","HEAD"],text=True).strip()
    except: return "nogit"
def LGBM(): return lgb.LGBMClassifier(n_estimators=200,max_depth=6,learning_rate=0.05,num_leaves=31,
    min_child_samples=20,class_weight="balanced",random_state=SEED,verbose=-1,n_jobs=1,deterministic=True,force_row_wise=True)

# dimensionless ratios among magnitude features (numerator, denominator)
RATIOS=[("j_noise_to_blur","s_noise_res_std","s_blur_mag"),
        ("j_flow_to_jitter","s_flow_mag","t_landmark_jitter"),
        ("j_accel_to_jitter","t_landmark_accel_var","t_landmark_jitter"),
        ("j_prnu_to_noise","s_prnu_energy","s_noise_res_std"),
        ("j_rppg_snr_to_prom","t_rppg_snr","t_rppg_peak_prominence"),
        ("j_texture_to_noise","t_texture_warp_residual","s_noise_res_std"),
        ("j_skinjit_to_flow","t_skin_color_jitter","s_flow_mag"),
        ("j_interpup_to_rigid","t_interpupillary_std","t_rigid_dist_var"),
        ("j_nosebridge_to_rigid","t_nose_bridge_std","t_rigid_dist_var"),
        ("j_noisevmr_to_res","s_noise_vmr","s_noise_res_std"),
        ("j_dctstd_to_texture","t_dct_temporal_std","t_texture_warp_residual"),
        ("j_residentropy_to_specent","t_residual_entropy","t_noise_spectral_entropy")]
J_FEATS=[r[0] for r in RATIOS]
def add_ratios(df):
    d=df.copy()
    for name,num,den in RATIOS:
        d[name]=np.abs(pd.to_numeric(d[num],errors="coerce"))/(np.abs(pd.to_numeric(d[den],errors="coerce"))+EPS)
        d[name]=d[name].replace([np.inf,-np.inf],np.nan)
    return d

def ff_set(name):
    o=pd.read_csv(f"{F}/ffpp_{'original' if name=='real' else name}_c23.csv")
    r=pd.read_csv(f"{F}/roi_{'original' if name=='real' else name}_c23.csv")
    o["_b"]=o.video_path.map(bn); r["_b"]=r.video_path.map(bn)
    return make_splits(o.merge(r[["_b"]+G1],on="_b",how="inner"))
FF={k:add_ratios(ff_set(k)) for k in ["real"]+MAN}
FC=sorted([c for c in FF["real"].columns if c[:2] in ("s_","t_")])
C53=FC+G1; C53J=FC+G1+J_FEATS
def imp(df,cols):
    d=df.copy()
    for c in cols: d[c]=pd.to_numeric(d[c],errors="coerce").replace([np.inf,-np.inf],np.nan)
    d[cols]=d[cols].fillna(d.loc[d.partition=="train",cols].median())
    return d
FF={k:imp(v,C53J) for k,v in FF.items()}

cd=pd.read_csv(f"{F}/celebdf_features.csv"); cd["_b"]=cd.video_path.map(bn)
g1=pd.read_csv(f"{TD}/G1_celebdf_dev.csv"); g1["_b"]=g1.video_path.map(bn)
CD=add_ratios(cd.merge(g1[["_b"]+G1],on="_b",how="inner"))

def boot_ci(y,pa,pb,n=1500,s=SEED):
    rng=np.random.RandomState(s); d=[]
    for _ in range(n):
        i=rng.randint(0,len(y),len(y))
        if len(np.unique(y[i]))<2: continue
        d.append(roc_auc_score(y[i],pa[i])-roc_auc_score(y[i],pb[i]))
    return round(float(np.percentile(d,2.5)),4),round(float(np.percentile(d,97.5)),4)
def fit_pred(tr,te,cols,quantile=False):
    Xtr,Xte=tr[cols].values,te[cols].values
    if quantile:
        qt=QuantileTransformer(output_distribution="normal",n_quantiles=min(1000,len(tr)),random_state=SEED).fit(Xtr)
        Xtr,Xte=qt.transform(Xtr),qt.transform(Xte)
    sc=StandardScaler().fit(Xtr); m=LGBM().fit(sc.transform(Xtr),tr.label.values.astype(int))
    return m.predict_proba(sc.transform(Xte))[:,1]

# ---- IN-DIST per manip (val): 53 vs 53+J-ratios ----
indist=[]
for m in MAN:
    tr=pd.concat([FF["real"][FF["real"].partition=="train"],FF[m][FF[m].partition=="train"]],ignore_index=True)
    va=pd.concat([FF["real"][FF["real"].partition=="val"],  FF[m][FF[m].partition=="val"]],  ignore_index=True)
    y=va.label.values.astype(int)
    p53=fit_pred(tr,va,C53); pj=fit_pred(tr,va,C53J)
    a,aj=roc_auc_score(y,p53),roc_auc_score(y,pj); lo,hi=boot_ci(y,pj,p53)
    indist.append(dict(manip=m,auc_53=round(a,4),auc_53J=round(aj,4),delta=round(aj-a,4),ci=[lo,hi]))
m53=np.mean([r["auc_53"] for r in indist]); m53j=np.mean([r["auc_53J"] for r in indist])

# ---- CROSS-DATASET (celebdf_dev): 53 vs 53+J-ratios  AND  53 raw vs 53 quantile-aligned ----
trall=pd.concat([FF["real"][FF["real"].partition=="train"]]+[FF[m][FF[m].partition=="train"] for m in MAN],ignore_index=True)
med=trall[C53J].median()
CDi=CD.copy()
for c in C53J: CDi[c]=pd.to_numeric(CDi[c],errors="coerce").replace([np.inf,-np.inf],np.nan).fillna(med[c])
yc=CDi.label.values.astype(int)
pc53=fit_pred(trall,CDi,C53); pcj=fit_pred(trall,CDi,C53J)
pcq=fit_pred(trall,CDi,C53,quantile=True)          # J-b quantile alignment of the 53-D rep
c53,cj,cq=roc_auc_score(yc,pc53),roc_auc_score(yc,pcj),roc_auc_score(yc,pcq)
jlo,jhi=boot_ci(yc,pcj,pc53); qlo,qhi=boot_ci(yc,pcq,pc53)

def cohen(df,f):
    a=df[df.label==0][f].astype(float); b=df[df.label==1][f].astype(float)
    sp=np.sqrt(((len(a)-1)*a.var()+(len(b)-1)*b.var())/max(len(a)+len(b)-2,1))
    return round(float((b.mean()-a.mean())/(sp+1e-9)),3)
valpool=pd.concat([FF["real"][FF["real"].partition=="val"]]+[FF[m][FF[m].partition=="val"] for m in MAN],ignore_index=True)
dcoh={j:dict(val_d=cohen(valpool,j),celebdf_dev_d=cohen(CDi,j)) for j in J_FEATS}

res=dict(provenance=dict(script="exp_trackD_J_eval.py",git_commit=commit(),seed=SEED,date=datetime.date.today().isoformat(),
    axis_dev_only=True,sealed_touched=False,J_ratio_features=J_FEATS,note="J-cheap; J-extract deferred"),
    in_distribution=dict(per_manip=indist,mean_53=round(m53,4),mean_53J=round(m53j,4),mean_delta=round(m53j-m53,4)),
    cross_dataset_celebdf_dev=dict(n=int(len(CDi)),auc_53=round(c53,4),
        auc_53_plus_Jratios=round(cj,4),delta_ratios=round(cj-c53,4),ci_ratios=[jlo,jhi],
        auc_53_quantile_aligned=round(cq,4),delta_quantile=round(cq-c53,4),ci_quantile=[qlo,qhi]),
    per_feature_cohens_d=dcoh)
json.dump(res,open(f"{OUT}/trackD_J_dev.json","w"),indent=1)
print("="*70);print("TRACK D — GROUP J (domain-invariant reformulations, cheap) — DEV");print("="*70)
print("IN-DIST (FF++ val) 53 vs 53+J-ratios:")
for r in indist: print(f"  {r['manip']:15s} 53={r['auc_53']:.4f} 53+J={r['auc_53J']:.4f} Δ={r['delta']:+.4f} CI{r['ci']}")
print(f"  MEAN            53={m53:.4f} 53+J={m53j:.4f} Δ={m53j-m53:+.4f}")
print(f"CROSS (celebdf_dev n={len(CDi)}):")
print(f"  53={c53:.4f}  53+J-ratios={cj:.4f} Δ={cj-c53:+.4f} CI[{jlo},{jhi}]")
print(f"  53 quantile-aligned={cq:.4f} Δ={cq-c53:+.4f} CI[{qlo},{qhi}]")
print("per-feature Cohen's d (val | celebdf_dev):")
for j,dd in dcoh.items(): print(f"  {j:26s} {dd['val_d']:+.3f} | {dd['celebdf_dev_d']:+.3f}")
print(f"saved {OUT}/trackD_J_dev.json (commit {commit()})")
