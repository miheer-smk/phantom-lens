#!/usr/bin/env python3
"""DeLong significance tests on stored prediction scores (identity-disjoint test sets).
(A) 53-D vs 50-D per manipulation — is the G1 gain significant?
(B) Per-pillar: full-50 vs leave-one-pillar-out, Holm-corrected across 20 pillars per dataset.
All from actual model prediction scores; no estimated p-values (§0)."""
import os,sys,json,subprocess,datetime
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore"); sys.path.insert(0,"src")
from protocol import make_splits; import roi_config as RC
from delong import delong_roc_test, holm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
SEED=42; F="features"; OUT="results_clean"
def base(p): return os.path.basename(str(p))
G1=RC.CANDIDATE_GROUPS["G1_mouth_instability"]
PILLARS=json.load(open("splits/pillar_map.json"))
MAN={"Deepfakes":"deepfakes","Face2Face":"face2face","FaceSwap":"faceswap","NeuralTextures":"neuraltextures"}
def load(name,roi=False):
    df=pd.read_csv(f"{F}/ffpp_{name}_c23.csv") if name!="real" else pd.read_csv(f"{F}/ffpp_original_c23.csv")
    return df
real=pd.read_csv(f"{F}/ffpp_original_c23.csv"); FC=sorted([c for c in real.columns if c[:2] in ("s_","t_")])
def clean(df,cols):
    d=df.copy()
    for c in cols: d[c]=pd.to_numeric(d[c],errors="coerce").replace([np.inf,-np.inf],np.nan); d[c]=d[c].fillna(d[c].median())
    return d
def LGBM(): return lgb.LGBMClassifier(n_estimators=200,max_depth=6,learning_rate=0.05,num_leaves=31,min_child_samples=20,class_weight="balanced",random_state=SEED,verbose=-1,n_jobs=-1)
def commit():
    try: return subprocess.check_output(["git","rev-parse","--short","HEAD"],text=True).strip()
    except: return "nogit"
def scores(tr,te,cols):
    sc=StandardScaler().fit(tr[cols].values); clf=LGBM(); clf.fit(sc.transform(tr[cols].values),tr['label'].values.astype(int))
    return clf.predict_proba(sc.transform(te[cols].values))[:,1]

# merge original + G1 per manip (for A); original only (for B)
realm=make_splits(pd.read_csv(f"{F}/ffpp_original_c23.csv"))
roiR=pd.read_csv(f"{F}/roi_original_c23.csv"); roiR["_b"]=roiR.video_path.map(base)
def with_g1(name):
    o=pd.read_csv(f"{F}/ffpp_{name}_c23.csv") if name!="real" else pd.read_csv(f"{F}/ffpp_original_c23.csv")
    r=pd.read_csv(f"{F}/roi_{'original' if name=='real' else name}_c23.csv");
    o["_b"]=o.video_path.map(base); r["_b"]=r.video_path.map(base)
    m=o.merge(r[["_b"]+G1],on="_b",how="inner"); return make_splits(m)
Mg={k:with_g1(v) for k,v in MAN.items()}; realg=with_g1("real")

resA=[]
print("="*74+"\n(A) DeLong: 53-D vs 50-D per manipulation (identity-disjoint test)\n"+"="*74)
for disp,short in MAN.items():
    man=clean(Mg[disp],FC+G1); rl=clean(realg,FC+G1)
    tr=pd.concat([rl[rl.partition=="train"],man[man.partition=="train"]],ignore_index=True)
    te=pd.concat([rl[rl.partition=="test"], man[man.partition=="test"]], ignore_index=True)
    y=te['label'].values.astype(int)
    p50=scores(tr,te,FC); p53=scores(tr,te,FC+G1)
    a53,a50,z,p=delong_roc_test(y,p53,p50)
    resA.append(dict(manip=disp,auc_50=round(a50,4),auc_53=round(a53,4),delta=round(a53-a50,4),z=round(z,3),p_value=p))
    sig="SIGNIFICANT" if p<0.05 else "n.s."
    print(f"  {disp:15s} 50-D={a50:.4f} 53-D={a53:.4f} Δ={a53-a50:+.4f}  DeLong z={z:+.3f} p={p:.4g}  [{sig}]")

print("\n"+"="*74+"\n(B) DeLong: full-50 vs leave-one-pillar-out (Holm-corrected per dataset)\n"+"="*74)
resB=[]
# in-distribution manips
def eval_pillars(dsname, tr, te):
    y=te['label'].values.astype(int); pf=scores(tr,te,FC); full=roc_auc_score(y,pf)
    raw=[]; rows=[]
    for pil,feats in PILLARS.items():
        keep=[c for c in FC if c not in feats]
        pa=scores(tr,te,keep)
        af,aa,z,p=delong_roc_test(y,pf,pa)  # full vs ablated
        raw.append(p); rows.append([pil,round(full,4),round(aa,4),round(full-aa,4),round(z,3),p])
    hp=holm(raw)
    for r,hpi in zip(rows,hp):
        resB.append(dict(dataset=dsname,pillar=r[0],full_auc=r[1],loGo_auc=r[2],delta=r[3],z=r[4],p_value=r[5],p_holm=round(hpi,4)))
    # print significant-after-Holm
    sigs=[r for r,hpi in zip(rows,hp) if hpi<0.05]
    print(f"  {dsname}: {len(sigs)}/20 pillars significant after Holm — "+
          (", ".join(f"{r[0]}(Δ{r[3]:+.3f},p={hpi:.3f})" for r,hpi in zip(rows,hp) if hpi<0.05) or "none"))
    return
for disp,short in MAN.items():
    man=clean(pd.read_csv(f"{F}/ffpp_{short}_c23.csv"),FC); man=make_splits(man)
    rl=clean(pd.read_csv(f"{F}/ffpp_original_c23.csv"),FC); rl=make_splits(rl)
    tr=pd.concat([rl[rl.partition=="train"],man[man.partition=="train"]],ignore_index=True)
    te=pd.concat([rl[rl.partition=="test"], man[man.partition=="test"]], ignore_index=True)
    eval_pillars(disp,tr,te)
# CelebDF zero-shot
cd=clean(pd.read_csv(f"{F}/celebdf_features.csv"),FC)
rl=make_splits(clean(pd.read_csv(f"{F}/ffpp_original_c23.csv"),FC))
trf=pd.concat([rl[rl.partition=="train"]]+[make_splits(clean(pd.read_csv(f"{F}/ffpp_{s}_c23.csv"),FC)).query("partition=='train'") for s in MAN.values()],ignore_index=True)
eval_pillars("CelebDF",trf,cd)

json.dump({"provenance":{"script":"run_delong.py","git_commit":commit(),"seed":SEED,"date":datetime.date.today().isoformat(),
    "test":"DeLong paired AUC; Holm across 20 pillars per dataset"},"A_53vs50":resA,"B_pillars":resB},
    open(f"{OUT}/delong.json","w"),indent=1)
pd.DataFrame(resA).to_csv(f"{OUT}/delong_53vs50.csv",index=False)
pd.DataFrame(resB).to_csv(f"{OUT}/delong_pillars.csv",index=False)
print(f"\nsaved {OUT}/delong.json, delong_53vs50.csv, delong_pillars.csv (commit {commit()})")
