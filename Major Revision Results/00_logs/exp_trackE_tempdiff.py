#!/usr/bin/env python3
"""Track E — TEMPORAL-DIFFERENCE / RELATIVE-FLICKER representation. DEV only; sealed=0. ZERO extraction
(reuses persisted per-frame spatial series). Hypothesis: every lever so far used ABSOLUTE feature magnitudes,
which shift across domains (why DA failed). Frame-to-frame differences d_t=x_t-x_{t-1} cancel the per-video/
per-domain DC offset; RELATIVE flicker mean|d|/median|x| is dimensionless (same transfer property as E1
order-stats and E4 face/bg ratio). Also targets the Celeb-DF fake artifact directly (temporal flicker).
Per spatial channel (13): td_meanabs, td_std, td_p90abs, td_max, td_relflicker(dimensionless), td_signchg(rate)
-> 13x6 = 78 TD features. Eval: base 196-D vs 196+78 vs TD-only. RandomForest, identity-grouped celebdf_dev CV,
real/fake recall separately. Threshold cross +0.03 (Holm across full ledger, applied at freeze).
PRE-REGISTERED prediction (written before measuring): cross POSITIVE (offset-invariant + dimensionless flicker
transfers); in-dist positive concentrated in F2F/NT (intermittent re-render flicker); DF/FS smaller.
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
SPATIAL13=["s_noise_vmr","s_noise_res_std","s_noise_hf_ratio","s_prnu_energy","s_prnu_face_periph",
           "s_shadow_score","s_face_bg_diff","s_benford_dev","s_block_artifact","s_dbl_compress",
           "s_blur_mag","s_flow_mag","s_flow_dir_consist"]
def method(p):
    for m,d in DIR.items():
        if f"/{d}/" in p: return m
    return "real" if "youtube" in p else ("celebdf" if "Celeb-DF" in p else "?")
def commit():
    try: return subprocess.check_output(["git","rev-parse","--short","HEAD"],text=True).strip()
    except: return "nogit"
def RF(): return RandomForestClassifier(n_estimators=400,max_depth=8,min_samples_leaf=5,class_weight="balanced",random_state=SEED,n_jobs=-1)
EPS=1e-8
TD_FEATS=[f"td_{s}_{k}" for s in SPATIAL13 for k in ("meanabs","std","p90abs","max","relflick","signchg")]

def td_for_video(g):
    g=g.sort_values("frame"); row={}
    for s in SPATIAL13:
        x=pd.to_numeric(g[s],errors="coerce").values.astype(float); x=x[np.isfinite(x)]
        if len(x)<3:
            for k in ("meanabs","std","p90abs","max","relflick","signchg"): row[f"td_{s}_{k}"]=0.0
            continue
        d=np.diff(x); ad=np.abs(d); scale=np.median(np.abs(x))+EPS
        row[f"td_{s}_meanabs"]=float(ad.mean()); row[f"td_{s}_std"]=float(d.std())
        row[f"td_{s}_p90abs"]=float(np.percentile(ad,90)); row[f"td_{s}_max"]=float(ad.max())
        row[f"td_{s}_relflick"]=float(ad.mean()/scale)                    # dimensionless
        row[f"td_{s}_signchg"]=float((np.diff(np.sign(d))!=0).mean()) if len(d)>1 else 0.0
    return pd.Series(row)

# build TD features from persisted per-frame series. ffpp keyed by FULL path (basename collisions exist);
# celebdf keyed by basename but basenames are unique within celebdf -> map to full path via plain_everyone.
def load_td(path):
    pf=pd.read_csv(path)
    key="video" if "video" in pf.columns else "video_path"
    td=pf.groupby(key).apply(td_for_video).reset_index().rename(columns={key:"key"})
    return td
print("building TD features from per-frame series ...",flush=True)
ev=pd.read_csv(f"{TE}/plain_everyone_E3.csv"); ev["src"]=ev.video_path.map(method)
cd_bn2path=ev[ev.src=="celebdf"].assign(bn=lambda d:d.video_path.map(lambda p:os.path.basename(str(p)))).set_index("bn")["video_path"].to_dict()
td_ff=load_td(f"{TE}/perframe_ffpp_trainval_fixed.csv"); td_ff["video_path"]=td_ff["key"]            # ffpp: key IS full path
td_cd=load_td(f"{TE}/perframe_celebdf_dev.csv"); td_cd["video_path"]=td_cd["key"].map(cd_bn2path)     # celebdf: basename -> full path
td=pd.concat([td_ff.drop(columns="key"), td_cd.dropna(subset=["video_path"]).drop(columns="key")],ignore_index=True)
print(f"  TD rows: {len(td)} (ffpp {len(td_ff)} + celebdf {td_cd['video_path'].notna().sum()})  cols: {len(TD_FEATS)}",flush=True)

before=len(ev); ev=ev.merge(td,on="video_path",how="inner");
print(f"  merged 196-D+TD: {len(ev)}/{before} videos matched",flush=True)
ALL=FEATS+TD_FEATS
for c in ALL: ev[c]=pd.to_numeric(ev[c],errors="coerce").replace([np.inf,-np.inf],np.nan)
ff=make_splits(ev[ev.src.isin(["real"]+MAN)].copy()); cd=ev[ev.src=="celebdf"].copy()
yc=cd.label.values.astype(int)
cd_ids=cd.video_path.map(lambda p:(re.findall(r"id(\d+)",str(p)) or [os.path.basename(str(p))])[0]).values
real_tr=ff[(ff.src=="real")&(ff.partition=="train")]
tr=pd.concat([real_tr.assign(label=0)]+[ff[(ff.src==m)&(ff.partition=="train")].assign(label=1) for m in MAN],ignore_index=True)
val={m:pd.concat([ff[(ff.src=='real')&(ff.partition=='val')],ff[(ff.src==m)&(ff.partition=='val')]],ignore_index=True) for m in MAN}
def cv(p):
    a=[roc_auc_score(yc[i],p[i]) for _,i in GroupKFold(5).split(p,yc,cd_ids) if len(np.unique(yc[i]))>1]
    return round(float(np.mean(a)),4),round(float(np.std(a)),4)
def rec(p,t=0.5):
    pr=(p>=t).astype(int); return round(float((pr[yc==0]==0).mean()),3),round(float((pr[yc==1]==1).mean()),3)
def run(cols):
    med=tr[cols].median(); TR=tr[cols].fillna(med); CD=cd[cols].fillna(med)
    sc=StandardScaler().fit(TR.values); m=RF().fit(sc.transform(TR.values),tr.label.values.astype(int))
    ys=[];ps=[]
    for mm in MAN: ys.append(val[mm].label.values.astype(int)); ps.append(m.predict_proba(sc.transform(val[mm][cols].fillna(med).values))[:,1])
    ind=round(roc_auc_score(np.concatenate(ys),np.concatenate(ps)),4)
    pc=m.predict_proba(sc.transform(CD.values))[:,1]; cm,cs=cv(pc); rr,fr=rec(pc)
    return dict(indist_auc=ind,celebdf_dev_cv_mean=cm,celebdf_dev_cv_std=cs,real_recall=rr,fake_recall=fr)
base=run(FEATS); stk=run(FEATS+TD_FEATS); tdonly=run(TD_FEATS)
res=dict(provenance=dict(script="exp_trackE_tempdiff.py",git_commit=commit(),seed=SEED,date=datetime.date.today().isoformat(),
    axis_dev_only=True,sealed_touched=False,classifier="RandomForest_d8",n_TD=len(TD_FEATS),extraction="none (reused per-frame)"),
    base_196D=base, plus_TD_274D=stk, TD_only_78D=tdonly,
    delta_cross=round(stk["celebdf_dev_cv_mean"]-base["celebdf_dev_cv_mean"],4),
    delta_indist=round(stk["indist_auc"]-base["indist_auc"],4))
os.makedirs(OUT,exist_ok=True); json.dump(res,open(f"{OUT}/trackE_tempdiff_dev.json","w"),indent=1)
print("="*76);print("TRACK E — TEMPORAL-DIFFERENCE / RELATIVE-FLICKER (RF; celebdf_dev CV)");print("="*76)
for tag,r in [("196-D base",base),("196+78 TD",stk),("TD-only 78",tdonly)]:
    print(f"  {tag:14s} in-dist {r['indist_auc']:.4f} | cross {r['celebdf_dev_cv_mean']:.4f} ±{r['celebdf_dev_cv_std']:.3f} | realRec {r['real_recall']} fakeRec {r['fake_recall']}")
print(f"\n  Δ (196+TD vs base): cross {res['delta_cross']:+.4f} | in-dist {res['delta_indist']:+.4f}  (bar cross +0.03)")
print(f"saved {OUT}/trackE_tempdiff_dev.json (commit {commit()})")
