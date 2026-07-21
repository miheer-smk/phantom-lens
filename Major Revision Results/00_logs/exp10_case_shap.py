#!/usr/bin/env python3
"""EXP-10 Case-level SHAP (R4, R5.6). 4 principled cases from identity-disjoint test predictions:
 TP correct-fake (highest-confidence correct), TN correct-real (lowest P_fake correct),
 FN fake->real (fake with lowest P_fake), FP CelebDF real->fake (real with highest P_fake).
Per case: P(fake), true class, top-5 push-to-fake, top-5 push-to-real, feature values, SHAP waterfall.
CAVEAT (in every caption): SHAP explains the classifier's output; it does NOT prove a feature
causally establishes manipulation."""
import os,sys,json,subprocess,datetime
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore"); sys.path.insert(0,"src")
from protocol import make_splits, assert_no_identity_overlap
import roi_config as RC
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb, shap
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
SEED=42; F="features"; OUT="results_clean"; FIG="Major Revision Results/03_figures/exp10_case_level_shap"
os.makedirs(FIG,exist_ok=True)
CAVEAT="SHAP explains the classifier's output; it does not prove a feature causally establishes manipulation."
def base(p): return os.path.basename(str(p))
def commit():
    try: return subprocess.check_output(["git","rev-parse","--short","HEAD"],text=True).strip()
    except: return "nogit"
G1=RC.CANDIDATE_GROUPS["G1_mouth_instability"]; MAN=["deepfakes","face2face","faceswap","neuraltextures"]
def with_g1(name):
    o=pd.read_csv(f"{F}/ffpp_{name}_c23.csv") if name!="real" else pd.read_csv(f"{F}/ffpp_original_c23.csv")
    r=pd.read_csv(f"{F}/roi_{'original' if name=='real' else name}_c23.csv")
    o["_b"]=o.video_path.map(base); r["_b"]=r.video_path.map(base)
    return make_splits(o.merge(r[["_b"]+G1],on="_b",how="inner"))
real=with_g1("real"); MANd={m:with_g1(m) for m in MAN}
FC=sorted([c for c in real.columns if c[:2] in ("s_","t_")]); COLS=FC+G1
from leakfree import split_impute, impute_with, pooled_train_median
def clean(df,cols):  # M1 fix: TRAIN-partition medians only
    return split_impute(df, cols)[0]
real=clean(real,COLS); MANd={m:clean(v,COLS) for m,v in MANd.items()}
ff_med=pooled_train_median([real]+list(MANd.values()),FC)  # for zero-shot CelebDF imputation
def LGBM(): return lgb.LGBMClassifier(n_estimators=200,max_depth=6,learning_rate=0.05,num_leaves=31,min_child_samples=20,class_weight="balanced",random_state=SEED,verbose=-1,n_jobs=-1)

# 53-D model on FF++ (train ids) -> test predictions (for TP/TN/FN)
for _df,_src in [(real,"real")]+[(MANd[m],m) for m in MAN]: _df["source"]=_src
tr=pd.concat([real[real.partition=="train"]]+[MANd[m][MANd[m].partition=="train"] for m in MAN],ignore_index=True)
te=pd.concat([real[real.partition=="test"]]+[MANd[m][MANd[m].partition=="test"] for m in MAN],ignore_index=True)
sc=StandardScaler().fit(tr[COLS].values); clf=LGBM(); clf.fit(sc.transform(tr[COLS].values),tr['label'].values.astype(int))
te=te.reset_index(drop=True); te["p"]=clf.predict_proba(sc.transform(te[COLS].values))[:,1]
expl=shap.TreeExplainer(clf); base_val=expl.expected_value
if isinstance(base_val,(list,np.ndarray)): base_val=float(np.ravel(base_val)[-1])

# 50-D model for CelebDF FP
cd=impute_with(pd.read_csv(f"{F}/celebdf_features.csv"),FC,ff_med)  # M1 fix: FF++ train median
tr50=pd.concat([real[real.partition=="train"]]+[MANd[m][MANd[m].partition=="train"] for m in MAN],ignore_index=True)
sc50=StandardScaler().fit(tr50[FC].values); clf50=LGBM(); clf50.fit(sc50.transform(tr50[FC].values),tr50['label'].values.astype(int))
cd["source"]="celebdf"; cd=cd.reset_index(drop=True); cd["p"]=clf50.predict_proba(sc50.transform(cd[FC].values))[:,1]
expl50=shap.TreeExplainer(clf50); base50=expl50.expected_value
if isinstance(base50,(list,np.ndarray)): base50=float(np.ravel(base50)[-1])

# ---- principled case selection ----
tp=te[(te.label==1)&(te.p>=0.5)].sort_values("p",ascending=False).iloc[0]   # highest-conf correct fake
tn=te[(te.label==0)&(te.p<0.5)].sort_values("p").iloc[0]                     # lowest P_fake correct real
fn=te[(te.label==1)&(te.p<0.5)].sort_values("p").iloc[0]                     # fake with lowest P_fake (worst miss)
fp=cd[(cd.label==0)&(cd.p>=0.5)].sort_values("p",ascending=False).iloc[0]    # CelebDF real with highest P_fake

def shap_case(row, cols, scaler, explainer, bval, tag, model):
    x=scaler.transform(row[cols].values.reshape(1,-1).astype(float))
    sv=explainer.shap_values(x)
    if isinstance(sv,list): sv=sv[1]
    sv=np.ravel(sv)[:len(cols)]
    order=np.argsort(sv)
    push_fake=[(cols[i],round(float(sv[i]),4),round(float(row[cols[i]]),4)) for i in order[::-1] if sv[i]>0][:5]
    push_real=[(cols[i],round(float(sv[i]),4),round(float(row[cols[i]]),4)) for i in order if sv[i]<0][:5]
    # waterfall
    try:
        e=shap.Explanation(values=sv,base_values=bval,data=row[cols].values.astype(float),feature_names=list(cols))
        plt.figure(); shap.plots.waterfall(e,max_display=12,show=False)
        plt.title(f"{tag}  P(fake)={row['p']:.3f}  true={'FAKE' if row['label']==1 else 'REAL'}",fontsize=9)
        plt.figtext(0.5,-0.02,CAVEAT,ha="center",fontsize=6,wrap=True)
        plt.tight_layout(); plt.savefig(f"{FIG}/case_shap_{tag}.png",dpi=130,bbox_inches="tight"); plt.close()
    except Exception as ex:
        print(f"  waterfall {tag} failed: {ex}")
    return dict(case=tag,video=base(row['video_path']),dataset=row.get('source','?'),
        p_fake=round(float(row['p']),4),true_class=("fake" if row['label']==1 else "real"),
        top5_push_to_fake=push_fake,top5_push_to_real=push_real,model=model,caveat=CAVEAT)

cases=[shap_case(tp,COLS,sc,expl,base_val,"tp","53-D FF++"),
       shap_case(tn,COLS,sc,expl,base_val,"tn","53-D FF++"),
       shap_case(fn,COLS,sc,expl,base_val,"fn","53-D FF++"),
       shap_case(fp,FC,sc50,expl50,base50,"fp","50-D CelebDF")]
json.dump(dict(provenance=dict(script="exp10_case_shap.py",git_commit=commit(),seed=SEED,date=datetime.date.today().isoformat(),
    selection="TP=highest-conf correct fake; TN=lowest P_fake correct real; FN=fake w/ lowest P_fake; FP=CelebDF real w/ highest P_fake",
    caveat=CAVEAT),cases=cases),open(f"{OUT}/case_shap.json","w"),indent=2)
print("=== EXP-10 CASE-LEVEL SHAP ===")
for c in cases:
    print(f"\n[{c['case'].upper()}] {c['video']} ({c['dataset']}) P(fake)={c['p_fake']} true={c['true_class']} [{c['model']}]")
    print("  push->FAKE:", [f"{n}={s:+.3f}(v={v})" for n,s,v in c['top5_push_to_fake'][:3]])
    print("  push->REAL:", [f"{n}={s:+.3f}(v={v})" for n,s,v in c['top5_push_to_real'][:3]])
print(f"\nsaved {OUT}/case_shap.json; waterfalls in {FIG} (commit {commit()})")
