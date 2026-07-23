#!/usr/bin/env python3
"""EXP-11 statistical wrap-up (R3.13). Paired tests from ACTUAL prediction scores (never fabricated).
 DeLong: full-50 vs top-3, full-50 vs top-10 (per manip), c23 vs c40 (per manip) — Holm-corrected.
 McNemar: baseline θ=0.50 vs val-calibrated threshold (CelebDF paired predictions).
 Wilcoxon signed-rank: full-50 vs top-3 across 10 identity-grouped CV folds.
 Bootstrap 95% CIs on AUC differences. Identity-disjoint, seed 42."""
import os,sys,json,subprocess,datetime
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore"); sys.path.insert(0,"src")
from protocol import make_splits, clip_identities
from delong import delong_roc_test, holm
from scipy.stats import wilcoxon
from statsmodels.stats.contingency_tables import mcnemar
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, balanced_accuracy_score
import lightgbm as lgb
SEED=42; F="features"; OUT="results_clean"
MAN=["deepfakes","face2face","faceswap","neuraltextures"]
def base(p): return os.path.basename(str(p))
def commit():
    try: return subprocess.check_output(["git","rev-parse","--short","HEAD"],text=True).strip()
    except: return "nogit"
def load(name,comp="c23"):
    return make_splits(pd.read_csv(f"{F}/ffpp_{'original' if name=='real' else name}_{comp}.csv"))
from leakfree import split_impute, impute_with, pooled_train_median
def clean(df,FC):  # M1 fix: TRAIN-partition medians only
    return split_impute(df, FC)[0]
real=load("real"); MANd={m:load(m) for m in MAN}
FC=sorted([c for c in real.columns if c[:2] in ("s_","t_")])
real=clean(real,FC); MANd={m:clean(v,FC) for m,v in MANd.items()}
ff_med=pooled_train_median([real]+list(MANd.values()),FC)  # for zero-shot CelebDF imputation
def LGBM(): return lgb.LGBMClassifier(n_estimators=200,max_depth=6,learning_rate=0.05,num_leaves=31,min_child_samples=20,class_weight="balanced",random_state=SEED,verbose=-1,n_jobs=-1)
def scores(cols,tr,te):
    sc=StandardScaler().fit(tr[cols].values); m=LGBM(); m.fit(sc.transform(tr[cols].values),tr['label'].values.astype(int))
    return m.predict_proba(sc.transform(te[cols].values))[:,1]
def boot_diff_ci(y,pa,pb,n=2000,s=SEED):
    rng=np.random.RandomState(s); d=[]
    for _ in range(n):
        i=rng.randint(0,len(y),len(y))
        if len(np.unique(y[i]))<2: continue
        d.append(roc_auc_score(y[i],pa[i])-roc_auc_score(y[i],pb[i]))
    return round(float(np.percentile(d,2.5)),4),round(float(np.percentile(d,97.5)),4)

results={"delong":[],"mcnemar":[],"wilcoxon":[]}
# rank features on train+val importance (no test leak)
trv_all=pd.concat([real[real.partition.isin(["train","val"])]]+[MANd[m][MANd[m].partition.isin(["train","val"])] for m in MAN],ignore_index=True)
rk=LGBM().fit(StandardScaler().fit_transform(trv_all[FC].values),trv_all['label'].values.astype(int))
order=[FC[i] for i in np.argsort(rk.feature_importances_)[::-1]]; top3=order[:3]; top10=order[:10]

# ---- (A) DeLong full-50 vs top-3 / top-10, per manip ----
raw=[]; tmp=[]
for m,md in MANd.items():
    tr=pd.concat([real[real.partition.isin(["train","val"])],md[md.partition.isin(["train","val"])]],ignore_index=True)
    te=pd.concat([real[real.partition=="test"],md[md.partition=="test"]],ignore_index=True); y=te['label'].values.astype(int)
    pf=scores(FC,tr,te)
    for label,cols in [("full50_vs_top3",top3),("full50_vs_top10",top10)]:
        pk=scores(cols,tr,te); af,ak,z,p=delong_roc_test(y,pf,pk)
        lo,hi=boot_diff_ci(y,pf,pk)
        tmp.append(dict(comparison=f"{label}[{m}]",test="DeLong",auc_full=round(af,4),auc_sub=round(ak,4),
            auc_diff=round(af-ak,4),z=round(z,3),p_value=p,ci95=[lo,hi])); raw.append(p)
# ---- c23 vs c40 per manip (paired on matched test videos) ----
for m in MAN:
    md23=MANd[m]; md40=clean(load(m,"c40"),FC); r40=clean(load("real","c40"),FC)
    tr23=pd.concat([real[real.partition.isin(["train","val"])],md23[md23.partition.isin(["train","val"])]],ignore_index=True)
    tr40=pd.concat([r40[r40.partition.isin(["train","val"])],md40[md40.partition.isin(["train","val"])]],ignore_index=True)
    te23=pd.concat([real[real.partition=="test"],md23[md23.partition=="test"]],ignore_index=True)
    te40=pd.concat([r40[r40.partition=="test"],md40[md40.partition=="test"]],ignore_index=True)
    te23["_b"]=te23.video_path.map(base); te40["_b"]=te40.video_path.map(base)
    mg=te23[["_b","label"]].merge(te40[["_b"]],on="_b")  # matched videos
    te23m=te23[te23._b.isin(mg._b)].sort_values("_b"); te40m=te40[te40._b.isin(mg._b)].sort_values("_b")
    y=te23m['label'].values.astype(int)
    p23=scores(FC,tr23,te23m); p40=scores(FC,tr40,te40m)
    a23,a40,z,p=delong_roc_test(y,p23,p40); lo,hi=boot_diff_ci(y,p23,p40)
    tmp.append(dict(comparison=f"c23_vs_c40[{m}]",test="DeLong",auc_full=round(a23,4),auc_sub=round(a40,4),
        auc_diff=round(a23-a40,4),z=round(z,3),p_value=p,ci95=[lo,hi])); raw.append(p)
hp=holm(raw)
for r,h in zip(tmp,hp): r["p_holm"]=round(h,4); results["delong"].append(r)

# ---- (B) McNemar: CelebDF baseline θ=0.50 vs val-calibrated threshold ----
cd=impute_with(pd.read_csv(f"{F}/celebdf_features.csv"),FC,ff_med)  # M1 fix: FF++ train median
trv=pd.concat([real[real.partition.isin(["train"])]]+[MANd[m][MANd[m].partition=="train"] for m in MAN],ignore_index=True)
va=pd.concat([real[real.partition=="val"]]+[MANd[m][MANd[m].partition=="val"] for m in MAN],ignore_index=True)
sc=StandardScaler().fit(trv[FC].values); clf=LGBM().fit(sc.transform(trv[FC].values),trv['label'].values.astype(int))
pv=clf.predict_proba(sc.transform(va[FC].values))[:,1]; yv=va['label'].values.astype(int)
grid=np.linspace(0.01,0.99,99); f1th=grid[np.argmax([f1_score(yv,(pv>=t).astype(int),average='macro') for t in grid])]
pc=clf.predict_proba(sc.transform(cd[FC].values))[:,1]; yc=cd['label'].values.astype(int)
pred_base=(pc>=0.5).astype(int); pred_cal=(pc>=f1th).astype(int)
correct_b=(pred_base==yc); correct_c=(pred_cal==yc)
b=int(np.sum(correct_b&~correct_c)); c=int(np.sum(~correct_b&correct_c))
mc=mcnemar([[int(np.sum(correct_b&correct_c)),b],[c,int(np.sum(~correct_b&~correct_c))]],exact=False,correction=True)
results["mcnemar"].append(dict(comparison="CelebDF baseline(0.50) vs val-calibrated(%.3f)"%f1th,test="McNemar",
    statistic=round(float(mc.statistic),4),p_value=float(mc.pvalue),b_only_base_correct=b,c_only_cal_correct=c))

# ---- (C) Wilcoxon: full-50 vs top-3 across 10 identity-grouped folds ----
allc=pd.concat([real[real.partition!="test"]]+[MANd[m][MANd[m].partition!="test"] for m in MAN],ignore_index=True)
groups=allc['video_path'].map(lambda p: sorted(clip_identities(p))[0]); gkf=GroupKFold(10)
af_full=[]; af_top3=[]
for tri,vai in gkf.split(allc[FC].values,allc['label'].values,groups):
    trf=allc.iloc[tri]; vaf=allc.iloc[vai]; yv2=vaf['label'].values.astype(int)
    af_full.append(roc_auc_score(yv2,scores(FC,trf,vaf))); af_top3.append(roc_auc_score(yv2,scores(top3,trf,vaf)))
w,pw=wilcoxon(af_full,af_top3)
results["wilcoxon"].append(dict(comparison="full50_vs_top3 across 10 folds",test="Wilcoxon signed-rank",
    statistic=round(float(w),4),p_value=float(pw),mean_full=round(float(np.mean(af_full)),4),mean_top3=round(float(np.mean(af_top3)),4)))

json.dump(dict(provenance=dict(script="exp11_stats.py",git_commit=commit(),seed=SEED,date=datetime.date.today().isoformat(),
    note="all p-values computed from actual prediction scores; DeLong Holm-corrected across family"),
    top3=top3,top10=top10,**results),open(f"{OUT}/statistical_tests.json","w"),indent=2)
rows=results["delong"]+results["mcnemar"]+results["wilcoxon"]
pd.DataFrame(rows).to_csv(f"{OUT}/statistical_tests.csv",index=False)
print("=== EXP-11 STATISTICAL TESTS ===\n-- DeLong (Holm-corrected) --")
for r in results["delong"]: print(f"  {r['comparison']:26s} Δ={r['auc_diff']:+.4f} z={r['z']:+.2f} p={r['p_value']:.2e} p_holm={r['p_holm']:.3f}")
print("-- McNemar --")
for r in results["mcnemar"]: print(f"  {r['comparison']}: stat={r['statistic']} p={r['p_value']:.3e} (b={r['b_only_base_correct']}, c={r['c_only_cal_correct']})")
print("-- Wilcoxon --")
for r in results["wilcoxon"]: print(f"  {r['comparison']}: stat={r['statistic']} p={r['p_value']:.3e} (full {r['mean_full']} vs top3 {r['mean_top3']})")
print(f"\nsaved {OUT}/statistical_tests.csv, statistical_tests.json (commit {commit()})")
