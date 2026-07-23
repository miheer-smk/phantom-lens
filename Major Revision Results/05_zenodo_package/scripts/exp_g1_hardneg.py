#!/usr/bin/env python3
"""G1 — Hard-negative analysis on the CLEAN, identity-disjoint Deepfakes TEST split.
Retires the leaky old exp5 (which trained AND tested on the same full ffpp_fake.csv -> 13/957).
Recipe (identity-disjoint, seed 42, locked LightGBM, M1 train-only imputer):
  Train : real_train + Deepfakes/Face2Face/FaceSwap/NeuralTextures TRAIN identities (multi-manip, as old exp5)
  Test  : Deepfakes TEST identities (fakes) + real TEST (for TN/context)
Hard negatives = Deepfakes test fakes with P(fake) < 0.5 (predicted real). Reports count, rate,
confidence distribution, and the K hardest cases with their most fake-atypical features.
"""
import os, sys, json, subprocess, datetime
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore"); sys.path.insert(0, "src")
from protocol import make_splits, assert_no_identity_overlap
from leakfree import split_impute, pooled_train_median
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

SEED=42; np.random.seed(SEED); F="features"; OUT="results_clean"
def basen(p): return os.path.basename(str(p))
def commit():
    try: return subprocess.check_output(["git","rev-parse","--short","HEAD"],text=True).strip()
    except: return "nogit"
FILES={"real":"ffpp_original_c23.csv","Deepfakes":"ffpp_deepfakes_c23.csv",
       "Face2Face":"ffpp_face2face_c23.csv","FaceSwap":"ffpp_faceswap_c23.csv",
       "NeuralTextures":"ffpp_neuraltextures_c23.csv"}
raw={k:pd.read_csv(f"{F}/{v}") for k,v in FILES.items()}
FC=sorted([c for c in raw["real"].columns if c[:2] in ("s_","t_")])
P={k:split_impute(v,FC)[0] for k,v in raw.items()}   # M1 train-only imputer
MAN=["Deepfakes","Face2Face","FaceSwap","NeuralTextures"]

df_test=P["Deepfakes"][P["Deepfakes"].partition=="test"]      # fakes
real_test=P["real"][P["real"].partition=="test"]             # reals (context)

def LGBM(): return lgb.LGBMClassifier(n_estimators=200,max_depth=6,learning_rate=0.05,num_leaves=31,
    min_child_samples=20,class_weight="balanced",random_state=SEED,verbose=-1,n_jobs=-1)

def analyse(recipe, train_frames):
    assert_no_identity_overlap([(f,"train") for f in train_frames]+[(df_test,"test"),(real_test,"test")])
    tr=pd.concat(train_frames,ignore_index=True)
    sc=StandardScaler().fit(tr[FC].values); clf=LGBM(); clf.fit(sc.transform(tr[FC].values),tr['label'].values.astype(int))
    pf=clf.predict_proba(sc.transform(df_test[FC].values))[:,1]      # P(fake) on Deepfakes TEST fakes
    n_fakes=len(pf); n_fn=int((pf<0.5).sum())
    yte=np.r_[np.zeros(len(real_test)),np.ones(len(df_test))]
    pte=clf.predict_proba(sc.transform(pd.concat([real_test,df_test])[FC].values))[:,1]
    auc_test=round(roc_auc_score(yte,pte),4)
    fake_mean=sc.transform(df_test[FC].values).mean(0); hard=[]
    for i in np.argsort(pf)[:15]:
        dev=sc.transform(df_test[FC].values[i:i+1])[0]-fake_mean
        top=np.argsort(-np.abs(dev))[:4]
        hard.append(dict(video=basen(df_test.iloc[i]["video_path"]),p_fake=round(float(pf[i]),4),
            atypical_features={FC[j]:round(float(dev[j]),2) for j in top}))
    return pf, dict(recipe=recipe, n_train=int(len(tr)), n_deepfakes_test_fakes=n_fakes,
        n_real_test=int(len(real_test)), test_auc_real_vs_deepfakes=auc_test,
        n_false_negatives=n_fn, false_negative_rate=round(n_fn/n_fakes,4),
        pfake_min=round(float(pf.min()),4), pfake_mean=round(float(pf.mean()),4),
        pfake_p05=round(float(np.percentile(pf,5)),4), pfake_median=round(float(np.median(pf)),4),
        hardest_cases=hard)

indist_frames=[P["real"][P["real"].partition=="train"], P["Deepfakes"][P["Deepfakes"].partition=="train"]]
multi_frames=[P["real"][P["real"].partition=="train"]]+[P[m][P[m].partition=="train"] for m in MAN]
print("identity-disjoint assertion (both recipes)...", flush=True)
pf_ind, r_ind = analyse("in_distribution (real+Deepfakes TRAIN) — matches locked 0.9706 detector", indist_frames)
pf_mm , r_mm  = analyse("multi_manip (real+all4 TRAIN) — matches old exp5 recipe", multi_frames)

res=dict(provenance=dict(script="Major Revision Results/00_logs/exp_g1_hardneg.py",git_commit=commit(),
    seed=SEED,date=datetime.date.today().isoformat(),
    protocol="identity-disjoint Deepfakes TEST; M1 train-only imputer; LGBM locked; two training recipes"),
    hard_negative_definition="Deepfakes TEST fake with P(fake)<0.5 (predicted real)",
    in_distribution=r_ind, multi_manip=r_mm,
    retires="results/exp5 (leaky: trained+tested on same ffpp_fake.csv -> 13/957)")
json.dump(res,open(f"{OUT}/hardneg_deepfakes.json","w"),indent=1)
# primary reporting = in-distribution model (the reported Deepfakes detector)
pf=pf_ind; n_fakes=r_ind["n_deepfakes_test_fakes"]; n_fn=r_ind["n_false_negatives"]; auc_test=r_ind["test_auc_real_vs_deepfakes"]; hard=r_ind["hardest_cases"]

# figure: P(fake) distribution over Deepfakes test with FN region shaded
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig,ax=plt.subplots(figsize=(6,3.4),dpi=300)
ax.hist(pf,bins=40,color="#2166AC",edgecolor="white",linewidth=0.3)
ax.axvline(0.5,color="#B2182B",lw=1.5,ls="--",label="decision θ=0.5")
ax.axvspan(0,0.5,color="#B2182B",alpha=0.07)
ax.set_xlabel("P(fake) — Deepfakes TEST videos (identity-disjoint)"); ax.set_ylabel("count")
ax.set_title(f"Deepfakes hard negatives: {n_fn}/{n_fakes} FN ({100*n_fn/n_fakes:.1f}%), test AUC={auc_test:.3f}",fontsize=9)
ax.legend(fontsize=8,frameon=False); fig.tight_layout()
figdir="Major Revision Results/03_figures/expG1_hard_negatives"; os.makedirs(figdir,exist_ok=True)
fig.savefig(f"{figdir}/deepfakes_hardneg_pfake_dist.png"); plt.close(fig)

print("="*66); print("G1 — CLEAN HARD-NEGATIVE ANALYSIS (Deepfakes, identity-disjoint TEST)"); print("="*66)
print("  [old leaky exp5: 13/957 = 1.36% FN, trained+tested on same ffpp_fake.csv]")
for r in (r_ind, r_mm):
    print(f"  {r['recipe']}")
    print(f"     n_train={r['n_train']}  DF_test_fakes={r['n_deepfakes_test_fakes']}  test AUC(real vs DF)={r['test_auc_real_vs_deepfakes']}")
    print(f"     FALSE NEGATIVES (P(fake)<0.5): {r['n_false_negatives']}/{r['n_deepfakes_test_fakes']} = {100*r['false_negative_rate']:.2f}%"
          f"  | P(fake) min={r['pfake_min']} median={r['pfake_median']} mean={r['pfake_mean']}")
print(f"  hardest case (in-dist): {hard[0]['video']} P(fake)={hard[0]['p_fake']}")
print(f"saved {OUT}/hardneg_deepfakes.json + figure (commit {commit()})")
