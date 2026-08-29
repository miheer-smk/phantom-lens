#!/usr/bin/env python
"""R1-C8 SHAP faithfulness: top-5 |SHAP| masking with TRAIN-ONLY medians vs 100 random 5-masks."""
import json, time
from _common import base_parser, out_path, REPO
import numpy as np, pandas as pd
from scipy.stats import wilcoxon, spearmanr
from sklearn.preprocessing import StandardScaler
from classifiers.prism_pipeline import (MANIPULATIONS, feature_dir, load_config, feature_columns,
                                        load_split, assign_partition, train_median_impute, descriptor_groups)
import lightgbm as lgb, shap

def main():
    ap=base_parser(__doc__); ap.add_argument("--n_random",type=int,default=100)
    ap.add_argument("--ablation",help="group_ablation.csv, to correlate against")
    a=ap.parse_args()
    F=feature_dir(a.features); params=load_config(REPO); id2split=load_split(REPO)
    groups=descriptor_groups(REPO)
    raw={"real":pd.read_csv(F/"ffpp_original_c23.csv")}
    for m in MANIPULATIONS: raw[m]=pd.read_csv(F/f"ffpp_{m}_c23.csv")
    cols=feature_columns(raw["real"]); P={k:assign_partition(v,id2split) for k,v in raw.items()}
    part=lambda k,p: P[k][P[k].partition==p]
    res={"generated_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
         "n_random":a.n_random,"seed":a.seed,"primary":{},"secondary":{}}
    ab=pd.read_csv(a.ablation) if a.ablation else None
    print(f"{'manip':16s} {'n':>5s} {'dC top5':>10s} {'dC rand':>10s} {'ratio':>7s} {'p':>11s}")
    for m in MANIPULATIONS:
        tr_r=pd.concat([part("real","train"),part(m,"train")],ignore_index=True)
        te_r=pd.concat([part("real","test"), part(m,"test")], ignore_index=True)
        tr,(te,),_=train_median_impute(tr_r,[te_r],cols)
        Xtr=tr[cols].values.astype(float); ytr=tr.label.values.astype(int)
        sc=StandardScaler().fit(Xtr)
        clf=lgb.LGBMClassifier(**{k:v for k,v in params.items() if k!="verbose"},verbose=-1).fit(sc.transform(Xtr),ytr)
        med=np.median(sc.transform(Xtr),axis=0)          # TRAIN-only medians, in model space
        Xte=sc.transform(te[cols].values.astype(float))
        p0=clf.predict_proba(Xte)[:,1]; yh=(p0>=0.5).astype(int); C0=np.where(yh==1,p0,1-p0)
        sv=shap.TreeExplainer(clf).shap_values(Xte)
        if isinstance(sv,list): sv=sv[1]
        if sv.ndim==3: sv=sv[:,:,1]
        top=np.argsort(-np.abs(sv),axis=1)[:,:5]
        Xm=Xte.copy()
        for i in range(len(Xte)): Xm[i,top[i]]=med[top[i]]
        pm=clf.predict_proba(Xm)[:,1]; dT=C0-np.where(yh==1,pm,1-pm)
        rng=np.random.default_rng(a.seed); acc=np.zeros((len(Xte),a.n_random))
        for b in range(a.n_random):
            Xr=Xte.copy(); idx=np.array([rng.choice(len(cols),5,replace=False) for _ in range(len(Xte))])
            for i in range(len(Xte)): Xr[i,idx[i]]=med[idx[i]]
            pr=clf.predict_proba(Xr)[:,1]; acc[:,b]=C0-np.where(yh==1,pr,1-pr)
        dR=acc.mean(axis=1); st,pv=wilcoxon(dT,dR)
        res["primary"][m]=dict(n=int(len(Xte)),median_dC_top5=round(float(np.median(dT)),5),
            median_dC_random=round(float(np.median(dR)),5),
            ratio=round(float(np.median(dT)/max(np.median(dR),1e-9)),2),
            wilcoxon_p=float(pv),rank_biserial=round(float(2*np.mean(dT>dR)-1),4))
        r=res["primary"][m]
        print(f"{m:16s} {r['n']:5d} {r['median_dC_top5']:10.5f} {r['median_dC_random']:10.5f} "
              f"{r['ratio']:7.1f} {pv:11.3g}")
        if ab is not None:
            gs={g:float(np.abs(sv[:,[cols.index(f) for f in fs if f in cols]]).mean()) for g,fs in groups.items()}
            s=ab[ab.manipulation==m]
            common=[g for g in gs if g in set(s.group)]
            rho,pp=spearmanr([gs[g] for g in common],[float(s[s.group==g].delta_auc.iloc[0]) for g in common])
            res["secondary"][m]=dict(spearman_rho=round(float(rho),4),p_value=float(pp),n_groups=len(common),
                convention="delta_auc = full - loGo, so POSITIVE rho means SHAP is CONCORDANT with ablation utility")
    if res["secondary"]:
        print("\ngroup |SHAP| vs leave-one-group-out deltaAUC:")
        for k,v in res["secondary"].items(): print(f"  {k:16s} rho={v['spearman_rho']:+.4f}  p={v['p_value']:.4g}")
    o=out_path(a,"shap_faithfulness.json"); json.dump(res,open(o,"w"),indent=1); print(f"\n-> {o}")

if __name__ == "__main__": main()
