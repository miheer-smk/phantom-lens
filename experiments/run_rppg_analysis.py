#!/usr/bin/env python
"""Standalone physiological baselines: POS, CHROM and the shipped POS+CHROM dual,
using ONLY the rPPG descriptors, per manipulation and zero-shot."""
import json, time
from _common import base_parser, out_path, REPO
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from classifiers.prism_pipeline import MANIPULATIONS, feature_dir, load_config, load_split, assign_partition
import lightgbm as lgb
POS=["pos_snr","pos_peak_prom","pos_interreg_corr","pos_harmonic"]
CHR=["chrom_snr","chrom_peak_prom","chrom_interreg_corr","chrom_harmonic"]
METHODS=[("current(POS+CHROM)",POS+CHR),("POS",POS),("CHROM",CHR)]

def main():
    a=base_parser(__doc__).parse_args()
    F=feature_dir(a.features); params=load_config(REPO); id2split=load_split(REPO)
    need=[F/f"rppg_{n}_c23.csv" for n in ["original"]+MANIPULATIONS]
    missing=[p.name for p in need if not p.exists()]
    if missing:
        print("UNAVAILABLE - missing rPPG matrices:", missing); return
    def load(n): return assign_partition(pd.read_csv(F/f"rppg_{n}_c23.csv"), id2split)
    def clean(d,cols):
        d=d.copy()
        for c in cols: d[c]=pd.to_numeric(d[c],errors="coerce").replace([np.inf,-np.inf],np.nan)
        d[cols]=d[cols].fillna(d.loc[d.partition=="train",cols].median()); return d
    res={"generated_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
         "note":"rPPG descriptors only; no other PRISM feature","per_manipulation":{},"zero_shot":{}}
    real=load("original")
    print(f"{'manip':16s} {'method':20s} {'AUC':>8s}  n")
    for m in MANIPULATIONS:
        fk=load(m); res["per_manipulation"][m]={}
        for name,cols in METHODS:
            rr,ff=clean(real,cols),clean(fk,cols)
            tr=pd.concat([rr[rr.partition=="train"],ff[ff.partition=="train"]],ignore_index=True)
            te=pd.concat([rr[rr.partition=="test"], ff[ff.partition=="test"]], ignore_index=True)
            sc=StandardScaler().fit(tr[cols].values)
            c=lgb.LGBMClassifier(**{k:v for k,v in params.items() if k!="verbose"},verbose=-1).fit(
                sc.transform(tr[cols].values),tr.label.values.astype(int))
            au=float(roc_auc_score(te.label.values.astype(int),c.predict_proba(sc.transform(te[cols].values))[:,1]))
            res["per_manipulation"][m][name]=dict(auc=round(au,4),n=int(len(te)))
            print(f"{m:16s} {name:20s} {au:8.4f}  {len(te)}")
    cdp=F/"rppg_celebdf.csv"
    if cdp.exists():
        cd=pd.read_csv(cdp); print()
        for name,cols in METHODS:
            parts=[clean(load(x),cols) for x in ["original"]+MANIPULATIONS]
            tr=pd.concat([d[d.partition=="train"] for d in parts],ignore_index=True)
            med=tr[cols].median(); c2=cd.copy()
            for c_ in cols: c2[c_]=pd.to_numeric(c2[c_],errors="coerce").replace([np.inf,-np.inf],np.nan)
            c2[cols]=c2[cols].fillna(med)
            sc=StandardScaler().fit(tr[cols].values)
            m_=lgb.LGBMClassifier(**{k:v for k,v in params.items() if k!="verbose"},verbose=-1).fit(
                sc.transform(tr[cols].values),tr.label.values.astype(int))
            au=float(roc_auc_score(c2.label.values.astype(int),m_.predict_proba(sc.transform(c2[cols].values))[:,1]))
            res["zero_shot"][name]=dict(auc=round(au,4),n=int(len(c2)))
            print(f"{'CelebDF':16s} {name:20s} {au:8.4f}  {len(c2)}")
    else:
        res["zero_shot"]={"status":"UNAVAILABLE","reason":"rppg_celebdf.csv not in PRISM_FEATURES"}
        print("\n  zero-shot UNAVAILABLE - rppg_celebdf.csv not present")
    o=out_path(a,"rppg_analysis.json"); json.dump(res,open(o,"w"),indent=1); print(f"\n-> {o}")

if __name__ == "__main__": main()
