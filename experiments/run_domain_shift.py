#!/usr/bin/env python
"""R1-C3B class-conditional cross-domain feature shift: SMD, Wasserstein-1, KS, per descriptor."""
import json, time
from _common import base_parser, out_path, REPO
import numpy as np, pandas as pd
from scipy.stats import wasserstein_distance, ks_2samp
from classifiers.prism_pipeline import MANIPULATIONS, feature_dir, feature_columns, descriptor_groups

def stats(a_,b_):
    a_=pd.to_numeric(a_,errors="coerce").dropna().values; b_=pd.to_numeric(b_,errors="coerce").dropna().values
    if len(a_)<5 or len(b_)<5: return None
    sa,sb=a_.std(ddof=1),b_.std(ddof=1); pool=np.sqrt((sa**2+sb**2)/2)
    mu,sd=a_.mean(),(sa if sa>0 else 1.0)
    return dict(smd=float((b_.mean()-a_.mean())/pool) if pool>0 else np.nan,
                w1=float(wasserstein_distance((a_-mu)/sd,(b_-mu)/sd)),
                ks=float(ks_2samp(a_,b_).statistic))

def main():
    a=base_parser(__doc__).parse_args()
    F=feature_dir(a.features)
    ffr=pd.read_csv(F/"ffpp_original_c23.csv"); cols=feature_columns(ffr)
    g2={f:g for g,fs in descriptor_groups(REPO).items() for f in fs}
    S={"real":ffr[cols].apply(pd.to_numeric,errors="coerce"),
       "fake":pd.concat([pd.read_csv(F/f"ffpp_{m}_c23.csv")[cols] for m in MANIPULATIONS],
                        ignore_index=True).apply(pd.to_numeric,errors="coerce")}
    targets={}
    cd=F/"celebdf_features.csv"
    if cd.exists():
        d=pd.read_csv(cd)
        targets["CelebDF"]={"real":d[d.label==0][cols].apply(pd.to_numeric,errors="coerce"),
                            "fake":d[d.label==1][cols].apply(pd.to_numeric,errors="coerce")}
    rows=[]
    for t,byc in targets.items():
        for cls,tgt in byc.items():
            for c in cols:
                st=stats(S[cls][c],tgt[c])
                if st: rows.append(dict(target=t,klass=cls,descriptor=c,group=g2.get(c,"other"),**st))
    d=pd.DataFrame(rows)
    if not len(d): print("no target matrices found in PRISM_FEATURES"); return
    o=out_path(a,"domain_shift.json")
    d.to_csv(str(o).replace(".json",".csv"),index=False)
    print("=== mean |SMD| by group, per target and class ===")
    piv=d.pivot_table(index=["target","group"],columns="klass",values="smd",aggfunc=lambda x:np.mean(np.abs(x)))
    print(piv.round(3).to_string())
    print("\n=== does the REAL class shift more than the FAKE class? ===")
    summ={}
    for t in targets:
        r=d[(d.target==t)&(d.klass=="real")].smd.abs().mean(); f=d[(d.target==t)&(d.klass=="fake")].smd.abs().mean()
        summ[t]=dict(real=round(float(r),4),fake=round(float(f),4),ratio=round(float(r/f),4))
        print(f"  {t:14s} real={r:.3f} fake={f:.3f} ratio={r/f:.3f}  "
              f"{'REAL shifts more' if r>f else 'fake shifts more'}")
    json.dump(dict(generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
        n_rows=len(d),real_vs_fake=summ,
        top_shifts=d.reindex(d.smd.abs().sort_values(ascending=False).index).head(15).to_dict("records")),
        open(o,"w"),indent=1)
    print(f"\n-> {o}")

if __name__ == "__main__": main()
