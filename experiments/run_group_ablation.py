#!/usr/bin/env python
"""Leave-one-group-out ablation over the 20 Table A2 implementation groups.
delta_auc = full_auc - leaveOneGroupOut_auc, so POSITIVE means the group is USEFUL."""
import json, time
from _common import base_parser, out_path, REPO
import pandas as pd
from sklearn.metrics import roc_auc_score
from classifiers.prism_pipeline import (MANIPULATIONS, feature_dir, load_config, feature_columns,
                                        load_split, assign_partition, train_median_impute,
                                        fit_prism, score, descriptor_groups)

def main():
    a = base_parser(__doc__).parse_args()
    F=feature_dir(a.features); params=load_config(REPO); id2split=load_split(REPO)
    groups=descriptor_groups(REPO)
    raw={"real":pd.read_csv(F/"ffpp_original_c23.csv")}
    for m in MANIPULATIONS: raw[m]=pd.read_csv(F/f"ffpp_{m}_c23.csv")
    cols=feature_columns(raw["real"]); P={k:assign_partition(v,id2split) for k,v in raw.items()}
    part=lambda k,p: P[k][P[k].partition==p]
    rows=[]
    print(f"{'manip':16s} {'group':26s} {'full':>7s} {'loGo':>7s} {'delta':>8s}")
    for m in MANIPULATIONS:
        tr_r=pd.concat([part("real","train"),part(m,"train")],ignore_index=True)
        te_r=pd.concat([part("real","test"), part(m,"test")], ignore_index=True)
        tr,(te,),_=train_median_impute(tr_r,[te_r],cols)
        y=te.label.values.astype(int)
        sc,clf=fit_prism(tr[cols].values.astype(float),tr.label.values.astype(int),params)
        full=float(roc_auc_score(y,score(sc,clf,te[cols].values.astype(float))))
        for gname,feats in groups.items():
            keep=[c for c in cols if c not in set(feats)]
            sc2,clf2=fit_prism(tr[keep].values.astype(float),tr.label.values.astype(int),params)
            lo=float(roc_auc_score(y,score(sc2,clf2,te[keep].values.astype(float))))
            rows.append(dict(manipulation=m,group=gname,n_group_features=len(feats),
                             full_auc=round(full,4),loGo_auc=round(lo,4),delta_auc=round(full-lo,4),
                             n_test=int(len(y))))
            print(f"{m:16s} {gname:26s} {full:7.4f} {lo:7.4f} {full-lo:+8.4f}")
    o=out_path(a,"group_ablation.json")
    json.dump(dict(generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
        convention="delta_auc = full - leaveOneGroupOut; POSITIVE = group is USEFUL",
        rows=rows),open(o,"w"),indent=1)
    pd.DataFrame(rows).to_csv(str(o).replace(".json",".csv"),index=False)
    print(f"\n-> {o}")

if __name__ == "__main__": main()
