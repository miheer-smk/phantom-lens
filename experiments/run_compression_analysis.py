#!/usr/bin/env python
"""c23 vs c40 robustness, and cross-compression transfer (train c23 -> test c40)."""
import json, time
from _common import base_parser, out_path, REPO
import pandas as pd
from sklearn.metrics import roc_auc_score
from classifiers.prism_pipeline import (MANIPULATIONS, feature_dir, load_config, feature_columns,
                                        load_split, assign_partition, train_median_impute,
                                        fit_prism, score, metrics)

def main():
    a=base_parser(__doc__).parse_args()
    F=feature_dir(a.features); params=load_config(REPO); id2split=load_split(REPO)
    cols=None; rows=[]
    print(f"{'manip':16s} {'c23':>8s} {'c40':>8s} {'delta':>8s} {'c23->c40':>9s}")
    for m in MANIPULATIONS:
        got={}
        for comp in ("c23","c40"):
            r=assign_partition(pd.read_csv(F/f"ffpp_original_{comp}.csv"),id2split)
            f=assign_partition(pd.read_csv(F/f"ffpp_{m}_{comp}.csv"),id2split)
            if cols is None: cols=feature_columns(r)
            tr_r=pd.concat([r[r.partition=="train"],f[f.partition=="train"]],ignore_index=True)
            te_r=pd.concat([r[r.partition=="test"], f[f.partition=="test"]], ignore_index=True)
            tr,(te,),_=train_median_impute(tr_r,[te_r],cols)
            sc,clf=fit_prism(tr[cols].values.astype(float),tr.label.values.astype(int),params)
            p=score(sc,clf,te[cols].values.astype(float))
            got[comp]=dict(model=(sc,clf),te=te,
                           **metrics(te.label.values.astype(int),p,te.source_video_id.values,a.seed))
        sc,clf=got["c23"]["model"]; te40=got["c40"]["te"]
        xp=score(sc,clf,te40[cols].values.astype(float))
        cross=float(roc_auc_score(te40.label.values.astype(int),xp))
        d=got["c40"]["auc"]-got["c23"]["auc"]   # published convention: c40 - c23, NEGATIVE = degradation
        rows.append(dict(manipulation=m,c23_auc=got["c23"]["auc"],c23_ci=got["c23"]["grouped_ci"],
                         c40_auc=got["c40"]["auc"],c40_ci=got["c40"]["grouped_ci"],
                         delta_auc=round(d,4),cross_c23train_c40test_auc=round(cross,4),
                         c23_mcc=got["c23"]["mcc"],c40_mcc=got["c40"]["mcc"]))
        print(f"{m:16s} {got['c23']['auc']:8.4f} {got['c40']['auc']:8.4f} {d:+8.4f} {cross:9.4f}")
    o=out_path(a,"compression_analysis.json")
    json.dump(dict(generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
        convention="delta_auc = c40_auc - c23_auc, matching the published table; NEGATIVE means c40 degrades",rows=rows),open(o,"w"),indent=1)
    print(f"\n-> {o}")

if __name__ == "__main__": main()
