#!/usr/bin/env python
"""R1-C6: Logistic Regression vs Random Forest vs LightGBM on identical matrices and split.
LR/RF select hyperparameters on the VALIDATION partition only; LightGBM is frozen with no search."""
import json, time, yaml
from itertools import product
from _common import base_parser, out_path, REPO
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from classifiers.prism_pipeline import (MANIPULATIONS, feature_dir, load_config, feature_columns,
                                        load_split, assign_partition, train_median_impute, metrics)
import lightgbm as lgb

def main():
    a = base_parser(__doc__).parse_args()
    F = feature_dir(a.features); params = load_config(REPO); id2split = load_split(REPO)
    lrc = yaml.safe_load(open(REPO/"configs"/"logistic_regression.yaml"))
    rfc = yaml.safe_load(open(REPO/"configs"/"random_forest.yaml"))
    raw = {"real": pd.read_csv(F/"ffpp_original_c23.csv")}
    for m in MANIPULATIONS: raw[m] = pd.read_csv(F/f"ffpp_{m}_c23.csv")
    cols = feature_columns(raw["real"])
    P = {k: assign_partition(v, id2split) for k, v in raw.items()}
    part = lambda k,p: P[k][P[k].partition==p]
    res = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "protocol": "train-only fit; LR/RF tuned on validation only; LightGBM frozen, 0 configs",
           "search_budget": {"LR": len(lrc["search"]["C"]),
                             "RF": len(rfc["search"]["n_estimators"])*len(rfc["search"]["max_depth"])*len(rfc["search"]["min_samples_leaf"]),
                             "LightGBM": 0},
           "in_distribution": {}}
    print(f"{'manip':16s} {'clf':9s} {'AUC':>7s} {'grouped CI':>18s} {'mF1':>7s} {'MCC':>7s}")
    for m in MANIPULATIONS:
        tr_r = pd.concat([part("real","train"), part(m,"train")], ignore_index=True)
        va_r = pd.concat([part("real","val"),   part(m,"val")],   ignore_index=True)
        te_r = pd.concat([part("real","test"),  part(m,"test")],  ignore_index=True)
        tr,(va,te),_ = train_median_impute(tr_r, [va_r, te_r], cols)
        sc = StandardScaler().fit(tr[cols].values.astype(float))
        X = {k: sc.transform(d[cols].values.astype(float)) for k,d in (("tr",tr),("va",va),("te",te))}
        y = {k: d.label.values.astype(int) for k,d in (("tr",tr),("va",va),("te",te))}
        g = te.source_video_id.values
        res["in_distribution"][m] = {}
        best=None
        for C in lrc["search"]["C"]:
            mdl=LogisticRegression(C=C,**{k:v for k,v in lrc["fixed"].items()}).fit(X["tr"],y["tr"])
            s=roc_auc_score(y["va"],mdl.predict_proba(X["va"])[:,1])
            if best is None or s>best[0]: best=(s,C,mdl)
        res["in_distribution"][m]["LR"]={**metrics(y["te"],best[2].predict_proba(X["te"])[:,1],g,a.seed),
                                          "selected":{"C":best[1],"val_auc":round(best[0],4)}}
        best=None
        for ne,md,ml in product(rfc["search"]["n_estimators"],rfc["search"]["max_depth"],rfc["search"]["min_samples_leaf"]):
            mdl=RandomForestClassifier(n_estimators=ne,max_depth=md,min_samples_leaf=ml,
                 **{k:v for k,v in rfc["fixed"].items()},n_jobs=-1).fit(X["tr"],y["tr"])
            s=roc_auc_score(y["va"],mdl.predict_proba(X["va"])[:,1])
            if best is None or s>best[0]: best=(s,(ne,md,ml),mdl)
        res["in_distribution"][m]["RF"]={**metrics(y["te"],best[2].predict_proba(X["te"])[:,1],g,a.seed),
            "selected":dict(zip(("n_estimators","max_depth","min_samples_leaf"),best[1]),val_auc=round(best[0],4))}
        gb=lgb.LGBMClassifier(**{k:v for k,v in params.items() if k!="verbose"},verbose=-1).fit(X["tr"],y["tr"])
        res["in_distribution"][m]["LightGBM"]={**metrics(y["te"],gb.predict_proba(X["te"])[:,1],g,a.seed),
                                               "selected":{"frozen":True,"search_performed":False}}
        for c in ("LR","RF","LightGBM"):
            r=res["in_distribution"][m][c]
            print(f"{m:16s} {c:9s} {r['auc']:7.4f} {str(r['grouped_ci']):>18s} {r['macro_f1']:7.4f} {r['mcc']:7.4f}")
    o=out_path(a,"classifier_comparison.json"); json.dump(res,open(o,"w"),indent=1); print(f"\n-> {o}")

if __name__ == "__main__": main()
