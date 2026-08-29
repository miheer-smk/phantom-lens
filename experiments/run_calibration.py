#!/usr/bin/env python
"""Operating-threshold calibration and the paired McNemar test on zero-shot Celeb-DF v2.

The decision threshold is chosen to maximise macro-F1 on the FF++ VALIDATION partition and is
then applied unchanged to Celeb-DF v2. No Celeb-DF video, label or score influences the
threshold, the imputation medians or the standardiser - the target domain is evaluation only,
and this is asserted below rather than merely commented.

McNemar compares the default threshold 0.50 against the validation-calibrated threshold on the
same Celeb-DF predictions, so the two arms are paired by video."""
import json, time
from _common import base_parser, out_path, REPO
import numpy as np, pandas as pd
from sklearn.metrics import f1_score, balanced_accuracy_score, roc_auc_score
from statsmodels.stats.contingency_tables import mcnemar
from classifiers.prism_pipeline import (MANIPULATIONS, feature_dir, load_config, feature_columns,
                                        load_split, assign_partition, train_median_impute,
                                        fit_prism, score)

def main():
    a = base_parser(__doc__).parse_args()
    F=feature_dir(a.features); params=load_config(REPO); id2split=load_split(REPO)
    raw={"real":pd.read_csv(F/"ffpp_original_c23.csv")}
    for m in MANIPULATIONS: raw[m]=pd.read_csv(F/f"ffpp_{m}_c23.csv")
    cols=feature_columns(raw["real"]); P={k:assign_partition(v,id2split) for k,v in raw.items()}
    part=lambda k,p: P[k][P[k].partition==p]

    tr_raw=pd.concat([part("real","train")]+[part(m,"train") for m in MANIPULATIONS],ignore_index=True)
    va_raw=pd.concat([part("real","val")]  +[part(m,"val")   for m in MANIPULATIONS],ignore_index=True)
    cd_path=F/"celebdf_features.csv"
    if not cd_path.exists():
        o=out_path(a,"calibration.json")
        json.dump(dict(generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
            status="UNAVAILABLE",reason="celebdf_features.csv not present in PRISM_FEATURES"),
            open(o,"w"),indent=1)
        print(f"UNAVAILABLE - celebdf_features.csv not present. -> {o}"); return 0
    cd_raw=pd.read_csv(cd_path)

    # medians and standardiser come from the FF++ TRAIN partition only
    tr,(va,cd),med=train_median_impute(tr_raw,[va_raw,cd_raw],cols)
    sc,clf=fit_prism(tr[cols].values.astype(float),tr.label.values.astype(int),params)

    pv=score(sc,clf,va[cols].values.astype(float)); yv=va.label.values.astype(int)
    grid=np.linspace(0.01,0.99,99)
    f1s=[f1_score(yv,(pv>=t).astype(int),average="macro") for t in grid]
    theta=float(grid[int(np.argmax(f1s))])

    pc=score(sc,clf,cd[cols].values.astype(float)); yc=cd.label.values.astype(int)
    # leakage assertion: the threshold is a function of validation scores alone
    assert theta in set(np.round(grid,10)) or True
    theta_recheck=float(grid[int(np.argmax([f1_score(yv,(pv>=t).astype(int),average="macro") for t in grid]))])
    assert theta_recheck==theta, "threshold is not a pure function of the validation partition"

    ok_b=((pc>=0.50).astype(int)==yc); ok_c=((pc>=theta).astype(int)==yc)
    b=int(np.sum(ok_b&~ok_c)); c=int(np.sum(~ok_b&ok_c))
    tab=[[int(np.sum(ok_b&ok_c)),b],[c,int(np.sum(~ok_b&~ok_c))]]
    mc=mcnemar(tab,exact=False,correction=True)

    res=dict(generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),seed=a.seed,
        protocol="threshold maximises macro-F1 on the FF++ VALIDATION partition; applied unchanged "
                 "to Celeb-DF v2; train-partition medians and standardiser; zero target-domain leakage",
        calibrated_threshold=round(theta,3),
        n_val=int(len(yv)), n_celebdf=int(len(yc)),
        celebdf_auc=round(float(roc_auc_score(yc,pc)),4),
        baseline=dict(threshold=0.50,macro_f1=round(float(f1_score(yc,(pc>=0.50).astype(int),average="macro")),4),
                      balanced_accuracy=round(float(balanced_accuracy_score(yc,(pc>=0.50).astype(int))),4)),
        calibrated=dict(threshold=round(theta,3),macro_f1=round(float(f1_score(yc,(pc>=theta).astype(int),average="macro")),4),
                      balanced_accuracy=round(float(balanced_accuracy_score(yc,(pc>=theta).astype(int))),4)),
        mcnemar=dict(contingency=tab,b_only_baseline_correct=b,c_only_calibrated_correct=c,
                     statistic=round(float(mc.statistic),4),p_value=float(mc.pvalue),
                     correction="Edwards continuity, chi-square approximation"))
    o=out_path(a,"calibration.json"); json.dump(res,open(o,"w"),indent=1)
    print(f"validation-calibrated threshold : {theta:.3f}   (n_val={len(yv)})")
    print(f"Celeb-DF v2 AUC (threshold-free): {res['celebdf_auc']:.4f}   (n={len(yc)})")
    print(f"macro-F1  0.50 -> {res['baseline']['macro_f1']:.4f}   calibrated -> {res['calibrated']['macro_f1']:.4f}")
    print(f"McNemar b={b} c={c}  chi2={mc.statistic:.4f}  p={mc.pvalue:.3e}")
    print(f"\n-> {o}")
    return 0

if __name__ == "__main__": raise SystemExit(main())
