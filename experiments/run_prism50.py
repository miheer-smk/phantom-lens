#!/usr/bin/env python
"""Reproduce the primary PRISM-50 results from the released feature matrices.

    export PRISM_FEATURES=/path/to/features
    python experiments/run_prism50.py

Reproduces Table 7 (in-distribution per manipulation), Table 8 (leave-one-manipulation-out)
and Table 13 (zero-shot Celeb-DF v2), and checks each against the published value.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from classifiers.prism_pipeline import (MANIPULATIONS, feature_dir, load_config, feature_columns,
                                        load_split, assign_partition, assert_no_identity_overlap,
                                        train_median_impute, fit_prism, score)
from sklearn.metrics import roc_auc_score

PUBLISHED = {"table7": {"deepfakes": 0.9706, "face2face": 0.8096, "faceswap": 0.9631, "neuraltextures": 0.7867},
             "table8": {"deepfakes": 0.7039, "face2face": 0.6904, "faceswap": 0.5978, "neuraltextures": 0.5221},
             "table13": 0.6322}
TOL = 5e-5   # print precision of a 4-dp published value

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features"); ap.add_argument("--out", default=str(REPO / "results" / "reproduction.json"))
    a = ap.parse_args()
    F = feature_dir(a.features); params = load_config(REPO); id2split = load_split(REPO)
    raw = {"real": pd.read_csv(F / "ffpp_original_c23.csv")}
    for m in MANIPULATIONS:
        raw[m] = pd.read_csv(F / f"ffpp_{m}_c23.csv")
    cols = feature_columns(raw["real"])
    assert len(cols) == 50, f"expected 50 descriptors, found {len(cols)}"
    P = {k: assign_partition(v, id2split) for k, v in raw.items()}
    part = lambda k, p: P[k][P[k].partition == p]
    checks = []
    def chk(name, obs, exp):
        ok = exp is None or abs(obs - exp) < TOL
        checks.append(dict(check=name, observed=round(obs, 6), published=exp, reproduces=bool(ok)))
        print(f"  {name:34s} {obs:.4f}   published {exp}   {'OK' if ok else '*** DIFFERS ***'}")

    print("Table 7 - in-distribution, identity-disjoint, fit on TRAIN only")
    for m in MANIPULATIONS:
        tr_raw = pd.concat([part("real", "train"), part(m, "train")], ignore_index=True)
        te_raw = pd.concat([part("real", "test"), part(m, "test")], ignore_index=True)
        assert_no_identity_overlap([(tr_raw, "train"), (te_raw, "test")])
        tr, (te,), _ = train_median_impute(tr_raw, [te_raw], cols)
        sc, clf = fit_prism(tr[cols].values.astype(float), tr.label.values.astype(int), params)
        p = score(sc, clf, te[cols].values.astype(float))
        chk(f"Table 7 {m}", roc_auc_score(te.label.values.astype(int), p), PUBLISHED["table7"][m])

    print("Table 8 - leave-one-manipulation-out")
    for held in MANIPULATIONS:
        others = [x for x in MANIPULATIONS if x != held]
        tr_raw = pd.concat([part("real", "train")] + [part(o, "train") for o in others], ignore_index=True)
        te_raw = pd.concat([part("real", "test"), part(held, "test")], ignore_index=True)
        assert_no_identity_overlap([(tr_raw, "train"), (te_raw, "test")])
        tr, (te,), _ = train_median_impute(tr_raw, [te_raw], cols)
        sc, clf = fit_prism(tr[cols].values.astype(float), tr.label.values.astype(int), params)
        p = score(sc, clf, te[cols].values.astype(float))
        chk(f"Table 8 LOMO {held}", roc_auc_score(te.label.values.astype(int), p), PUBLISHED["table8"][held])

    cd_path = F / "celebdf_features.csv"
    if cd_path.exists():
        print("Table 13 - zero-shot Celeb-DF v2")
        tr_raw = pd.concat([part("real", "train")] + [part(m, "train") for m in MANIPULATIONS], ignore_index=True)
        cd_raw = pd.read_csv(cd_path)
        tr, (cd,), _ = train_median_impute(tr_raw, [cd_raw], cols)   # FF++ TRAIN median only
        sc, clf = fit_prism(tr[cols].values.astype(float), tr.label.values.astype(int), params)
        p = score(sc, clf, cd[cols].values.astype(float))
        chk("Table 13 CelebDF zero-shot", roc_auc_score(cd.label.values.astype(int), p), PUBLISHED["table13"])
    else:
        print("  Table 13 skipped - celebdf_features.csv not present")

    n_bad = sum(1 for c in checks if not c["reproduces"])
    json.dump(dict(generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   tolerance=TOL, config=params, checks=checks,
                   all_reproduce=bool(n_bad == 0)), open(a.out, "w"), indent=1)
    print(f"\n{len(checks) - n_bad}/{len(checks)} reproduce within {TOL}. -> {a.out}")
    return 1 if n_bad else 0

if __name__ == "__main__":
    sys.exit(main())
