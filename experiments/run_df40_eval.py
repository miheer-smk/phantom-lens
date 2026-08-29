#!/usr/bin/env python
"""R1-C4 DF40 zero-shot evaluation with the frozen FF++-trained model.

Runs the substrate + missingness GATE first and refuses to score any method that fails it,
mirroring the pre-registered protocol: treatment is decided on substrate evidence, not on AUC.
"""
import json, sys, time
from _common import base_parser, out_path, REPO
import numpy as np, pandas as pd
from classifiers.prism_pipeline import (MANIPULATIONS, feature_dir, load_config, feature_columns,
                                        load_split, assign_partition, train_median_impute,
                                        fit_prism, score, metrics)
from evaluation.substrate import check_substrate, summarise

def main():
    ap=base_parser(__doc__)
    ap.add_argument("--df40", help="JSONL of extracted DF40 PRISM-50 features")
    a=ap.parse_args()
    F=feature_dir(a.features); params=load_config(REPO); id2split=load_split(REPO)
    src=a.df40 or (F/"df40_prism50.jsonl")
    if not str(src) or not __import__("os").path.exists(src):
        print(f"UNAVAILABLE - DF40 feature file not found: {src}"); return 0
    raw={"real":pd.read_csv(F/"ffpp_original_c23.csv")}
    for m in MANIPULATIONS: raw[m]=pd.read_csv(F/f"ffpp_{m}_c23.csv")
    cols=feature_columns(raw["real"]); P={k:assign_partition(v,id2split) for k,v in raw.items()}
    tr_r=pd.concat([P[k][P[k].partition=="train"] for k in P],ignore_index=True)
    ff_test=P["real"][P["real"].partition=="test"]
    cd=F/"celebdf_features.csv"
    cdr=pd.read_csv(cd)[lambda d: d.label==0] if cd.exists() else None
    rows=[json.loads(l) for l in open(src)]
    ok=pd.DataFrame([r for r in rows if r.get("ok")])
    print("=== GATE: substrate check, run BEFORE any AUC ===")
    gate={}
    for m in sorted(ok.manipulation.unique()):
        s=ok[ok.manipulation==m]
        reps=[check_substrate(v,n_frames=300,native_fps=25.0,is_cropped_face=False,container="video")
              for v in s.video_id]
        g=summarise(reps)
        evaluable = g["fully_in_domain_rate"]>=0.90 and g["mean_n_unreliable"]<1.0
        gate[m]=dict(**g,evaluable=bool(evaluable))
        print(f"  {m:14s} in_domain={g['fully_in_domain_rate']:.3f}  "
              f"{'EVALUABLE' if evaluable else 'NOT EVALUABLE - no AUC will be reported'}")
    res={"generated_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
         "gate":gate,"methods":{},"protocol":"frozen FF++-trained model; zero DF40 training/imputation/thresholds"}
    print("\n=== evaluation (gated methods only) ===")
    aucs=[]
    for m,gi in gate.items():
        if not gi["evaluable"]:
            res["methods"][m]={"status":"NOT EVALUABLE","reason":"failed the substrate gate"}; continue
        s=ok[ok.manipulation==m]
        reals=[d for d in (ff_test,cdr) if d is not None]
        real=pd.concat(reals,ignore_index=True)
        fake=pd.DataFrame([r["features"] if "features" in r else r for _,r in s.iterrows()])
        fake=fake[[c for c in cols if c in fake.columns]]
        tr,(re_,fa_),_=train_median_impute(tr_r,[real,fake],cols)
        sc,clf=fit_prism(tr[cols].values.astype(float),tr.label.values.astype(int),params)
        p=np.r_[score(sc,clf,re_[cols].values.astype(float)),score(sc,clf,fa_[cols].values.astype(float))]
        y=np.r_[np.zeros(len(re_)),np.ones(len(fa_))].astype(int)
        # grouping key must be homogeneous: FF++ reals group by source id, Celeb-DF reals by
        # subject id, DF40 fakes by their own source id. Coerce everything to str.
        if "source_video_id" in re_.columns:
            gr = re_.source_video_id.fillna("").astype(str)
            gr = gr.where(gr != "", pd.Series(
                [f"real_{i}" for i in range(len(re_))], index=re_.index))
        else:
            gr = pd.Series([f"real_{i}" for i in range(len(re_))], index=re_.index)
        g = np.r_[gr.values.astype(str), s.source_video_id.astype(str).values]
        mt=metrics(y,p,g,a.seed); res["methods"][m]=mt; aucs.append(mt["auc"])
        print(f"  {m:14s} AUC={mt['auc']:.4f} CI={mt['grouped_ci']} mF1={mt['macro_f1']:.4f} MCC={mt['mcc']:.4f}")
    if aucs:
        res["macro_average"]=dict(scope=[m for m,g in gate.items() if g["evaluable"]],
            unweighted_mean_auc=round(float(np.mean(aucs)),4),sd=round(float(np.std(aucs,ddof=1)),4),
            note="per-method values must be reported alongside; the macro conceals the spread")
        print(f"\n  macro over {len(aucs)} evaluable methods: {np.mean(aucs):.4f} (sd {np.std(aucs,ddof=1):.4f})")
    o=out_path(a,"df40_eval.json"); json.dump(res,open(o,"w"),indent=1); print(f"\n-> {o}")

if __name__ == "__main__": sys.exit(main() or 0)
