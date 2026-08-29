#!/usr/bin/env python
"""R1-C7 identity-grouped cluster bootstrap over the released per-video scores."""
import json, time
from _common import base_parser, out_path, REPO
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from classifiers.prism_pipeline import grouped_ci

def iid_ci(y, s, n=2000, seed=42):
    rng = np.random.default_rng(seed); out = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2: continue
        out.append(roc_auc_score(y[i], s[i]))
    return float(np.percentile(out,2.5)), float(np.percentile(out,97.5))

def main():
    ap = base_parser(__doc__); ap.add_argument("--n_boot", type=int, default=2000)
    a = ap.parse_args()
    pv = pd.read_csv(REPO / "results" / "final_video_scores" / "prism_per_video_scores.csv")
    rows = []
    print(f"{'row':34s} {'AUC':>7s} {'iid width':>10s} {'grouped width':>14s} {'ratio':>7s} {'skip':>5s}")
    for (model, manip), s in pv.groupby(["model","manipulation"]):
        if s.true_label.nunique() < 2: continue
        y = s.true_label.values.astype(int); sc = s.pred_prob.values; g = s.source_id.astype(str).values
        pt = float(roc_auc_score(y, sc))
        glo, ghi, sk, ng = grouped_ci(y, sc, g, n_boot=a.n_boot, seed=a.seed)
        ilo, ihi = iid_ci(y, sc, n=a.n_boot, seed=a.seed)
        gw = (ghi-glo) if glo is not None else None; iw = ihi-ilo
        rows.append(dict(model=model, manipulation=manip, n=int(len(s)), n_groups=ng,
                         point_auc=round(pt,6), iid_ci=[round(ilo,4),round(ihi,4)],
                         grouped_ci=[round(glo,4),round(ghi,4)] if glo else None,
                         width_ratio=round(gw/iw,3) if gw else None, skipped=sk,
                         degenerate_grouping=bool(ng < 5)))
        print(f"{model+' '+str(manip):34s} {pt:7.4f} {iw:10.4f} "
              f"{(f'{gw:.4f}' if gw else 'n/a'):>14s} {(f'{gw/iw:.3f}' if gw else 'n/a'):>7s} {sk:5d}")
    o = out_path(a, "grouped_bootstrap.json")
    json.dump(dict(generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   n_boot=a.n_boot, seed=a.seed,
                   method="cluster bootstrap over source identities; duplicated groups REPLICATE rows",
                   rows=rows), open(o,"w"), indent=1)
    print(f"\n-> {o}")
    print("NOTE: rows flagged degenerate_grouping have too few clusters for a defensible interval.")

if __name__ == "__main__": main()
