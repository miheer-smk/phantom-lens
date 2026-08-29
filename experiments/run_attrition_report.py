#!/usr/bin/env python
"""R1-C5 attrition report with class-asymmetry testing. Reads splits/evaluation_manifest.csv only."""
import json, time
from _common import base_parser, out_path, REPO
import numpy as np, pandas as pd
from scipy.stats import fisher_exact
VOCAB = ["file_missing","decode_failure","mediapipe_no_face","insufficient_valid_frames",
         "feature_computation_failure","retained"]

def main():
    a = base_parser(__doc__).parse_args()
    d = pd.read_csv(REPO / "splits" / "evaluation_manifest.csv")
    def grp(r):
        if r.dataset == "FFpp": return f"FF++ {r.manipulation} {r.compression}"
        if r.dataset == "CelebDF": return f"Celeb-DF v2 {r.manipulation}"
        if r.dataset == "DF40": return f"DF40 {r.manipulation}"
        return r.dataset
    d["group"] = d.apply(grp, axis=1)
    rows = []
    for (g, cls), s in d.groupby(["group", "class"], dropna=False):
        n = len(s); ret = int(s.video_retained.sum())
        rec = dict(group=g, klass=cls, input_videos=n, processed=ret, excluded=n-ret,
                   exclusion_pct=round(100*(n-ret)/n, 2),
                   median_landmark_success_pct=round(100*float(np.nanmedian(
                       pd.to_numeric(s.landmark_success_ratio, errors="coerce"))), 2)
                       if s.landmark_success_ratio.notna().any() else None)
        for v in VOCAB: rec["excl_"+v] = int((s.exclusion_reason == v).sum())
        rows.append(rec)
    T = pd.DataFrame(rows).sort_values(["group","klass"])
    tests = []
    for ds, s in d.groupby("dataset"):
        piv = s.groupby("class").agg(ret=("video_retained","sum"), n=("video_retained","size"))
        if not {"real","fake"}.issubset(piv.index): continue
        rr, nr = int(piv.loc["real","ret"]), int(piv.loc["real","n"])
        rf, nf = int(piv.loc["fake","ret"]), int(piv.loc["fake","n"])
        odds, p = fisher_exact([[rr, nr-rr], [rf, nf-rf]])
        tests.append(dict(dataset=ds, real_retention=round(rr/nr,4), fake_retention=round(rf/nf,4),
                          odds_ratio=round(float(odds),4), p_uncorrected=float(p)))
    order = np.argsort([t["p_uncorrected"] for t in tests]); m = len(tests); prev = 0.0
    for rank, i in enumerate(order):
        adj = min(1.0, max(prev, (m-rank)*tests[i]["p_uncorrected"])); prev = adj
        tests[i]["p_holm"] = float(adj); tests[i]["significant_holm"] = bool(adj < 0.05)
    print(T[["group","klass","input_videos","processed","excluded","exclusion_pct"]].to_string(index=False))
    print("\nFisher's exact, real vs fake retention:")
    for t in tests:
        print(f"  {t['dataset']:14s} real {t['real_retention']:.4f}  fake {t['fake_retention']:.4f}  "
              f"OR={t['odds_ratio']:.4f}  p={t['p_uncorrected']:.4g}  p_holm={t['p_holm']:.4g}"
              f"{'  ***' if t['significant_holm'] else ''}")
    o = out_path(a, "attrition_report.json")
    json.dump(dict(generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   vocabulary=VOCAB, table=T.to_dict("records"), fisher_tests=tests),
              open(o, "w"), indent=1)
    print(f"\n-> {o}")

if __name__ == "__main__": main()
