#!/usr/bin/env python
"""Regenerate every Table 19 and Section 6.8 value from the RELEASED per-video score CSVs.

No checkpoint, no crops, no dataset access. The three score files in
results/per_video_scores/ are sufficient to recompute the whole baseline comparison, including
the paired DeLong statistic, and each value is checked against the published figure.

Regenerating the score files themselves needs a locally trained checkpoint; the training script
and configuration are in baselines/. See README."""
import json, time
from _common import base_parser, out_path, REPO
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, matthews_corrcoef
from scipy import stats

PUBLISHED = {"xception_ffpp": {"deepfakes":0.9939,"face2face":0.9943,"faceswap":0.9937,"neuraltextures":0.9772},
             "xception_celebdf_shared": 0.8211, "prism_celebdf_shared": 0.6322, "delong_z": 15.426}
TOL = 5e-4

def grouped_ci(y, s, g, n=2000, seed=42):
    """Identity-grouped cluster bootstrap. Duplicated groups REPLICATE their rows."""
    g = np.asarray(g).astype(str)
    assert len(y) == len(s) == len(g), f"misaligned: {len(y)}/{len(s)}/{len(g)}"
    u = np.unique(g); idx = {x: np.flatnonzero(g == x) for x in u}
    rng = np.random.default_rng(seed); out = []
    for _ in range(n):
        i = np.concatenate([idx[x] for x in rng.choice(u, len(u), True)])
        if len(np.unique(y[i])) > 1:
            out.append(roc_auc_score(y[i], s[i]))
    if not out: return None, None, len(u)
    return round(float(np.percentile(out, 2.5)), 4), round(float(np.percentile(out, 97.5)), 4), len(u)


def _auc_var(y, s):
    """DeLong structural components for one score vector."""
    y = np.asarray(y); s = np.asarray(s)
    pos, neg = s[y == 1], s[y == 0]
    m, n = len(pos), len(neg)
    v01 = np.array([(np.sum(neg < p) + 0.5*np.sum(neg == p))/n for p in pos])
    v10 = np.array([(np.sum(pos > q) + 0.5*np.sum(pos == q))/m for q in neg])
    return v01, v10, m, n

def delong(y, s1, s2):
    a1, a2 = roc_auc_score(y, s1), roc_auc_score(y, s2)
    x1, y1, m, n = _auc_var(y, s1); x2, y2, _, _ = _auc_var(y, s2)
    s11 = np.cov(np.vstack([x1, x2]))/m; s00 = np.cov(np.vstack([y1, y2]))/n
    S = s11 + s00
    var = S[0,0] + S[1,1] - 2*S[0,1]
    z = (a1 - a2)/np.sqrt(var)
    return a1, a2, float(z), float(2*stats.norm.sf(abs(z)))

def main():
    a = base_parser(__doc__).parse_args()
    P = REPO/"results"/"per_video_scores"
    xc = pd.read_csv(P/"xception_per_video_scores.csv")
    ls = pd.read_csv(P/"lsda_per_video_scores.csv")
    pr = pd.read_csv(P/"prism_per_video_scores.csv")
    split = pd.read_csv(REPO/"splits"/"ffpp_identity_split.csv")
    id2s = dict(zip(split.source_video_id.astype(str), split.partition))
    sid = lambda v: str(v).split("_")[0].lstrip("0") or "0"
    checks = []
    def chk(label, got, pub, y=None, s=None, groups=None):
        """R1-C2 Step 6 requires AUC, grouped 95% CI, macro-F1, MCC and n for EVERY method."""
        ok = pub is None or abs(got-pub) < TOL
        row = dict(quantity=label, value=round(float(got),4), published=pub, reproduces=bool(ok))
        if y is not None and s is not None:
            pred = (np.asarray(s) >= 0.5).astype(int)
            row.update(n=int(len(y)),
                       macro_f1=round(float(f1_score(y, pred, average="macro")), 4),
                       mcc=round(float(matthews_corrcoef(y, pred)), 4))
            if groups is not None:
                lo, hi, ng = grouped_ci(np.asarray(y), np.asarray(s), np.asarray(groups))
                row.update(grouped_ci=[lo, hi], n_groups=ng)
        checks.append(row)
        flag = "OK" if ok else "*** DIFFERS ***"
        print(f"  {label:42s} {got:.4f}   published {pub}   {flag}" if pub is not None
              else f"  {label:42s} {got:.4f}")

    print("Table 19 - Xception, FF++ test partition, unified aggregation")
    xc["partition"] = xc.video.astype(str).map(sid).map(id2s)
    real = xc[(xc.dataset=="real") & (xc.partition=="test")]
    for m, pub in PUBLISHED["xception_ffpp"].items():
        fk = xc[(xc.dataset==m) & (xc.partition=="test")]
        sub = pd.concat([real.assign(label=0), fk.assign(label=1)], ignore_index=True)
        gsrc = sub.video.astype(str).map(sid).values
        chk(f"Xception {m}", roc_auc_score(sub.label.values, sub.p.values), pub,
            y=sub.label.values, s=sub.p.values, groups=gsrc)

    print("\nTable 19 - shared Celeb-DF v2 intersection")
    xcd = xc[xc.dataset=="celebdf"][["video","label","p"]].rename(columns={"p":"pX"})
    pcd = pr[pr.model=="PRISM_50D_zeroshot"][["video","true_label","pred_prob"]].rename(
              columns={"true_label":"label","pred_prob":"pP"})
    mg = pcd.merge(xcd[["video","pX"]], on="video", how="inner")
    print(f"  shared population n = {len(mg)}   (published comparison used 6121)")
    y = mg.label.values.astype(int)
    ax, ap, z, pv = delong(y, mg.pX.values, mg.pP.values)
    import re as _re
    gid = mg.video.astype(str).map(lambda v: (_re.match(r"(id\d+)", v).group(1)
                                              if _re.match(r"(id\d+)", v) else "youtube_real")).values
    chk("Xception Celeb-DF (shared)", ax, PUBLISHED["xception_celebdf_shared"],
        y=y, s=mg.pX.values, groups=gid)
    chk("PRISM Celeb-DF (shared)",    ap, PUBLISHED["prism_celebdf_shared"],
        y=y, s=mg.pP.values, groups=gid)
    chk("paired DeLong z",             z, PUBLISHED["delong_z"])
    print(f"  DeLong p = {pv:.3g}")

    print("\nSection 6.8 - LSDA (no published value; this revision's measurement)")
    for t in ("deepfakes","face2face","faceswap","neuraltextures","celebdf"):
        sub = ls[ls.target==t]
        if len(sub):
            gl = sub.video.astype(str).map(lambda v: v.split("_")[0]).values
            chk(f"LSDA {t}", roc_auc_score(sub.label.values, sub.p.values), None,
                y=sub.label.values, s=sub.p.values, groups=gl)

    n_bad = sum(1 for c in checks if not c["reproduces"])
    o = out_path(a, "table19_comparison.json")
    json.dump(dict(generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   tolerance=TOL, shared_n=int(len(mg)), checks=checks,
                   all_reproduce=bool(n_bad==0),
                   note="recomputed entirely from released per-video scores; no checkpoint required"),
              open(o,"w"), indent=1)
    n_pub = sum(1 for c in checks if c["published"] is not None)
    print(f"\n{n_pub-n_bad}/{n_pub} published values reproduce within {TOL}. -> {o}")
    return 1 if n_bad else 0

if __name__ == "__main__": raise SystemExit(main())
