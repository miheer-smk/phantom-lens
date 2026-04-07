"""
Phantom Lens V4 — PKL Diagnostic
==================================
Fast text-only analysis of V4 features before training.
Runs in under 2 minutes on any dataset size.

Usage:
    python analyze_v4_pkl.py
    python analyze_v4_pkl.py --pkl data/precomputed_features_v4.pkl
"""

import os
import sys
import time
import pickle
import argparse
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

PKL_PATH = "data/precomputed_features_v4.pkl"

FEATURE_NAMES = [
    "noise_variance",               # 0
    "noise_kurtosis",               # 1
    "noise_entropy",                # 2
    "prnu_correlation",             # 3
    "prnu_energy",                  # 4
    "shadow_consistency",           # 5
    "shadow_direction",             # 6
    "shadow_intensity",             # 7
    "dct_energy_dist",              # 8
    "dct_block_artifacts",          # 9
    "dct_ratio",                    # 10
    "codec_qp_estimate",            # 11
    "codec_bitrate_var",            # 12
    "flow_consistency",             # 13
    "blend_boundary_grad_ratio",    # 14
    "blend_grad_dir_coherence",     # 15
    "blend_boundary_vs_bg_ratio",   # 16
    "freq_peak_energy_ratio",       # 17
    "freq_peak_spacing_regularity", # 18
    "freq_face_bg_spectral_kl",     # 19
]

PILLAR_MAP = {
    "P1 Noise":          [0, 1, 2],
    "P2 PRNU":           [3, 4],
    "P4 Shadow":         [5, 6, 7],
    "P6 DCT":            [8, 9, 10],
    "P7 Codec":          [11, 12],
    "P9 Flow":           [13],
    "Blend Boundary":    [14, 15, 16],
    "Freq Checkerboard": [17, 18, 19],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl", default=PKL_PATH)
    args = parser.parse_args()

    t0 = time.time()

    # ── Load ──────────────────────────────────────────────────────────────
    print("=" * 70)
    print("PHANTOM LENS V4 — PKL DIAGNOSTIC")
    print("=" * 70)

    if not os.path.exists(args.pkl):
        print(f"[FATAL] File not found: {args.pkl}")
        sys.exit(1)

    with open(args.pkl, "rb") as f:
        data = pickle.load(f)

    features = np.array(data["features"], dtype=np.float64)
    labels = np.array(data["labels"], dtype=np.int64)
    video_ids = data.get("video_ids", [str(i) for i in range(len(labels))])
    sources = data.get("dataset_sources", ["unknown"] * len(labels))
    generators = data.get("generator_types", ["unknown"] * len(labels))

    N, D = features.shape
    n_real = int((labels == 0).sum())
    n_fake = int((labels == 1).sum())

    # ── 1. Dataset overview ───────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("1. DATASET OVERVIEW")
    print(f"{'─'*70}")
    print(f"  File:       {args.pkl}")
    print(f"  Size:       {os.path.getsize(args.pkl) / 1024 / 1024:.1f} MB")
    print(f"  Samples:    {N}")
    print(f"  Features:   {D}")
    print(f"  Real:       {n_real} ({100*n_real/N:.1f}%)")
    print(f"  Fake:       {n_fake} ({100*n_fake/N:.1f}%)")
    print(f"  Ratio:      1:{n_fake/max(n_real,1):.2f}")

    src_arr = np.array(sources)
    print(f"\n  Per-source:")
    print(f"    {'Source':<22}  {'Total':>7}  {'Real':>7}  {'Fake':>7}")
    print(f"    {'-'*22}  {'-'*7}  {'-'*7}  {'-'*7}")
    for src in sorted(set(sources)):
        mask = src_arr == src
        sr = int(((labels == 0) & mask).sum())
        sf = int(((labels == 1) & mask).sum())
        print(f"    {src:<22}  {mask.sum():7d}  {sr:7d}  {sf:7d}")

    gen_arr = np.array(generators)
    print(f"\n  Per-generator:")
    for gen in sorted(set(generators)):
        n = int((gen_arr == gen).sum())
        print(f"    {gen:<22}  {n:7d}")

    # NaN/Inf check
    n_nan = np.isnan(features).sum()
    n_inf = np.isinf(features).sum()
    if n_nan > 0 or n_inf > 0:
        print(f"\n  [WARN] NaN={n_nan}, Inf={n_inf} in features!")
        features = np.nan_to_num(features, nan=0.5, posinf=0.5, neginf=0.5)
    else:
        print(f"\n  [OK] No NaN/Inf values")

    # ── Shared train/test split for all analyses ──────────────────────────
    rng = np.random.RandomState(42)
    perm = rng.permutation(N)
    split = int(N * 0.8)
    tr, te = perm[:split], perm[split:]

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(features[tr])
    X_te = scaler.transform(features[te])
    y_tr, y_te = labels[tr], labels[te]

    # ── 2. Per-feature AUC vs real/fake ───────────────────────────────────
    print(f"\n{'─'*70}")
    print("2. PER-FEATURE DISCRIMINABILITY (real vs fake)")
    print(f"{'─'*70}")

    feat_results = []
    for j in range(D):
        try:
            clf = LogisticRegression(max_iter=500, C=1.0, random_state=42)
            clf.fit(X_tr[:, j:j+1], y_tr)
            p = clf.predict_proba(X_te[:, j:j+1])[:, 1]
            auc = roc_auc_score(y_te, p)
        except Exception:
            auc = 0.5

        # Cohen's d on raw features
        rv = features[:, j][labels == 0]
        fv = features[:, j][labels == 1]
        ps = np.sqrt(((len(rv)-1)*rv.var(ddof=1) + (len(fv)-1)*fv.var(ddof=1)) /
                     (len(rv)+len(fv)-2))
        d = (rv.mean() - fv.mean()) / max(ps, 1e-12)

        fname = FEATURE_NAMES[j] if j < len(FEATURE_NAMES) else f"dim_{j}"
        pillar = "?"
        for pname, dims in PILLAR_MAP.items():
            if j in dims:
                pillar = pname
                break

        feat_results.append((j, fname, pillar, auc, d))

    # Sort by AUC descending
    feat_results.sort(key=lambda x: x[3], reverse=True)

    print(f"  {'Dim':>3}  {'Feature':<35}  {'Pillar':<18}  {'AUC':>7}  {'Cohen d':>9}  {'Verdict':<10}")
    print(f"  {'-'*3}  {'-'*35}  {'-'*18}  {'-'*7}  {'-'*9}  {'-'*10}")
    for j, fname, pillar, auc, d in feat_results:
        if abs(d) > 0.5:
            v = "STRONG"
        elif abs(d) > 0.3:
            v = "MODERATE"
        else:
            v = "WEAK"
        print(f"  {j:3d}  {fname:<35}  {pillar:<18}  {auc:7.4f}  {d:+9.4f}  {v:<10}")

    # Per-pillar summary
    print(f"\n  Per-pillar summary:")
    print(f"    {'Pillar':<22}  {'Mean AUC':>9}  {'Max AUC':>9}  {'Verdict':<10}")
    print(f"    {'-'*22}  {'-'*9}  {'-'*9}  {'-'*10}")
    for pname, dims in PILLAR_MAP.items():
        pillar_aucs = [r[3] for r in feat_results if r[0] in dims]
        mean_a = np.mean(pillar_aucs)
        max_a = np.max(pillar_aucs)
        v = "STRONG" if max_a > 0.75 else ("MODERATE" if max_a > 0.60 else "WEAK")
        print(f"    {pname:<22}  {mean_a:9.4f}  {max_a:9.4f}  {v:<10}")

    # ── 3. Overall confound AUC ───────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("3. OVERALL CONFOUND AUC (all 20 features -> dataset source)")
    print(f"{'─'*70}")

    le = LabelEncoder()
    src_labels = le.fit_transform(sources)
    n_classes = len(le.classes_)
    print(f"  Sources: {list(le.classes_)} ({n_classes} classes)")

    src_tr, src_te = src_labels[tr], src_labels[te]

    if n_classes < 2:
        print("  [SKIP] Only one source")
        overall_confound = 0.5
    else:
        clf = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs',
                                  random_state=42)
        clf.fit(X_tr, src_tr)

        if n_classes == 2:
            p = clf.predict_proba(X_te)[:, 1]
            overall_confound = roc_auc_score(src_te, p)
        else:
            p = clf.predict_proba(X_te)
            overall_confound = roc_auc_score(src_te, p,
                                              multi_class='ovr', average='weighted')

        if overall_confound > 0.90:
            cv = "CONFOUNDED"
        elif overall_confound > 0.75:
            cv = "MODERATE"
        elif overall_confound > 0.60:
            cv = "ACCEPTABLE"
        else:
            cv = "GOOD"

        print(f"\n  Overall confound AUC:  {overall_confound:.4f}  [{cv}]")
        print(f"  V3 base was:           0.9638  [CONFOUNDED]")
        print(f"  V3 + CelebDF was:      0.9087  [CONFOUNDED]")
        print(f"  Change from V3 base:   {overall_confound - 0.9638:+.4f}")

    # ── 4. Per-feature confound AUC ───────────────────────────────────────
    print(f"\n{'─'*70}")
    print("4. PER-FEATURE CONFOUND AUC (each feature -> dataset source)")
    print(f"{'─'*70}")

    confound_results = []
    n_confounded = 0

    for j in range(D):
        try:
            clf_j = LogisticRegression(max_iter=500, random_state=42)
            clf_j.fit(X_tr[:, j:j+1], src_tr)
            if n_classes == 2:
                p = clf_j.predict_proba(X_te[:, j:j+1])[:, 1]
                a = roc_auc_score(src_te, p)
            else:
                p = clf_j.predict_proba(X_te[:, j:j+1])
                a = roc_auc_score(src_te, p, multi_class='ovr', average='weighted')
        except Exception:
            a = 0.5

        fname = FEATURE_NAMES[j] if j < len(FEATURE_NAMES) else f"dim_{j}"
        status = "CONFOUNDED" if a > 0.80 else "ok"
        if a > 0.80:
            n_confounded += 1
        confound_results.append((j, fname, a, status))

    # Sort by confound AUC descending
    confound_results.sort(key=lambda x: x[2], reverse=True)

    print(f"  {'Dim':>3}  {'Feature':<35}  {'Confound AUC':>12}  {'Status':<12}")
    print(f"  {'-'*3}  {'-'*35}  {'-'*12}  {'-'*12}")
    for j, fname, a, status in confound_results:
        flag = " <<<" if status == "CONFOUNDED" else ""
        print(f"  {j:3d}  {fname:<35}  {a:12.4f}  {status:<12}{flag}")

    print(f"\n  Confounded features (AUC > 0.80): {n_confounded} / {D}")
    if n_confounded > 0:
        print(f"  Worst offenders:")
        for j, fname, a, status in confound_results[:3]:
            if a > 0.80:
                print(f"    dim {j:2d} {fname:<35} AUC={a:.4f}")

    # ── 5. Feature correlation matrix ─────────────────────────────────────
    print(f"\n{'─'*70}")
    print("5. FEATURE CORRELATION — Top 5 most correlated pairs")
    print(f"{'─'*70}")

    corr = np.corrcoef(features.T)  # (20, 20)
    pairs = []
    for i in range(D):
        for j in range(i+1, D):
            pairs.append((i, j, abs(corr[i, j])))
    pairs.sort(key=lambda x: x[2], reverse=True)

    print(f"  {'Dim A':>5}  {'Feature A':<30}  {'Dim B':>5}  {'Feature B':<30}  {'|corr|':>7}")
    print(f"  {'-'*5}  {'-'*30}  {'-'*5}  {'-'*30}  {'-'*7}")
    for i, j, c in pairs[:5]:
        fi = FEATURE_NAMES[i] if i < len(FEATURE_NAMES) else f"dim_{i}"
        fj = FEATURE_NAMES[j] if j < len(FEATURE_NAMES) else f"dim_{j}"
        warn = " REDUNDANT" if c > 0.90 else ""
        print(f"  {i:5d}  {fi:<30}  {j:5d}  {fj:<30}  {c:7.4f}{warn}")

    n_high_corr = sum(1 for _, _, c in pairs if c > 0.90)
    if n_high_corr > 0:
        print(f"\n  [WARN] {n_high_corr} feature pairs have |corr| > 0.90")
    else:
        print(f"\n  [OK] No highly redundant feature pairs")

    # New vs old pillar orthogonality
    old_dims = list(range(14))
    new_dims = list(range(14, 20))
    cross_corrs = []
    for i in old_dims:
        for j in new_dims:
            cross_corrs.append(abs(corr[i, j]))
    max_cross = max(cross_corrs)
    mean_cross = np.mean(cross_corrs)
    print(f"\n  New pillar orthogonality:")
    print(f"    Mean |corr| (old vs new): {mean_cross:.4f}")
    print(f"    Max  |corr| (old vs new): {max_cross:.4f}")
    if max_cross < 0.50:
        print(f"    [OK] New pillars provide orthogonal signal")
    elif max_cross < 0.75:
        print(f"    [OK] New pillars moderately independent")
    else:
        print(f"    [WARN] New pillars overlap with existing features")

    # ── 6. Final verdict ──────────────────────────────────────────────────
    elapsed = time.time() - t0

    print(f"\n{'─'*70}")
    print("6. FINAL VERDICT")
    print(f"{'─'*70}")

    issues = []

    if overall_confound > 0.90:
        issues.append(f"Overall confound AUC {overall_confound:.4f} > 0.90 — still heavily confounded")
    elif overall_confound > 0.85:
        issues.append(f"Overall confound AUC {overall_confound:.4f} > 0.85 — moderate confounding")

    best_disc_auc = max(r[3] for r in feat_results)
    if best_disc_auc < 0.60:
        issues.append(f"Best feature AUC is only {best_disc_auc:.4f} — no strong discriminative features")

    new_aucs = [r[3] for r in feat_results if r[0] >= 14]
    best_new = max(new_aucs) if new_aucs else 0.5
    if best_new < 0.55:
        issues.append(f"New pillars best AUC = {best_new:.4f} — not contributing signal")

    if n_confounded > 10:
        issues.append(f"{n_confounded}/20 features confounded — majority carry dataset identity")

    if n_nan > 0 or n_inf > 0:
        issues.append(f"Found {n_nan} NaN and {n_inf} Inf values")

    ratio = n_fake / max(n_real, 1)
    if ratio > 3.0 or ratio < 0.33:
        issues.append(f"Severe class imbalance: ratio 1:{ratio:.2f}")

    print()
    if len(issues) == 0:
        print("  VERDICT: READY TO TRAIN")
        print("    No blocking issues. Run train_v4.py.")
    elif len(issues) <= 2 and overall_confound < 0.90:
        print("  VERDICT: READY TO TRAIN (with caution)")
        print("    Minor issues:")
        for iss in issues:
            print(f"      - {iss}")
        print("    Proceed with train_v4.py but monitor these.")
    else:
        print("  VERDICT: NEEDS FIXING")
        print("    Blocking issues:")
        for iss in issues:
            print(f"      - {iss}")

    print(f"\n  Key numbers:")
    print(f"    Overall confound AUC:     {overall_confound:.4f}")
    print(f"    Best discriminative AUC:  {best_disc_auc:.4f}")
    print(f"    Best new pillar AUC:      {best_new:.4f}")
    print(f"    Confounded features:      {n_confounded}/{D}")
    print(f"    Highly correlated pairs:  {n_high_corr}")
    print(f"    Analysis time:            {elapsed:.1f}s")

    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()