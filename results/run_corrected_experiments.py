#!/usr/bin/env python3
"""
Corrected experiment re-runs:
  - Real samples split (no overlap between train/test)
  - 10-fold stratified cross-validation
  - 95% confidence intervals on CV AUC
  - Comparison with previous results
"""
import json, os, warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
import lightgbm as lgb

warnings.filterwarnings("ignore")

BASE = "/home/iiitn/Miheer_project_FE/phantom-lens"

# ── Previous results for comparison ──────────────────────────────────────────
PREVIOUS = {
    "Deepfakes→Face2Face":       0.6809,
    "Multi-manip→NeuralTextures": 0.9965,
    "Multi-manip→FaceShifter":    0.9957,
}

CLF_SHORT = {
    "LogisticRegression": "LR",
    "RandomForest":       "RF",
    "LightGBM":           "LGBM",
}

EXPERIMENTS = [
    (
        "Deepfakes→Face2Face",
        ["features/ffpp_real_train.csv", "features/ffpp_fake.csv"],
        ["features/ffpp_real_test.csv",  "features/ffpp_face2face.csv"],
        "results/corrected_deepfakes_to_face2face",
    ),
    (
        "Multi-manip→NeuralTextures",
        ["features/ffpp_real_train.csv", "features/ffpp_fake.csv",
         "features/ffpp_face2face.csv",  "features/ffpp_faceswap.csv",
         "features/ffpp_faceshifter.csv"],
        ["features/ffpp_real_test.csv",  "features/ffpp_neuraltextures.csv"],
        "results/corrected_multi_to_neuraltextures",
    ),
    (
        "Multi-manip→FaceShifter",
        ["features/ffpp_real_train.csv", "features/ffpp_fake.csv",
         "features/ffpp_face2face.csv",  "features/ffpp_faceswap.csv"],
        ["features/ffpp_real_test.csv",  "features/ffpp_faceshifter.csv"],
        "results/corrected_multi_to_faceshifter",
    ),
]


def load_data(file_list):
    dfs = [pd.read_csv(f"{BASE}/{f}") for f in file_list]
    df  = pd.concat(dfs, ignore_index=True)
    fc  = sorted([c for c in df.columns if c.startswith("s_") or c.startswith("t_")])
    df[fc] = df[fc].replace([np.inf, -np.inf], np.nan)
    for c in fc:
        df[c] = df[c].fillna(df[c].median())
    return df[fc].values.astype(np.float64), df["label"].values.astype(int), fc


def ci95(aucs):
    n  = len(aucs)
    m  = np.mean(aucs)
    se = stats.sem(aucs)
    lo, hi = stats.t.interval(0.95, df=n-1, loc=m, scale=se)
    return m, np.std(aucs), lo, hi


def run_experiment(name, train_files, test_files, out_dir):
    os.makedirs(f"{BASE}/{out_dir}", exist_ok=True)

    print(f"\n{'='*65}")
    print(f"  {name}")
    print(f"{'='*65}")

    X_tr, y_tr, fc = load_data(train_files)
    X_te, y_te, _  = load_data(test_files)

    print(f"  Train : {len(y_tr)}  (Real={( y_tr==0).sum()}, Fake={(y_tr==1).sum()})")
    print(f"  Test  : {len(y_te)}  (Real={(y_te==0).sum()}, Fake={(y_te==1).sum()})")
    print(f"  Features: {len(fc)}")

    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_te_sc = scaler.transform(X_te)

    classifiers = {
        "LogisticRegression": LogisticRegression(
            C=1.0, penalty="l2", solver="lbfgs", max_iter=2000,
            class_weight="balanced", random_state=42),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=10,
            class_weight="balanced", random_state=42, n_jobs=-1),
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            num_leaves=31, min_child_samples=20,
            class_weight="balanced", random_state=42, verbose=-1),
    }

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    results = {}
    best_auc = 0.0
    best_name = None

    print(f"\n  {'Classifier':<22} {'CV AUC mean':>12} {'± std':>8} {'95% CI':>22} {'Test AUC':>10}")
    print(f"  {'-'*78}")

    for clf_name, clf_template in classifiers.items():
        # ── 10-fold CV on training set ───────────────────────────────────────
        fold_aucs = []
        for tr_idx, val_idx in skf.split(X_tr_sc, y_tr):
            m = clone(clf_template)
            m.fit(X_tr_sc[tr_idx], y_tr[tr_idx])
            p = m.predict_proba(X_tr_sc[val_idx])[:, 1]
            fold_aucs.append(roc_auc_score(y_tr[val_idx], p))

        cv_mean, cv_std, ci_lo, ci_hi = ci95(fold_aucs)

        # ── Full train → test evaluation ─────────────────────────────────────
        clf = clone(clf_template)
        clf.fit(X_tr_sc, y_tr)
        y_prob   = clf.predict_proba(X_te_sc)[:, 1]
        test_auc = roc_auc_score(y_te, y_prob)

        short = CLF_SHORT[clf_name]
        ci_str = f"[{ci_lo:.4f}, {ci_hi:.4f}]"
        print(f"  {short:<22} {cv_mean:>12.4f} {cv_std:>8.4f} {ci_str:>22} {test_auc:>10.4f}")

        results[clf_name] = {
            "cv_fold_aucs": fold_aucs,
            "cv_auc_mean":  cv_mean,
            "cv_auc_std":   cv_std,
            "cv_ci_lo":     ci_lo,
            "cv_ci_hi":     ci_hi,
            "test_auc":     test_auc,
        }

        if test_auc > best_auc:
            best_auc  = test_auc
            best_name = clf_name

    results["best_classifier"] = best_name
    results["best_test_auc"]   = best_auc
    results["experiment_name"] = name

    # ── CI on test AUC via bootstrap ─────────────────────────────────────────
    clf_best = clone(classifiers[best_name])
    clf_best.fit(X_tr_sc, y_tr)
    y_prob_best = clf_best.predict_proba(X_te_sc)[:, 1]

    rng = np.random.RandomState(42)
    boot_aucs = []
    for _ in range(2000):
        idx = rng.choice(len(y_te), size=len(y_te), replace=True)
        if len(np.unique(y_te[idx])) < 2:
            continue
        boot_aucs.append(roc_auc_score(y_te[idx], y_prob_best[idx]))
    test_ci_lo = np.percentile(boot_aucs, 2.5)
    test_ci_hi = np.percentile(boot_aucs, 97.5)

    results["best_test_ci_lo"] = test_ci_lo
    results["best_test_ci_hi"] = test_ci_hi

    print(f"\n  Best: {CLF_SHORT[best_name]}  Test AUC={best_auc:.4f}  "
          f"Bootstrap 95% CI=[{test_ci_lo:.4f}, {test_ci_hi:.4f}]")

    # ── Compare with previous ─────────────────────────────────────────────────
    prev = PREVIOUS.get(name)
    if prev is not None:
        delta = best_auc - prev
        flag  = "⚠  SIGNIFICANT CHANGE" if abs(delta) > 0.02 else "✓  within tolerance"
        print(f"  Previous AUC={prev:.4f}  Δ={delta:+.4f}  {flag}")
        results["previous_auc"] = prev
        results["delta"]        = delta

    with open(f"{BASE}/{out_dir}/results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    return {
        "name":       name,
        "best_model": CLF_SHORT[best_name],
        "test_auc":   best_auc,
        "ci_lo":      test_ci_lo,
        "ci_hi":      test_ci_hi,
        "prev_auc":   prev,
        "delta":      best_auc - prev if prev else None,
    }


# ── Run all ───────────────────────────────────────────────────────────────────
summary = []
for exp_name, tr_files, te_files, out_dir in EXPERIMENTS:
    row = run_experiment(exp_name, tr_files, te_files, out_dir)
    summary.append(row)

# ── Final table ───────────────────────────────────────────────────────────────
print(f"\n\n{'='*65}")
print("  FINAL CORRECTED RESULTS TABLE")
print(f"{'='*65}")
print(f"  {'Train':<22} {'Test':<20} {'Model':>6} {'AUC':>8} {'95% CI':>22} {'vs prev':>10} {'Flag'}")
print(f"  {'-'*95}")

for row in summary:
    train_part, test_part = row["name"].split("→")
    delta_str = f"{row['delta']:+.4f}" if row["delta"] is not None else "—"
    flag = "⚠ SIG" if row["delta"] is not None and abs(row["delta"]) > 0.02 else "✓"
    ci_str = f"[{row['ci_lo']:.4f}, {row['ci_hi']:.4f}]"
    print(f"  {train_part:<22} {test_part:<20} {row['best_model']:>6} "
          f"{row['test_auc']:>8.4f} {ci_str:>22} {delta_str:>10} {flag}")

print()
