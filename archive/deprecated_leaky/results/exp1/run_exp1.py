#!/usr/bin/env python3
"""
Experiment 1 — Multi-manipulation evaluation (c23).

Training: real_train + all 4 manipulation types (one unified model)
Testing:  for each manipulation — real_test + that manipulation type

Reports per-manipulation: AUC, F1, Precision, Recall, MCC
Plots combined ROC curves.
"""

import os, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    matthews_corrcoef, roc_curve, classification_report,
    ConfusionMatrixDisplay, confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
import lightgbm as lgb
from scipy import stats

warnings.filterwarnings("ignore")

BASE   = "/home/iiitn/Miheer_project_FE/phantom-lens"
OUTDIR = f"{BASE}/results/exp1"
os.makedirs(OUTDIR, exist_ok=True)

# ── Dataset registry ──────────────────────────────────────────────────────────
MANIPULATIONS = {
    "Deepfakes":      "features/ffpp_fake.csv",
    "Face2Face":      "features/ffpp_face2face.csv",
    "FaceSwap":       "features/ffpp_faceswap.csv",
    "NeuralTextures": "features/ffpp_neuraltextures.csv",
}

TRAIN_FILES = [
    "features/ffpp_real_train.csv",
    "features/ffpp_fake.csv",
    "features/ffpp_face2face.csv",
    "features/ffpp_faceswap.csv",
    "features/ffpp_neuraltextures.csv",
]

REAL_TEST = "features/ffpp_real_test.csv"

CLF_COLORS = {
    "LogisticRegression": "#3498db",
    "RandomForest":       "#e67e22",
    "LightGBM":           "#9b59b6",
}
MANIP_COLORS = {
    "Deepfakes":      "#e74c3c",
    "Face2Face":      "#2ecc71",
    "FaceSwap":       "#3498db",
    "NeuralTextures": "#f39c12",
}


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_df(file_list):
    dfs = [pd.read_csv(f"{BASE}/{f}") for f in file_list]
    df  = pd.concat(dfs, ignore_index=True)
    fc  = sorted([c for c in df.columns if c.startswith("s_") or c.startswith("t_")])
    df[fc] = df[fc].replace([np.inf, -np.inf], np.nan)
    for c in fc:
        df[c] = df[c].fillna(df[c].median())
    return df[fc].values.astype(np.float64), df["label"].values.astype(int), fc


def bootstrap_ci(y_true, y_prob, n_boot=2000, seed=42):
    rng   = np.random.RandomState(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.choice(len(y_true), size=len(y_true), replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        boots.append(roc_auc_score(y_true[idx], y_prob[idx]))
    return np.percentile(boots, 2.5), np.percentile(boots, 97.5)


# ── Step 1: Train unified multi-manipulation model ───────────────────────────
print("=" * 65)
print("  EXPERIMENT 1 — Multi-manipulation evaluation (c23)")
print("=" * 65)

print(f"\n[1/3] Loading training data ...")
X_tr, y_tr, feat_cols = load_df(TRAIN_FILES)
scaler = StandardScaler()
X_tr_sc = scaler.fit_transform(X_tr)

n_real = (y_tr == 0).sum()
n_fake = (y_tr == 1).sum()
print(f"  Train : {len(y_tr)} samples  (Real={n_real}, Fake={n_fake})")
print(f"  Features: {len(feat_cols)}")

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

# ── Step 2: 10-fold CV on training set ───────────────────────────────────────
print(f"\n[2/3] 10-fold CV on training set ...")
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
cv_results = {}
for clf_name, clf_template in classifiers.items():
    fold_aucs = []
    for tr_idx, val_idx in skf.split(X_tr_sc, y_tr):
        m = clone(clf_template)
        m.fit(X_tr_sc[tr_idx], y_tr[tr_idx])
        p = m.predict_proba(X_tr_sc[val_idx])[:, 1]
        fold_aucs.append(roc_auc_score(y_tr[val_idx], p))
    m_auc  = np.mean(fold_aucs)
    s_auc  = np.std(fold_aucs)
    ci     = stats.t.interval(0.95, df=len(fold_aucs)-1,
                               loc=m_auc, scale=stats.sem(fold_aucs))
    cv_results[clf_name] = {
        "fold_aucs": fold_aucs, "mean": m_auc, "std": s_auc,
        "ci_lo": ci[0], "ci_hi": ci[1],
    }
    short = {"LogisticRegression":"LR","RandomForest":"RF","LightGBM":"LGBM"}[clf_name]
    print(f"  {short:<6}  CV AUC={m_auc:.4f} ± {s_auc:.4f}  "
          f"95%CI=[{ci[0]:.4f}, {ci[1]:.4f}]")

# Fit final models on full training set
trained_clfs = {}
for clf_name, clf_template in classifiers.items():
    m = clone(clf_template)
    m.fit(X_tr_sc, y_tr)
    trained_clfs[clf_name] = m

# ── Step 3: Evaluate per manipulation type ───────────────────────────────────
print(f"\n[3/3] Per-manipulation evaluation ...")

# Threshold: use 0.5 on balanced training classes
THRESHOLD = 0.5

all_results  = {}
roc_data     = {}   # for combined ROC plot

for manip_name, manip_csv in MANIPULATIONS.items():
    X_te, y_te, _ = load_df([REAL_TEST, manip_csv])
    X_te_sc = scaler.transform(X_te)

    n_real_te = (y_te == 0).sum()
    n_fake_te = (y_te == 1).sum()

    print(f"\n  ── {manip_name}  (Real={n_real_te}, Fake={n_fake_te}) ──")

    manip_results = {}
    best_auc = 0.0
    best_clf = None

    for clf_name, clf in trained_clfs.items():
        short = {"LogisticRegression":"LR","RandomForest":"RF","LightGBM":"LGBM"}[clf_name]

        y_prob = clf.predict_proba(X_te_sc)[:, 1]
        y_pred = (y_prob >= THRESHOLD).astype(int)

        auc   = roc_auc_score(y_te, y_prob)
        f1    = f1_score(y_te, y_pred, zero_division=0)
        prec  = precision_score(y_te, y_pred, zero_division=0)
        rec   = recall_score(y_te, y_pred, zero_division=0)
        mcc   = matthews_corrcoef(y_te, y_pred)
        ci_lo, ci_hi = bootstrap_ci(y_te, y_prob)

        manip_results[clf_name] = {
            "auc": auc, "f1": f1, "precision": prec,
            "recall": rec, "mcc": mcc,
            "ci_lo": ci_lo, "ci_hi": ci_hi,
        }

        print(f"    {short:<6}  AUC={auc:.4f} [{ci_lo:.4f},{ci_hi:.4f}]  "
              f"F1={f1:.4f}  P={prec:.4f}  R={rec:.4f}  MCC={mcc:.4f}")

        if auc > best_auc:
            best_auc = auc
            best_clf = clf_name
            best_fpr, best_tpr, _ = roc_curve(y_te, y_prob)
            best_y_prob = y_prob

    all_results[manip_name] = {
        "classifiers": manip_results,
        "best_classifier": best_clf,
        "best_auc": best_auc,
    }
    roc_data[manip_name] = {
        "fpr": best_fpr.tolist(), "tpr": best_tpr.tolist(),
        "auc": best_auc, "clf": best_clf,
        "y_te": y_te, "y_prob": best_y_prob,
    }

# ── Summary table ─────────────────────────────────────────────────────────────
print(f"\n\n{'='*80}")
print("  SUMMARY — Best classifier per manipulation")
print(f"{'='*80}")
print(f"  {'Manipulation':<20} {'Model':>6} {'AUC':>8} {'95% CI':>20} "
      f"{'F1':>7} {'Prec':>7} {'Recall':>8} {'MCC':>7}")
print(f"  {'-'*82}")

summary_rows = []
for manip_name, res in all_results.items():
    bc   = res["best_classifier"]
    r    = res["classifiers"][bc]
    short = {"LogisticRegression":"LR","RandomForest":"RF","LightGBM":"LGBM"}[bc]
    ci_str = f"[{r['ci_lo']:.4f},{r['ci_hi']:.4f}]"
    print(f"  {manip_name:<20} {short:>6} {r['auc']:>8.4f} {ci_str:>20} "
          f"{r['f1']:>7.4f} {r['precision']:>7.4f} {r['recall']:>8.4f} {r['mcc']:>7.4f}")
    summary_rows.append({
        "Manipulation": manip_name,
        "Best Model": short,
        "AUC": round(r["auc"], 4),
        "95% CI Lo": round(r["ci_lo"], 4),
        "95% CI Hi": round(r["ci_hi"], 4),
        "F1": round(r["f1"], 4),
        "Precision": round(r["precision"], 4),
        "Recall": round(r["recall"], 4),
        "MCC": round(r["mcc"], 4),
    })

# ── Save summary CSV ──────────────────────────────────────────────────────────
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(f"{OUTDIR}/summary.csv", index=False)
print(f"\n  Saved: results/exp1/summary.csv")

# ── Combined ROC curve plot ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6))
ax.plot([0,1],[0,1],"k--",lw=1,alpha=0.4,label="Random (AUC=0.50)")

for manip_name, rd in roc_data.items():
    short_clf = {"LogisticRegression":"LR","RandomForest":"RF","LightGBM":"LGBM"}[rd["clf"]]
    label = f"{manip_name} [{short_clf}]  AUC={rd['auc']:.4f}"
    ax.plot(rd["fpr"], rd["tpr"],
            color=MANIP_COLORS[manip_name], lw=2.2, label=label)

ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title("ROC Curves — Multi-manipulation Model\n(Trained on all 4 types, tested per type)",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=9, loc="lower right")
ax.set_xlim([-0.01, 1.01])
ax.set_ylim([-0.01, 1.01])
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/roc_combined.png", dpi=180)
plt.close()
print(f"  Saved: results/exp1/roc_combined.png")

# ── Per-manipulation metric bar chart ─────────────────────────────────────────
metrics   = ["AUC", "F1", "Precision", "Recall", "MCC"]
n_manips  = len(summary_rows)
n_metrics = len(metrics)
x         = np.arange(n_manips)
width     = 0.15

fig, ax = plt.subplots(figsize=(12, 5))
metric_colors = ["#2ecc71","#3498db","#9b59b6","#e74c3c","#f39c12"]
for i, metric in enumerate(metrics):
    vals = [r[metric] for r in summary_rows]
    ax.bar(x + i*width, vals, width, label=metric,
           color=metric_colors[i], alpha=0.85, edgecolor="white")

ax.set_xticks(x + width*2)
ax.set_xticklabels([r["Manipulation"] for r in summary_rows], fontsize=11)
ax.set_ylim(0, 1.1)
ax.set_ylabel("Score", fontsize=12)
ax.set_title("Per-manipulation Metrics — Multi-manipulation Model", fontsize=13, fontweight="bold")
ax.legend(fontsize=9)
ax.axhline(0.5, color="gray", linestyle="--", lw=0.8, alpha=0.5)
for i, row in enumerate(summary_rows):
    ax.text(x[i]+width*2, row["AUC"]+0.02, f"{row['AUC']:.3f}",
            ha="center", fontsize=8, fontweight="bold", color="#2ecc71")
plt.tight_layout()
plt.savefig(f"{OUTDIR}/metrics_bar.png", dpi=180)
plt.close()
print(f"  Saved: results/exp1/metrics_bar.png")

# ── Confusion matrices (best clf per manipulation) ────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, (manip_name, rd) in zip(axes, roc_data.items()):
    y_pred = (rd["y_prob"] >= THRESHOLD).astype(int)
    cm = confusion_matrix(rd["y_te"], y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Real","Fake"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"{manip_name}\nAUC={rd['auc']:.4f}", fontsize=10, fontweight="bold")
plt.suptitle("Confusion Matrices — Best Classifier per Manipulation",
             fontsize=12, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/confusion_matrices.png", dpi=180, bbox_inches="tight")
plt.close()
print(f"  Saved: results/exp1/confusion_matrices.png")

# ── Save full JSON ─────────────────────────────────────────────────────────────
save_results = {
    "cv_results": {k: {kk: vv for kk, vv in v.items() if kk != "fold_aucs"}
                   for k, v in cv_results.items()},
    "per_manipulation": all_results,
    "summary": summary_rows,
}
with open(f"{OUTDIR}/results.json", "w") as f:
    json.dump(save_results, f, indent=2, default=str)
print(f"  Saved: results/exp1/results.json")

print(f"\n{'='*65}")
print(f"  Experiment 1 complete. All outputs in results/exp1/")
print(f"{'='*65}\n")
