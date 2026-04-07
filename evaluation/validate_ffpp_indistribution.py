#!/usr/bin/env python3
"""
PHANTOM LENS — FF++ In-Distribution Validation
===============================================
Complete validation suite for a model trained on FF++ (Real vs Deepfakes).
Covers ALL in-distribution metrics: Precision, Recall, F1, Confusion Matrix,
ROC/PR Curves, Threshold Optimization, Calibration, Bootstrap CI, and SHAP.

Usage:
    python validate_ffpp_indistribution.py

Outputs saved to: results_ffpp_validation/
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    roc_auc_score, roc_curve, auc,
    precision_recall_curve, average_precision_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    f1_score, precision_score, recall_score, accuracy_score,
    brier_score_loss, log_loss, matthews_corrcoef
)
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.manifold import TSNE

warnings.filterwarnings("ignore")

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    print("[WARN] LightGBM not found — skipping LGBM classifier.")

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("[WARN] SHAP not found. Run: pip install shap — SHAP plots will be skipped.")

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
OUTPUT_DIR    = "results_ffpp_validation"
N_SPLITS      = 5        # Cross-validation folds
BOOTSTRAP_N   = 1000     # Bootstrap iterations for CIs
RANDOM_STATE  = 42
THRESHOLD_STEP = 0.01    # Threshold sweep resolution
TSNE_SAMPLE   = 800      # Subsample for t-SNE

os.makedirs(OUTPUT_DIR, exist_ok=True)
np.random.seed(RANDOM_STATE)

PALETTE = {
    "real":   "#4a90d9",
    "fake":   "#e24b4a",
    "green":  "#27ae60",
    "orange": "#ef9f27",
    "purple": "#8e44ad",
    "dark":   "#2c3e50",
}

# ─────────────────────────────────────────────────────────────
# SECTION 0 — LOAD & PREPROCESS
# ─────────────────────────────────────────────────────────────
print("=" * 65)
print("  PHANTOM LENS — FF++ In-Distribution Validation")
print("=" * 65)

real_df = pd.read_csv("features/ffpp_real.csv")
fake_df = pd.read_csv("features/ffpp_fake.csv")
real_df["label"] = 0
fake_df["label"] = 1

df = pd.concat([real_df, fake_df], ignore_index=True)

feature_cols = sorted([c for c in df.columns
                        if c.startswith("s_") or c.startswith("t_")])

for col in feature_cols:
    df[col] = df[col].fillna(df[col].median())
df = df.replace([np.inf, -np.inf], np.nan)
for col in feature_cols:
    df[col] = df[col].fillna(df[col].median())

X_raw = df[feature_cols].values.astype(np.float64)
y     = df["label"].values.astype(int)

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

n_real = (y == 0).sum()
n_fake = (y == 1).sum()

print(f"\nReal : {n_real:,}")
print(f"Fake : {n_fake:,}")
print(f"Features : {len(feature_cols)}")
print(f"Class balance : {n_real/(n_real+n_fake)*100:.1f}% Real / "
      f"{n_fake/(n_real+n_fake)*100:.1f}% Fake")

# ─────────────────────────────────────────────────────────────
# SECTION 1 — CLASSIFIERS
# ─────────────────────────────────────────────────────────────
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                      random_state=RANDOM_STATE)

classifiers = {
    "LogisticRegression": LogisticRegression(
        C=1.0, max_iter=2000, class_weight="balanced",
        random_state=RANDOM_STATE),
    "RandomForest": RandomForestClassifier(
        n_estimators=200, max_depth=8, class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1),
}
if HAS_LGBM:
    classifiers["LightGBM"] = lgb.LGBMClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        class_weight="balanced", random_state=RANDOM_STATE, verbose=-1)

# Gather cross-val predicted probabilities for every classifier
cv_proba = {}
cv_preds = {}

print(f"\nRunning {N_SPLITS}-fold cross-validation ...\n")
for name, clf in classifiers.items():
    proba = cross_val_predict(clf, X_scaled, y, cv=skf,
                              method="predict_proba")[:, 1]
    pred  = (proba >= 0.5).astype(int)
    cv_proba[name] = proba
    cv_preds[name] = pred
    print(f"  ✓ {name}")

# ─────────────────────────────────────────────────────────────
# HELPER — Bootstrap CI
# ─────────────────────────────────────────────────────────────
def bootstrap_metric(y_true, y_score_or_pred, metric_fn,
                     n=BOOTSTRAP_N, ci=0.95, is_proba=True):
    """Return (mean, lower, upper) with confidence interval."""
    scores = []
    n_samples = len(y_true)
    for _ in range(n):
        idx = np.random.choice(n_samples, n_samples, replace=True)
        yt  = y_true[idx]
        yp  = y_score_or_pred[idx]
        if len(np.unique(yt)) < 2:
            continue
        try:
            scores.append(metric_fn(yt, yp))
        except Exception:
            pass
    scores = np.array(scores)
    alpha  = (1 - ci) / 2
    return scores.mean(), np.percentile(scores, alpha*100), np.percentile(scores, (1-alpha)*100)


# ─────────────────────────────────────────────────────────────
# SECTION 2 — CORE CLASSIFICATION REPORT + SUMMARY TABLE
# ─────────────────────────────────────────────────────────────
print("\n" + "─" * 65)
print("  SECTION 2 — Classification Reports (threshold = 0.50)")
print("─" * 65)

summary_rows = []

for name, proba in cv_proba.items():
    pred = cv_preds[name]
    print(f"\n{'═'*50}")
    print(f"  {name}")
    print('═'*50)
    print(classification_report(y, pred, target_names=["Real", "Fake"],
                                 digits=4))

    acc  = accuracy_score(y, pred)
    prec = precision_score(y, pred, zero_division=0)
    rec  = recall_score(y, pred, zero_division=0)
    f1   = f1_score(y, pred, zero_division=0)
    mcc  = matthews_corrcoef(y, pred)
    auc_ = roc_auc_score(y, proba)
    ap   = average_precision_score(y, proba)
    bs   = brier_score_loss(y, proba)
    ll   = log_loss(y, proba)

    # Bootstrap CI for AUC and F1
    auc_mean, auc_lo, auc_hi = bootstrap_metric(
        y, proba, roc_auc_score)
    f1_mean, f1_lo, f1_hi = bootstrap_metric(
        y, pred, f1_score, is_proba=False)

    summary_rows.append({
        "Classifier":    name,
        "Accuracy":      round(acc,  4),
        "Precision":     round(prec, 4),
        "Recall":        round(rec,  4),
        "F1":            round(f1,   4),
        "F1_CI_lo":      round(f1_lo, 4),
        "F1_CI_hi":      round(f1_hi, 4),
        "ROC-AUC":       round(auc_, 4),
        "AUC_CI_lo":     round(auc_lo, 4),
        "AUC_CI_hi":     round(auc_hi, 4),
        "Avg Precision": round(ap,   4),
        "Brier Score":   round(bs,   4),
        "Log Loss":      round(ll,   4),
        "MCC":           round(mcc,  4),
    })

    print(f"  ROC-AUC : {auc_:.4f}  [{auc_lo:.4f} – {auc_hi:.4f}]  (95% CI)")
    print(f"  F1      : {f1:.4f}  [{f1_lo:.4f} – {f1_hi:.4f}]  (95% CI)")
    print(f"  MCC     : {mcc:.4f}")
    print(f"  Brier   : {bs:.4f}")
    print(f"  LogLoss : {ll:.4f}")

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(f"{OUTPUT_DIR}/00_validation_summary.csv", index=False)
print(f"\nSaved: 00_validation_summary.csv")


# ─────────────────────────────────────────────────────────────
# SECTION 3 — CONFUSION MATRICES  (one per classifier)
# ─────────────────────────────────────────────────────────────
print("\nGenerating confusion matrices ...")

n_clf = len(classifiers)
fig, axes = plt.subplots(1, n_clf, figsize=(5 * n_clf, 4.5))
if n_clf == 1:
    axes = [axes]

for ax, (name, pred) in zip(axes, cv_preds.items()):
    cm = confusion_matrix(y, pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Real", "Fake"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    tn, fp, fn, tp = cm.ravel()
    ax.set_title(
        f"{name}\n"
        f"TN={tn}  FP={fp}\n"
        f"FN={fn}  TP={tp}",
        fontsize=9, fontweight="bold"
    )
    ax.set_xlabel("Predicted", fontsize=9)
    ax.set_ylabel("True", fontsize=9)

plt.suptitle("Confusion Matrices — FF++ (5-fold CV, threshold=0.50)",
             fontsize=12, fontweight="bold", y=1.03)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/01_confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 01_confusion_matrices.png")


# ─────────────────────────────────────────────────────────────
# SECTION 4 — ROC CURVES + AUC with 95% CI BAND
# ─────────────────────────────────────────────────────────────
print("Generating ROC curves ...")

clf_colors = [PALETTE["real"], PALETTE["real"],
              PALETTE["green"], PALETTE["orange"], PALETTE["purple"]]

fig, ax = plt.subplots(figsize=(7, 6))

for i, (name, proba) in enumerate(cv_proba.items()):
    fpr, tpr, _ = roc_curve(y, proba)
    auc_val      = auc(fpr, tpr)
    color        = list(PALETTE.values())[i % len(PALETTE)]

    # Bootstrap CI band on ROC
    tpr_list = []
    base_fpr = np.linspace(0, 1, 200)
    for _ in range(300):
        idx = np.random.choice(len(y), len(y), replace=True)
        if len(np.unique(y[idx])) < 2:
            continue
        fpr_b, tpr_b, _ = roc_curve(y[idx], proba[idx])
        tpr_list.append(np.interp(base_fpr, fpr_b, tpr_b))
    tpr_arr  = np.array(tpr_list)
    tpr_lo   = np.percentile(tpr_arr, 2.5, axis=0)
    tpr_hi   = np.percentile(tpr_arr, 97.5, axis=0)

    ax.plot(fpr, tpr, lw=2, color=color,
            label=f"{name}  AUC={auc_val:.4f}")
    ax.fill_between(base_fpr, tpr_lo, tpr_hi, color=color, alpha=0.12)

ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
ax.set_xlabel("False Positive Rate", fontsize=11)
ax.set_ylabel("True Positive Rate", fontsize=11)
ax.set_title("ROC Curves — FF++ In-Distribution (5-fold CV)\n"
             "Shaded = 95% CI via Bootstrap",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=9, loc="lower right")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/02_roc_curves_with_ci.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 02_roc_curves_with_ci.png")


# ─────────────────────────────────────────────────────────────
# SECTION 5 — PRECISION–RECALL CURVES
# ─────────────────────────────────────────────────────────────
print("Generating Precision–Recall curves ...")

fig, ax = plt.subplots(figsize=(7, 6))
baseline = n_fake / (n_real + n_fake)

for i, (name, proba) in enumerate(cv_proba.items()):
    prec_curve, rec_curve, _ = precision_recall_curve(y, proba)
    ap = average_precision_score(y, proba)
    color = list(PALETTE.values())[i % len(PALETTE)]
    ax.plot(rec_curve, prec_curve, lw=2, color=color,
            label=f"{name}  AP={ap:.4f}")

ax.axhline(y=baseline, color="gray", linestyle="--", lw=1,
           label=f"Random (AP={baseline:.3f})")
ax.set_xlabel("Recall", fontsize=11)
ax.set_ylabel("Precision", fontsize=11)
ax.set_title("Precision–Recall Curves — FF++ In-Distribution (5-fold CV)",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1.05])
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/03_pr_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 03_pr_curves.png")


# ─────────────────────────────────────────────────────────────
# SECTION 6 — THRESHOLD SWEEP  (Precision / Recall / F1 vs θ)
# ─────────────────────────────────────────────────────────────
print("Generating threshold sweep plots ...")

fig, axes = plt.subplots(1, n_clf, figsize=(7 * n_clf, 5))
if n_clf == 1:
    axes = [axes]

for ax, (name, proba) in zip(axes, cv_proba.items()):
    thresholds = np.arange(0.01, 1.0, THRESHOLD_STEP)
    precs, recs, f1s, accs = [], [], [], []

    for t in thresholds:
        pred_t = (proba >= t).astype(int)
        precs.append(precision_score(y, pred_t, zero_division=0))
        recs.append(recall_score(y, pred_t, zero_division=0))
        f1s.append(f1_score(y, pred_t, zero_division=0))
        accs.append(accuracy_score(y, pred_t))

    best_idx  = int(np.argmax(f1s))
    best_t    = thresholds[best_idx]
    best_f1   = f1s[best_idx]

    ax.plot(thresholds, precs, color=PALETTE["real"],   lw=2, label="Precision")
    ax.plot(thresholds, recs,  color=PALETTE["real"],   lw=2, linestyle="--", label="Recall")
    ax.plot(thresholds, f1s,   color=PALETTE["green"],  lw=2.5, label="F1")
    ax.plot(thresholds, accs,  color=PALETTE["orange"], lw=1.5, linestyle=":", label="Accuracy")
    ax.axvline(best_t, color=PALETTE["dark"], lw=1.5, linestyle="-.",
               label=f"Best F1 θ={best_t:.2f} (F1={best_f1:.3f})")

    ax.set_xlabel("Decision Threshold (θ)", fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title(f"{name}\nThreshold Sweep", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])

    print(f"  {name}: Best F1={best_f1:.4f} at θ={best_t:.2f}")

plt.suptitle("Threshold Sweep — Precision / Recall / F1 vs θ  (FF++ CV)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/04_threshold_sweep.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 04_threshold_sweep.png")


# ─────────────────────────────────────────────────────────────
# SECTION 7 — OPTIMAL THRESHOLD CONFUSION MATRICES
# ─────────────────────────────────────────────────────────────
print("Generating optimal-threshold confusion matrices ...")

fig, axes = plt.subplots(1, n_clf, figsize=(5 * n_clf, 4.5))
if n_clf == 1:
    axes = [axes]

for ax, (name, proba) in zip(axes, cv_proba.items()):
    thresholds = np.arange(0.01, 1.0, THRESHOLD_STEP)
    f1s = [f1_score(y, (proba >= t).astype(int), zero_division=0)
           for t in thresholds]
    best_t = thresholds[int(np.argmax(f1s))]
    pred_t = (proba >= best_t).astype(int)

    cm   = confusion_matrix(y, pred_t)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Real", "Fake"])
    disp.plot(ax=ax, colorbar=False, cmap="Greens")
    tn, fp, fn, tp = cm.ravel()
    f1_opt = f1_score(y, pred_t)
    ax.set_title(
        f"{name}  θ={best_t:.2f}\n"
        f"TN={tn}  FP={fp}  FN={fn}  TP={tp}\n"
        f"F1={f1_opt:.4f}",
        fontsize=8.5, fontweight="bold"
    )

plt.suptitle("Confusion Matrices at Optimal F1 Threshold — FF++ (CV)",
             fontsize=12, fontweight="bold", y=1.03)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/05_confusion_optimal_threshold.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 05_confusion_optimal_threshold.png")


# ─────────────────────────────────────────────────────────────
# SECTION 8 — PER-FOLD METRIC VARIANCE
# ─────────────────────────────────────────────────────────────
print("Computing per-fold metric variance ...")

fold_metrics = {name: {"AUC": [], "F1": [], "Precision": [], "Recall": []}
                for name in classifiers}

for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_scaled, y)):
    X_tr, X_val = X_scaled[train_idx], X_scaled[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]

    for name, clf in classifiers.items():
        clf.fit(X_tr, y_tr)
        prob = clf.predict_proba(X_val)[:, 1]
        pred = (prob >= 0.5).astype(int)

        fold_metrics[name]["AUC"].append(roc_auc_score(y_val, prob))
        fold_metrics[name]["F1"].append(f1_score(y_val, pred, zero_division=0))
        fold_metrics[name]["Precision"].append(precision_score(y_val, pred, zero_division=0))
        fold_metrics[name]["Recall"].append(recall_score(y_val, pred, zero_division=0))

metric_names = ["AUC", "F1", "Precision", "Recall"]
fig, axes = plt.subplots(1, len(metric_names), figsize=(16, 5))

for ax, metric in zip(axes, metric_names):
    data   = [fold_metrics[n][metric] for n in classifiers]
    labels = list(classifiers.keys())
    bp = ax.boxplot(data, labels=labels, patch_artist=True, notch=False,
                    medianprops=dict(color="black", linewidth=2))
    colors_box = [PALETTE["real"], PALETTE["real"], PALETTE["green"]]
    for patch, color in zip(bp["boxes"], colors_box[:len(bp["boxes"])]):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    for i, (vals, label) in enumerate(zip(data, labels)):
        ax.scatter([i+1]*len(vals), vals, s=40, zorder=3,
                   color=colors_box[i % len(colors_box)], alpha=0.9)
    ax.set_title(metric, fontsize=11, fontweight="bold")
    ax.set_ylabel("Score", fontsize=10)
    ax.set_ylim([max(0, min(min(d) for d in data) - 0.05), 1.02])
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="x", labelrotation=15, labelsize=8)

plt.suptitle(f"Per-Fold Metric Variance — FF++ ({N_SPLITS}-Fold CV)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/06_per_fold_variance.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 06_per_fold_variance.png")


# ─────────────────────────────────────────────────────────────
# SECTION 9 — SCORE DISTRIBUTIONS (Real vs Fake probability)
# ─────────────────────────────────────────────────────────────
print("Generating score distribution plots ...")

fig, axes = plt.subplots(1, n_clf, figsize=(6 * n_clf, 5))
if n_clf == 1:
    axes = [axes]

for ax, (name, proba) in zip(axes, cv_proba.items()):
    thresholds = np.arange(0.01, 1.0, THRESHOLD_STEP)
    f1s    = [f1_score(y, (proba >= t).astype(int), zero_division=0)
              for t in thresholds]
    best_t = thresholds[int(np.argmax(f1s))]

    real_scores = proba[y == 0]
    fake_scores = proba[y == 1]

    bins = np.linspace(0, 1, 50)
    ax.hist(real_scores, bins=bins, alpha=0.55, color=PALETTE["real"],
            label=f"Real  (n={len(real_scores)})", density=True)
    ax.hist(fake_scores, bins=bins, alpha=0.55, color=PALETTE["fake"],
            label=f"Fake  (n={len(fake_scores)})", density=True)
    ax.axvline(0.5,    color="gray",         lw=1.5, linestyle="--", label="θ=0.50")
    ax.axvline(best_t, color=PALETTE["dark"], lw=1.5, linestyle="-.", label=f"Optimal θ={best_t:.2f}")
    ax.set_xlabel("Predicted P(Fake)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title(f"{name}\nScore Distributions", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

plt.suptitle("Predicted Score Distributions — Real vs Fake (FF++ CV)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/07_score_distributions.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 07_score_distributions.png")


# ─────────────────────────────────────────────────────────────
# SECTION 10 — CALIBRATION CURVES
# ─────────────────────────────────────────────────────────────
print("Generating calibration curves ...")

fig, axes = plt.subplots(1, n_clf, figsize=(6 * n_clf, 5))
if n_clf == 1:
    axes = [axes]

for ax, (name, proba) in zip(axes, cv_proba.items()):
    frac_pos, mean_pred = calibration_curve(y, proba, n_bins=15,
                                            strategy="uniform")
    bs = brier_score_loss(y, proba)
    ax.plot(mean_pred, frac_pos, "s-", lw=2,
            color=PALETTE["real"], label=f"Model  (Brier={bs:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    ax.set_xlabel("Mean Predicted Probability", fontsize=11)
    ax.set_ylabel("Fraction of Positives", fontsize=11)
    ax.set_title(f"{name}\nCalibration Curve", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

plt.suptitle("Probability Calibration — FF++ In-Distribution",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/08_calibration_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 08_calibration_curves.png")


# ─────────────────────────────────────────────────────────────
# SECTION 11 — ERROR ANALYSIS  (FP / FN deep dive)
# ─────────────────────────────────────────────────────────────
print("Running error analysis ...")

# Use best classifier (LightGBM if available, else RandomForest)
best_clf_name = "LightGBM" if "LightGBM" in cv_proba else "RandomForest"
proba_best    = cv_proba[best_clf_name]

thresholds = np.arange(0.01, 1.0, THRESHOLD_STEP)
f1s        = [f1_score(y, (proba_best >= t).astype(int), zero_division=0)
              for t in thresholds]
best_t     = thresholds[int(np.argmax(f1s))]
pred_best  = (proba_best >= best_t).astype(int)

fp_mask = (pred_best == 1) & (y == 0)   # Real predicted as Fake
fn_mask = (pred_best == 0) & (y == 1)   # Fake predicted as Real
tp_mask = (pred_best == 1) & (y == 1)
tn_mask = (pred_best == 0) & (y == 0)

fp_scores = proba_best[fp_mask]
fn_scores = proba_best[fn_mask]
tp_scores = proba_best[tp_mask]
tn_scores = proba_best[tn_mask]

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
groups = [
    (tn_scores, "True Negatives (TN)\nReal → Real",   PALETTE["real"],   axes[0, 0]),
    (fp_scores, "False Positives (FP)\nReal → Fake",  PALETTE["orange"], axes[0, 1]),
    (fn_scores, "False Negatives (FN)\nFake → Real",  PALETTE["purple"], axes[1, 0]),
    (tp_scores, "True Positives (TP)\nFake → Fake",   PALETTE["green"],  axes[1, 1]),
]
for scores, title, color, ax in groups:
    if len(scores) == 0:
        ax.text(0.5, 0.5, "No samples", ha="center", va="center")
        ax.set_title(title, fontsize=10, fontweight="bold")
        continue
    ax.hist(scores, bins=30, color=color, alpha=0.7, edgecolor="white")
    ax.axvline(best_t, color="black", lw=1.5, linestyle="--",
               label=f"θ={best_t:.2f}")
    ax.set_xlabel("P(Fake)", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title(f"{title}\n(n={len(scores)})", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

plt.suptitle(f"Error Analysis — {best_clf_name}  (optimal θ={best_t:.2f})\nFF++ In-Distribution",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/09_error_analysis.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 09_error_analysis.png")


# ─────────────────────────────────────────────────────────────
# SECTION 12 — FEATURE IMPORTANCE (best classifier)
# ─────────────────────────────────────────────────────────────
print("Generating feature importance plot ...")

best_clf_obj = classifiers[best_clf_name]
best_clf_obj.fit(X_scaled, y)    # Refit on full data for importance

if hasattr(best_clf_obj, "feature_importances_"):
    importances = best_clf_obj.feature_importances_
elif hasattr(best_clf_obj, "coef_"):
    importances = np.abs(best_clf_obj.coef_[0])
else:
    importances = np.ones(len(feature_cols))

feat_imp = sorted(zip(feature_cols, importances),
                  key=lambda x: x[1], reverse=True)
top_n   = min(25, len(feat_imp))
fi_names = [x[0] for x in feat_imp[:top_n]]
fi_vals  = [x[1] for x in feat_imp[:top_n]]
fi_colors = [PALETTE["green"] if n.startswith("t_") else PALETTE["real"]
             for n in fi_names]

fig, ax = plt.subplots(figsize=(10, 8))
ax.barh(range(top_n), fi_vals[::-1], color=fi_colors[::-1],
        edgecolor="white", linewidth=0.5)
ax.set_yticks(range(top_n))
ax.set_yticklabels(fi_names[::-1], fontsize=8.5)
ax.set_xlabel("Feature Importance", fontsize=11)
ax.set_title(f"Feature Importance — {best_clf_name}  (Top {top_n})\n"
             f"Green=Temporal  Red=Spatial",
             fontsize=12, fontweight="bold")
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/10_feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 10_feature_importance.png")


# ─────────────────────────────────────────────────────────────
# SECTION 13 — SHAP  (if available)
# ─────────────────────────────────────────────────────────────
if HAS_SHAP:
    print("Generating SHAP summary plot ...")
    sample_idx = np.random.choice(len(X_scaled), min(400, len(X_scaled)), replace=False)
    X_shap     = X_scaled[sample_idx]

    try:
        explainer   = shap.TreeExplainer(best_clf_obj)
        shap_values = explainer.shap_values(X_shap)

        # LightGBM / RF may return list [real, fake] — take fake class
        if isinstance(shap_values, list):
            sv = shap_values[1]
        else:
            sv = shap_values

        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(sv, X_shap, feature_names=feature_cols,
                          show=False, plot_type="bar", max_display=20)
        plt.title(f"SHAP Feature Importance — {best_clf_name}",
                  fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/11_shap_bar.png", dpi=150, bbox_inches="tight")
        plt.close()

        fig, ax = plt.subplots(figsize=(10, 9))
        shap.summary_plot(sv, X_shap, feature_names=feature_cols,
                          show=False, max_display=20)
        plt.title(f"SHAP Beeswarm — {best_clf_name}",
                  fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/12_shap_beeswarm.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("Saved: 11_shap_bar.png + 12_shap_beeswarm.png")
    except Exception as e:
        print(f"  [WARN] SHAP failed: {e}")
else:
    print("  [SKIP] SHAP not installed — run: pip install shap")


# ─────────────────────────────────────────────────────────────
# SECTION 14 — t-SNE  (fixed)
# ─────────────────────────────────────────────────────────────
print("Running t-SNE ...")

sample_idx  = np.random.choice(len(X_scaled), TSNE_SAMPLE, replace=False)
X_sample    = X_scaled[sample_idx]
y_sample    = y[sample_idx]
prob_sample = cv_proba[best_clf_name][sample_idx]

tsne   = TSNE(n_components=2, random_state=RANDOM_STATE,
              perplexity=30, max_iter=1000)        # max_iter replaces n_iter
X_tsne = tsne.fit_transform(X_sample)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left — ground truth labels
for label, name_, color in [(0, "Real", PALETTE["real"]),
                              (1, "Fake", PALETTE["fake"])]:
    mask = y_sample == label
    axes[0].scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                    c=color, label=name_, alpha=0.45, s=14, edgecolors="none")
axes[0].set_title("t-SNE — Ground Truth Labels", fontsize=11, fontweight="bold")
axes[0].legend(fontsize=9)
axes[0].set_xlabel("Dim 1"); axes[0].set_ylabel("Dim 2")
axes[0].grid(alpha=0.2)

# Right — predicted probability colour map
sc = axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1],
                     c=prob_sample, cmap="RdBu_r",
                     alpha=0.55, s=14, edgecolors="none",
                     vmin=0, vmax=1)
plt.colorbar(sc, ax=axes[1], label="P(Fake)")
axes[1].set_title("t-SNE — Predicted P(Fake)", fontsize=11, fontweight="bold")
axes[1].set_xlabel("Dim 1"); axes[1].set_ylabel("Dim 2")
axes[1].grid(alpha=0.2)

plt.suptitle(f"t-SNE Feature Space — FF++ (n={TSNE_SAMPLE}, {best_clf_name})",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/13_tsne.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 13_tsne.png")


# ─────────────────────────────────────────────────────────────
# SECTION 15 — BOOTSTRAP METRIC DISTRIBUTION PLOT
# ─────────────────────────────────────────────────────────────
print("Generating bootstrap CI plots ...")

boot_metrics = ["AUC", "F1", "Precision", "Recall"]

def _boot_all(y_true, proba, n=500):
    """Return n bootstrap samples of AUC, F1, Prec, Rec at θ=0.5."""
    results = {m: [] for m in boot_metrics}
    for _ in range(n):
        idx = np.random.choice(len(y_true), len(y_true), replace=True)
        yt, yp = y_true[idx], proba[idx]
        if len(np.unique(yt)) < 2:
            continue
        pred = (yp >= 0.5).astype(int)
        results["AUC"].append(roc_auc_score(yt, yp))
        results["F1"].append(f1_score(yt, pred, zero_division=0))
        results["Precision"].append(precision_score(yt, pred, zero_division=0))
        results["Recall"].append(recall_score(yt, pred, zero_division=0))
    return results

fig, axes = plt.subplots(len(classifiers), len(boot_metrics),
                         figsize=(16, 4 * n_clf))
if n_clf == 1:
    axes = axes.reshape(1, -1)

for row_i, (name, proba) in enumerate(cv_proba.items()):
    boot_res = _boot_all(y, proba, n=500)
    for col_j, metric in enumerate(boot_metrics):
        ax    = axes[row_i, col_j]
        vals  = np.array(boot_res[metric])
        lo    = np.percentile(vals, 2.5)
        hi    = np.percentile(vals, 97.5)
        mu    = vals.mean()
        ax.hist(vals, bins=40, color=list(PALETTE.values())[row_i % 6],
                alpha=0.7, edgecolor="white")
        ax.axvline(mu, color="black", lw=2, linestyle="-",  label=f"Mean={mu:.3f}")
        ax.axvline(lo, color="black", lw=1, linestyle="--", label=f"95% CI")
        ax.axvline(hi, color="black", lw=1, linestyle="--")
        ax.set_title(f"{name}\n{metric}={mu:.3f} [{lo:.3f}–{hi:.3f}]",
                     fontsize=8.5, fontweight="bold")
        ax.set_xlabel(metric, fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

plt.suptitle("Bootstrap Distributions — 95% CI (FF++ In-Distribution)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/14_bootstrap_ci.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 14_bootstrap_ci.png")


# ─────────────────────────────────────────────────────────────
# FINAL PRINTED SUMMARY
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  VALIDATION COMPLETE — FINAL SUMMARY")
print("=" * 65)
print(f"\n  Dataset : {n_real} Real + {n_fake} Fake  ({n_real+n_fake} total)")
print(f"  Features: {len(feature_cols)}")
print(f"  CV Folds: {N_SPLITS}\n")
print(f"  {'Classifier':<22} {'AUC':>7} {'F1':>7} {'Prec':>7} {'Recall':>7} {'MCC':>7}")
print("  " + "─" * 57)
for row in summary_rows:
    print(f"  {row['Classifier']:<22} {row['ROC-AUC']:>7.4f} "
          f"{row['F1']:>7.4f} {row['Precision']:>7.4f} "
          f"{row['Recall']:>7.4f} {row['MCC']:>7.4f}")

print(f"\n  Outputs saved to → {OUTPUT_DIR}/")
print("=" * 65)

print("\n  Output files:")
files = sorted(os.listdir(OUTPUT_DIR))
for f in files:
    size = os.path.getsize(os.path.join(OUTPUT_DIR, f))
    print(f"    {f:<50} {size/1024:>6.1f} KB")