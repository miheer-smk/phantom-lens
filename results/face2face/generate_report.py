#!/usr/bin/env python3
"""
Generate a detailed PDF report for Phantom Lens Face2Face experiment.
Includes hyperparameters, all metrics, Cohen's d, ROC curves, fold details.
"""

import json
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch
from matplotlib import rcParams
import warnings
warnings.filterwarnings("ignore")

# ── styling ──────────────────────────────────────────────────────────────────
rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 150,
})

COLORS = {
    "LR":  "#4878CF",
    "RF":  "#6ACC65",
    "LGB": "#D65F5F",
    "large":  "#e24b4a",
    "medium": "#ef9f27",
    "small":  "#378add",
    "neg":    "#aaaaaa",
    "header": "#2C3E50",
    "row_odd":"#F2F5F9",
    "row_even":"#FFFFFF",
}

RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_PDF     = os.path.join(RESULTS_DIR, "phantom_lens_face2face_report.pdf")

# ── load data ─────────────────────────────────────────────────────────────────
with open(os.path.join(RESULTS_DIR, "results.json")) as f:
    R = json.load(f)

# Load models and scaler to get actual sklearn params
def load_pkl(name):
    p = os.path.join(RESULTS_DIR, f"model_{name}.pkl")
    with open(p, "rb") as f:
        return pickle.load(f)

lr_model  = load_pkl("LogisticRegression")
rf_model  = load_pkl("RandomForest")
lgb_model = load_pkl("LightGBM")

# Load feature data for ROC curves
feat_dir = os.path.join(RESULTS_DIR, "../../features")
train_df = pd.concat([
    pd.read_csv(os.path.join(feat_dir, "ffpp_real.csv")),
    pd.read_csv(os.path.join(feat_dir, "ffpp_fake.csv")),
], ignore_index=True)
test_df = pd.concat([
    pd.read_csv(os.path.join(feat_dir, "ffpp_real.csv")),
    pd.read_csv(os.path.join(feat_dir, "ffpp_face2face.csv")),
], ignore_index=True)

feature_cols = sorted([c for c in train_df.columns if c.startswith("s_") or c.startswith("t_")])
X_train = train_df[feature_cols].values.astype(np.float64)
y_train = train_df["label"].values.astype(int)
X_test  = test_df[feature_cols].values.astype(np.float64)
y_test  = test_df["label"].values.astype(int)

scaler_path = os.path.join(RESULTS_DIR, "scaler.pkl")
with open(scaler_path, "rb") as f:
    scaler = pickle.load(f)
X_train_sc = scaler.transform(X_train)
X_test_sc  = scaler.transform(X_test)

from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix

# ── helper: table drawing ─────────────────────────────────────────────────────
def draw_table(ax, headers, rows, col_widths=None, title=None,
               header_color=COLORS["header"], fontsize=8.5):
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=11, fontweight="bold", pad=8,
                     color=COLORS["header"])
    n_cols = len(headers)
    n_rows = len(rows)
    if col_widths is None:
        col_widths = [1.0 / n_cols] * n_cols

    # Normalise widths to [0,1]
    total = sum(col_widths)
    cw = [w / total for w in col_widths]

    row_h = 0.85 / (n_rows + 1)          # +1 for header
    y0    = 0.93

    # header
    x = 0.0
    for i, (h, w) in enumerate(zip(headers, cw)):
        ax.add_patch(FancyBboxPatch((x, y0 - row_h), w - 0.002, row_h,
                                    boxstyle="square,pad=0",
                                    facecolor=header_color, edgecolor="white",
                                    lw=0.5, transform=ax.transAxes, clip_on=False))
        ax.text(x + w / 2, y0 - row_h / 2, h,
                ha="center", va="center", fontsize=fontsize,
                fontweight="bold", color="white", transform=ax.transAxes)
        x += w

    # rows
    for ri, row in enumerate(rows):
        bg = COLORS["row_odd"] if ri % 2 == 0 else COLORS["row_even"]
        y = y0 - (ri + 2) * row_h
        x = 0.0
        for ci, (val, w) in enumerate(zip(row, cw)):
            ax.add_patch(FancyBboxPatch((x, y), w - 0.002, row_h,
                                        boxstyle="square,pad=0",
                                        facecolor=bg, edgecolor="#dddddd",
                                        lw=0.3, transform=ax.transAxes, clip_on=False))
            ax.text(x + w / 2, y + row_h / 2, str(val),
                    ha="center", va="center", fontsize=fontsize,
                    transform=ax.transAxes)
            x += w


def add_page_header(fig, title, subtitle=""):
    fig.text(0.5, 0.97, title, ha="center", va="top",
             fontsize=14, fontweight="bold", color=COLORS["header"])
    if subtitle:
        fig.text(0.5, 0.94, subtitle, ha="center", va="top",
                 fontsize=9, color="#555555")


def add_footer(fig, page_num, total_pages):
    fig.text(0.5, 0.015, f"Phantom Lens V2  |  Face2Face Cross-Dataset Experiment  |  Page {page_num}/{total_pages}",
             ha="center", va="bottom", fontsize=7.5, color="#888888")
    fig.text(0.015, 0.015, "PRISM V3 Physics Features  |  FF++ Dataset",
             ha="left", va="bottom", fontsize=7, color="#aaaaaa")


# ═══════════════════════════════════════════════════════════════════════════════
# BUILD PDF
# ═══════════════════════════════════════════════════════════════════════════════
TOTAL_PAGES = 9
print(f"Generating PDF report → {OUT_PDF}")

with PdfPages(OUT_PDF) as pdf:

    # ─── PAGE 1: TITLE PAGE ────────────────────────────────────────────────
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor("#1B2631")

    fig.text(0.5, 0.72, "PHANTOM LENS V2", ha="center", fontsize=32,
             fontweight="bold", color="white")
    fig.text(0.5, 0.63, "PRISM Physics Feature Extractor", ha="center",
             fontsize=18, color="#AED6F1")
    fig.text(0.5, 0.55, "Cross-Dataset Deepfake Detection Report",
             ha="center", fontsize=14, color="#85C1E9")

    fig.add_artist(plt.Line2D([0.12, 0.88], [0.50, 0.50],
                               color="#AED6F1", lw=1.5, transform=fig.transFigure))

    meta = [
        ("Train Set",    "FF++ Deepfakes (fake, 957)  +  FF++ Real (960)"),
        ("Test Set",     "FF++ Face2Face (fake, 960)  +  FF++ Real (960)"),
        ("Features",     "50 PRISM Physics Features (13 Spatial + 37 Temporal)"),
        ("Classifiers",  "Logistic Regression  |  Random Forest  |  LightGBM"),
        ("CV Strategy",  "5-Fold Stratified Cross-Validation"),
        ("GPU Backend",  "CuPy 14.0.1 on NVIDIA GB10"),
    ]
    for i, (k, v) in enumerate(meta):
        y = 0.44 - i * 0.055
        fig.text(0.22, y, f"{k}:", ha="left", fontsize=10,
                 fontweight="bold", color="#AED6F1")
        fig.text(0.42, y, v, ha="left", fontsize=10, color="white")

    fig.text(0.5, 0.06, "Generated by Phantom Lens Pipeline",
             ha="center", fontsize=9, color="#888888")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    # ─── PAGE 2: EXPERIMENT OVERVIEW + DATASET TABLE ───────────────────────
    fig = plt.figure(figsize=(11, 8.5))
    add_page_header(fig, "Experiment Overview", "Dataset Statistics & Evaluation Protocol")
    add_footer(fig, 2, TOTAL_PAGES)

    gs = gridspec.GridSpec(3, 2, figure=fig,
                           top=0.88, bottom=0.08, hspace=0.55, wspace=0.35)

    # Dataset table
    ax_ds = fig.add_subplot(gs[0, :])
    draw_table(ax_ds,
        headers=["Split", "Source", "Manipulation", "Label", "Videos", "Features"],
        rows=[
            ["Train", "FF++ (c23)", "Deepfakes",   "1 (Fake)",  "1001 total / 957 extracted", "50"],
            ["Train", "FF++ (c23)", "Original",    "0 (Real)",  "1000 total / 960 extracted", "50"],
            ["Test",  "FF++ (c23)", "Face2Face",   "1 (Fake)",  "1000 total / 960 extracted", "50"],
            ["Test",  "FF++ (c23)", "Original",    "0 (Real)",  "1000 total / 960 extracted", "50"],
        ],
        col_widths=[0.9, 1.0, 1.1, 1.0, 1.6, 0.7],
        title="Dataset Summary",
    )

    # Evaluation protocol
    ax_pr = fig.add_subplot(gs[1, :])
    draw_table(ax_pr,
        headers=["Step", "Description", "Details"],
        rows=[
            ["1", "Feature Extraction",    "precompute_features_best_gpu.py  |  max_frames=300  |  workers=4–8"],
            ["2", "Train/Test Split",       "Train: Deepfakes+Real  →  Test: Face2Face+Real (cross-manipulation)"],
            ["3", "Feature Scaling",        "StandardScaler (fit on train, transform test)"],
            ["4", "Cross-Validation",       "StratifiedKFold (n_splits=5, shuffle=True, random_state=42)"],
            ["5", "Model Selection",        "Best model = highest Test AUC on cross-dataset evaluation"],
            ["6", "Effect Size Analysis",   "Cohen's d per feature on scaled test set (real vs fake)"],
        ],
        col_widths=[0.5, 1.5, 3.5],
        title="Evaluation Protocol",
    )

    # Summary numbers
    ax_sum = fig.add_subplot(gs[2, :])
    draw_table(ax_sum,
        headers=["Metric", "LogisticRegression", "RandomForest", "LightGBM"],
        rows=[
            ["CV AUC (mean ± std)",
             f"{R['LogisticRegression']['cv_auc_mean']:.4f} ± {R['LogisticRegression']['cv_auc_std']:.4f}",
             f"{R['RandomForest']['cv_auc_mean']:.4f} ± {R['RandomForest']['cv_auc_std']:.4f}",
             f"{R['LightGBM']['cv_auc_mean']:.4f} ± {R['LightGBM']['cv_auc_std']:.4f}"],
            ["Test AUC (cross-dataset)",
             f"{R['LogisticRegression']['test_auc']:.4f}",
             f"{R['RandomForest']['test_auc']:.4f}",
             f"{R['LightGBM']['test_auc']:.4f}"],
            ["Test Accuracy",
             f"{R['LogisticRegression']['test_acc']:.4f}",
             f"{R['RandomForest']['test_acc']:.4f}",
             f"{R['LightGBM']['test_acc']:.4f}"],
            ["Best Model", "", "✓  RandomForest  AUC=0.6809", ""],
        ],
        col_widths=[2.0, 1.8, 1.8, 1.8],
        title="Summary Results",
    )

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    # ─── PAGE 3: HYPERPARAMETERS ────────────────────────────────────────────
    fig = plt.figure(figsize=(11, 8.5))
    add_page_header(fig, "Classifier Hyperparameters",
                    "All parameters used for training (from sklearn / lightgbm)")
    add_footer(fig, 3, TOTAL_PAGES)

    gs = gridspec.GridSpec(3, 1, figure=fig,
                           top=0.88, bottom=0.08, hspace=0.55)

    # LogisticRegression params
    ax_lr = fig.add_subplot(gs[0])
    lr_params = lr_model.get_params()
    draw_table(ax_lr,
        headers=["Parameter", "Value", "Description"],
        rows=[
            ["C",             str(lr_params.get("C", 1.0)),            "Inverse regularisation strength (L2)"],
            ["penalty",       str(lr_params.get("penalty","l2")),       "Regularisation type"],
            ["solver",        str(lr_params.get("solver","lbfgs")),     "Optimisation algorithm"],
            ["max_iter",      str(lr_params.get("max_iter",2000)),      "Max iterations for convergence"],
            ["class_weight",  str(lr_params.get("class_weight","balanced")), "Handles class imbalance"],
            ["random_state",  str(lr_params.get("random_state",42)),    "Reproducibility seed"],
            ["multi_class",   str(lr_params.get("multi_class","auto")), "Multi-class strategy"],
        ],
        col_widths=[1.5, 1.2, 3.5],
        title="Logistic Regression Hyperparameters",
        header_color="#2471A3",
    )

    # RandomForest params
    ax_rf = fig.add_subplot(gs[1])
    rf_params = rf_model.get_params()
    draw_table(ax_rf,
        headers=["Parameter", "Value", "Description"],
        rows=[
            ["n_estimators",    str(rf_params.get("n_estimators",200)),       "Number of decision trees"],
            ["max_depth",       str(rf_params.get("max_depth",8)),            "Maximum depth of each tree"],
            ["min_samples_leaf",str(rf_params.get("min_samples_leaf",10)),    "Min samples at leaf node"],
            ["class_weight",    str(rf_params.get("class_weight","balanced")), "Handles class imbalance"],
            ["n_jobs",          str(rf_params.get("n_jobs",-1)),              "Parallel jobs (-1 = all cores)"],
            ["random_state",    str(rf_params.get("random_state",42)),        "Reproducibility seed"],
            ["criterion",       str(rf_params.get("criterion","gini")),       "Split quality measure"],
        ],
        col_widths=[1.5, 1.2, 3.5],
        title="Random Forest Hyperparameters",
        header_color="#1E8449",
    )

    # LightGBM params
    ax_lgb = fig.add_subplot(gs[2])
    lgb_params = lgb_model.get_params()
    draw_table(ax_lgb,
        headers=["Parameter", "Value", "Description"],
        rows=[
            ["n_estimators",      str(lgb_params.get("n_estimators",200)),      "Number of boosting rounds"],
            ["max_depth",         str(lgb_params.get("max_depth",6)),           "Maximum tree depth"],
            ["learning_rate",     str(lgb_params.get("learning_rate",0.05)),    "Shrinkage rate per step"],
            ["num_leaves",        str(lgb_params.get("num_leaves",31)),         "Max leaves per tree"],
            ["min_child_samples", str(lgb_params.get("min_child_samples",20)),  "Min data in one leaf"],
            ["class_weight",      str(lgb_params.get("class_weight","balanced")), "Handles class imbalance"],
            ["random_state",      str(lgb_params.get("random_state",42)),       "Reproducibility seed"],
        ],
        col_widths=[1.5, 1.2, 3.5],
        title="LightGBM Hyperparameters",
        header_color="#922B21",
    )

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    # ─── PAGE 4: CROSS-VALIDATION RESULTS PER FOLD ─────────────────────────
    fig = plt.figure(figsize=(11, 8.5))
    add_page_header(fig, "Cross-Validation Results",
                    "5-Fold Stratified CV on Training Set (Deepfakes + Real)")
    add_footer(fig, 4, TOTAL_PAGES)

    gs = gridspec.GridSpec(2, 2, figure=fig,
                           top=0.88, bottom=0.08, hspace=0.55, wspace=0.3)

    # Per-fold AUC table
    ax_fold = fig.add_subplot(gs[0, :])
    fold_rows = []
    for fold in range(5):
        fold_rows.append([
            f"Fold {fold+1}",
            f"{R['LogisticRegression']['cv_fold_aucs'][fold]:.4f}",
            f"{R['RandomForest']['cv_fold_aucs'][fold]:.4f}",
            f"{R['LightGBM']['cv_fold_aucs'][fold]:.4f}",
        ])
    fold_rows.append([
        "Mean ± Std",
        f"{R['LogisticRegression']['cv_auc_mean']:.4f} ± {R['LogisticRegression']['cv_auc_std']:.4f}",
        f"{R['RandomForest']['cv_auc_mean']:.4f} ± {R['RandomForest']['cv_auc_std']:.4f}",
        f"{R['LightGBM']['cv_auc_mean']:.4f} ± {R['LightGBM']['cv_auc_std']:.4f}",
    ])
    draw_table(ax_fold,
        headers=["Fold", "LogisticRegression AUC", "RandomForest AUC", "LightGBM AUC"],
        rows=fold_rows,
        col_widths=[0.8, 2.0, 2.0, 2.0],
        title="AUC per Fold",
    )

    # CV AUC boxplot
    ax_box = fig.add_subplot(gs[1, 0])
    data = [
        R['LogisticRegression']['cv_fold_aucs'],
        R['RandomForest']['cv_fold_aucs'],
        R['LightGBM']['cv_fold_aucs'],
    ]
    bp = ax_box.boxplot(data, patch_artist=True, widths=0.5,
                        medianprops=dict(color="white", lw=2))
    for patch, col in zip(bp['boxes'], [COLORS["LR"], COLORS["RF"], COLORS["LGB"]]):
        patch.set_facecolor(col)
    ax_box.set_xticks([1, 2, 3])
    ax_box.set_xticklabels(["LR", "RF", "LGB"])
    ax_box.set_ylabel("AUC")
    ax_box.set_title("CV AUC Distribution (5 Folds)", fontweight="bold")
    ax_box.set_ylim(0.93, 1.01)
    ax_box.grid(axis="y", alpha=0.4)

    # CV AUC per fold line chart
    ax_line = fig.add_subplot(gs[1, 1])
    folds = list(range(1, 6))
    for name, color, key in [
        ("LR",  COLORS["LR"],  "LogisticRegression"),
        ("RF",  COLORS["RF"],  "RandomForest"),
        ("LGB", COLORS["LGB"], "LightGBM"),
    ]:
        ax_line.plot(folds, R[key]['cv_fold_aucs'], marker='o',
                     label=name, color=color, lw=2, markersize=6)
    ax_line.set_xlabel("Fold")
    ax_line.set_ylabel("AUC")
    ax_line.set_title("CV AUC per Fold", fontweight="bold")
    ax_line.set_xticks(folds)
    ax_line.legend(fontsize=8)
    ax_line.set_ylim(0.93, 1.01)
    ax_line.grid(alpha=0.3)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    # ─── PAGE 5: TEST SET METRICS TABLE ────────────────────────────────────
    fig = plt.figure(figsize=(11, 8.5))
    add_page_header(fig, "Test Set Evaluation",
                    "Cross-Dataset: Trained on Deepfakes → Tested on Face2Face")
    add_footer(fig, 5, TOTAL_PAGES)

    gs = gridspec.GridSpec(3, 1, figure=fig,
                           top=0.88, bottom=0.08, hspace=0.55)

    # Main metrics
    ax_m = fig.add_subplot(gs[0])
    draw_table(ax_m,
        headers=["Metric", "LogisticRegression", "RandomForest", "LightGBM"],
        rows=[
            ["Test AUC",
             f"{R['LogisticRegression']['test_auc']:.4f}",
             f"{R['RandomForest']['test_auc']:.4f} ★ Best",
             f"{R['LightGBM']['test_auc']:.4f}"],
            ["Test Accuracy",
             f"{R['LogisticRegression']['test_acc']:.4f}",
             f"{R['RandomForest']['test_acc']:.4f}",
             f"{R['LightGBM']['test_acc']:.4f}"],
            ["Precision (Fake)",
             f"{R['LogisticRegression']['test_precision']:.4f}",
             f"{R['RandomForest']['test_precision']:.4f}",
             f"{R['LightGBM']['test_precision']:.4f}"],
            ["Recall (Fake)",
             f"{R['LogisticRegression']['test_recall']:.4f}",
             f"{R['RandomForest']['test_recall']:.4f}",
             f"{R['LightGBM']['test_recall']:.4f}"],
            ["F1-Score (Fake)",
             f"{R['LogisticRegression']['test_f1']:.4f}",
             f"{R['RandomForest']['test_f1']:.4f}",
             f"{R['LightGBM']['test_f1']:.4f}"],
        ],
        col_widths=[1.6, 2.0, 2.0, 2.0],
        title="Cross-Dataset Test Set Metrics (Face2Face, 1920 samples)",
    )

    # Per-class classification reports
    for row_idx, (clf, model) in enumerate(
            [("LogisticRegression", lr_model),
             ("RandomForest", rf_model)], start=1):
        ax_c = fig.add_subplot(gs[row_idx])
        y_prob = model.predict_proba(X_test_sc)[:, 1]
        y_pred = model.predict(X_test_sc)
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        rows_clf = [
            ["Real (0)",  f"{tn/(tn+fp):.4f}", f"{tn/(tn+fn):.4f}",
             f"{2*(tn/(tn+fp))*(tn/(tn+fn))/((tn/(tn+fp))+(tn/(tn+fn))+1e-8):.4f}",
             str(tn+fn)],
            ["Fake (1)",  f"{tp/(tp+fp):.4f}", f"{tp/(tp+fn):.4f}",
             f"{2*(tp/(tp+fp))*(tp/(tp+fn))/((tp/(tp+fp))+(tp/(tp+fn))+1e-8):.4f}",
             str(tp+fp)],
            ["Macro avg",
             f"{((tn/(tn+fp))+(tp/(tp+fp)))/2:.4f}",
             f"{((tn/(tn+fn))+(tp/(tp+fn)))/2:.4f}",
             f"{(2*(tn/(tn+fp))*(tn/(tn+fn))/((tn/(tn+fp))+(tn/(tn+fn))+1e-8) + 2*(tp/(tp+fp))*(tp/(tp+fn))/((tp/(tp+fp))+(tp/(tp+fn))+1e-8))/2:.4f}",
             "1920"],
            [f"TP={tp}  FP={fp}  TN={tn}  FN={fn}", "", "", "", ""],
        ]
        draw_table(ax_c,
            headers=["Class", "Precision", "Recall", "F1-Score", "Support"],
            rows=rows_clf,
            col_widths=[2.5, 1.3, 1.3, 1.3, 1.0],
            title=f"Classification Report — {clf}",
        )

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    # ─── PAGE 6: ROC CURVES (all 3) + Confusion Matrices ──────────────────
    fig = plt.figure(figsize=(11, 8.5))
    add_page_header(fig, "ROC Curves & Confusion Matrices",
                    "Cross-Dataset Evaluation on Face2Face Test Set")
    add_footer(fig, 6, TOTAL_PAGES)

    gs = gridspec.GridSpec(2, 3, figure=fig,
                           top=0.88, bottom=0.08, hspace=0.45, wspace=0.35)

    models_info = [
        ("LogisticRegression", lr_model,  COLORS["LR"]),
        ("RandomForest",       rf_model,  COLORS["RF"]),
        ("LightGBM",           lgb_model, COLORS["LGB"]),
    ]

    for col, (name, model, color) in enumerate(models_info):
        y_prob = model.predict_proba(X_test_sc)[:, 1]
        y_pred = model.predict(X_test_sc)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_val = roc_auc_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred)

        # ROC
        ax_roc = fig.add_subplot(gs[0, col])
        ax_roc.plot(fpr, tpr, color=color, lw=2, label=f"AUC={auc_val:.4f}")
        ax_roc.fill_between(fpr, tpr, alpha=0.1, color=color)
        ax_roc.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
        ax_roc.set_xlabel("FPR")
        ax_roc.set_ylabel("TPR")
        short = {"LogisticRegression": "LR", "RandomForest": "RF"}.get(name, name)
        ax_roc.set_title(f"{short}\nAUC = {auc_val:.4f}", fontweight="bold")
        ax_roc.legend(fontsize=7)
        ax_roc.grid(alpha=0.3)

        # Confusion matrix
        ax_cm = fig.add_subplot(gs[1, col])
        im = ax_cm.imshow(cm, cmap="Blues", aspect="auto")
        for i in range(2):
            for j in range(2):
                ax_cm.text(j, i, str(cm[i, j]), ha="center", va="center",
                           fontsize=12, fontweight="bold",
                           color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax_cm.set_xticks([0, 1])
        ax_cm.set_yticks([0, 1])
        ax_cm.set_xticklabels(["Pred Real", "Pred Fake"], fontsize=8)
        ax_cm.set_yticklabels(["True Real", "True Fake"], fontsize=8)
        ax_cm.set_title(f"Confusion Matrix\n{short}", fontweight="bold")
        plt.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    # ─── PAGE 7: COMBINED ROC + CV vs TEST AUC BAR ─────────────────────────
    fig = plt.figure(figsize=(11, 8.5))
    add_page_header(fig, "Performance Comparison",
                    "CV vs Cross-Dataset AUC Comparison")
    add_footer(fig, 7, TOTAL_PAGES)

    gs = gridspec.GridSpec(1, 2, figure=fig,
                           top=0.86, bottom=0.12, hspace=0.4, wspace=0.35)

    # Combined ROC
    ax_roc_all = fig.add_subplot(gs[0])
    for name, model, color in models_info:
        y_prob = model.predict_proba(X_test_sc)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_val = roc_auc_score(y_test, y_prob)
        short = {"LogisticRegression": "LR", "RandomForest": "RF"}.get(name, name)
        ax_roc_all.plot(fpr, tpr, color=color, lw=2.5, label=f"{short} (AUC={auc_val:.4f})")
    ax_roc_all.fill_between([0, 1], [0, 1], alpha=0.05, color="gray")
    ax_roc_all.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random")
    ax_roc_all.set_xlabel("False Positive Rate")
    ax_roc_all.set_ylabel("True Positive Rate")
    ax_roc_all.set_title("ROC Curves — All Models (Face2Face Test)", fontweight="bold")
    ax_roc_all.legend(fontsize=9)
    ax_roc_all.grid(alpha=0.3)

    # CV vs Test AUC bar chart
    ax_bar = fig.add_subplot(gs[1])
    names_short = ["LR", "RF", "LGB"]
    cv_aucs  = [R[k]['cv_auc_mean']  for k in ["LogisticRegression", "RandomForest", "LightGBM"]]
    cv_stds  = [R[k]['cv_auc_std']   for k in ["LogisticRegression", "RandomForest", "LightGBM"]]
    tst_aucs = [R[k]['test_auc']     for k in ["LogisticRegression", "RandomForest", "LightGBM"]]
    colors_bar = [COLORS["LR"], COLORS["RF"], COLORS["LGB"]]

    x = np.arange(3)
    w = 0.35
    b1 = ax_bar.bar(x - w/2, cv_aucs, w, yerr=cv_stds, capsize=4,
                    color=colors_bar, alpha=0.85, label="CV AUC (train / Deepfakes)")
    b2 = ax_bar.bar(x + w/2, tst_aucs, w,
                    color=colors_bar, alpha=0.45, label="Test AUC (cross-dataset / Face2Face)",
                    hatch="//")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(names_short)
    ax_bar.set_ylabel("AUC")
    ax_bar.set_title("CV AUC vs Cross-Dataset Test AUC", fontweight="bold")
    ax_bar.set_ylim(0.0, 1.08)
    ax_bar.axhline(0.5, color="gray", lw=1, linestyle="--", alpha=0.6, label="Random (0.5)")
    ax_bar.legend(fontsize=7.5)
    ax_bar.grid(axis="y", alpha=0.3)
    for bar, v in zip(b1, cv_aucs):
        ax_bar.text(bar.get_x() + bar.get_width()/2, v + 0.01, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=7.5)
    for bar, v in zip(b2, tst_aucs):
        ax_bar.text(bar.get_x() + bar.get_width()/2, v + 0.01, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=7.5)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    # ─── PAGE 8: COHEN'S d EFFECT SIZE ─────────────────────────────────────
    fig = plt.figure(figsize=(11, 8.5))
    add_page_header(fig, "Feature Discriminative Power — Cohen's d",
                    "Effect size of each feature (Real vs Fake) on scaled test set")
    add_footer(fig, 8, TOTAL_PAGES)

    gs = gridspec.GridSpec(1, 2, figure=fig,
                           top=0.88, bottom=0.06, hspace=0.4, wspace=0.4)

    # Cohen's d bar chart (all 50 features)
    ax_d = fig.add_subplot(gs[0])
    sorted_d = sorted(R["cohens_d"].items(), key=lambda x: x[1], reverse=True)
    names_d  = [x[0].replace("t_","t:").replace("s_","s:") for x in sorted_d]
    vals_d   = [x[1] for x in sorted_d]
    bar_cols = [COLORS["large"] if v > 0.8
                else COLORS["medium"] if v > 0.5
                else COLORS["small"] if v > 0.2
                else COLORS["neg"] for v in vals_d]

    ax_d.barh(range(len(vals_d)), vals_d[::-1], color=bar_cols[::-1])
    ax_d.set_yticks(range(len(names_d)))
    ax_d.set_yticklabels(names_d[::-1], fontsize=6.5)
    ax_d.axvline(0.8, color=COLORS["large"],  lw=1.5, ls="--", alpha=0.7, label="Large (>0.8)")
    ax_d.axvline(0.5, color=COLORS["medium"], lw=1.5, ls="--", alpha=0.7, label="Medium (>0.5)")
    ax_d.axvline(0.2, color=COLORS["small"],  lw=1.5, ls="--", alpha=0.7, label="Small (>0.2)")
    ax_d.set_xlabel("Cohen's d")
    ax_d.set_title("All 50 Features — Cohen's d", fontweight="bold")
    ax_d.legend(fontsize=7, loc="lower right")
    ax_d.grid(axis="x", alpha=0.3)

    # Cohen's d table (top 25)
    ax_dt = fig.add_subplot(gs[1])
    top25 = sorted_d[:25]
    def effect_label(v):
        if v > 0.8: return "Large"
        if v > 0.5: return "Medium"
        if v > 0.2: return "Small"
        return "Negligible"

    draw_table(ax_dt,
        headers=["#", "Feature", "Cohen's d", "Effect"],
        rows=[[str(i+1), n, f"{v:.4f}", effect_label(v)]
              for i, (n, v) in enumerate(top25)],
        col_widths=[0.3, 2.0, 0.8, 0.8],
        title="Top 25 Features by Effect Size",
        fontsize=7.5,
    )

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    # ─── PAGE 9: SUMMARY & CONCLUSIONS ─────────────────────────────────────
    fig = plt.figure(figsize=(11, 8.5))
    add_page_header(fig, "Summary & Conclusions",
                    "Phantom Lens V2 — Face2Face Cross-Dataset Experiment")
    add_footer(fig, 9, TOTAL_PAGES)

    gs = gridspec.GridSpec(2, 2, figure=fig,
                           top=0.88, bottom=0.08, hspace=0.55, wspace=0.35)

    # Final results table
    ax_final = fig.add_subplot(gs[0, :])
    draw_table(ax_final,
        headers=["Model", "CV AUC", "CV Std", "CV Acc",
                 "Test AUC", "Test Acc", "Precision", "Recall", "F1"],
        rows=[
            ["LogisticRegression",
             f"{R['LogisticRegression']['cv_auc_mean']:.4f}",
             f"±{R['LogisticRegression']['cv_auc_std']:.4f}",
             f"{R['LogisticRegression']['cv_acc_mean']:.4f}",
             f"{R['LogisticRegression']['test_auc']:.4f}",
             f"{R['LogisticRegression']['test_acc']:.4f}",
             f"{R['LogisticRegression']['test_precision']:.4f}",
             f"{R['LogisticRegression']['test_recall']:.4f}",
             f"{R['LogisticRegression']['test_f1']:.4f}"],
            ["RandomForest ★",
             f"{R['RandomForest']['cv_auc_mean']:.4f}",
             f"±{R['RandomForest']['cv_auc_std']:.4f}",
             f"{R['RandomForest']['cv_acc_mean']:.4f}",
             f"{R['RandomForest']['test_auc']:.4f}",
             f"{R['RandomForest']['test_acc']:.4f}",
             f"{R['RandomForest']['test_precision']:.4f}",
             f"{R['RandomForest']['test_recall']:.4f}",
             f"{R['RandomForest']['test_f1']:.4f}"],
            ["LightGBM",
             f"{R['LightGBM']['cv_auc_mean']:.4f}",
             f"±{R['LightGBM']['cv_auc_std']:.4f}",
             f"{R['LightGBM']['cv_acc_mean']:.4f}",
             f"{R['LightGBM']['test_auc']:.4f}",
             f"{R['LightGBM']['test_acc']:.4f}",
             f"{R['LightGBM']['test_precision']:.4f}",
             f"{R['LightGBM']['test_recall']:.4f}",
             f"{R['LightGBM']['test_f1']:.4f}"],
        ],
        col_widths=[1.6, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8],
        title="Complete Results Summary  (★ = Best Cross-Dataset AUC)",
        fontsize=8,
    )

    # Key findings
    ax_kf = fig.add_subplot(gs[1, 0])
    draw_table(ax_kf,
        headers=["Finding", "Detail"],
        rows=[
            ["Best Model",          "RandomForest (Test AUC 0.6809)"],
            ["Best CV Model",       "LightGBM (CV AUC 0.9802)"],
            ["CV → Test Gap",       "~0.30 AUC drop (expected: cross-manipulation)"],
            ["Top Feature",         "t_coupling_consistency (Cohen d=0.62, Medium)"],
            ["Physics Pillars",     "Geometry & Temporal stability most transferable"],
            ["LightGBM Precision",  "1.0000 — zero false positives but very low recall"],
            ["Train samples",       "1917  (960 real + 957 Deepfakes fake)"],
            ["Test samples",        "1920  (960 real + 960 Face2Face fake)"],
        ],
        col_widths=[1.8, 2.5],
        title="Key Findings",
        fontsize=8,
    )

    # Pillar contribution
    ax_pil = fig.add_subplot(gs[1, 1])
    pillar_map = {
        "P1 Noise Physics":      ["s_noise_vmr","s_noise_res_std","s_noise_hf_ratio"],
        "P2 PRNU":               ["s_prnu_energy","s_prnu_face_periph"],
        "P4 Shadow/Light":       ["s_shadow_score","s_face_bg_diff"],
        "P6 Compression":        ["s_benford_dev","s_block_artifact","s_dbl_compress"],
        "P8 Blur":               ["s_blur_mag"],
        "P9 Optical Flow":       ["s_flow_mag","s_flow_dir_consist"],
        "T1 Noise Stability":    ["t_noise_temporal_corr","t_noise_corr_std","t_noise_spectral_entropy"],
        "T2 rPPG Cardiac":       ["t_rppg_snr","t_rppg_peak_prominence","t_rppg_interregion_corr","t_rppg_harmonic_ratio"],
        "T4 SSIM Stability":     ["t_face_ssim_mean","t_face_ssim_std","t_face_ssim_min"],
        "T6 Landmarks":          ["t_landmark_jitter","t_landmark_accel_var","t_landmark_velocity_autocorr","t_jaw_chin_rigidity"],
        "T7 Rigid Geometry":     ["t_rigid_dist_var","t_interpupillary_std","t_nose_bridge_std"],
        "T13 Blur Coupling":     ["t_motion_blur_coupling","t_coupling_consistency"],
    }
    pillar_rows = []
    for pname, feats in pillar_map.items():
        mean_d = np.mean([R["cohens_d"].get(f, 0) for f in feats])
        pillar_rows.append([pname, str(len(feats)), f"{mean_d:.4f}",
                            effect_label(mean_d)])
    pillar_rows.sort(key=lambda x: float(x[2]), reverse=True)
    draw_table(ax_pil,
        headers=["Physics Pillar", "# Feats", "Mean d", "Effect"],
        rows=pillar_rows,
        col_widths=[1.8, 0.6, 0.7, 0.8],
        title="Physics Pillar Mean Effect Size",
        fontsize=7.5,
    )

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

print(f"\nPDF report saved → {OUT_PDF}")
print(f"Pages: {TOTAL_PAGES}")
