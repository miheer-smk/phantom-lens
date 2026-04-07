#!/usr/bin/env python3
"""
Experiment 2 — Compression comparison (DeepFakes).

Checks availability of c0, c23, c40 compression levels.
Extracts features for missing CSVs, trains classifiers,
produces comparison bar chart. Gracefully handles missing data.
"""

import os, json, subprocess, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, matthews_corrcoef
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
import lightgbm as lgb
from scipy import stats

warnings.filterwarnings("ignore")

BASE   = "/home/iiitn/Miheer_project_FE/phantom-lens"
OUTDIR = f"{BASE}/results/exp2"
os.makedirs(OUTDIR, exist_ok=True)

FFPP_ROOT = "/home/iiitn/Miheer_Project/data/ffpp_official"
VENV_PY   = "/home/iiitn/Miheer_Project/phantomlens_linux_env/bin/python3"

# ── Step 1: Availability check ────────────────────────────────────────────────
print("=" * 65)
print("  EXPERIMENT 2 — Compression comparison (DeepFakes only)")
print("=" * 65)

COMPRESSION_LEVELS = {
    "c0":  {"label": "c0 (raw/uncompressed)",   "quality": "raw"},
    "c23": {"label": "c23 (light compression)",  "quality": "light"},
    "c40": {"label": "c40 (heavy compression)",  "quality": "heavy"},
}

COMP_COLORS = {
    "c0":  "#2ecc71",
    "c23": "#3498db",
    "c40": "#e74c3c",
}

print("\n[Step 1] Checking FF++ compression folder availability ...")
available   = {}
unavailable = []

for comp, info in COMPRESSION_LEVELS.items():
    fake_path = f"{FFPP_ROOT}/manipulated_sequences/Deepfakes/{comp}/videos"
    real_path = f"{FFPP_ROOT}/original_sequences/youtube/{comp}/videos"

    fake_ok = os.path.isdir(fake_path) and len(os.listdir(fake_path)) > 0
    real_ok = os.path.isdir(real_path) and len(os.listdir(real_path)) > 0

    if fake_ok and real_ok:
        n_fake = len(os.listdir(fake_path))
        n_real = len(os.listdir(real_path))
        print(f"  ✓  {comp}  ({info['label']})  "
              f"Fake={n_fake} videos, Real={n_real} videos")
        available[comp] = {
            "fake_dir": fake_path,
            "real_dir": real_path,
            "n_fake":   n_fake,
            "n_real":   n_real,
        }
    else:
        status_fake = f"fake_dir={'FOUND' if fake_ok else 'MISSING'}"
        status_real = f"real_dir={'FOUND' if real_ok else 'MISSING'}"
        print(f"  ✗  {comp}  ({info['label']})  — UNAVAILABLE  [{status_fake}, {status_real}]")
        unavailable.append(comp)

print(f"\n  Available  : {list(available.keys())}")
print(f"  Unavailable: {unavailable}  (skipping — will not fail)")

if not available:
    print("\n  [ABORT] No compression levels available. Exiting.")
    exit(0)


# ── Step 2: Feature CSV availability / extraction ────────────────────────────
print("\n[Step 2] Checking feature CSV availability ...")

def csv_path_for(comp, kind):
    """kind: 'fake' or 'real'"""
    return f"{BASE}/features/ffpp_deepfakes_{comp}_{kind}.csv"

def load_df(file_list):
    dfs = [pd.read_csv(f) for f in file_list]
    df  = pd.concat(dfs, ignore_index=True)
    fc  = sorted([c for c in df.columns if c.startswith("s_") or c.startswith("t_")])
    df[fc] = df[fc].replace([np.inf, -np.inf], np.nan)
    for c in fc:
        df[c] = df[c].fillna(df[c].median())
    return df[fc].values.astype(np.float64), df["label"].values.astype(int), fc

def extract_features(video_dir, output_csv, label, workers=8):
    """Run GPU feature extractor via subprocess."""
    extractor = f"{BASE}/features/precompute_features_best_gpu.py"
    cmd = [
        VENV_PY, extractor,
        "--video_dir", video_dir,
        "--output",    output_csv,
        "--label",     str(label),
        "--max_frames","120",
        "--workers",   str(workers),
    ]
    print(f"  Running extractor → {output_csv}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [ERROR] Extractor failed:\n{result.stderr[-500:]}")
        return False
    # Print summary line
    for line in result.stdout.splitlines():
        if "Done" in line or "Success" in line or "Features saved" in line:
            print(f"    {line.strip()}")
    return True

# c23 real features: use the already-split ffpp_real_train/test CSVs (no leakage)
# For other compression levels, we create new real CSVs from their respective folders

comp_csvs = {}  # comp → {"fake": path, "real": path}

for comp in available:
    # c23: use the canonical, already-validated CSVs directly
    if comp == "c23":
        fake_csv = f"{BASE}/features/ffpp_fake.csv"
        real_csv = f"{BASE}/features/ffpp_real.csv"
    else:
        fake_csv = csv_path_for(comp, "fake")
        real_csv = csv_path_for(comp, "real")

    # ── Fake features ────────────────────────────────────────────────────────
    if os.path.exists(fake_csv):
        df_check = pd.read_csv(fake_csv)
        print(f"  ✓  {comp} fake CSV exists — {len(df_check)} rows  ({fake_csv})")
    else:
        print(f"  ✗  {comp} fake CSV missing — extracting ...")
        ok = extract_features(available[comp]["fake_dir"], fake_csv, label=1)
        if not ok:
            print(f"  [SKIP] {comp} fake extraction failed.")
            continue

    # ── Real features ─────────────────────────────────────────────────────────
    if comp == "c23":
        # Reuse existing split CSVs — no leakage
        real_train_csv = f"{BASE}/features/ffpp_real_train.csv"
        real_test_csv  = f"{BASE}/features/ffpp_real_test.csv"
        print(f"  ✓  {comp} real CSV: using existing train/test split (no leakage)")
        comp_csvs[comp] = {
            "fake":       fake_csv,
            "real_train": real_train_csv,
            "real_test":  real_test_csv,
        }
        continue

    if os.path.exists(real_csv):
        df_check = pd.read_csv(real_csv)
        print(f"  ✓  {comp} real CSV exists — {len(df_check)} rows  ({real_csv})")
    else:
        print(f"  ✗  {comp} real CSV missing — extracting ...")
        ok = extract_features(available[comp]["real_dir"], real_csv, label=0)
        if not ok:
            print(f"  [SKIP] {comp} real extraction failed.")
            continue

    # For non-c23 levels: split real CSV 80/20 fresh
    real_df = pd.read_csv(real_csv)
    from sklearn.model_selection import train_test_split
    r_train, r_test = train_test_split(real_df, test_size=0.2, random_state=42, shuffle=True)
    real_train_csv = real_csv.replace(".csv", "_train.csv")
    real_test_csv  = real_csv.replace(".csv", "_test.csv")
    r_train.to_csv(real_train_csv, index=False)
    r_test.to_csv(real_test_csv,   index=False)
    print(f"  ✓  {comp} real split: train={len(r_train)}, test={len(r_test)}")

    comp_csvs[comp] = {
        "fake":       fake_csv,
        "real_train": real_train_csv,
        "real_test":  real_test_csv,
    }

# Fallback for c23 if not added above
if "c23" in available and "c23" not in comp_csvs:
    comp_csvs["c23"] = {
        "fake":       f"{BASE}/features/ffpp_fake.csv",
        "real_train": f"{BASE}/features/ffpp_real_train.csv",
        "real_test":  f"{BASE}/features/ffpp_real_test.csv",
    }

if not comp_csvs:
    print("\n  [ABORT] No valid CSVs to work with.")
    exit(0)


# ── Step 3: Train & evaluate per compression level ────────────────────────────
print(f"\n[Step 3] Training and evaluation ...")

classifiers = {
    "LR": LogisticRegression(
        C=1.0, penalty="l2", solver="lbfgs", max_iter=2000,
        class_weight="balanced", random_state=42),
    "RF": RandomForestClassifier(
        n_estimators=200, max_depth=8, min_samples_leaf=10,
        class_weight="balanced", random_state=42, n_jobs=-1),
    "LGBM": lgb.LGBMClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        num_leaves=31, min_child_samples=20,
        class_weight="balanced", random_state=42, verbose=-1),
}

def bootstrap_ci(y_true, y_prob, n_boot=2000, seed=42):
    rng = np.random.RandomState(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        if len(np.unique(y_true[idx])) < 2: continue
        boots.append(roc_auc_score(y_true[idx], y_prob[idx]))
    return np.percentile(boots, 2.5), np.percentile(boots, 97.5)

all_results = {}

for comp, paths in comp_csvs.items():
    label = COMPRESSION_LEVELS[comp]["label"]
    print(f"\n  ── {label} ──")

    # Load train
    X_tr, y_tr, fc = load_df([paths["real_train"], paths["fake"]])
    # Load test
    X_te, y_te, _  = load_df([paths["real_test"],  paths["fake"]])

    print(f"    Train: {len(y_tr)}  (Real={(y_tr==0).sum()}, Fake={(y_tr==1).sum()})")
    print(f"    Test : {len(y_te)}  (Real={(y_te==0).sum()}, Fake={(y_te==1).sum()})")

    scaler    = StandardScaler()
    X_tr_sc   = scaler.fit_transform(X_tr)
    X_te_sc   = scaler.transform(X_te)

    # 5-fold CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    comp_res = {}
    best_auc = 0.0; best_clf = None; best_y_prob = None

    print(f"    {'Clf':<6}  {'CV AUC':>10}  {'Test AUC':>10}  {'F1':>7}  "
          f"{'Prec':>7}  {'Recall':>8}  {'MCC':>7}  {'95% CI':>22}")
    print(f"    {'-'*87}")

    for clf_name, clf_template in classifiers.items():
        fold_aucs = []
        for tr_idx, val_idx in skf.split(X_tr_sc, y_tr):
            m = clone(clf_template)
            m.fit(X_tr_sc[tr_idx], y_tr[tr_idx])
            fold_aucs.append(
                roc_auc_score(y_tr[val_idx],
                              m.predict_proba(X_tr_sc[val_idx])[:, 1])
            )
        cv_mean = np.mean(fold_aucs)
        cv_std  = np.std(fold_aucs)

        clf = clone(clf_template)
        clf.fit(X_tr_sc, y_tr)
        y_prob = clf.predict_proba(X_te_sc)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        test_auc = roc_auc_score(y_te, y_prob)
        f1       = f1_score(y_te, y_pred, zero_division=0)
        prec     = precision_score(y_te, y_pred, zero_division=0)
        rec      = recall_score(y_te, y_pred, zero_division=0)
        mcc      = matthews_corrcoef(y_te, y_pred)
        ci_lo, ci_hi = bootstrap_ci(y_te, y_prob)

        ci_str = f"[{ci_lo:.4f},{ci_hi:.4f}]"
        print(f"    {clf_name:<6}  {cv_mean:>10.4f}  {test_auc:>10.4f}  {f1:>7.4f}  "
              f"{prec:>7.4f}  {rec:>8.4f}  {mcc:>7.4f}  {ci_str:>22}")

        comp_res[clf_name] = {
            "cv_auc_mean": cv_mean, "cv_auc_std": cv_std,
            "test_auc": test_auc, "f1": f1,
            "precision": prec, "recall": rec, "mcc": mcc,
            "ci_lo": ci_lo, "ci_hi": ci_hi,
        }

        if test_auc > best_auc:
            best_auc = test_auc; best_clf = clf_name; best_y_prob = y_prob

    all_results[comp] = {
        "label": label,
        "classifiers": comp_res,
        "best_classifier": best_clf,
        "best_auc": best_auc,
        "y_te": y_te.tolist(),
        "y_prob_best": best_y_prob.tolist() if best_y_prob is not None else [],
    }

    print(f"    → Best: {best_clf}  AUC={best_auc:.4f}")


# ── Summary table ─────────────────────────────────────────────────────────────
print(f"\n\n{'='*75}")
print("  SUMMARY TABLE — Compression comparison (DeepFakes)")
print(f"{'='*75}")
print(f"  {'Compression':<28} {'Best Model':>10} {'AUC':>8} {'F1':>7} "
      f"{'Prec':>7} {'Recall':>8} {'MCC':>7}")
print(f"  {'-'*78}")

summary_rows = []
for comp, res in all_results.items():
    bc = res["best_classifier"]
    r  = res["classifiers"][bc]
    print(f"  {res['label']:<28} {bc:>10} {r['test_auc']:>8.4f} {r['f1']:>7.4f} "
          f"{r['precision']:>7.4f} {r['recall']:>8.4f} {r['mcc']:>7.4f}")
    summary_rows.append({
        "Compression": res["label"],
        "Best Model": bc,
        "AUC": round(r["test_auc"], 4),
        "CI Lo": round(r["ci_lo"], 4),
        "CI Hi": round(r["ci_hi"], 4),
        "F1": round(r["f1"], 4),
        "Precision": round(r["precision"], 4),
        "Recall": round(r["recall"], 4),
        "MCC": round(r["mcc"], 4),
    })

pd.DataFrame(summary_rows).to_csv(f"{OUTDIR}/summary.csv", index=False)
print(f"\n  Saved: results/exp2/summary.csv")


# ── Step 4: Visualizations ────────────────────────────────────────────────────
print(f"\n[Step 4] Generating visualizations ...")

metrics  = ["AUC", "F1", "Precision", "Recall", "MCC"]
clf_keys = list(classifiers.keys())
comp_list = list(all_results.keys())
n_comps  = len(comp_list)

# ── Plot 1: Grouped bar chart — AUC per compression × classifier ──────────────
fig, ax = plt.subplots(figsize=(max(7, n_comps*3), 5))
x     = np.arange(n_comps)
width = 0.25
clf_colors = {"LR": "#3498db", "RF": "#e67e22", "LGBM": "#9b59b6"}

for i, clf_name in enumerate(clf_keys):
    auc_vals = [all_results[c]["classifiers"][clf_name]["test_auc"]
                for c in comp_list]
    ci_lo = [all_results[c]["classifiers"][clf_name]["ci_lo"] for c in comp_list]
    ci_hi = [all_results[c]["classifiers"][clf_name]["ci_hi"] for c in comp_list]
    errs  = [np.array(auc_vals) - np.array(ci_lo),
             np.array(ci_hi) - np.array(auc_vals)]
    bars = ax.bar(x + i*width, auc_vals, width,
                  label=clf_name, color=clf_colors[clf_name],
                  alpha=0.85, edgecolor="white")
    ax.errorbar(x + i*width, auc_vals,
                yerr=errs, fmt="none", color="black", capsize=4, linewidth=1.2)
    for bar, v in zip(bars, auc_vals):
        ax.text(bar.get_x() + bar.get_width()/2,
                v + 0.015, f"{v:.4f}",
                ha="center", fontsize=8, fontweight="bold")

ax.set_xticks(x + width)
ax.set_xticklabels([all_results[c]["label"] for c in comp_list], fontsize=10)
ax.set_ylim(0.4, 1.15)
ax.set_ylabel("Test AUC", fontsize=12)
ax.set_title("AUC Comparison across Compression Levels — DeepFakes\n"
             "(Error bars = 95% bootstrap CI)", fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
ax.axhline(0.5, color="gray", linestyle="--", lw=0.9, alpha=0.5, label="Chance")
ax.grid(axis="y", alpha=0.3)

if len(comp_list) > 1:
    note = "Note: c0 and/or c40 not available in this dataset installation."
    ax.text(0.5, -0.12, note, transform=ax.transAxes, fontsize=8,
            ha="center", color="gray", style="italic")

plt.tight_layout()
plt.savefig(f"{OUTDIR}/bar_auc_compression.png", dpi=180)
plt.close()
print(f"  Saved: results/exp2/bar_auc_compression.png")


# ── Plot 2: Multi-metric bar chart (all metrics, best clf per comp) ────────────
fig, ax = plt.subplots(figsize=(max(7, n_comps*3.5), 5))
metric_colors = {"AUC": "#2ecc71","F1": "#3498db","Precision": "#9b59b6",
                 "Recall": "#e74c3c","MCC": "#f39c12"}
x     = np.arange(n_comps)
width = 0.15

for i, metric in enumerate(metrics):
    vals = [summary_rows[j][metric] for j in range(len(summary_rows))]
    bars = ax.bar(x + i*width, vals, width, label=metric,
                  color=metric_colors[metric], alpha=0.85, edgecolor="white")

ax.set_xticks(x + width*2)
ax.set_xticklabels([r["Compression"] for r in summary_rows], fontsize=10)
ax.set_ylim(0, 1.15)
ax.set_ylabel("Score", fontsize=12)
ax.set_title("All Metrics — Best Classifier per Compression Level\n(DeepFakes)", fontsize=13, fontweight="bold")
ax.legend(fontsize=9)
ax.axhline(0.5, color="gray", linestyle="--", lw=0.8, alpha=0.4)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/bar_metrics_compression.png", dpi=180)
plt.close()
print(f"  Saved: results/exp2/bar_metrics_compression.png")


# ── Plot 3: ROC curve(s) ───────────────────────────────────────────────────────
from sklearn.metrics import roc_curve
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot([0,1],[0,1],"k--",lw=1,alpha=0.4,label="Random (AUC=0.50)")

for comp, res in all_results.items():
    y_te   = np.array(res["y_te"])
    y_prob = np.array(res["y_prob_best"])
    fpr, tpr, _ = roc_curve(y_te, y_prob)
    auc = res["best_auc"]
    bc  = res["best_classifier"]
    ax.plot(fpr, tpr, color=COMP_COLORS[comp], lw=2.2,
            label=f"{res['label']} [{bc}]  AUC={auc:.4f}")

ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title("ROC Curves — DeepFakes by Compression Level",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=9, loc="lower right")
ax.set_xlim([-0.01, 1.01]); ax.set_ylim([-0.01, 1.01])
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/roc_compression.png", dpi=180)
plt.close()
print(f"  Saved: results/exp2/roc_compression.png")


# ── Availability report ───────────────────────────────────────────────────────
avail_report = {
    "available_compressions":   list(available.keys()),
    "unavailable_compressions": unavailable,
    "reason_unavailable":       "Folders not present in dataset installation. "
                                "Only c23 (light compression) was downloaded.",
    "experiments_run":          list(all_results.keys()),
    "results":                  {k: {kk: vv for kk, vv in v.items()
                                     if kk not in ("y_te","y_prob_best")}
                                 for k, v in all_results.items()},
    "summary": summary_rows,
}
with open(f"{OUTDIR}/results.json", "w") as f:
    json.dump(avail_report, f, indent=2, default=str)
print(f"  Saved: results/exp2/results.json")

print(f"\n{'='*65}")
print(f"  Experiment 2 complete. Outputs in results/exp2/")
print(f"  Unavailable: {unavailable} — skipped gracefully.")
print(f"{'='*65}\n")
