#!/usr/bin/env python3
"""
Experiment 5 — Hard Negatives Analysis (DeepFakes).

Uses same multi-manipulation training setup as Exp1:
  Train : real_train + Deepfakes + Face2Face + FaceSwap + NeuralTextures
  Test  : real_test  + Deepfakes  (full ffpp_fake.csv)

Best classifier from Exp1: LightGBM (AUC=0.9995 on Deepfakes)

Identifies false negatives: fake videos predicted as real.
Reports video_path, confidence score, and feature profiles.
"""

import os, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, confusion_matrix, classification_report,
    ConfusionMatrixDisplay,
)
import lightgbm as lgb

warnings.filterwarnings("ignore")

BASE   = "/home/iiitn/Miheer_project_FE/phantom-lens"
OUTDIR = f"{BASE}/results/exp5"
os.makedirs(OUTDIR, exist_ok=True)

TRAIN_FILES = [
    "features/ffpp_real_train.csv",
    "features/ffpp_fake.csv",
    "features/ffpp_face2face.csv",
    "features/ffpp_faceswap.csv",
    "features/ffpp_neuraltextures.csv",
]
REAL_TEST  = "features/ffpp_real_test.csv"
FAKE_TEST  = "features/ffpp_fake.csv"

LGBM_PARAMS = dict(
    n_estimators=200, max_depth=6, learning_rate=0.05,
    num_leaves=31, min_child_samples=20,
    class_weight="balanced", random_state=42, verbose=-1,
)
THRESHOLD = 0.5

# ── Data loading ──────────────────────────────────────────────────────────────
def load_df_raw(file_list):
    """Load CSVs, return full DataFrame (with video_path) and feature cols."""
    dfs = [pd.read_csv(f"{BASE}/{f}") for f in file_list]
    df  = pd.concat(dfs, ignore_index=True)
    fc  = sorted([c for c in df.columns if c.startswith("s_") or c.startswith("t_")])
    df[fc] = df[fc].replace([np.inf, -np.inf], np.nan)
    for c in fc:
        df[c] = df[c].fillna(df[c].median())
    return df, fc


print("=" * 65)
print("  EXPERIMENT 5 — Hard Negatives Analysis (DeepFakes)")
print("=" * 65)

# ── Train ─────────────────────────────────────────────────────────────────────
print("\n[1/4] Training multi-manip LightGBM ...")
train_df, feat_cols = load_df_raw(TRAIN_FILES)
X_tr = train_df[feat_cols].values.astype(np.float64)
y_tr = train_df["label"].values.astype(int)

scaler   = StandardScaler()
X_tr_sc  = scaler.fit_transform(X_tr)

clf = lgb.LGBMClassifier(**LGBM_PARAMS)
clf.fit(pd.DataFrame(X_tr_sc, columns=feat_cols), y_tr, feature_name=feat_cols)
print(f"  Train: {len(y_tr)}  (Real={(y_tr==0).sum()}, Fake={(y_tr==1).sum()})")

# ── Test: real_test + deepfakes ───────────────────────────────────────────────
print("\n[2/4] Predicting on test set (real_test + deepfakes) ...")
real_test_df, _ = load_df_raw([REAL_TEST])
fake_test_df, _ = load_df_raw([FAKE_TEST])

# Keep original video_path for tracing
real_test_df["split"] = "real"
fake_test_df["split"] = "fake"
test_df_full = pd.concat([real_test_df, fake_test_df], ignore_index=True)

X_te     = test_df_full[feat_cols].values.astype(np.float64)
X_te_sc  = scaler.transform(X_te)
y_te     = test_df_full["label"].values.astype(int)

y_prob = clf.predict_proba(pd.DataFrame(X_te_sc, columns=feat_cols))[:, 1]
y_pred = (y_prob >= THRESHOLD).astype(int)

auc    = roc_auc_score(y_te, y_prob)
print(f"  Test: {len(y_te)}  (Real={(y_te==0).sum()}, Fake={(y_te==1).sum()})")
print(f"  Test AUC: {auc:.4f}")
print(classification_report(y_te, y_pred, target_names=["Real","Fake"]))

# ── False negative identification ─────────────────────────────────────────────
print("\n[3/4] Identifying false negatives ...")

fake_mask   = y_te == 1
fn_mask     = (y_te == 1) & (y_pred == 0)   # fake predicted as real
tp_mask     = (y_te == 1) & (y_pred == 1)   # correctly detected fakes

n_fakes     = int(fake_mask.sum())
n_fn        = int(fn_mask.sum())
fn_pct      = 100 * n_fn / n_fakes
n_tp        = int(tp_mask.sum())

print(f"  Total fakes in test  : {n_fakes}")
print(f"  True positives (TP)  : {n_tp}  ({100*n_tp/n_fakes:.1f}%)")
print(f"  False negatives (FN) : {n_fn}  ({fn_pct:.1f}%)")

# Build FN DataFrame with confidence scores
fn_df = test_df_full[fn_mask].copy()
fn_df["confidence_fake"] = y_prob[fn_mask]      # P(fake) — low = hard to detect
fn_df["predicted_label"] = y_pred[fn_mask]       # 0 = predicted real
fn_df["true_label"]      = y_te[fn_mask]          # 1 = is fake

fn_out = fn_df[["video_path", "confidence_fake", "true_label",
                 "predicted_label"] + feat_cols].copy()
fn_out = fn_out.sort_values("confidence_fake", ascending=True)  # lowest conf first

# Also build TP DataFrame for comparison
tp_df = test_df_full[tp_mask].copy()
tp_df["confidence_fake"] = y_prob[tp_mask]

# ── Summary stats ─────────────────────────────────────────────────────────────
print(f"\n  FN confidence scores (P(fake)):")
print(f"    Mean  : {fn_df['confidence_fake'].mean():.4f}")
print(f"    Std   : {fn_df['confidence_fake'].std():.4f}")
print(f"    Min   : {fn_df['confidence_fake'].min():.4f}")
print(f"    Max   : {fn_df['confidence_fake'].max():.4f}")
print(f"\n  TP confidence scores (P(fake)) for comparison:")
print(f"    Mean  : {tp_df['confidence_fake'].mean():.4f}")
print(f"    Min   : {tp_df['confidence_fake'].min():.4f}")
print(f"    Max   : {tp_df['confidence_fake'].max():.4f}")

# ── Feature profile: FN vs TP ─────────────────────────────────────────────────
print(f"\n  Top features most different between FN and TP (mean diff):")
fn_feats = fn_df[feat_cols]
tp_feats = tp_df[feat_cols]
diffs = (fn_feats.mean() - tp_feats.mean()).abs().sort_values(ascending=False)
for feat, d in diffs.head(10).items():
    fn_val = fn_feats[feat].mean()
    tp_val = tp_feats[feat].mean()
    print(f"    {feat:<38}  FN={fn_val:>8.4f}  TP={tp_val:>8.4f}  Δ={d:>7.4f}")

# ── Save tables ───────────────────────────────────────────────────────────────
fn_out[["video_path","confidence_fake","true_label","predicted_label"]].to_csv(
    f"{OUTDIR}/false_negatives.csv", index=False
)
fn_out.to_csv(f"{OUTDIR}/false_negatives_with_features.csv", index=False)
print(f"\n  Saved: results/exp5/false_negatives.csv")
print(f"  Saved: results/exp5/false_negatives_with_features.csv")

# ── Print top 20 FN table ──────────────────────────────────────────────────────
print(f"\n  {'#':>3}  {'Confidence P(fake)':>20}  {'Video'}")
print(f"  {'-'*80}")
for i, (_, row) in enumerate(fn_out.head(30).iterrows(), 1):
    vname = os.path.basename(str(row["video_path"]))
    print(f"  {i:>3}  {row['confidence_fake']:>20.4f}  {vname}")
if n_fn > 30:
    print(f"  ... ({n_fn - 30} more — see false_negatives.csv)")

# ── Step 4: Visualizations ────────────────────────────────────────────────────
print(f"\n[4/4] Generating visualizations ...")

# ── Plot 1: Confidence score distribution FN vs TP ───────────────────────────
fig, ax = plt.subplots(figsize=(9, 4))
bins = np.linspace(0, 1, 41)
ax.hist(tp_df["confidence_fake"], bins=bins, alpha=0.65,
        color="#2ecc71", label=f"True Positives (n={n_tp})")
ax.hist(fn_df["confidence_fake"], bins=bins, alpha=0.8,
        color="#e74c3c", label=f"False Negatives (n={n_fn})")
ax.axvline(THRESHOLD, color="black", linestyle="--", lw=1.5, label=f"Threshold={THRESHOLD}")
ax.set_xlabel("Model Confidence P(fake)", fontsize=12)
ax.set_ylabel("Count", fontsize=12)
ax.set_title(f"Confidence Score Distribution — DeepFakes Test Set\n"
             f"False Negatives: {n_fn}/{n_fakes} ({fn_pct:.1f}%)", fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/confidence_distribution.png", dpi=180)
plt.close()
print(f"  Saved: results/exp5/confidence_distribution.png")

# ── Plot 2: Per-video confidence bar for FN (sorted) ─────────────────────────
fig, ax = plt.subplots(figsize=(max(8, min(n_fn * 0.35, 20)), 4.5))
fn_sorted = fn_out.sort_values("confidence_fake")
vnames    = [os.path.basename(str(p))[:12] for p in fn_sorted["video_path"]]
confs     = fn_sorted["confidence_fake"].values
cmap_col  = ["#e74c3c" if c < 0.2 else "#e67e22" if c < 0.35 else "#f1c40f"
             for c in confs]
bars = ax.bar(range(len(confs)), confs, color=cmap_col, alpha=0.85, edgecolor="white")
ax.axhline(THRESHOLD, color="black", linestyle="--", lw=1.5,
           label=f"Decision threshold ({THRESHOLD})")
ax.set_xticks(range(len(vnames)))
ax.set_xticklabels(vnames, rotation=75, ha="right", fontsize=7)
ax.set_ylabel("P(fake) confidence", fontsize=11)
ax.set_ylim(0, 0.7)
ax.set_title(f"False Negative Confidence Scores — DeepFakes\n"
             f"(All {n_fn} missed fakes, sorted by confidence)",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=9)
p1 = mpatches.Patch(color="#e74c3c", label="P(fake) < 0.20  (very hard)")
p2 = mpatches.Patch(color="#e67e22", label="P(fake) 0.20–0.35")
p3 = mpatches.Patch(color="#f1c40f", label="P(fake) 0.35–0.50")
ax.legend(handles=[p1, p2, p3], fontsize=8, loc="upper left")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/fn_confidence_bar.png", dpi=180, bbox_inches="tight")
plt.close()
print(f"  Saved: results/exp5/fn_confidence_bar.png")

# ── Plot 3: Feature profile FN vs TP vs Real (radar-style bar) ────────────────
top10_diff_feats = diffs.head(10).index.tolist()
fn_means  = fn_feats[top10_diff_feats].mean()
tp_means  = tp_feats[top10_diff_feats].mean()
real_df   = test_df_full[y_te==0]
real_means = real_df[top10_diff_feats].mean()

x     = np.arange(len(top10_diff_feats))
width = 0.28
fig, ax = plt.subplots(figsize=(13, 5))
ax.bar(x - width, real_means.values, width, label="Real (TN)", color="#3498db", alpha=0.8)
ax.bar(x,         tp_means.values,   width, label="Fake TP",   color="#2ecc71", alpha=0.8)
ax.bar(x + width, fn_means.values,   width, label="Fake FN",   color="#e74c3c", alpha=0.8)
ax.set_xticks(x)
labels_short = [f.replace("t_","").replace("s_","") for f in top10_diff_feats]
ax.set_xticklabels(labels_short, rotation=35, ha="right", fontsize=9)
ax.set_ylabel("Feature value (original scale)", fontsize=11)
ax.set_title("Feature Profile: Real vs Fake TP vs Fake FN\n"
             "(Top-10 features with largest FN–TP difference)",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/feature_profile_fn_vs_tp.png", dpi=180)
plt.close()
print(f"  Saved: results/exp5/feature_profile_fn_vs_tp.png")

# ── Plot 4: Confusion matrix ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5, 4))
cm   = confusion_matrix(y_te, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=["Real","Fake"])
disp.plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title(f"Confusion Matrix — DeepFakes\nAUC={auc:.4f}", fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTDIR}/confusion_matrix.png", dpi=180)
plt.close()
print(f"  Saved: results/exp5/confusion_matrix.png")

# ── Save JSON summary ──────────────────────────────────────────────────────────
summary = {
    "test_auc":         auc,
    "n_fakes_total":    n_fakes,
    "n_true_positives": n_tp,
    "n_false_negatives":n_fn,
    "fn_pct":           fn_pct,
    "fn_confidence": {
        "mean":  float(fn_df["confidence_fake"].mean()),
        "std":   float(fn_df["confidence_fake"].std()),
        "min":   float(fn_df["confidence_fake"].min()),
        "max":   float(fn_df["confidence_fake"].max()),
    },
    "tp_confidence": {
        "mean": float(tp_df["confidence_fake"].mean()),
        "min":  float(tp_df["confidence_fake"].min()),
        "max":  float(tp_df["confidence_fake"].max()),
    },
    "top10_distinguishing_features": diffs.head(10).to_dict(),
    "false_negatives": fn_out[["video_path","confidence_fake"]].to_dict(orient="records"),
}
with open(f"{OUTDIR}/results.json", "w") as f:
    json.dump(summary, f, indent=2, default=str)
print(f"  Saved: results/exp5/results.json")

print(f"\n{'='*65}")
print(f"  Experiment 5 complete. Outputs in results/exp5/")
print(f"{'='*65}\n")
