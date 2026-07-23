#!/usr/bin/env python3
"""
Experiment 3 — Feature Ablation (LightGBM only).

Pipeline:
  Train : real_train + Deepfakes + Face2Face + FaceSwap + FaceShifter
  Test  : real_test  + NeuralTextures   (unseen manipulation type)

Steps:
  1. Train full LightGBM → rank features via SHAP (fallback: feature_importances_)
  2. Create subsets: top-3, top-10, top-20, all-50
  3. 10-fold CV + held-out test for each subset
  4. Bar chart + table
"""

import os, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from scipy import stats

warnings.filterwarnings("ignore")

BASE   = "/home/iiitn/Miheer_project_FE/phantom-lens"
OUTDIR = f"{BASE}/results/exp3"
os.makedirs(OUTDIR, exist_ok=True)

TRAIN_FILES = [
    "features/ffpp_real_train.csv",
    "features/ffpp_fake.csv",
    "features/ffpp_face2face.csv",
    "features/ffpp_faceswap.csv",
    "features/ffpp_faceshifter.csv",
]
TEST_FILES = [
    "features/ffpp_real_test.csv",
    "features/ffpp_neuraltextures.csv",
]

SUBSETS   = [3, 10, 20, 50]
LGBM_PARAMS = dict(
    n_estimators=200, max_depth=6, learning_rate=0.05,
    num_leaves=31, min_child_samples=20,
    class_weight="balanced", random_state=42, verbose=-1,
)

# ── Data loader ───────────────────────────────────────────────────────────────
def load_df(file_list):
    dfs = [pd.read_csv(f"{BASE}/{f}") for f in file_list]
    df  = pd.concat(dfs, ignore_index=True)
    fc  = sorted([c for c in df.columns if c.startswith("s_") or c.startswith("t_")])
    df[fc] = df[fc].replace([np.inf, -np.inf], np.nan)
    for c in fc:
        df[c] = df[c].fillna(df[c].median())
    return df, fc

def ci95_t(values):
    n  = len(values)
    m  = np.mean(values)
    se = stats.sem(values)
    lo, hi = stats.t.interval(0.95, df=n-1, loc=m, scale=se)
    return m, np.std(values), lo, hi

def bootstrap_auc_ci(y_true, y_prob, n_boot=2000, seed=42):
    rng   = np.random.RandomState(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        if len(np.unique(y_true[idx])) < 2: continue
        boots.append(roc_auc_score(y_true[idx], y_prob[idx]))
    return np.percentile(boots, 2.5), np.percentile(boots, 97.5)

# ══════════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("  EXPERIMENT 3 — Feature Ablation (LightGBM)")
print("=" * 65)

# ── Load data ─────────────────────────────────────────────────────────────────
print("\n[Data] Loading train / test ...")
train_df, feat_cols = load_df(TRAIN_FILES)
test_df,  _         = load_df(TEST_FILES)

X_tr_full = train_df[feat_cols].values.astype(np.float64)
y_tr      = train_df["label"].values.astype(int)
X_te_full = test_df[feat_cols].values.astype(np.float64)
y_te      = test_df["label"].values.astype(int)

scaler    = StandardScaler()
X_tr_sc   = scaler.fit_transform(X_tr_full)
X_te_sc   = scaler.transform(X_te_full)

print(f"  Train : {len(y_tr)}  (Real={(y_tr==0).sum()}, Fake={(y_tr==1).sum()})")
print(f"  Test  : {len(y_te)}  (Real={(y_te==0).sum()}, Fake={(y_te==1).sum()})")
print(f"  Features: {len(feat_cols)}")

# ── Step 1: Train full model → feature ranking ────────────────────────────────
print("\n[Step 1] Training full LightGBM (50 features) ...")

full_lgbm = lgb.LGBMClassifier(**LGBM_PARAMS)
full_lgbm.fit(
    pd.DataFrame(X_tr_sc, columns=feat_cols), y_tr,
    feature_name=feat_cols,
)

# ── SHAP importance ───────────────────────────────────────────────────────────
shap_available = False
shap_ranking   = None

try:
    import shap
    print("  Computing SHAP values (TreeExplainer) ...")
    explainer   = shap.TreeExplainer(full_lgbm)
    shap_values = explainer.shap_values(
        pd.DataFrame(X_tr_sc[:500], columns=feat_cols)   # subsample for speed
    )
    # shap_values may be list [neg_class, pos_class] or single array
    if isinstance(shap_values, list):
        sv = shap_values[1]   # positive (fake) class
    else:
        sv = shap_values

    mean_abs_shap = np.abs(sv).mean(axis=0)
    shap_ranking  = sorted(
        zip(feat_cols, mean_abs_shap), key=lambda x: x[1], reverse=True
    )
    shap_available = True
    print("  ✓ SHAP importance computed successfully")

except Exception as e:
    print(f"  ✗ SHAP failed ({e}) — falling back to LightGBM feature_importances_")

if not shap_available:
    imp = full_lgbm.feature_importances_
    shap_ranking = sorted(
        zip(feat_cols, imp.astype(float)), key=lambda x: x[1], reverse=True
    )

importance_method = "SHAP (mean |φ|)" if shap_available else "LightGBM gain importance"
ranked_features   = [name for name, _ in shap_ranking]
ranked_scores     = [score for _, score in shap_ranking]

print(f"\n  Ranking method: {importance_method}")
print(f"\n  {'Rank':>4}  {'Feature':<38}  {'Score':>10}")
print(f"  {'-'*57}")
for i, (feat, score) in enumerate(shap_ranking[:20], 1):
    print(f"  {i:>4}  {feat:<38}  {score:>10.4f}")
if len(shap_ranking) > 20:
    print(f"  ... ({len(shap_ranking)-20} more features)")

# Save ranking
ranking_df = pd.DataFrame({
    "rank":    range(1, len(ranked_features)+1),
    "feature": ranked_features,
    "score":   ranked_scores,
    "method":  importance_method,
})
ranking_df.to_csv(f"{OUTDIR}/feature_ranking.csv", index=False)
print(f"\n  Saved: results/exp3/feature_ranking.csv")

# ── Step 2 + 3: Train & evaluate each subset ──────────────────────────────────
print(f"\n[Step 2+3] Ablation — 10-fold CV + test AUC per subset ...")

skf     = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
results = {}

print(f"\n  {'Subset':>8}  {'CV AUC':>10}  {'± std':>7}  "
      f"{'95% CI (CV)':>22}  {'Test AUC':>10}  {'Test 95% CI':>22}")
print(f"  {'-'*90}")

for k in SUBSETS:
    top_feats = ranked_features[:k]
    fi = [feat_cols.index(f) for f in top_feats]

    X_tr_k = X_tr_sc[:, fi]
    X_te_k = X_te_sc[:, fi]

    # 10-fold CV
    fold_aucs = []
    for tr_idx, val_idx in skf.split(X_tr_k, y_tr):
        m = lgb.LGBMClassifier(**LGBM_PARAMS)
        m.fit(
            pd.DataFrame(X_tr_k[tr_idx], columns=top_feats), y_tr[tr_idx],
            feature_name=top_feats,
        )
        p = m.predict_proba(
            pd.DataFrame(X_tr_k[val_idx], columns=top_feats)
        )[:, 1]
        fold_aucs.append(roc_auc_score(y_tr[val_idx], p))

    cv_mean, cv_std, cv_lo, cv_hi = ci95_t(fold_aucs)

    # Full train → test
    clf_final = lgb.LGBMClassifier(**LGBM_PARAMS)
    clf_final.fit(
        pd.DataFrame(X_tr_k, columns=top_feats), y_tr,
        feature_name=top_feats,
    )
    y_prob    = clf_final.predict_proba(
        pd.DataFrame(X_te_k, columns=top_feats)
    )[:, 1]
    test_auc  = roc_auc_score(y_te, y_prob)
    t_lo, t_hi = bootstrap_auc_ci(y_te, y_prob)

    cv_ci_str  = f"[{cv_lo:.4f},{cv_hi:.4f}]"
    te_ci_str  = f"[{t_lo:.4f},{t_hi:.4f}]"
    feat_label = f"Top {k}" if k < 50 else "All 50"
    print(f"  {feat_label:>8}  {cv_mean:>10.4f}  {cv_std:>7.4f}  "
          f"{cv_ci_str:>22}  {test_auc:>10.4f}  {te_ci_str:>22}")

    results[k] = {
        "n_features":     k,
        "features":       top_feats,
        "fold_aucs":      fold_aucs,
        "cv_mean":        cv_mean,
        "cv_std":         cv_std,
        "cv_ci_lo":       cv_lo,
        "cv_ci_hi":       cv_hi,
        "test_auc":       test_auc,
        "test_ci_lo":     t_lo,
        "test_ci_hi":     t_hi,
    }

# ── Summary table ─────────────────────────────────────────────────────────────
print(f"\n\n  {'='*65}")
print(f"  ABLATION SUMMARY")
print(f"  {'='*65}")
print(f"  {'Subset':<12} {'Features':<50}")
for k in SUBSETS:
    feat_label = f"Top {k}" if k < 50 else "All 50"
    feats_str  = ", ".join(results[k]["features"][:3]) + ("..." if k > 3 else "")
    print(f"  {feat_label:<12} {feats_str}")

summary_rows = []
for k, res in results.items():
    feat_label = f"Top {k}" if k < 50 else "All 50"
    summary_rows.append({
        "Subset":      feat_label,
        "N Features":  k,
        "CV AUC":      round(res["cv_mean"], 4),
        "CV Std":      round(res["cv_std"],  4),
        "CV CI Lo":    round(res["cv_ci_lo"], 4),
        "CV CI Hi":    round(res["cv_ci_hi"], 4),
        "Test AUC":    round(res["test_auc"], 4),
        "Test CI Lo":  round(res["test_ci_lo"], 4),
        "Test CI Hi":  round(res["test_ci_hi"], 4),
    })

pd.DataFrame(summary_rows).to_csv(f"{OUTDIR}/ablation_summary.csv", index=False)
print(f"\n  Saved: results/exp3/ablation_summary.csv")

# ── Step 4: Visualizations ────────────────────────────────────────────────────
print(f"\n[Step 4] Generating visualizations ...")

subset_labels = [f"Top {k}" if k < 50 else "All 50" for k in SUBSETS]
test_aucs     = [results[k]["test_auc"]    for k in SUBSETS]
cv_means      = [results[k]["cv_mean"]     for k in SUBSETS]
test_ci_lo    = [results[k]["test_ci_lo"]  for k in SUBSETS]
test_ci_hi    = [results[k]["test_ci_hi"]  for k in SUBSETS]
cv_ci_lo      = [results[k]["cv_ci_lo"]    for k in SUBSETS]
cv_ci_hi      = [results[k]["cv_ci_hi"]    for k in SUBSETS]

x     = np.arange(len(SUBSETS))
width = 0.35

# ── Plot 1: AUC vs feature subset (test + CV side-by-side) ───────────────────
fig, ax = plt.subplots(figsize=(9, 5))

te_err = [np.array(test_aucs) - np.array(test_ci_lo),
          np.array(test_ci_hi) - np.array(test_aucs)]
cv_err = [np.array(cv_means)  - np.array(cv_ci_lo),
          np.array(cv_ci_hi)  - np.array(cv_means)]

b1 = ax.bar(x - width/2, test_aucs, width, label="Test AUC",
            color="#2ecc71", alpha=0.88, edgecolor="white")
ax.errorbar(x - width/2, test_aucs, yerr=te_err,
            fmt="none", color="black", capsize=5, linewidth=1.5)

b2 = ax.bar(x + width/2, cv_means, width, label="CV AUC (10-fold mean)",
            color="#3498db", alpha=0.88, edgecolor="white")
ax.errorbar(x + width/2, cv_means, yerr=cv_err,
            fmt="none", color="black", capsize=5, linewidth=1.5)

for bar, v in zip(b1, test_aucs):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.006,
            f"{v:.4f}", ha="center", fontsize=9, fontweight="bold", color="#1a7a43")
for bar, v in zip(b2, cv_means):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.006,
            f"{v:.4f}", ha="center", fontsize=9, fontweight="bold", color="#1a4f7a")

ax.set_xticks(x)
ax.set_xticklabels([f"{lbl}\n({k} feats)" for lbl, k in zip(subset_labels, SUBSETS)],
                   fontsize=10)
ax.set_ylim(0.5, 1.08)
ax.set_ylabel("AUC", fontsize=12)
ax.set_title(f"Feature Ablation — LightGBM\n"
             f"Feature ranking: {importance_method}\n"
             f"Train: Multi-manip  |  Test: NeuralTextures (unseen)",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=10)
ax.axhline(0.5, color="gray",  linestyle="--", lw=0.8, alpha=0.4)
ax.axhline(results[50]["test_auc"], color="#e74c3c", linestyle=":",
           lw=1.2, alpha=0.7, label=f"Full-50 Test AUC={results[50]['test_auc']:.4f}")
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/bar_ablation_auc.png", dpi=180)
plt.close()
print(f"  Saved: results/exp3/bar_ablation_auc.png")

# ── Plot 2: Test AUC line chart with CI shading ───────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(SUBSETS, test_aucs, "o-", color="#2ecc71", lw=2.5,
        markersize=8, label="Test AUC", zorder=3)
ax.fill_between(SUBSETS, test_ci_lo, test_ci_hi,
                color="#2ecc71", alpha=0.2, label="Test 95% CI")
ax.plot(SUBSETS, cv_means, "s--", color="#3498db", lw=2,
        markersize=7, label="CV AUC (10-fold)", zorder=3)
ax.fill_between(SUBSETS, cv_ci_lo, cv_ci_hi,
                color="#3498db", alpha=0.15, label="CV 95% CI")
for xi, yi in zip(SUBSETS, test_aucs):
    ax.annotate(f"{yi:.4f}", (xi, yi), textcoords="offset points",
                xytext=(5, 6), fontsize=9, color="#1a7a43", fontweight="bold")
ax.set_xlabel("Number of Features", fontsize=12)
ax.set_ylabel("AUC", fontsize=12)
ax.set_xticks(SUBSETS)
ax.set_xticklabels([str(k) for k in SUBSETS])
ax.set_ylim(max(0.5, min(test_aucs+cv_ci_lo) - 0.05), 1.02)
ax.set_title(f"AUC vs Feature Subset Size — LightGBM Ablation\n"
             f"({importance_method})", fontsize=12, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/line_ablation_auc.png", dpi=180)
plt.close()
print(f"  Saved: results/exp3/line_ablation_auc.png")

# ── Plot 3: SHAP / importance bar chart (top-20) ──────────────────────────────
top20_names  = [n.replace("t_","").replace("s_","") for n in ranked_features[:20]]
top20_scores = ranked_scores[:20]
colors_bar   = ["#e74c3c" if s > np.percentile(ranked_scores, 80)
                else "#f39c12" if s > np.percentile(ranked_scores, 50)
                else "#3498db" for s in top20_scores]

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(range(20), top20_scores[::-1], color=colors_bar[::-1], alpha=0.85)
ax.set_yticks(range(20))
ax.set_yticklabels(top20_names[::-1], fontsize=9)
ax.set_xlabel(importance_method, fontsize=11)
ax.set_title(f"Top-20 Feature Importances — LightGBM\n({importance_method})",
             fontsize=12, fontweight="bold")
for bar, v in zip(bars, top20_scores[::-1]):
    ax.text(v + max(top20_scores)*0.01, bar.get_y() + bar.get_height()/2,
            f"{v:.4f}", va="center", fontsize=8)

# Mark subset boundaries
subset_boundaries = [3, 10, 20]
boundary_y        = [19 - k + 0.5 for k in subset_boundaries]
for by, k in zip(boundary_y, subset_boundaries):
    if k <= 20:
        ax.axhline(by, color="gray", linestyle="--", lw=1, alpha=0.6)
        ax.text(max(top20_scores)*0.95, by + 0.3, f"Top-{k}",
                fontsize=8, color="gray", ha="right")

plt.tight_layout()
plt.savefig(f"{OUTDIR}/feature_importance_top20.png", dpi=180)
plt.close()
print(f"  Saved: results/exp3/feature_importance_top20.png")

# ── Plot 4: Fold-level CV boxplot per subset ──────────────────────────────────
fold_data = [results[k]["fold_aucs"] for k in SUBSETS]
fig, ax = plt.subplots(figsize=(8, 4.5))
bp = ax.boxplot(fold_data, patch_artist=True, notch=False,
                medianprops=dict(color="black", linewidth=2))
palette = ["#2ecc71", "#3498db", "#9b59b6", "#e74c3c"]
for patch, color in zip(bp["boxes"], palette):
    patch.set_facecolor(color); patch.set_alpha(0.7)
ax.set_xticklabels([f"{lbl}\n({k} feats)" for lbl, k in zip(subset_labels, SUBSETS)], fontsize=10)
ax.set_ylabel("AUC", fontsize=12)
ax.set_title("CV Fold AUC Distribution per Feature Subset\n(LightGBM, 10-fold)",
             fontsize=12, fontweight="bold")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/boxplot_cv_folds.png", dpi=180)
plt.close()
print(f"  Saved: results/exp3/boxplot_cv_folds.png")

# ── Save JSON ─────────────────────────────────────────────────────────────────
save_obj = {
    "importance_method": importance_method,
    "shap_available":    shap_available,
    "feature_ranking":   [[n, float(s)] for n, s in shap_ranking],
    "ablation_results":  {
        str(k): {kk: vv for kk, vv in v.items() if kk != "fold_aucs"}
        for k, v in results.items()
    },
    "summary": summary_rows,
}
with open(f"{OUTDIR}/results.json", "w") as f:
    json.dump(save_obj, f, indent=2)
print(f"  Saved: results/exp3/results.json")

print(f"\n{'='*65}")
print(f"  Experiment 3 complete. Outputs in results/exp3/")
print(f"{'='*65}\n")
