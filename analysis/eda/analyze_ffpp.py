#!/usr/bin/env python3
"""
PHANTOM LENS — FF++ In-Depth Feature Analysis
==============================================
Complete distribution and discriminative analysis of
physics features on FF++ dataset (Real vs Deepfakes).

Usage:
    python analyze_ffpp.py

Outputs saved to: results_ffpp_analysis/
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
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.manifold import TSNE

warnings.filterwarnings("ignore")

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

OUTPUT_DIR = "results_ffpp_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
print("=" * 60)
print("PHANTOM LENS — FF++ In-Depth Analysis")
print("=" * 60)

real_df = pd.read_csv("features/ffpp_real.csv")
fake_df = pd.read_csv("features/ffpp_fake.csv")

real_df["label"] = 0
fake_df["label"] = 1

df = pd.concat([real_df, fake_df], ignore_index=True)

feature_cols = sorted([c for c in df.columns
                       if c.startswith("s_") or c.startswith("t_")])

# Fill NaN
for col in feature_cols:
    df[col] = df[col].fillna(df[col].median())
df = df.replace([np.inf, -np.inf], np.nan)
for col in feature_cols:
    df[col] = df[col].fillna(df[col].median())

X = df[feature_cols].values.astype(np.float64)
y = df["label"].values.astype(int)

real_df_clean = df[df["label"] == 0]
fake_df_clean = df[df["label"] == 1]

print(f"Real: {len(real_df_clean)} | Fake (Deepfakes): {len(fake_df_clean)}")
print(f"Features: {len(feature_cols)}")


# ─────────────────────────────────────────────
# 2. COHEN'S D FOR ALL FEATURES
# ─────────────────────────────────────────────
print("\nComputing Cohen's d...")

cohens_d = {}
for col in feature_cols:
    r = real_df_clean[col].dropna().values
    f = fake_df_clean[col].dropna().values
    n1, n2 = len(r), len(f)
    if n1 < 2 or n2 < 2:
        cohens_d[col] = 0.0
        continue
    pooled = np.sqrt(((n1-1)*r.std()**2 + (n2-1)*f.std()**2) / (n1+n2-2))
    cohens_d[col] = abs(r.mean() - f.mean()) / (pooled + 1e-8)

sorted_d = sorted(cohens_d.items(), key=lambda x: x[1], reverse=True)

print(f"\n{'Rank':>4} {'Feature':<40} {'Cohen d':>8} {'Effect':>12}")
print("-" * 70)
for rank, (fname, d) in enumerate(sorted_d, 1):
    effect = "Large" if d > 0.8 else "Medium" if d > 0.5 else "Small" if d > 0.2 else "Negligible"
    print(f"{rank:4d} {fname:<40} {d:8.4f} {effect:>12}")


# ─────────────────────────────────────────────
# 3. COHEN'S D BAR CHART
# ─────────────────────────────────────────────
top_n = min(30, len(sorted_d))
names = [x[0] for x in sorted_d[:top_n]]
values = [x[1] for x in sorted_d[:top_n]]
bar_colors = ["#e24b4a" if v > 0.8 else "#ef9f27" if v > 0.5 else "#4a90d9" for v in values]

fig, ax = plt.subplots(figsize=(11, 9))
bars = ax.barh(range(top_n), values[::-1], color=bar_colors[::-1], edgecolor='white', linewidth=0.5)
ax.set_yticks(range(top_n))
ax.set_yticklabels(names[::-1], fontsize=8.5)
ax.axvline(x=0.8, color="red", linestyle="--", alpha=0.6, linewidth=1.5, label="Large effect (d=0.8)")
ax.axvline(x=0.5, color="orange", linestyle="--", alpha=0.6, linewidth=1.5, label="Medium effect (d=0.5)")
ax.axvline(x=0.2, color="gray", linestyle=":", alpha=0.5, linewidth=1, label="Small effect (d=0.2)")
ax.set_xlabel("Cohen's d (Effect Size)", fontsize=11)
ax.set_title("Feature Discriminative Strength — FF++ (Real vs Deepfakes)\n"
             f"n={len(real_df_clean)} real + {len(fake_df_clean)} fake", fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/01_cohens_d_ranking.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSaved: 01_cohens_d_ranking.png")


# ─────────────────────────────────────────────
# 4. KDE DISTRIBUTION PLOTS — TOP 12 FEATURES
# ─────────────────────────────────────────────
print("Generating KDE distribution plots...")

top12 = [x[0] for x in sorted_d[:12]]
fig, axes = plt.subplots(3, 4, figsize=(18, 12))
axes = axes.flatten()

for i, col in enumerate(top12):
    ax = axes[i]
    r_vals = real_df_clean[col].dropna().values
    f_vals = fake_df_clean[col].dropna().values

    # KDE
    try:
        r_kde = stats.gaussian_kde(r_vals)
        f_kde = stats.gaussian_kde(f_vals)
        xmin = min(r_vals.min(), f_vals.min())
        xmax = max(r_vals.max(), f_vals.max())
        xrange = np.linspace(xmin, xmax, 300)
        ax.fill_between(xrange, r_kde(xrange), alpha=0.4, color='#4a90d9', label='Real')
        ax.fill_between(xrange, f_kde(xrange), alpha=0.4, color='#e24b4a', label='Fake')
        ax.plot(xrange, r_kde(xrange), color='#2166ac', linewidth=1.5)
        ax.plot(xrange, f_kde(xrange), color='#c0392b', linewidth=1.5)
    except Exception:
        ax.hist(r_vals, bins=30, alpha=0.5, color='#4a90d9', label='Real', density=True)
        ax.hist(f_vals, bins=30, alpha=0.5, color='#e24b4a', label='Fake', density=True)

    d = cohens_d[col]
    _, pval = stats.mannwhitneyu(r_vals, f_vals, alternative='two-sided')
    stars = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
    ax.set_title(f"{col}\nd={d:.3f} {stars}", fontsize=8.5, fontweight='bold')
    ax.set_xlabel("Value", fontsize=7)
    ax.set_ylabel("Density", fontsize=7)
    ax.tick_params(labelsize=7)
    if i == 0:
        ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

plt.suptitle("Feature Distributions — FF++ Real vs Deepfakes (Top 12 by Cohen's d)",
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/02_kde_distributions_top12.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: 02_kde_distributions_top12.png")


# ─────────────────────────────────────────────
# 5. VIOLIN PLOTS — TOP 16 FEATURES
# ─────────────────────────────────────────────
print("Generating violin plots...")

top16 = [x[0] for x in sorted_d[:16]]
fig, axes = plt.subplots(4, 4, figsize=(18, 14))
axes = axes.flatten()

for i, col in enumerate(top16):
    ax = axes[i]
    r_vals = real_df_clean[col].dropna().values
    f_vals = fake_df_clean[col].dropna().values

    vp = ax.violinplot([r_vals, f_vals], positions=[0, 1], showmedians=True,
                       showextrema=True)
    vp['bodies'][0].set_facecolor('#4a90d9')
    vp['bodies'][0].set_alpha(0.6)
    vp['bodies'][1].set_facecolor('#e24b4a')
    vp['bodies'][1].set_alpha(0.6)
    for pc in ['cmedians', 'cmaxes', 'cmins', 'cbars']:
        vp[pc].set_color('black')
        vp[pc].set_linewidth(1.2)

    ax.scatter([0]*len(r_vals), r_vals, alpha=0.15, s=3, color='#2166ac')
    ax.scatter([1]*len(f_vals), f_vals, alpha=0.15, s=3, color='#c0392b')

    d = cohens_d[col]
    _, pval = stats.mannwhitneyu(r_vals, f_vals, alternative='two-sided')
    stars = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
    ax.set_title(f"{col}\nd={d:.3f} {stars}", fontsize=8, fontweight='bold')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Real', 'Fake'], fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(axis='y', alpha=0.2)

plt.suptitle("Violin Plots — FF++ Real vs Deepfakes (Top 16 Features)",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/03_violin_plots_top16.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: 03_violin_plots_top16.png")


# ─────────────────────────────────────────────
# 6. CORRELATION HEATMAP
# ─────────────────────────────────────────────
print("Generating correlation heatmap...")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
df_scaled = pd.DataFrame(X_scaled, columns=feature_cols)

top20_cols = [x[0] for x in sorted_d[:20]]
corr = df_scaled[top20_cols].corr()

fig, ax = plt.subplots(figsize=(14, 12))
im = ax.imshow(corr.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
plt.colorbar(im, ax=ax, shrink=0.8)
ax.set_xticks(range(len(top20_cols)))
ax.set_yticks(range(len(top20_cols)))
ax.set_xticklabels(top20_cols, rotation=45, ha='right', fontsize=7.5)
ax.set_yticklabels(top20_cols, fontsize=7.5)

for i in range(len(top20_cols)):
    for j in range(len(top20_cols)):
        val = corr.values[i, j]
        color = 'white' if abs(val) > 0.5 else 'black'
        ax.text(j, i, f"{val:.2f}", ha='center', va='center',
                fontsize=6, color=color)

ax.set_title("Feature Correlation Matrix — Top 20 Features (FF++)",
             fontsize=12, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/04_correlation_heatmap.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: 04_correlation_heatmap.png")


# ─────────────────────────────────────────────
# 7. PER-FEATURE AUC (UNIVARIATE)
# ─────────────────────────────────────────────
print("Computing per-feature univariate AUC...")

feature_aucs = {}
for col in feature_cols:
    vals = df[col].values
    try:
        auc_val = roc_auc_score(y, vals)
        feature_aucs[col] = max(auc_val, 1 - auc_val)
    except Exception:
        feature_aucs[col] = 0.5

sorted_aucs = sorted(feature_aucs.items(), key=lambda x: x[1], reverse=True)

top_n = min(30, len(sorted_aucs))
auc_names = [x[0] for x in sorted_aucs[:top_n]]
auc_values = [x[1] for x in sorted_aucs[:top_n]]
auc_colors = ["#e24b4a" if v > 0.8 else "#ef9f27" if v > 0.7 else "#4a90d9" for v in auc_values]

fig, ax = plt.subplots(figsize=(11, 9))
ax.barh(range(top_n), auc_values[::-1], color=auc_colors[::-1],
        edgecolor='white', linewidth=0.5)
ax.set_yticks(range(top_n))
ax.set_yticklabels(auc_names[::-1], fontsize=8.5)
ax.axvline(x=0.8, color="red", linestyle="--", alpha=0.6, linewidth=1.5, label="AUC=0.8")
ax.axvline(x=0.7, color="orange", linestyle="--", alpha=0.6, linewidth=1.5, label="AUC=0.7")
ax.axvline(x=0.5, color="gray", linestyle=":", alpha=0.5, linewidth=1, label="Random (AUC=0.5)")
ax.set_xlabel("Univariate AUC", fontsize=11)
ax.set_xlim(0.45, 1.0)
ax.set_title("Per-Feature Univariate AUC — FF++ (Real vs Deepfakes)",
             fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/05_per_feature_auc.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: 05_per_feature_auc.png")


# ─────────────────────────────────────────────
# 8. INTRA-DATASET ROC CURVES (CROSS-VALIDATION)
# ─────────────────────────────────────────────
print("Running 5-fold CV for intra-dataset ROC curves...")

classifiers = {
    "LogisticRegression": LogisticRegression(C=1.0, max_iter=2000,
        class_weight="balanced", random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=200, max_depth=8,
        class_weight="balanced", random_state=42, n_jobs=-1),
}
if HAS_LGBM:
    classifiers["LightGBM"] = lgb.LGBMClassifier(n_estimators=200,
        max_depth=6, learning_rate=0.05, class_weight="balanced",
        random_state=42, verbose=-1)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = {}

fig, ax = plt.subplots(figsize=(7, 6))
colors_clf = ['#2166ac', '#e24b4a', '#27ae60']

for idx, (clf_name, clf) in enumerate(classifiers.items()):
    y_scores = cross_val_predict(clf, X_scaled, y, cv=skf,
                                  method='predict_proba')[:, 1]
    auc_val = roc_auc_score(y, y_scores)
    fpr, tpr, _ = roc_curve(y, y_scores)
    cv_results[clf_name] = auc_val
    ax.plot(fpr, tpr, lw=2, color=colors_clf[idx],
            label=f"{clf_name} (AUC={auc_val:.4f})")
    print(f"  {clf_name}: CV AUC = {auc_val:.4f}")

ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
ax.set_xlabel("False Positive Rate", fontsize=11)
ax.set_ylabel("True Positive Rate", fontsize=11)
ax.set_title("Intra-Dataset ROC Curves — FF++ (5-fold CV)\nReal vs Deepfakes",
             fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/06_intra_dataset_roc.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: 06_intra_dataset_roc.png")


# ─────────────────────────────────────────────
# 9. TEMPORAL vs SPATIAL FEATURE COMPARISON
# ─────────────────────────────────────────────
print("Generating spatial vs temporal comparison...")

spatial_cols = [c for c in feature_cols if c.startswith("s_")]
temporal_cols = [c for c in feature_cols if c.startswith("t_")]

spatial_d = [cohens_d[c] for c in spatial_cols]
temporal_d = [cohens_d[c] for c in temporal_cols]

fig, axes = plt.subplots(1, 2, figsize=(13, 6))

# Box comparison
axes[0].boxplot([spatial_d, temporal_d], labels=['Spatial (13)', 'Temporal (37)'],
                patch_artist=True,
                boxprops=dict(facecolor='lightblue', alpha=0.7))
for i, (data, color) in enumerate(zip([spatial_d, temporal_d],
                                       ['#4a90d9', '#27ae60'])):
    axes[0].scatter([i+1]*len(data), data, alpha=0.5, s=20, color=color, zorder=3)
axes[0].axhline(y=0.8, color='red', linestyle='--', alpha=0.6, label='Large (d=0.8)')
axes[0].axhline(y=0.5, color='orange', linestyle='--', alpha=0.6, label='Medium (d=0.5)')
axes[0].set_ylabel("Cohen's d", fontsize=11)
axes[0].set_title("Spatial vs Temporal Feature Strength", fontsize=11, fontweight='bold')
axes[0].legend(fontsize=8)
axes[0].grid(axis='y', alpha=0.3)

# Count by effect size
categories = ['Large\n(d>0.8)', 'Medium\n(d>0.5)', 'Small\n(d>0.2)', 'Weak\n(d<0.2)']
s_counts = [
    sum(1 for d in spatial_d if d > 0.8),
    sum(1 for d in spatial_d if 0.5 < d <= 0.8),
    sum(1 for d in spatial_d if 0.2 < d <= 0.5),
    sum(1 for d in spatial_d if d <= 0.2),
]
t_counts = [
    sum(1 for d in temporal_d if d > 0.8),
    sum(1 for d in temporal_d if 0.5 < d <= 0.8),
    sum(1 for d in temporal_d if 0.2 < d <= 0.5),
    sum(1 for d in temporal_d if d <= 0.2),
]

x = np.arange(len(categories))
w = 0.35
axes[1].bar(x - w/2, s_counts, w, label='Spatial', color='#4a90d9', alpha=0.8)
axes[1].bar(x + w/2, t_counts, w, label='Temporal', color='#27ae60', alpha=0.8)
axes[1].set_xticks(x)
axes[1].set_xticklabels(categories, fontsize=9)
axes[1].set_ylabel("Number of Features", fontsize=11)
axes[1].set_title("Feature Count by Effect Size Category", fontsize=11, fontweight='bold')
axes[1].legend(fontsize=9)
axes[1].grid(axis='y', alpha=0.3)

for bar in axes[1].patches:
    h = bar.get_height()
    if h > 0:
        axes[1].text(bar.get_x() + bar.get_width()/2., h + 0.1,
                     str(int(h)), ha='center', va='bottom', fontsize=9)

plt.suptitle("Spatial vs Temporal Physics Feature Analysis — FF++",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/07_spatial_vs_temporal.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: 07_spatial_vs_temporal.png")


# ─────────────────────────────────────────────
# 10. t-SNE VISUALIZATION
# ─────────────────────────────────────────────
print("Running t-SNE (this may take 1-2 minutes)...")

sample_size = min(800, len(X_scaled))
idx = np.random.choice(len(X_scaled), sample_size, replace=False)
X_sample = X_scaled[idx]
y_sample = y[idx]

tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
X_tsne = tsne.fit_transform(X_sample)

fig, ax = plt.subplots(figsize=(8, 7))
colors_map = {0: '#4a90d9', 1: '#e24b4a'}
labels_map = {0: 'Real', 1: 'Deepfakes'}

for label in [0, 1]:
    mask = y_sample == label
    ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
               c=colors_map[label], label=labels_map[label],
               alpha=0.5, s=15, edgecolors='none')

for label in [0, 1]:
    mask = y_sample == label
    cx = X_tsne[mask, 0].mean()
    cy = X_tsne[mask, 1].mean()
    ax.scatter(cx, cy, c=colors_map[label], s=200, marker='*',
               edgecolors='black', linewidth=1, zorder=5)
    ax.annotate(f"{labels_map[label]}\ncentroid", (cx, cy),
                textcoords="offset points", xytext=(10, 5), fontsize=8,
                fontweight='bold', color=colors_map[label])

ax.set_title(f"t-SNE — FF++ Physics Feature Space\n"
             f"(n={sample_size} samples, 50 physics features)",
             fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.set_xlabel("t-SNE Dim 1", fontsize=10)
ax.set_ylabel("t-SNE Dim 2", fontsize=10)
ax.grid(alpha=0.2)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/08_tsne_ffpp.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: 08_tsne_ffpp.png")


# ─────────────────────────────────────────────
# 11. TEMPORAL FEATURE TIME-SERIES SUMMARY
# ─────────────────────────────────────────────
print("Generating feature mean comparison bar chart...")

top8_temporal = [c for c in [x[0] for x in sorted_d] if c.startswith("t_")][:8]

fig, axes = plt.subplots(2, 4, figsize=(18, 8))
axes = axes.flatten()

for i, col in enumerate(top8_temporal):
    ax = axes[i]
    r_mean = real_df_clean[col].mean()
    f_mean = fake_df_clean[col].mean()
    r_std = real_df_clean[col].std()
    f_std = fake_df_clean[col].std()

    bars = ax.bar(['Real', 'Fake'], [r_mean, f_mean],
                  yerr=[r_std, f_std], capsize=5,
                  color=['#4a90d9', '#e24b4a'], alpha=0.8,
                  error_kw=dict(elinewidth=1.5, ecolor='black'))
    d = cohens_d[col]
    _, pval = stats.mannwhitneyu(
        real_df_clean[col].dropna().values,
        fake_df_clean[col].dropna().values,
        alternative='two-sided')
    stars = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
    ax.set_title(f"{col}\nd={d:.3f} {stars}", fontsize=8.5, fontweight='bold')
    ax.set_ylabel("Mean ± Std", fontsize=8)
    ax.tick_params(labelsize=8)
    ax.grid(axis='y', alpha=0.3)

plt.suptitle("Top 8 Temporal Features — Mean ± Std (FF++ Real vs Deepfakes)",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/09_temporal_mean_comparison.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: 09_temporal_mean_comparison.png")


# ─────────────────────────────────────────────
# 12. SUMMARY STATISTICS TABLE
# ─────────────────────────────────────────────
print("\nGenerating summary statistics...")

summary_rows = []
for col in feature_cols:
    r = real_df_clean[col].dropna()
    f = fake_df_clean[col].dropna()
    _, pval = stats.mannwhitneyu(r.values, f.values, alternative='two-sided')
    summary_rows.append({
        'feature': col,
        'real_mean': r.mean(),
        'real_std': r.std(),
        'fake_mean': f.mean(),
        'fake_std': f.std(),
        'cohens_d': cohens_d[col],
        'p_value': pval,
        'univariate_auc': feature_aucs[col],
        'effect': ('Large' if cohens_d[col] > 0.8 else
                   'Medium' if cohens_d[col] > 0.5 else
                   'Small' if cohens_d[col] > 0.2 else 'Negligible')
    })

summary_df = pd.DataFrame(summary_rows).sort_values('cohens_d', ascending=False)
summary_df.to_csv(f"{OUTPUT_DIR}/feature_summary_statistics.csv", index=False)
print(f"Saved: feature_summary_statistics.csv")


# ─────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("ANALYSIS COMPLETE — SUMMARY")
print("=" * 60)
print(f"\nDataset: {len(real_df_clean)} Real + {len(fake_df_clean)} Fake (FF++ Deepfakes)")
print(f"Features: {len(feature_cols)} total")
print(f"\nEffect size breakdown:")
large = sum(1 for d in cohens_d.values() if d > 0.8)
medium = sum(1 for d in cohens_d.values() if 0.5 < d <= 0.8)
small = sum(1 for d in cohens_d.values() if 0.2 < d <= 0.5)
negligible = sum(1 for d in cohens_d.values() if d <= 0.2)
print(f"  Large (d>0.8):      {large} features")
print(f"  Medium (d>0.5):     {medium} features")
print(f"  Small (d>0.2):      {small} features")
print(f"  Negligible (d<0.2): {negligible} features")
print(f"\nTop 5 features by Cohen's d:")
for rank, (fname, d) in enumerate(sorted_d[:5], 1):
    print(f"  {rank}. {fname}: d={d:.4f}")
print(f"\nIntra-dataset CV AUC:")
for clf_name, auc_val in cv_results.items():
    print(f"  {clf_name}: {auc_val:.4f}")
print(f"\nAll outputs saved to: {OUTPUT_DIR}/")
print("=" * 60)