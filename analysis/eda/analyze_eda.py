# Phantom Lens Research Framework
# Developer: Miheer Satish Kulkarni | IIIT Nagpur | 2026
# All Rights Reserved


"""
Phantom Lens — EDA & Correlation Analysis
Generates: Correlation Matrix, Box Plots, Violin Plots, PDF Distributions
Usage: python analyze_eda.py
"""

import os
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ─── CONFIG ──────────────────────────────────────────────────
PKL_PATH    = "data/precomputed_features.pkl"
RESULTS_DIR = "results/eda"
os.makedirs(RESULTS_DIR, exist_ok=True)

PILLAR_NAMES = [
    "P1: VMR",           # dim 0
    "P1: Res Std",       # dim 1
    "P1: HF Ratio",      # dim 2
    "P2: PRNU Energy",   # dim 3
    "P2: Face/Periph",   # dim 4
    "P3: RG Corr",       # dim 5
    "P3: BG Corr",       # dim 6
    "P4: Face-BG Diff",  # dim 7
    "P4: Spec Ratio",    # dim 8
    "P4: Shadow",        # dim 9
    "P5: Drift Mean",    # dim 10
    "P5: Drift Std",     # dim 11
    "P6: Benford Dev",   # dim 12
    "P6: Block Art",     # dim 13
    "P6: Dbl Compress",  # dim 14
    "P7: Res Mean",      # dim 15
    "P7: Res Var",       # dim 16
    "P8: Blur Mag",      # dim 17
    "P8: Dir Consist",   # dim 18
    "P9: Flow Mag",      # dim 19
    "P9: Boundary",      # dim 20
    "P10: RG Shift",     # dim 21
    "P10: BG Shift",     # dim 22
    "P10: Edge/Ctr",     # dim 23
]

SHORT_NAMES = [
    "VMR", "ResStd", "HFRatio",
    "PRNU_E", "PRNU_R",
    "RGCorr", "BGCorr",
    "FBDiff", "SpecR", "Shadow",
    "DriftM", "DriftS",
    "Benford", "BlockA", "DblComp",
    "ResM", "ResV",
    "BlurM", "DirC",
    "FlowM", "Bound",
    "RGShft", "BGShft", "EdgeCtr",
]

REAL_COLOR = '#4A90D9'
FAKE_COLOR = '#E05C5C'

# ─── LOAD DATA ───────────────────────────────────────────────
print("Loading pkl...")
with open(PKL_PATH, 'rb') as f:
    data = pickle.load(f)

features = data['features'].astype(np.float32)
labels   = np.array(data['labels'], dtype=int)

real_mask = labels == 0
fake_mask = labels == 1

X_real = features[real_mask]
X_fake = features[fake_mask]

print(f"Loaded: {len(labels)} samples | Real: {real_mask.sum()} | Fake: {fake_mask.sum()}")

# Subsample for plots (max 5000 per class for speed)
np.random.seed(42)
def subsample(X, n=5000):
    if len(X) > n:
        idx = np.random.choice(len(X), n, replace=False)
        return X[idx]
    return X

X_real_s = subsample(X_real)
X_fake_s = subsample(X_fake)

print(f"Subsampled for plots: {len(X_real_s)} real | {len(X_fake_s)} fake\n")

# ═══════════════════════════════════════════════════════════════
# PLOT 1: PEARSON CORRELATION MATRIX — REAL
# ═══════════════════════════════════════════════════════════════
print("Generating Pearson Correlation Matrix (Real)...")
corr_real = np.corrcoef(X_real_s.T)

fig, ax = plt.subplots(figsize=(16, 14))
im = ax.imshow(corr_real, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
plt.colorbar(im, ax=ax, shrink=0.8, label='Pearson Correlation')
ax.set_xticks(range(24))
ax.set_yticks(range(24))
ax.set_xticklabels(SHORT_NAMES, rotation=90, fontsize=8)
ax.set_yticklabels(SHORT_NAMES, fontsize=8)
ax.set_title('Pearson Correlation Matrix — Real Videos\n24-dim Physics Feature Space',
             fontsize=14, fontweight='bold', pad=20)
# Annotate cells
for i in range(24):
    for j in range(24):
        val = corr_real[i, j]
        color = 'white' if abs(val) > 0.6 else 'black'
        ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                fontsize=5.5, color=color)
plt.tight_layout()
out = os.path.join(RESULTS_DIR, 'corr_pearson_real.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved → {out}")

# ═══════════════════════════════════════════════════════════════
# PLOT 2: PEARSON CORRELATION MATRIX — FAKE
# ═══════════════════════════════════════════════════════════════
print("Generating Pearson Correlation Matrix (Fake)...")
corr_fake = np.corrcoef(X_fake_s.T)

fig, ax = plt.subplots(figsize=(16, 14))
im = ax.imshow(corr_fake, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
plt.colorbar(im, ax=ax, shrink=0.8, label='Pearson Correlation')
ax.set_xticks(range(24))
ax.set_yticks(range(24))
ax.set_xticklabels(SHORT_NAMES, rotation=90, fontsize=8)
ax.set_yticklabels(SHORT_NAMES, fontsize=8)
ax.set_title('Pearson Correlation Matrix — Fake Videos\n24-dim Physics Feature Space',
             fontsize=14, fontweight='bold', pad=20)
for i in range(24):
    for j in range(24):
        val = corr_fake[i, j]
        color = 'white' if abs(val) > 0.6 else 'black'
        ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                fontsize=5.5, color=color)
plt.tight_layout()
out = os.path.join(RESULTS_DIR, 'corr_pearson_fake.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved → {out}")

# ═══════════════════════════════════════════════════════════════
# PLOT 3: CORRELATION DIFFERENCE (Real - Fake)
# ═══════════════════════════════════════════════════════════════
print("Generating Correlation Difference Matrix...")
corr_diff = corr_real - corr_fake

fig, ax = plt.subplots(figsize=(16, 14))
im = ax.imshow(corr_diff, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
plt.colorbar(im, ax=ax, shrink=0.8, label='Correlation Difference (Real - Fake)')
ax.set_xticks(range(24))
ax.set_yticks(range(24))
ax.set_xticklabels(SHORT_NAMES, rotation=90, fontsize=8)
ax.set_yticklabels(SHORT_NAMES, fontsize=8)
ax.set_title('Correlation Difference Matrix (Real minus Fake)\nLarge values = pillar pairs that behave differently in real vs fake',
             fontsize=14, fontweight='bold', pad=20)
for i in range(24):
    for j in range(24):
        val = corr_diff[i, j]
        color = 'white' if abs(val) > 0.5 else 'black'
        ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                fontsize=5.5, color=color)
plt.tight_layout()
out = os.path.join(RESULTS_DIR, 'corr_difference.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved → {out}")

# ═══════════════════════════════════════════════════════════════
# PLOT 4: SPEARMAN CORRELATION MATRIX — COMBINED
# ═══════════════════════════════════════════════════════════════
print("Generating Spearman Correlation Matrix...")
X_all_s = np.vstack([X_real_s, X_fake_s])
spearman_matrix = np.zeros((24, 24))
for i in range(24):
    for j in range(24):
        rho, _ = stats.spearmanr(X_all_s[:, i], X_all_s[:, j])
        spearman_matrix[i, j] = rho

fig, ax = plt.subplots(figsize=(16, 14))
im = ax.imshow(spearman_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
plt.colorbar(im, ax=ax, shrink=0.8, label='Spearman Correlation')
ax.set_xticks(range(24))
ax.set_yticks(range(24))
ax.set_xticklabels(SHORT_NAMES, rotation=90, fontsize=8)
ax.set_yticklabels(SHORT_NAMES, fontsize=8)
ax.set_title('Spearman Correlation Matrix — All Samples\n(Non-parametric, robust to outliers)',
             fontsize=14, fontweight='bold', pad=20)
for i in range(24):
    for j in range(24):
        val = spearman_matrix[i, j]
        color = 'white' if abs(val) > 0.6 else 'black'
        ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                fontsize=5.5, color=color)
plt.tight_layout()
out = os.path.join(RESULTS_DIR, 'corr_spearman.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved → {out}")

# ═══════════════════════════════════════════════════════════════
# PLOT 5: BOX PLOTS — ALL 24 FEATURES
# ═══════════════════════════════════════════════════════════════
print("Generating Box Plots...")

# Normalize for display
scaler = StandardScaler()
X_real_norm = scaler.fit_transform(X_real_s)
X_fake_norm = scaler.transform(X_fake_s)

fig, axes = plt.subplots(6, 4, figsize=(20, 24))
fig.suptitle('Box Plots — All 24 Physics Features\nReal (blue) vs Fake (red)',
             fontsize=16, fontweight='bold', y=1.01)

for idx in range(24):
    row, col = idx // 4, idx % 4
    ax = axes[row][col]

    real_data = X_real_norm[:, idx]
    fake_data = X_fake_norm[:, idx]

    bp_real = ax.boxplot(real_data, positions=[1], widths=0.5,
                         patch_artist=True, notch=True,
                         boxprops=dict(facecolor=REAL_COLOR, alpha=0.7),
                         medianprops=dict(color='white', linewidth=2),
                         whiskerprops=dict(color=REAL_COLOR),
                         capprops=dict(color=REAL_COLOR),
                         flierprops=dict(marker='o', color=REAL_COLOR,
                                        alpha=0.3, markersize=2))

    bp_fake = ax.boxplot(fake_data, positions=[2], widths=0.5,
                         patch_artist=True, notch=True,
                         boxprops=dict(facecolor=FAKE_COLOR, alpha=0.7),
                         medianprops=dict(color='white', linewidth=2),
                         whiskerprops=dict(color=FAKE_COLOR),
                         capprops=dict(color=FAKE_COLOR),
                         flierprops=dict(marker='o', color=FAKE_COLOR,
                                        alpha=0.3, markersize=2))

    ax.set_title(PILLAR_NAMES[idx], fontsize=9, fontweight='bold')
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Real', 'Fake'], fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xlim(0.3, 2.7)

    # Add mean markers
    ax.scatter([1], [real_data.mean()], color='navy', zorder=5,
               marker='D', s=20)
    ax.scatter([2], [fake_data.mean()], color='darkred', zorder=5,
               marker='D', s=20)

plt.tight_layout()
out = os.path.join(RESULTS_DIR, 'boxplots_all_features.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved → {out}")

# ═══════════════════════════════════════════════════════════════
# PLOT 6: VIOLIN PLOTS — ALL 24 FEATURES
# ═══════════════════════════════════════════════════════════════
print("Generating Violin Plots...")

fig, axes = plt.subplots(6, 4, figsize=(20, 24))
fig.suptitle('Violin Plots — All 24 Physics Features\nReal (blue) vs Fake (red) | Shows full distribution shape',
             fontsize=16, fontweight='bold', y=1.01)

for idx in range(24):
    row, col = idx // 4, idx % 4
    ax = axes[row][col]

    real_data = X_real_norm[:, idx]
    fake_data = X_fake_norm[:, idx]

    # Clip outliers for cleaner violin shape
    real_clip = np.clip(real_data, np.percentile(real_data, 1), np.percentile(real_data, 99))
    fake_clip = np.clip(fake_data, np.percentile(fake_data, 1), np.percentile(fake_data, 99))

    vp_real = ax.violinplot(real_clip, positions=[1], widths=0.6,
                            showmeans=True, showmedians=True)
    vp_fake = ax.violinplot(fake_clip, positions=[2], widths=0.6,
                            showmeans=True, showmedians=True)

    # Color violins
    for pc in vp_real['bodies']:
        pc.set_facecolor(REAL_COLOR)
        pc.set_alpha(0.7)
    for part in ['cmeans', 'cmedians', 'cbars', 'cmaxes', 'cmins']:
        if part in vp_real:
            vp_real[part].set_color('navy')

    for pc in vp_fake['bodies']:
        pc.set_facecolor(FAKE_COLOR)
        pc.set_alpha(0.7)
    for part in ['cmeans', 'cmedians', 'cbars', 'cmaxes', 'cmins']:
        if part in vp_fake:
            vp_fake[part].set_color('darkred')

    ax.set_title(PILLAR_NAMES[idx], fontsize=9, fontweight='bold')
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Real', 'Fake'], fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xlim(0.3, 2.7)

plt.tight_layout()
out = os.path.join(RESULTS_DIR, 'violinplots_all_features.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved → {out}")

# ═══════════════════════════════════════════════════════════════
# PLOT 7: PDF (KDE) PLOTS — ALL 24 FEATURES
# ═══════════════════════════════════════════════════════════════
print("Generating PDF/KDE Distribution Plots...")

fig, axes = plt.subplots(6, 4, figsize=(20, 24))
fig.suptitle('PDF (KDE) Distribution Plots — All 24 Physics Features\nReal (blue) vs Fake (red) | Overlap = harder to distinguish',
             fontsize=16, fontweight='bold', y=1.01)

for idx in range(24):
    row, col = idx // 4, idx % 4
    ax = axes[row][col]

    real_data = X_real_norm[:, idx]
    fake_data = X_fake_norm[:, idx]

    # Clip for KDE
    lo = min(np.percentile(real_data, 1), np.percentile(fake_data, 1))
    hi = max(np.percentile(real_data, 99), np.percentile(fake_data, 99))
    x_range = np.linspace(lo, hi, 200)

    # Add tiny jitter to avoid singular covariance
    real_data = real_data + np.random.normal(0, 1e-6, len(real_data))
    fake_data = fake_data + np.random.normal(0, 1e-6, len(fake_data))

    # KDE for real
    kde_real = stats.gaussian_kde(real_data)
    ax.fill_between(x_range, kde_real(x_range), alpha=0.4, color=REAL_COLOR, label='Real')
    ax.plot(x_range, kde_real(x_range), color=REAL_COLOR, linewidth=1.5)

    # KDE for fake
    kde_fake = stats.gaussian_kde(fake_data)
    ax.fill_between(x_range, kde_fake(x_range), alpha=0.4, color=FAKE_COLOR, label='Fake')
    ax.plot(x_range, kde_fake(x_range), color=FAKE_COLOR, linewidth=1.5)

    # Overlap area
    y_min = np.minimum(kde_real(x_range), kde_fake(x_range))
    ax.fill_between(x_range, y_min, alpha=0.3, color='purple', label='Overlap')

    ax.set_title(PILLAR_NAMES[idx], fontsize=9, fontweight='bold')
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.3)
    if idx == 0:
        ax.legend(fontsize=7)

plt.tight_layout()
out = os.path.join(RESULTS_DIR, 'pdf_kde_all_features.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved → {out}")

# ═══════════════════════════════════════════════════════════════
# PLOT 8: STATISTICAL SUMMARY — MEAN DIFFERENCE PER FEATURE
# ═══════════════════════════════════════════════════════════════
print("Generating Statistical Summary Plot...")

real_means = X_real_s.mean(axis=0)
fake_means = X_fake_s.mean(axis=0)
real_stds  = X_real_s.std(axis=0)
fake_stds  = X_fake_s.std(axis=0)

# Cohen's d effect size
pooled_std = np.sqrt((real_stds**2 + fake_stds**2) / 2)
cohens_d   = np.abs(real_means - fake_means) / (pooled_std + 1e-10)

fig, axes = plt.subplots(2, 1, figsize=(18, 12))
fig.suptitle('Statistical Summary — Real vs Fake per Feature',
             fontsize=14, fontweight='bold')

x = np.arange(24)
width = 0.35

# Mean comparison
ax1 = axes[0]
bars1 = ax1.bar(x - width/2, real_means, width, label='Real Mean',
                color=REAL_COLOR, alpha=0.8)
bars2 = ax1.bar(x + width/2, fake_means, width, label='Fake Mean',
                color=FAKE_COLOR, alpha=0.8)
ax1.errorbar(x - width/2, real_means, yerr=real_stds,
             fmt='none', color='navy', capsize=3, linewidth=1)
ax1.errorbar(x + width/2, fake_means, yerr=fake_stds,
             fmt='none', color='darkred', capsize=3, linewidth=1)
ax1.set_xticks(x)
ax1.set_xticklabels(SHORT_NAMES, rotation=45, ha='right', fontsize=8)
ax1.set_ylabel('Mean Value', fontsize=11)
ax1.set_title('Feature Means: Real vs Fake (error bars = std)', fontsize=12)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3, axis='y')

# Cohen's d
ax2 = axes[1]
colors_d = [FAKE_COLOR if d > 0.5 else REAL_COLOR if d > 0.2 else 'gray'
            for d in cohens_d]
bars = ax2.bar(x, cohens_d, color=colors_d, alpha=0.8, edgecolor='white')
ax2.axhline(y=0.2, color='green', linestyle='--', alpha=0.7, label='Small effect (0.2)')
ax2.axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='Medium effect (0.5)')
ax2.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='Large effect (0.8)')
ax2.set_xticks(x)
ax2.set_xticklabels(SHORT_NAMES, rotation=45, ha='right', fontsize=8)
ax2.set_ylabel("Cohen's d (Effect Size)", fontsize=11)
ax2.set_title("Cohen's d Effect Size per Feature\n(Red = large effect = strong discriminator)", fontsize=12)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3, axis='y')

# Label top features
top5_d = np.argsort(cohens_d)[::-1][:5]
for i in top5_d:
    ax2.text(i, cohens_d[i] + 0.02, f'{cohens_d[i]:.2f}',
             ha='center', va='bottom', fontsize=7, fontweight='bold')

plt.tight_layout()
out = os.path.join(RESULTS_DIR, 'statistical_summary.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved → {out}")

# ═══════════════════════════════════════════════════════════════
# PRINT SUMMARY
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("EDA SUMMARY")
print("="*55)

print("\nTop 5 features by Cohen's d (strongest discriminators):")
for rank, i in enumerate(np.argsort(cohens_d)[::-1][:5], 1):
    print(f"  {rank}. {PILLAR_NAMES[i]:20s}  d={cohens_d[i]:.3f}")

print("\nBottom 5 features by Cohen's d (weakest discriminators):")
for rank, i in enumerate(np.argsort(cohens_d)[:5], 1):
    print(f"  {rank}. {PILLAR_NAMES[i]:20s}  d={cohens_d[i]:.3f}")

print("\nHighly correlated feature pairs in Real (|r| > 0.7):")
for i in range(24):
    for j in range(i+1, 24):
        if abs(corr_real[i, j]) > 0.7:
            print(f"  {SHORT_NAMES[i]:10s} — {SHORT_NAMES[j]:10s}  r={corr_real[i,j]:.3f}")

print("\nAll EDA plots saved to results/eda/")
print("  corr_pearson_real.png")
print("  corr_pearson_fake.png")
print("  corr_difference.png")
print("  corr_spearman.png")
print("  boxplots_all_features.png")
print("  violinplots_all_features.png")
print("  pdf_kde_all_features.png")
print("  statistical_summary.png")