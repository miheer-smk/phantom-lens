# Phantom Lens Research Framework
# Developer: Miheer Satish Kulkarni | IIIT Nagpur | 2026
# All Rights Reserved


"""
Phantom Lens — PCA Analysis & Visualization
Run this AFTER training to visualize the 24-dim feature space.
Usage: python analyze_pca.py
"""

import os
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ─── CONFIG ──────────────────────────────────────────────────
PKL_PATH    = "data/precomputed_features.pkl"
RESULTS_DIR = "results"
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

# ─── LOAD DATA ───────────────────────────────────────────────

print("Loading pkl...")
with open(PKL_PATH, 'rb') as f:
    data = pickle.load(f)

features = data['features'].astype(np.float32)
labels   = np.array(data['labels'], dtype=int)
sources  = data.get('dataset_sources', ['unknown'] * len(labels))

real_mask = labels == 0
fake_mask = labels == 1
print(f"Loaded: {len(labels)} samples | Real: {real_mask.sum()} | Fake: {fake_mask.sum()}")

# Normalize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# ─── PLOT 1: PCA 2D (Real vs Fake) ───────────────────────────

print("Running PCA (2D)...")
pca2 = PCA(n_components=2, random_state=42)
X_2d = pca2.fit_transform(X_scaled)

# Subsample for readability (max 5000 points per class)
np.random.seed(42)
def subsample(mask, n=5000):
    idx = np.where(mask)[0]
    if len(idx) > n:
        idx = np.random.choice(idx, n, replace=False)
    return idx

real_idx = subsample(real_mask)
fake_idx = subsample(fake_mask)

fig, ax = plt.subplots(figsize=(10, 8))
ax.scatter(X_2d[real_idx, 0], X_2d[real_idx, 1],
           c='steelblue', alpha=0.35, s=8, label='Real')
ax.scatter(X_2d[fake_idx, 0], X_2d[fake_idx, 1],
           c='tomato', alpha=0.35, s=8, label='Fake')
ax.set_xlabel(f'PC1 ({pca2.explained_variance_ratio_[0]*100:.1f}% variance)', fontsize=12)
ax.set_ylabel(f'PC2 ({pca2.explained_variance_ratio_[1]*100:.1f}% variance)', fontsize=12)
ax.set_title('PCA of 24-dim Physics Feature Space\nReal vs Fake', fontsize=14, fontweight='bold')
ax.legend(fontsize=12, markerscale=4)
ax.grid(True, alpha=0.3)

total_var = pca2.explained_variance_ratio_.sum() * 100
ax.text(0.02, 0.98, f'Total variance explained: {total_var:.1f}%',
        transform=ax.transAxes, fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
out = os.path.join(RESULTS_DIR, 'pca_2d_real_vs_fake.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved → {out}")

# ─── PLOT 2: PCA Variance Explained ──────────────────────────

print("Running PCA (full)...")
pca_full = PCA(random_state=42)
pca_full.fit(X_scaled)

cumvar = np.cumsum(pca_full.explained_variance_ratio_) * 100
indvar = pca_full.explained_variance_ratio_ * 100

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.bar(range(1, 25), indvar, color='steelblue', edgecolor='white')
ax1.set_xlabel('Principal Component', fontsize=12)
ax1.set_ylabel('Variance Explained (%)', fontsize=12)
ax1.set_title('Individual Variance per PC', fontsize=13, fontweight='bold')
ax1.set_xticks(range(1, 25))
ax1.grid(True, alpha=0.3, axis='y')

ax2.plot(range(1, 25), cumvar, 'bo-', linewidth=2, markersize=5)
ax2.axhline(y=90, color='red', linestyle='--', alpha=0.7, label='90% threshold')
ax2.axhline(y=95, color='orange', linestyle='--', alpha=0.7, label='95% threshold')
n90 = int(np.searchsorted(cumvar, 90)) + 1
ax2.axvline(x=n90, color='red', linestyle=':', alpha=0.5)
ax2.set_xlabel('Number of Components', fontsize=12)
ax2.set_ylabel('Cumulative Variance (%)', fontsize=12)
ax2.set_title('Cumulative Variance Explained', fontsize=13, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.text(n90 + 0.3, 50, f'{n90} PCs\nfor 90%', fontsize=9, color='red')

plt.tight_layout()
out = os.path.join(RESULTS_DIR, 'pca_variance_explained.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved → {out}")

# ─── PLOT 3: PCA Loadings (which pillars matter most) ─────────

fig, ax = plt.subplots(figsize=(14, 6))
loadings = pca_full.components_[:2]  # PC1 and PC2

x = np.arange(24)
width = 0.35
bars1 = ax.bar(x - width/2, loadings[0], width, label='PC1', color='steelblue', alpha=0.8)
bars2 = ax.bar(x + width/2, loadings[1], width, label='PC2', color='tomato', alpha=0.8)

ax.set_xticks(x)
ax.set_xticklabels(PILLAR_NAMES, rotation=45, ha='right', fontsize=8)
ax.set_ylabel('Loading', fontsize=12)
ax.set_title('PCA Loadings — Which Features Drive Separation\n(PC1 & PC2)', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.axhline(0, color='black', linewidth=0.8)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
out = os.path.join(RESULTS_DIR, 'pca_loadings.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved → {out}")

# ─── PLOT 4: PCA 2D by Dataset Source ────────────────────────

unique_sources = sorted(set(sources))
colors = plt.cm.tab10(np.linspace(0, 1, len(unique_sources)))
color_map = dict(zip(unique_sources, colors))

fig, ax = plt.subplots(figsize=(12, 8))
for src in unique_sources:
    src_idx = [i for i, s in enumerate(sources) if s == src]
    src_idx = np.array(src_idx)
    # Subsample
    if len(src_idx) > 3000:
        src_idx = np.random.choice(src_idx, 3000, replace=False)
    marker = 'o' if any(labels[i] == 0 for i in src_idx[:5]) else '^'
    ax.scatter(X_2d[src_idx, 0], X_2d[src_idx, 1],
               color=color_map[src], alpha=0.4, s=8,
               label=src, marker=marker)

ax.set_xlabel(f'PC1 ({pca2.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
ax.set_ylabel(f'PC2 ({pca2.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
ax.set_title('PCA by Dataset Source\n(○ = real datasets, △ = fake datasets)', fontsize=13, fontweight='bold')
ax.legend(fontsize=8, markerscale=4, loc='best', ncol=2)
ax.grid(True, alpha=0.3)

plt.tight_layout()
out = os.path.join(RESULTS_DIR, 'pca_2d_by_source.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved → {out}")

# ─── SUMMARY ─────────────────────────────────────────────────

print("\n" + "="*50)
print("PCA SUMMARY")
print("="*50)
print(f"Dims needed for 90% variance : {n90}")
print(f"PC1 variance                 : {indvar[0]:.1f}%")
print(f"PC2 variance                 : {indvar[1]:.1f}%")
print(f"Top 5 features driving PC1   :")
top5 = np.argsort(np.abs(loadings[0]))[::-1][:5]
for rank, idx in enumerate(top5, 1):
    print(f"  {rank}. {PILLAR_NAMES[idx]:20s}  loading={loadings[0][idx]:+.3f}")

print("\nAll PCA plots saved to results/")
print("  pca_2d_real_vs_fake.png")
print("  pca_variance_explained.png")
print("  pca_loadings.png")
print("  pca_2d_by_source.png")

# ─── t-SNE ───────────────────────────────────────────────────
# t-SNE is slow on large datasets so we subsample to 8000 total
# We use PCA to 12 dims first (keeps 90% variance) then run t-SNE
# This is standard practice and much faster

print("\n" + "="*50)
print("t-SNE ANALYSIS")
print("="*50)
print("Note: t-SNE subsamples to 8000 points and pre-reduces via PCA(12). This is standard practice.")

from sklearn.manifold import TSNE

# Subsample equally from real and fake — 4000 each
np.random.seed(42)
real_all = np.where(real_mask)[0]
fake_all = np.where(fake_mask)[0]
n_each = min(4000, len(real_all), len(fake_all))
real_sample = np.random.choice(real_all, n_each, replace=False)
fake_sample = np.random.choice(fake_all, n_each, replace=False)
sample_idx  = np.concatenate([real_sample, fake_sample])
sample_labels = labels[sample_idx]
sample_sources = [sources[i] for i in sample_idx]

# Pre-reduce to 12 dims with PCA first (standard t-SNE trick)
print(f"Running PCA(12) pre-reduction on {len(sample_idx)} samples...")
pca12 = PCA(n_components=12, random_state=42)
X_pca12 = pca12.fit_transform(X_scaled[sample_idx])

# Run t-SNE
print("Running t-SNE (this takes 1-3 minutes)...")
tsne = TSNE(
    n_components=2,
    perplexity=40,
    learning_rate=200,
    max_iter=1000,
    random_state=42,
    n_jobs=-1
)
X_tsne = tsne.fit_transform(X_pca12)
print("t-SNE done!")

# ── t-SNE PLOT 1: Real vs Fake ────────────────────────────────
real_t = sample_labels == 0
fake_t = sample_labels == 1

fig, ax = plt.subplots(figsize=(10, 8))
ax.scatter(X_tsne[real_t, 0], X_tsne[real_t, 1],
           c='steelblue', alpha=0.35, s=8, label=f'Real (n={real_t.sum()})')
ax.scatter(X_tsne[fake_t, 0], X_tsne[fake_t, 1],
           c='tomato', alpha=0.35, s=8, label=f'Fake (n={fake_t.sum()})')
ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
ax.set_title('t-SNE of 24-dim Physics Feature Space\nReal vs Fake', fontsize=14, fontweight='bold')
ax.legend(fontsize=12, markerscale=4)
ax.grid(True, alpha=0.3)
ax.text(0.02, 0.98,
        'Perplexity=40 | PCA(12) pre-reduction | n=8000',
        transform=ax.transAxes, fontsize=9,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
out = os.path.join(RESULTS_DIR, 'tsne_real_vs_fake.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved → {out}")

# ── t-SNE PLOT 2: By Dataset Source ──────────────────────────
unique_src = sorted(set(sample_sources))
colors_t = plt.cm.tab10(np.linspace(0, 1, len(unique_src)))
cmap_t = dict(zip(unique_src, colors_t))

fig, ax = plt.subplots(figsize=(12, 8))
for src in unique_src:
    sidx = [i for i, s in enumerate(sample_sources) if s == src]
    sidx = np.array(sidx)
    marker = 'o' if any(sample_labels[i] == 0 for i in sidx[:5]) else '^'
    ax.scatter(X_tsne[sidx, 0], X_tsne[sidx, 1],
               color=cmap_t[src], alpha=0.4, s=8,
               label=src, marker=marker)

ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
ax.set_title('t-SNE by Dataset Source\n(○ = real datasets, △ = fake datasets)', fontsize=13, fontweight='bold')
ax.legend(fontsize=8, markerscale=4, loc='best', ncol=2)
ax.grid(True, alpha=0.3)

plt.tight_layout()
out = os.path.join(RESULTS_DIR, 'tsne_by_source.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved → {out}")

print("\nAll t-SNE plots saved to results/")
print("  tsne_real_vs_fake.png")
print("  tsne_by_source.png")