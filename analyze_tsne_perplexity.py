# Phantom Lens Research Framework
# Developer: Miheer Satish Kulkarni | IIIT Nagpur | 2026
# All Rights Reserved


"""
Phantom Lens — t-SNE Perplexity Sweep
Shows how the feature space looks at different perplexity values.
Usage: python analyze_tsne_perplexity.py
"""

import os
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# ─── CONFIG ──────────────────────────────────────────────────
PKL_PATH      = "data/precomputed_features.pkl"
RESULTS_DIR   = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

PERPLEXITIES  = [5, 15, 30, 50, 100]   # the sweep
N_EACH        = 2000                    # 2000 real + 2000 fake = 4000 total (fast)
RANDOM_STATE  = 42

# ─── LOAD & PREPARE ──────────────────────────────────────────
print("Loading pkl...")
with open(PKL_PATH, 'rb') as f:
    data = pickle.load(f)

features = data['features'].astype(np.float32)
labels   = np.array(data['labels'], dtype=int)

real_mask = labels == 0
fake_mask = labels == 1
print(f"Loaded: {len(labels)} samples | Real: {real_mask.sum()} | Fake: {fake_mask.sum()}")

# Normalize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# Subsample equally
np.random.seed(RANDOM_STATE)
real_idx = np.random.choice(np.where(real_mask)[0], N_EACH, replace=False)
fake_idx = np.random.choice(np.where(fake_mask)[0], N_EACH, replace=False)
sample_idx    = np.concatenate([real_idx, fake_idx])
sample_labels = labels[sample_idx]

# PCA pre-reduction to 12 dims (keeps 90% variance, speeds up t-SNE a lot)
print(f"PCA(12) pre-reduction on {len(sample_idx)} samples...")
pca12 = PCA(n_components=12, random_state=RANDOM_STATE)
X_pca12 = pca12.fit_transform(X_scaled[sample_idx])

real_mask_s = sample_labels == 0
fake_mask_s = sample_labels == 1

# ─── PERPLEXITY SWEEP — GRID PLOT ────────────────────────────
print(f"\nRunning t-SNE for {len(PERPLEXITIES)} perplexity values...")
print("This will take several minutes — one run per perplexity.\n")

n_cols = len(PERPLEXITIES)
fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))
fig.suptitle(
    't-SNE Perplexity Sweep — 24-dim Physics Feature Space\nReal vs Fake (n=4000)',
    fontsize=15, fontweight='bold', y=1.02
)

tsne_results = {}

for i, perp in enumerate(PERPLEXITIES):
    print(f"  Running perplexity={perp}...")
    tsne = TSNE(
        n_components=2,
        perplexity=perp,
        learning_rate=200,
        max_iter=1000,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    X_tsne = tsne.fit_transform(X_pca12)
    tsne_results[perp] = X_tsne
    print(f"  Done perplexity={perp}")

    ax = axes[i]
    ax.scatter(X_tsne[real_mask_s, 0], X_tsne[real_mask_s, 1],
               c='steelblue', alpha=0.4, s=6, label='Real')
    ax.scatter(X_tsne[fake_mask_s, 0], X_tsne[fake_mask_s, 1],
               c='tomato', alpha=0.4, s=6, label='Fake')
    ax.set_title(f'Perplexity = {perp}', fontsize=12, fontweight='bold')
    ax.set_xlabel('Dim 1', fontsize=9)
    ax.set_ylabel('Dim 2', fontsize=9)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.2)
    if i == 0:
        ax.legend(fontsize=9, markerscale=3)

plt.tight_layout()
out = os.path.join(RESULTS_DIR, 'tsne_perplexity_sweep.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSaved grid → {out}")

# ─── INDIVIDUAL PLOTS PER PERPLEXITY ─────────────────────────
print("\nSaving individual plots...")
for perp, X_tsne in tsne_results.items():
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(X_tsne[real_mask_s, 0], X_tsne[real_mask_s, 1],
               c='steelblue', alpha=0.4, s=8, label=f'Real (n={real_mask_s.sum()})')
    ax.scatter(X_tsne[fake_mask_s, 0], X_tsne[fake_mask_s, 1],
               c='tomato', alpha=0.4, s=8, label=f'Fake (n={fake_mask_s.sum()})')
    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax.set_title(
        f't-SNE — Perplexity={perp}\n24-dim Physics Feature Space | Real vs Fake',
        fontsize=13, fontweight='bold'
    )
    ax.legend(fontsize=11, markerscale=3)
    ax.grid(True, alpha=0.3)
    ax.text(0.02, 0.98,
            f'Perplexity={perp} | PCA(12) pre-reduction | n=4000',
            transform=ax.transAxes, fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, f'tsne_perplexity_{perp}.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {out}")

# ─── SUMMARY ─────────────────────────────────────────────────
print("\n" + "="*55)
print("t-SNE PERPLEXITY SWEEP COMPLETE")
print("="*55)
print("What to look for:")
print("  Low perplexity (5-15)  : tight local clusters, may look fragmented")
print("  Mid perplexity (30-50) : best balance of local and global structure")
print("  High perplexity (100)  : smoother, more global view of separation")
print("\nFiles saved:")
print("  results/tsne_perplexity_sweep.png  <- all 5 in one grid")
for perp in PERPLEXITIES:
    print(f"  results/tsne_perplexity_{perp}.png")