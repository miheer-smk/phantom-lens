# Phantom Lens Research Framework
# Developer: Miheer Satish Kulkarni | IIIT Nagpur | 2026
# All Rights Reserved


"""
Phantom Lens — t-SNE Iterations Sweep
Shows how t-SNE evolves and stabilizes across different iteration counts.
Perplexity fixed at 30 (optimal from perplexity sweep).
Usage: python analyze_tsne_iterations.py
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

ITERATIONS    = [250, 500, 750, 1000, 2000]   # the sweep
PERPLEXITY    = 30                             # fixed at optimal value
N_EACH        = 2000                           # 2000 real + 2000 fake = 4000 total
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

# Subsample equally — same samples across all runs for fair comparison
np.random.seed(RANDOM_STATE)
real_idx = np.random.choice(np.where(real_mask)[0], N_EACH, replace=False)
fake_idx = np.random.choice(np.where(fake_mask)[0], N_EACH, replace=False)
sample_idx    = np.concatenate([real_idx, fake_idx])
sample_labels = labels[sample_idx]

# PCA pre-reduction to 12 dims
print(f"PCA(12) pre-reduction on {len(sample_idx)} samples...")
pca12 = PCA(n_components=12, random_state=RANDOM_STATE)
X_pca12 = pca12.fit_transform(X_scaled[sample_idx])

real_mask_s = sample_labels == 0
fake_mask_s = sample_labels == 1

print(f"\nFixed perplexity = {PERPLEXITY}")
print(f"Sweeping iterations: {ITERATIONS}")
print(f"Estimated time: {len(ITERATIONS)*1.5:.0f}–{len(ITERATIONS)*3:.0f} minutes\n")

# ─── ITERATIONS SWEEP ────────────────────────────────────────
tsne_results = {}

for itr in ITERATIONS:
    print(f"  Running iterations={itr}...")
    tsne = TSNE(
        n_components=2,
        perplexity=PERPLEXITY,
        learning_rate=200,
        max_iter=itr,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    X_tsne = tsne.fit_transform(X_pca12)
    tsne_results[itr] = X_tsne
    print(f"  Done iterations={itr}")

print("\nAll t-SNE runs complete! Saving plots...\n")

# ─── GRID PLOT ───────────────────────────────────────────────
n_cols = len(ITERATIONS)
fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))
fig.suptitle(
    f't-SNE Iterations Sweep — 24-dim Physics Feature Space\n'
    f'Real vs Fake (n=4000) | Fixed Perplexity={PERPLEXITY}',
    fontsize=15, fontweight='bold', y=1.02
)

for i, itr in enumerate(ITERATIONS):
    X_tsne = tsne_results[itr]
    ax = axes[i]
    ax.scatter(X_tsne[real_mask_s, 0], X_tsne[real_mask_s, 1],
               c='steelblue', alpha=0.4, s=6, label='Real')
    ax.scatter(X_tsne[fake_mask_s, 0], X_tsne[fake_mask_s, 1],
               c='tomato', alpha=0.4, s=6, label='Fake')
    ax.set_title(f'Iterations = {itr}', fontsize=12, fontweight='bold')
    ax.set_xlabel('Dim 1', fontsize=9)
    ax.set_ylabel('Dim 2', fontsize=9)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.2)
    if i == 0:
        ax.legend(fontsize=9, markerscale=3)

plt.tight_layout()
out = os.path.join(RESULTS_DIR, 'tsne_iterations_sweep.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved grid → {out}")

# ─── INDIVIDUAL PLOTS ────────────────────────────────────────
print("Saving individual plots...")
for itr, X_tsne in tsne_results.items():
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(X_tsne[real_mask_s, 0], X_tsne[real_mask_s, 1],
               c='steelblue', alpha=0.4, s=8, label=f'Real (n={real_mask_s.sum()})')
    ax.scatter(X_tsne[fake_mask_s, 0], X_tsne[fake_mask_s, 1],
               c='tomato', alpha=0.4, s=8, label=f'Fake (n={fake_mask_s.sum()})')
    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax.set_title(
        f't-SNE — Iterations={itr}\n'
        f'24-dim Physics Feature Space | Perplexity={PERPLEXITY} | Real vs Fake',
        fontsize=13, fontweight='bold'
    )
    ax.legend(fontsize=11, markerscale=3)
    ax.grid(True, alpha=0.3)
    ax.text(0.02, 0.98,
            f'Iterations={itr} | Perplexity={PERPLEXITY} | PCA(12) pre-reduction | n=4000',
            transform=ax.transAxes, fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, f'tsne_iterations_{itr}.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {out}")

# ─── SUMMARY ─────────────────────────────────────────────────
print("\n" + "="*55)
print("t-SNE ITERATIONS SWEEP COMPLETE")
print("="*55)
print("What to look for:")
print("  100-250 iterations  : not converged yet, messy clusters")
print("  500 iterations      : taking shape, separation emerging")
print("  1000 iterations     : converged, stable separation")
print("  2000 iterations     : fully converged, no further change")
print("\nFiles saved:")
print("  results/tsne_iterations_sweep.png  <- all 5 in one grid")
for itr in ITERATIONS:
    print(f"  results/tsne_iterations_{itr}.png")