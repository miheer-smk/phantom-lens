"""
Phantom Lens V3 — t-SNE Analysis
Author: Miheer Satish Kulkarni, IIIT Nagpur
"""
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import os
import warnings
warnings.filterwarnings('ignore')

PKL_PATH   = "data/precomputed_features_v3_base.pkl"
OUTPUT_DIR = "results/v3_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

LIVE_DIMS = [i for i in range(30) if i != 20]
REAL_C = '#185FA5'
FAKE_C = '#A32D2D'
CELEB_C = '#3B6D11'
FFPP_C  = '#854F0B'
GEN_COLORS = {
    'deepfakes':     '#E24B4A',
    'face2face':     '#D85A30',
    'faceshifter':   '#BA7517',
    'faceswap':      '#854F0B',
    'neuraltextures':'#533AB7',
    'real':          '#185FA5',
    'real_youtube':  '#1D9E75',
}

print("Loading pkl...")
with open(PKL_PATH, 'rb') as f:
    data = pickle.load(f)
features  = np.array(data['features'], dtype=np.float32)[:, LIVE_DIMS]
labels    = np.array(data['labels'],   dtype=np.int32)
sources   = np.array(data['dataset_sources'])
gen_types = np.array(data['generator_types'])

# Subsample balanced
np.random.seed(42)
N = 2500
r_idx = np.random.choice(np.where(labels==0)[0], min(N,(labels==0).sum()), replace=False)
f_idx = np.random.choice(np.where(labels==1)[0], min(N,(labels==1).sum()), replace=False)
idx   = np.concatenate([r_idx, f_idx])
X_viz = features[idx]
y_viz = labels[idx]
s_viz = sources[idx]
g_viz = gen_types[idx]

scaler = StandardScaler()
X_sc   = scaler.fit_transform(X_viz)

print(f"Running t-SNE on {len(idx)} samples (takes 3-5 min)...")
tsne = TSNE(n_components=2, perplexity=40, n_iter=800,
            random_state=42, init='pca', learning_rate='auto')
emb  = tsne.fit_transform(X_sc)
kl   = tsne.kl_divergence_
print(f"t-SNE done. KL={kl:.3f}")

sep = np.linalg.norm(emb[y_viz==0].mean(0) - emb[y_viz==1].mean(0))

# ── FIGURE 1: Real vs Fake + Density ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.patch.set_facecolor('#F8F9FA')

ax = axes[0]
ax.set_facecolor('white')
ax.scatter(emb[y_viz==1,0], emb[y_viz==1,1], c=FAKE_C, s=8, alpha=0.4, linewidths=0, label=f'Fake (n={int((y_viz==1).sum())})')
ax.scatter(emb[y_viz==0,0], emb[y_viz==0,1], c=REAL_C, s=8, alpha=0.4, linewidths=0, label=f'Real (n={int((y_viz==0).sum())})')
rc = emb[y_viz==0].mean(0); fc = emb[y_viz==1].mean(0)
ax.plot(*rc, 'o', color=REAL_C, ms=14, zorder=5)
ax.plot(*fc, 'o', color=FAKE_C, ms=14, zorder=5)
ax.annotate('Real centroid', rc, xytext=(rc[0]+4, rc[1]+4), fontsize=9, color=REAL_C, fontweight='bold')
ax.annotate('Fake centroid', fc, xytext=(fc[0]+4, fc[1]+4), fontsize=9, color=FAKE_C, fontweight='bold')
ax.set_title(f'V3 t-SNE — Real vs Fake\nKL={kl:.3f}  Cluster sep={sep:.1f}', fontsize=11, fontweight='bold')
ax.legend(markerscale=3, fontsize=10)
ax.set_xlabel('t-SNE dim 1', fontsize=10)
ax.set_ylabel('t-SNE dim 2', fontsize=10)
ax.spines[['top','right']].set_visible(False)

ax2 = axes[1]
ax2.set_facecolor('white')
for src, color, mk in [('celebvhq', CELEB_C, 'o'), ('ffpp_official', FFPP_C, 's')]:
    for lbl, ls_mk in [(0, '^'), (1, 'v')]:
        mask = (s_viz == src) & (y_viz == lbl)
        if mask.sum() == 0: continue
        lname = f'{src} {"real" if lbl==0 else "fake"}'
        ax2.scatter(emb[mask,0], emb[mask,1], c=color if lbl==0 else FAKE_C,
                    s=6, alpha=0.35, linewidths=0, marker=ls_mk, label=lname)
ax2.set_title('V3 t-SNE — By Dataset + Label', fontsize=11, fontweight='bold')
ax2.legend(markerscale=2.5, fontsize=7, loc='upper right')
ax2.set_xlabel('t-SNE dim 1', fontsize=10)
ax2.set_ylabel('t-SNE dim 2', fontsize=10)
ax2.spines[['top','right']].set_visible(False)

plt.suptitle('Phantom Lens V3 — t-SNE Feature Space\n29-dim physics features | 5,000 samples',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'tsne_main.png'), dpi=150, bbox_inches='tight', facecolor='#F8F9FA')
plt.close()
print("Saved: tsne_main.png")

# ── FIGURE 2: Coloured by manipulation type ───────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 8))
fig.patch.set_facecolor('#F8F9FA')
ax.set_facecolor('white')
unique_gens = sorted(set(g_viz))
for gen in unique_gens:
    mask = g_viz == gen
    color = GEN_COLORS.get(gen, '#888780')
    ax.scatter(emb[mask,0], emb[mask,1], c=color, s=7,
               alpha=0.4, linewidths=0, label=f'{gen} (n={mask.sum()})')
ax.set_title(f'V3 t-SNE — By Manipulation Type\nKL={kl:.3f}', fontsize=12, fontweight='bold')
ax.legend(markerscale=2.5, fontsize=8, loc='upper right')
ax.set_xlabel('t-SNE dim 1', fontsize=10)
ax.set_ylabel('t-SNE dim 2', fontsize=10)
ax.spines[['top','right']].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'tsne_by_manipulation.png'), dpi=150, bbox_inches='tight', facecolor='#F8F9FA')
plt.close()
print("Saved: tsne_by_manipulation.png")

print(f"\nt-SNE complete. Cluster separation: {sep:.2f}")