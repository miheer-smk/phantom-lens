"""
Phantom Lens V3 — PCA Analysis
Author: Miheer Satish Kulkarni, IIIT Nagpur
"""
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
import warnings
warnings.filterwarnings('ignore')

PKL_PATH   = "data/precomputed_features_v3_base.pkl"
OUTPUT_DIR = "results/v3_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

LIVE_DIMS = [i for i in range(30) if i != 20]
DIM_NAMES = [
    'P1_vmr','P1_resStd','P1_hfRatio',
    'P2_prnu_E','P2_faceRatio',
    'P3_rgCorr','P3_bgCorr',
    'P4_faceBG','P4_specular','P4_shadow',
    'P5_driftM','P5_driftS',
    'P6_benford','P6_blockArt','P6_dblComp',
    'P7_resM','P7_resV',
    'P8_blurMag','P8_blurDir',
    'P9_flowMag',
    'P10_rgShift','P10_bgShift','P10_edgeCtr',
    'P11_eyeAsym','P11_tempStab','P11_geom',
    'P12_colorTemp','P12_fillLight','P12_satCons'
]
REAL_C = '#185FA5'
FAKE_C = '#A32D2D'
CELEB_C = '#3B6D11'
FFPP_C  = '#854F0B'

print("Loading pkl...")
with open(PKL_PATH, 'rb') as f:
    data = pickle.load(f)
features = np.array(data['features'], dtype=np.float32)[:, LIVE_DIMS]
labels   = np.array(data['labels'],   dtype=np.int32)
sources  = np.array(data['dataset_sources'])
print(f"Shape: {features.shape} | Real: {(labels==0).sum()} | Fake: {(labels==1).sum()}")

# Subsample
np.random.seed(42)
N = 3000
r_idx = np.random.choice(np.where(labels==0)[0], min(N,(labels==0).sum()), replace=False)
f_idx = np.random.choice(np.where(labels==1)[0], min(N,(labels==1).sum()), replace=False)
idx   = np.concatenate([r_idx, f_idx])
X_viz = features[idx]
y_viz = labels[idx]
s_viz = sources[idx]

scaler  = StandardScaler()
X_sc    = scaler.fit_transform(X_viz)

print("Running PCA...")
pca    = PCA(n_components=10, random_state=42)
X_pca  = pca.fit_transform(X_sc)
var_ex = pca.explained_variance_ratio_

# ── FIGURE 1: Scree + PC1 vs PC2 real/fake + PC1 vs PC2 by source ────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.patch.set_facecolor('#F8F9FA')

ax = axes[0]
ax.set_facecolor('white')
ax.bar(range(1,11), var_ex*100, color='#185FA5', alpha=0.85, edgecolor='white')
ax.plot(range(1,11), np.cumsum(var_ex)*100, 'r-o', lw=2, ms=5, label='Cumulative')
ax.set_xlabel('Principal Component', fontsize=11)
ax.set_ylabel('Explained Variance (%)', fontsize=11)
ax.set_title('Scree Plot', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.spines[['top','right']].set_visible(False)
for i, v in enumerate(var_ex):
    ax.text(i+1, v*100+0.3, f'{v*100:.1f}%', ha='center', fontsize=7)

ax2 = axes[1]
ax2.set_facecolor('white')
ax2.scatter(X_pca[y_viz==1,0], X_pca[y_viz==1,1], c=FAKE_C, s=6, alpha=0.35, linewidths=0, label='Fake')
ax2.scatter(X_pca[y_viz==0,0], X_pca[y_viz==0,1], c=REAL_C, s=6, alpha=0.35, linewidths=0, label='Real')
rc = X_pca[y_viz==0].mean(0); fc = X_pca[y_viz==1].mean(0)
ax2.plot(*rc, 'o', color=REAL_C, ms=12, zorder=5)
ax2.plot(*fc, 'o', color=FAKE_C, ms=12, zorder=5)
sep = np.linalg.norm(rc - fc)
ax2.set_xlabel(f'PC1 ({var_ex[0]*100:.1f}%)', fontsize=11)
ax2.set_ylabel(f'PC2 ({var_ex[1]*100:.1f}%)', fontsize=11)
ax2.set_title(f'PC1 vs PC2 — Real vs Fake\nCluster sep={sep:.1f}', fontsize=11, fontweight='bold')
ax2.legend(markerscale=3, fontsize=10)
ax2.spines[['top','right']].set_visible(False)

ax3 = axes[2]
ax3.set_facecolor('white')
for src, color, mk in [('celebvhq', CELEB_C, 'o'), ('ffpp_official', FFPP_C, 's')]:
    idx2 = np.where(s_viz == src)[0]
    ax3.scatter(X_pca[idx2,0], X_pca[idx2,1], c=color, s=6, alpha=0.35, linewidths=0, marker=mk, label=src)
ax3.set_xlabel(f'PC1 ({var_ex[0]*100:.1f}%)', fontsize=11)
ax3.set_ylabel(f'PC2 ({var_ex[1]*100:.1f}%)', fontsize=11)
ax3.set_title('PC1 vs PC2 — By Dataset', fontsize=11, fontweight='bold')
ax3.legend(markerscale=3, fontsize=9)
ax3.spines[['top','right']].set_visible(False)

plt.suptitle('Phantom Lens V3 — PCA Analysis\n29-dim Physics Feature Space | 6,000 samples',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'pca_main.png'), dpi=150, bbox_inches='tight', facecolor='#F8F9FA')
plt.close()
print("Saved: pca_main.png")

# ── FIGURE 2: PCA Loadings ────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 9))
fig.patch.set_facecolor('#F8F9FA')
for ax, pc_idx, title in [(axes[0], 0, f'PC1 Loadings ({var_ex[0]*100:.1f}% variance)'),
                            (axes[1], 1, f'PC2 Loadings ({var_ex[1]*100:.1f}% variance)')]:
    ax.set_facecolor('white')
    loadings = pca.components_[pc_idx]
    colors_l = ['#3B6D11' if l > 0 else '#A32D2D' for l in loadings]
    ax.barh(range(len(DIM_NAMES)), loadings, color=colors_l, alpha=0.85, height=0.7, edgecolor='white')
    ax.axvline(0, color='black', lw=0.8)
    ax.set_yticks(range(len(DIM_NAMES)))
    ax.set_yticklabels(DIM_NAMES, fontsize=8)
    ax.set_xlabel('Loading', fontsize=11)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.spines[['top','right']].set_visible(False)
plt.suptitle("Phantom Lens V3 — PCA Feature Loadings\nWhich physics features drive the principal components",
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'pca_loadings.png'), dpi=150, bbox_inches='tight', facecolor='#F8F9FA')
plt.close()
print("Saved: pca_loadings.png")

# ── FIGURE 3: Loss curves from training report ────────────────────────────────
# Note: actual loss values not saved in pkl, showing AUC summary instead
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor('#F8F9FA')
ax.set_facecolor('white')
seeds = [42, 123, 777, 999, 2024]
aucs  = [0.8999, 0.8973, 0.9117, 0.8945, 0.9027]
colors_s = ['#185FA5','#A32D2D','#3B6D11','#854F0B','#533AB7']
bars = ax.bar(range(len(seeds)), aucs, color=colors_s, alpha=0.85, width=0.5, edgecolor='white')
ax.axhline(0.8961, color='black', ls='--', lw=1.5, label='V1 baseline (0.8961)')
ax.set_xticks(range(len(seeds)))
ax.set_xticklabels([f'Seed {s}' for s in seeds], fontsize=10)
ax.set_ylim([0.88, 0.92])
ax.set_ylabel('Best Validation AUC', fontsize=12)
ax.set_title('Phantom Lens V3 — Multi-Seed Training Results\nPillarAttentionClassifier | 29 live dims | 176,000 samples',
             fontsize=12, fontweight='bold')
for bar, auc in zip(bars, aucs):
    diff = auc - 0.8961
    mk   = f'+{diff:.4f}' if diff > 0 else f'{diff:.4f}'
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.0005,
            f'{auc:.4f}\n({mk})', ha='center', va='bottom', fontsize=9)
ax.legend(fontsize=11)
ax.spines[['top','right']].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'training_auc_summary.png'), dpi=150, bbox_inches='tight', facecolor='#F8F9FA')
plt.close()
print("Saved: training_auc_summary.png")

print("\nPCA analysis complete.")
print(f"Top PC1 features: {[DIM_NAMES[i] for i in np.argsort(np.abs(pca.components_[0]))[::-1][:3]]}")
print(f"Top PC2 features: {[DIM_NAMES[i] for i in np.argsort(np.abs(pca.components_[1]))[::-1][:3]]}")
print(f"Cumulative variance PC1+PC2: {(var_ex[0]+var_ex[1])*100:.1f}%")