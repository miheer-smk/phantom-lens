"""
Phantom Lens — Complete Analysis for Professor
Generates:
  1. t-SNE per individual dataset (V2 and V3)
  2. t-SNE on mixed dataset (V2 and V3)
  3. Detailed dataset training table
Author: Miheer Satish Kulkarni, IIIT Nagpur
"""
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
import os
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = "results/professor_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── COLORS ────────────────────────────────────────────────────────────────────
REAL_C  = '#185FA5'
FAKE_C  = '#A32D2D'
CELEB_C = '#3B6D11'
FFPP_C  = '#854F0B'
CDF_C   = '#533AB7'

# ── LOAD PKLS ─────────────────────────────────────────────────────────────────
print("Loading PKLs...")

with open('data/precomputed_features.pkl', 'rb') as f:
    v2 = pickle.load(f)
v2_X   = np.array(v2['features'], dtype=np.float32)
v2_y   = np.array(v2['labels'],   dtype=np.int32)
v2_src = np.array(v2['dataset_sources'])
print(f"V2: {v2_X.shape} | Real:{(v2_y==0).sum()} Fake:{(v2_y==1).sum()}")

with open('data/precomputed_features_v3_base.pkl', 'rb') as f:
    v3b = pickle.load(f)
v3b_X   = np.array(v3b['features'], dtype=np.float32)
v3b_y   = np.array(v3b['labels'],   dtype=np.int32)
v3b_src = np.array(v3b['dataset_sources'])
print(f"V3 base: {v3b_X.shape} | Real:{(v3b_y==0).sum()} Fake:{(v3b_y==1).sum()}")

with open('data/precomputed_features_v3_with_celebdf.pkl', 'rb') as f:
    v3f = pickle.load(f)
v3f_X   = np.array(v3f['features'], dtype=np.float32)
v3f_y   = np.array(v3f['labels'],   dtype=np.int32)
v3f_src = np.array(v3f['dataset_sources'])
print(f"V3 full: {v3f_X.shape} | Real:{(v3f_y==0).sum()} Fake:{(v3f_y==1).sum()}")

# ── HELPER: RUN TSNE ──────────────────────────────────────────────────────────
def run_tsne(X, seed=42, n=2000):
    np.random.seed(seed)
    if len(X) > n:
        idx = np.random.choice(len(X), n, replace=False)
        X = X[idx]
        return_idx = idx
    else:
        return_idx = np.arange(len(X))
    sc  = StandardScaler()
    Xs  = sc.fit_transform(X)
    tsne = TSNE(n_components=2, perplexity=35, n_iter=800,
                random_state=seed, init='pca', learning_rate='auto')
    emb = tsne.fit_transform(Xs)
    return emb, return_idx

def sep(emb, y):
    if len(np.unique(y)) < 2:
        return 0.0
    return float(np.linalg.norm(
        emb[y==0].mean(0) - emb[y==1].mean(0)))

# ── TASK 1: t-SNE PER INDIVIDUAL DATASET ──────────────────────────────────────
print("\n" + "="*55)
print("TASK 1 — t-SNE per individual dataset")
print("="*55)

datasets_v2 = {
    'FF++ (V2 features)':     (v2_X[v2_src=='ffpp_official'],
                                v2_y[v2_src=='ffpp_official']),
    'CelebVHQ (V2 features)': (v2_X[v2_src=='celebvhq'],
                                v2_y[v2_src=='celebvhq']),
}
datasets_v3 = {
    'FF++ (V3 features)':      (v3b_X[v3b_src=='ffpp_official'],
                                 v3b_y[v3b_src=='ffpp_official']),
    'CelebVHQ (V3 features)':  (v3b_X[v3b_src=='celebvhq'],
                                 v3b_y[v3b_src=='celebvhq']),
    'CelebDF fake (V3 full)':  (v3f_X[v3f_src=='celebdf_fake'],
                                 v3f_y[v3f_src=='celebdf_fake']),
}

all_datasets = {**datasets_v2, **datasets_v3}
n_plots = len(all_datasets)
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.patch.set_facecolor('#F8F9FA')
axes = axes.flatten()

seps = {}
for ax, (name, (X, y)) in zip(axes, all_datasets.items()):
    ax.set_facecolor('white')
    if len(X) < 20:
        ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center',
                transform=ax.transAxes)
        ax.set_title(name, fontsize=10, fontweight='bold')
        continue

    print(f"  Running t-SNE: {name} ({len(X)} samples)...")
    emb, idx = run_tsne(X, n=1500)
    y_sub = y[idx]
    s = sep(emb, y_sub)
    seps[name] = s

    ax.scatter(emb[y_sub==1,0], emb[y_sub==1,1],
               c=FAKE_C, s=8, alpha=0.4, linewidths=0, label='Fake')
    ax.scatter(emb[y_sub==0,0], emb[y_sub==0,1],
               c=REAL_C, s=8, alpha=0.4, linewidths=0, label='Real')

    if (y_sub==0).sum() > 0:
        rc = emb[y_sub==0].mean(0)
        ax.plot(*rc, 'o', color=REAL_C, ms=10, zorder=5)
    if (y_sub==1).sum() > 0:
        fc = emb[y_sub==1].mean(0)
        ax.plot(*fc, 'o', color=FAKE_C, ms=10, zorder=5)

    real_n = int((y_sub==0).sum())
    fake_n = int((y_sub==1).sum())
    ax.set_title(f'{name}\nReal={real_n} Fake={fake_n} | Sep={s:.1f}',
                 fontsize=9, fontweight='bold')
    ax.legend(markerscale=3, fontsize=8)
    ax.set_xlabel('t-SNE 1', fontsize=8)
    ax.set_ylabel('t-SNE 2', fontsize=8)
    ax.spines[['top','right']].set_visible(False)

# hide last empty axis
axes[-1].set_visible(False)

plt.suptitle('Phantom Lens — t-SNE Per Individual Dataset\n'
             'V2 (24-dim) and V3 (30-dim) Feature Versions',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
out1 = os.path.join(OUTPUT_DIR, 'task1_tsne_per_dataset.png')
plt.savefig(out1, dpi=150, bbox_inches='tight', facecolor='#F8F9FA')
plt.close()
print(f"  Saved: {out1}")

print(f"\n  Cluster separation summary:")
for name, s in seps.items():
    print(f"    {name:<35} sep={s:.2f}")

# ── TASK 2: t-SNE ON MIXED DATASET ────────────────────────────────────────────
print("\n" + "="*55)
print("TASK 2 — t-SNE on mixed dataset")
print("="*55)

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.patch.set_facecolor('#F8F9FA')

# V2 mixed — real vs fake
print("  V2 mixed t-SNE (real vs fake)...")
np.random.seed(42)
n_each = 800
idx_r = np.random.choice(np.where(v2_y==0)[0], min(n_each,(v2_y==0).sum()), replace=False)
idx_f = np.random.choice(np.where(v2_y==1)[0], min(n_each,(v2_y==1).sum()), replace=False)
idx_v2 = np.concatenate([idx_r, idx_f])
X_v2m = v2_X[idx_v2]; y_v2m = v2_y[idx_v2]; s_v2m = v2_src[idx_v2]
emb_v2, _ = run_tsne(X_v2m, n=len(X_v2m))
s_v2 = sep(emb_v2, y_v2m)

ax = axes[0,0]; ax.set_facecolor('white')
ax.scatter(emb_v2[y_v2m==1,0], emb_v2[y_v2m==1,1], c=FAKE_C, s=8, alpha=0.4, linewidths=0, label='Fake')
ax.scatter(emb_v2[y_v2m==0,0], emb_v2[y_v2m==0,1], c=REAL_C, s=8, alpha=0.4, linewidths=0, label='Real')
ax.set_title(f'V2 Mixed — Real vs Fake\nSep={s_v2:.1f}', fontsize=10, fontweight='bold')
ax.legend(markerscale=3, fontsize=8); ax.spines[['top','right']].set_visible(False)

# V2 mixed — by source
ax = axes[0,1]; ax.set_facecolor('white')
for src, color, mk in [('ffpp_official',FFPP_C,'o'),('celebvhq',CELEB_C,'s')]:
    mask = s_v2m == src
    ax.scatter(emb_v2[mask,0], emb_v2[mask,1], c=color, s=8, alpha=0.4,
               linewidths=0, marker=mk, label=src)
ax.set_title(f'V2 Mixed — By Dataset Source\nSep={s_v2:.1f}', fontsize=10, fontweight='bold')
ax.legend(markerscale=3, fontsize=8); ax.spines[['top','right']].set_visible(False)

# V2 mixed — by source AND label
ax = axes[0,2]; ax.set_facecolor('white')
combos = [('ffpp_official',0,FFPP_C,'^','FF++ Real'),
          ('ffpp_official',1,FAKE_C,'v','FF++ Fake'),
          ('celebvhq',0,CELEB_C,'^','CelebVHQ Real')]
for src, lbl, color, mk, name in combos:
    mask = (s_v2m==src) & (y_v2m==lbl)
    ax.scatter(emb_v2[mask,0], emb_v2[mask,1], c=color, s=8, alpha=0.4,
               linewidths=0, marker=mk, label=name)
ax.set_title('V2 Mixed — Source + Label', fontsize=10, fontweight='bold')
ax.legend(markerscale=3, fontsize=7); ax.spines[['top','right']].set_visible(False)

# V3 full mixed
print("  V3 full mixed t-SNE...")
np.random.seed(42)
n_each3 = 600
idx_r3 = np.random.choice(np.where(v3f_y==0)[0], min(n_each3,(v3f_y==0).sum()), replace=False)
idx_f3 = np.random.choice(np.where(v3f_y==1)[0], min(n_each3*2,(v3f_y==1).sum()), replace=False)
idx_v3 = np.concatenate([idx_r3, idx_f3])
X_v3m = v3f_X[idx_v3]; y_v3m = v3f_y[idx_v3]; s_v3m = v3f_src[idx_v3]
emb_v3, _ = run_tsne(X_v3m, n=len(X_v3m))
s_v3 = sep(emb_v3, y_v3m)

ax = axes[1,0]; ax.set_facecolor('white')
ax.scatter(emb_v3[y_v3m==1,0], emb_v3[y_v3m==1,1], c=FAKE_C, s=8, alpha=0.4, linewidths=0, label='Fake')
ax.scatter(emb_v3[y_v3m==0,0], emb_v3[y_v3m==0,1], c=REAL_C, s=8, alpha=0.4, linewidths=0, label='Real')
ax.set_title(f'V3 Full Mixed — Real vs Fake\nSep={s_v3:.1f}', fontsize=10, fontweight='bold')
ax.legend(markerscale=3, fontsize=8); ax.spines[['top','right']].set_visible(False)

ax = axes[1,1]; ax.set_facecolor('white')
for src, color, mk in [('ffpp_official',FFPP_C,'o'),('celebvhq',CELEB_C,'s'),('celebdf_fake',CDF_C,'D')]:
    mask = s_v3m == src
    ax.scatter(emb_v3[mask,0], emb_v3[mask,1], c=color, s=8, alpha=0.4,
               linewidths=0, marker=mk, label=src)
ax.set_title(f'V3 Full Mixed — By Dataset Source\nSep={s_v3:.1f}', fontsize=10, fontweight='bold')
ax.legend(markerscale=3, fontsize=7); ax.spines[['top','right']].set_visible(False)

ax = axes[1,2]; ax.set_facecolor('white')
combos3 = [('ffpp_official',0,FFPP_C,'^','FF++ Real'),
           ('ffpp_official',1,FAKE_C,'v','FF++ Fake'),
           ('celebvhq',0,CELEB_C,'^','CelebVHQ Real'),
           ('celebdf_fake',1,CDF_C,'v','CelebDF Fake')]
for src, lbl, color, mk, name in combos3:
    mask = (s_v3m==src) & (y_v3m==lbl)
    ax.scatter(emb_v3[mask,0], emb_v3[mask,1], c=color, s=8, alpha=0.4,
               linewidths=0, marker=mk, label=name)
ax.set_title('V3 Full Mixed — Source + Label', fontsize=10, fontweight='bold')
ax.legend(markerscale=3, fontsize=7); ax.spines[['top','right']].set_visible(False)

plt.suptitle('Phantom Lens — t-SNE on Mixed Dataset\n'
             'Top row: V2 (24-dim) | Bottom row: V3 Full (30-dim + CelebDF)',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
out2 = os.path.join(OUTPUT_DIR, 'task2_tsne_mixed_dataset.png')
plt.savefig(out2, dpi=150, bbox_inches='tight', facecolor='#F8F9FA')
plt.close()
print(f"  Saved: {out2}")

# ── TASK 3: DETAILED DATASET TABLE ────────────────────────────────────────────
print("\n" + "="*55)
print("TASK 3 — Detailed dataset training table")
print("="*55)

LIVE_V2 = list(range(24))
LIVE_V3 = [i for i in range(30) if i != 20]

PILLAR_NAMES_V2 = [
    'P1_vmr','P1_resStd','P1_hfRatio',
    'P2_prnu_E','P2_faceRatio',
    'P3_rgCorr','P3_bgCorr',
    'P4_faceBG','P4_specular','P4_shadow',
    'P5_driftM','P5_driftS',
    'P6_benford','P6_blockArt','P6_dblComp',
    'P7_resM','P7_resV',
    'P8_blurMag','P8_blurDir',
    'P9_flowMag','P9_boundary',
    'P10_rgShift','P10_bgShift','P10_edgeCtr',
]

def cohens_d(a, b):
    pooled = np.sqrt((a.std()**2 + b.std()**2) / 2 + 1e-8)
    return float(abs(a.mean() - b.mean()) / pooled)

def per_source_auc(X, y, src, source_name, live_dims):
    mask = src == source_name
    Xm = X[mask][:, live_dims]
    ym = y[mask]
    if len(np.unique(ym)) < 2 or len(ym) < 20:
        return None
    sc = StandardScaler()
    Xs = sc.fit_transform(Xm)
    lr = LogisticRegression(max_iter=300, C=1.0)
    from sklearn.model_selection import cross_val_score
    aucs = cross_val_score(lr, Xs, ym, cv=3, scoring='roc_auc')
    return float(aucs.mean())

# Compute pillar AUCs per dataset for V2
print("\n  Computing pillar AUCs for V2...")
pillar_groups_v2 = [[0,1,2],[3,4],[5,6],[7,8,9],[10,11],
                    [12,13,14],[15,16],[17,18],[19,20],[21,22,23]]
pillar_names_short = ['P1 Noise','P2 PRNU','P3 Bayer','P4 Shadow',
                      'P5 Specular','P6 DCT','P7 Codec','P8 Blur',
                      'P9 Flow','P10 Chromatic']

from sklearn.model_selection import cross_val_score

def pillar_auc(X, y, dims):
    if len(np.unique(y)) < 2:
        return 0.5
    Xp = X[:, dims]
    sc = StandardScaler()
    Xs = sc.fit_transform(Xp)
    lr = LogisticRegression(max_iter=300)
    try:
        return float(cross_val_score(lr, Xs, y, cv=3, scoring='roc_auc').mean())
    except:
        return 0.5

# Dataset breakdown
print("\n  Building dataset breakdown...")
report_lines = []
report_lines.append("="*70)
report_lines.append("PHANTOM LENS — DETAILED DATASET REPORT")
report_lines.append("="*70)

# V2 breakdown
report_lines.append("\nV2 PKL (24-dim features)")
report_lines.append("-"*50)
for src in sorted(set(v2_src)):
    mask = v2_src == src
    real = int(((v2_y==0) & mask).sum())
    fake = int(((v2_y==1) & mask).sum())
    total = int(mask.sum())
    report_lines.append(f"  {src:<20} total={total:6d}  real={real:6d}  fake={fake:6d}")
report_lines.append(f"  {'TOTAL':<20} total={len(v2_y):6d}  real={(v2_y==0).sum():6d}  fake={(v2_y==1).sum():6d}")

# V3 base breakdown
report_lines.append("\nV3 BASE PKL (30-dim features, FF++ + CelebVHQ)")
report_lines.append("-"*50)
for src in sorted(set(v3b_src)):
    mask = v3b_src == src
    real = int(((v3b_y==0) & mask).sum())
    fake = int(((v3b_y==1) & mask).sum())
    total = int(mask.sum())
    report_lines.append(f"  {src:<20} total={total:6d}  real={real:6d}  fake={fake:6d}")
report_lines.append(f"  {'TOTAL':<20} total={len(v3b_y):6d}  real={(v3b_y==0).sum():6d}  fake={(v3b_y==1).sum():6d}")

# V3 full breakdown
report_lines.append("\nV3 FULL PKL (30-dim features, FF++ + CelebVHQ + CelebDF fake)")
report_lines.append("-"*50)
for src in sorted(set(v3f_src)):
    mask = v3f_src == src
    real = int(((v3f_y==0) & mask).sum())
    fake = int(((v3f_y==1) & mask).sum())
    total = int(mask.sum())
    report_lines.append(f"  {src:<20} total={total:6d}  real={real:6d}  fake={fake:6d}")
report_lines.append(f"  {'TOTAL':<20} total={len(v3f_y):6d}  real={(v3f_y==0).sum():6d}  fake={(v3f_y==1).sum():6d}")

# Train/test split (80/20)
report_lines.append("\nTRAIN/TEST SPLIT (80/20 video-level)")
report_lines.append("-"*50)
for pkl_name, X, y, src in [
    ("V2", v2_X, v2_y, v2_src),
    ("V3 base", v3b_X, v3b_y, v3b_src),
    ("V3 full", v3f_X, v3f_y, v3f_src),
]:
    n = len(y)
    train = int(n * 0.80)
    val   = n - train
    report_lines.append(f"  {pkl_name:<10} total={n:6d}  train~{train:6d}  val~{val:6d}")

# Per-pillar AUC on full dataset
report_lines.append("\nPER-PILLAR AUC — V2 features on full V2 pkl")
report_lines.append("-"*50)
report_lines.append(f"  {'Pillar':<15} {'AUC':>6}  {'Cohen d':>8}  {'Discriminative?':>16}")
report_lines.append("  " + "-"*48)

for i, (name, dims) in enumerate(zip(pillar_names_short, pillar_groups_v2)):
    auc = pillar_auc(v2_X, v2_y, dims)
    real_vals = v2_X[v2_y==0][:, dims[0]]
    fake_vals = v2_X[v2_y==1][:, dims[0]]
    d = cohens_d(real_vals, fake_vals)
    level = "STRONG" if auc > 0.80 else "MODERATE" if auc > 0.70 else "WEAK"
    report_lines.append(f"  {name:<15} {auc:>6.4f}  {d:>8.3f}  {level:>16}")
    print(f"    {name}: AUC={auc:.4f} d={d:.3f}")

# Per-pillar AUC on V3 pkl
report_lines.append("\nPER-PILLAR AUC — V3 features on V3 base pkl")
report_lines.append("-"*50)
pillar_groups_v3 = [[0,1,2],[3,4],[5,6],[7,8,9],[10,11],
                    [12,13,14],[15,16],[17,18],[19],
                    [20,21,22],[23,24,25],[26,27,28]]
pillar_names_v3 = ['P1 Noise','P2 PRNU','P3 Bayer','P4 Shadow',
                   'P5 Specular','P6 DCT','P7 Codec','P8 Blur',
                   'P9 Flow','P10 Chromatic','P11 EyeSym','P12 Illum']

LIVE_V3_dims = [i for i in range(30) if i != 20]
X_v3_live = v3b_X[:, LIVE_V3_dims]

report_lines.append(f"  {'Pillar':<15} {'AUC':>6}  {'Cohen d':>8}  {'Discriminative?':>16}")
report_lines.append("  " + "-"*48)

for name, dims in zip(pillar_names_v3, pillar_groups_v3):
    valid_dims = [d for d in dims if d < X_v3_live.shape[1]]
    if not valid_dims:
        continue
    auc = pillar_auc(X_v3_live, v3b_y, valid_dims)
    real_vals = X_v3_live[v3b_y==0][:, valid_dims[0]]
    fake_vals = X_v3_live[v3b_y==1][:, valid_dims[0]]
    d = cohens_d(real_vals, fake_vals)
    level = "STRONG" if auc > 0.80 else "MODERATE" if auc > 0.70 else "WEAK"
    report_lines.append(f"  {name:<15} {auc:>6.4f}  {d:>8.3f}  {level:>16}")
    print(f"    {name}: AUC={auc:.4f} d={d:.3f}")

# Dataset confounding
report_lines.append("\nDATASET CONFOUNDING ANALYSIS")
report_lines.append("-"*50)
report_lines.append("  Can features predict dataset origin (not real/fake)?")

for pkl_name, X, y, src in [
    ("V2", v2_X, v2_y, v2_src),
    ("V3 base", v3b_X[:, LIVE_V3_dims], v3b_y, v3b_src),
]:
    y_src = (src == 'ffpp_official').astype(int)
    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    lr = LogisticRegression(max_iter=200)
    auc = cross_val_score(lr, Xs, y_src, cv=3, scoring='roc_auc').mean()
    report_lines.append(f"  {pkl_name:<12} Dataset prediction AUC: {auc:.4f}  "
                        f"({'CONFOUNDED' if auc>0.70 else 'OK'})")

# Cross-dataset results
report_lines.append("\nCROSS-DATASET EVALUATION RESULTS")
report_lines.append("-"*50)
report_lines.append("  Model tested on Celeb-DF v2 (zero-shot, never seen during training)")
report_lines.append(f"  {'Model':<30} {'Cross-Dataset AUC':>18}")
report_lines.append("  " + "-"*50)
report_lines.append(f"  {'V1 MLP baseline':<30} {'0.4923':>18}")
report_lines.append(f"  {'V3 with codec norm':<30} {'0.4324':>18}")
report_lines.append(f"  {'V3 without codec norm':<30} {'0.5551 (best)':>18}")
report_lines.append(f"  {'V3 + CelebDF fake training':<30} {'0.4451':>18}")

report_lines.append("\n" + "="*70)
report_lines.append("END OF REPORT")
report_lines.append("="*70)

report_path = os.path.join(OUTPUT_DIR, 'task3_detailed_dataset_report.txt')
with open(report_path, 'w') as f:
    f.write('\n'.join(report_lines))
print(f"\n  Saved: {report_path}")

# Print to console too
for line in report_lines:
    print(line)

print(f"\n{'='*55}")
print("ALL TASKS COMPLETE")
print(f"{'='*55}")
print(f"Output folder: {OUTPUT_DIR}")
print(f"  task1_tsne_per_dataset.png")
print(f"  task2_tsne_mixed_dataset.png")
print(f"  task3_detailed_dataset_report.txt")