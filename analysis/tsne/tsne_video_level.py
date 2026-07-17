"""
Phantom Lens — Video-Level t-SNE on V2 PKL
Uses precomputed_features.pkl directly — no re-extraction needed

Run: phantomlens_env\Scripts\python.exe tsne_video_level.py
Time: ~10-15 minutes
Output: tsne_video_results\
"""

import pickle, os, warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')

# ── CONFIG ────────────────────────────────────────────────────────
PKL_PATH = "data/precomputed_features.pkl"
OUT_DIR  = "tsne_video_results"
SEED     = 42
os.makedirs(OUT_DIR, exist_ok=True)

# ── LOAD ──────────────────────────────────────────────────────────
print("Loading PKL...")
with open(PKL_PATH, "rb") as f:
    data = pickle.load(f)

X   = np.array(data['features'],        dtype=np.float32)
y   = np.array(data['labels'],          dtype=np.int32)
src = np.array(data['dataset_sources'])
gt  = np.array(data['generator_types'])

print(f"Total : {len(X)} videos | {X.shape[1]} dims")
print(f"FF++  : {(src=='ffpp_official').sum()} | "
      f"real={int(((src=='ffpp_official')&(y==0)).sum())} "
      f"fake={int(((src=='ffpp_official')&(y==1)).sum())}")
print(f"CelebVHQ: {(src=='celebvhq').sum()} | "
      f"real={int(((src=='celebvhq')&(y==0)).sum())} "
      f"fake={int(((src=='celebvhq')&(y==1)).sum())}")

# ── NORMALISE ─────────────────────────────────────────────────────
scaler = StandardScaler()
X_sc   = scaler.fit_transform(X)

# ── COLOURS ───────────────────────────────────────────────────────
TYPE_COL = {
    'real_youtube':   '#22c55e',
    'real':           '#16a34a',
    'deepfakes':      '#ef4444',
    'face2face':      '#f59e0b',
    'faceshifter':    '#8b5cf6',
    'faceswap':       '#ec4899',
    'neuraltextures': '#14b8a6',
}
SRC_COL = {
    'ffpp_official': '#3b82f6',
    'celebvhq':      '#f59e0b',
}

def run_tsne(X_sub, tag=""):
    perp = min(40, max(5, len(X_sub)//20))
    print(f"  t-SNE: {len(X_sub)} videos (perplexity={perp}) {tag}...",
          end=" ", flush=True)
    ts  = TSNE(n_components=2, perplexity=perp,
               random_state=SEED, n_iter=1000, verbose=0)
    emb = ts.fit_transform(X_sub)
    print("done.")
    return emb

def subsample(mask_real, mask_fake, n=1500, seed=SEED):
    rng = np.random.default_rng(seed)
    ri  = np.where(mask_real)[0]
    fi  = np.where(mask_fake)[0]
    n   = min(n, len(ri), len(fi))
    ri  = rng.choice(ri, n, replace=False)
    fi  = rng.choice(fi, n, replace=False)
    idx = np.concatenate([ri, fi])
    rng.shuffle(idx)
    return idx, n

def patch(label, color):
    return mpatches.Patch(color=color, label=label)

# ═══════════════════════════════════════════════════════════════
# PLOT 1 — FF++ : Real vs All Fakes (video level)
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("PLOT 1: FF++ Real vs All Fakes (VIDEO level)")
print("="*55)

ff_mask  = src == 'ffpp_official'
idx1, n1 = subsample(ff_mask & (y==0), ff_mask & (y==1), n=1500)
emb1     = run_tsne(X_sc[idx1], "FF++ all")

fig, ax = plt.subplots(figsize=(10,8))
for cls, col, lbl in [(0,'#22c55e','Real (YouTube)'),
                       (1,'#ef4444','Fake — all types')]:
    m = y[idx1] == cls
    ax.scatter(emb1[m,0], emb1[m,1], c=col,
               alpha=0.5, s=18,
               label=f"{lbl} (n={m.sum()})", zorder=3-cls)
ax.set_title("FF++  |  Real vs All Fakes  |  VIDEO-LEVEL\n"
             "Each dot = one video  |  All temporal physics pillars active",
             fontsize=13, fontweight='bold')
ax.legend(fontsize=11, markerscale=2)
ax.set_xlabel("t-SNE Dim 1", fontsize=11)
ax.set_ylabel("t-SNE Dim 2", fontsize=11)
ax.grid(True, alpha=0.2)
plt.tight_layout()
p = os.path.join(OUT_DIR, "plot1_ffpp_real_vs_allfakes_VIDEO.png")
plt.savefig(p, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved → {p}")

# ═══════════════════════════════════════════════════════════════
# PLOT 2 — FF++ : Real vs Each Fake Type (video level)
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("PLOT 2: FF++ Real vs Each Fake Type (VIDEO level)")
print("="*55)

fake_types = ['deepfakes','face2face','faceshifter',
              'faceswap','neuraltextures']
real_ff_mask = (src=='ffpp_official') & (gt=='real_youtube')

fig, axes = plt.subplots(2, 3, figsize=(18,12))
axes_flat  = axes.flatten()

for i, ftype in enumerate(fake_types):
    ax       = axes_flat[i]
    fake_mask = (src=='ffpp_official') & (gt==ftype)
    idx_i, n_i = subsample(real_ff_mask, fake_mask, n=1000)

    print(f"  {ftype}: real={n_i}, fake={n_i}")
    emb_i = run_tsne(X_sc[idx_i], ftype)

    for cls, col, lbl in [(0,'#22c55e','Real'),
                           (1, TYPE_COL[ftype], ftype)]:
        m = y[idx_i] == cls
        ax.scatter(emb_i[m,0], emb_i[m,1], c=col,
                   alpha=0.5, s=16,
                   label=f"{lbl} (n={m.sum()})", zorder=3-cls)

    ax.set_title(f"Real  vs  {ftype}", fontsize=11, fontweight='bold')
    ax.legend(fontsize=9, markerscale=2)
    ax.grid(True, alpha=0.2)
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")

axes_flat[-1].axis('off')
fig.suptitle("FF++  |  Real vs Each Fake Type  |  VIDEO-LEVEL\n"
             "Each dot = one full video  |  Phantom Lens V2 All Pillars",
             fontsize=14, fontweight='bold')
plt.tight_layout()
p = os.path.join(OUT_DIR, "plot2_ffpp_per_faketype_VIDEO.png")
plt.savefig(p, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved → {p}")

# ═══════════════════════════════════════════════════════════════
# PLOT 3 — CelebVHQ : Real vs Fake (video level)
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("PLOT 3: CelebVHQ Real vs Fake (VIDEO level)")
print("="*55)

cv_mask   = src == 'celebvhq'
idx3, n3  = subsample(cv_mask & (y==0), cv_mask & (y==1), n=1500)
emb3      = run_tsne(X_sc[idx3], "CelebVHQ")

fig, ax = plt.subplots(figsize=(10,8))
for cls, col, lbl in [(0,'#22c55e','Real'),
                       (1,'#ef4444','Fake')]:
    m = y[idx3] == cls
    ax.scatter(emb3[m,0], emb3[m,1], c=col,
               alpha=0.5, s=18,
               label=f"{lbl} (n={m.sum()})", zorder=3-cls)
ax.set_title("CelebVHQ  |  Real vs Fake  |  VIDEO-LEVEL\n"
             "Each dot = one video  |  All physics pillars active",
             fontsize=13, fontweight='bold')
ax.legend(fontsize=11, markerscale=2)
ax.set_xlabel("t-SNE Dim 1"); ax.set_ylabel("t-SNE Dim 2")
ax.grid(True, alpha=0.2)
plt.tight_layout()
p = os.path.join(OUT_DIR, "plot3_celebvhq_real_vs_fake_VIDEO.png")
plt.savefig(p, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved → {p}")

# ═══════════════════════════════════════════════════════════════
# PLOT 4 — Dataset Confound (the smoking gun)
# Same data coloured two ways: by source vs by real/fake
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("PLOT 4: Dataset Confound — Source vs Label colouring")
print("="*55)

rng4  = np.random.default_rng(SEED)
n4    = 1500
ff_idx = rng4.choice(np.where(src=='ffpp_official')[0],
                      min(n4, (src=='ffpp_official').sum()),
                      replace=False)
cv_idx = rng4.choice(np.where(src=='celebvhq')[0],
                      min(n4, (src=='celebvhq').sum()),
                      replace=False)
idx4   = np.concatenate([ff_idx, cv_idx])
rng4.shuffle(idx4)

print(f"  FF++={len(ff_idx)}, CelebVHQ={len(cv_idx)}")
emb4 = run_tsne(X_sc[idx4], "confound")

fig, axes = plt.subplots(1, 2, figsize=(18,8))

# Left — colour by dataset source
for s, col in SRC_COL.items():
    m = src[idx4] == s
    axes[0].scatter(emb4[m,0], emb4[m,1], c=col,
                    alpha=0.45, s=14,
                    label=f"{s} (n={m.sum()})", zorder=2)
axes[0].set_title("Coloured by DATASET SOURCE\n"
                  "← Codec bias: datasets separate cleanly",
                  fontsize=12, fontweight='bold')
axes[0].legend(fontsize=10, markerscale=2)
axes[0].set_xlabel("t-SNE Dim 1"); axes[0].set_ylabel("t-SNE Dim 2")
axes[0].grid(True, alpha=0.2)

# Right — colour by real/fake
for cls, col, lbl in [(0,'#22c55e','Real'), (1,'#ef4444','Fake')]:
    m = y[idx4] == cls
    axes[1].scatter(emb4[m,0], emb4[m,1], c=col,
                    alpha=0.45, s=14,
                    label=f"{lbl} (n={m.sum()})", zorder=2)
axes[1].set_title("Coloured by REAL vs FAKE\n"
                  "← Physics discrimination across datasets",
                  fontsize=12, fontweight='bold')
axes[1].legend(fontsize=10, markerscale=2)
axes[1].set_xlabel("t-SNE Dim 1"); axes[1].set_ylabel("t-SNE Dim 2")
axes[1].grid(True, alpha=0.2)

fig.suptitle("Codec Bias Diagnosis  |  Same Data, Two Colourings  |  VIDEO-LEVEL\n"
             "Left shows dataset separation (problem)  |  "
             "Right shows real/fake separation (physics signal)",
             fontsize=13, fontweight='bold')
plt.tight_layout()
p = os.path.join(OUT_DIR, "plot4_confound_source_vs_label_VIDEO.png")
plt.savefig(p, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved → {p}")

# ═══════════════════════════════════════════════════════════════
# PLOT 5 — 4-PANEL matching professor's diagram (VIDEO level)
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("PLOT 5: 4-panel professor diagram (VIDEO level)")
print("="*55)

fig, axes = plt.subplots(2, 2, figsize=(16,14))

panels = [
    ("DA — FF++ Real vs Deepfakes",
     (src=='ffpp_official')&(gt=='real_youtube'),
     (src=='ffpp_official')&(gt=='deepfakes'),
     '#ef4444', 'Deepfakes'),
    ("DB — FF++ Real vs All Fakes",
     (src=='ffpp_official')&(y==0),
     (src=='ffpp_official')&(y==1),
     '#ef4444', 'All Fakes'),
    ("DC — CelebVHQ Real vs Fake",
     (src=='celebvhq')&(y==0),
     (src=='celebvhq')&(y==1),
     '#f59e0b', 'CelebVHQ Fake'),
    ("DD — All Sources Real vs Fake",
     y==0, y==1,
     '#ef4444', 'All Fakes'),
]

for idx5, (title, real_m, fake_m, fcol, flbl) in enumerate(panels):
    ax       = axes[idx5//2][idx5%2]
    idx_p, n_p = subsample(real_m, fake_m, n=800)
    print(f"  {title}: real={n_p}, fake={n_p}")
    emb_p = run_tsne(X_sc[idx_p], title)

    for cls, col, lbl in [(0,'#22c55e','Real'),
                           (1, fcol, flbl)]:
        m = y[idx_p] == cls
        ax.scatter(emb_p[m,0], emb_p[m,1], c=col,
                   alpha=0.5, s=16,
                   label=f"{lbl} (n={m.sum()})", zorder=3-cls)

    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, markerscale=2)
    ax.grid(True, alpha=0.2)
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")

fig.suptitle("Phantom Lens V2  |  Physics Features t-SNE  |  VIDEO-LEVEL\n"
             "Each dot = one full video  |  All 10 pillars + temporal features active",
             fontsize=14, fontweight='bold')
plt.tight_layout()
p = os.path.join(OUT_DIR, "plot5_4panel_professor_VIDEO.png")
plt.savefig(p, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved → {p}")

# ═══════════════════════════════════════════════════════════════
# DONE
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("ALL DONE — Results in:", OUT_DIR)
print("="*55)
print()
print("Show professor in this order:")
print("  1. plot5  ← matches your diagram, video level")
print("  2. plot2  ← per fake type separation")
print("  3. plot4  ← codec bias smoking gun (LEFT=problem, RIGHT=physics)")
print("  4. plot1  ← overall FF++ separation")
print("  5. plot3  ← CelebVHQ separation")
print()
print("Key message:")
print("  Video-level = temporal pillars active = stronger separation")
print("  Compare with image-level plots to show temporal pillars matter")