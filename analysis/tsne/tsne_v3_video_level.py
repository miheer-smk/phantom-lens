# Phantom Lens Research Framework
# Developer: Miheer Satish Kulkarni | IIIT Nagpur | 2026
# All Rights Reserved

"""
Phantom Lens V3 — Video-Level t-SNE Visualisation
30-dimensional physics features (P1-P12)

Usage:
  phantomlens_env\Scripts\python.exe tsne_v3_video_level.py

Output folder: tsne_v3_results\
"""

import os
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# ── CONFIG ────────────────────────────────────────────────────────────────────

PKL_PATH   = "data/precomputed_features_v3_with_celebdf.pkl"
OUT_DIR    = "tsne_v3_results"
SAMPLE_CAP = 1500
TSNE_PERP  = 40
TSNE_ITER  = 1000
SEED       = 42

os.makedirs(OUT_DIR, exist_ok=True)
rng = np.random.default_rng(SEED)

# ── COLOURS ───────────────────────────────────────────────────────────────────

COLOUR = {
    'real':           '#4CAF50',
    'real_youtube':   '#4CAF50',
    'deepfakes':      '#F44336',
    'face2face':      '#FF9800',
    'faceshifter':    '#9C27B0',
    'faceswap':       '#E91E63',
    'neuraltextures': '#00BCD4',
    'fake_all':       '#F44336',
    'ffpp':           '#2196F3',
    'celebvhq':       '#FF9800',
    'celebdf':        '#9C27B0',
}

# ── LOAD PKL ──────────────────────────────────────────────────────────────────

print("=" * 60)
print("PHANTOM LENS V3 — Video-Level t-SNE")
print("=" * 60)
print(f"\nLoading {PKL_PATH} ...")

with open(PKL_PATH, 'rb') as f:
    data = pickle.load(f)

features  = np.array(data['features'],        dtype=np.float32)
labels    = np.array(data['labels'],          dtype=np.int32)
sources   = np.array(data['dataset_sources'])
gentypes  = np.array(data['generator_types'])
video_ids = np.array(data['video_ids'])

# Fix NaN/Inf from any bad extractions
features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

print(f"Total samples  : {len(labels):,}")
print(f"Feature dims   : {features.shape[1]}")
print(f"Real           : {(labels==0).sum():,}")
print(f"Fake           : {(labels==1).sum():,}")
print(f"Datasets       : {np.unique(sources)}")
print(f"Generator types: {np.unique(gentypes)}")

# ── AGGREGATE PER-FRAME → PER-VIDEO ──────────────────────────────────────────

unique_vids = np.unique(video_ids)
if len(unique_vids) < len(video_ids):
    print(f"\nAggregating {len(video_ids):,} frames → {len(unique_vids):,} videos...")
    agg_feat, agg_lab, agg_src, agg_gen = [], [], [], []
    for vid in unique_vids:
        m = video_ids == vid
        agg_feat.append(features[m].mean(axis=0))
        agg_lab.append(labels[m][0])
        agg_src.append(sources[m][0])
        agg_gen.append(gentypes[m][0])
    features = np.array(agg_feat, dtype=np.float32)
    labels   = np.array(agg_lab,  dtype=np.int32)
    sources  = np.array(agg_src)
    gentypes = np.array(agg_gen)
    print(f"Done — {len(labels):,} video-level vectors")
else:
    print("\nAlready video-level ✓")

# ── HELPERS ───────────────────────────────────────────────────────────────────

def sample(mask, n):
    idx = np.where(mask)[0]
    if len(idx) <= n:
        return idx
    return rng.choice(idx, n, replace=False)

def run_tsne(feat_subset):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(feat_subset)
    tsne = TSNE(n_components=2, perplexity=TSNE_PERP,
                n_iter=TSNE_ITER, random_state=SEED,
                n_jobs=-1, verbose=0)
    return tsne.fit_transform(scaled)

def dot(ax, emb, colour, label, alpha=0.55, size=18):
    ax.scatter(emb[:,0], emb[:,1], c=colour, label=label,
               s=size, alpha=alpha, linewidths=0.3,
               edgecolors='white', zorder=2)

def style(ax, title=""):
    ax.set_title(title, fontsize=10, fontweight='bold', pad=6)
    ax.set_xlabel("t-SNE Dim 1", fontsize=8)
    ax.set_ylabel("t-SNE Dim 2", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    leg = ax.legend(fontsize=7, markerscale=1.2,
                    framealpha=0.9, edgecolor='#cccccc')
    leg.get_frame().set_linewidth(0.5)

# ── MASKS ─────────────────────────────────────────────────────────────────────

ffpp_mask  = sources == 'ffpp_official'
cvhq_mask  = sources == 'celebvhq'
cdf_mask   = sources == 'celebdf'

real_mask  = labels == 0
fake_mask  = labels == 1

ffpp_real  = ffpp_mask & real_mask
ffpp_fake  = ffpp_mask & fake_mask
cvhq_real  = cvhq_mask & real_mask
cvhq_fake  = cvhq_mask & fake_mask

# ── PLOT 1 — FF++ Real vs All Fakes ──────────────────────────────────────────

print("\n[1/5] FF++ Real vs All Fakes...")
ir = sample(ffpp_real, SAMPLE_CAP)
if_= sample(ffpp_fake, SAMPLE_CAP)
idx = np.concatenate([ir, if_])
print(f"      real={len(ir):,}  fake={len(if_):,}  running t-SNE...", end='', flush=True)
emb = run_tsne(features[idx])
print(" done.")

lab = labels[idx]
fig, ax = plt.subplots(figsize=(8,6))
dot(ax, emb[lab==0], COLOUR['real'],     f"Real (YouTube) (n={len(ir)})")
dot(ax, emb[lab==1], COLOUR['fake_all'], f"Fake — all types (n={len(if_)})")
style(ax,
    "FF++  |  Real vs All Fakes  |  VIDEO-LEVEL\n"
    "Each dot = one video  |  All temporal physics pillars active")
fig.tight_layout()
p1 = os.path.join(OUT_DIR, "plot1_ffpp_real_vs_allfakes.png")
fig.savefig(p1, dpi=150, bbox_inches='tight')
plt.close()
print(f"      Saved → {p1}")

# ── PLOT 2 — FF++ Real vs Each Fake Type ─────────────────────────────────────

print("\n[2/5] FF++ Real vs Each Fake Type...")
fake_types = ['deepfakes','face2face','faceshifter','faceswap','neuraltextures']
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
axes = axes.flatten()

for i, ft in enumerate(fake_types):
    ax = axes[i]
    ft_mask = ffpp_mask & (gentypes == ft)
    ir = sample(ffpp_real, SAMPLE_CAP)
    if_ = sample(ft_mask, SAMPLE_CAP)
    if len(if_) == 0:
        ax.text(0.5, 0.5, f"No {ft} samples",
                ha='center', va='center', transform=ax.transAxes)
        continue
    idx = np.concatenate([ir, if_])
    print(f"      {ft}: real={len(ir)}, fake={len(if_)}  t-SNE...", end='', flush=True)
    emb = run_tsne(features[idx])
    print(" done.")
    lab = labels[idx]
    dot(ax, emb[lab==0], COLOUR['real'], f"Real (n={len(ir)})")
    dot(ax, emb[lab==1], COLOUR[ft],    f"{ft} (n={len(if_)})")
    style(ax, f"Real vs {ft}")

axes[5].set_visible(False)
fig.suptitle(
    "FF++  |  Real vs Each Fake Type  |  VIDEO-LEVEL\n"
    "Each dot = one full video  |  Phantom Lens V3 All Pillars",
    fontsize=13, fontweight='bold', y=1.01)
fig.tight_layout()
p2 = os.path.join(OUT_DIR, "plot2_ffpp_per_faketype.png")
fig.savefig(p2, dpi=150, bbox_inches='tight')
plt.close()
print(f"      Saved → {p2}")

# ── PLOT 3 — CelebVHQ Real vs Fake ───────────────────────────────────────────

print("\n[3/5] CelebVHQ Real vs Fake...")
nr = cvhq_real.sum()
nf = cvhq_fake.sum()
print(f"      CelebVHQ real={nr:,}  fake={nf:,}")

fig, ax = plt.subplots(figsize=(8,6))
if nr == 0 or nf == 0:
    ax.text(0.5, 0.5,
            f"CelebVHQ: real={nr}  fake={nf}\n"
            "Need both classes for separation plot",
            ha='center', va='center', fontsize=12,
            transform=ax.transAxes)
    style(ax, "CelebVHQ  |  Real vs Fake  |  VIDEO-LEVEL")
else:
    ir  = sample(cvhq_real, SAMPLE_CAP)
    if_ = sample(cvhq_fake, SAMPLE_CAP)
    idx = np.concatenate([ir, if_])
    print(f"      sampled real={len(ir)}, fake={len(if_)}  t-SNE...", end='', flush=True)
    emb = run_tsne(features[idx])
    print(" done.")
    lab = labels[idx]
    dot(ax, emb[lab==0], COLOUR['real'],     f"Real (n={len(ir)})")
    dot(ax, emb[lab==1], COLOUR['fake_all'], f"Fake (n={len(if_)})")
    style(ax,
        "CelebVHQ  |  Real vs Fake  |  VIDEO-LEVEL\n"
        "Each dot = one video  |  All physics pillars active")
fig.tight_layout()
p3 = os.path.join(OUT_DIR, "plot3_celebvhq_real_vs_fake.png")
fig.savefig(p3, dpi=150, bbox_inches='tight')
plt.close()
print(f"      Saved → {p3}")

# ── PLOT 4 — Codec Bias Diagnosis ────────────────────────────────────────────

print("\n[4/5] Codec Bias Diagnosis...")
all_sources = np.unique(sources)
idx_parts = []
for src in all_sources:
    idx_parts.append(sample(sources == src, SAMPLE_CAP))
idx = np.concatenate(idx_parts)
print(f"      total={len(idx):,}  t-SNE...", end='', flush=True)
emb = run_tsne(features[idx])
print(" done.")

src_arr = sources[idx]
lab_arr = labels[idx]

fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(16, 7))

# Left — by dataset
src_colours = [COLOUR.get(s, '#999999') for s in all_sources]
for src, col in zip(all_sources, src_colours):
    m = src_arr == src
    dot(ax_l, emb[m], col, f"{src} (n={m.sum()})", alpha=0.5)
style(ax_l,
    "Coloured by DATASET SOURCE\n← Codec bias: datasets separate cleanly")

# Right — by real/fake
nr = (lab_arr==0).sum()
nf = (lab_arr==1).sum()
dot(ax_r, emb[lab_arr==0], COLOUR['real'],     f"Real (n={nr})")
dot(ax_r, emb[lab_arr==1], COLOUR['fake_all'], f"Fake (n={nf})")
style(ax_r,
    "Coloured by REAL vs FAKE\n← Physics discrimination across datasets")

fig.suptitle(
    "Codec Bias Diagnosis  |  Same Data, Two Colourings  |  VIDEO-LEVEL\n"
    "Left shows dataset separation (problem)  |  "
    "Right shows real/fake separation (physics signal)",
    fontsize=11, fontweight='bold')
fig.tight_layout()
p4 = os.path.join(OUT_DIR, "plot4_codec_bias_diagnosis.png")
fig.savefig(p4, dpi=150, bbox_inches='tight')
plt.close()
print(f"      Saved → {p4}")

# ── PLOT 5 — 4-Panel Professor Diagram ───────────────────────────────────────

print("\n[5/5] 4-Panel Professor Diagram...")

panels = [
    ("DA — FF++ Real vs Deepfakes",
     ffpp_real,
     ffpp_mask & (gentypes == 'deepfakes'),
     COLOUR['deepfakes'], 'Deepfakes'),
    ("DB — FF++ Real vs All Fakes",
     ffpp_real,
     ffpp_fake,
     COLOUR['fake_all'], 'All Fakes'),
    ("DC — CelebVHQ Real vs Fake",
     cvhq_real,
     cvhq_fake,
     COLOUR['fake_all'], 'CelebVHQ Fake'),
    ("DD — All Sources Real vs Fake",
     real_mask,
     fake_mask,
     COLOUR['fake_all'], 'All Fakes'),
]

fig, axes = plt.subplots(2, 2, figsize=(16, 14))
for ax, (title, rm, fm, fc, flabel) in zip(axes.flatten(), panels):
    nr = rm.sum()
    nf = fm.sum()
    if nr == 0 or nf == 0:
        ax.text(0.5, 0.5, f"Insufficient data\nreal={nr}  fake={nf}",
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title, fontsize=10, fontweight='bold')
        continue
    cap = min(SAMPLE_CAP, nr, nf)
    ir  = sample(rm, cap)
    if_ = sample(fm, cap)
    idx = np.concatenate([ir, if_])
    print(f"      {title}: real={len(ir)}, fake={len(if_)}  t-SNE...",
          end='', flush=True)
    emb = run_tsne(features[idx])
    print(" done.")
    lab = labels[idx]
    dot(ax, emb[lab==0], COLOUR['real'], f"Real (n={len(ir)})")
    dot(ax, emb[lab==1], fc,             f"{flabel} (n={len(if_)})")
    style(ax, title)

fig.suptitle(
    "Phantom Lens V3  |  Physics Features t-SNE  |  VIDEO-LEVEL\n"
    "Each dot = one full video  |  All 10 pillars + temporal features active",
    fontsize=13, fontweight='bold')
fig.tight_layout()
p5 = os.path.join(OUT_DIR, "plot5_4panel_professor.png")
fig.savefig(p5, dpi=150, bbox_inches='tight')
plt.close()
print(f"      Saved → {p5}")

# ── DONE ─────────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("ALL DONE — Results in:", OUT_DIR)
print("="*60)
print(f"""
  plot1 — FF++ Real vs All Fakes       → plot1_ffpp_real_vs_allfakes.png
  plot2 — FF++ per fake type           → plot2_ffpp_per_faketype.png
  plot3 — CelebVHQ Real vs Fake        → plot3_celebvhq_real_vs_fake.png
  plot4 — Codec Bias Diagnosis         → plot4_codec_bias_diagnosis.png
  plot5 — 4-panel professor diagram    → plot5_4panel_professor.png

Open D:\\PhantomLens\\tsne_v3_results\\ in File Explorer to see plots.
""")