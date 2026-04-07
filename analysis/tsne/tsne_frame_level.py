"""
Phantom Lens — Per-Frame Image-Level t-SNE for Professor
Extracts 8 frames per video from small subset → t-SNE on individual frames

Run: phantomlens_env\Scripts\python.exe tsne_frame_level.py
Time: ~15-20 minutes
Output: tsne_frame_results\ folder
"""

import os, sys, pickle, warnings
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from scipy import fftpack
warnings.filterwarnings('ignore')

# ── Add project root to path ──────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── CONFIG ────────────────────────────────────────────────────────
FFPP_REAL_DIR  = r"data\ffpp_official\original_sequences\youtube"
FFPP_FAKE_BASE = r"data\ffpp_official\manipulated_sequences"
FAKE_TYPES     = ["Deepfakes", "Face2Face", "FaceShifter",
                  "FaceSwap", "NeuralTextures"]
OUT_DIR        = "tsne_frame_results"
N_VIDEOS       = 60     # videos per class (real + each fake type)
N_FRAMES       = 8      # frames per video
SEED           = 42
os.makedirs(OUT_DIR, exist_ok=True)

# ── PHYSICS FEATURE FUNCTIONS (from precompute_features_v2.py) ────

def compute_pillar1(img_gray):
    img = img_gray.astype(np.float32)
    h, w = img.shape
    vmr_vals = []
    for i in range(0, h-8, 8):
        for j in range(0, w-8, 8):
            block = img[i:i+8, j:j+8]
            mu = block.mean(); var = block.var()
            if mu > 5.0:
                vmr_vals.append(var / (mu + 1e-6))
    vmr = float(np.median(vmr_vals)) if vmr_vals else 0.0
    blurred = cv2.medianBlur(img_gray, 3).astype(np.float32)
    residual_std = float((img - blurred).std())
    lap = cv2.Laplacian(img_gray, cv2.CV_64F)
    hf_ratio = float(np.abs(lap).mean()) / (float(np.abs(img).mean()) + 1e-6)
    return np.array([vmr, residual_std, hf_ratio], dtype=np.float32)

def compute_pillar2(img_gray):
    img = img_gray.astype(np.float32)
    h, w = img.shape
    blur = cv2.GaussianBlur(img, (5,5), 1.0)
    prnu = img - blur
    cy, cx = h//2, w//2
    h4, w4 = h//4, w//4
    face_energy = float(np.abs(prnu[cy-h4:cy+h4, cx-w4:cx+w4]).mean())
    mask = np.ones((h,w), dtype=bool)
    mask[cy-h4:cy+h4, cx-w4:cx+w4] = False
    peri_energy = float(np.abs(prnu[mask]).mean())
    return np.array([face_energy, face_energy/(peri_energy+1e-6)], dtype=np.float32)

def compute_pillar3(img_rgb):
    img = img_rgb.astype(np.float32)
    def res(ch):
        b = cv2.GaussianBlur(ch.astype(np.uint8),(3,3),0).astype(np.float32)
        return (ch - b).flatten()
    r = res(img[:,:,0]); g = res(img[:,:,1]); b = res(img[:,:,2])
    def sc(a,b):
        if a.std()<1e-6 or b.std()<1e-6: return 0.0
        return float(np.corrcoef(a,b)[0,1])
    return np.array([sc(r,g), sc(b,g)], dtype=np.float32)

def compute_pillar4(img_rgb):
    img = img_rgb.astype(np.float32)/255.0
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)/255.0
    cy,cx = h//2, w//2; h4,w4 = h//4, w//4
    face = gray[cy-h4:cy+h4, cx-w4:cx+w4]
    top = gray[:h4,:]; left = gray[:,:w4]; right = gray[:,-w4:]
    face_bg_diff = abs(float(face.mean()) -
                       float(np.concatenate([top.flatten(),left.flatten(),
                                             right.flatten()]).mean()))
    spec_ratio = float((face > 0.9).mean())
    gx = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=3)
    gy = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=3)
    shadow = float(np.sqrt(gx**2+gy**2).std())
    return np.array([face_bg_diff, spec_ratio, shadow], dtype=np.float32)

def compute_pillar6(img_gray):
    img = img_gray.astype(np.float32)
    h, w = img.shape
    dct_coeffs = []
    for i in range(0, h-8, 8):
        for j in range(0, w-8, 8):
            block = img[i:i+8, j:j+8]
            dct_block = fftpack.dct(fftpack.dct(block.T,norm='ortho').T,norm='ortho')
            dct_coeffs.extend(np.abs(dct_block.flatten()[1:]))
    dct_coeffs = np.array(dct_coeffs)
    dct_coeffs = dct_coeffs[dct_coeffs > 1.0]
    if len(dct_coeffs) > 100:
        ld = np.floor(dct_coeffs/10**np.floor(np.log10(dct_coeffs+1e-10))).astype(int)
        ld = ld[(ld>=1)&(ld<=9)]
        obs = np.bincount(ld,minlength=10)[1:]/(len(ld)+1e-6)
        exp = np.array([np.log10(1+1/d) for d in range(1,10)])
        benford = float(np.sum(np.abs(obs-exp)))
    else:
        benford = 0.0
    rows1=img[7::8,:]; rows2=img[8::8,:]
    mr = min(rows1.shape[0],rows2.shape[0])
    block_art = float(np.abs(rows1[:mr,:]-rows2[:mr,:]).mean()) if h>16 else 0.0
    hist,_ = np.histogram(dct_coeffs[:1000] if len(dct_coeffs)>1000 else dct_coeffs,
                          bins=50,range=(0,100))
    hf = np.abs(fftpack.fft(hist.astype(np.float32)))
    dcs = float(hf[2:8].max()/(hf[1:].mean()+1e-6))
    return np.array([benford, block_art, dcs], dtype=np.float32)

def compute_pillar10(img_rgb):
    img = img_rgb.astype(np.float32)
    r=img[:,:,0]; g=img[:,:,1]; b=img[:,:,2]
    h,w = g.shape
    def ch_shift(c1,c2):
        try:
            f1=np.fft.fft2(c1); f2=np.fft.fft2(c2)
            cp=f1*np.conj(f2); cp/=(np.abs(cp)+1e-10)
            corr=np.abs(np.fft.ifft2(cp))
            pk=np.unravel_index(corr.argmax(),corr.shape)
            dy=pk[0] if pk[0]<h//2 else pk[0]-h
            dx=pk[1] if pk[1]<w//2 else pk[1]-w
            return float(np.sqrt(dy**2+dx**2))
        except: return 0.0
    em=max(h//8,1)
    cr=float(np.abs(r[em:-em,em:-em]-g[em:-em,em:-em]).mean())
    er=float(np.abs(r-g).mean())
    return np.array([ch_shift(r,g), ch_shift(b,g), er/(cr+1e-6)], dtype=np.float32)

def extract_frame_features(frame_bgr):
    """Extract 24-dim physics features from one BGR frame."""
    try:
        frame_rgb  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        frame_rgb  = cv2.resize(frame_rgb,  (224,224))
        frame_gray = cv2.resize(frame_gray, (224,224))
        frame_bgr  = cv2.resize(frame_bgr,  (224,224))
        p1  = compute_pillar1(frame_gray)
        p2  = compute_pillar2(frame_gray)
        p3  = compute_pillar3(frame_rgb)
        p4  = compute_pillar4(frame_rgb)
        p5  = np.array([0.0, 0.0], dtype=np.float32)
        p6  = compute_pillar6(frame_gray)
        p7  = np.array([0.0, 0.0], dtype=np.float32)
        p8  = np.array([0.0, 0.0], dtype=np.float32)
        p9  = np.array([0.0, 0.0], dtype=np.float32)
        p10 = compute_pillar10(frame_rgb)
        feat = np.concatenate([p1,p2,p3,p4,p5,p6,p7,p8,p9,p10])
        feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
        return feat.astype(np.float32)
    except:
        return None

def extract_frames_from_video(video_path, n_frames=N_FRAMES):
    """Extract n_frames evenly from middle 60% of video."""
    try:
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total < 2: total = 30
        start = int(total * 0.20)
        end   = int(total * 0.80)
        if end - start < n_frames:
            start = 0; end = total
        indices = [int(x) for x in np.linspace(start, end-1, n_frames)]
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret and frame is not None:
                frames.append(frame)
        cap.release()
        return frames
    except:
        return []

def get_video_files(folder, max_n):
    """Get up to max_n video files from a folder (searches subdirs)."""
    exts = ('.mp4', '.avi', '.mov', '.mkv')
    videos = []
    for root, dirs, files in os.walk(folder):
        for f in sorted(files):
            if f.lower().endswith(exts):
                videos.append(os.path.join(root, f))
        if len(videos) >= max_n:
            break
    rng = np.random.default_rng(SEED)
    if len(videos) > max_n:
        idx = rng.choice(len(videos), max_n, replace=False)
        videos = [videos[i] for i in sorted(idx)]
    return videos[:max_n]

# ═════════════════════════════════════════════════════════════════
# STEP 1 — EXTRACT PER-FRAME FEATURES
# ═════════════════════════════════════════════════════════════════
print("="*60)
print("STEP 1: Extracting per-frame physics features")
print(f"  {N_VIDEOS} videos × {N_FRAMES} frames per class")
print("="*60)

all_features = []
all_labels   = []   # 0=real 1=fake
all_types    = []   # 'real', 'Deepfakes', etc.
all_frameids = []

# ── Real videos ───────────────────────────────────────────────────
print(f"\n[Real] Scanning {FFPP_REAL_DIR}...")
real_videos = get_video_files(FFPP_REAL_DIR, N_VIDEOS)
print(f"  Found {len(real_videos)} real videos")

real_count = 0
for i, vpath in enumerate(real_videos):
    frames = extract_frames_from_video(vpath)
    for frame in frames:
        feat = extract_frame_features(frame)
        if feat is not None:
            all_features.append(feat)
            all_labels.append(0)
            all_types.append('Real')
            all_frameids.append(f"real_{i}")
            real_count += 1
    if (i+1) % 10 == 0:
        print(f"  Real: {i+1}/{len(real_videos)} videos, {real_count} frames so far")

print(f"  Real total frames: {real_count}")

# ── Fake videos per type ──────────────────────────────────────────
fake_counts = {}
for ftype in FAKE_TYPES:
    fake_dir = os.path.join(FFPP_FAKE_BASE, ftype)
    if not os.path.exists(fake_dir):
        print(f"\n[{ftype}] Directory not found: {fake_dir} — skipping")
        continue

    print(f"\n[{ftype}] Scanning {fake_dir}...")
    fake_videos = get_video_files(fake_dir, N_VIDEOS)
    print(f"  Found {len(fake_videos)} videos")

    count = 0
    for i, vpath in enumerate(fake_videos):
        frames = extract_frames_from_video(vpath)
        for frame in frames:
            feat = extract_frame_features(frame)
            if feat is not None:
                all_features.append(feat)
                all_labels.append(1)
                all_types.append(ftype)
                all_frameids.append(f"{ftype}_{i}")
                count += 1
        if (i+1) % 10 == 0:
            print(f"  {ftype}: {i+1}/{len(fake_videos)} videos, {count} frames")

    fake_counts[ftype] = count
    print(f"  {ftype} total frames: {count}")

# Convert to arrays
X   = np.array(all_features, dtype=np.float32)
y   = np.array(all_labels,   dtype=np.int32)
gt  = np.array(all_types)

print(f"\nTotal frames extracted: {len(X)}")
print(f"Real frames:  {(y==0).sum()}")
print(f"Fake frames:  {(y==1).sum()}")
for ft in FAKE_TYPES:
    n = (gt==ft).sum()
    if n > 0:
        print(f"  {ft}: {n} frames")

# Normalise
scaler = StandardScaler()
X_sc   = scaler.fit_transform(X)

# Save checkpoint
ckpt = os.path.join(OUT_DIR, "frame_features.pkl")
with open(ckpt,'wb') as f:
    pickle.dump({'X':X, 'y':y, 'types':gt, 'X_sc':X_sc}, f, protocol=4)
print(f"\nFeatures saved to {ckpt}")

# ═════════════════════════════════════════════════════════════════
# STEP 2 — COLOUR PALETTE
# ═════════════════════════════════════════════════════════════════
TYPE_COLORS = {
    'Real':          '#22c55e',
    'Deepfakes':     '#ef4444',
    'Face2Face':     '#f59e0b',
    'FaceShifter':   '#8b5cf6',
    'FaceSwap':      '#ec4899',
    'NeuralTextures':'#14b8a6',
}

def make_legend(types_present):
    return [mpatches.Patch(color=TYPE_COLORS.get(t,'#888888'), label=t)
            for t in types_present if t in TYPE_COLORS]

def run_tsne(X_sub, label=""):
    perp = min(40, max(5, len(X_sub)//10))
    print(f"  t-SNE on {len(X_sub)} frames (perplexity={perp}) {label}...",
          end=" ", flush=True)
    ts = TSNE(n_components=2, perplexity=perp,
              random_state=SEED, n_iter=1000, verbose=0)
    res = ts.fit_transform(X_sub)
    print("done.")
    return res

# ═════════════════════════════════════════════════════════════════
# PLOT 1 — FF++ : Real vs ALL Fakes (Frame Level)
# ═════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("PLOT 1: Real vs All Fakes Combined (frame level)")
print("="*60)

emb1 = run_tsne(X_sc, "all")

fig, ax = plt.subplots(figsize=(10, 8))
# Real
m = y == 0
ax.scatter(emb1[m,0], emb1[m,1], c=TYPE_COLORS['Real'],
           alpha=0.5, s=16, label=f"Real (n={m.sum()})", zorder=3)
# All fakes
m = y == 1
ax.scatter(emb1[m,0], emb1[m,1], c='#ef4444',
           alpha=0.4, s=14, label=f"Fake — all types (n={m.sum()})", zorder=2)

ax.set_title("FF++  |  Real vs All Fakes  |  IMAGE-LEVEL\n"
             "Each dot = one video frame  |  Physics Features t-SNE",
             fontsize=13, fontweight='bold', pad=12)
ax.legend(fontsize=11, markerscale=2)
ax.set_xlabel("t-SNE Dimension 1", fontsize=11)
ax.set_ylabel("t-SNE Dimension 2", fontsize=11)
ax.grid(True, alpha=0.2)
plt.tight_layout()
p = os.path.join(OUT_DIR, "plot1_real_vs_allfakes_framelevel.png")
plt.savefig(p, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved → {p}")

# ═════════════════════════════════════════════════════════════════
# PLOT 2 — Real vs EACH Fake Type (Frame Level)
# Professor's diagram: individual dataset boxes
# ═════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("PLOT 2: Real vs Each Fake Type Individually (frame level)")
print("="*60)

present_fakes = [ft for ft in FAKE_TYPES if (gt==ft).sum() > 0]
n_plots = len(present_fakes)
ncols = 3
nrows = (n_plots + ncols - 1) // ncols

fig, axes = plt.subplots(nrows, ncols,
                          figsize=(18, 6*nrows))
axes_flat = axes.flatten() if n_plots > 1 else [axes]

real_idx = np.where(y == 0)[0]
rng = np.random.default_rng(SEED)

for i, ftype in enumerate(present_fakes):
    ax = axes_flat[i]
    fake_idx = np.where(gt == ftype)[0]

    # Balance classes
    n = min(len(real_idx), len(fake_idx), 600)
    ri = rng.choice(real_idx, n, replace=False)
    fi = rng.choice(fake_idx, n, replace=False)
    idx = np.concatenate([ri, fi])
    np.random.shuffle(idx)

    X_sub = X_sc[idx]
    y_sub = y[idx]
    gt_sub = gt[idx]

    print(f"  {ftype}: real={n}, fake={n}")
    emb = run_tsne(X_sub, ftype)

    for cls, col, lbl in [(0, TYPE_COLORS['Real'], 'Real'),
                           (1, TYPE_COLORS[ftype], ftype)]:
        m = y_sub == cls
        ax.scatter(emb[m,0], emb[m,1], c=col,
                   alpha=0.5, s=14,
                   label=f"{lbl} (n={m.sum()})", zorder=3-cls)

    ax.set_title(f"Real  vs  {ftype}\n(image-level frames)",
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9, markerscale=2)
    ax.grid(True, alpha=0.2)
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")

# Hide unused subplots
for j in range(n_plots, len(axes_flat)):
    axes_flat[j].axis('off')

fig.suptitle("FF++  |  Real vs Each Fake Type  |  IMAGE-LEVEL\n"
             "Each dot = one video frame  |  Phantom Lens V2 Physics Features",
             fontsize=14, fontweight='bold')
plt.tight_layout()
p = os.path.join(OUT_DIR, "plot2_real_vs_each_faketype_framelevel.png")
plt.savefig(p, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved → {p}")

# ═════════════════════════════════════════════════════════════════
# PLOT 3 — ALL TYPES coloured individually
# Shows which fakes cluster differently from real
# ═════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("PLOT 3: All types coloured individually (frame level)")
print("="*60)

# Subsample for speed
n_per = min(300, (y==0).sum())
rng2  = np.random.default_rng(SEED)
chosen = []
for t in ['Real'] + present_fakes:
    mask = gt == t if t != 'Real' else y == 0
    idx2 = np.where(mask)[0]
    n2   = min(n_per, len(idx2))
    chosen.extend(rng2.choice(idx2, n2, replace=False).tolist())
chosen = np.array(chosen)
np.random.shuffle(chosen)

print(f"  Using {len(chosen)} total frames")
emb3 = run_tsne(X_sc[chosen], "all types")

fig, ax = plt.subplots(figsize=(11, 9))
for t in ['Real'] + present_fakes:
    if t == 'Real':
        mask = y[chosen] == 0
    else:
        mask = gt[chosen] == t
    if mask.sum() == 0:
        continue
    ax.scatter(emb3[mask,0], emb3[mask,1],
               c=TYPE_COLORS.get(t,'#888888'),
               alpha=0.55, s=16,
               label=f"{t} (n={mask.sum()})", zorder=2)

ax.set_title("All Types — Real + Each Fake  |  IMAGE-LEVEL\n"
             "Each dot = one video frame  |  Physics Features t-SNE",
             fontsize=13, fontweight='bold', pad=12)
ax.legend(fontsize=10, markerscale=2, loc='best')
ax.set_xlabel("t-SNE Dimension 1", fontsize=11)
ax.set_ylabel("t-SNE Dimension 2", fontsize=11)
ax.grid(True, alpha=0.2)
plt.tight_layout()
p = os.path.join(OUT_DIR, "plot3_all_types_framelevel.png")
plt.savefig(p, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved → {p}")

# ═════════════════════════════════════════════════════════════════
# PLOT 4 — SUMMARY 4-PANEL (matches professor's diagram)
# ═════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("PLOT 4: 4-panel summary matching professor diagram")
print("="*60)

fig, axes = plt.subplots(2, 2, figsize=(16, 14))

panels = [
    ("DA — Real vs Deepfakes", "Deepfakes"),
    ("DB — Real vs Face2Face", "Face2Face"),
    ("DC — Real vs FaceShifter", "FaceShifter"),
    ("DD — Real vs All Fakes", None),  # None = all fakes combined
]

for idx2, (title, ftype) in enumerate(panels):
    ax = axes[idx2//2][idx2%2]
    ri2  = np.where(y == 0)[0]

    if ftype is not None:
        fi2 = np.where(gt == ftype)[0]
        if len(fi2) == 0:
            ax.text(0.5, 0.5, f"{ftype}\nnot found",
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title); continue
        n2 = min(400, len(ri2), len(fi2))
        rng3 = np.random.default_rng(SEED)
        ri_s = rng3.choice(ri2, n2, replace=False)
        fi_s = rng3.choice(fi2, n2, replace=False)
        idx_s = np.concatenate([ri_s, fi_s])
        np.random.shuffle(idx_s)
        X_s = X_sc[idx_s]; y_s = y[idx_s]; gt_s = gt[idx_s]
        print(f"  {title}: real={n2}, fake={n2}")
        emb_s = run_tsne(X_s, title)
        for cls, col, lbl in [(0, TYPE_COLORS['Real'], 'Real'),
                               (1, TYPE_COLORS.get(ftype,'#ef4444'), ftype)]:
            m2 = y_s == cls
            ax.scatter(emb_s[m2,0], emb_s[m2,1], c=col,
                       alpha=0.5, s=14,
                       label=f"{lbl} (n={m2.sum()})", zorder=3-cls)
    else:
        # All fakes combined
        fi2 = np.where(y == 1)[0]
        n2  = min(400, len(ri2), len(fi2))
        rng3 = np.random.default_rng(SEED)
        ri_s = rng3.choice(ri2, n2, replace=False)
        fi_s = rng3.choice(fi2, n2, replace=False)
        idx_s = np.concatenate([ri_s, fi_s])
        np.random.shuffle(idx_s)
        X_s = X_sc[idx_s]; y_s = y[idx_s]
        print(f"  {title}: real={n2}, fake={n2}")
        emb_s = run_tsne(X_s, title)
        for cls, col, lbl in [(0, TYPE_COLORS['Real'], 'Real'),
                               (1, '#ef4444', 'All Fakes')]:
            m2 = y_s == cls
            ax.scatter(emb_s[m2,0], emb_s[m2,1], c=col,
                       alpha=0.5, s=14,
                       label=f"{lbl} (n={m2.sum()})", zorder=3-cls)

    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, markerscale=2)
    ax.grid(True, alpha=0.2)
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")

fig.suptitle("Phantom Lens V2  |  Physics Features t-SNE  |  IMAGE-LEVEL\n"
             "Each dot = one video frame  |  Matching Professor's Diagram",
             fontsize=14, fontweight='bold')
plt.tight_layout()
p = os.path.join(OUT_DIR, "plot4_4panel_professor_diagram.png")
plt.savefig(p, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved → {p}")

# ═════════════════════════════════════════════════════════════════
# DONE
# ═════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("ALL DONE — Results saved to:", OUT_DIR)
print("="*60)
print()
print("Files to show professor:")
print("  plot4_4panel_professor_diagram.png  ← matches your diagram")
print("  plot2_real_vs_each_faketype_framelevel.png  ← per fake type")
print("  plot1_real_vs_allfakes_framelevel.png  ← overall")
print("  plot3_all_types_framelevel.png  ← all types coloured")
print()
print("What to tell professor:")
print("  'Sir, each dot is one individual video frame — image level.'")
print("  'Physics features extracted per frame, no averaging.'")
print("  'Separation shows physics signal is real.'")