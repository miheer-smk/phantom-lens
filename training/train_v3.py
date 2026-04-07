# Phantom Lens Research Framework
# Developer: Miheer Satish Kulkarni | IIIT Nagpur | 2026
# All Rights Reserved

"""
Phantom Lens / PRISM — Training Pipeline V3
============================================
Built on train_v2.py with these changes:
  1. Input: 30-dim V3 features (dead dim 20 removed -> 29 live dims)
  2. PillarAttentionClassifier replaces flat MLP
  3. Per-source codec normalisation for domain adaptation
  4. 5 seeds instead of 3 (target variance <= 0.0007)
  5. Per-dataset AUC breakdown in validation
  6. Pillar attention weights logged and visualised
  7. WeightedRandomSampler handles 96k real vs 80k fake imbalance
  8. Full metrics: AUC, F1, precision, recall, confusion matrix
  9. Loss curves, ROC curves, training report -- all saved

Target:
  FF++ validation AUC  : > 0.8961 (beat V1)
  Variance (5 seeds)   : <= 0.0007
  Celeb-DF v2 AUC      : 0.83-0.87 (cross-dataset test after training)

Author: Miheer Satish Kulkarni, IIIT Nagpur
"""

import os
import pickle
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.metrics import (roc_auc_score, roc_curve, confusion_matrix,
                              f1_score, precision_score, recall_score)
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# ── CONFIG ────────────────────────────────────────────────────────────────────

PKL_PATH = "data/precomputed_features_v3_with_celebdf.pkl"
CHECKPOINT_DIR = "checkpoints"
RESULTS_DIR    = "results/v3"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR,    exist_ok=True)

DEAD_DIMS  = [20]
TOTAL_DIMS = 30
LIVE_DIMS  = [i for i in range(TOTAL_DIMS) if i not in DEAD_DIMS]
INPUT_DIM  = len(LIVE_DIMS)   # 29

BATCH_SIZE    = 256
MAX_EPOCHS    = 60
LR            = 0.001
WEIGHT_DECAY  = 1e-3
DROPOUT_RATES = (0.40, 0.30, 0.20)
PATIENCE      = 8
SEEDS         = [42, 123, 777, 999, 2024]
VAL_SPLIT     = 0.20

AUG_PROB          = 0.5
NOISE_STD         = 0.02
BRIGHTNESS_RANGE  = 0.15
COMPRESSION_RANGE = (0.6, 1.4)

CODEC_DIMS = [12, 13, 14, 15, 16]

PILLAR_GROUPS = [
    [0, 1, 2],
    [3, 4],
    [5, 6],
    [7, 8, 9],
    [10, 11],
    [12, 13, 14],
    [15, 16],
    [17, 18],
    [19],
    [20, 21, 22],
    [23, 24, 25],
    [26, 27, 28],
]

PILLAR_NAMES = [
    'P1 Noise', 'P2 PRNU', 'P3 Bayer', 'P4 Shadow',
    'P5 Specular', 'P6 DCT', 'P7 Codec', 'P8 Blur',
    'P9 Flow', 'P10 Chromatic', 'P11 EyeSym', 'P12 Illum'
]

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE} | Input dims: {INPUT_DIM} | Pillars: {len(PILLAR_GROUPS)}")


# ── MODEL ─────────────────────────────────────────────────────────────────────

class PillarAttentionClassifier(nn.Module):
    def __init__(self, pillar_groups=PILLAR_GROUPS, dropout_rates=DROPOUT_RATES):
        super().__init__()
        self.pillar_groups = pillar_groups
        n = len(pillar_groups)
        self.pillar_encoders = nn.ModuleList([
            nn.Sequential(nn.Linear(len(g), 8), nn.ReLU(), nn.BatchNorm1d(8))
            for g in pillar_groups
        ])
        self.attention = nn.Sequential(nn.Linear(8*n, n), nn.Softmax(dim=1))
        self.classifier = nn.Sequential(
            nn.Linear(8*n, 64), nn.ReLU(), nn.BatchNorm1d(64), nn.Dropout(dropout_rates[0]),
            nn.Linear(64, 32),  nn.ReLU(), nn.BatchNorm1d(32), nn.Dropout(dropout_rates[1]),
            nn.Linear(32, 16),  nn.ReLU(), nn.BatchNorm1d(16), nn.Dropout(dropout_rates[2]),
            nn.Linear(16, 1),   nn.Sigmoid()
        )

    def forward(self, x):
        reps = [enc(x[:, g]) for g, enc in zip(self.pillar_groups, self.pillar_encoders)]
        stacked = torch.cat(reps, dim=1)
        weights = self.attention(stacked)
        attended = torch.cat([r * weights[:, i:i+1] for i, r in enumerate(reps)], dim=1)
        return self.classifier(attended).squeeze(1)

    def get_attention_weights(self, x):
        with torch.no_grad():
            reps = [enc(x[:, g]) for g, enc in zip(self.pillar_groups, self.pillar_encoders)]
            stacked = torch.cat(reps, dim=1)
            weights = self.attention(stacked)
        return weights.cpu().numpy()


# ── CODEC NORMALISATION ───────────────────────────────────────────────────────

def per_source_codec_normalize(X, sources, fit_stats=None):
    X = X.copy()
    stats = {}
    for src in set(sources):
        rows = np.where(np.array([s == src for s in sources]))[0]
        if fit_stats is None:
            mu = X[rows][:, CODEC_DIMS].mean(axis=0)
            sd = X[rows][:, CODEC_DIMS].std(axis=0) + 1e-8
            stats[src] = (mu, sd)
        else:
            if src in fit_stats:
                mu, sd = fit_stats[src]
            else:
                mu = np.mean([v[0] for v in fit_stats.values()], axis=0)
                sd = np.mean([v[1] for v in fit_stats.values()], axis=0)
        X[np.ix_(rows, CODEC_DIMS)] = (X[rows][:, CODEC_DIMS] - mu) / sd
    return X, stats


# ── AUGMENTATION ──────────────────────────────────────────────────────────────

def augment_features(features, labels):
    aug = features.copy()
    for i in range(len(aug)):
        if np.random.random() < AUG_PROB:
            aug[i] += np.random.normal(0, NOISE_STD, aug.shape[1]).astype(np.float32)
        if np.random.random() < AUG_PROB:
            shift = np.random.uniform(-BRIGHTNESS_RANGE, BRIGHTNESS_RANGE)
            aug[i, 0:3]  *= (1 + shift)
            aug[i, 3:5]  *= (1 + shift)
        if np.random.random() < AUG_PROB:
            cf = np.random.uniform(*COMPRESSION_RANGE)
            aug[i, 12:17] *= cf
        if np.random.random() < 0.2:
            aug[i, np.random.randint(0, aug.shape[1])] = 0.0
    return aug


# ── DATA LOADING ──────────────────────────────────────────────────────────────

def load_data():
    print(f"\nLoading: {PKL_PATH}")
    with open(PKL_PATH, 'rb') as f:
        data = pickle.load(f)
    features  = np.array(data['features'], dtype=np.float32)
    labels    = np.array(data['labels'],   dtype=np.float32)
    video_ids = data.get('video_ids', [str(i) for i in range(len(labels))])
    sources   = data.get('dataset_sources', ['unknown'] * len(labels))
    assert features.shape[1] == TOTAL_DIMS, f"Expected {TOTAL_DIMS} dims"
    features = features[:, LIVE_DIMS]
    real = int((labels==0).sum()); fake = int((labels==1).sum())
    print(f"  Total: {len(labels)} | Real: {real} | Fake: {fake} | Dims: {features.shape[1]}")
    src_arr = np.array(sources)
    for src in sorted(set(sources)):
        sr = int(((src_arr==src)&(labels==0)).sum())
        sf = int(((src_arr==src)&(labels==1)).sum())
        print(f"  {src:<20} real={sr:6d}  fake={sf:6d}")
    return features, labels, video_ids, sources


# ── VIDEO LEVEL SPLIT ─────────────────────────────────────────────────────────

def video_level_split(features, labels, video_ids, sources, val_split=0.20, seed=42):
    np.random.seed(seed)
    vid_to_idx = defaultdict(list)
    for i, vid in enumerate(video_ids):
        vid_to_idx[vid].append(i)
    unique_vids = list(vid_to_idx.keys())
    vid_labels  = {v: labels[vid_to_idx[v][0]] for v in unique_vids}
    real_vids   = [v for v in unique_vids if vid_labels[v] == 0]
    fake_vids   = [v for v in unique_vids if vid_labels[v] == 1]
    np.random.shuffle(real_vids); np.random.shuffle(fake_vids)
    n_rv = max(1, int(len(real_vids)*val_split))
    n_fv = max(1, int(len(fake_vids)*val_split))
    val_vids = set(real_vids[:n_rv] + fake_vids[:n_fv])
    trn_vids = set(real_vids[n_rv:] + fake_vids[n_fv:])
    trn_idx = [i for v in trn_vids for i in vid_to_idx[v]]
    val_idx = [i for v in val_vids  for i in vid_to_idx[v]]
    src_arr = np.array(sources)
    print(f"  Video split: {len(trn_vids)} train | {len(val_vids)} val")
    print(f"  Sample split: {len(trn_idx)} train | {len(val_idx)} val")
    return (features[trn_idx], labels[trn_idx], src_arr[trn_idx],
            features[val_idx], labels[val_idx], src_arr[val_idx])


# ── EVALUATION ────────────────────────────────────────────────────────────────

def full_evaluation(model, X_val_sc, y_val, src_val, seed):
    model.eval()
    with torch.no_grad():
        probs = model(torch.FloatTensor(X_val_sc).to(DEVICE)).cpu().numpy()
    preds = (probs > 0.5).astype(int)
    auc   = roc_auc_score(y_val, probs)
    acc   = float(np.mean(preds == y_val))
    f1    = f1_score(y_val, preds, zero_division=0)
    prec  = precision_score(y_val, preds, zero_division=0)
    rec   = recall_score(y_val, preds, zero_division=0)
    cm    = confusion_matrix(y_val, preds)
    print(f"\n{'='*50}")
    print(f"EVALUATION RESULTS (seed={seed})")
    print(f"{'='*50}")
    print(f"AUC       : {auc:.4f}")
    print(f"Accuracy  : {acc:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    print(f"\nPer-dataset AUC breakdown:")
    for src in sorted(set(src_val)):
        idx = np.where(src_val == src)[0]
        if len(idx) < 10: continue
        sl = y_val[idx]; sp = probs[idx]
        if len(np.unique(sl)) < 2: continue
        print(f"  {src:<25} AUC={roc_auc_score(sl, sp):.4f}  (n={len(idx)})")
    fpr, tpr, _ = roc_curve(y_val, probs)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, 'b-', lw=2, label=f'AUC = {auc:.4f}')
    plt.plot([0,1],[0,1],'k--')
    plt.axhline(y=0.8961, color='gray', ls=':', lw=1.2, label='V1 baseline (0.8961)')
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve — Phantom Lens V3 (seed={seed})')
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(RESULTS_DIR, f'roc_curve_seed{seed}.png'), dpi=150, bbox_inches='tight')
    plt.close()
    return {'auc': auc, 'acc': acc, 'f1': f1, 'precision': prec, 'recall': rec, 'probs': probs}


def save_loss_curves(train_losses, val_losses, val_aucs, seed):
    epochs = range(1, len(train_losses)+1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss')
    ax1.plot(epochs, val_losses,   'r-', label='Val Loss')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
    ax1.set_title(f'Loss Curves (seed={seed})'); ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2.plot(epochs, val_aucs, 'g-', label='Val AUC')
    ax2.axhline(y=0.8961, color='gray', ls='--', lw=1.2, label='V1 baseline')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('AUC')
    ax2.set_title(f'Validation AUC (seed={seed})'); ax2.legend(); ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f'training_curves_seed{seed}.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Loss curves saved: results/v3/training_curves_seed{seed}.png")


# ── ABLATION STUDY ────────────────────────────────────────────────────────────

def run_ablation(features, labels, video_ids, sources):
    print("\n" + "="*60)
    print("ABLATION STUDY — V3 (29-dim, 12 pillars)")
    print("="*60)
    experiments = [
        ("P1 Noise only",               [0,1,2]),
        ("P2 PRNU only",                [3,4]),
        ("P4 Shadow only",              [7,8,9]),
        ("P6 DCT only",                 [12,13,14]),
        ("P8 Blur only",                [17,18]),
        ("P11 EyeSym only (NEW)",       [23,24,25]),
        ("P12 Illumination only (NEW)", [26,27,28]),
        ("Top-3 pillars (1+2+4)",       [0,1,2,3,4,7,8,9]),
        ("Original 10 pillars",         list(range(23))),
        ("All 12 pillars V3",           list(range(29))),
    ]
    ablation_results = []
    for exp_name, fmask in experiments:
        print(f"\n  {exp_name}")
        np.random.seed(42); torch.manual_seed(42)
        X_tr, y_tr, src_tr, X_val, y_val, src_val = video_level_split(
            features, labels, video_ids, sources, val_split=VAL_SPLIT, seed=42)
        scaler = StandardScaler()
        X_tr_sc  = scaler.fit_transform(X_tr[:, fmask]).astype(np.float32)
        X_val_sc = scaler.transform(X_val[:, fmask]).astype(np.float32)
        abl_groups = [[i] for i in range(len(fmask))]
        model = PillarAttentionClassifier(pillar_groups=abl_groups).to(DEVICE)
        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        criterion = nn.BCELoss()
        n_real = int((y_tr==0).sum()); n_fake = int((y_tr==1).sum())
        w_r = len(y_tr)/(2*n_real); w_f = len(y_tr)/(2*n_fake)
        sw  = np.where(y_tr==0, w_r, w_f)
        sampler = WeightedRandomSampler(torch.FloatTensor(sw), len(sw), replacement=True)
        tr_dl  = DataLoader(TensorDataset(torch.FloatTensor(augment_features(X_tr_sc, y_tr)),
                                           torch.FloatTensor(y_tr)), batch_size=BATCH_SIZE, sampler=sampler)
        val_dl = DataLoader(TensorDataset(torch.FloatTensor(X_val_sc), torch.FloatTensor(y_val)),
                            batch_size=BATCH_SIZE, shuffle=False)
        best_auc = 0.0; pat = 0
        for ep in range(60):
            model.train()
            for Xb, yb in tr_dl:
                optimizer.zero_grad()
                criterion(model(Xb.to(DEVICE)), yb.to(DEVICE)).backward()
                optimizer.step()
            model.eval()
            with torch.no_grad():
                probs = model(torch.FloatTensor(X_val_sc).to(DEVICE)).cpu().numpy()
            auc = roc_auc_score(y_val, probs)
            if auc > best_auc: best_auc = auc; pat = 0
            else:
                pat += 1
                if pat >= 15: break
        ablation_results.append((exp_name, best_auc))
        print(f"  -> AUC: {best_auc:.4f}")
    print("\n" + "="*60)
    print("ABLATION SUMMARY")
    print("="*60)
    for name, auc in ablation_results:
        diff = auc - 0.8961
        bar  = "X" * int(auc * 50)
        print(f"  {name:<40} AUC={auc:.4f}  {diff:+.4f}")
    return ablation_results


# ── TRAIN ONE SEED ────────────────────────────────────────────────────────────

def train_one_seed(features, labels, video_ids, sources, seed):
    print(f"\n{'='*60}")
    print(f"TRAINING SEED {seed}")
    print(f"{'='*60}")
    torch.manual_seed(seed); np.random.seed(seed)

    X_tr, y_tr, src_tr, X_val, y_val, src_val = video_level_split(
        features, labels, video_ids, sources, val_split=VAL_SPLIT, seed=seed)


    # Per-video L2 normalisation — removes dataset-level bias
    X_tr_pv  = X_tr  / (np.linalg.norm(X_tr,  axis=1, keepdims=True) + 1e-8)
    X_val_pv = X_val / (np.linalg.norm(X_val, axis=1, keepdims=True) + 1e-8)
    scaler   = StandardScaler()
    X_tr_sc  = scaler.fit_transform(X_tr_pv).astype(np.float32)
    X_val_sc = scaler.transform(X_val_pv).astype(np.float32)
    X_tr_aug = augment_features(X_tr_sc, y_tr)
    # Remove codec normalisation — it was adding dataset bias back

    n_real = int((y_tr==0).sum()); n_fake = int((y_tr==1).sum())
    w_real = len(y_tr)/(2*n_real); w_fake = len(y_tr)/(2*n_fake)
    sw     = np.where(y_tr==0, w_real, w_fake)
    sampler = WeightedRandomSampler(torch.FloatTensor(sw), len(sw), replacement=True)
    print(f"  Imbalance: real={n_real} fake={n_fake} w_real={w_real:.3f} w_fake={w_fake:.3f}")

    tr_dl  = DataLoader(TensorDataset(torch.FloatTensor(X_tr_aug), torch.FloatTensor(y_tr)),
                        batch_size=BATCH_SIZE, sampler=sampler)
    val_dl = DataLoader(TensorDataset(torch.FloatTensor(X_val_sc), torch.FloatTensor(y_val)),
                        batch_size=BATCH_SIZE, shuffle=False)

    model     = PillarAttentionClassifier().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS, eta_min=1e-5)
    criterion = nn.BCELoss()

    best_auc = 0.0; best_state = None; pat_ctr = 0
    train_losses = []; val_losses = []; val_aucs = []

    for epoch in range(1, MAX_EPOCHS+1):
        model.train()
        epoch_loss = 0.0
        for Xb, yb in tr_dl:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward(); optimizer.step()
            epoch_loss += loss.item()
        avg_tr = epoch_loss / len(tr_dl)
        train_losses.append(avg_tr)
        scheduler.step()

        model.eval()
        vp, vl, vl_sum = [], [], 0.0
        with torch.no_grad():
            for Xb, yb in val_dl:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                probs = model(Xb)
                vl_sum += criterion(probs, yb).item()
                vp.extend(probs.cpu().numpy())
                vl.extend(yb.cpu().numpy())
        avg_val = vl_sum / len(val_dl)
        val_losses.append(avg_val)
        vp_arr = np.array(vp); vl_arr = np.array(vl)
        auc = roc_auc_score(vl_arr, vp_arr)
        acc = float(np.mean((vp_arr > 0.5) == vl_arr))
        val_aucs.append(auc)

        print(f"Epoch {epoch:3d}/{MAX_EPOCHS} | TrLoss={avg_tr:.4f} | ValLoss={avg_val:.4f} | AUC={auc:.4f} | Acc={acc:.4f}")

        if auc > best_auc:
            best_auc   = auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            pat_ctr    = 0
            print(f"  -> Best model saved (AUC={best_auc:.4f})")
        else:
            pat_ctr += 1
            if pat_ctr >= PATIENCE:
                print(f"  Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    model.eval()
    sample_x = torch.FloatTensor(X_val_sc[:min(512, len(X_val_sc))]).to(DEVICE)
    attn_w   = model.get_attention_weights(sample_x).mean(axis=0)
    print(f"\n  Pillar attention weights:")
    for name, w in zip(PILLAR_NAMES, attn_w):
        print(f"    {name:<15} {w:.4f}  {'X'*int(w*200)}")

    save_loss_curves(train_losses, val_losses, val_aucs, seed)
    metrics = full_evaluation(model, X_val_sc, vl_arr, src_val, seed)

    return (best_auc, best_state, scaler, {},
            train_losses, val_losses, val_aucs, attn_w,
            X_val_sc, vl_arr, src_val, metrics)


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    print("="*60)
    print("Phantom Lens V3 — Training Pipeline")
    print(f"Device: {DEVICE} | Input: {INPUT_DIM} dims | Pillars: {len(PILLAR_GROUPS)}")
    print("="*60)

    features, labels, video_ids, sources = load_data()

    all_aucs = []; all_metrics = []; all_attn = []; seed_val_aucs = []
    best_auc_ever = 0.0; best_state_ever = None
    best_scaler_ever = None; best_codec_ever = None; best_seed_ever = None

    for seed in SEEDS:
        (auc, state, scaler, _,
         tr_losses, val_losses, val_aucs,
         attn_w, X_val_sc, y_val, src_val,
         metrics) = train_one_seed(features, labels, video_ids, sources, seed)

        all_aucs.append(auc); all_metrics.append(metrics)
        all_attn.append(attn_w); seed_val_aucs.append(val_aucs)

        ckpt_path = os.path.join(CHECKPOINT_DIR, f"v3_seed{seed}_AUC{int(auc*10000)}.pt")
        torch.save({
            'model_state': state, 'scaler_mean': scaler.mean_,
            'scaler_std': scaler.scale_,
            'live_dims': LIVE_DIMS, 'dead_dims': DEAD_DIMS,
            'input_dim': INPUT_DIM, 'pillar_groups': PILLAR_GROUPS,
            'auc': auc, 'seed': seed, 'version': 'v3',
        }, ckpt_path)
        print(f"  Checkpoint: {ckpt_path}")

        if auc > best_auc_ever:
            best_auc_ever = auc; best_state_ever = state
            best_scaler_ever = scaler; best_codec_ever = {}
            best_seed_ever = seed

    best_path = os.path.join(CHECKPOINT_DIR, f"best_model_v3_AUC{int(best_auc_ever*10000)}.pt")
    torch.save({
        'model_state': best_state_ever, 'scaler_mean': best_scaler_ever.mean_,
        'scaler_std': best_scaler_ever.scale_, 'codec_stats': best_codec_ever,
        'live_dims': LIVE_DIMS, 'dead_dims': DEAD_DIMS,
        'input_dim': INPUT_DIM, 'pillar_groups': PILLAR_GROUPS,
        'auc': best_auc_ever, 'version': 'v3',
    }, best_path)
    shutil.copy(best_path, os.path.join(CHECKPOINT_DIR, "best_model_v3.pt"))

    mean_auc  = float(np.mean(all_aucs))
    std_auc   = float(np.std(all_aucs))
    mean_attn = np.mean(all_attn, axis=0)

    print(f"\n{'='*60}")
    print("MULTI-SEED SUMMARY")
    print(f"{'='*60}")
    for seed, auc in zip(SEEDS, all_aucs):
        diff = auc - 0.8961
        mk   = "BEAT V1" if diff > 0 else "below V1"
        print(f"  Seed {seed:>4} : AUC={auc:.4f}  {diff:+.4f}  [{mk}]")
    print(f"\n  Mean AUC  : {mean_auc:.4f} +/- {std_auc:.4f}")
    print(f"  Best AUC  : {best_auc_ever:.4f} (seed {best_seed_ever})")
    print(f"  V1 base   : 0.8961 +/- 0.0007")
    print(f"  Change    : {mean_auc-0.8961:+.4f}")
    if std_auc <= 0.0007:
        print(f"  Variance  : PASSED (<= 0.0007)")
    else:
        print(f"  Variance  : FAILED ({std_auc:.4f} > 0.0007)")

    ablation_results = run_ablation(features, labels, video_ids, sources)

    report_path = os.path.join(RESULTS_DIR, 'v3_training_report.txt')
    with open(report_path, 'w') as f:
        f.write("PHANTOM LENS V3 — TRAINING REPORT\n")
        f.write("="*60 + "\n\n")
        f.write(f"Architecture : PillarAttentionClassifier\n")
        f.write(f"Input dims   : {INPUT_DIM}\n")
        f.write(f"Pillars      : {len(PILLAR_GROUPS)}\n\n")
        f.write(f"Mean AUC : {mean_auc:.4f} +/- {std_auc:.4f}\n")
        f.write(f"Best AUC : {best_auc_ever:.4f}\n")
        f.write(f"V1 base  : 0.8961 +/- 0.0007\n")
        f.write(f"Change   : {mean_auc-0.8961:+.4f}\n\n")
        f.write("Per-seed:\n")
        for seed, auc, m in zip(SEEDS, all_aucs, all_metrics):
            f.write(f"  Seed {seed}: AUC={auc:.4f} F1={m['f1']:.4f} Acc={m['acc']:.4f}\n")
        f.write("\nPillar attention (mean):\n")
        for name, w in zip(PILLAR_NAMES, mean_attn):
            f.write(f"  {name}: {w:.4f}\n")
        f.write("\nAblation:\n")
        for name, auc in ablation_results:
            f.write(f"  {name:<40} AUC={auc:.4f}\n")

    print(f"\n  Report: {report_path}")
    print(f"  Best checkpoint: {best_path}")
    print(f"\nTraining complete!")
    print(f"Final AUC: {mean_auc:.4f} +/- {std_auc:.4f}")


if __name__ == "__main__":
    main()