# Phantom Lens Research Framework
# Developer: Miheer Satish Kulkarni | IIIT Nagpur | 2026
# All Rights Reserved

"""
Phantom Lens — Cross-Dataset Evaluator V3
Tests V3 model on Celeb-DF v2 (never seen during training)
Also compares against V1 result (0.4923) for honest comparison

Usage:
  python eval_crossdataset_v3.py

Author: Miheer Satish Kulkarni, IIIT Nagpur
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.metrics import (roc_auc_score, roc_curve,
                              accuracy_score, f1_score,
                              confusion_matrix, precision_score,
                              recall_score)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from precompute_features_v3 import extract_features_from_video
    print("V3 feature extractor loaded")
except ImportError:
    print("ERROR: Cannot import precompute_features_v3.py")
    sys.exit(1)

# ── CONFIG ────────────────────────────────────────────────────────────────────
CHECKPOINT   = "checkpoints/best_model_v3_AUC9177.pt"
REAL_FOLDER  = "data/celebdf/real"
FAKE_FOLDER  = "data/celebdf/fake"
OUTPUT_DIR   = "results/cross_dataset_v3"
N_FRAMES     = 16     # V3 uses 16 frames
MAX_VIDEOS   = 500
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# V1 reference result for comparison
V1_CROSSDATASET_AUC = 0.4923
V1_INDIST_AUC       = 0.8961
V3_INDIST_AUC       = 0.9117

# Dead dims from training
DEAD_DIMS = [20]
TOTAL_DIMS = 30
LIVE_DIMS  = [i for i in range(TOTAL_DIMS) if i not in DEAD_DIMS]

# Pillar groups
PILLAR_GROUPS = [
    [0,1,2],[3,4],[5,6],[7,8,9],[10,11],
    [12,13,14],[15,16],[17,18],[19],
    [20,21,22],[23,24,25],[26,27,28],
]


# ── MODEL ─────────────────────────────────────────────────────────────────────
class PillarAttentionClassifier(nn.Module):
    def __init__(self, pillar_groups=PILLAR_GROUPS,
                 dropout_rates=(0.25, 0.15, 0.10)):
        super().__init__()
        self.pillar_groups = pillar_groups
        n = len(pillar_groups)
        self.pillar_encoders = nn.ModuleList([
            nn.Sequential(nn.Linear(len(g), 8), nn.ReLU(), nn.BatchNorm1d(8))
            for g in pillar_groups
        ])
        self.attention = nn.Sequential(
            nn.Linear(8*n, n), nn.Softmax(dim=1))
        self.classifier = nn.Sequential(
            nn.Linear(8*n, 64), nn.ReLU(), nn.BatchNorm1d(64), nn.Dropout(dropout_rates[0]),
            nn.Linear(64, 32),  nn.ReLU(), nn.BatchNorm1d(32), nn.Dropout(dropout_rates[1]),
            nn.Linear(32, 16),  nn.ReLU(), nn.BatchNorm1d(16), nn.Dropout(dropout_rates[2]),
            nn.Linear(16, 1),   nn.Sigmoid()
        )

    def forward(self, x):
        reps    = [enc(x[:, g]) for g, enc in
                   zip(self.pillar_groups, self.pillar_encoders)]
        stacked = torch.cat(reps, dim=1)
        weights = self.attention(stacked)
        attended = torch.cat([r * weights[:, i:i+1]
                               for i, r in enumerate(reps)], dim=1)
        return self.classifier(attended).squeeze(1)


# ── LOAD MODEL ────────────────────────────────────────────────────────────────
def load_model():
    if not os.path.exists(CHECKPOINT):
        print(f"ERROR: Checkpoint not found: {CHECKPOINT}")
        sys.exit(1)
    ckpt  = torch.load(CHECKPOINT, map_location=DEVICE)
    model = PillarAttentionClassifier(
        pillar_groups=ckpt.get('pillar_groups', PILLAR_GROUPS)
    ).to(DEVICE)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    scaler_mean  = ckpt['scaler_mean']
    scaler_std   = ckpt['scaler_std']
    codec_stats  = ckpt.get('codec_stats', {})
    live_dims    = ckpt.get('live_dims', LIVE_DIMS)
    print(f"Model loaded | Training AUC: {ckpt.get('auc',0):.4f}")
    print(f"Device: {DEVICE} | Live dims: {len(live_dims)}")
    return model, scaler_mean, scaler_std, codec_stats, live_dims


# ── PREDICT ONE VIDEO ─────────────────────────────────────────────────────────
def predict_video(video_path, model, scaler_mean, scaler_std,
                  codec_stats, live_dims):
    try:
        feats = extract_features_from_video(
            video_path, n_frames=N_FRAMES)
        if len(feats) == 0:
            return None

        feat_arr  = np.array(feats, dtype=np.float32)
        feat_mean = feat_arr.mean(axis=0, keepdims=True)  # (1, 30)

        # Remove dead dims
        feat_live = feat_mean[:, live_dims]               # (1, 29)

        # Apply codec normalisation using 'celebvhq' stats as proxy
        # (closest domain to Celeb-DF v2 in terms of real video stats)
        codec_dims = [12, 13, 14, 15, 16]
        if 'celebvhq' in codec_stats:
            mu, sd = codec_stats['celebvhq']
            feat_live[0, codec_dims] = (
                feat_live[0, codec_dims] - mu) / (sd + 1e-8)
        elif len(codec_stats) > 0:
            mu = np.mean([v[0] for v in codec_stats.values()], axis=0)
            sd = np.mean([v[1] for v in codec_stats.values()], axis=0)
            feat_live[0, codec_dims] = (
                feat_live[0, codec_dims] - mu) / (sd + 1e-8)

        # Scale
        feat_sc = (feat_live - scaler_mean) / (scaler_std + 1e-8)
        tensor  = torch.FloatTensor(feat_sc).to(DEVICE)

        with torch.no_grad():
            prob = float(model(tensor).cpu().numpy()[0])
        return prob

    except Exception:
        return None


# ── EVALUATE FOLDER ───────────────────────────────────────────────────────────
def evaluate_folder(folder, label, model, scaler_mean, scaler_std,
                    codec_stats, live_dims, max_v, desc):
    videos = sorted([
        f for f in os.listdir(folder)
        if f.lower().endswith(('.mp4', '.avi', '.mov'))
    ])[:max_v]

    if len(videos) == 0:
        print(f"ERROR: No videos in {folder}")
        return []

    results = []
    failed  = 0
    for vf in tqdm(videos, desc=desc):
        prob = predict_video(
            os.path.join(folder, vf),
            model, scaler_mean, scaler_std,
            codec_stats, live_dims)
        if prob is None:
            failed += 1
            continue
        results.append((prob, label))

    lbl = 'REAL' if label == 0 else 'FAKE'
    print(f"  {lbl}: {len(results)} processed | {failed} failed")
    return results


# ── COMPUTE METRICS ───────────────────────────────────────────────────────────
def compute_and_print(results, model_name):
    probs  = np.array([r[0] for r in results])
    labels = np.array([r[1] for r in results])

    auc   = roc_auc_score(labels, probs)
    preds = (probs >= 0.5).astype(int)
    acc   = accuracy_score(labels, preds)
    f1    = f1_score(labels, preds, zero_division=0)
    prec  = precision_score(labels, preds, zero_division=0)
    rec   = recall_score(labels, preds, zero_division=0)
    cm    = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()
    tpr = tp / (tp + fn + 1e-8)
    tnr = tn / (tn + fp + 1e-8)

    if   auc >= 0.90: verdict = "EXCEPTIONAL — IEEE TIFS ready"
    elif auc >= 0.85: verdict = "VERY GOOD — publishable"
    elif auc >= 0.80: verdict = "GOOD — strong result"
    elif auc >= 0.75: verdict = "ACCEPTABLE"
    elif auc >= 0.65: verdict = "IMPROVING"
    else:             verdict = "NEEDS MORE WORK"

    print(f"\n{'='*60}")
    print(f"  {model_name} — CELEB-DF V2 CROSS-DATASET RESULTS")
    print(f"{'='*60}")
    print(f"  Samples  : {int((labels==0).sum())} real | "
          f"{int((labels==1).sum())} fake")
    print(f"  AUC      : {auc:.4f}   {verdict}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1       : {f1:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  TPR fake : {tpr:.4f}  (fake detection rate)")
    print(f"  TNR real : {tnr:.4f}  (real detection rate)")
    print(f"  CM: TN={tn} FP={fp} FN={fn} TP={tp}")
    print(f"\n  V1 in-distribution AUC  : {V1_INDIST_AUC:.4f}")
    print(f"  V3 in-distribution AUC  : {V3_INDIST_AUC:.4f}")
    print(f"  V1 cross-dataset AUC    : {V1_CROSSDATASET_AUC:.4f}")
    print(f"  V3 cross-dataset AUC    : {auc:.4f}")
    improvement = auc - V1_CROSSDATASET_AUC
    print(f"  Improvement over V1     : {improvement:+.4f}")
    print(f"{'='*60}")

    return auc, probs, labels


# ── SAVE COMPARISON PLOT ──────────────────────────────────────────────────────
def save_comparison_plot(probs, labels, v3_auc):
    fpr, tpr, _ = roc_curve(labels, probs)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor('#F8F9FA')

    # ROC curve
    ax = axes[0]
    ax.set_facecolor('white')
    ax.plot([0,1],[0,1],'k--',alpha=0.4,lw=1.2,label='Random (0.50)')
    ax.axhline(V1_CROSSDATASET_AUC, color='#A32D2D', ls=':',
               lw=1.5, alpha=0.8,
               label=f'V1 cross-dataset ({V1_CROSSDATASET_AUC:.4f})')
    ax.plot(fpr, tpr, color='#185FA5', lw=2.5,
            label=f'V3 cross-dataset ({v3_auc:.4f})')
    ax.fill_between(fpr, tpr, alpha=0.08, color='#185FA5')
    ax.set_xlim([0,1]); ax.set_ylim([0,1.05])
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title('ROC Curve — Celeb-DF v2\nV1 vs V3 Cross-Dataset',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.spines[['top','right']].set_visible(False)

    # AUC comparison bar
    ax2 = axes[1]
    ax2.set_facecolor('white')
    scenarios = ['V1\nFF++ val\n(in-dist)',
                 'V3\nFF++ val\n(in-dist)',
                 'V1\nCeleb-DF\n(cross)',
                 'V3\nCeleb-DF\n(cross)']
    aucs   = [V1_INDIST_AUC, V3_INDIST_AUC,
              V1_CROSSDATASET_AUC, v3_auc]
    colors = ['#185FA5','#3B6D11','#A32D2D',
              '#3B6D11' if v3_auc > 0.80 else '#854F0B']
    bars = ax2.bar(scenarios, aucs, color=colors,
                   alpha=0.85, width=0.5, edgecolor='white')
    ax2.axhline(0.85, color='navy', ls='--', lw=1.5,
                label='IEEE TIFS target (0.85)')
    ax2.axhline(0.50, color='red', ls=':', lw=1.2, alpha=0.5,
                label='Random (0.50)')
    ax2.set_ylim([0.3, 1.0])
    ax2.set_ylabel('AUC', fontsize=11)
    ax2.set_title('AUC Comparison\nV1 vs V3',
                  fontsize=11, fontweight='bold')
    for bar, auc in zip(bars, aucs):
        ax2.text(bar.get_x()+bar.get_width()/2,
                 bar.get_height()+0.01,
                 f'{auc:.4f}', ha='center', va='bottom',
                 fontsize=10, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.spines[['top','right']].set_visible(False)

    plt.suptitle('Phantom Lens / PRISM — Cross-Dataset Evaluation\n'
                 'V3 PillarAttentionClassifier vs V1 MLP',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'v3_vs_v1_crossdataset.png')
    plt.savefig(out, dpi=150, bbox_inches='tight',
                facecolor='#F8F9FA')
    plt.close()
    print(f"\n  Plot saved: {out}")


# ── SAVE RESULTS ──────────────────────────────────────────────────────────────
def save_results(v3_auc, probs, labels):
    preds = (probs >= 0.5).astype(int)
    acc   = accuracy_score(labels, preds)
    f1    = f1_score(labels, preds, zero_division=0)
    improvement = v3_auc - V1_CROSSDATASET_AUC

    txt = os.path.join(OUTPUT_DIR, 'v3_crossdataset_results.txt')
    with open(txt, 'w') as f:
        f.write("PHANTOM LENS V3 — CROSS-DATASET RESULTS\n")
        f.write("="*55 + "\n")
        f.write("Model   : best_model_v3_AUC9177.pt\n")
        f.write("Dataset : Celeb-DF v2 (ZERO-SHOT)\n")
        f.write("="*55 + "\n\n")
        f.write(f"V1 in-dist AUC       : {V1_INDIST_AUC:.4f}\n")
        f.write(f"V3 in-dist AUC       : {V3_INDIST_AUC:.4f}\n")
        f.write(f"V1 cross-dataset AUC : {V1_CROSSDATASET_AUC:.4f}\n")
        f.write(f"V3 cross-dataset AUC : {v3_auc:.4f}\n")
        f.write(f"Improvement          : {improvement:+.4f}\n\n")
        f.write(f"Accuracy : {acc:.4f}\n")
        f.write(f"F1       : {f1:.4f}\n")
    print(f"  Results saved: {txt}")


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "="*60)
    print("  Phantom Lens V3 — Cross-Dataset Evaluation")
    print("  Celeb-DF v2 | Zero-shot | No retraining")
    print("="*60 + "\n")

    model, scaler_mean, scaler_std, codec_stats, live_dims = load_model()
    print()

    print("Processing REAL videos...")
    real_r = evaluate_folder(
        REAL_FOLDER, 0, model,
        scaler_mean, scaler_std,
        codec_stats, live_dims,
        MAX_VIDEOS, "Real videos")

    print("\nProcessing FAKE videos...")
    fake_r = evaluate_folder(
        FAKE_FOLDER, 1, model,
        scaler_mean, scaler_std,
        codec_stats, live_dims,
        MAX_VIDEOS, "Fake videos")

    all_r = real_r + fake_r
    if len(all_r) < 20:
        print("ERROR: Too few results.")
        sys.exit(1)

    v3_auc, probs, labels = compute_and_print(
        all_r, "V3 PillarAttentionClassifier")
    save_comparison_plot(probs, labels, v3_auc)
    save_results(v3_auc, probs, labels)

    print("\n  Output: results/cross_dataset_v3/")
    print("  Done.\n")


if __name__ == "__main__":
    main()