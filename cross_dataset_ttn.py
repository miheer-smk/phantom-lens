"""
Phantom Lens / PRISM — Cross-Dataset Eval with Test-Time Normalisation
Quick fix: no retraining, same checkpoint, just fixes the scaler problem.
Author: Miheer Satish Kulkarni, IIIT Nagpur
"""

import os, sys
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from precompute_features_v2 import extract_features_from_video
    print("Feature extractor loaded successfully")
except ImportError:
    print("ERROR: Cannot import precompute_features_v2.py")
    sys.exit(1)

CHECKPOINT  = "checkpoints/best_model_24dim_AUC8961.pt"
REAL_FOLDER = "data/celebdf/real"
FAKE_FOLDER = "data/celebdf/fake"
OUTPUT_DIR  = "results/cross_dataset"
N_FRAMES    = 8
MAX_VIDEOS  = 500
INPUT_DIM   = 24
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(OUTPUT_DIR, exist_ok=True)

class PhysicsClassifier(nn.Module):
    def __init__(self, input_dim=24):
        super().__init__()
        self.network = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.BatchNorm1d(64), nn.Dropout(0.3),
            nn.Linear(64, 32), nn.ReLU(),
            nn.BatchNorm1d(32), nn.Dropout(0.2),
            nn.Linear(32, 16), nn.ReLU(),
            nn.BatchNorm1d(16), nn.Dropout(0.1),
            nn.Linear(16, 1), nn.Sigmoid()
        )
    def forward(self, x):
        return self.network(x).squeeze(1)

def load_model():
    ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
    model = PhysicsClassifier(input_dim=INPUT_DIM).to(DEVICE)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    mean = ckpt['scaler_mean']
    std  = ckpt['scaler_std']
    print(f"Model loaded | Training AUC: {ckpt.get('auc',0):.4f} | Device: {DEVICE}")
    return model, mean, std

def extract_all_features(folder_path, label, max_videos, desc):
    videos = sorted([f for f in os.listdir(folder_path)
                     if f.lower().endswith(('.mp4','.avi','.mov','.mkv'))])[:max_videos]
    feats, labels, failed = [], [], 0
    for vf in tqdm(videos, desc=desc):
        f = extract_features_from_video(os.path.join(folder_path, vf), n_frames=N_FRAMES)
        if len(f) == 0:
            failed += 1
            continue
        feats.append(np.array(f, dtype=np.float32).mean(axis=0))
        labels.append(label)
    print(f"  {'REAL' if label==0 else 'FAKE'}: {len(feats)} extracted | {failed} failed")
    return np.array(feats, dtype=np.float32), np.array(labels, dtype=np.int32)

def get_probs(features, model, mean, std):
    scaled = (features - mean) / (std + 1e-8)
    with torch.no_grad():
        probs = model(torch.FloatTensor(scaled).to(DEVICE)).cpu().numpy()
    return probs

def show_metrics(probs, labels, name):
    auc   = roc_auc_score(labels, probs)
    preds = (probs >= 0.5).astype(int)
    acc   = accuracy_score(labels, preds)
    f1    = f1_score(labels, preds, zero_division=0)
    cm    = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()
    tpr = tp / (tp + fn + 1e-8)
    tnr = tn / (tn + fp + 1e-8)
    if   auc >= 0.90: v = "EXCEPTIONAL"
    elif auc >= 0.85: v = "VERY GOOD"
    elif auc >= 0.80: v = "GOOD"
    elif auc >= 0.75: v = "ACCEPTABLE"
    elif auc >= 0.65: v = "IMPROVING"
    else:             v = "NEEDS MORE WORK"
    print(f"\n  [{name}]")
    print(f"  AUC      : {auc:.4f}  {v}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1       : {f1:.4f}")
    print(f"  TPR fake : {tpr:.4f}  |  TNR real : {tnr:.4f}")
    return auc

def main():
    print("\n"+"="*60)
    print("  Phantom Lens — Cross-Dataset TTN Fix")
    print("  Celeb-DF v2 | No retraining | Same checkpoint")
    print("="*60+"\n")

    model, train_mean, train_std = load_model()

    # ── Extract features ──────────────────────────────────────────────────────
    print("\nExtracting Celeb-DF v2 features...")
    real_f, real_l = extract_all_features(REAL_FOLDER, 0, MAX_VIDEOS, "Real")
    fake_f, fake_l = extract_all_features(FAKE_FOLDER, 1, MAX_VIDEOS, "Fake")
    X = np.vstack([real_f, fake_f])
    y = np.concatenate([real_l, fake_l])
    print(f"\nTotal: {len(y)} samples | Shape: {X.shape}")

    # ── Method A: original FF++ scaler ────────────────────────────────────────
    print("\n"+"="*60)
    print("  METHOD COMPARISON")
    print("="*60)
    probs_a = get_probs(X, model, train_mean, train_std)
    auc_a   = show_metrics(probs_a, y, "Method A: FF++ scaler (original result)")

    # ── Method B: test-time normalisation ─────────────────────────────────────
    sc = StandardScaler()
    sc.fit(X)
    probs_b = get_probs(X, model,
                        sc.mean_.astype(np.float32),
                        sc.scale_.astype(np.float32))
    auc_b   = show_metrics(probs_b, y, "Method B: Test-time norm (Celeb-DF stats)")

    # ── Method C: hybrid ──────────────────────────────────────────────────────
    h_mean = ((train_mean + sc.mean_.astype(np.float32)) / 2.0)
    h_std  = ((train_std  + sc.scale_.astype(np.float32)) / 2.0)
    probs_c = get_probs(X, model, h_mean, h_std)
    auc_c   = show_metrics(probs_c, y, "Method C: Hybrid norm (average of both)")

    # ── Summary ───────────────────────────────────────────────────────────────
    best = max(auc_a, auc_b, auc_c)
    print("\n"+"="*60)
    print("  SUMMARY")
    print("="*60)
    print(f"  In-distribution FF++ AUC  : 0.8961")
    print(f"  Method A (FF++ scaler)    : {auc_a:.4f}  (original)")
    print(f"  Method B (test-time norm) : {auc_b:.4f}  <-- quick fix")
    print(f"  Method C (hybrid norm)    : {auc_c:.4f}  <-- balanced")
    print(f"\n  Best AUC      : {best:.4f}")
    print(f"  Improvement   : {best - auc_a:+.4f} over original")
    print("="*60)

    # ── ROC plot ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9,7))
    ax.set_facecolor('#FAFAFA')
    ax.plot([0,1],[0,1],'k--',alpha=0.4,lw=1.2,label='Random (0.50)')
    ax.axhline(y=0.8961,color='gray',ls=':',lw=1.2,
               label='In-distribution FF++ (0.8961)')
    for probs,auc,color,lbl in [
        (probs_a,auc_a,'#888780',f'A: FF++ scaler ({auc_a:.4f})'),
        (probs_b,auc_b,'#A32D2D',f'B: Test-time norm ({auc_b:.4f})'),
        (probs_c,auc_c,'#185FA5',f'C: Hybrid norm ({auc_c:.4f})'),
    ]:
        fpr_r,tpr_r,_ = roc_curve(y, probs)
        ax.plot(fpr_r,tpr_r,color=color,lw=2.2,label=lbl)
    ax.set_xlim([0,1]); ax.set_ylim([0,1.05])
    ax.set_xlabel('False Positive Rate',fontsize=12)
    ax.set_ylabel('True Positive Rate',fontsize=12)
    ax.set_title('Phantom Lens / PRISM\nCeleb-DF v2 Normalisation Comparison\n'
                 'No retraining — same checkpoint',
                 fontsize=12,fontweight='bold')
    ax.legend(loc='lower right',fontsize=10)
    ax.grid(True,alpha=0.3)
    ax.spines[['top','right']].set_visible(False)
    out = os.path.join(OUTPUT_DIR,'celebdf_ttn_comparison.png')
    plt.tight_layout()
    plt.savefig(out,dpi=150,bbox_inches='tight')
    plt.close()
    print(f"\n  ROC curve saved: {out}")

    # ── Save txt ──────────────────────────────────────────────────────────────
    txt = os.path.join(OUTPUT_DIR,'celebdf_ttn_results.txt')
    with open(txt,'w') as f:
        f.write("PHANTOM LENS — TTN COMPARISON RESULTS\n")
        f.write("="*50+"\n")
        f.write(f"In-dist FF++ AUC  : 0.8961\n")
        f.write(f"Method A AUC      : {auc_a:.4f}\n")
        f.write(f"Method B AUC      : {auc_b:.4f}\n")
        f.write(f"Method C AUC      : {auc_c:.4f}\n")
        f.write(f"Best improvement  : {best-auc_a:+.4f}\n")
    print(f"  Results saved  : {txt}")
    print("\n  Done.\n")

if __name__ == "__main__":
    main()