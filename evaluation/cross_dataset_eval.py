"""
Phantom Lens / PRISM — Cross-Dataset Evaluation
Tests trained model on Celeb-DF v2 (never seen during training)
Zero-shot evaluation — no retraining whatsoever

Author: Miheer Satish Kulkarni, IIIT Nagpur
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    accuracy_score, f1_score,
    confusion_matrix
)
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
    print("Make sure this script is in D:\\PhantomLens\\")
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
    if not os.path.exists(CHECKPOINT):
        print(f"ERROR: Checkpoint not found at {CHECKPOINT}")
        sys.exit(1)
    ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
    model = PhysicsClassifier(input_dim=INPUT_DIM).to(DEVICE)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    scaler_mean = ckpt['scaler_mean']
    scaler_std  = ckpt['scaler_std']
    saved_auc   = ckpt.get('auc', 'unknown')
    print(f"Model loaded | Training AUC: {saved_auc:.4f} | Device: {DEVICE}")
    return model, scaler_mean, scaler_std

def predict_video(video_path, model, scaler_mean, scaler_std):
    try:
        feats = extract_features_from_video(video_path, n_frames=N_FRAMES)
        if len(feats) == 0:
            return None
        feat_arr    = np.array(feats, dtype=np.float32)
        feat_mean   = feat_arr.mean(axis=0, keepdims=True)
        feat_scaled = (feat_mean - scaler_mean) / (scaler_std + 1e-8)
        tensor      = torch.FloatTensor(feat_scaled).to(DEVICE)
        with torch.no_grad():
            prob = float(model(tensor).cpu().numpy()[0])
        return prob
    except Exception:
        return None

def evaluate_folder(folder_path, label, model, scaler_mean, scaler_std, max_videos, desc):
    if not os.path.exists(folder_path):
        print(f"ERROR: Folder not found: {folder_path}")
        return []
    videos = sorted([f for f in os.listdir(folder_path)
                     if f.lower().endswith(('.mp4','.avi','.mov','.mkv'))])[:max_videos]
    if len(videos) == 0:
        print(f"ERROR: No videos found in {folder_path}")
        return []
    results = []
    failed  = 0
    for vf in tqdm(videos, desc=desc):
        prob = predict_video(os.path.join(folder_path, vf), model, scaler_mean, scaler_std)
        if prob is None:
            failed += 1
            continue
        results.append((prob, label))
    print(f"  {'REAL' if label==0 else 'FAKE'}: {len(results)} processed | {failed} failed")
    return results

def compute_metrics(all_results):
    probs  = np.array([r[0] for r in all_results])
    labels = np.array([r[1] for r in all_results])
    auc    = roc_auc_score(labels, probs)
    preds  = (probs >= 0.5).astype(int)
    acc    = accuracy_score(labels, preds)
    f1     = f1_score(labels, preds, zero_division=0)
    cm     = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()
    tpr = tp / (tp + fn + 1e-8)
    fpr = fp / (fp + tn + 1e-8)
    tnr = tn / (tn + fp + 1e-8)
    real_count = int((labels==0).sum())
    fake_count = int((labels==1).sum())

    print("\n" + "="*55)
    print("  CELEB-DF V2 — CROSS-DATASET RESULTS")
    print("="*55)
    print(f"  Samples tested  : {real_count} real | {fake_count} fake")
    print(f"  Total           : {len(all_results)}")
    print("-"*55)
    print(f"  AUC             : {auc:.4f}   <-- KEY METRIC")
    print(f"  Accuracy        : {acc:.4f}")
    print(f"  F1 Score        : {f1:.4f}")
    print(f"  Sensitivity TPR : {tpr:.4f}  (fake detection rate)")
    print(f"  Specificity TNR : {tnr:.4f}  (real detection rate)")
    print(f"  False Pos Rate  : {fpr:.4f}")
    print("-"*55)
    print(f"  Confusion Matrix:")
    print(f"    True Real  predicted Real  : {tn}")
    print(f"    True Real  predicted Fake  : {fp}")
    print(f"    True Fake  predicted Real  : {fn}")
    print(f"    True Fake  predicted Fake  : {tp}")
    print("="*55)
    gap = 0.8961 - auc
    print(f"\n  In-distribution AUC (FF++) : 0.8961")
    print(f"  Cross-dataset AUC (CelebDF): {auc:.4f}")
    print(f"  Generalisation gap         : {gap:+.4f}")
    if   auc >= 0.90: verdict = "EXCEPTIONAL — directly publishable in IEEE TIFS"
    elif auc >= 0.85: verdict = "VERY GOOD — strong cross-dataset result"
    elif auc >= 0.80: verdict = "GOOD — physics features are generalising"
    elif auc >= 0.75: verdict = "ACCEPTABLE — needs improvement before TIFS"
    else:             verdict = "NEEDS WORK — investigate which pillars are dropping"
    print(f"\n  Verdict : {verdict}")
    print("="*55)
    return {'auc':auc,'acc':acc,'f1':f1,'tpr':tpr,'fpr':fpr,'tnr':tnr,
            'probs':probs,'labels':labels,'n_real':real_count,'n_fake':fake_count}

def save_roc_curve(metrics):
    fpr_arr, tpr_arr, _ = roc_curve(metrics['labels'], metrics['probs'])
    auc = metrics['auc']
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.set_facecolor('#FAFAFA')
    ax.plot([0,1],[0,1],'k--',alpha=0.4,lw=1.5,label='Random (AUC=0.50)')
    ax.axhline(y=0.8961,color='#185FA5',ls=':',lw=1.5,alpha=0.7,
               label='In-distribution FF++ (AUC=0.8961)')
    ax.plot(fpr_arr,tpr_arr,color='#A32D2D',lw=2.5,
            label=f'Celeb-DF v2 Cross-Dataset (AUC={auc:.4f})')
    ax.fill_between(fpr_arr, tpr_arr, alpha=0.08, color='#A32D2D')
    ax.set_xlim([0.0,1.0]); ax.set_ylim([0.0,1.05])
    ax.set_xlabel('False Positive Rate',fontsize=12)
    ax.set_ylabel('True Positive Rate',fontsize=12)
    ax.set_title('Phantom Lens / PRISM\nCross-Dataset ROC — Celeb-DF v2\n'
                 'Zero-shot: never trained on Celeb-DF v2',
                 fontsize=12,fontweight='bold')
    ax.legend(loc='lower right',fontsize=10)
    ax.grid(True,alpha=0.3)
    ax.spines[['top','right']].set_visible(False)
    out = os.path.join(OUTPUT_DIR,'celebdf_roc_curve.png')
    plt.tight_layout()
    plt.savefig(out,dpi=150,bbox_inches='tight')
    plt.close()
    print(f"\n  ROC curve saved : {out}")

def save_results(metrics):
    out = os.path.join(OUTPUT_DIR,'celebdf_results.txt')
    with open(out,'w') as f:
        f.write("PHANTOM LENS / PRISM — CROSS-DATASET EVALUATION\n")
        f.write("="*55+"\n")
        f.write("Model   : best_model_24dim_AUC8961.pt\n")
        f.write("Dataset : Celeb-DF v2 (ZERO-SHOT — never trained on this)\n")
        f.write("="*55+"\n\n")
        f.write(f"Samples : {metrics['n_real']} real | {metrics['n_fake']} fake\n\n")
        f.write(f"AUC         : {metrics['auc']:.4f}\n")
        f.write(f"Accuracy    : {metrics['acc']:.4f}\n")
        f.write(f"F1 Score    : {metrics['f1']:.4f}\n")
        f.write(f"Sensitivity : {metrics['tpr']:.4f}\n")
        f.write(f"Specificity : {metrics['tnr']:.4f}\n")
        f.write(f"FPR         : {metrics['fpr']:.4f}\n\n")
        f.write(f"In-dist AUC (FF++) : 0.8961\n")
        f.write(f"Cross-dataset AUC  : {metrics['auc']:.4f}\n")
        f.write(f"Gap                : {0.8961-metrics['auc']:+.4f}\n")
    print(f"  Results saved   : {out}")

def main():
    print("\n"+"="*55)
    print("  Phantom Lens / PRISM")
    print("  Cross-Dataset Evaluation — Celeb-DF v2")
    print("  Author: Miheer Satish Kulkarni, IIIT Nagpur")
    print("="*55+"\n")
    model, scaler_mean, scaler_std = load_model()
    print()
    print("Processing REAL videos...")
    real_results = evaluate_folder(REAL_FOLDER, 0, model, scaler_mean, scaler_std, MAX_VIDEOS, "Real videos")
    print("\nProcessing FAKE videos...")
    fake_results = evaluate_folder(FAKE_FOLDER, 1, model, scaler_mean, scaler_std, MAX_VIDEOS, "Fake videos")
    all_results  = real_results + fake_results
    if len(all_results) < 20:
        print("ERROR: Too few results. Check folder paths.")
        sys.exit(1)
    metrics = compute_metrics(all_results)
    save_roc_curve(metrics)
    save_results(metrics)
    print("\n  All output saved to: results/cross_dataset/")
    print("  Done.\n")

if __name__ == "__main__":
    main()