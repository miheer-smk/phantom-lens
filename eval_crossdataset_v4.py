"""
Phantom Lens V4 — Cross-Dataset Evaluator (Clean Version)
==========================================================
No residualisation. Pure self-normalisation on test features.
Author: Miheer Satish Kulkarni, IIIT Nagpur
"""
import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (roc_auc_score, accuracy_score, f1_score,
                              confusion_matrix, roc_curve)
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ── CONFIG ────────────────────────────────────────────────────────────────────
CHECKPOINT = "checkpoints/best_model_v4_AUC9127.pt"
TEST_PKL   = "data/v4_test.pkl"
OUTPUT_DIR = "results/v4_crossdataset"
os.makedirs(OUTPUT_DIR, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── MODEL ─────────────────────────────────────────────────────────────────────
PILLAR_GROUPS = [
    [0, 1, 2],
    [3, 4],
    [5, 6, 7],
    [8, 9, 10],
    [13],
    [14, 15, 16],
    [17, 18, 19],
]
DROPOUT_RATES = (0.40, 0.30, 0.20)

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


def main():
    print("="*60)
    print("Phantom Lens V4 — Cross-Dataset Evaluation")
    print("Celeb-DF v2 | Zero-shot | Self-normalisation")
    print("="*60)

    # Load model
    print(f"\nLoading: {CHECKPOINT}")
    ckpt  = torch.load(CHECKPOINT, map_location=DEVICE)
    model = PillarAttentionClassifier(
        pillar_groups=ckpt.get('pillar_groups', PILLAR_GROUPS)
    ).to(DEVICE)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f"Model loaded | Training AUC: {ckpt.get('auc', 0):.4f}")

    # Load test data
    print(f"\nLoading: {TEST_PKL}")
    with open(TEST_PKL, 'rb') as f:
        test_data = pickle.load(f)
    features = np.array(test_data['features'], dtype=np.float32)
    labels   = np.array(test_data['labels'],   dtype=np.int32)
    n_real   = int((labels==0).sum())
    n_fake   = int((labels==1).sum())
    print(f"Samples: {len(labels)} | Real: {n_real} | Fake: {n_fake}")

    # Self-normalisation — normalise test set using its own statistics
    print("\nApplying self-normalisation (test set own statistics)...")
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features).astype(np.float32)
    print("Done.")

    # Run inference
    print("\nRunning inference...")
    all_probs = []
    batch_size = 512
    with torch.no_grad():
        for i in range(0, len(features_scaled), batch_size):
            batch = torch.FloatTensor(features_scaled[i:i+batch_size]).to(DEVICE)
            probs = model(batch).cpu().numpy()
            all_probs.extend(probs)
    probs_arr = np.array(all_probs)

    # Metrics
    auc   = roc_auc_score(labels, probs_arr)
    preds = (probs_arr >= 0.5).astype(int)
    acc   = accuracy_score(labels, preds)
    f1    = f1_score(labels, preds, zero_division=0)
    cm    = confusion_matrix(labels, preds)
    tn, fp, fn_v, tp = cm.ravel()
    tpr = tp / (tp + fn_v + 1e-8)
    tnr = tn / (tn + fp + 1e-8)

    print(f"\n{'='*60}")
    print(f"  V4 CROSS-DATASET RESULTS — CELEB-DF V2")
    print(f"{'='*60}")
    print(f"  Samples  : {n_real} real | {n_fake} fake")
    print(f"  AUC      : {auc:.4f}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1       : {f1:.4f}")
    print(f"  TPR fake : {tpr:.4f}")
    print(f"  TNR real : {tnr:.4f}")
    print(f"  CM: TN={tn} FP={fp} FN={fn_v} TP={tp}")
    print(f"\n  --- COMPARISON ---")
    print(f"  V1 cross-dataset AUC  : 0.4923")
    print(f"  V3 best cross-dataset : 0.5551")
    print(f"  V4 cross-dataset AUC  : {auc:.4f}")
    print(f"  Improvement over V3   : {auc-0.5551:+.4f}")

    if auc >= 0.75:
        status = "EXCELLENT — IEEE TIFS ready"
    elif auc >= 0.70:
        status = "GOOD — proceed to SBI"
    elif auc >= 0.65:
        status = "MODERATE — SBI needed"
    elif auc >= 0.60:
        status = "IMPROVING"
    elif auc >= 0.55:
        status = "MILD IMPROVEMENT"
    else:
        status = "NEEDS MORE WORK"
    print(f"  STATUS: {status}")
    print(f"{'='*60}")

    # ROC plot
    fpr, tpr_c, _ = roc_curve(labels, probs_arr)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0,1],[0,1],'k--',alpha=0.4,lw=1.2,label='Random (0.50)')
    ax.axhline(0.5551,color='#A32D2D',ls=':',lw=1.5,label='V3 best (0.5551)')
    ax.axhline(0.4923,color='gray',ls=':',lw=1.2,label='V1 (0.4923)')
    ax.plot(fpr, tpr_c, color='#185FA5', lw=2.5, label=f'V4 ({auc:.4f})')
    ax.fill_between(fpr, tpr_c, alpha=0.08, color='#185FA5')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'V4 Cross-Dataset — Celeb-DF v2\nAUC={auc:.4f}', fontweight='bold')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'v4_crossdataset_roc.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  ROC saved: {out}")

    # Save results
    res = os.path.join(OUTPUT_DIR, 'v4_crossdataset_results.txt')
    with open(res, 'w') as f:
        f.write(f"V1 cross-dataset : 0.4923\n")
        f.write(f"V3 cross-dataset : 0.5551\n")
        f.write(f"V4 cross-dataset : {auc:.4f}\n")
        f.write(f"Improvement      : {auc-0.5551:+.4f}\n")
        f.write(f"TPR fake         : {tpr:.4f}\n")
        f.write(f"TNR real         : {tnr:.4f}\n")
    print(f"  Results saved: {res}")


if __name__ == "__main__":
    main()