"""
Quick test - V3 model WITHOUT codec normalisation on Celeb-DF v2
Uses correct class names from train_v3.py
"""
import torch
import numpy as np
import os
import sys
sys.path.insert(0, '.')

from precompute_features_v3 import extract_features_from_video
from train_v3 import PillarAttentionClassifier, LIVE_DIMS, PILLAR_GROUPS
from sklearn.metrics import roc_auc_score

print("Loading model...")
ckpt = torch.load('checkpoints/best_model_v3_AUC9116.pt', map_location='cpu')
m    = PillarAttentionClassifier()
m.load_state_dict(ckpt['model_state'])
m.eval()
sm = ckpt['scaler_mean']
ss = ckpt['scaler_std']
print(f"Model loaded | Training AUC: {ckpt.get('auc',0):.4f}")

results = []
for folder, label, desc in [
    ('data/celebdf/real', 0, 'REAL'),
    ('data/celebdf/fake', 1, 'FAKE'),
]:
    videos = sorted([f for f in os.listdir(folder)
                     if f.endswith('.mp4')])[:100]
    ok = 0
    for vf in videos:
        feats = extract_features_from_video(
            os.path.join(folder, vf), n_frames=16)
        if not feats:
            continue
        feat    = np.array(feats, dtype=np.float32).mean(0, keepdims=True)
        feat_lv = feat[:, LIVE_DIMS]
        feat_sc = (feat_lv - sm) / (ss + 1e-8)
        with torch.no_grad():
            p = float(m(torch.FloatTensor(feat_sc)).numpy()[0])
        results.append((p, label))
        ok += 1
    print(f"  {desc}: {ok} processed")

probs  = np.array([r[0] for r in results])
labels = np.array([r[1] for r in results])

auc       = roc_auc_score(labels, probs)
real_mean = probs[labels==0].mean()
fake_mean = probs[labels==1].mean()

print(f"\n{'='*50}")
print(f"AUC without codec norm : {auc:.4f}")
print(f"Mean prob REAL videos  : {real_mean:.4f}")
print(f"Mean prob FAKE videos  : {fake_mean:.4f}")
print(f"V1 cross-dataset AUC   : 0.4923")
print(f"V3 with codec norm     : 0.4324")
print(f"V3 without codec norm  : {auc:.4f}")
if auc > 0.4923:
    print("VERDICT: Better than V1 — codec norm was hurting")
elif auc > 0.4324:
    print("VERDICT: Better than codec norm version — partial fix")
else:
    print("VERDICT: Deeper problem — not codec norm")
print(f"{'='*50}")