"""
Quick diagnostic — V4 cross-dataset feature analysis
"""
import pickle
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Load test data
print("Loading test data...")
with open('data/v4_test.pkl', 'rb') as f:
    test = pickle.load(f)
features = np.array(test['features'], dtype=np.float32)
labels   = np.array(test['labels'])

real = features[labels==0]
fake = features[labels==1]

print(f"\nTest set: {len(labels)} samples | real={len(real)} fake={len(fake)}")

# Feature stats per class
print("\nFeature means — real vs fake:")
print(f"{'Dim':<5} {'Real':>10} {'Fake':>10} {'Diff':>10} {'Direction'}")
print("-"*45)
for i in range(20):
    rm = real[:,i].mean()
    fm = fake[:,i].mean()
    diff = rm - fm
    direction = "real>fake" if diff > 0 else "fake>real"
    print(f"  {i:<3} {rm:>10.4f} {fm:>10.4f} {diff:>+10.4f}  {direction}")

# Per-feature AUC on test set
print("\nPer-feature AUC on test set (no model needed):")
print(f"{'Dim':<5} {'AUC':>8} {'Signal?'}")
print("-"*25)
best_auc = 0
for i in range(20):
    try:
        auc = roc_auc_score(labels, features[:,i])
        if auc < 0.5:
            auc = 1 - auc  # flip if inverted
        sig = "YES ***" if auc > 0.60 else "mild" if auc > 0.55 else "no"
        if auc > best_auc: best_auc = auc
        print(f"  {i:<3} {auc:>8.4f}  {sig}")
    except:
        print(f"  {i:<3} {'error':>8}")

print(f"\nBest single feature AUC on CelebDF: {best_auc:.4f}")

# Simple logistic regression on raw test features
print("\nLogistic regression on raw test features (train=80%, test=20%):")
np.random.seed(42)
idx = np.random.permutation(len(labels))
split = int(len(labels)*0.8)
tr_idx, te_idx = idx[:split], idx[split:]

sc = StandardScaler()
X_tr = sc.fit_transform(features[tr_idx])
X_te = sc.transform(features[te_idx])
y_tr = labels[tr_idx]
y_te = labels[te_idx]

lr = LogisticRegression(max_iter=1000, C=1.0)
lr.fit(X_tr, y_tr)
probs = lr.predict_proba(X_te)[:,1]
auc = roc_auc_score(y_te, probs)
print(f"  LR AUC on CelebDF features alone: {auc:.4f}")
print(f"  (This tells us if V4 features carry signal on CelebDF)")

if auc > 0.65:
    print("  FEATURES ARE DISCRIMINATIVE — model training issue")
elif auc > 0.55:
    print("  FEATURES HAVE MILD SIGNAL — need better training strategy")
else:
    print("  FEATURES LACK SIGNAL ON CELEBDF — feature redesign needed")