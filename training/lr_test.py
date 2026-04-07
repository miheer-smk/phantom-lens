"""
Quick LR cross-dataset test
"""
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

print("Loading data...")
with open('data/v4_train.pkl','rb') as f:
    train = pickle.load(f)
with open('data/v4_test.pkl','rb') as f:
    test = pickle.load(f)

X_tr = np.array(train['features'], dtype=np.float32)
y_tr = np.array(train['labels'])
X_te = np.array(test['features'], dtype=np.float32)
y_te = np.array(test['labels'])

print(f"Train: {len(y_tr)} | Test: {len(y_te)}")

sc = StandardScaler()
X_tr_sc = sc.fit_transform(X_tr)
X_te_sc = sc.transform(X_te)

print("\nTraining LR on training data, testing on CelebDF...")
for C in [0.01, 0.1, 1.0]:
    lr = LogisticRegression(max_iter=1000, C=C)
    lr.fit(X_tr_sc, y_tr)
    probs = lr.predict_proba(X_te_sc)[:,1]
    auc = roc_auc_score(y_te, probs)
    print(f"  C={C:<6} AUC={auc:.4f}")

print("\nNow with test self-normalisation...")
sc2 = StandardScaler()
X_te_self = sc2.fit_transform(X_te)
for C in [0.01, 0.1, 1.0]:
    lr = LogisticRegression(max_iter=1000, C=C)
    lr.fit(X_tr_sc, y_tr)
    probs = lr.predict_proba(X_te_self)[:,1]
    auc = roc_auc_score(y_te, probs)
    print(f"  C={C:<6} AUC={auc:.4f} (self-norm)")

print("\nDone.")