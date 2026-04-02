import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

d   = pickle.load(open('data/precomputed_features_v3_base.pkl','rb'))
X   = np.array(d['features'])[:, [i for i in range(30) if i != 20]]
src = np.array(d['dataset_sources'])
y   = (src == 'ffpp_official').astype(int)

# Before normalisation
sc1  = StandardScaler()
auc1 = cross_val_score(LogisticRegression(max_iter=200),
                        sc1.fit_transform(X), y,
                        cv=3, scoring='roc_auc').mean()

# After per-video L2 normalisation
X_pv = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
sc2  = StandardScaler()
auc2 = cross_val_score(LogisticRegression(max_iter=200),
                        sc2.fit_transform(X_pv), y,
                        cv=3, scoring='roc_auc').mean()

print(f"Dataset AUC before norm : {auc1:.4f}  {'CONFOUNDED' if auc1>0.70 else 'OK'}")
print(f"Dataset AUC after norm  : {auc2:.4f}  {'STILL CONFOUNDED' if auc2>0.70 else 'FIXED'}")
print()
if auc2 < auc1:
    print(f"Improvement: {auc1-auc2:.4f} reduction in confounding")
if auc2 <= 0.70:
    print("Ready to retrain with per-video normalisation")
else:
    print("Need additional deconfounding steps")