import pickle, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

d   = pickle.load(open('data/precomputed_features_v3_with_celebdf.pkl','rb'))
X   = np.array(d['features'])[:, [i for i in range(30) if i!=20]]
src = np.array(d['dataset_sources'])

print(f"Total: {len(src)}")
print(f"Sources: {dict(zip(*np.unique(src, return_counts=True)))}")

y   = (src == 'ffpp_official').astype(int)
sc  = StandardScaler()
Xs  = sc.fit_transform(X)
lr  = LogisticRegression(max_iter=200)
auc = cross_val_score(lr, Xs, y, cv=3, scoring='roc_auc').mean()

print(f"\nDataset AUC (old pkl) : 0.9638")
print(f"Dataset AUC (new pkl) : {auc:.4f}")
print(f"Improvement           : {0.9638-auc:.4f}")
if auc > 0.70:
    print("STILL CONFOUNDED — but training anyway, diversity helps")
else:
    print("CONFOUND REDUCED — ready to retrain")