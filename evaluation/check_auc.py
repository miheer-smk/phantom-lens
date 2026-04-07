# Phantom Lens Research Framework
# Developer: Miheer Satish Kulkarni | IIIT Nagpur | 2026
# All Rights Reserved



import pickle
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import sys
sys.path.insert(0, 'D:/PhantomLens')
from train_v2 import PhysicsClassifier

d = pickle.load(open('data/precomputed_features.pkl', 'rb'))
features = np.array(d['features'])
labels = np.array(d['labels'])
sources = np.array(d['dataset_sources'])

ckpt = torch.load('checkpoints/best_model_seed42_AUC8963.pt', map_location='cpu')
scaler = StandardScaler()
scaler.mean_ = ckpt['scaler_mean']
scaler.scale_ = ckpt['scaler_std']
X = scaler.transform(features)

model = PhysicsClassifier(input_dim=24).to('cpu')
model.load_state_dict(ckpt['model_state'])
model.eval()

X_t = torch.tensor(X, dtype=torch.float32)
with torch.no_grad():
    probs = torch.sigmoid(model(X_t)).numpy()

ffpp_mask = sources == 'ffpp_official'
print('FF++ AUC:   ', round(roc_auc_score(labels[ffpp_mask], probs[ffpp_mask]), 4))
print('CelebVHQ AUC:', round(roc_auc_score(labels[~ffpp_mask], probs[~ffpp_mask]), 4))
print('Overall AUC: ', round(roc_auc_score(labels, probs), 4))