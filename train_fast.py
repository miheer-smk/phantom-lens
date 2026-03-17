# Phantom Lens Research Framework
# Developer: Miheer Satish Kulkarni | IIIT Nagpur | 2026
# All Rights Reserved


import torch
import torch.nn as nn
import numpy as np
import pickle
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score
import os

# ── Load precomputed features ─────────────────────────────────────────────────
print("Loading precomputed features...")
with open('data/precomputed_features.pkl', 'rb') as f:
    data = pickle.load(f)

features = data['features']
labels = data['labels']
print(f"Loaded {len(features)} samples")

# ── Dataset ───────────────────────────────────────────────────────────────────
class FeatureDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
    def __len__(self):
        return len(self.features)
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# Train/val split
n = len(features)
idx = np.random.permutation(n)
train_idx = idx[:int(0.8*n)]
val_idx = idx[int(0.8*n):]

train_dataset = FeatureDataset(features[train_idx], labels[train_idx])
val_dataset = FeatureDataset(features[val_idx], labels[val_idx])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")

# ── Model ─────────────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class FusionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

model = FusionNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.BCELoss()
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

# ── Training ──────────────────────────────────────────────────────────────────
EPOCHS = 50
best_auc = 0.0
os.makedirs('checkpoints', exist_ok=True)

for epoch in range(EPOCHS):
    model.train()
    train_losses = []
    for feats, labs in train_loader:
        feats, labs = feats.to(device), labs.to(device)
        optimizer.zero_grad()
        preds = model(feats)
        loss = criterion(preds, labs)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for feats, labs in val_loader:
            feats = feats.to(device)
            preds = model(feats)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labs.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    if len(np.unique(all_labels)) > 1:
        auc = roc_auc_score(all_labels, all_preds)
        acc = accuracy_score(all_labels, (all_preds > 0.5).astype(int))
    else:
        auc = 0.0
        acc = 0.0

    scheduler.step()

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss={np.mean(train_losses):.4f} | AUC={auc:.4f} | Acc={acc:.4f}")

    if auc > best_auc:
        best_auc = auc
        torch.save(model.state_dict(), 'checkpoints/best_model.pt')
        print(f"  ✓ Best model saved (AUC={auc:.4f})")

print(f"\nTraining complete. Best AUC: {best_auc:.4f}")
