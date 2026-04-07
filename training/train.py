# Phantom Lens Research Framework
# Developer: Miheer Satish Kulkarni | IIIT Nagpur | 2026
# All Rights Reserved


import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import cv2

from src.utils.dataset import PhantomLensDataset
from src.pillars.pillar1_noise import compute_pillar1_score
from src.pillars.pillar2_light import compute_pillar2_score
from src.pillars.pillar3_compression import compute_pillar3_score

# ── Config ────────────────────────────────────────────────────────────────────
DATA_ROOT = "data/"
N_FRAMES = 8
EPOCHS = 30
LR = 1e-3
CHECKPOINT_DIR = "checkpoints/"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ── Feature Extraction ────────────────────────────────────────────────────────
def extract_features(frames_numpy):
    T = frames_numpy.shape[0]
    f1_scores = []
    for t in range(T):
        r = compute_pillar1_score(frames_numpy[t])
        f1_scores.append(r['f1_raw'])
    f1_mean = float(np.mean(f1_scores))
    f1_std = float(np.std(f1_scores))

    frames_gray = []
    for t in range(T):
        gray = cv2.cvtColor((frames_numpy[t]*255).astype(np.uint8),
                            cv2.COLOR_RGB2GRAY).astype(np.float32)/255.0
        frames_gray.append(gray)
    r2 = compute_pillar2_score(frames_gray)
    f2_raw = r2['f2_raw']
    f2_var = r2['temporal_variance']

    f3_scores = []
    for t in range(min(T, 4)):
        r3 = compute_pillar3_score(frames_numpy[t])
        f3_scores.append(r3['f3_raw'])
    f3_mean = float(np.mean(f3_scores))
    f3_benford = float(np.mean([compute_pillar3_score(frames_numpy[t])['benford_deviation']
                                for t in range(min(T, 4))]))

    features = np.array([
        f1_mean, f1_std,
        f2_raw, f2_var,
        f3_mean, f3_benford,
        np.clip(f1_mean * f2_raw, 0, 10),
        np.clip(f1_mean * f3_mean, 0, 10),
    ], dtype=np.float32)

    return features

# ── Model ─────────────────────────────────────────────────────────────────────
class FusionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

# ── Data ──────────────────────────────────────────────────────────────────────
train_dataset = PhantomLensDataset(DATA_ROOT, split='train', n_frames=N_FRAMES)
val_dataset = PhantomLensDataset(DATA_ROOT, split='val', n_frames=N_FRAMES)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# ── Training ──────────────────────────────────────────────────────────────────
model = FusionNet()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.BCELoss()
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
best_val_loss = float('inf')

for epoch in range(EPOCHS):
    model.train()
    train_losses = []

    for frames_tensor, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        frames_np = frames_tensor[0].permute(0, 2, 3, 1).numpy()
        features = extract_features(frames_np)
        features_tensor = torch.tensor(features).unsqueeze(0)
        optimizer.zero_grad()
        pred = model(features_tensor)
        loss = criterion(pred, labels)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    model.eval()
    val_losses = []
    all_preds, all_labels = [], []

    with torch.no_grad():
        for frames_tensor, labels in val_loader:
            frames_np = frames_tensor[0].permute(0, 2, 3, 1).numpy()
            features = extract_features(frames_np)
            features_tensor = torch.tensor(features).unsqueeze(0)
            pred = model(features_tensor)
            loss = criterion(pred, labels)
            val_losses.append(loss.item())
            all_preds.append(pred.item())
            all_labels.append(labels.item())

    from sklearn.metrics import roc_auc_score
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    if len(np.unique(all_labels)) > 1:
        auc = roc_auc_score(all_labels, all_preds)
    else:
        auc = 0.0

    print(f"Epoch {epoch+1}: Train Loss={np.mean(train_losses):.4f} | Val Loss={np.mean(val_losses):.4f} | AUC={auc:.4f}")

    scheduler.step()

    if np.mean(val_losses) < best_val_loss:
        best_val_loss = np.mean(val_losses)
        torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, 'best_model.pt'))
        print(f"  ✓ Model saved")

print(f"\nDone. Best Val Loss: {best_val_loss:.4f}")