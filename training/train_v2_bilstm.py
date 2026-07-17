"""
Phantom Lens — BiLSTM Experiment on V2 Features
Safe: original model (AUC 0.8961) never touched

Run MORNING: phantomlens_env\Scripts\python.exe train_v2_bilstm.py
Time: ~30-45 minutes
Output: checkpoints/bilstm_experiment/
"""

import os, pickle, warnings
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
warnings.filterwarnings('ignore')

# ── CONFIG ────────────────────────────────────────────────────────
SEQ_PKL    = "data/v2_sequences.pkl"
CKPT_DIR   = "checkpoints/bilstm_experiment"
os.makedirs(CKPT_DIR, exist_ok=True)

SEEDS        = [42, 123, 777, 999, 2024]
N_FRAMES     = 8
N_DIMS       = 24
HIDDEN       = 64
BATCH_SIZE   = 128
MAX_EPOCHS   = 80
LR           = 0.001
WEIGHT_DECAY = 1e-3
PATIENCE     = 10
VAL_SPLIT    = 0.20
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("="*60)
print("BiLSTM EXPERIMENT — V2 FEATURES")
print(f"Device: {DEVICE}")
print("Original MLP baseline: AUC 0.8961 ± 0.0007")
print("="*60)

# ── LOAD DATA ─────────────────────────────────────────────────────
print(f"\nLoading {SEQ_PKL}...")
with open(SEQ_PKL,'rb') as f: data=pickle.load(f)

X_raw = data['sequences']   # (N, 8, 24)
y_all = data['labels']      # (N,)
src   = data['sources']
gt    = data['gen_types']

print(f"  Sequences: {X_raw.shape}")
print(f"  Real: {(y_all==0).sum()} | Fake: {(y_all==1).sum()}")

# ── TEMPORAL DELTA AUGMENTATION ───────────────────────────────────
def add_temporal_deltas(X):
    """
    Augment sequence with frame-to-frame differences.
    Physics: GAN frames are partially independent — delta captures violations.
    Input:  (N, 8, 24)
    Output: (N, 8, 48)  [original + delta, delta padded with zeros at t=0]
    """
    N, T, D = X.shape
    deltas = np.zeros_like(X)
    deltas[:, 1:, :] = X[:, 1:, :] - X[:, :-1, :]  # delta[0] = 0
    return np.concatenate([X, deltas], axis=2)  # (N, 8, 48)

print("\nApplying temporal delta augmentation...")
X_aug = add_temporal_deltas(X_raw)   # (N, 8, 48)
print(f"  Augmented shape: {X_aug.shape}")

# ── DATASET ───────────────────────────────────────────────────────
class SeqDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self):  return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]

# ── MODEL ─────────────────────────────────────────────────────────
class PhysicsBiLSTM(nn.Module):
    """
    BiLSTM over per-frame physics pillar sequence.
    Input:  (batch, 8, 48)  — 8 frames, 48 features (24 original + 24 delta)
    Output: (batch, 1)      — fake probability logit
    """
    def __init__(self, input_dim=48, hidden=64, n_layers=2, dropout=0.3):
        super().__init__()

        # Per-frame encoder: maps 48-dim → 64-dim
        self.frame_encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
        )

        # BiLSTM: reads encoded sequence
        self.bilstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )

        # Attention over timesteps
        self.attn = nn.Linear(hidden*2, 1)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden*2, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.40),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.30),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        # x: (batch, 8, 48)
        B, T, D = x.shape

        # Encode each frame independently
        x_flat = x.reshape(B*T, D)
        enc    = self.frame_encoder(x_flat)   # (B*T, 64)
        enc    = enc.reshape(B, T, 64)        # (B, 8, 64)

        # BiLSTM over frame sequence
        lstm_out, _ = self.bilstm(enc)        # (B, 8, 128)

        # Temporal attention: which frames matter most?
        attn_w = torch.softmax(self.attn(lstm_out), dim=1)  # (B, 8, 1)
        context = (attn_w * lstm_out).sum(dim=1)             # (B, 128)

        # Classify
        out = self.classifier(context)        # (B, 1)
        return out.squeeze(1)                 # (B,)

    def get_attention_weights(self, x):
        """Returns attention weights for interpretability."""
        B, T, D = x.shape
        x_flat = x.reshape(B*T, D)
        enc    = self.frame_encoder(x_flat).reshape(B, T, 64)
        lstm_out, _ = self.bilstm(enc)
        attn_w = torch.softmax(self.attn(lstm_out), dim=1)
        return attn_w.squeeze(-1)  # (B, 8)

# ── TRAINING LOOP ─────────────────────────────────────────────────
def train_one_seed(seed):
    print(f"\n{'─'*40}")
    print(f"  Seed {seed}")
    print(f"{'─'*40}")
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_aug, y_all, test_size=VAL_SPLIT,
        stratify=y_all, random_state=seed)

    # Normalise per feature (across N and T dims)
    N_tr, T, D = X_tr.shape
    X_tr_flat  = X_tr.reshape(-1, D)
    scaler     = StandardScaler()
    X_tr_flat  = scaler.fit_transform(X_tr_flat)
    X_tr       = X_tr_flat.reshape(N_tr, T, D).astype(np.float32)

    N_val      = X_val.shape[0]
    X_val_flat = X_val.reshape(-1, D)
    X_val_flat = scaler.transform(X_val_flat)
    X_val      = X_val_flat.reshape(N_val, T, D).astype(np.float32)

    # Sampler
    n_real = int((y_tr==0).sum()); n_fake = int((y_tr==1).sum())
    w_real = len(y_tr)/(2*n_real); w_fake = len(y_tr)/(2*n_fake)
    weights = np.where(y_tr==0, w_real, w_fake)
    sampler = WeightedRandomSampler(
        torch.tensor(weights,dtype=torch.float32),
        num_samples=len(y_tr), replacement=True)

    tr_loader  = DataLoader(SeqDataset(X_tr, y_tr),
                             batch_size=BATCH_SIZE, sampler=sampler)
    val_loader = DataLoader(SeqDataset(X_val, y_val),
                             batch_size=BATCH_SIZE, shuffle=False)

    # Model
    model    = PhysicsBiLSTM(input_dim=48, hidden=HIDDEN).to(DEVICE)
    pos_w    = torch.tensor([n_real/n_fake], dtype=torch.float32).to(DEVICE)
    criterion= nn.BCEWithLogitsLoss(pos_weight=pos_w)
    optimizer= torch.optim.AdamW(model.parameters(),
                                  lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler= torch.optim.lr_scheduler.CosineAnnealingLR(
                  optimizer, T_max=MAX_EPOCHS)

    best_auc = 0.0; patience_cnt = 0
    best_path= os.path.join(CKPT_DIR, f"bilstm_seed{seed}.pt")

    for epoch in range(MAX_EPOCHS):
        # Train
        model.train()
        for X_b, y_b in tr_loader:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            optimizer.zero_grad()
            logits = model(X_b)
            loss   = criterion(logits, y_b)
            loss.backward(); optimizer.step()
        scheduler.step()

        # Validate
        model.eval()
        all_probs=[]; all_y=[]
        with torch.no_grad():
            for X_b, y_b in val_loader:
                logits = model(X_b.to(DEVICE))
                probs  = torch.sigmoid(logits).cpu().numpy()
                all_probs.extend(probs); all_y.extend(y_b.numpy())

        auc = roc_auc_score(all_y, all_probs)

        if auc > best_auc:
            best_auc = auc; patience_cnt = 0
            torch.save({'model':model.state_dict(),
                        'scaler':scaler, 'auc':auc,
                        'seed':seed}, best_path)
        else:
            patience_cnt += 1

        if (epoch+1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d} | Val AUC: {auc:.4f} "
                  f"| Best: {best_auc:.4f} | Patience: {patience_cnt}/{PATIENCE}")

        if patience_cnt >= PATIENCE:
            print(f"  Early stop at epoch {epoch+1}")
            break

    print(f"  Seed {seed} BEST AUC: {best_auc:.4f}")
    return best_auc

# ── RUN ALL SEEDS ─────────────────────────────────────────────────
all_aucs = []
for seed in SEEDS:
    auc = train_one_seed(seed)
    all_aucs.append(auc)

mean_auc = np.mean(all_aucs)
std_auc  = np.std(all_aucs)

# ── FINAL REPORT ──────────────────────────────────────────────────
print("\n" + "="*60)
print("FINAL RESULTS")
print("="*60)
print(f"  BiLSTM AUCs per seed: {[f'{a:.4f}' for a in all_aucs]}")
print(f"  BiLSTM Mean AUC:  {mean_auc:.4f} ± {std_auc:.4f}")
print()
print(f"  MLP Baseline:     0.8961 ± 0.0007")
print(f"  BiLSTM Result:    {mean_auc:.4f} ± {std_auc:.4f}")
diff = mean_auc - 0.8961
print(f"  Difference:       {diff:+.4f}")
print()
if mean_auc > 0.8961:
    print("  ✅ BiLSTM IMPROVES over MLP baseline")
    print(f"  Temporal modeling adds +{diff:.4f} AUC")
elif mean_auc > 0.88:
    print("  ⚠️ BiLSTM roughly MATCHES MLP baseline")
    print("  Temporal features don't hurt — comparable performance")
else:
    print("  ❌ BiLSTM UNDERPERFORMS MLP on this dataset")
    print("  MLP averaged features are sufficient for V2")
print()
print("  Checkpoints saved to:", CKPT_DIR)
print("  Original model UNTOUCHED: checkpoints/best_model_24dim_AUC8961.pt")

# Save summary
summary = {
    'bilstm_aucs': all_aucs,
    'bilstm_mean': mean_auc,
    'bilstm_std':  std_auc,
    'mlp_mean':    0.8961,
    'mlp_std':     0.0007,
}
with open(os.path.join(CKPT_DIR,'results_summary.pkl'),'wb') as f:
    pickle.dump(summary, f)
print(f"\nSummary saved to {CKPT_DIR}/results_summary.pkl")