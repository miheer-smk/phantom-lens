# Phantom Lens Research Framework
# Developer: Miheer Satish Kulkarni | IIIT Nagpur | 2026
# All Rights Reserved



"""
Phantom Lens / PRISM — Training Pipeline V2
Video-level split | Dropout | Augmentation | Early stopping
Ablation study | Cross-dataset eval | Multi-seed | Full metrics
Author: Miheer Satish Kulkarni, IIIT Nagpur
"""

import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score
from sklearn.metrics import precision_score, recall_score
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# ─── CONFIG ──────────────────────────────────────────────────────────────────

PKL_PATH      = "data/precomputed_features.pkl"
CHECKPOINT_DIR = "checkpoints"
RESULTS_DIR    = "results"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Training hyperparameters
BATCH_SIZE    = 512
MAX_EPOCHS    = 100
LR            = 0.001
WEIGHT_DECAY  = 1e-4
DROPOUT_RATES = [0.3, 0.2, 0.1]
PATIENCE      = 20          # early stopping patience
SEEDS         = [42, 123, 777]  # 3 seeds for stability
VAL_SPLIT     = 0.20

# Augmentation
AUG_PROB      = 0.5         # probability of applying each augmentation
NOISE_STD     = 0.02        # Gaussian noise std
BRIGHTNESS_RANGE = 0.15     # ±15% brightness shift
COMPRESSION_RANGE = (0.7, 1.0)  # simulated compression factor

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ─── MODEL ───────────────────────────────────────────────────────────────────

class PhysicsClassifier(nn.Module):
    """
    4-layer MLP with BatchNorm and Dropout.
    Input: 12-dim physics feature vector
    Output: probability of being FAKE
    """
    def __init__(self, input_dim=12, dropout_rates=[0.3, 0.2, 0.1]):
        super().__init__()
        self.network = nn.Sequential(
            nn.BatchNorm1d(input_dim),

            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout_rates[0]),

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(dropout_rates[1]),

            nn.Linear(32, 16),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Dropout(dropout_rates[2]),

            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x).squeeze(1)


# ─── DATA LOADING ────────────────────────────────────────────────────────────

def load_data():
    """Load pkl and return features, labels, video_ids, sources."""
    print("Loading pkl...")
    with open(PKL_PATH, 'rb') as f:
        data = pickle.load(f)

    features = data['features'].astype(np.float32)
    labels   = np.array(data['labels'], dtype=np.float32)

    # Handle legacy pkl without video_ids
    if 'video_ids' in data:
        video_ids = data['video_ids']
    else:
        video_ids = [f'video_{i}' for i in range(len(labels))]

    if 'dataset_sources' in data:
        sources = data['dataset_sources']
    else:
        sources = ['unknown'] * len(labels)

    # Sanity check
    assert features.shape[0] == len(labels), "Feature/label mismatch"
    assert features.shape[1] == 24, f"Expected 24-dim features, got {features.shape[1]}"

    real = int(np.sum(labels == 0))
    fake = int(np.sum(labels == 1))
    print(f"Loaded: {len(labels)} samples | Real: {real} | Fake: {fake}")

    return features, labels, video_ids, sources


def video_level_split(features, labels, video_ids, sources, val_split=0.20, seed=42):
    """
    Split by VIDEO ID — all frames from same video stay in same split.
    This prevents AUC inflation from same-video frames in train and val.
    """
    np.random.seed(seed)

    # Group indices by video_id
    video_to_indices = defaultdict(list)
    for i, vid in enumerate(video_ids):
        video_to_indices[vid].append(i)

    # Get unique video IDs with their labels
    unique_videos = list(video_to_indices.keys())
    video_labels = {}
    for vid in unique_videos:
        indices = video_to_indices[vid]
        video_labels[vid] = labels[indices[0]]

    # Stratified split by video
    real_videos = [v for v in unique_videos if video_labels[v] == 0]
    fake_videos = [v for v in unique_videos if video_labels[v] == 1]

    np.random.shuffle(real_videos)
    np.random.shuffle(fake_videos)

    n_real_val = max(1, int(len(real_videos) * val_split))
    n_fake_val = max(1, int(len(fake_videos) * val_split))

    val_videos  = set(real_videos[:n_real_val] + fake_videos[:n_fake_val])
    train_videos = set(real_videos[n_real_val:] + fake_videos[n_fake_val:])

    # Get indices
    train_idx = []
    val_idx   = []
    for vid in unique_videos:
        indices = video_to_indices[vid]
        if vid in train_videos:
            train_idx.extend(indices)
        else:
            val_idx.extend(indices)

    train_idx = np.array(train_idx)
    val_idx   = np.array(val_idx)

    print(f"Video-level split: {len(train_videos)} train videos | "
          f"{len(val_videos)} val videos")
    print(f"Sample split: {len(train_idx)} train | {len(val_idx)} val")

    return (features[train_idx], labels[train_idx],
            [sources[i] for i in train_idx],
            features[val_idx], labels[val_idx],
            [sources[i] for i in val_idx])


# ─── AUGMENTATION ────────────────────────────────────────────────────────────

def augment_features(features, labels):
    """
    Physics-aware feature augmentation.
    Simulates effects of compression, noise, and brightness changes
    on the physics feature vector.
    """
    aug_features = features.copy()
    n = len(aug_features)

    for i in range(n):
        # Gaussian noise augmentation
        if np.random.random() < AUG_PROB:
            noise = np.random.normal(0, NOISE_STD, aug_features.shape[1])
            aug_features[i] += noise.astype(np.float32)

        # Brightness shift (affects pillar 1 and 2 features)
        if np.random.random() < AUG_PROB:
            shift = np.random.uniform(-BRIGHTNESS_RANGE, BRIGHTNESS_RANGE)
            aug_features[i, :3] *= (1 + shift)   # pillar 1 features
            aug_features[i, 3:5] *= (1 + shift)  # pillar 2 features

        # Compression simulation (affects pillar 6 DCT features)
        if np.random.random() < AUG_PROB:
            comp_factor = np.random.uniform(*COMPRESSION_RANGE)
            aug_features[i, 12:15] *= comp_factor  # pillar 6 features

        # Feature dropout (randomly zero one feature)
        if np.random.random() < 0.2:
            zero_idx = np.random.randint(0, aug_features.shape[1])
            aug_features[i, zero_idx] = 0.0

    return aug_features


# ─── TRAINING ────────────────────────────────────────────────────────────────

def train_one_seed(features, labels, video_ids, sources, seed, feature_mask=None):
    """
    Train model with one random seed.
    feature_mask: list of feature indices to use (for ablation)
    Returns best validation AUC and model.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Video-level split
    (X_train, y_train, src_train,
     X_val, y_val, src_val) = video_level_split(
        features, labels, video_ids, sources,
        val_split=VAL_SPLIT, seed=seed
    )

    # Apply feature mask for ablation
    if feature_mask is not None:
        X_train = X_train[:, feature_mask]
        X_val   = X_val[:, feature_mask]
        input_dim = len(feature_mask)
    else:
        input_dim = features.shape[1]

    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_val_scaled   = scaler.transform(X_val).astype(np.float32)

    # Augment training data
    X_train_aug = augment_features(X_train_scaled, y_train)

    # Class weights for imbalance
    n_real = int(np.sum(y_train == 0))
    n_fake = int(np.sum(y_train == 1))
    w_real = len(y_train) / (2 * n_real)
    w_fake = len(y_train) / (2 * n_fake)
    sample_weights = np.where(y_train == 0, w_real, w_fake)
    sampler = WeightedRandomSampler(
        weights=torch.FloatTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True
    )

    # DataLoaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train_aug),
        torch.FloatTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val_scaled),
        torch.FloatTensor(y_val)
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              sampler=sampler)
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                              shuffle=False)

    # Model
    model = PhysicsClassifier(
        input_dim=input_dim,
        dropout_rates=DROPOUT_RATES
    ).to(DEVICE)

    # Loss with class weights
    pos_weight = torch.tensor([w_fake / w_real]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    criterion_bce = nn.BCELoss()

    # Optimizer + scheduler
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MAX_EPOCHS, eta_min=1e-5
    )

    best_auc    = 0.0
    best_model  = None
    patience_counter = 0
    train_losses = []
    val_losses   = []
    val_aucs     = []

    for epoch in range(1, MAX_EPOCHS + 1):
        # Training
        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion_bce(preds, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        scheduler.step()

        # Validation
        model.eval()
        val_preds_list = []
        val_labels_list = []
        val_loss_total = 0.0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                preds = model(X_batch)
                loss = criterion_bce(preds, y_batch)
                val_loss_total += loss.item()
                val_preds_list.extend(preds.cpu().numpy())
                val_labels_list.extend(y_batch.cpu().numpy())

        avg_val_loss = val_loss_total / len(val_loader)
        val_losses.append(avg_val_loss)

        val_preds_arr  = np.array(val_preds_list)
        val_labels_arr = np.array(val_labels_list)
        val_auc = roc_auc_score(val_labels_arr, val_preds_arr)
        val_aucs.append(val_auc)

        val_acc = float(np.mean((val_preds_arr > 0.5) == val_labels_arr))

        print(f"Epoch {epoch:3d}/{MAX_EPOCHS} | "
              f"Train Loss={avg_train_loss:.4f} | "
              f"Val Loss={avg_val_loss:.4f} | "
              f"AUC={val_auc:.4f} | "
              f"Acc={val_acc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            best_model = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            print(f"  ✓ Best model saved (AUC={best_auc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch} (patience={PATIENCE})")
                break

    return best_auc, best_model, scaler, train_losses, val_losses, val_aucs, \
           X_val_scaled, y_val, src_val


# ─── EVALUATION ──────────────────────────────────────────────────────────────

def full_evaluation(model, scaler, X_val, y_val, src_val, seed, results_dir):
    """Compute and save full evaluation metrics."""
    model.eval()
    X_tensor = torch.FloatTensor(X_val).to(DEVICE)

    with torch.no_grad():
        probs = model(X_tensor).cpu().numpy()

    preds_binary = (probs > 0.5).astype(int)

    auc  = roc_auc_score(y_val, probs)
    acc  = float(np.mean(preds_binary == y_val))
    f1   = f1_score(y_val, preds_binary)
    prec = precision_score(y_val, preds_binary, zero_division=0)
    rec  = recall_score(y_val, preds_binary, zero_division=0)
    cm   = confusion_matrix(y_val, preds_binary)

    print(f"\n{'='*50}")
    print(f"EVALUATION RESULTS (seed={seed})")
    print(f"{'='*50}")
    print(f"AUC       : {auc:.4f}")
    print(f"Accuracy  : {acc:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"Confusion Matrix:\n{cm}")

    # Per-dataset AUC breakdown
    unique_sources = list(set(src_val))
    print(f"\nPer-dataset AUC breakdown:")
    for src in sorted(unique_sources):
        indices = [i for i, s in enumerate(src_val) if s == src]
        if len(indices) < 10:
            continue
        src_labels = y_val[indices]
        src_probs  = probs[indices]
        if len(np.unique(src_labels)) < 2:
            continue
        src_auc = roc_auc_score(src_labels, src_probs)
        print(f"  {src:30s} AUC={src_auc:.4f} (n={len(indices)})")

    # Save ROC curve
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_val, probs)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'AUC = {auc:.4f}')
    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve — Phantom Lens V1 (seed={seed})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(results_dir, f'roc_curve_seed{seed}.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    return {
        'auc': auc, 'acc': acc, 'f1': f1,
        'precision': prec, 'recall': rec
    }


def save_loss_curves(train_losses, val_losses, val_aucs, seed, results_dir):
    """Save training curves."""
    epochs = range(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, train_losses, 'b-', label='Train Loss')
    ax1.plot(epochs, val_losses,   'r-', label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'Training vs Validation Loss (seed={seed})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, val_aucs, 'g-', label='Val AUC')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('AUC')
    ax2.set_title(f'Validation AUC over Training (seed={seed})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'training_curves_seed{seed}.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Loss curves saved → results/training_curves_seed{seed}.png")


# ─── ABLATION STUDY ──────────────────────────────────────────────────────────

def run_ablation(features, labels, video_ids, sources):
    """
    Test each pillar individually and in combinations.
    This proves each pillar contributes independently.
    """
    print("\n" + "="*60)
    print("ABLATION STUDY")
    print("="*60)

    # Feature index groups
    pillar1_idx = [0, 1, 2]        # f1_vmr, f1_residual_std, f1_high_freq
    pillar2_idx = [3, 4, 5]        # f2_lighting, f2_specular, f2_shadow
    pillar3_idx = [6, 7, 8]        # f3_benford, f3_block, f3_quant
    all_idx     = list(range(12))  # all 12 features including cross terms

    experiments = [
        ("Pillar 1 only (Noise)",        pillar1_idx),
        ("Pillar 2 only (Light)",         pillar2_idx),
        ("Pillar 3 only (Compression)",   pillar3_idx),
        ("Pillar 1 + 2",                  pillar1_idx + pillar2_idx),
        ("Pillar 1 + 3",                  pillar1_idx + pillar3_idx),
        ("Pillar 2 + 3",                  pillar2_idx + pillar3_idx),
        ("All 3 Pillars (full model)",    all_idx),
    ]

    ablation_results = []

    for exp_name, feature_mask in experiments:
        print(f"\nExperiment: {exp_name}")
        auc, _, _, _, _, _, _, _, _ = train_one_seed(
            features, labels, video_ids, sources,
            seed=42, feature_mask=feature_mask
        )
        ablation_results.append((exp_name, auc))
        print(f"  → AUC: {auc:.4f}")

    print("\n" + "="*60)
    print("ABLATION SUMMARY")
    print("="*60)
    for name, auc in ablation_results:
        bar = "█" * int(auc * 50)
        print(f"  {name:35s} AUC={auc:.4f} {bar}")

    return ablation_results


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Phantom Lens V1 — Training Pipeline V2")
    print(f"Device: {DEVICE}")
    print("=" * 60)

    # Load data
    features, labels, video_ids, sources = load_data()

    # Multi-seed training
    all_aucs = []
    all_metrics = []

    for seed in SEEDS:
        print(f"\n{'='*60}")
        print(f"TRAINING SEED {seed}")
        print(f"{'='*60}")

        (best_auc, best_model_state, scaler,
         train_losses, val_losses, val_aucs,
         X_val, y_val, src_val) = train_one_seed(
            features, labels, video_ids, sources, seed=seed
        )

        all_aucs.append(best_auc)

        # Save loss curves
        save_loss_curves(train_losses, val_losses, val_aucs, seed, RESULTS_DIR)

        # Full evaluation
        model = PhysicsClassifier(input_dim=24, dropout_rates=DROPOUT_RATES).to(DEVICE)
        model.load_state_dict(best_model_state)

        metrics = full_evaluation(model, scaler, X_val, y_val, src_val,
                                  seed, RESULTS_DIR)
        all_metrics.append(metrics)

        # Save checkpoint
        ckpt_path = os.path.join(
            CHECKPOINT_DIR,
            f"best_model_seed{seed}_AUC{int(best_auc*10000)}.pt"
        )
        torch.save({
            'model_state': best_model_state,
            'scaler_mean': scaler.mean_,
            'scaler_std':  scaler.scale_,
            'auc':         best_auc,
            'seed':        seed
        }, ckpt_path)
        print(f"Checkpoint saved: {ckpt_path}")

    # Multi-seed summary
    print(f"\n{'='*60}")
    print("MULTI-SEED SUMMARY")
    print(f"{'='*60}")
    mean_auc = np.mean(all_aucs)
    std_auc  = np.std(all_aucs)
    print(f"AUC across {len(SEEDS)} seeds: {mean_auc:.4f} ± {std_auc:.4f}")
    for i, (seed, auc) in enumerate(zip(SEEDS, all_aucs)):
        print(f"  Seed {seed}: AUC={auc:.4f}")

    # Save best overall model
    best_seed_idx = np.argmax(all_aucs)
    best_seed = SEEDS[best_seed_idx]
    best_overall_auc = all_aucs[best_seed_idx]
    print(f"\nBest model: seed={best_seed}, AUC={best_overall_auc:.4f}")

    best_ckpt = os.path.join(
        CHECKPOINT_DIR,
        f"best_model_seed{best_seed}_AUC{int(best_overall_auc*10000)}.pt"
    )
    import shutil
    shutil.copy(best_ckpt, os.path.join(CHECKPOINT_DIR, "best_model.pt"))
    print(f"Best model saved as checkpoints/best_model.pt")

    # Run ablation study
    ablation_results = run_ablation(features, labels, video_ids, sources)

    # Save full results report
    report_path = os.path.join(RESULTS_DIR, "training_report.txt")
    with open(report_path, 'w') as f:
        f.write("PHANTOM LENS V1 — TRAINING REPORT\n")
        f.write("="*60 + "\n\n")
        f.write(f"AUC: {mean_auc:.4f} ± {std_auc:.4f}\n\n")
        f.write("Per-seed results:\n")
        for seed, auc, metrics in zip(SEEDS, all_aucs, all_metrics):
            f.write(f"  Seed {seed}: AUC={auc:.4f} | "
                    f"F1={metrics['f1']:.4f} | "
                    f"Acc={metrics['acc']:.4f}\n")
        f.write("\nAblation Study:\n")
        for name, auc in ablation_results:
            f.write(f"  {name:35s} AUC={auc:.4f}\n")

    print(f"\nFull report saved: {report_path}")
    print("\nTraining complete!")
    print(f"Final AUC: {mean_auc:.4f} ± {std_auc:.4f}")


if __name__ == "__main__":
    main()