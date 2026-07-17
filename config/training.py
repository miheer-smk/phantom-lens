# Phantom Lens Research Framework
# Developer: Miheer Satish Kulkarni | IIIT Nagpur | 2026
# All Rights Reserved

"""
Centralized training hyperparameter configuration.

These reflect the production V4 defaults.  Per-script overrides can
shadow individual names; this module serves as the single source of truth.
"""

# ── Reproducibility ───────────────────────────────────────────────────────────
SEEDS = [42, 123, 777, 999, 2024]

# ── Optimisation ─────────────────────────────────────────────────────────────
BATCH_SIZE   = 256
MAX_EPOCHS   = 80
LR           = 0.001
WEIGHT_DECAY = 1e-3
PATIENCE     = 10          # early-stopping patience (epochs)

# ── Architecture ─────────────────────────────────────────────────────────────
DROPOUT_RATES = (0.40, 0.30, 0.20)

# ── Feature dimensions per version ───────────────────────────────────────────
FEATURE_DIM = {
    "v1":   8,
    "v2":   8,
    "v3":  50,
    "v4":  20,
    "best": 50,
}

# ── Pillar groups for V4 attention model ─────────────────────────────────────
# Each tuple: (pillar_name, start_dim, end_dim_exclusive)
PILLAR_GROUPS_V4 = [
    ("noise",      0,  3),
    ("prnu",       3,  5),
    ("shadow",     5,  8),
    ("dct",        8, 11),
    ("flow",      11, 13),
    ("blend",     13, 17),
    ("checkerbd", 17, 20),
]

# ── SBI augmentation probabilities ───────────────────────────────────────────
SBI_PROB          = 0.30   # fraction of real samples to augment
SBI_NOISE_REDUCE  = 0.40
SBI_PRNU_REDUCE   = 0.40
SBI_DCT_BOOST     = 0.25
SBI_BLEND_PERTURB = 0.30

# ── Frame sampling ────────────────────────────────────────────────────────────
FRAMES_PER_VIDEO = 8        # default; some scripts use 16
FRAME_RESOLUTION = 224      # pixels (square)
