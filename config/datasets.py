# Phantom Lens Research Framework
# Developer: Miheer Satish Kulkarni | IIIT Nagpur | 2026
# All Rights Reserved

"""
Centralized dataset path configuration.

Override by setting environment variables or editing this file.
All paths are relative to the project root unless absolute.
"""

import os

# ── Raw video roots ──────────────────────────────────────────────────────────
FFPP_ROOT            = os.getenv("FFPP_ROOT",            "data/ffpp")
CELEBDF_ROOT         = os.getenv("CELEBDF_ROOT",         "data/celebdf")
CELEBVHQ_ROOT        = os.getenv("CELEBVHQ_ROOT",        "data/celebvhq")
DEEPERFORENSICS_ROOT = os.getenv("DEEPERFORENSICS_ROOT", "data/deeperforensics")
DFFD_ROOT            = os.getenv("DFFD_ROOT",            "data/dffd")
WILDDEEPFAKE_ROOT    = os.getenv("WILDDEEPFAKE_ROOT",    "data/wilddeepfake")

# ── Processed frame roots (after prepare_*.py) ───────────────────────────────
FRAMES_ROOT = os.getenv("FRAMES_ROOT", "data/frames")

# ── Pre-computed feature pickle paths ────────────────────────────────────────
V1_PKL   = os.getenv("V1_PKL",   "data/v1_train.pkl")
V2_PKL   = os.getenv("V2_PKL",   "data/v2_train.pkl")
V3_PKL   = os.getenv("V3_PKL",   "data/v3_train.pkl")
V4_PKL   = os.getenv("V4_PKL",   "data/v4_train.pkl")
BEST_PKL = os.getenv("BEST_PKL", "data/best_train.pkl")

# ── Cross-dataset evaluation targets ─────────────────────────────────────────
CELEBDF_EVAL_PKL = os.getenv("CELEBDF_EVAL_PKL", "data/celebdf_eval.pkl")

# ── Checkpoint directory ──────────────────────────────────────────────────────
CHECKPOINT_DIR = os.getenv("CHECKPOINT_DIR", "checkpoints")
