#!/usr/bin/env python3
"""
Cross-Dataset Evaluation — CelebDF v2
======================================
Step 1 : Extract PRISM V3 features from CelebDF (real + fake).
         Saves → features/celebdf_real.csv, features/celebdf_fake.csv
Step 2 : Train multi-manipulation LightGBM on FF++ features (no recomputation).
Step 3 : Test zero-shot on CelebDF.
Step 4 : Report AUC.

NOTE: Uses iterdir()-based recursive walker because rglob/find fails on this
      filesystem mount (NTFS-like directory permissions).
"""

import csv
import json
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

N_WORKERS   = 8    # parallel feature extraction processes
MAX_FRAMES  = 150  # frames sampled per video (300→150 halves extraction time;
                   # rPPG min is 60 frames, so 150 retains all temporal signals)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE    = Path("/home/iiitn/Miheer_project_FE/phantom-lens")
OUTDIR  = BASE / "results" / "exp_celebdf"
FEAT    = BASE / "features"
OUTDIR.mkdir(parents=True, exist_ok=True)

REAL_ROOT = BASE / "data" / "celebdf" / "real"
FAKE_ROOT = BASE / "data" / "celebdf" / "fake"

REAL_CSV  = FEAT / "celebdf_real.csv"
FAKE_CSV  = FEAT / "celebdf_fake.csv"

# FF++ training CSVs (multi-manipulation, same as Exp1)
TRAIN_FILES = [
    FEAT / "ffpp_real_train.csv",
    FEAT / "ffpp_fake.csv",
    FEAT / "ffpp_face2face.csv",
    FEAT / "ffpp_faceswap.csv",
    FEAT / "ffpp_neuraltextures.csv",
]

LGBM_PARAMS = dict(
    n_estimators=200, max_depth=6, learning_rate=0.05,
    num_leaves=31, min_child_samples=20,
    class_weight="balanced", random_state=42, verbose=-1,
)

VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".mov"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def walk_videos(root: Path):
    """Recursively collect video files using iterdir (rglob broken on this FS)."""
    found = []
    stack = [root]
    while stack:
        current = stack.pop()
        try:
            for entry in current.iterdir():
                if entry.is_dir():
                    stack.append(entry)
                elif entry.suffix.lower() in VIDEO_EXTS:
                    found.append(entry)
        except PermissionError:
            pass
    found.sort()
    return found


def load_feature_module():
    """Import process_single_video from the PRISM extractor."""
    sys.path.insert(0, str(FEAT))
    from precompute_features_best import process_single_video, ALL_FEATURE_NAMES
    return process_single_video, ALL_FEATURE_NAMES


# Module-level worker so ProcessPoolExecutor can pickle it.
_FEAT_PATH = str(FEAT)
def _worker_fn(args):
    video_path, label, max_frames = args
    sys.path.insert(0, _FEAT_PATH)
    from precompute_features_best import process_single_video
    return process_single_video(video_path, label, max_frames=max_frames)


def extract_to_csv(video_paths, label, out_csv, feature_names, workers=1):
    """Extract features for a list of videos and write to CSV (parallel-safe)."""
    header = ["video_path", "label"] + feature_names
    mode   = "w"

    existing = set()
    if out_csv.exists():
        try:
            df_ex = pd.read_csv(out_csv)
            existing = set(df_ex["video_path"].astype(str))
            mode = "a"
            print(f"  Resuming — {len(existing)} already done, skipping.")
        except Exception:
            pass

    todo = [p for p in video_paths if str(p) not in existing]
    print(f"  Processing {len(todo)} videos (label={label}) → {out_csv.name}")

    if not todo:
        return

    tasks = [(str(p), label, MAX_FRAMES) for p in todo]
    write_header = (mode == "w")

    with open(out_csv, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if write_header:
            writer.writeheader()

        ok = fail = done = 0

        if workers <= 1:
            for t in tasks:
                result = _worker_fn(t)
                done += 1
                if result is not None:
                    writer.writerow(result)
                    f.flush()
                    ok += 1
                else:
                    fail += 1
                if done % 50 == 0 or done == len(tasks):
                    print(f"    [{done}/{len(tasks)}]  ok={ok}  failed={fail}", flush=True)
        else:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(_worker_fn, t): t[0] for t in tasks}
                for fut in as_completed(futures):
                    done += 1
                    try:
                        result = fut.result()
                    except Exception as e:
                        result = None
                    if result is not None:
                        writer.writerow(result)
                        f.flush()
                        ok += 1
                    else:
                        fail += 1
                    if done % 50 == 0 or done == len(tasks):
                        print(f"    [{done}/{len(tasks)}]  ok={ok}  failed={fail}", flush=True)

    print(f"  Done: {ok} extracted, {fail} failed → {out_csv}", flush=True)


def load_df(csv_paths):
    """Load and concatenate CSVs; return (DataFrame, feature_cols)."""
    dfs = [pd.read_csv(p) for p in csv_paths]
    df  = pd.concat(dfs, ignore_index=True)
    fc  = sorted([c for c in df.columns if c.startswith("s_") or c.startswith("t_")])
    df[fc] = df[fc].replace([np.inf, -np.inf], np.nan)
    for c in fc:
        df[c] = df[c].fillna(df[c].median())
    return df, fc


# ══════════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("  Cross-Dataset Evaluation — CelebDF v2")
print("=" * 65)

# ── STEP 1: Feature extraction ────────────────────────────────────────────────
print(f"\n[1/4] Loading PRISM extractor (workers={N_WORKERS}) ...")
_, FEATURE_NAMES = load_feature_module()   # just grab names; workers import independently

real_videos = walk_videos(REAL_ROOT)
fake_videos = walk_videos(FAKE_ROOT)
print(f"  Found {len(real_videos)} real videos under {REAL_ROOT.relative_to(BASE)}")
print(f"  Found {len(fake_videos)} fake videos under {FAKE_ROOT.relative_to(BASE)}")

if not real_videos:
    sys.exit("ERROR: No real videos found. Check path: data/celebdf/real")
if not fake_videos:
    sys.exit("ERROR: No fake videos found. Check path: data/celebdf/fake")

print("\n[2/4] Extracting features ...")
print("  --- Real videos ---")
extract_to_csv(real_videos, label=0, out_csv=REAL_CSV,
               feature_names=FEATURE_NAMES, workers=N_WORKERS)

print("  --- Fake videos ---")
extract_to_csv(fake_videos, label=1, out_csv=FAKE_CSV,
               feature_names=FEATURE_NAMES, workers=N_WORKERS)

# ── STEP 2: Train on FF++ multi-manipulation ──────────────────────────────────
print("\n[3/4] Training multi-manipulation LightGBM on FF++ ...")

import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

train_df, feat_cols = load_df(TRAIN_FILES)
X_tr = train_df[feat_cols].values.astype(np.float64)
y_tr = train_df["label"].values.astype(int)

scaler  = StandardScaler()
X_tr_sc = scaler.fit_transform(X_tr)

clf = lgb.LGBMClassifier(**LGBM_PARAMS)
clf.fit(pd.DataFrame(X_tr_sc, columns=feat_cols), y_tr, feature_name=feat_cols)
print(f"  Train: {len(y_tr)} samples  (Real={(y_tr==0).sum()}, Fake={(y_tr==1).sum()})")

# ── STEP 3: Test on CelebDF ───────────────────────────────────────────────────
print("\n[4/4] Evaluating on CelebDF ...")

if not REAL_CSV.exists() or not FAKE_CSV.exists():
    sys.exit("ERROR: Feature CSVs missing — extraction may have failed.")

test_df, _ = load_df([REAL_CSV, FAKE_CSV])

# Align columns: some features may be missing if all videos failed
missing = [c for c in feat_cols if c not in test_df.columns]
for c in missing:
    test_df[c] = 0.0

X_te    = test_df[feat_cols].values.astype(np.float64)
X_te_sc = scaler.transform(X_te)
y_te    = test_df["label"].values.astype(int)

y_prob  = clf.predict_proba(pd.DataFrame(X_te_sc, columns=feat_cols))[:, 1]
auc     = roc_auc_score(y_te, y_prob)

n_real  = int((y_te == 0).sum())
n_fake  = int((y_te == 1).sum())

print(f"\n  Test set : {len(y_te)} videos  (Real={n_real}, Fake={n_fake})")
print(f"\n  ╔══════════════════════════════════╗")
print(f"  ║  Cross-Dataset AUC (CelebDF v2)  ║")
print(f"  ║           AUC = {auc:.4f}           ║")
print(f"  ╚══════════════════════════════════╝")
print(f"\n  (Random-chance baseline = 0.5000)")

# ── Save JSON summary ─────────────────────────────────────────────────────────
summary = {
    "experiment": "cross_dataset_celebdf",
    "train_set":  "FF++ multi-manipulation (real_train + Deepfakes + Face2Face + FaceSwap + NeuralTextures)",
    "test_set":   "CelebDF v2 (Celeb-real + YouTube-real + Celeb-synthesis)",
    "classifier": "LightGBM",
    "n_train":    int(len(y_tr)),
    "n_test_real": n_real,
    "n_test_fake": n_fake,
    "n_test_total": int(len(y_te)),
    "cross_dataset_auc": float(auc),
    "random_baseline_auc": 0.5,
    "feature_csv_real": str(REAL_CSV),
    "feature_csv_fake": str(FAKE_CSV),
}
with open(OUTDIR / "results.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\n  Saved: results/exp_celebdf/results.json")
print(f"\n{'='*65}")
print(f"  Cross-dataset evaluation complete.")
print(f"{'='*65}\n")
