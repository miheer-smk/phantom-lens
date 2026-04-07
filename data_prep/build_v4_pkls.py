"""
Phantom Lens V4 — Build Train and Test PKLs from Checkpoints
"""
import pickle
import numpy as np
import os

CKPT_DIR   = "data/v4_checkpoints"
TRAIN_PKL  = "data/v4_train.pkl"
TEST_PKL   = "data/v4_test.pkl"
CELEBVHQ_SUBSAMPLE = 8000
CELEBVHQ_FOLDER    = 6

FOLDER_NAMES = {
    0:"FF++ Real",1:"FF++ Deepfakes",2:"FF++ Face2Face",
    3:"FF++ FaceSwap",4:"FF++ NeuralTextures",5:"FF++ FaceShifter",
    6:"CelebVHQ Real",7:"CelebDF Fake",8:"CelebDF Real",
}

def load_ckpt(idx):
    path = os.path.join(CKPT_DIR, f"v4_ckpt_folder_{idx:02d}.pkl")
    if not os.path.exists(path):
        print(f"  [MISSING] {path}"); return None
    with open(path,"rb") as f:
        data = pickle.load(f)
    # Convert features to list if numpy array
    if isinstance(data["features"], np.ndarray):
        data["features"] = list(data["features"])
    n    = len(data["features"])
    real = sum(1 for l in data["labels"] if l == 0)
    fake = n - real
    print(f"  Loaded folder {idx} ({FOLDER_NAMES.get(idx,'?')}): {n} | real={real} fake={fake}")
    return data

def subsample_ckpt(data, n_keep, seed=42):
    n = len(data["features"])
    if n <= n_keep:
        print(f"    No subsampling needed ({n} <= {n_keep})")
        return data
    rng = np.random.RandomState(seed)
    idx = sorted(rng.choice(n, n_keep, replace=False).tolist())
    print(f"    Subsampled {n} -> {n_keep}")
    return {
        "features":        [data["features"][i]        for i in idx],
        "labels":          [data["labels"][i]           for i in idx],
        "video_ids":       [data["video_ids"][i]        for i in idx],
        "dataset_sources": [data["dataset_sources"][i]  for i in idx],
        "generator_types": [data["generator_types"][i]  for i in idx],
    }

def merge_ckpts(folder_indices, subsample_map=None):
    all_f, all_l, all_v, all_s, all_g = [], [], [], [], []
    for idx in folder_indices:
        data = load_ckpt(idx)
        if data is None:
            continue
        if subsample_map and idx in subsample_map:
            data = subsample_ckpt(data, subsample_map[idx])
        all_f.extend(data["features"])
        all_l.extend(data["labels"])
        all_v.extend(data["video_ids"])
        all_s.extend(data["dataset_sources"])
        all_g.extend(data["generator_types"])
    if not all_f:
        return None
    return {
        "features":        np.array(all_f, dtype=np.float64),
        "labels":          np.array(all_l, dtype=np.int64),
        "video_ids":       all_v,
        "dataset_sources": all_s,
        "generator_types": all_g,
    }

def print_summary(data, name):
    features = data["features"]
    labels   = data["labels"]
    sources  = data["dataset_sources"]
    n        = len(labels)
    nr       = int((labels==0).sum())
    nf       = int((labels==1).sum())
    print(f"\n  {name}:")
    print(f"    Total : {n}")
    print(f"    Real  : {nr} ({100*nr/n:.1f}%)")
    print(f"    Fake  : {nf} ({100*nf/n:.1f}%)")
    print(f"    Dims  : {features.shape[1]}")
    src_arr = np.array(sources)
    for src in sorted(set(sources)):
        mask = src_arr == src
        sr = int(((labels==0) & mask).sum())
        sf = int(((labels==1) & mask).sum())
        print(f"    {src:<22} real={sr:6d}  fake={sf:6d}")
    nn = np.isnan(features).sum()
    ni = np.isinf(features).sum()
    print(f"    {'[OK] No NaN/Inf' if nn==0 and ni==0 else f'[WARN] NaN={nn} Inf={ni}'}")

def main():
    print("="*65)
    print("PHANTOM LENS V4 — Build Train and Test PKLs")
    print("="*65)
    os.makedirs("data", exist_ok=True)

    print("\nChecking checkpoints:")
    available = []
    for idx in range(9):
        path   = os.path.join(CKPT_DIR, f"v4_ckpt_folder_{idx:02d}.pkl")
        exists = os.path.exists(path)
        name   = FOLDER_NAMES.get(idx, f"folder_{idx}")
        print(f"  Folder {idx} ({name:<25}): {'READY' if exists else 'MISSING'}")
        if exists:
            available.append(idx)

    # Build training pkl
    print(f"\n{'─'*65}")
    print("Building TRAINING pkl (FF++ all + CelebVHQ 8000, NO CelebDF)...")
    train_folders = [f for f in [0,1,2,3,4,5,6] if f in available]
    missing       = [f for f in [0,1,2,3,4,5,6] if f not in available]
    if missing:
        print(f"  [WARN] Missing: {[FOLDER_NAMES[f] for f in missing]}")
    train_data = merge_ckpts(
        train_folders,
        subsample_map={CELEBVHQ_FOLDER: CELEBVHQ_SUBSAMPLE}
    )
    if train_data is not None:
        with open(TRAIN_PKL, "wb") as f:
            pickle.dump(train_data, f, protocol=4)
        size_mb = os.path.getsize(TRAIN_PKL) / 1024 / 1024
        print(f"\n  Saved: {TRAIN_PKL} ({size_mb:.1f} MB)")
        print_summary(train_data, "TRAINING PKL")

    # Build test pkl
    print(f"\n{'─'*65}")
    print("Building TEST pkl (CelebDF Real + Fake ONLY)...")
    test_folders = [f for f in [7,8] if f in available]
    missing_test = [f for f in [7,8] if f not in available]
    if missing_test:
        print(f"  [WARN] Missing: {[FOLDER_NAMES[f] for f in missing_test]}")
    test_data = merge_ckpts(test_folders)
    if test_data is not None:
        with open(TEST_PKL, "wb") as f:
            pickle.dump(test_data, f, protocol=4)
        size_mb = os.path.getsize(TEST_PKL) / 1024 / 1024
        print(f"\n  Saved: {TEST_PKL} ({size_mb:.1f} MB)")
        print_summary(test_data, "TEST PKL")

    # Summary
    print(f"\n{'='*65}")
    print("SUMMARY")
    print(f"{'='*65}")
    for path, name in [(TRAIN_PKL,"Training"), (TEST_PKL,"Test")]:
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / 1024 / 1024
            print(f"  {name:<12}: {path} ({size_mb:.1f} MB) READY")
        else:
            print(f"  {name:<12}: NOT BUILT YET")
    print(f"\nNext steps:")
    print(f"  1. python analyze_v4_pkl.py --pkl data/v4_train.pkl")
    print(f"  2. python train_v4.py")
    print(f"{'='*65}")

if __name__ == "__main__":
    main()