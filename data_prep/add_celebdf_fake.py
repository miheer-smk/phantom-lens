"""
Phantom Lens V3 — Add Celeb-DF v2 Fake Videos to Training PKL
Precomputes 30-dim features for Celeb-DF v2 fake videos
Merges into existing precomputed_features_v3_base.pkl
Author: Miheer Satish Kulkarni, IIIT Nagpur
"""
import os
import sys
import pickle
import numpy as np
from tqdm import tqdm
sys.path.insert(0, '.')
from precompute_features_v3 import extract_features_from_video

FAKE_FOLDER  = "data/celebdf/fake"
BASE_PKL     = "data/precomputed_features_v3_base.pkl"
OUTPUT_PKL   = "data/precomputed_features_v3_with_celebdf.pkl"
N_FRAMES     = 16
CHECKPOINT   = "data/celebdf_fake_checkpoint.pkl"

print("="*60)
print("  Adding Celeb-DF v2 Fake Videos to V3 PKL")
print("="*60)

# Load existing pkl
print(f"\nLoading base pkl: {BASE_PKL}")
with open(BASE_PKL, 'rb') as f:
    data = pickle.load(f)

features  = list(data['features'])
labels    = list(data['labels'])
video_ids = list(data.get('video_ids', [str(i) for i in range(len(labels))]))
sources   = list(data.get('dataset_sources', ['unknown']*len(labels)))
gen_types = list(data.get('generator_types', ['unknown']*len(labels)))

print(f"  Current samples: {len(labels)}")
print(f"  Real: {sum(1 for l in labels if l==0)}")
print(f"  Fake: {sum(1 for l in labels if l==1)}")

# Load checkpoint if exists
done_videos = set()
if os.path.exists(CHECKPOINT):
    with open(CHECKPOINT, 'rb') as f:
        ckpt = pickle.load(f)
    done_videos = set(ckpt['done_videos'])
    print(f"\nCheckpoint found: {len(done_videos)} videos already processed")

# Get all fake videos
videos = sorted([f for f in os.listdir(FAKE_FOLDER)
                 if f.lower().endswith('.mp4')])
remaining = [v for v in videos if v not in done_videos]
print(f"\nCeleb-DF v2 fake videos: {len(videos)} total | {len(remaining)} remaining")

# Process videos
new_features = []
new_labels   = []
new_vids     = []
new_sources  = []
new_gens     = []
failed = 0

for vf in tqdm(remaining, desc="CelebDF Fake"):
    path  = os.path.join(FAKE_FOLDER, vf)
    feats = extract_features_from_video(path, n_frames=N_FRAMES)
    if not feats:
        failed += 1
        continue
    for i, f_vec in enumerate(feats):
        new_features.append(np.array(f_vec, dtype=np.float32))
        new_labels.append(1)   # fake
        new_vids.append(f"celebdf_fake_{vf}_{i}")
        new_sources.append('celebdf_fake')
        new_gens.append('celebdf_synthesis')
    done_videos.add(vf)

    # Save checkpoint every 200 videos
    if len(done_videos) % 200 == 0:
        with open(CHECKPOINT, 'wb') as f:
            pickle.dump({'done_videos': list(done_videos)}, f)

print(f"\nProcessed: {len(done_videos)} ok | {failed} failed")
print(f"New samples added: {len(new_features)}")

# Merge
features  += new_features
labels    += new_labels
video_ids += new_vids
sources   += new_sources
gen_types += new_gens

# Save merged pkl
merged = {
    'features':         np.array(features, dtype=np.float32),
    'labels':           np.array(labels,   dtype=np.float32),
    'video_ids':        video_ids,
    'dataset_sources':  sources,
    'generator_types':  gen_types,
}

print(f"\nSaving merged pkl: {OUTPUT_PKL}")
with open(OUTPUT_PKL, 'wb') as f:
    pickle.dump(merged, f)

total = len(labels)
real  = sum(1 for l in labels if l == 0)
fake  = sum(1 for l in labels if l == 1)
src_u = set(sources)

print(f"\n{'='*60}")
print(f"  MERGE COMPLETE")
print(f"{'='*60}")
print(f"  Total samples : {total}")
print(f"  Real          : {real}")
print(f"  Fake          : {fake}")
print(f"  Sources       : {sorted(src_u)}")
print(f"  Saved to      : {OUTPUT_PKL}")
print(f"{'='*60}")

# Clean up checkpoint
if os.path.exists(CHECKPOINT):
    os.remove(CHECKPOINT)
    print("  Checkpoint cleaned up")