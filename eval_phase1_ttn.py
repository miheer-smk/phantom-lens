"""
Phantom Lens V3 — Test-Time Normalisation Evaluator
Phase 1: Normalise each test video by its own frame statistics
No retraining needed — works with existing checkpoint

Key change from previous evaluator:
  Before: normalise using TRAINING set mean/std
  Now:    normalise each video using ITS OWN frame statistics
  Why:    removes absolute codec baseline, keeps relative deviation signal

Also: only uses strong pillars (drops P3, P5, P8, P10, P12)

Author: Miheer Satish Kulkarni, IIIT Nagpur
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import (roc_auc_score, accuracy_score, f1_score,
                              confusion_matrix, roc_curve)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from precompute_features_v3 import extract_features_from_video
from train_v3 import PillarAttentionClassifier, PILLAR_GROUPS

# ── CONFIG ────────────────────────────────────────────────────────────────────
CHECKPOINT  = "checkpoints/best_model_v3_AUC9177.pt"
REAL_FOLDER = "data/celebdf/real"
FAKE_FOLDER = "data/celebdf/fake"
OUTPUT_DIR  = "results/phase1_ttn"
N_FRAMES    = 16
MAX_VIDEOS  = 500
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Strong pillars only — drop P3(5,6), P5(10,11), P8(17,18), P10(20,21,22), P12(26,27,28)
# From 29 live dims keep:
# P1(0,1,2) P2(3,4) P4(7,8,9) P6(12,13,14) P7(15,16) P9(19) P11(23,24,25)
STRONG_DIMS = [0,1,2, 3,4, 7,8,9, 12,13,14, 15,16, 19, 23,24,25]
DEAD_DIMS   = [20]
LIVE_DIMS   = [i for i in range(30) if i not in DEAD_DIMS]

print(f"Device: {DEVICE}")
print(f"Strong dims: {len(STRONG_DIMS)} out of 29 live dims")
print(f"Dropped weak pillars: P3 Bayer, P5 Specular, P8 Blur, P10 Chromatic, P12 Illum")


# ── MODEL ─────────────────────────────────────────────────────────────────────
def load_model():
    ckpt  = torch.load(CHECKPOINT, map_location=DEVICE)
    model = PillarAttentionClassifier(
        pillar_groups=ckpt.get('pillar_groups', PILLAR_GROUPS)
    ).to(DEVICE)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    sm = ckpt['scaler_mean']
    ss = ckpt['scaler_std']
    print(f"Model loaded | Training AUC: {ckpt.get('auc',0):.4f}")
    return model, sm, ss


# ── TEST-TIME NORMALISATION ───────────────────────────────────────────────────
def predict_with_ttn(video_path, model, train_sm, train_ss):
    """
    Test-time normalisation:
    1. Extract all frame features
    2. Normalise by THIS VIDEO's own mean/std (not training stats)
    3. Aggregate normalised frames
    4. Apply training scaler (now on normalised features)
    5. Predict
    """
    try:
        feats = extract_features_from_video(video_path, n_frames=N_FRAMES)
        if len(feats) < 2:
            return None

        feat_arr = np.array(feats, dtype=np.float32)  # (n_frames, 30)

        # Remove dead dim
        feat_arr = feat_arr[:, LIVE_DIMS]              # (n_frames, 29)

        # Keep only strong dims
        feat_strong = feat_arr[:, STRONG_DIMS]         # (n_frames, 16)

        # TEST-TIME NORMALISATION — use this video's own statistics
        vid_mean = feat_strong.mean(axis=0, keepdims=True)
        vid_std  = feat_strong.std(axis=0,  keepdims=True) + 1e-8
        feat_ttn = (feat_strong - vid_mean) / vid_std  # normalised per video

        # Aggregate frames → single video vector
        feat_video = feat_ttn.mean(axis=0, keepdims=True)  # (1, 16)

        # Apply training scaler on strong dims only
        sm_strong = train_sm[STRONG_DIMS]
        ss_strong = train_ss[STRONG_DIMS] + 1e-8
        feat_sc   = (feat_video - sm_strong) / ss_strong

        # Pad back to 29 dims for model (zero-fill dropped dims)
        feat_full = np.zeros((1, 29), dtype=np.float32)
        feat_full[0, STRONG_DIMS] = feat_sc[0]

        with torch.no_grad():
            prob = float(model(torch.FloatTensor(feat_full).to(DEVICE)).cpu().numpy()[0])
        return prob

    except Exception as e:
        return None


def predict_original(video_path, model, train_sm, train_ss):
    """Original method — use training stats for normalisation"""
    try:
        feats = extract_features_from_video(video_path, n_frames=N_FRAMES)
        if not feats:
            return None
        feat_arr  = np.array(feats, dtype=np.float32).mean(0, keepdims=True)
        feat_live = feat_arr[:, LIVE_DIMS]
        feat_sc   = (feat_live - train_sm) / (train_ss + 1e-8)
        with torch.no_grad():
            prob = float(model(torch.FloatTensor(feat_sc).to(DEVICE)).cpu().numpy()[0])
        return prob
    except:
        return None


# ── EVALUATE ──────────────────────────────────────────────────────────────────
def evaluate(folder, label, model, sm, ss, desc, use_ttn=True):
    videos  = sorted([f for f in os.listdir(folder)
                      if f.lower().endswith('.mp4')])[:MAX_VIDEOS]
    results = []
    failed  = 0
    fn      = predict_with_ttn if use_ttn else predict_original
    for vf in tqdm(videos, desc=desc):
        p = fn(os.path.join(folder, vf), model, sm, ss)
        if p is None:
            failed += 1
            continue
        results.append((p, label))
    lbl = 'REAL' if label==0 else 'FAKE'
    print(f"  {lbl}: {len(results)} processed | {failed} failed")
    return results


def compute_metrics(results, name):
    probs  = np.array([r[0] for r in results])
    labels = np.array([r[1] for r in results])
    auc    = roc_auc_score(labels, probs)
    preds  = (probs >= 0.5).astype(int)
    acc    = accuracy_score(labels, preds)
    f1     = f1_score(labels, preds, zero_division=0)
    cm     = confusion_matrix(labels, preds)
    tn, fp, fn_v, tp = cm.ravel()
    tpr = tp / (tp + fn_v + 1e-8)
    tnr = tn / (tn + fp + 1e-8)
    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")
    print(f"  AUC      : {auc:.4f}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1       : {f1:.4f}")
    print(f"  TPR fake : {tpr:.4f}")
    print(f"  TNR real : {tnr:.4f}")
    print(f"  CM: TN={tn} FP={fp} FN={fn_v} TP={tp}")
    return auc, probs, labels


def main():
    print("\n" + "="*55)
    print("  Phantom Lens — Phase 1: Test-Time Normalisation")
    print("  Celeb-DF v2 Zero-Shot Evaluation")
    print("="*55 + "\n")

    model, sm, ss = load_model()

    # Method 1 — Original (training stats normalisation)
    print("\nMethod 1: Original normalisation (training stats)")
    r_orig = evaluate(REAL_FOLDER, 0, model, sm, ss, "Real (orig)", use_ttn=False)
    f_orig = evaluate(FAKE_FOLDER, 1, model, sm, ss, "Fake (orig)", use_ttn=False)
    auc_orig, _, _ = compute_metrics(r_orig + f_orig, "ORIGINAL METHOD")

    # Method 2 — Test-time normalisation
    print("\nMethod 2: Test-time normalisation (per-video stats)")
    r_ttn = evaluate(REAL_FOLDER, 0, model, sm, ss, "Real (TTN)", use_ttn=True)
    f_ttn = evaluate(FAKE_FOLDER, 1, model, sm, ss, "Fake (TTN)", use_ttn=True)
    auc_ttn, probs_ttn, labels_ttn = compute_metrics(r_ttn + f_ttn, "TEST-TIME NORMALISATION")

    # Comparison
    print(f"\n{'='*55}")
    print(f"  PHASE 1 RESULTS SUMMARY")
    print(f"{'='*55}")
    print(f"  V1 baseline (original)  : 0.4923")
    print(f"  V3 best so far          : 0.5551")
    print(f"  Original method (now)   : {auc_orig:.4f}")
    print(f"  Test-time norm (Phase 1): {auc_ttn:.4f}")
    gain = auc_ttn - 0.5551
    print(f"  Gain over best V3       : {gain:+.4f}")
    if auc_ttn > 0.65:
        print(f"  STATUS: GOOD — proceed to Phase 2 (rPPG)")
    elif auc_ttn > 0.58:
        print(f"  STATUS: MODERATE — TTN helps, continue Phase 2")
    else:
        print(f"  STATUS: MINIMAL GAIN — deeper problem, reconsider")
    print(f"{'='*55}")

    # Save ROC plot
    fpr, tpr_curve, _ = roc_curve(labels_ttn, probs_ttn)
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor('#F8F9FA')
    ax.set_facecolor('white')
    ax.plot([0,1],[0,1],'k--',alpha=0.4,lw=1.2,label='Random (0.50)')
    ax.axhline(0.5551, color='#A32D2D', ls=':', lw=1.5,
               label=f'V3 best so far (0.5551)')
    ax.plot(fpr, tpr_curve, color='#185FA5', lw=2.5,
            label=f'Phase 1 TTN ({auc_ttn:.4f})')
    ax.fill_between(fpr, tpr_curve, alpha=0.08, color='#185FA5')
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title(f'Phase 1: Test-Time Normalisation\nCeleb-DF v2 Cross-Dataset AUC={auc_ttn:.4f}',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.spines[['top','right']].set_visible(False)
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'phase1_roc.png')
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#F8F9FA')
    plt.close()
    print(f"\n  ROC plot saved: {out}")

    # Save results
    with open(os.path.join(OUTPUT_DIR, 'phase1_results.txt'), 'w') as f:
        f.write("PHASE 1 RESULTS\n")
        f.write("="*40 + "\n")
        f.write(f"V1 baseline       : 0.4923\n")
        f.write(f"V3 best so far    : 0.5551\n")
        f.write(f"Original method   : {auc_orig:.4f}\n")
        f.write(f"Test-time norm    : {auc_ttn:.4f}\n")
        f.write(f"Gain              : {auc_ttn-0.5551:+.4f}\n")
    print(f"  Results saved: {OUTPUT_DIR}/phase1_results.txt")


if __name__ == "__main__":
    main()