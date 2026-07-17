# Leakage-Free Evaluation Protocol (Methods-ready text)

## Identity-level splitting

FaceForensics++ clips are named `<target>_<source>.mp4` for manipulations and `<id>.mp4`
for pristine videos; the same source identities recur across Deepfakes, Face2Face, FaceSwap,
NeuralTextures and the pristine set. Splitting by clip therefore leaks identities across
folds. We adopt the **official FaceForensics++ split** of the 1000 source sequences into
720 training / 140 validation / 140 test identities (`splits/ffpp_official_split.json`,
seed 42). Every clip is assigned to a partition by its source identity: a pristine clip by
its single id, a manipulated clip by its (target, source) pair, which the official split
guarantees to lie in the same partition. An assertion (`assert_no_identity_overlap`,
`src/protocol.py`) runs at the start of every experiment and fails loudly if any source
identity appears in more than one of {train, val, test}. No fake clip, and no clip derived
from a training identity, ever appears in a test or cross-validation fold.

## Three evaluation regimes

1. **In-distribution (per manipulation).** Train on {pristine + that manipulation} restricted
   to *training* identities; test on {pristine + that manipulation} restricted to *test*
   identities. Reported per manipulation: AUC, macro-F1, precision, recall, MCC, with
   bootstrap 95% CIs (2000 resamples, seed 42).

2. **Cross-manipulation (leave-one-manipulation-out).** Train on {pristine + three
   manipulations} (training identities); test on {pristine + the held-out fourth
   manipulation} (test identities). The held-out manipulation is unseen in both identity and
   manipulation type — the honest generalization measure.

3. **Cross-dataset (zero-shot).** Train on FF++ (pristine + all four manipulations, training
   identities); test on Celeb-DF v2, which is never seen during training, feature design,
   scaling, or model selection.

## Anti-leakage guarantees

- StandardScaler is fit on training rows only and applied to test rows.
- No hyperparameter, feature, or calibration is tuned on FF++ test identities or on Celeb-DF v2.
- The validation partition is used for feature/model selection (Track C); the test partition
  is touched only for the final reported number.
- Every reported number is produced by a deterministic (seed 42), re-runnable script whose
  name, git commit, input-CSV SHA-256, and date are recorded in `LOCKED_NUMBERS.md`.

## Retired (leaky) protocol — not used

The prior per-manipulation protocol placed every manipulation's fake CSV in both the training
set and each per-manipulation test set (`results/exp1/run_exp1.py`, `TRAIN_FILES`), so 100%
of test fakes were seen in training. The resulting near-perfect numbers (FaceSwap 0.9999,
NeuralTextures 0.9991, combined "0.9939") are leakage artifacts and are permanently retired.
