# Reproduction Gate Progress

## Feature-level validation (PASSED — decisive)
Re-extracted features vs original ground truth (results/exp5/false_negatives_with_features.csv,
13 Deepfakes c23 videos): **49/50 features machine-identical**, 1 (t_noise_spectral_entropy)
1% drift. => opencv 4.11 extraction reproduces originals exactly. Extraction/opencv NOT a risk.

## Gate run 1 (exp1 per-manip, LightGBM) — MISSED
Used OFFICIAL FF++ split (693/134 reals). Results below target:
  Deepfakes 0.9374 (t0.9709) · Face2Face 0.8031 (t0.8818) · FaceSwap 0.8729 (t0.9996 LGBM) · NeuralTextures 0.7205 (t0.9989 LGBM)
Diagnosis: NOT features/versions (features proven identical). Cause = WRONG SPLIT.

## BUG FOUND (from their own code)
results/exp2/run_exp2.py:176 creates the real split as:
  train_test_split(real_df, test_size=0.2, random_state=42, shuffle=True)
=> random 80/20 seed42, NOT the official JSON split. Corroborated: exp5/results.json
Deepfakes test has 957 fakes -> 0.9709 (matches my 956-row deepfakes extraction).

## Gate run 2 (random 80/20 split, seed 42) — RUNNING
Targets (their exp1/results.json LightGBM): DF 0.9709 · F2F 0.8818 · FS 0.9996 · NT 0.9989.
Result: see gate_exp1_run2.txt.

## Note
exp3 All-50 headline 0.9939 additionally needs FaceShifter (train)+NeuralTextures(test);
FaceShifter not yet downloaded. Will fetch after per-manip gate passes.

## Gate run 2 (random 80/20 seed42 split) — STILL MISSED (split was not the cause)
DF 0.9352 · F2F 0.8045 · FS 0.8714 · NT 0.7219 (vs their LGBM DF .9709/F2F .8818/FS .9996/NT .9989)
Split barely changed anything (NT 0.7205->0.7219). Hypothesis rejected.

## DECISIVE DIAGNOSTIC: training-set 10-fold CV
MY multi-manip LightGBM CV AUC = **0.7685** vs THEIR **0.9212** (exp1/results.json;
corroborated by exp3 All-50 CV 0.9230). This is a training-only metric (no test leakage).
=> Same classifier+params+CV protocol, so MY TRAINING FEATURE MATRIX differs from theirs
   at population level — DESPITE 13 Deepfakes-FN videos matching to machine precision.

## Feature health (mine): CLEAN
0% NaN, 0% inf, 0 all-zero rows across real+4 manip (~960 each). Extraction not broken.

## Working hypothesis (under test)
The 13 machine-identical videos were Deepfakes FAKES. Real-video features never verified.
If my REALS differ from theirs (or specific manips), real-vs-fake separation drops 0.92->0.77.
Per-manip real-vs-fake CV breakdown running to localize (Deepfakes vs F2F vs FS vs NT).

## IMPORTANT (honest status)
Feature extraction reproduces (proven on Deepfakes). But full-pipeline AUC does NOT yet
reproduce — training CV 0.77 vs 0.92. Root cause NOT yet found. NOT a versions issue.
Candidates: (a) real-video feature mismatch, (b) a manip-specific extraction diff,
(c) their published CV used different data (e.g. more samples/frames). Investigating.

## PER-MANIP real-vs-fake CLEAN CV (localizes gap) — KEY RESULT
  Deepfakes 0.980 · FaceSwap 0.957 · Face2Face 0.780 · NeuralTextures 0.733
Pattern: DF & FS separate well; F2F & NT do NOT. Multi-manip CV 0.77 driven by F2F/NT.

## HONEST CONCLUSION (for Miheer)
1. Feature EXTRACTION reproduces: Deepfakes features machine-identical to original (proven).
2. Published in-distribution AUCs do NOT cleanly reproduce:
   - my multi-manip 10-fold CV = 0.77 vs published 0.92 (exp1 & exp3 both record ~0.92)
   - per-manip: DF/FS reproduce well; F2F/NT much lower than published.
3. Published per-manip test protocol is LEAKY (all fake rows appear in BOTH train and the
   per-manip test set). Even reproducing that exact leaky protocol, my NeuralTextures test
   AUC = 0.72 vs their 0.9989 — their model scored all NT fakes ~1.0 (recall 1.0), mine ~0.75.
   With identical LGBM params, identical features would memorize identically → their NT/F2F
   feature values likely differ from mine (I could only ground-truth-verify Deepfakes).
4. Interpretation options (NEEDS MIHEER):
   (a) Published NT/FS ~0.999 were inflated by leaky train=test-fakes + overfit; honest
       reproducible separability is DF .98/FS .96/F2F .78/NT .73 — a DEFENSIBLE, more modest story.
   (b) An undocumented pipeline/data detail differs for F2F/NT extraction.
   Either way: the near-perfect NeuralTextures/FaceSwap numbers are a reviewer red-flag and
   may not survive scrutiny; better discovered now than by a reviewer.

## CelebDF zero-shot GATE (the "clean" cross-dataset number) — ALSO LOWER
My reproduction: AUC 0.6143 vs published 0.6989 (Δ -0.085); AP 0.905 (pub 0.924);
macro-F1 0.550 (0.625); real-rec 0.385 (0.402); fake-rec 0.778 (0.874); MCC 0.128 (0.254).
Repo's own exp_celebdf/results.json = 0.6867; PDF = 0.6989 (their runs already varied ~0.02).
My 0.614 is ~0.07 below even their lower value. Consistent with my multi-manip training
model being weaker (CV 0.77 vs their 0.92), which propagates to zero-shot.

## OVERALL HONEST PICTURE (both leaky AND clean numbers come out lower in my reproduction)
- Deepfakes features: machine-identical (proven). Clean real-vs-DF CV = 0.98 (EXCELLENT).
- But multi-manip training CV 0.77 vs 0.92, and CelebDF 0.61 vs 0.69 — driven by F2F/NT
  features separating much worse in my extraction (0.78/0.73).
- Two forces at play: (1) proven leakage inflating per-manip 0.999s; (2) a residual gap
  where even clean numbers reproduce ~0.07-0.15 low.
- CANNOT fully close (2) without the ORIGINAL model_LightGBM.pkl or original ffpp_*.csv
  feature files — the decisive artifacts. The regenerated F2F/NT features may differ from
  the originals in a way I can't verify (no F2F/NT ground-truth CSV exists in repo).

## "TRY HARDER" — legitimate CelebDF reproduction attempts (all reported, none cherry-picked)
CelebDF zero-shot, published 0.6989:
  A faithful (real_train 80% + 4 manip)  = 0.6143
  B all reals (960) + 4 manip            = 0.6172
  split-seed sweep {0,1,42,123,777}      = 0.597 / 0.633 / 0.614 / 0.615 / 0.618
  => BEST legitimate config = 0.633 (seed 0), still -0.066 below 0.6989. Gap NOT closable
     via honest config choices. My honest CelebDF ≈ 0.61-0.63.

## DECISIVE: OLD 30-feat pkl reveals WHERE the high numbers came from
precomputed_features_v3_with_celebdf.pkl is FRAME-LEVEL (16000 frames/manip). Random 5-fold
CV on frames (frames of same video in train AND test = frame-level leakage) gives:
  real vs deepfakes .9948 · face2face .9984 · faceswap .9972 · neuraltextures .9967  (ALL ~0.99)
=> Under leaky frame-level eval EVERYTHING looks near-perfect. This is the classic deepfake
   pitfall (must split by video, not frame). My VIDEO-LEVEL honest numbers (DF .98/F2F .78/
   FS .96/NT .73) are correct; the published near-perfect numbers trace to leakage
   (file-level in exp1, frame-level in earlier iterations).

## HONEST FINAL: I cannot match 0.9939/0.6989 by legitimate means.
Best honest: in-dist DF 0.98 (clean, strong), F2F 0.78, FS 0.96, NT 0.73; CelebDF ~0.62.
The published targets require reintroducing leakage or seed-selection — neither valid.

## NEW-METHODOLOGY experiments to legitimately boost cross-dataset (target 0.70+)
Tested on real data. Baseline CelebDF 0.617. Results (ALL reported):
  QuantileTransform robust norm      0.609 (worse)
  physics-invariant subset (physio+geom) 0.583 (WORSE — hypothesis refuted)
  invariant + quantile               0.575
  camera/codec features only         0.624 (slightly BEST — opposite of hypothesis!)
  CORAL (UDA)                        0.578 (worse)
  invariant + CORAL                 0.542 (worse)
=> NONE reach 0.70. Quick DA/feature-selection/normalization tricks do NOT help here.
   My physical intuition (physiology/geometry = domain-invariant) was EMPIRICALLY WRONG:
   rPPG/blink/landmark features are noisy & weak cross-dataset; compression features carry
   more signal. Honest cross-dataset ceiling with current features ≈ 0.62-0.63.
CONCLUSION: 0.70+/0.99+ NOT achievable via post-hoc tricks. Would need genuine new
   feature extraction and/or multi-dataset real training — real research, uncertain payoff.

## COMPRESSION AUGMENTATION (same-domain more-data, user's "no domain shift" ask) — PARTIAL
Add c40 versions of same FF++ videos to training (real+DF c40 ready; F2F/FS/NT c40 extracting):
  baseline c23-only              CelebDF AUC 0.617 (real_rec 0.35, fake_rec 0.79)
  + real40                       AUC 0.585 (real_rec 0.74!, fake_rec 0.36)
  + real40 + deepfakes40         AUC 0.605 (real_rec 0.62, fake_rec 0.53)
=> AUC (ranking) NOT improved by partial aug. BUT real-recall jumped 0.35->0.74 (the core
   cross-dataset failure) — it's a threshold/balance shift, not a ranking gain. Partial
   (imbalanced: added compressed reals+DF but not compressed F2F/FS/NT). Full c40 test
   pending — may rebalance. Honest expectation: modest at best on AUC.

## HONEST META (after many legitimate attempts)
Tried: feature-subset selection, quantile norm, CORAL/UDA, compression augmentation (partial).
NONE cleanly improve cross-dataset AUC beyond ~0.62. More same-domain data cannot fix:
  (a) F2F/NT in-distribution ceiling (0.73-0.78) = feature-SEPARABILITY limit, not data quantity;
  (b) cross-dataset gap = domain shift, which same-domain data by definition doesn't bridge.
Keeping methodology + same-domain data alone will NOT reach 0.99+/0.70+. Honest.
