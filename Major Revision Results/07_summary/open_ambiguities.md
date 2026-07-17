# Open Ambiguities & Stop-and-Ask Items (Rule 10)

Running list. Each item halts the affected work until Miheer resolves it. Do NOT guess past
any of these. Status: 🔴 open / 🟢 resolved.

---

## A4 — 🔴 MASTER BLOCKER: Where are the data, CSVs, and model?

The examined repo copy (`Downloads/phantom-lens-main`, from the zip) is **code-only**. Absent:
extracted 50-feature CSVs (`features/`, gitignored), trained model (`checkpoints/`, `*.pkl`,
gitignored), all raw video (`data/`, gitignored). A full-machine search found none of them.
**Nothing can be computed until these are supplied.** Options for Miheer:
1. Point me to the real working directory / drive / server where the feature CSVs + trained
   LightGBM model live (fastest path — unblocks Phase 1 immediately).
2. Supply the raw FF++ (c23 + c40) and CelebDF-v2 videos so features can be re-extracted
   (slow; also needs the extraction env; risks drift vs original numbers — Rule 6).
3. Supply a subset (e.g. just the CSVs + `best_train.pkl` + `celebdf_eval.pkl`) to unblock
   the CSV-only Phase-1 experiments while videos are arranged for Phase 2.

## A2 — 🟢 RESOLVED via document_pdf.pdf: canonical = 0.6989

`Downloads/document_pdf.pdf` (PRISM V3 results report) is the SOURCE of the reviewer doc's
numbers. Canonical CelebDF: **AUC 0.6989**, AP 0.9243, Acc 0.8124, MCC 0.2537, F1(fake)0.8901,
F1(macro)0.6252, Recall(fake)0.8745, Recall(real)0.4020, confusion TN324/FP482/FN668/TP4655
(806 real+5323 fake). GATE TARGETS THESE. The repo's `exp_celebdf/results.json` (0.6867,
TN242/FP564/FN413/TP4910) is a SUPERSEDED earlier run — same script `run_celebdf_eval.py`,
different result → the 0.6989 pipeline used different CSVs/config and its artifacts are NOT in
this repo copy (not in corrected_*/ either). So the exact 0.6989 pipeline isn't fully
recoverable from disk; gate aims for 0.6989 but may land in 0.687–0.699 band. In-dist 0.9939
is the more solidly reproducible anchor.

## A10 — 🔴 CV fold count: PDF says "5-fold", code/reviewer-doc say 10-fold

document_pdf.pdf p2 labels the ablation "FF++ c23, 5-fold CV"; but `run_exp3.py` uses
StratifiedKFold(n_splits=10) and the reviewer doc (Exp 2) says "ten folds". CV AUC column
(0.9230 for All-50) matches the 10-fold code artifact. Likely a PDF labeling error. Reconcile
before Exp 2. Also: PDF p3 compression (old exp2) used a SEPARATE protocol — Deepfakes-only,
balanced 950 real/950 fake, 80/20 split, max_frames=120 — NOT the official split; c40 was
extracted for Deepfakes ONLY. New Exp 3 must do all 4 manip c23/c40 (reviewer protocol);
can cross-check DF intra c23=0.9841 / c40=0.9505 against PDF p3.

## A1 — 🔴 Experiment 1 feature-group granularity

The reviewer example table names 5 coarse groups (Noise physics / PRNU-inspired residual /
rPPG / Landmark geometry / Blink dynamics), covering only 19 of 50 features. The code defines
20 fine pillars covering all 50 (see `00_logs/feature_group_mapping.md`). Decision needed:
- (a) Ablate at the 20-pillar granularity (most faithful to code), or
- (b) Ablate at a coarse ~5–7 group scheme matching the reviewers' framing, or
- (c) Both (recommended — coarse for the response letter, fine for the appendix).
Also: does "PRNU-inspired residual" include the temporal PRNU pillar T3, or spatial P2 only?

## A3 — 🔴 c40 compression features (Exp 3)

Manuscript/README state only c23 was tested; no c40 features exist here. Does Miheer have
c40 feature CSVs for DeepFakes/Face2Face/FaceSwap/NeuralTextures somewhere? If not, Exp 3's
c40 arm is a new video re-extraction (Phase-1.5, needs c40 videos + extraction env).

## A5 — 🔴 Phase-2 additional dataset (Exp 6) choice + license

Reviewer options: WildDeepfake (recommended), DFDC preview, DeeperForensics-1.0, Celeb-DF++,
FakeAVCeleb. `data_prep/` already has `prepare_wilddeepfake.py`, `prepare_deeperforensics.py`,
`prepare_dffd.py` — so tooling exists. Which dataset, and is its license/access cleared?
No large download will start without explicit go-ahead (per Section 6 / Phase-2 gating).

## A6 — 🔴 Authorize Python environment build

No scientific-Python packages are installed and no project venv exists. Building from
`requirements.txt` is a prerequisite for literally everything. On this aarch64 (ARM64)
machine, wheel availability for mediapipe / lightgbm / opencv-python must be verified.
Authorize me to create a venv and install, and confirm CPU vs GPU requirements file.

## A7 — 🟢 LARGELY RESOLVED: mediapipe version MATCHES original; opencv minor

MAJOR UPDATE: repo `dockerfile` explicitly downgrades the extractor's mediapipe:
`RUN sed -i 's/mediapipe==0.10.21/mediapipe==0.10.18/' requirements_gpu.txt`
→ the `0.10.21` pin in requirements_gpu.txt was NEVER used; the ORIGINAL extraction ran on
**mediapipe 0.10.18** — the EXACT version now installed. So the primary feature-drift risk is
GONE (same landmark model). Remaining minor variable: opencv — I have 4.11.0 (headless); the
original ran inside `nvcr.io/nvidia/pytorch:24.04` whose bundled opencv is ~4.7. opencv version
affects only: (a) the noise-physics circle-mask (already fixed to be numerically identical),
(b) Farneback optical flow & DCT (numerically stable across versions). Expected impact
negligible. Also bash-history shows a CPU env with unpinned `opencv-python-headless`. The
GATE confirms; if any material mismatch remains it's most plausibly opencv or the ~1%
face-detection survival difference (my CelebDF real=798 vs original 806), not mediapipe.

---

## A9 — 🟡 FF++ split: official 720/140/140 fetched & validated; gate will confirm

Per Miheer's "no guessing — use official split" instruction: fetched official FF++ splits
(ondyari/FaceForensics master) → `01_splits/ffpp_official/{train,val,test}.json`. Validated:
train 720 / val 140 / test 140 unique original IDs, total 1000, all pairwise-disjoint (no
leakage). Routing plan: originals by ID; manipulated (source_target) by pair's split. Also
yields the FF++ VAL split Exp 4 needs. STILL TO CONFIRM EMPIRICALLY AT THE GATE: that this
split + max_frames=300 reproduces All-50 FF++ AUC 0.9939 (±tol). If not → report discrepancy,
do not ship. Original CSVs referenced only real_train/real_test (no val) — the original may
have folded val into train; gate will clarify.

Classifier config CONFIRMED: LGBM(class_weight="balanced", random_state=42, verbose=-1);
10-fold CV = StratifiedKFold(n_splits=10, shuffle=True, random_state=42); bootstrap seed=42.

## A8 — 🟢 RESOLVED: max_frames per dataset (extraction reproduction parameter)

Miheer suggested 478; cross-check shows **478 = MediaPipe Face Mesh landmark count**
(`src/precompute_features_best.py:138` "Extract 478 landmarks", `refine_landmarks=True` →
468+10 iris), NOT a frame budget. The paper cites 478 *landmarks*. Actual max_frames used
by the ORIGINAL runs (verified in code):
  - **FF++ training features (the 0.9939 model): max_frames = 300**
    (`results/face2face/generate_report.py:218` protocol table "max_frames=300"; corroborated
    by `run_celebdf_eval.py:29` comment "300→150 halves extraction time"). exp1 & exp3 both
    load the same pre-extracted `features/ffpp_*.csv` → extracted at 300.
  - **CelebDF-v2 cross-dataset test: max_frames = 150** (`run_celebdf_eval.py:29`).
  - Old compression exp (exp2): max_frames = 120 (`run_exp2.py:111`) — ⚠️ carry into new
    Experiment 3: decide whether to reproduce Exp 3 at 120 or standardize to 300/c-level.
DECISION LOCKED: extract CelebDF @150 (running), FF++ @300. Original FF++ CSV set needed:
ffpp_real_train.csv, ffpp_real_test.csv, ffpp_fake.csv (Deepfakes), ffpp_face2face.csv,
ffpp_faceswap.csv, ffpp_neuraltextures.csv (+ ffpp_faceshifter.csv only for old exp3, skipped).

### Resolved
- **A4 (data location)** — 🟢 CSVs confirmed deleted (machine-wide content search). Path is
  now: download FF++ (c23+c40) + CelebDF-v2 → re-extract features → rebuild CSVs. FF++ c23+c40
  download IN PROGRESS (2026-07-15).
- **A6 (env build)** — 🟢 Core analysis + extraction env built & import-verified in `.venv`
  (see environment.txt). torch/Xception deferred to Phase 2.
- **c0 vs c23/c40 scope** — 🟢 User confirmed c23+c40 (raw/c0 dropped: not needed, disk-infeasible).
