# DEVIATIONS.md — where the R2 run differs from the manuscript's description

Every place what was actually run differs from what the manuscript describes, and why.
A deviation is not a defect: defects are contradictions (see `DEFECTS.md`), deviations are documented
differences of procedure.

---

## DEV-001 — Environment cloned from the original venv rather than rebuilt from `requirements.txt`
**Phase:** 0.3 · **Date:** 2026-08-27

**Manuscript/plan says:** build `env/` with the pinned versions in §2, from scratch.

**What was done:** the interpreter that *produced the published numbers* was found intact at
`~/Downloads/phantom-lens-main/.venv`. It was cloned byte-for-byte to `env/prism` (`rsync -a`,
`pyvenv.cfg` and script shebangs repointed) instead of pip-installing afresh.

**Why:** a fresh `pip install` on aarch64 may resolve to differently-built wheels for the same version
string (LightGBM in particular compiles per-platform). Cloning removes that risk entirely and gives
strictly higher reproduction fidelity, which is the whole point of the pin.

**Verification:** `env/frozen_requirements.txt` (88 packages) is a **byte-identical `diff`** against
`legacy/phantomlens/requirements_snapshot.txt`. Every §2 pinned version verified at import:
Python 3.12.3 · NumPy 1.26.4 · SciPy 1.15.3 · scikit-learn 1.7.2 · LightGBM 4.6.0 · OpenCV 4.11.0 ·
MediaPipe 0.10.18 · PyTorch 2.11.0+cu128 (`cuda.is_available() == True`) · timm 1.0.28.
No version substitutions were required; the "stop and report" clause was not triggered.

**Residual risk:** none identified. Phase 12 still ships a `requirements.txt`/`environment.yml` for
third parties, whose build may differ — that is inherent and will be stated in the release README.

---

## DEV-002 — `.venv` excluded from the archived legacy copies
**Phase:** 0.1 · **Date:** 2026-08-27

The 7.9 GB `.venv` inside `phantom-lens-main(revised).zip` was not extracted into `legacy/`, and
`data_xception/` was excluded from the *zip* copy only (it is present in full in the checkout copy).
These are build artifacts, not research artifacts, and are captured instead by DEV-001 and by
`env/frozen_requirements.txt`. Inventory counts in `results/P0_inventory.json` are therefore of
research artifacts, 91,411 files.

---

## DEV-003 — Reproduction-gate tolerance interpreted against published *precision*
**Phase:** 0.4 · **Date:** 2026-08-27

**Plan says:** delta < 1e-6 from saved score files → PASS exact; < 5e-4 from re-fitting → PASS with a note.

**Issue:** the manuscript reports AUCs rounded to **4 decimal places**, so an exact reproduction can
differ from the printed value by up to 5e-5 purely from rounding. Every observed delta fell in
[1.9e-6, 4.6e-5] — i.e. all are strictly larger than the 1e-6 "exact" threshold, yet all round to the
published 4-dp value **exactly**.

**What was done:** a `PASS_ROUNDING` verdict was added, defined as `round(observed,4) ==
round(published,4)`. It is recorded per-check as `matches_at_published_4dp` in
`results/P0_reproduction_gate.json`, alongside the raw delta, so no information is hidden.

**Why this is not a weakening of the gate:** the 1e-6 criterion is only meaningful against a full-precision
reference, which the manuscript does not provide. Where full precision *was* available — the Table 18
model sizes — the reproduction is genuinely bit-exact (`PASS_EXACT`, delta 0.0 on all four configs).
Note also that the "saved scores" and "refit from features" paths agree with each other to ~1e-9,
confirming the pipeline, not just the stored outputs.

### Amendment (GATE 0, 2026-08-27) — the criterion, stated formally

The two-tier rule replacing the original 1e-6 / 5e-4 bars, now asserted in
`results/P0_reproduction_gate.json → interpretation_rules`:

1. **Against saved score files: EXACT match required.** Re-deriving a statistic from the very same
   per-video probabilities that produced it must be numerically exact. Any nonzero delta there would
   indicate a different statistic, population or estimator — never rounding.
2. **Against manuscript text: agreement within print precision required.**

**The print-precision argument.** The manuscript prints AUCs to 4 decimal places. A value printed as
`x` therefore has a true value in `[x − 5e-5, x + 5e-5)`, so **5e-5 is the maximum deviation
attributable to print precision alone**. The largest delta observed anywhere in the gate is
**4.580e-5** (`table8_lomo_face2face_auc`), which is **strictly below** that bound. Every observed
delta is consequently explained in full by the manuscript's own rounding, and **no protocol drift is
implied by any of them**. No delta requires a further explanation.

Any delta at or above 5e-5 is *not* explicable by print precision and is reclassified **FAIL** — the
amendment applies this rule programmatically; it reclassified nothing, because none reached the bound.

**Two independent corroborations** that this is print precision and not a tolerance being stretched to
fit: (a) the saved-score path and the refit-from-features path are both full precision and agree with
*each other* to ~1e-9; (b) where the artifacts do contain a full-precision reference — the Table 18
model sizes — reproduction is bit-exact at delta 0.0 on all four configurations.

---

## DEV-004 — Table 18 model size re-derived with the profiling recipe, not the evaluation recipe
**Phase:** 0.4 · **Date:** 2026-08-27

The published 611.2 KB comes from `exp5_runtime.py`, which trains its profiling model on the
**train+val** partitions with whole-frame median imputation and `n_jobs=1`. The *evaluation* model
(`baseline_clean.py`) trains on **train only** with the M1 train-only imputer, and serialises to
601.6 KB (native) / 606.2 KB (joblib bundle).

Both were computed and both are recorded. The Table 18 check is scored against the profiling recipe,
which is the one that produced the published figure; it reproduces bit-exactly (611.2 KB, delta 0.0),
as do the other three configs (501.9 / 585.0 / 590.1 KB).

**Manuscript implication:** Table 18 should state that the profiled model is fit on train+val. Two
distinct models are currently described by one number.

---

## DEV-005 — GPU baseline track runs in a second venv (`env/dfb`), not the frozen `env/prism`
**Phase:** 7.1 · **Date:** 2026-08-27

**Plan implies:** one pinned environment for the whole revision.

**What was done:** DeepfakeBench's dependency closure is incompatible with the PRISM pin — it pulls
`numpy>=2` transitively, which breaks MediaPipe 0.10.18 and therefore the entire feature extractor.
`env/dfb` was cloned from `env/prism` and extended with 17 additional packages; NumPy was then pinned
back to 1.26.4 inside it, because `np.sctypes` (removed in NumPy 2.0) breaks DeepfakeBench's own
import chain.

**Why this is safe:** the two tracks share no code path. PRISM feature extraction, classifier fitting,
bootstrap, SHAP and domain-shift analysis all run in `env/prism`; only the deep baselines run in
`env/dfb`. Their only interface is a per-video score CSV.

**Verification:** `env/prism` was re-checked after every install in `env/dfb` and remains a
**byte-identical `pip freeze` diff** against `env/frozen_requirements.txt`. `env/dfb` is frozen to
`env/frozen_requirements_dfb.txt` (135 packages). Both ship in the Phase 12 release.

---

## DEV-006 — LAA-Net is not available through DeepfakeBench
**Phase:** 7.1 · **Date:** 2026-08-27

The brief states DeepfakeBench "already supports LSDA, VideoMAE, FTCN, Face X-ray and others, which
cuts implementation cost substantially". Verified true for LSDA, VideoMAE and FTCN — all three pass a
CUDA forward pass. **LAA-Net is not in the registry** (35 detectors enumerated in
`results/P7_1_dfb_probe.json`; no `laanet`).

Phase 7.3 must therefore build LAA-Net from its official repository, whose aarch64/CUDA-12.8 build is
unverified. Against the pre-agreed cut order for optional baselines (VideoMAE first, then
LAA-Net), this is a ranked, accepted risk rather than a blocker.

---

## DEV-007 — Table 5 (standalone rPPG) must be reproduced with train+val fitting
**Phase:** 8 (recorded at Phase 1.3) · **Date:** 2026-08-27

Table 5's published values come from `exp9_analyze.py`, which fits on **train + val** (n_test = 1337),
not train-only. Phase 8 must match that protocol or it will report a spurious reproduction failure
against POS 0.5178 / POS+CHROM 0.4782 / CHROM 0.4663. See `results/P1_defect001.md` §3 for the full
list of six train+val experiments. If the author selects Option B for DEFECT-001, Table 5 is
regenerated train-only and these reference values change by design.

---

## DEV-008 — LSDA evaluates Celeb-DF on the FULL release, not the official 518-video test list
**Phase:** 7.6 / item H · **Date:** 2026-08-28

**DeepfakeBench's convention:** `rearrange.py` reads `List_of_testing_videos.txt` and assigns exactly
those 518 videos to the `test` split; everything else becomes `train`.

**Why that could not be used.** PRISM's published zero-shot result is over the **entire Celeb-DF v2
release** (6121 of 6529 after extraction gates), not the official test split — see DEFECT-004. If
LSDA were evaluated on the 518-video split, its Celeb-DF row would share **no common population**
with PRISM's, and the R1-C2 comparison table would be comparing different corpora in the same column.

**What was done.** A full-coverage `List_of_testing_videos.txt` was generated for the DeepfakeBench
mirror, listing all 6529 videos so every one lands in `test`. The **official 518-video list is
preserved alongside** as `List_of_testing_videos.OFFICIAL.txt`, unmodified, and the real dataset
directory is untouched (the mirror is symlinks).

**Consequence, which must be stated in the manuscript.** LSDA's Celeb-DF figure here is **not
comparable to published Celeb-DF benchmark numbers**, which conventionally use the 518-video test
split. It is constructed to be comparable to *our* PRISM and Xception rows and to nothing else. Any
cross-paper comparison of the LSDA Celeb-DF value would be invalid.

**Verification:** the generated JSON yields test = 890 real + 5639 fake = 6529, matching the release
exactly. Intersection with PRISM's 6121 is computed in `results/PH_common_population.json`.

---

## DEV-009 — released feature matrices carried absolute home paths; now dataset-relative

**Opened:** 2026-08-28 (Phase 12 pre-check) · **Applied:** same day

**Found.** Every shipped feature matrix stored `video_path` as an absolute path on the authors'
machine — `<HOME>/Datasets/...` — **27,095 rows across 17 CSVs**, plus **3,688** `file_path`
values in `df40_prism50.jsonl` carrying `<HOME>/prism_r2/data/`. The archive is destined for a
DOI-assigning public repository. The earlier repo sanitisation predated the inclusion of
`features/`, so these were never scanned.

**Change.** The path prefix is removed; paths are now dataset-relative. Done as a **textual
substitution**, not a pandas round-trip.

**Why that distinction matters — a real trap, caught by a control.** The first attempt rewrote the
CSVs through `pandas.read_csv`/`to_csv`. The reproduction gate then returned **7/9**, and the
obvious reading was that the path change had altered results. It had not. A control that
round-tripped the files through pandas *without touching any path* also returned 7/9: the cause was
**float reformatting on write**, not the path. The byte-preserving substitution returns **9/9**,
identical to the untouched archive.

Had the control been skipped, the conclusion would have been the exact opposite of the truth.

**Verification.** All ten runners execute against the sanitised matrices; the reproduction gate is
9/9; the calibration threshold (0.510), McNemar χ² (25.4697, p = 4.4939e-07) and the DF40 macro
(0.7717, sd 0.1491) are unchanged. A full scan of the archive — text and binary — returns no
`/home/`, no `/Users/`, no `C:\Users` and no email address.

**Documented consequence.** `_video_seed` hashes the path *string*, so the per-video RNG seed was
never portable: a re-extraction from any other location already produced different sampler draws.
Relativising the stored path does not create this, but it makes it visible, and the README now
states plainly that re-extraction will not match the shipped matrices and that the shipped matrices
are what reproduce the published tables. Related: DEFECT-008.

**Not done, and flagged instead.** The working repository's own git history still contains these
absolute paths in earlier commits. **No history has been rewritten.** If the release is a snapshot
of `repo/` — an archive or a fresh repository — the history never ships and nothing further is
needed. If the working history is ever published, it must be scrubbed first, and that is a decision
to take deliberately rather than as a side effect.

---

## DEV-010 — verification-script defect: wrong normalisation, silently accepted
**Opened:** 2026-08-29 (D2) · **Status:** FIXED same day · **No published value affected**

> **Numbering note.** The ruling designated this DEV-009. That number was already taken earlier the
> same night by the absolute-paths entry, so it is recorded here as **DEV-010**. Flagged rather than
> renumbered, because silently reusing an identifier is how two different findings become one.

`scripts/pD2_xception_rerun.py` hard-coded normalisation constants `[0.5,0.5,0.5]`. The Xception
checkpoint was trained with ImageNet constants `[0.485,0.456,0.406] / [0.229,0.224,0.225]`,
documented at `legacy/phantomlens/Major Revision Results/00_logs/xception_train.py:14` and used
again by `exp_g9_xception_predictions.py:14`.

**The failure was silent.** No exception, no warning. The model accepted inputs it had never been
trained on and returned well-formed probabilities in [0,1], from which a plausible AUC and a
plausible DeLong statistic were computed. Every Table 19 row was depressed by 0.026–0.062. This is
the same class as DEFECT-009: **a wrong result that looks like a result.**

**Fix.** The normalisation is now *read from the training script* rather than restated —
`_training_normalisation()` parses MEAN/STD out of `xception_train.py` and raises if the file is
missing or unparseable rather than falling back to a default. A restated constant can drift from
the checkpoint silently, which is precisely the failure being guarded against.

The same latent bug existed in `pD1_lsda_eval.py`, which used `cfg.get("mean",[0.5,0.5,0.5])` — a
silent substitution had the training config lacked the key. The keys are now required. (The LSDA
config does carry `[0.5,0.5,0.5]` genuinely, so D1's results were never affected.)

**Archive gap, flagged not fixed.** The ruling framed this as a hazard for "a third party re-running
from the archive". They cannot currently hit it: **neither baseline scoring script is in
`repo/`**, so the archive cannot reproduce Table 19 at all. That is a larger gap than the assertion
and is left as a decision — porting the baselines would mean shipping checkpoints and crop
manifests, which is a scope question, not a bug fix.

---

## Verification direction — standing rule
**Adopted:** 2026-08-29, from the DEV-010 near-miss

**When a re-run contradicts a published value, read the original artifacts before drafting any
disclosure. Never the reverse order.**

**The reasoning failure, recorded because it matters more than the bug.** At Step 1 the evidence was
read correctly: five rows, every deviation negative, a systematic pattern indicating one upstream
cause rather than five independent errors. The error was in the next step — assuming that single
cause sat in the *published pipeline* rather than in the *re-run*. Both were consistent with the
evidence; only one was checked.

A verification tool that errs toward finding defects is as dangerous as one that misses them. The
first D2 result was on a path to being written up as a disclosure of an irreproducibility that
does not exist — a fabricated defect, in an externally facing document, four days before a hard
deadline. The read-only artifact inspection that caught it took eleven minutes.

**What made it work:** going in to look for a documented *aggregation*, the first thing the file
showed was that the original aggregation was already a per-crop mean — identical to the re-run.
That refuted the standing hypothesis and forced attention to the next line, where the real
difference was. Looking for the wrong thing in the right place still found it; drafting first would
not have.
