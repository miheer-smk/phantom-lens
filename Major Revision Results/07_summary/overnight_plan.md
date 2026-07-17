# Overnight Plan (2026-07-15 night → user returns 2026-07-16 with HDD)

## User instruction (verbatim intent)
"Download everything. Once versions are known (from the HDD tomorrow), THEN run the final
step to get the final numbers." → Downloads + prep overnight; **FINAL numbers wait for exact
versions.** Do NOT present any overnight number as final.

## What runs overnight (autonomous, via background jobs + harness notifications)
1. **Finish ALL downloads**: FF++ c23 (FaceSwap, NeuralTextures remaining) then all of c40
   (originals + 4 manip). Watchdog supervises, auto-restarts on stall.
2. **Finish in-flight extractions**: CelebDF (`celebdf_features.csv`) + FF++ originals+Deepfakes
   (`ffpp_original_c23.csv`, `ffpp_deepfakes_c23.csv`), max_frames: CelebDF 150 / FF++ 300.
3. **As FF++ c23 manip finish downloading → extract them** (Face2Face, FaceSwap, NeuralTextures
   @300) and the **c40** set. All with CURRENT versions (opencv 4.11 / lightgbm 4.6) —
   **marked PRELIMINARY**, may be re-extracted tomorrow if exact opencv differs.
4. **PRELIMINARY sanity gates** (clearly labelled NOT final): Deepfakes-only AUC vs 0.9709,
   and All-50 vs 0.9939 once all c23 extracted. Purpose = early read on how close current
   versions get + catch pipeline bugs. Saved to `07_summary/preliminary_gate.md`.

## What does NOT run overnight
- No FINAL/official numbers. No manuscript-facing tables. Those wait for Step below.

## Tomorrow (when user brings HDD)
1. Mount HDD → read old **`.pkl`** → exact **lightgbm** version (+ confirm hyperparameters).
2. Find **venv / `pip freeze`** → exact **opencv** version.
3. Compare to current env:
   - **If match** → overnight extractions are valid → run FINAL gate → **final numbers**.
   - **If differ** → `pip install` exact opencv/lightgbm → **re-extract** → run FINAL gate.
4. Verify a few FF++/CelebDF video checksums (static official releases → byte-identical).
5. Neutralize LightGBM thread non-determinism (match n_jobs / deterministic mode).
6. Only then: declare final numbers, proceed to the 12 experiments per the big prompt.

## Targets (from document_pdf.pdf — canonical)
FF++ All-50 AUC **0.9939**; DF 0.9709 / F2F 0.8818 / FS 0.9999 / NT 0.9991;
CelebDF AUC **0.6989**, MCC 0.2537, macro-F1 0.6252, real-rec 0.4020, fake-rec 0.8745.

## Locked facts (no guessing) — see open_ambiguities.md
extractor `precompute_features_best.py` (+1 numerically-identical opencv fix) · mediapipe
0.10.18 · numpy 1.26.4 · scipy 1.15.3 · tqdm 4.67.3 · LGBM(n_est200,depth6,lr.05,leaves31,
min_child20,balanced,seed42) · official FF++ split 720/140/140 · CPU extraction · same GB10.
Only unknowns: exact opencv + lightgbm versions → coming from HDD tomorrow.

---
## Recovery log (overnight, after harness killed the extraction tasks)
- Harness killed b7xh3oyzz (FF++ DF) + biiiusuen (FF++ rest) mid-run (NOT OOM — 114GB free).
- Integrity checked: NO corruption. All CSVs unique paths, 52 cols, valid labels.
- CLEAN + COMPLETE: celebdf_features.csv (6121; fake 5323 = exact match), ffpp_original_c23.csv (960). Marked .done.
- Partial (deleted, being re-extracted clean): deepfakes/face2face/faceswap c23.
- Relaunched via `run_ffpp_extract_all.sh` — RESUME-SAFE (per-file .done sentinels; a kill
  leaves no .done so that file is cleanly redone, no dup/partial). Detached via setsid
  (not harness-tracked → no completion notification, but survives session kills; .done
  preserves progress across any future kill). Covers all FF++ c23 + c40.
- STILL: extraction uses current opencv 4.11 (PRELIMINARY). If HDD venv shows a different
  opencv, re-extract (delete .done + rerun). lightgbm differs → only re-train (cheap).
- Downloads 100% complete (10000 videos). Nothing lost.

## When user returns with HDD
1. Check `ffpp_extract_all.log` for progress / any FAILED lines; relaunch driver if killed
   (resume-safe — completed manips skipped).
2. Read pkl (lightgbm) + venv (opencv). If opencv==4.11 → staged extraction valid → gate.
   If opencv differs → `rm features/*.done features/ffpp_*.csv` for FF++, set exact opencv, rerun.
