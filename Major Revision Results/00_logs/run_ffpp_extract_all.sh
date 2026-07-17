#!/bin/bash
# Resume-safe FF++ extraction: per-file .done sentinels. A kill mid-file leaves no .done,
# so the file is cleanly re-done from scratch next run (no append-dup, no partial kept).
set -u
cd /home/iiitn/Downloads/phantom-lens-main
source .venv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
LOG="Major Revision Results/00_logs/ffpp_extract_all.log"
ROOT=/home/iiitn/Datasets/FaceForensics++
MF=300; W=12
ts(){ date -u +%H:%M:%SZ; }
run(){ # $1=video_dir $2=label $3=outcsv
  local done="${3}.done"
  if [ -f "$done" ]; then echo "[$(ts)] SKIP(done) $3 rows=$(($(wc -l < "$3" 2>/dev/null||echo 1)-1))" >> "$LOG"; return; fi
  rm -f "$3"
  echo "[$(ts)] extracting $3 (label $2) ..." >> "$LOG"
  if python src/precompute_features_best.py --video_dir "$1" --output "$3" --label "$2" --max_frames $MF --workers $W >> "$LOG" 2>&1; then
    touch "$done"; echo "[$(ts)] DONE $3 rows=$(($(wc -l < "$3")-1))" >> "$LOG"
  else
    echo "[$(ts)] FAILED $3 (no .done written; will retry next run)" >> "$LOG"
  fi
}
echo "[$(ts)] === FF++ extract-all START (resume-safe) mf=$MF w=$W ===" >> "$LOG"
# c23
run "$ROOT/original_sequences/youtube/c23/videos"           0 features/ffpp_original_c23.csv
run "$ROOT/manipulated_sequences/Deepfakes/c23/videos"      1 features/ffpp_deepfakes_c23.csv
run "$ROOT/manipulated_sequences/Face2Face/c23/videos"      1 features/ffpp_face2face_c23.csv
run "$ROOT/manipulated_sequences/FaceSwap/c23/videos"       1 features/ffpp_faceswap_c23.csv
run "$ROOT/manipulated_sequences/NeuralTextures/c23/videos" 1 features/ffpp_neuraltextures_c23.csv
# c40
run "$ROOT/original_sequences/youtube/c40/videos"           0 features/ffpp_original_c40.csv
run "$ROOT/manipulated_sequences/Deepfakes/c40/videos"      1 features/ffpp_deepfakes_c40.csv
run "$ROOT/manipulated_sequences/Face2Face/c40/videos"      1 features/ffpp_face2face_c40.csv
run "$ROOT/manipulated_sequences/FaceSwap/c40/videos"       1 features/ffpp_faceswap_c40.csv
run "$ROOT/manipulated_sequences/NeuralTextures/c40/videos" 1 features/ffpp_neuraltextures_c40.csv
echo "[$(ts)] === FF++ extract-all COMPLETE ===" >> "$LOG"
