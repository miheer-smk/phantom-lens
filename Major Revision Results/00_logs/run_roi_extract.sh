#!/bin/bash
set -u
cd /home/iiitn/Downloads/phantom-lens-main
source .venv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
LOG="Major Revision Results/00_logs/roi_extract.log"
ROOT=/home/iiitn/Datasets/FaceForensics++
MF=150; W=8
ts(){ date -u +%H:%M:%SZ; }
run(){ # dir label out
  [ -f "$3.done" ] && { echo "[$(ts)] SKIP $3"; return; } >> "$LOG"
  rm -f "$3"
  echo "[$(ts)] extracting $3 ..." >> "$LOG"
  if python3 src/extract_roi_features.py --video_dir "$1" --output "$3" --label "$2" --max_frames $MF --workers $W >> "$LOG" 2>&1; then
    touch "$3.done"; echo "[$(ts)] DONE $3 rows=$(($(wc -l < "$3")-1))" >> "$LOG"
  else echo "[$(ts)] FAILED $3" >> "$LOG"; fi
}
echo "[$(ts)] === ROI extract START (real+F2F+NT c23, mf=$MF w=$W) ===" >> "$LOG"
run "$ROOT/original_sequences/youtube/c23/videos"           0 features/roi_original_c23.csv
run "$ROOT/manipulated_sequences/Face2Face/c23/videos"      1 features/roi_face2face_c23.csv
run "$ROOT/manipulated_sequences/NeuralTextures/c23/videos" 1 features/roi_neuraltextures_c23.csv
echo "[$(ts)] === ROI extract COMPLETE ===" >> "$LOG"
