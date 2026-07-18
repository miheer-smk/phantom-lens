#!/bin/bash
set -u; cd /home/iiitn/Downloads/phantom-lens-main; source .venv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
LOG="Major Revision Results/00_logs/roi_dffs.log"; ROOT=/home/iiitn/Datasets/FaceForensics++
ts(){ date -u +%H:%M:%SZ; }
run(){ [ -f "$3.done" ] && return; rm -f "$3"
  echo "[$(ts)] $3" >> "$LOG"
  python3 src/extract_roi_features.py --video_dir "$1" --output "$3" --label "$2" --max_frames 150 --workers 10 >> "$LOG" 2>&1 && touch "$3.done"
  echo "[$(ts)] DONE $3 rows=$(($(wc -l < "$3")-1))" >> "$LOG"; }
echo "[$(ts)] === ROI DF+FS START ===" >> "$LOG"
run "$ROOT/manipulated_sequences/Deepfakes/c23/videos" 1 features/roi_deepfakes_c23.csv
run "$ROOT/manipulated_sequences/FaceSwap/c23/videos"  1 features/roi_faceswap_c23.csv
echo "[$(ts)] === ROI DF+FS COMPLETE ===" >> "$LOG"
