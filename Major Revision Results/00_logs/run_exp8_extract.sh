#!/bin/bash
set -u; cd /home/iiitn/Downloads/phantom-lens-main; source .venv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
LOG="Major Revision Results/00_logs/exp8_extract.log"; ROOT=/home/iiitn/Datasets/FaceForensics++
ts(){ date -u +%H:%M:%SZ; }
run(){ [ -f "$3.done" ] && return; rm -f "$3"
  echo "[$(ts)] $3" >> "$LOG"
  python3 "Major Revision Results/00_logs/exp8_residual_extract.py" --video_dir "$1" --output "$3" --label "$2" --workers 12 >> "$LOG" 2>&1 && touch "$3.done"
  echo "[$(ts)] done $3 rows=$(($(wc -l < "$3")-1))" >> "$LOG"; }
echo "[$(ts)] === EXP-8 residual extraction START ===" >> "$LOG"
run "$ROOT/original_sequences/youtube/c23/videos"           0 features/residual_original_c23.csv
run "$ROOT/manipulated_sequences/Deepfakes/c23/videos"      1 features/residual_deepfakes_c23.csv
run "$ROOT/manipulated_sequences/Face2Face/c23/videos"      1 features/residual_face2face_c23.csv
run "$ROOT/manipulated_sequences/FaceSwap/c23/videos"       1 features/residual_faceswap_c23.csv
run "$ROOT/manipulated_sequences/NeuralTextures/c23/videos" 1 features/residual_neuraltextures_c23.csv
echo "[$(ts)] === EXP-8 residual extraction COMPLETE ===" >> "$LOG"
