#!/bin/bash
set -u; cd /home/iiitn/Downloads/phantom-lens-main; source .venv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
LOG="Major Revision Results/00_logs/exp9_extract.log"; ROOT=/home/iiitn/Datasets/FaceForensics++
ts(){ date -u +%H:%M:%SZ; }
run(){ [ -f "$3.done" ] && return; rm -f "$3"
  echo "[$(ts)] $3" >> "$LOG"
  python3 "Major Revision Results/00_logs/exp9_rppg_extract.py" --video_dir "$1" --output "$3" --label "$2" --workers 12 >> "$LOG" 2>&1 && touch "$3.done"
  echo "[$(ts)] done $3 rows=$(($(wc -l < "$3")-1))" >> "$LOG"; }
echo "[$(ts)] === EXP-9 rPPG extraction START ===" >> "$LOG"
for comp in c23 c40; do
  run "$ROOT/original_sequences/youtube/$comp/videos"           0 features/rppg_original_$comp.csv
  run "$ROOT/manipulated_sequences/Deepfakes/$comp/videos"      1 features/rppg_deepfakes_$comp.csv
  run "$ROOT/manipulated_sequences/Face2Face/$comp/videos"      1 features/rppg_face2face_$comp.csv
  run "$ROOT/manipulated_sequences/FaceSwap/$comp/videos"       1 features/rppg_faceswap_$comp.csv
  run "$ROOT/manipulated_sequences/NeuralTextures/$comp/videos" 1 features/rppg_neuraltextures_$comp.csv
done
echo "[$(ts)] === EXP-9 rPPG extraction COMPLETE ===" >> "$LOG"
