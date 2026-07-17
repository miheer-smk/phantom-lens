#!/bin/bash
set -u
cd /home/iiitn/Downloads/phantom-lens-main
source .venv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
LOG="Major Revision Results/00_logs/ffpp_rest_extract.log"
ROOT=/home/iiitn/Datasets/FaceForensics++
MF=300; W=14
ts(){ date -u +%H:%M:%SZ; }
run(){ # $1=video_dir $2=label $3=outcsv
  if [ -f "$3" ]; then echo "[$(ts)] SKIP existing $3 (rows=$(($(wc -l < "$3")-1)))" >> "$LOG"; return; fi
  echo "[$(ts)] extracting $3 (label $2) ..." >> "$LOG"
  python src/precompute_features_best.py --video_dir "$1" --output "$3" --label "$2" --max_frames $MF --workers $W >> "$LOG" 2>&1
  echo "[$(ts)] done $3: rows=$(($(wc -l < "$3" 2>/dev/null || echo 1)-1))" >> "$LOG"
}
echo "[$(ts)] === FF++ REST extraction START (c23 F2F/FS/NT + all c40) mf=$MF w=$W ===" > "$LOG"
run "$ROOT/manipulated_sequences/Face2Face/c23/videos"      1 features/ffpp_face2face_c23.csv
run "$ROOT/manipulated_sequences/FaceSwap/c23/videos"       1 features/ffpp_faceswap_c23.csv
run "$ROOT/manipulated_sequences/NeuralTextures/c23/videos" 1 features/ffpp_neuraltextures_c23.csv
run "$ROOT/original_sequences/youtube/c40/videos"           0 features/ffpp_original_c40.csv
run "$ROOT/manipulated_sequences/Deepfakes/c40/videos"      1 features/ffpp_deepfakes_c40.csv
run "$ROOT/manipulated_sequences/Face2Face/c40/videos"      1 features/ffpp_face2face_c40.csv
run "$ROOT/manipulated_sequences/FaceSwap/c40/videos"       1 features/ffpp_faceswap_c40.csv
run "$ROOT/manipulated_sequences/NeuralTextures/c40/videos" 1 features/ffpp_neuraltextures_c40.csv
echo "[$(ts)] === FF++ REST extraction COMPLETE ===" >> "$LOG"
