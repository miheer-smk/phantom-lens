#!/bin/bash
set -u
cd /home/iiitn/Downloads/phantom-lens-main
source .venv/bin/activate
# limit per-worker intra-op threads to reduce contention with the concurrent CelebDF job
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
LOG="Major Revision Results/00_logs/ffpp_df_extract.log"
ROOT=/home/iiitn/Datasets/FaceForensics++
MF=300; W=10
ts(){ date -u +%H:%M:%SZ; }
echo "[$(ts)] === FF++ DF-gate extraction START (max_frames=$MF workers=$W, c23 originals+Deepfakes) ===" > "$LOG"
# Originals (real=0) c23
python src/precompute_features_best.py --video_dir "$ROOT/original_sequences/youtube/c23/videos" \
    --output features/ffpp_original_c23.csv --label 0 --max_frames $MF --workers $W >> "$LOG" 2>&1
echo "[$(ts)] originals done: rows=$(($(wc -l < features/ffpp_original_c23.csv 2>/dev/null || echo 1)-1))" >> "$LOG"
# Deepfakes (fake=1) c23
python src/precompute_features_best.py --video_dir "$ROOT/manipulated_sequences/Deepfakes/c23/videos" \
    --output features/ffpp_deepfakes_c23.csv --label 1 --max_frames $MF --workers $W >> "$LOG" 2>&1
echo "[$(ts)] deepfakes done: rows=$(($(wc -l < features/ffpp_deepfakes_c23.csv 2>/dev/null || echo 1)-1))" >> "$LOG"
echo "[$(ts)] === FF++ DF-gate extraction COMPLETE ===" >> "$LOG"
