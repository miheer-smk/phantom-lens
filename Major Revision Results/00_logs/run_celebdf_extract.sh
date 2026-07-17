#!/bin/bash
set -u
cd /home/iiitn/Downloads/phantom-lens-main
source .venv/bin/activate
LOG="Major Revision Results/00_logs/celebdf_extract.log"
OUT=features/celebdf_features.csv
ROOT=/home/iiitn/Datasets/Celeb-DF-v2
MF=150; W=16
ts(){ date -u +%H:%M:%SZ; }
echo "[$(ts)] === CelebDF extraction START (max_frames=$MF workers=$W) ===" > "$LOG"
echo "[$(ts)] seed=42 (extraction is deterministic; no RNG). mediapipe=0.10.18 opencv=4.11" >> "$LOG"
# Celeb-real (real=0) fresh
python src/precompute_features_best.py --video_dir "$ROOT/Celeb-real" --output "$OUT" --label 0 --max_frames $MF --workers $W >> "$LOG" 2>&1
echo "[$(ts)] Celeb-real done. rows=$(($(wc -l < "$OUT")-1))" >> "$LOG"
# YouTube-real (real=0) append
python src/precompute_features_best.py --video_dir "$ROOT/YouTube-real" --output "$OUT" --label 0 --max_frames $MF --workers $W --append >> "$LOG" 2>&1
echo "[$(ts)] YouTube-real done. rows=$(($(wc -l < "$OUT")-1))" >> "$LOG"
# Celeb-synthesis (fake=1) append
python src/precompute_features_best.py --video_dir "$ROOT/Celeb-synthesis" --output "$OUT" --label 1 --max_frames $MF --workers $W --append >> "$LOG" 2>&1
echo "[$(ts)] Celeb-synthesis done. rows=$(($(wc -l < "$OUT")-1))" >> "$LOG"
echo "[$(ts)] === CelebDF extraction COMPLETE totalrows=$(($(wc -l < "$OUT")-1)) ===" >> "$LOG"
