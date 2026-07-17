#!/bin/bash
# Downloads FF++ c23 + c40 for the datasets required by the 12 reviewer experiments.
# Resumable (download_ffpp.py skips existing files). TOS auto-agreed via empty stdin.
set -u
SCRIPT=/home/iiitn/Downloads/download_ffpp.py
OUT=/home/iiitn/Datasets/FaceForensics++
LOG="/home/iiitn/Downloads/phantom-lens-main/Major Revision Results/00_logs/download_monitor.log"
SERVER=EU2
DATASETS=(original Deepfakes Face2Face FaceSwap NeuralTextures)
ts(){ date -u +%H:%M:%SZ; }
echo "[$(ts)] === c23+c40 driver START (server $SERVER) ===" >> "$LOG"
for COMP in c23 c40; do
  for DS in "${DATASETS[@]}"; do
    echo "[$(ts)] downloading $DS @ $COMP ..." >> "$LOG"
    echo "" | python3 "$SCRIPT" "$OUT" -d "$DS" -c "$COMP" -t videos --server "$SERVER" \
        >> "$LOG" 2>&1
    echo "[$(ts)] finished $DS @ $COMP (rc=$?) size-so-far=$(du -sh "$OUT" 2>/dev/null | cut -f1)" >> "$LOG"
  done
done
echo "[$(ts)] === c23+c40 driver COMPLETE ===" >> "$LOG"
