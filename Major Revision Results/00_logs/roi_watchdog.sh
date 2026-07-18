#!/bin/bash
set -u
cd /home/iiitn/Downloads/phantom-lens-main
LOG="Major Revision Results/00_logs/roi_watchdog.log"
DRV="Major Revision Results/00_logs/run_roi_extract.sh"
ALERTS="Major Revision Results/07_summary/ALERTS.md"
ts(){ date -u +%Y-%m-%dT%H:%M:%SZ; }
ndone(){ ls features/roi_*_c23.csv.done 2>/dev/null | wc -l; }
echo "[$(ts)] [roi-watchdog] start" >> "$LOG"
prev=-1; stall=0
while true; do
  if [ "$(ndone)" -ge 3 ] || grep -q "ROI extract COMPLETE" "Major Revision Results/00_logs/roi_extract.log" 2>/dev/null; then
    echo "[$(ts)] [roi-watchdog] complete — exit" >> "$LOG"; exit 0
  fi
  sleep 120
  cur=$(( $(cat features/roi_*_c23.csv 2>/dev/null | wc -l) ))
  up=$(ps -eo args 2>/dev/null | grep -c '[e]xtract_roi_features')
  drv=$(ps -eo args 2>/dev/null | grep -c '[r]un_roi_extract.sh')
  [ "$cur" -le "$prev" ] && stall=$((stall+1)) || stall=0; prev=$cur
  echo "[$(ts)] rows=$cur done=$(ndone)/3 workers=$up driver=$drv stall=$stall" >> "$LOG"
  if [ "$drv" -eq 0 ] && [ "$up" -eq 0 ]; then
    echo "- **$(ts)** — ROI (Track C) extraction driver DOWN -> auto-relaunched." >> "$ALERTS"
    echo "[$(ts)] [roi-watchdog] relaunch" >> "$LOG"
    setsid bash "$DRV" >/dev/null 2>&1 &
    stall=0
  fi
done
