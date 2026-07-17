#!/bin/bash
# Supervises the FF++ c23+c40 driver: keeps it alive, restarts on death or 20-min stall.
# Exits 0 when the driver logs COMPLETE. Logs every action.
set -u
BASE=/home/iiitn/Downloads/phantom-lens-main
LOG="$BASE/Major Revision Results/00_logs/download_monitor.log"
DRIVER="$BASE/Major Revision Results/00_logs/run_ffpp_download.sh"
OUT=/home/iiitn/Datasets/FaceForensics++
ts(){ date -u +%H:%M:%SZ; }
prev=-1; stall=0
echo "[$(ts)] [watchdog] started (pid $$)" >> "$LOG"
while true; do
  if grep -q "driver COMPLETE" "$LOG"; then
    echo "[$(ts)] [watchdog] driver COMPLETE detected — exiting." >> "$LOG"; exit 0
  fi
  sleep 300
  cur=$(du -sb "$OUT" 2>/dev/null | cut -f1); cur=${cur:-0}
  driver_up=$(pgrep -f run_ffpp_download.sh | head -1)
  dl_up=$(pgrep -f download_ffpp.py | head -1)
  hr=$(printf '%d' "$cur" | awk '{printf "%.1fGB", $1/1073741824}')
  if [ "$cur" -le "$prev" ]; then stall=$((stall+1)); else stall=0; fi
  prev=$cur
  echo "[$(ts)] [watchdog] size=$hr driver=${driver_up:-DOWN} dl=${dl_up:-none} stall_ticks=$stall" >> "$LOG"
  if [ -z "$driver_up" ]; then
    echo "[$(ts)] [watchdog] driver DOWN & not complete -> relaunching (resumes, skips existing)" >> "$LOG"
    nohup bash "$DRIVER" >/dev/null 2>&1 &
    stall=0
  elif [ "$stall" -ge 4 ]; then
    echo "[$(ts)] [watchdog] STALL 20min no growth -> killing dl+driver and relaunching" >> "$LOG"
    pkill -f download_ffpp.py 2>/dev/null; pkill -f run_ffpp_download.sh 2>/dev/null; sleep 3
    nohup bash "$DRIVER" >/dev/null 2>&1 &
    stall=0
  fi
done
