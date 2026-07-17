#!/bin/bash
# Self-healing watchdog for FF++ feature extraction.
# Monitors: driver alive, progress (stall), memory (leak), disk. Auto-corrects; logs alerts.
# Resume-safe: relaunching run_ffpp_extract_all.sh skips .done files, redoes in-progress cleanly.
set -u
cd /home/iiitn/Downloads/phantom-lens-main
BASE="Major Revision Results/00_logs"
DRIVER="$BASE/run_ffpp_extract_all.sh"
LOG="$BASE/extract_watchdog.log"
ALERTS="Major Revision Results/07_summary/ALERTS.md"
EXLOG="$BASE/ffpp_extract_all.log"
ts(){ date -u +%Y-%m-%dT%H:%M:%SZ; }
totrows(){ local t=0; for f in features/ffpp_*_c23.csv features/ffpp_*_c40.csv; do [ -f "$f" ] && t=$((t + $(wc -l < "$f") - 1)); done; echo $t; }
ndone(){ ls features/ffpp_*.done 2>/dev/null | wc -l; }
alert(){ echo "- **$(ts)** — $1" >> "$ALERTS"; echo "[$(ts)] ALERT: $1" >> "$LOG"; }
relaunch(){ setsid bash "$DRIVER" >/dev/null 2>&1 & echo "[$(ts)] ACTION: relaunched extraction driver (resume-safe)" >> "$LOG"; }

echo "[$(ts)] [watchdog] START (pid $$)" >> "$LOG"
prev=$(totrows); stall=0
while true; do
  # stop condition: all 10 FF++ sets done
  if [ "$(ndone)" -ge 10 ] || grep -q "extract-all COMPLETE" "$EXLOG" 2>/dev/null; then
    echo "[$(ts)] [watchdog] all FF++ extraction COMPLETE — exiting." >> "$LOG"; exit 0
  fi
  sleep 120
  cur=$(totrows)
  memused=$(free -m | awk 'NR==2{print $3}')       # MB
  memavail=$(free -m | awk 'NR==2{print $7}')       # MB
  diskfree=$(df -m / | awk 'NR==2{print $4}')       # MB
  drv=$(ps -eo args 2>/dev/null | grep -c '[r]un_ffpp_extract_all.sh')
  wk=$(ps -eo args 2>/dev/null | grep -c '[p]recompute_features_best.py')
  [ "$cur" -le "$prev" ] && stall=$((stall+1)) || stall=0
  prev=$cur
  echo "[$(ts)] rows=$cur done=$(ndone)/10 driver=$drv workers=$wk mem_used=${memused}MB avail=${memavail}MB disk=${diskfree}MB stall=$stall" >> "$LOG"

  # CORRECT: disk critically low -> alert + stop (unrecoverable without intervention)
  if [ "$diskfree" -lt 20000 ]; then alert "Disk critically low (${diskfree}MB free). Extraction paused-risk. NEEDS ATTENTION."; fi
  # CORRECT: driver dead but work remains -> relaunch
  if [ "$drv" -eq 0 ]; then
    alert "Extraction driver was DOWN (rows=$cur, done=$(ndone)/10) — auto-relaunched."
    relaunch; stall=0; sleep 30; continue
  fi
  # CORRECT: memory near-exhaustion (leak) -> recycle workers via driver restart (resume-safe)
  if [ "$memavail" -lt 12000 ]; then
    alert "Memory low (avail ${memavail}MB) — recycling workers (kill+relaunch driver to reset leak)."
    pkill -9 -f precompute_features_best.py 2>/dev/null; pkill -9 -f run_ffpp_extract_all.sh 2>/dev/null
    sleep 5; relaunch; stall=0; sleep 30; continue
  fi
  # CORRECT: hard stall (>~14 min no new rows) with driver alive -> restart
  if [ "$stall" -ge 7 ]; then
    alert "Stalled ${stall} checks (~14min, no new rows) — kill+relaunch to unstick."
    pkill -9 -f precompute_features_best.py 2>/dev/null; pkill -9 -f run_ffpp_extract_all.sh 2>/dev/null
    sleep 5; relaunch; stall=0
  fi
done
