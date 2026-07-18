#!/bin/bash
# Robust incremental backup of all revision results. Run after EVERY new result.
# Creates a fresh timestamped tarball in 2 on-disk locations + keeps a 'latest' pointer.
set -u
cd /home/iiitn/Downloads/phantom-lens-main
STAMP=$(date +%Y%m%d_%H%M%S)
DST1=/home/iiitn/phantom_lens_revision_backups
DST2=/home/iiitn/Datasets/_phantomlens_backups
mkdir -p "$DST1" "$DST2"
NAME="phantomlens_revision_${STAMP}.tar.gz"
# commit any new results first (provenance)
git add -A 2>/dev/null && git commit -q -m "auto-backup checkpoint ${STAMP}" 2>/dev/null
tar czf "$DST1/$NAME" --exclude='.venv' --exclude='__pycache__' \
  "Major Revision Results" results_clean splits src config evaluation training \
  LOCKED_NUMBERS.md requirements*.txt README.md pyproject.toml .git \
  features/*.csv features/*.done 2>/dev/null
if gzip -t "$DST1/$NAME" 2>/dev/null; then
  cp "$DST1/$NAME" "$DST2/$NAME"
  ln -sf "$NAME" "$DST1/LATEST.tar.gz"
  # keep only the 6 most recent to save space
  ls -t "$DST1"/phantomlens_revision_*.tar.gz 2>/dev/null | tail -n +7 | xargs -r rm -f
  ls -t "$DST2"/phantomlens_revision_*.tar.gz 2>/dev/null | tail -n +7 | xargs -r rm -f
  echo "OK backup $NAME ($(du -h "$DST1/$NAME"|cut -f1)) -> 2 locations, verified"
else
  echo "ERROR: backup archive failed gzip test — NOT propagated"; exit 1
fi
