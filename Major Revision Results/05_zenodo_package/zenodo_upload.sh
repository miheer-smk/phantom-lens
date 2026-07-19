#!/usr/bin/env bash
# Zenodo deposition — creates a DRAFT and uploads the reproducibility package.
# It DELIBERATELY DOES NOT PUBLISH: no permanent DOI is minted by this script.
# Publishing (which mints the DOI that goes in the paper) is a manual click in the
# Zenodo web UI, after Miheer + co-author review the draft. Author decision by design.
#
# Prerequisites:
#   1. Fill in ALL "TODO" fields in zenodo_metadata.json (authors, license, ORCIDs, paper DOI).
#      The script refuses to run while any TODO remains.
#   2. export ZENODO_TOKEN=<your personal access token>   (scope: deposit:write, deposit:actions)
#   3. (optional) export ZENODO_HOST=sandbox.zenodo.org    to dry-run on the sandbox first (recommended).
#
# Usage:  ./zenodo_upload.sh phantomlens_reproducibility_package.zip
set -euo pipefail

ZIP="${1:-phantomlens_reproducibility_package.zip}"
HERE="$(cd "$(dirname "$0")" && pwd)"
META="$HERE/zenodo_metadata.json"
HOST="${ZENODO_HOST:-zenodo.org}"          # sandbox.zenodo.org for a dry run
API="https://$HOST/api"

[ -n "${ZENODO_TOKEN:-}" ] || { echo "ERROR: export ZENODO_TOKEN first."; exit 1; }
[ -f "$ZIP" ]  || { echo "ERROR: archive '$ZIP' not found (run make_zenodo_zip first)."; exit 1; }
[ -f "$META" ] || { echo "ERROR: $META not found."; exit 1; }
if grep -q "TODO" "$META"; then
  echo "ERROR: zenodo_metadata.json still contains TODO placeholders."
  echo "       Fill in author list / license / ORCIDs / paper DOI before uploading."
  grep -n "TODO" "$META" | sed 's/^/   /'
  exit 1
fi

echo ">> host: $HOST   archive: $ZIP ($(du -h "$ZIP" | cut -f1))"
echo ">> creating DRAFT deposition ..."
DEP=$(curl -sf -X POST "$API/deposit/depositions?access_token=$ZENODO_TOKEN" \
      -H "Content-Type: application/json" -d '{}')
DEP_ID=$(echo "$DEP" | python3 -c 'import sys,json;print(json.load(sys.stdin)["id"])')
BUCKET=$(echo "$DEP" | python3 -c 'import sys,json;print(json.load(sys.stdin)["links"]["bucket"])')
echo "   deposition id: $DEP_ID"

echo ">> uploading archive to bucket ..."
curl -sf -X PUT "$BUCKET/$(basename "$ZIP")?access_token=$ZENODO_TOKEN" \
     --upload-file "$ZIP" -o /dev/null
echo "   upload complete."

echo ">> attaching metadata ..."
curl -sf -X PUT "$API/deposit/depositions/$DEP_ID?access_token=$ZENODO_TOKEN" \
     -H "Content-Type: application/json" -d @"$META" -o /dev/null

echo ""
echo ">> DRAFT ready — NOT published, no DOI minted."
echo ">> Review + publish manually here:"
echo "     https://$HOST/deposit/$DEP_ID"
echo ">> Publishing there mints the permanent DOI for the paper. Do that only after"
echo "   Miheer + co-author sign off on authors and metadata."
