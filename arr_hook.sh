#!/bin/sh
# arr_hook.sh - Radarr/Sonarr custom script hook for Bleeper
#
# Install:
#   1. Place this file somewhere accessible to the arr container
#   2. Radarr/Sonarr → Settings → Connect → Custom Script
#      Script path: /path/to/arr_hook.sh
#      Events: On Import, On Upgrade
#
# Configure the three variables below:

BLEEPER_URL="http://bleeper:5000"
BLEEPER_UPLOAD="/app/uploads"   # container-side path (inside bleeper container)

# ── Detect arr type and get file path ────────────────────────────────────────
if [ -n "$radarr_eventtype" ]; then
    EVENT="$radarr_eventtype"
    FILE_PATH="$radarr_moviefile_path"
elif [ -n "$sonarr_eventtype" ]; then
    EVENT="$sonarr_eventtype"
    FILE_PATH="$sonarr_episodefile_path"
else
    echo "[bleeper] No event detected - skipping"
    exit 0
fi

# Only act on import/upgrade events
case "$EVENT" in
    Download|EpisodeFileImport|MovieFileImport) ;;
    Test) echo "[bleeper] Test event received - OK"; exit 0 ;;
    *) echo "[bleeper] Skipping event: $EVENT"; exit 0 ;;
esac

if [ -z "$FILE_PATH" ]; then
    echo "[bleeper] No file path in event - skipping"
    exit 0
fi

FILENAME=$(basename "$FILE_PATH")
echo "[bleeper] Submitting: $FILENAME"

# ── POST to bleeper API ───────────────────────────────────────────────────────
RESPONSE=$(curl -s -w "\n%{http_code}" \
    -X POST "$BLEEPER_URL/api/process_full" \
    -H "Content-Type: application/json" \
    -d "{\"filename\": \"$FILENAME\"}" \
    --connect-timeout 10 \
    --max-time 30)

HTTP_CODE=$(echo "$RESPONSE" | tail -1)
BODY=$(echo "$RESPONSE" | head -1)

if [ "$HTTP_CODE" = "202" ]; then
    JOB_ID=$(echo "$BODY" | grep -o '"job_id":"[^"]*"' | cut -d'"' -f4)
    echo "[bleeper] Queued OK - job_id: $JOB_ID"
    exit 0
else
    echo "[bleeper] Failed - HTTP $HTTP_CODE: $BODY"
    exit 1
fi
