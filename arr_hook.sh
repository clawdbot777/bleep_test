#!/bin/sh
# arr_hook.sh - Radarr/Sonarr custom script hook for Bleeper
#
# Install:
#   1. Place this file somewhere accessible to the arr container
#   2. Radarr/Sonarr → Settings → Connect → Custom Script
#      Script path: /path/to/arr_hook.sh
#      Events: On Import, On Upgrade
#
# Configure the variables below:

BLEEPER_URL="http://bleeper:5000"

# Path translation: arr sees media at ARR_MEDIA_ROOT, bleeper sees it at BLEEPER_MEDIA_ROOT
# Check your volume mappings in each container to confirm these.
# Example: Radarr has /data/media mapped, bleeper has /media mapped → same host folder.
ARR_MEDIA_ROOT="/data/media"
BLEEPER_MEDIA_ROOT="/media"

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

# Translate path from arr's view → bleeper's view
BLEEPER_PATH="${FILE_PATH#$ARR_MEDIA_ROOT}"
BLEEPER_PATH="$BLEEPER_MEDIA_ROOT$BLEEPER_PATH"

FILENAME=$(basename "$FILE_PATH")
echo "[bleeper] Submitting: $FILENAME"
echo "[bleeper] Path (arr):     $FILE_PATH"
echo "[bleeper] Path (bleeper): $BLEEPER_PATH"

# ── POST to bleeper API ───────────────────────────────────────────────────────
RESPONSE=$(curl -s -w "\n%{http_code}" \
    -X POST "$BLEEPER_URL/api/process_full" \
    -H "Content-Type: application/json" \
    -d "{\"filename\": \"$BLEEPER_PATH\"}" \
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
