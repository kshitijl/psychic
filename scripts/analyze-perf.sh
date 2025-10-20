#!/usr/bin/env bash
# Analyze performance timings from the latest psychic run

set -euo pipefail

LOG_FILE="${HOME}/.local/share/psychic/app.log"

if [[ ! -f "$LOG_FILE" ]]; then
    echo "Error: Log file not found at $LOG_FILE" >&2
    exit 1
fi

# Get the latest session ID
LATEST_SESSION=$(grep -o '\[.*\]' "$LOG_FILE" | tail -1 | awk '{print $NF}' | tr -d ']')

if [[ -z "$LATEST_SESSION" ]]; then
    echo "Error: Could not find session ID in log file" >&2
    exit 1
fi

echo "Latest session: $LATEST_SESSION"
echo ""
echo "Timing breakdown (in milliseconds):"
echo "===================================="

# Extract TIMING logs for the latest session and parse with jq
grep "$LATEST_SESSION" "$LOG_FILE" | \
    grep 'TIMING' | \
    grep -o '{.*}' | \
    jq -r '"\(.ms | tonumber | . * 100 | round / 100)\t\(.op)\t\(if .count then "(\(.count) items)" else "" end)"' | \
    column -t -s $'\t'

echo ""
echo "Total startup time:"
grep "$LATEST_SESSION" "$LOG_FILE" | \
    grep 'TIMING' | \
    grep -o '{.*}' | \
    jq -s 'map(select(.op | contains("total") or contains("first_query_complete"))) | sort_by(.ms) | reverse | .[] | "\(.ms | tonumber | . * 100 | round / 100)ms  \(.op)"'
