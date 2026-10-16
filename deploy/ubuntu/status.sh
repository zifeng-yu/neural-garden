#!/usr/bin/env bash
# Read-only status: safe to run while either phase is working or awaiting review.
set -eu
BASE=${NG_BASE_DIR:-/home/admin/ng_run}
if [[ ! -d "$BASE/latest-run" ]]; then
    echo 'No run directory yet. Check latest.log for environment installation progress.'
    [[ ! -f "$BASE/latest.log" ]] || tail -n 12 "$BASE/latest.log"
    exit 0
fi
RUN=$(cd "$BASE/latest-run" && pwd -P)
printf 'Run: %s\n' "$RUN"
if [[ -f "$BASE/latest.pid" ]]; then
    pid=$(cat "$BASE/latest.pid")
    if [[ "$pid" =~ ^[0-9]+$ ]]; then
        ps -p "$pid" -o pid,etime,stat,args || true
    fi
fi
if [[ -f "$RUN/status.json" ]]; then
    cat "$RUN/status.json"
else
    echo 'Preparing environment; Python stage status is not available yet.'
fi
if [[ -f "$RUN/SUCCESS.txt" ]]; then
    cat "$RUN/SUCCESS.txt"
elif [[ -f "$RUN/FAILED.txt" ]]; then
    cat "$RUN/FAILED.txt"
    [[ ! -f "$RUN/failure.json" ]] || cat "$RUN/failure.json"
elif [[ -f "$RUN/PHASE1_READY.txt" && ! -e "$RUN/data" ]]; then
    cat "$RUN/PHASE1_READY.txt"
    echo 'Phase1 exits normally while awaiting review. No live PID is expected then.'
fi
printf '\nRecent launcher output:\n'
[[ ! -f "$BASE/latest.log" ]] || tail -n 12 "$BASE/latest.log"
