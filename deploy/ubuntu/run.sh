#!/usr/bin/env bash
# Run from /home/admin/ng_run. The companion runner.py and private .env live here too.
set -Eeuo pipefail
umask 077

SELF_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
BASE=${NG_BASE_DIR:-/home/admin/ng_run}
REPO_URL=${NG_REPO_URL:-https://github.com/zifeng-yu/neural-garden.git}
BRANCH=${NG_BRANCH:-master}
export NG_PILOT_DIR=${NG_PILOT_DIR:-/home/admin/pilot}
FOREGROUND=0
CHECK_ONLY=0
PHASE=phase1
REQUESTED_RUN=""
WORKERS=${NG_WORKERS:-2}
while (( $# )); do
    case "$1" in
        phase1|phase2) PHASE=$1; shift ;;
        --foreground) FOREGROUND=1; shift ;;
        --check-only) CHECK_ONLY=1; shift ;;
        --run) [[ $# -ge 2 ]] || { echo '--run needs a path' >&2; exit 2; }; REQUESTED_RUN=$2; shift 2 ;;
        --workers) [[ $# -ge 2 ]] || { echo '--workers needs a number' >&2; exit 2; }; WORKERS=$2; shift 2 ;;
        --help)
            echo 'bash run.sh phase1 [--workers 2] [--foreground] [--check-only]'
            echo 'bash run.sh phase2 [--run /home/admin/ng_run/runs/RUN-ID] [--foreground]'
            echo 'Default: phase1 only. Download its review archive before starting phase2.'
            exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done

die() { echo "ERROR: $*" >&2; exit 1; }
[[ "$WORKERS" =~ ^[1-8]$ ]] || die 'workers must be 1..8.'
[[ "$PHASE" != phase2 || "$CHECK_ONLY" == 0 ]] || die '--check-only is phase1 only.'
[[ "$PHASE" != phase1 || -z "$REQUESTED_RUN" ]] || die 'phase1 creates a fresh run; --run is phase2 only.'
[[ $(uname -s) == Linux && -r /etc/os-release ]] || die 'Ubuntu Linux is required.'
# This is the operating system's metadata, not the API-key file.
. /etc/os-release
[[ ${ID:-} == ubuntu ]] || die 'Use Ubuntu 22.04 or newer.'
[[ ${VERSION_ID%%.*} -ge 22 ]] || die 'Use Ubuntu 22.04 or newer.'
case $(uname -m) in x86_64|aarch64) ;; *) die 'Supported CPUs: x86_64 / aarch64.' ;; esac
[[ "$BASE" = /* ]] || die 'NG_BASE_DIR must be absolute.'
[[ -f "$SELF_DIR/runner.py" ]] || die 'runner.py must be beside run.sh.'
mkdir -p "$BASE" || die "Create $BASE and give the current user write access."
BASE=$(cd "$BASE" && pwd -P)
[[ -s "$BASE/.env" ]] || die "Put the supplied private .env in $BASE/.env first."
chmod 600 "$BASE/.env"
if [[ "$PHASE" == phase1 ]]; then
    [[ "$NG_PILOT_DIR" = /* && -d "$NG_PILOT_DIR" ]] || die 'Upload articles to /home/admin/pilot/ first (or set absolute NG_PILOT_DIR).'
    articles=("$NG_PILOT_DIR"/*.md)
    [[ -f "${articles[0]}" ]] || die 'The pilot directory must contain top-level .md articles.'
fi

# Install only OS prerequisites; application dependencies stay in a private venv.
missing=0
for cmd in git curl tar flock sha256sum timeout; do
    command -v "$cmd" >/dev/null || missing=1
done
[[ -s /etc/ssl/certs/ca-certificates.crt ]] || missing=1
if [[ "$missing" == 1 ]]; then
    sudo_cmd=()
    if [[ $EUID != 0 ]]; then
        command -v sudo >/dev/null || die 'Install git curl ca-certificates tar util-linux first.'
        sudo -v
        sudo_cmd=(sudo)
    fi
    "${sudo_cmd[@]}" apt-get -o Acquire::Retries=3 -o Acquire::http::Timeout=30 update
    "${sudo_cmd[@]}" apt-get install -y --no-install-recommends git curl ca-certificates tar util-linux
fi

if [[ ${NG_LOCK_INHERITED:-0} != 1 ]]; then
    exec 9>"$BASE/.run.lock"
    flock -n 9 || die 'Another indexing job is already running in this directory.'
fi
mkdir -p "$BASE/logs" "$BASE/runs" "$BASE/exports" "$BASE/tools"
# Resolve the phase2 target before background launch; never follow a later run.
if [[ "$PHASE" == phase2 ]]; then
    target=${REQUESTED_RUN:-$BASE/latest-run}
    [[ -d "$target" ]] || die 'No phase1 run found.'
    RUN=$(cd "$target" && pwd -P)
    [[ "$RUN" == "$BASE/runs/"* ]] || die '--run must be inside this base/runs directory.'
    [[ -f "$RUN/PHASE1_READY.txt" && -f "$RUN/phase1-review-manifest.json" ]] || die 'Phase1 review package is not ready.'
    [[ ! -e "$RUN/data" ]] || die 'Phase2 already has data; refusing to overwrite or resume a partial database.'
    [[ -x "$RUN/project/.venv/bin/python" && -f "$RUN/runner.py" ]] || die 'Run environment is missing; keep runs/ between phases.'
fi
if [[ "$FOREGROUND" == 0 ]]; then
    args=("$PHASE" --foreground --workers "$WORKERS")
    [[ "$CHECK_ONLY" == 0 ]] || args+=(--check-only)
    [[ "$PHASE" != phase2 ]] || args+=(--run "$RUN")
    launcher="$BASE/logs/launch-$PHASE-$(date -u +%Y%m%dT%H%M%SZ)-$$.log"
    NG_LOCK_INHERITED=1 NG_BASE_DIR="$BASE" nohup bash "$SELF_DIR/run.sh" "${args[@]}" \
        >"$launcher" 2>&1 </dev/null &
    echo "$!" >"$BASE/latest.pid"
    ln -sfn "$launcher" "$BASE/latest.log"
    echo "Started $PHASE, PID $(cat "$BASE/latest.pid")."
    echo "Progress: tail -f '$BASE/latest.log'"
    echo 'phase1 completion: PHASE1_READY.txt; phase2 completion: SUCCESS.txt.'
    exit 0
fi

if [[ "$PHASE" == phase1 ]]; then
    RUN="$BASE/runs/$(date -u +%Y%m%dT%H%M%SZ)-$$"
    mkdir "$RUN"
    cp "$SELF_DIR/runner.py" "$RUN/runner.py"
fi
ln -sfn "$RUN" "$BASE/latest-run"
trap 'rc=$?; if (( rc != 0 )); then printf "Phase: %s\nExit code: %s\nSee logs/status; this phase did NOT complete.\n" "$PHASE" "$rc" >"$RUN/FAILED.txt"; fi' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
echo "Phase: $PHASE; run directory: $RUN"

export UV_CACHE_DIR="$BASE/tools/uv-cache"
export UV_PYTHON_INSTALL_DIR="$BASE/tools/python"
export UV_PYTHON_PREFERENCE=only-managed
export UV_HTTP_TIMEOUT=${UV_HTTP_TIMEOUT:-300}
export UV_CONCURRENT_DOWNLOADS=${UV_CONCURRENT_DOWNLOADS:-4}
export PYTHONUNBUFFERED=1
unset VIRTUAL_ENV PYTHONPATH
PYTHON="$RUN/project/.venv/bin/python"
step() { "$PYTHON" "$RUN/runner.py" "$1" --project "$RUN/project" --run "$RUN" --base "$BASE" --workers "$WORKERS"; }

if [[ "$PHASE" == phase2 ]]; then
    # Use the exact interpreter, code, config and runner captured by phase1.
    # No Git pull, no re-extraction, and no automatic start after phase1.
    cd "$RUN/project"
    step phase2
    step verify
    step package
    echo "Phase2 complete: $RUN/SUCCESS.txt"
    exit 0
fi

# Never overwrite an existing checkout or dataset. Git is a source-only cache.
export GIT_TERMINAL_PROMPT=0
git check-ref-format --branch "$BRANCH" >/dev/null
CACHE="$BASE/repository.git"
if [[ ! -d "$CACHE" ]]; then
    # Publish the cache only after cloning succeeds, so a timeout cannot poison retries.
    timeout 300 git clone --bare --depth 1 --single-branch --branch "$BRANCH" "$REPO_URL" "$RUN/repository.git"
    mv "$RUN/repository.git" "$CACHE"
fi
[[ $(git --git-dir="$CACHE" remote get-url origin) == "$REPO_URL" ]] || die 'Cached origin differs from NG_REPO_URL.'
timeout 300 git --git-dir="$CACHE" fetch --depth 1 --prune origin "+refs/heads/$BRANCH:refs/heads/$BRANCH"
COMMIT=$(git --git-dir="$CACHE" rev-parse "refs/heads/$BRANCH^{commit}")
printf '%s\n' "$COMMIT" >"$RUN/commit.txt"
mkdir "$RUN/project"
git --git-dir="$CACHE" archive "$COMMIT" | tar -x -C "$RUN/project"
echo "Fetched commit: $COMMIT"

UV="$BASE/tools/uv/uv"
if [[ ! -x "$UV" ]]; then
    curl --fail --show-error --location --retry 3 --connect-timeout 15 --max-time 180 \
        https://astral.sh/uv/install.sh -o "$RUN/uv-install.sh"
    UV_INSTALL_DIR="$BASE/tools/uv" UV_NO_MODIFY_PATH=1 sh "$RUN/uv-install.sh"
fi
"$UV" --version
cd "$RUN/project"
"$UV" sync --locked --no-dev --python 3.12.10
[[ -f "$RUN/project/src/indexer_two_stage.py" ]] || die 'Push the new two-stage code to Git before running this package.'
step prepare
step probe
if [[ "$CHECK_ONLY" == 1 ]]; then
    echo 'DashScope checks passed. No extraction or indexing was performed.'
    exit 0
fi
# Export diagnostic logs even when a document fails outside knowledge extraction.
if step phase1; then
    step phase1-package
    echo "Phase1 complete: $RUN/PHASE1_READY.txt"
    echo 'STOPPED FOR REVIEW. Phase2 has NOT started.'
else
    step phase1-package
    exit 1
fi
