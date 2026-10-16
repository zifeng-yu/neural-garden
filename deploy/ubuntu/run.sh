#!/usr/bin/env bash
# Run from /home/admin/ng_run. The companion runner.py and private .env live here too.
set -Eeuo pipefail
umask 077

SELF_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
BASE=${NG_BASE_DIR:-/home/admin/ng_run}
REPO_URL=${NG_REPO_URL:-https://github.com/zifeng-yu/neural-garden.git}
BRANCH=${NG_BRANCH:-master}
FOREGROUND=0
CHECK_ONLY=0
for arg in "$@"; do
    case "$arg" in
        --foreground) FOREGROUND=1 ;;
        --check-only) CHECK_ONLY=1 ;;
        --help)
            echo 'bash run.sh [--foreground] [--check-only]'
            echo 'Default: start a detached, fresh build under /home/admin/ng_run/runs/.'
            exit 0 ;;
        *) echo "Unknown argument: $arg" >&2; exit 2 ;;
    esac
done

die() { echo "ERROR: $*" >&2; exit 1; }
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

# Install only OS prerequisites; application dependencies stay in a private venv.
missing=0
for cmd in git curl tar flock sha256sum; do
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
if [[ "$FOREGROUND" == 0 ]]; then
    args=(--foreground)
    [[ "$CHECK_ONLY" == 0 ]] || args+=(--check-only)
    launcher="$BASE/logs/launch-$(date -u +%Y%m%dT%H%M%SZ)-$$.log"
    NG_LOCK_INHERITED=1 NG_BASE_DIR="$BASE" nohup bash "$SELF_DIR/run.sh" "${args[@]}" \
        >"$launcher" 2>&1 </dev/null &
    echo "$!" >"$BASE/latest.pid"
    ln -sfn "$launcher" "$BASE/latest.log"
    echo "Started PID $(cat "$BASE/latest.pid")."
    echo "Progress: tail -f '$BASE/latest.log'"
    echo "Completion requires SUCCESS.txt in the latest run; starting is not success."
    exit 0
fi

RUN="$BASE/runs/$(date -u +%Y%m%dT%H%M%SZ)-$$"
mkdir "$RUN"
ln -sfn "$RUN" "$BASE/latest-run"
trap 'rc=$?; if (( rc != 0 )); then printf "Exit code: %s\nSee launcher log and status.json; this is NOT a completed dataset.\n" "$rc" >"$RUN/FAILED.txt"; fi' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
cp "$SELF_DIR/runner.py" "$RUN/runner.py"
echo "Run directory: $RUN"

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

export UV_CACHE_DIR="$BASE/tools/uv-cache"
export UV_PYTHON_INSTALL_DIR="$BASE/tools/python"
export UV_PYTHON_PREFERENCE=only-managed
export PYTHONUNBUFFERED=1
unset VIRTUAL_ENV PYTHONPATH
UV="$BASE/tools/uv/uv"
if [[ ! -x "$UV" ]]; then
    curl --fail --show-error --location --retry 3 --connect-timeout 15 --max-time 180 \
        https://astral.sh/uv/install.sh -o "$RUN/uv-install.sh"
    UV_INSTALL_DIR="$BASE/tools/uv" UV_NO_MODIFY_PATH=1 sh "$RUN/uv-install.sh"
fi
"$UV" --version
cd "$RUN/project"
"$UV" sync --locked --no-dev --python 3.12
PYTHON="$RUN/project/.venv/bin/python"

step() { "$PYTHON" "$RUN/runner.py" "$1" --project "$RUN/project" --run "$RUN" --base "$BASE"; }
step prepare
step probe
if [[ "$CHECK_ONLY" == 1 ]]; then
    echo 'DashScope LLM and embedding checks passed. No indexing was performed.'
    exit 0
fi
step index
# Separate processes ensure Chroma has closed before validating and packaging.
step verify
step package
echo "Finished. Read $RUN/SUCCESS.txt for the downloadable archive."
