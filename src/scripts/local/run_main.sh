#!/usr/bin/env bash
set -euo pipefail

# ── Run configuration ───────────────────────────────────────────────────────
# Edit these to configure the run, then execute this script with no arguments.
# Leave PARAM empty to omit --param and use the problem's default parameter.
PROBLEM="vanderpol"
PARAM=""
PROFILE="fast"
SEED=123
USE_DERIVMATCH=true
USE_COLLOCATION=true

# Resolve the repo root (three levels up from this script's directory).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

LOG_DIR="$REPO_ROOT/logs"
mkdir -p "$LOG_DIR"

# PID keeps the filename unique across concurrent runs launched in the same second.
LOG_FILE="$LOG_DIR/run_main_$(date +%Y%m%d_%H%M%S)_$$.log"

ARGS=(--problem "$PROBLEM" --profile "$PROFILE" --seed "$SEED" --computer local)
[[ -n "$PARAM" ]] && ARGS+=(--param "$PARAM")
[[ "$USE_DERIVMATCH" == false ]] && ARGS+=(--no-derivmatch)
[[ "$USE_COLLOCATION" == false ]] && ARGS+=(--no-collocation)

cd "$REPO_ROOT"
echo "Logging to $LOG_FILE"
julia --project=. src/run_main.jl "${ARGS[@]}" 2>&1 | tee "$LOG_FILE"


# Set the variables above and run from the project root with no arguments:
#   ./src/scripts/local/run_main.sh
# See `julia --project=. src/run_main.jl --help` for all flags.
