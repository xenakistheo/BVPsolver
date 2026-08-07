#!/usr/bin/env bash

# This script needs to be adapted. 
set -euo pipefail

# Resolve the repo root (three levels up from this script's directory).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

LOG_DIR="$REPO_ROOT/logs"
mkdir -p "$LOG_DIR"

# PID keeps the filename unique across concurrent runs launched in the same second.
LOG_FILE="$LOG_DIR/run_main_$(date +%Y%m%d_%H%M%S)_$$.log"

cd "$REPO_ROOT"
echo "Logging to $LOG_FILE"
julia --project=. src/run_main.jl "$@" 2>&1 | tee "$LOG_FILE"


# Run from the project root, e.g.:
#   ./src/scripts/orcd/run_main.sh --problem vanderpol --param=100 --profile fast --seed 123 --no-collocation
# See `julia --project=. src/run_main.jl --help` for all flags.
