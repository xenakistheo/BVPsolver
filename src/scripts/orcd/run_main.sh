#!/usr/bin/env bash

#SBATCH --partition=mit_normal
# use mit_quicktest for debugging, mit_normal for main runs
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-user=theodoros.xenakis.03@gmail.com
#SBATCH --mail-type=END,FAIL

set -euo pipefail

module load julia/1.12.6

# SLURM copies the submitted script to a spool directory before running it,
# so $BASH_SOURCE doesn't point at the repo. Use the submission directory
# instead (this script must be submitted via `sbatch` from the repo root).
REPO_ROOT="$SLURM_SUBMIT_DIR"

LOG_DIR="$REPO_ROOT/logs"
mkdir -p "$LOG_DIR"

# PID keeps the filename unique across concurrent runs launched in the same second.
LOG_FILE="$LOG_DIR/run_main_$(date +%Y%m%d_%H%M%S)_$$.log"
# Mirrors the %x_%j pattern in the #SBATCH --output/--error directives above,
# since SLURM doesn't expose the resolved filenames via env var.
OUT_FILE="$LOG_DIR/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out"
ERR_FILE="$LOG_DIR/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err"

cd "$REPO_ROOT"
echo "Logging to $LOG_FILE"
julia --project=. src/run_main.jl --computer orcd --out-file "$OUT_FILE" --err-file "$ERR_FILE" \
    "$@" 2>&1 | tee "$LOG_FILE"


echo "--------------------------------------"
echo "Done: $(date)"

# Run from the project root, e.g.:
#   ./src/scripts/orcd/run_main.sh --problem vanderpol --param=100 --profile fast --seed 123 --no-collocation
# See `julia --project=. src/run_main.jl --help` for all flags.
