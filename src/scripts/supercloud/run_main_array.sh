#!/usr/bin/env bash

#SBATCH --partition=xeon-p8
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=100:00:00
#SBATCH --output=logs/array_%A_%a.out
#SBATCH --error=logs/array_%A_%a.err
#SBATCH --mail-user=txenakis@mit.edu
#SBATCH --mail-type=END,FAIL

source /etc/profile

# Self-installed via juliaup (module system only offers julia/1.11.3, but this
# project's Manifest.toml requires >=1.12 for LinearAlgebra compat).
export PATH="$HOME/.juliaup/bin:$PATH"

set -euo pipefail

# SLURM copies the submitted script to a spool directory before running it,
# so $BASH_SOURCE doesn't point at the repo. Use the submission directory
# instead (this script must be submitted via `sbatch` from the repo root).
REPO_ROOT="$SLURM_SUBMIT_DIR"
cd "$REPO_ROOT"

LOG_DIR="$REPO_ROOT/logs"
mkdir -p "$LOG_DIR"

# ── Sweep definition ────────────────────────────────────────────────────────
# brusselator only (the other 6 problems run on ORCD, see
# src/scripts/orcd/run_main_array.sh) x 3 models x 7 pretraining/training
# combinations x 5 seeds = 105 tasks.
PROBLEMS=(brusselator)
MODELS=(stiff mlp GELU-scaled)
# "pretraining:training" pairs
CONFIGS=(
    "none:shooting"
    "derivmatch:shooting"
    "none:shapovalova"
    "derivmatch:shapovalova"
    "derivmatch:none"
    "none:collocation"
    "derivmatch:collocation"
)
SEEDS=(11 12 13 14 15)

TASKS=()
for problem in "${PROBLEMS[@]}"; do
    for model in "${MODELS[@]}"; do
        for cfg in "${CONFIGS[@]}"; do
            pretrain="${cfg%%:*}"
            train="${cfg##*:}"
            for seed in "${SEEDS[@]}"; do
                TASKS+=("--problem $problem --model $model --profile fast --seed $seed --pretraining $pretrain --training $train")
            done
        done
    done
done

TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is unset; submit this script with sbatch --array=...}"
if (( TASK_ID < 0 || TASK_ID >= ${#TASKS[@]} )); then
    echo "TASK_ID=$TASK_ID out of range (0-$(( ${#TASKS[@]} - 1 )))" >&2
    exit 1
fi
read -r -a ARGS <<< "${TASKS[$TASK_ID]}"

LOG_FILE="$LOG_DIR/run_main_${SLURM_ARRAY_JOB_ID}_${TASK_ID}.log"
# Mirrors the %A_%a pattern in the #SBATCH --output/--error directives above,
# since SLURM doesn't expose the resolved filenames via env var.
OUT_FILE="$LOG_DIR/array_${SLURM_ARRAY_JOB_ID}_${TASK_ID}.out"
ERR_FILE="$LOG_DIR/array_${SLURM_ARRAY_JOB_ID}_${TASK_ID}.err"

echo "Task $TASK_ID/${#TASKS[@]}: ${TASKS[$TASK_ID]}"
echo "Logging to $LOG_FILE"
julia --project=. src/run_main.jl --computer supercloud --out-file "$OUT_FILE" --err-file "$ERR_FILE" \
    "${ARGS[@]}" 2>&1 | tee "$LOG_FILE"

echo "--------------------------------------"
echo "Done: $(date)"

# ── Usage ────────────────────────────────────────────────────────────────
# --cpus-per-task/--mem reuse the values measured on ORCD (~1 core, ~3.3GB
# peak for the heaviest tested config, `pollu`). brusselator itself was never
# profiled -- it's the largest config (hidden=32, depth=3, dm_iters=30000,
# time_dependent=true), comparable in size to pollu but untested. Before
# trusting these numbers, do a single dry-run task and check `jobstats`:
#   sbatch --array=0-0 src/scripts/supercloud/run_main_array.sh
#
# Total tasks = 105. Checked this account's xeon-p8 association: MaxJobs=240,
# MaxSubmit=240 (per `sacctmgr show assoc ... format=Cluster,Partition,...`),
# and QOS "normal" has no additional MaxTRESPU/GrpTRES cap. 105 < 240, so no
# batching or throttle is needed -- submit the whole array in one shot.
# xeon-p8's MaxTime is 4-04:00:00 (=100h), so --time=100:00:00 above sits
# right at the partition ceiling.
#
# Run from the project root:
#   sbatch --array=0-104 src/scripts/supercloud/run_main_array.sh
