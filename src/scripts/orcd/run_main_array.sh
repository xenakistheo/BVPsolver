#!/usr/bin/env bash

#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs/array_%A_%a.out
#SBATCH --error=logs/array_%A_%a.err
#SBATCH --mail-user=theodoros.xenakis.03@gmail.com
#SBATCH --mail-type=END,FAIL

set -euo pipefail

module load julia/1.12.6

# SLURM copies the submitted script to a spool directory before running it,
# so $BASH_SOURCE doesn't point at the repo. Use the submission directory
# instead (this script must be submitted via `sbatch` from the repo root).
REPO_ROOT="$SLURM_SUBMIT_DIR"
cd "$REPO_ROOT"

LOG_DIR="$REPO_ROOT/logs"
mkdir -p "$LOG_DIR"

# ── Sweep definition ────────────────────────────────────────────────────────
# 6 problems (brusselator excluded, run separately on another cluster) x
# 3 models x 7 pretraining/training combinations x 5 seeds = 630 tasks.
PROBLEMS=(pollu rober vanderpol hires orego davis-skodje)
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
echo "Task $TASK_ID/${#TASKS[@]}: ${TASKS[$TASK_ID]}"
echo "Logging to $LOG_FILE"
julia --project=. src/run_main.jl --computer orcd "${ARGS[@]}" 2>&1 | tee "$LOG_FILE"

echo "--------------------------------------"
echo "Done: $(date)"

# ── Usage ────────────────────────────────────────────────────────────────
# Total tasks = 630, but this account's QOS caps MaxSubmitPU at 448 (queued +
# running jobs at once), so the array must be submitted in two batches. %48
# throttles concurrent running tasks to match the mem-bound QOS ceiling
# (MaxTRESPU cpu=96,mem=386G -> 386G / 8G per task = 48).
#
# Run from the project root:
#   sbatch --array=0-314%48   src/scripts/orcd/run_main_array.sh
#   sbatch --array=315-629%48 src/scripts/orcd/run_main_array.sh
#
# Submit the second batch once the first has drained enough to fit under
# MaxSubmitPU (check with `squeue -u $USER | wc -l`).
