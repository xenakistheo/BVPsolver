#!/usr/bin/env bash

#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --output=logs/ablation5_%A_%a.out
#SBATCH --error=logs/ablation5_%A_%a.err
#SBATCH --mail-user=txenakis@mit.edu
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

# ── Sweep definition (Ablation nr. 5 — sensitivity vs. stiffness; see
# arXiv:2508.01519 on the vanishing-gradient barrier for stiff neural ODEs) ──
# vanderpol + rober, derivmatch pretraining throughout, comparing
# stiff vs GELU-scaled models x collocation vs shooting training,
# every run with --track-sensitivity, x 20 seeds (500-519).
# 2 problems * 2 models * 2 trainings * 20 seeds = 160 tasks.
PROBLEMS=(vanderpol rober)
MODELS=(stiff GELU-scaled)
TRAININGS=(collocation shooting)
SEEDS=(500 501 502 503 504 505 506 507 508 509 510 511 512 513 514 515 516 517 518 519)

TASKS=()
for problem in "${PROBLEMS[@]}"; do
    for model in "${MODELS[@]}"; do
        for train in "${TRAININGS[@]}"; do
            for seed in "${SEEDS[@]}"; do
                TASKS+=("--problem $problem --model $model --profile fast --seed $seed --pretraining derivmatch --training $train --track-sensitivity")
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
OUT_FILE="$LOG_DIR/ablation5_${SLURM_ARRAY_JOB_ID}_${TASK_ID}.out"
ERR_FILE="$LOG_DIR/ablation5_${SLURM_ARRAY_JOB_ID}_${TASK_ID}.err"

echo "Task $TASK_ID/${#TASKS[@]}: ${TASKS[$TASK_ID]}"
echo "Logging to $LOG_FILE"
julia --project=. src/run_main.jl --computer orcd --out-file "$OUT_FILE" --err-file "$ERR_FILE" \
    "${ARGS[@]}" 2>&1 | tee "$LOG_FILE"

echo "--------------------------------------"
echo "Done: $(date)"

# ── Usage ────────────────────────────────────────────────────────────────
# Total tasks = 160. At --mem=8G, this account's QOS ceiling (MaxTRESPU
# mem=386G) caps concurrent running tasks at 386G / 8G ≈ 48, so %48
# throttles to that (same reasoning as run_main_array.sh).
#
# If a chunk of tasks times out or OOMs at 8G/3h, collect the failed task
# IDs and resubmit just those with `sbatch --array=<id1>,<id2>,...%N` and a
# higher --mem/--time.
#
# Run from the project root:
#   sbatch --array=0-159%48 src/scripts/orcd/run_ablation5_sensitivity_array.sh
