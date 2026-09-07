#!/usr/bin/env bash

#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --output=logs/ablation1_%A_%a.out
#SBATCH --error=logs/ablation1_%A_%a.err
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

# ── Sweep definition (new_runs.md, Ablation nr. 1) ──────────────────────────
# All 6 problems x 3 models x 2 training regimes (derivmatch pretraining
# followed by either shooting or collocation) x 10 seeds (20-29).
# 6 * 3 * 2 * 10 = 360 tasks.
PROBLEMS=(pollu rober vanderpol hires orego davis-skodje)
MODELS=(stiff mlp GELU-scaled)
# "pretraining:training" pairs
CONFIGS=(
    "derivmatch:shooting"
    "derivmatch:collocation"
)
SEEDS=(20 21 22 23 24 25 26 27 28 29)

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
OUT_FILE="$LOG_DIR/ablation1_${SLURM_ARRAY_JOB_ID}_${TASK_ID}.out"
ERR_FILE="$LOG_DIR/ablation1_${SLURM_ARRAY_JOB_ID}_${TASK_ID}.err"

echo "Task $TASK_ID/${#TASKS[@]}: ${TASKS[$TASK_ID]}"
echo "Logging to $LOG_FILE"
julia --project=. src/run_main.jl --computer orcd --out-file "$OUT_FILE" --err-file "$ERR_FILE" \
    "${ARGS[@]}" 2>&1 | tee "$LOG_FILE"

echo "--------------------------------------"
echo "Done: $(date)"

# ── Usage ────────────────────────────────────────────────────────────────
# Total tasks = 360. At --mem=8G, this account's QOS ceiling (MaxTRESPU
# mem=386G) caps concurrent running tasks at 386G / 8G ≈ 48, so %48
# throttles to that (same reasoning as run_main_array.sh).
#
# If a chunk of tasks times out or OOMs at 8G/3h (as happened previously
# for shooting/shapovalova on some problems, see run_main_array_heavy.sh),
# collect the failed task IDs and resubmit just those with
# `sbatch --array=<id1>,<id2>,...%N` and a higher --mem/--time.
#
# Run from the project root:
#   sbatch --array=0-359%48 src/scripts/orcd/run_ablation1_array.sh
