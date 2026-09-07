#!/usr/bin/env bash

#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --output=logs/ablation4_%A_%a.out
#SBATCH --error=logs/ablation4_%A_%a.err
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

# ── Sweep definition (new_runs.md, Ablation nr. 4) ──────────────────────────
# vanderpol only, sweeping mu (--param) over 7 values, comparing
# stiff+derivmatch+collocation vs GELU-scaled+derivmatch+shooting,
# x 10 seeds (40-49).
# 2 * 7 * 10 = 140 tasks.
PROBLEM="vanderpol"
MUS=(1 5 10 50 100 150 200)
# "model:pretraining:training" triples
CONFIGS=(
    "stiff:derivmatch:collocation"
    "GELU-scaled:derivmatch:shooting"
)
SEEDS=(40 41 42 43 44 45 46 47 48 49)

TASKS=()
for mu in "${MUS[@]}"; do
    for cfg in "${CONFIGS[@]}"; do
        model="${cfg%%:*}"
        rest="${cfg#*:}"
        pretrain="${rest%%:*}"
        train="${rest##*:}"
        for seed in "${SEEDS[@]}"; do
            TASKS+=("--problem $PROBLEM --param $mu --model $model --profile fast --seed $seed --pretraining $pretrain --training $train")
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
OUT_FILE="$LOG_DIR/ablation4_${SLURM_ARRAY_JOB_ID}_${TASK_ID}.out"
ERR_FILE="$LOG_DIR/ablation4_${SLURM_ARRAY_JOB_ID}_${TASK_ID}.err"

echo "Task $TASK_ID/${#TASKS[@]}: ${TASKS[$TASK_ID]}"
echo "Logging to $LOG_FILE"
julia --project=. src/run_main.jl --computer orcd --out-file "$OUT_FILE" --err-file "$ERR_FILE" \
    "${ARGS[@]}" 2>&1 | tee "$LOG_FILE"

echo "--------------------------------------"
echo "Done: $(date)"

# ── Usage ────────────────────────────────────────────────────────────────
# Total tasks = 140. At --mem=8G, this account's QOS ceiling (MaxTRESPU
# mem=386G) caps concurrent running tasks at 386G / 8G ≈ 48, so %48
# throttles to that (same reasoning as run_main_array.sh).
#
# High-mu (stiffer) van der Pol runs may run longer than the default
# 3h/8G budget. If a chunk of tasks times out or OOMs, collect the failed
# task IDs and resubmit just those with `sbatch --array=<id1>,<id2>,...%N`
# and a higher --mem/--time.
#
# Run from the project root:
#   sbatch --array=0-139%48 src/scripts/orcd/run_ablation4_vdp_array.sh
