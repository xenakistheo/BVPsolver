#!/usr/bin/env bash

#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --time=10:00:00
#SBATCH --output=logs/array_heavy_%A_%a.out
#SBATCH --error=logs/array_heavy_%A_%a.err
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

# ── Sweep definition ────────────────────────────────────────────────────────
# Reruns of the resource-bound failures from run_main_array.sh (jobs 20140853,
# 20198490): `shapovalova` training consistently timed out or OOM'd at
# 8G/3h across every problem and model, so every shapovalova combo is rerun
# here with more memory and time. A handful of specific `shooting` (problem,
# model, seed) combos also timed out and are added explicitly below, since
# `shooting` otherwise finishes fine at the default 8G/3h.
PROBLEMS=(pollu rober vanderpol hires orego davis-skodje)
MODELS=(stiff mlp GELU-scaled)
# "pretraining:training" pairs
CONFIGS=(
    "none:shapovalova"
    "derivmatch:shapovalova"
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

# Individually timed-out `shooting` tasks (not a full problem/model combo).
EXTRA_SHOOTING=(
    "--problem pollu --model stiff --profile fast --seed 11 --pretraining none --training shooting"
    "--problem pollu --model stiff --profile fast --seed 12 --pretraining none --training shooting"
    "--problem pollu --model stiff --profile fast --seed 13 --pretraining none --training shooting"
    "--problem pollu --model stiff --profile fast --seed 14 --pretraining none --training shooting"
    "--problem pollu --model stiff --profile fast --seed 15 --pretraining none --training shooting"
    "--problem pollu --model stiff --profile fast --seed 11 --pretraining derivmatch --training shooting"
    "--problem pollu --model stiff --profile fast --seed 12 --pretraining derivmatch --training shooting"
    "--problem pollu --model stiff --profile fast --seed 13 --pretraining derivmatch --training shooting"
    "--problem pollu --model stiff --profile fast --seed 14 --pretraining derivmatch --training shooting"
    "--problem pollu --model stiff --profile fast --seed 15 --pretraining derivmatch --training shooting"
    "--problem pollu --model mlp --profile fast --seed 11 --pretraining none --training shooting"
    "--problem pollu --model mlp --profile fast --seed 12 --pretraining none --training shooting"
    "--problem pollu --model mlp --profile fast --seed 13 --pretraining none --training shooting"
    "--problem pollu --model mlp --profile fast --seed 14 --pretraining none --training shooting"
    "--problem pollu --model mlp --profile fast --seed 15 --pretraining none --training shooting"
    "--problem pollu --model mlp --profile fast --seed 11 --pretraining derivmatch --training shooting"
    "--problem pollu --model mlp --profile fast --seed 12 --pretraining derivmatch --training shooting"
    "--problem pollu --model mlp --profile fast --seed 13 --pretraining derivmatch --training shooting"
    "--problem pollu --model mlp --profile fast --seed 14 --pretraining derivmatch --training shooting"
    "--problem pollu --model mlp --profile fast --seed 15 --pretraining derivmatch --training shooting"
    "--problem pollu --model GELU-scaled --profile fast --seed 14 --pretraining derivmatch --training shooting"
    "--problem pollu --model GELU-scaled --profile fast --seed 15 --pretraining derivmatch --training shooting"
    "--problem vanderpol --model stiff --profile fast --seed 15 --pretraining derivmatch --training shooting"
    "--problem vanderpol --model mlp --profile fast --seed 11 --pretraining derivmatch --training shooting"
    "--problem vanderpol --model mlp --profile fast --seed 12 --pretraining derivmatch --training shooting"
    "--problem vanderpol --model mlp --profile fast --seed 14 --pretraining derivmatch --training shooting"
    "--problem vanderpol --model GELU-scaled --profile fast --seed 15 --pretraining derivmatch --training shooting"
    "--problem hires --model stiff --profile fast --seed 14 --pretraining derivmatch --training shooting"
    "--problem hires --model mlp --profile fast --seed 12 --pretraining none --training shooting"
    "--problem orego --model stiff --profile fast --seed 11 --pretraining derivmatch --training shooting"
    "--problem orego --model stiff --profile fast --seed 13 --pretraining derivmatch --training shooting"
    "--problem orego --model stiff --profile fast --seed 14 --pretraining derivmatch --training shooting"
    "--problem orego --model mlp --profile fast --seed 15 --pretraining derivmatch --training shooting"
    "--problem orego --model GELU-scaled --profile fast --seed 12 --pretraining derivmatch --training shooting"
)
TASKS+=("${EXTRA_SHOOTING[@]}")

TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is unset; submit this script with sbatch --array=...}"
if (( TASK_ID < 0 || TASK_ID >= ${#TASKS[@]} )); then
    echo "TASK_ID=$TASK_ID out of range (0-$(( ${#TASKS[@]} - 1 )))" >&2
    exit 1
fi
read -r -a ARGS <<< "${TASKS[$TASK_ID]}"

LOG_FILE="$LOG_DIR/run_main_${SLURM_ARRAY_JOB_ID}_${TASK_ID}.log"
# Mirrors the %A_%a pattern in the #SBATCH --output/--error directives above,
# since SLURM doesn't expose the resolved filenames via env var.
OUT_FILE="$LOG_DIR/array_heavy_${SLURM_ARRAY_JOB_ID}_${TASK_ID}.out"
ERR_FILE="$LOG_DIR/array_heavy_${SLURM_ARRAY_JOB_ID}_${TASK_ID}.err"

echo "Task $TASK_ID/${#TASKS[@]}: ${TASKS[$TASK_ID]}"
echo "Logging to $LOG_FILE"
julia --project=. src/run_main.jl --computer orcd --out-file "$OUT_FILE" --err-file "$ERR_FILE" \
    "${ARGS[@]}" 2>&1 | tee "$LOG_FILE"

echo "--------------------------------------"
echo "Done: $(date)"

# ── Usage ────────────────────────────────────────────────────────────────
# Total tasks = 214 (180 shapovalova combos + 34 individual shooting reruns).
# At --mem=20G, this account's QOS ceiling (MaxTRESPU mem=386G) caps
# concurrent running tasks at 386G / 20G ≈ 19, so %19 throttles to that.
#
# Run from the project root:
#   sbatch --array=0-213%19 src/scripts/orcd/run_main_array_heavy.sh
