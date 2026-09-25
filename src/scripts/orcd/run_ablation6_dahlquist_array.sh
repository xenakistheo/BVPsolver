#!/usr/bin/env bash

#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/ablation6_%A_%a.out
#SBATCH --error=logs/ablation6_%A_%a.err
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

# ── Sweep definition (Ablation nr. 6 — Dahlquist dy/dt=lambda*y trajectory
# parameter sensitivity vs. stiffness; see sensitivity_ablation.md) ──────────
# dahlquist only, derivmatch pretraining throughout, comparing
# stiff vs GELU-scaled models x shooting vs derivmatch-only training,
# x 10 seeds (800-809), over a log-spaced negative-lambda sweep.
# "shooting" = derivmatch pretraining + shooting refinement
# (--pretraining derivmatch --training shooting); "derivmatch" = derivmatch
# pretraining alone, no further stage (--pretraining derivmatch --training
# none) -- both warm-start with derivmatch, matching every other ablation here
# (see run_ablation1-5*.sh) rather than shooting from a random init.
# trajectory_parameter_sensitivity_rms is computed unconditionally by
# run_main.jl for every run now, so no --track-sensitivity flag is needed.
# 5 lambdas * 2 models * 2 methods * 10 seeds = 200 tasks.
LAMBDAS=(-1 -10 -100 -1000 -10000)
MODELS=(stiff GELU-scaled)
METHODS=(shooting derivmatch)
SEEDS=(800 801 802 803 804 805 806 807 808 809)

TASKS=()
for lambda in "${LAMBDAS[@]}"; do
    for model in "${MODELS[@]}"; do
        for method in "${METHODS[@]}"; do
            training="none"
            [[ "$method" == "shooting" ]] && training="shooting"
            for seed in "${SEEDS[@]}"; do
                TASKS+=("--problem dahlquist --param $lambda --model $model --profile fast --seed $seed --pretraining derivmatch --training $training")
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
OUT_FILE="$LOG_DIR/ablation6_${SLURM_ARRAY_JOB_ID}_${TASK_ID}.out"
ERR_FILE="$LOG_DIR/ablation6_${SLURM_ARRAY_JOB_ID}_${TASK_ID}.err"

echo "Task $TASK_ID/${#TASKS[@]}: ${TASKS[$TASK_ID]}"
echo "Logging to $LOG_FILE"
julia --project=. src/run_main.jl --computer orcd --out-file "$OUT_FILE" --err-file "$ERR_FILE" \
    "${ARGS[@]}" 2>&1 | tee "$LOG_FILE"

echo "--------------------------------------"
echo "Done: $(date)"

# ── Usage ────────────────────────────────────────────────────────────────
# Total tasks = 200. At --mem=8G, this account's QOS ceiling (MaxTRESPU
# mem=386G) caps concurrent running tasks at 386G / 8G ≈ 48, so %48
# throttles to that (same reasoning as run_main_array.sh). Dahlquist is a
# trivial 1D linear problem (hidden=8, depth=2), so --time=00:30:00 is
# generous headroom over local smoke-test timings (~13s derivmatch, well
# under a minute for shooting at the default 1000 iters); raise it if a
# chunk of tasks times out.
#
# If a chunk of tasks times out or OOMs, collect the failed task IDs and
# resubmit just those with `sbatch --array=<id1>,<id2>,...%N` and a higher
# --mem/--time.
#
# Run from the project root:
#   sbatch --array=0-199%48 src/scripts/orcd/run_ablation6_dahlquist_array.sh
