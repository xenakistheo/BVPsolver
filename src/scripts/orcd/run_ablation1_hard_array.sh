#!/usr/bin/env bash

#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=07:00:00
#SBATCH --output=logs/ablation1_hard_%A_%a.out
#SBATCH --error=logs/ablation1_hard_%A_%a.err
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

# ── Sweep definition ─────────────────────────────────────────────────────────
# Reruns of the 21 run_ablation1_array.sh (job 22227913) tasks that were
# killed by the 3h time limit and are NOT pollu+stiff/mlp+shooting (that
# combo has never completed in this repo's history, even at 20G/10h in
# run_main_array_heavy.sh -- see run_ablation1_stiffmlp_pollu commands
# below instead). All 21 tasks here failed only on wall-clock: their
# successful siblings in the same job completed in well under 3h (hires
# GELU-scaled max 2.71h, orego GELU-scaled max 2.89h, pollu GELU-scaled
# historical max 3.68h), so 7h gives ample margin without guessing at a
# genuinely pathological runtime.
TASKS=(
    # pollu, GELU-scaled, shooting -- seeds 100-109 (all 10 failed at 3h;
    # historical completions for this combo run 1.0-3.7h)
    "--problem pollu --model GELU-scaled --profile fast --seed 100 --pretraining derivmatch --training shooting"
    "--problem pollu --model GELU-scaled --profile fast --seed 101 --pretraining derivmatch --training shooting"
    "--problem pollu --model GELU-scaled --profile fast --seed 102 --pretraining derivmatch --training shooting"
    "--problem pollu --model GELU-scaled --profile fast --seed 103 --pretraining derivmatch --training shooting"
    "--problem pollu --model GELU-scaled --profile fast --seed 104 --pretraining derivmatch --training shooting"
    "--problem pollu --model GELU-scaled --profile fast --seed 105 --pretraining derivmatch --training shooting"
    "--problem pollu --model GELU-scaled --profile fast --seed 106 --pretraining derivmatch --training shooting"
    "--problem pollu --model GELU-scaled --profile fast --seed 107 --pretraining derivmatch --training shooting"
    "--problem pollu --model GELU-scaled --profile fast --seed 108 --pretraining derivmatch --training shooting"
    "--problem pollu --model GELU-scaled --profile fast --seed 109 --pretraining derivmatch --training shooting"
    # vanderpol, stiff, shooting -- seeds 104, 107
    "--problem vanderpol --model stiff --profile fast --seed 104 --pretraining derivmatch --training shooting"
    "--problem vanderpol --model stiff --profile fast --seed 107 --pretraining derivmatch --training shooting"
    # vanderpol, mlp, shooting -- seeds 101, 106, 107, 108
    "--problem vanderpol --model mlp --profile fast --seed 101 --pretraining derivmatch --training shooting"
    "--problem vanderpol --model mlp --profile fast --seed 106 --pretraining derivmatch --training shooting"
    "--problem vanderpol --model mlp --profile fast --seed 107 --pretraining derivmatch --training shooting"
    "--problem vanderpol --model mlp --profile fast --seed 108 --pretraining derivmatch --training shooting"
    # vanderpol, GELU-scaled, shooting -- seed 106
    "--problem vanderpol --model GELU-scaled --profile fast --seed 106 --pretraining derivmatch --training shooting"
    # hires, stiff, shooting -- seed 105
    "--problem hires --model stiff --profile fast --seed 105 --pretraining derivmatch --training shooting"
    # orego, mlp, shooting -- seeds 102, 109
    "--problem orego --model mlp --profile fast --seed 102 --pretraining derivmatch --training shooting"
    "--problem orego --model mlp --profile fast --seed 109 --pretraining derivmatch --training shooting"
    # orego, GELU-scaled, shooting -- seed 109
    "--problem orego --model GELU-scaled --profile fast --seed 109 --pretraining derivmatch --training shooting"
)

TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is unset; submit this script with sbatch --array=...}"
if (( TASK_ID < 0 || TASK_ID >= ${#TASKS[@]} )); then
    echo "TASK_ID=$TASK_ID out of range (0-$(( ${#TASKS[@]} - 1 )))" >&2
    exit 1
fi
read -r -a ARGS <<< "${TASKS[$TASK_ID]}"

LOG_FILE="$LOG_DIR/run_main_${SLURM_ARRAY_JOB_ID}_${TASK_ID}.log"
# Mirrors the %A_%a pattern in the #SBATCH --output/--error directives above,
# since SLURM doesn't expose the resolved filenames via env var.
OUT_FILE="$LOG_DIR/ablation1_hard_${SLURM_ARRAY_JOB_ID}_${TASK_ID}.out"
ERR_FILE="$LOG_DIR/ablation1_hard_${SLURM_ARRAY_JOB_ID}_${TASK_ID}.err"

echo "Task $TASK_ID/${#TASKS[@]}: ${TASKS[$TASK_ID]}"
echo "Logging to $LOG_FILE"
julia --project=. src/run_main.jl --computer orcd --out-file "$OUT_FILE" --err-file "$ERR_FILE" \
    "${ARGS[@]}" 2>&1 | tee "$LOG_FILE"

echo "--------------------------------------"
echo "Done: $(date)"

# ── Usage ────────────────────────────────────────────────────────────────
# Total tasks = 21. At --mem=8G, this account's QOS ceiling (MaxTRESPU
# mem=386G) caps concurrent running tasks well above 21, so no %N
# throttle is needed here.
#
# Run from the project root:
#   sbatch --array=0-20 src/scripts/orcd/run_ablation1_hard_array.sh
