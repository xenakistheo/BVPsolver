#!/usr/bin/env bash
set -euo pipefail

# ── Dahlquist stiffness/sensitivity ablation ────────────────────────────────
# See sensitivity_ablation.md for the full experiment design. Runs the Cartesian
# product LAMBDAS x MODELS x METHODS x SEEDS through the existing
# src/run_main.jl entry point (no separate training pipeline) sequentially in
# this one process -- edit the arrays below to change the grid, everything
# else can be left alone.
#
# "shooting" here means derivmatch pretraining followed by shooting refinement
# (--pretraining derivmatch --training shooting); "derivmatch" means derivmatch
# pretraining with no further training stage (--pretraining derivmatch
# --training none), i.e. the network IS the derivmatch fit. Both arms warm-start
# with derivmatch rather than shooting from a random init, matching this
# repo's existing ablation convention (see src/scripts/orcd/run_ablation*.sh).
LAMBDAS=(-1 -10 -100 -1000 -10000)
MODELS=(stiff GELU-scaled)
METHODS=(shooting derivmatch)
SEEDS=(800 801 802 803 804 805 806 807 808 809)
PROBLEM="dahlquist"
PROFILE="fast"

# Resolve the repo root (three levels up from this script's directory).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

LOG_DIR="$REPO_ROOT/logs"
mkdir -p "$LOG_DIR"

n_total=$(( ${#LAMBDAS[@]} * ${#MODELS[@]} * ${#METHODS[@]} * ${#SEEDS[@]} ))
n_done=0

for lambda in "${LAMBDAS[@]}"; do
    for model in "${MODELS[@]}"; do
        for method in "${METHODS[@]}"; do
            training="none"
            [[ "$method" == "shooting" ]] && training="shooting"
            for seed in "${SEEDS[@]}"; do
                n_done=$((n_done + 1))
                echo "[$n_done/$n_total] Dahlquist | lambda=$lambda | architecture=$model | method=$method | seed=$seed"
                LOG_FILE="$LOG_DIR/dahlquist_${model}_${method}_seed${seed}_lambda${lambda}.log"
                julia --project=. src/run_main.jl \
                    --problem "$PROBLEM" --param "$lambda" --model "$model" --profile "$PROFILE" \
                    --seed "$seed" --pretraining derivmatch --training "$training" \
                    --computer local 2>&1 | tee "$LOG_FILE"
            done
        done
    done
done

echo "Dahlquist sensitivity ablation complete: $n_done/$n_total runs."

# ── Usage ────────────────────────────────────────────────────────────────
# Run from anywhere (paths are resolved relative to this script, not $PWD):
#   ./src/scripts/local/run_dahlquist_sensitivity_ablation.sh
#
# Default grid: 5 lambdas * 2 models * 2 methods * 10 seeds = 200 runs, executed
# sequentially in one process -- this can take a long time locally (each run is
# a full derivmatch fit, plus a shooting refinement for half the grid). Kill and
# rerun freely: every run writes its own data/runs/run_<id>/ with a fresh random
# id, so a partial/interrupted sweep never overwrites or corrupts prior runs --
# just be aware reruns will add duplicate (lambda, model, method, seed) entries
# rather than replacing them. Each run's trajectory_parameter_sensitivity_rms
# (plus lambda via "param", seed, model, training) lands in that run's
# data/runs/run_<id>/run_info.toml, ready for analysis/build_runs_index.py.
