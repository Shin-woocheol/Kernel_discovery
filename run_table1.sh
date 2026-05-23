#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Reproduce the "Ours" row of Table 1.
# Runs LLM-driven kernel discovery on the 5 paper benchmarks for 4 seeds each
# with the exact configuration reported in the paper.
#
# Usage:
#   bash run_table1.sh                   # all 5 benchmarks × seeds {0,1,2,3} on GPU 0
#   GPUS="0,1" bash run_table1.sh        # split jobs across GPU 0 and 1
#   BENCHMARKS="SVM_388" SEEDS="0" bash run_table1.sh   # single cell quick test
# ---------------------------------------------------------------------------
set -euo pipefail

# --- Activate conda env (default: `bo` from setup.sh) ---
ENV_NAME="${ENV_NAME:-bo}"
if [[ -z "${CONDA_DEFAULT_ENV:-}" || "${CONDA_DEFAULT_ENV}" != "${ENV_NAME}" ]]; then
    if ! command -v conda >/dev/null 2>&1; then
        echo "[ERROR] conda not found. Run setup.sh first." >&2
        exit 1
    fi
    CONDA_BASE="$(conda info --base)"
    # shellcheck disable=SC1091
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate "${ENV_NAME}"
fi

export BENCHMARKS="${BENCHMARKS:-SVM_388 Mopta08 Lasso-DNA Rover Humanoid}"
export SEEDS="${SEEDS:-0 1 2 3}"

# Paper config (kept identical to what produced Table 1; overridable via env vars)
export N_ITER="${N_ITER:-1000}"
export BATCH_SIZE="${BATCH_SIZE:-20}"
export N_INIT="${N_INIT:-20}"
export KERNEL_GENS="${KERNEL_GENS:-1}"
export KERNEL_MUTATIONS="${KERNEL_MUTATIONS:-1}"
export KERNEL_COMPOSITIONS="${KERNEL_COMPOSITIONS:-1}"
export POPULATION_SIZE="${POPULATION_SIZE:-10}"
export MAX_WORKERS="${MAX_WORKERS:-5}"
export CODEGEN_TEMP="${CODEGEN_TEMP:-0.3}"
export MAX_EVAL_TIME="${MAX_EVAL_TIME:-60.0}"
export MAX_RETRY="${MAX_RETRY:-0}"
export MODEL_SELECTION="${MODEL_SELECTION:-crps}"

GPUS="${GPUS:-0}"
MAX_JOBS_PER_GPU="${MAX_JOBS_PER_GPU:-1}"
MODEL_NAME="${MODEL_NAME:-gpt-4o}"

exec bash run_llm_bo_sweep.sh "$MAX_JOBS_PER_GPU" "$GPUS" "$MODEL_NAME"
