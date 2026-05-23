#!/usr/bin/env bash
set -euo pipefail


REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# --- Positional args ---
MAX_JOBS_PER_GPU="${1:-2}"
GPU_RANGE="${2:-0}"
MODEL_NAME="${3:-gpt-4o}"

# --- BO params ---
N_ITER="${N_ITER:-1000}"
N_INIT="${N_INIT:-20}"
BATCH_SIZE="${BATCH_SIZE:-20}"
KERNEL_GENS="${KERNEL_GENS:-1}"
KERNEL_MUTATIONS="${KERNEL_MUTATIONS:-2}"
KERNEL_COMPOSITIONS="${KERNEL_COMPOSITIONS:-2}"
POPULATION_SIZE="${POPULATION_SIZE:-10}"
MAX_RETRY="${MAX_RETRY:-1}"
CODEGEN_TEMP="${CODEGEN_TEMP:-0.3}"
MODEL_SELECTION="${MODEL_SELECTION:-crps}"
SOFTMAX_TEMPERATURE="${SOFTMAX_TEMPERATURE:-0.01}"
SIMPLICITY_PROMPT="${SIMPLICITY_PROMPT:-}"
SCORE_SUBSAMPLE="${SCORE_SUBSAMPLE:-1.0}"
MAX_EVAL_TIME="${MAX_EVAL_TIME:-60.0}"
FIT_BACKEND="${FIT_BACKEND:-scipy}"
ACQF_BACKEND="${ACQF_BACKEND:-scipy}"
WANDB="${WANDB:-}"
PROMPT_HIGHDIM="${PROMPT_HIGHDIM:-on}"
PROMPT_PSD="${PROMPT_PSD:-on}"
OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/hdbo/output}"

# --- Sweep config ---
if [[ -n "${BENCHMARKS:-}" ]]; then
    read -ra BENCHMARKS <<< "$BENCHMARKS"
else
    # BENCHMARKS=("Humanoid" "Rover" "Mopta08" "Lasso-DNA" "SVM_388")
    # BENCHMARKS=("Lasso-DNA" "Mopta08" "Rover" "SVM_388")
    BENCHMARKS=("SVM_388")
fi

if [[ -n "${SEEDS:-}" ]]; then
    read -ra SEEDS <<< "$SEEDS"
else
    SEEDS=(1 1)
    # SEEDS=(0 1)
fi


MAX_WORKERS="${MAX_WORKERS:-${POPULATION_SIZE}}"

# --- GPU slot setup ---
IFS=',' read -ra GPUS <<< "$GPU_RANGE"
NUM_GPUS="${#GPUS[@]}"
TOTAL_SLOTS=$(( NUM_GPUS * MAX_JOBS_PER_GPU ))

declare -a GPU_SLOTS=()
declare -a JOB_PIDS=()
for (( i = 0; i < TOTAL_SLOTS; i++ )); do
    GPU_SLOTS[$i]="${GPUS[$((i / MAX_JOBS_PER_GPU))]}"
    JOB_PIDS[$i]=0
done

echo "======================================================================"
echo "LLM-driven kernel discovery sweep (formula-level)"
echo "  GPUs: ${GPU_RANGE} (${NUM_GPUS} GPU(s) × ${MAX_JOBS_PER_GPU} jobs = ${TOTAL_SLOTS} slots)"
echo "  Model: ${MODEL_NAME}"
echo "  Benchmarks: ${BENCHMARKS[*]}"
echo "  Seeds: ${SEEDS[*]}"
echo "  n_iter=${N_ITER} n_init=${N_INIT} batch=${BATCH_SIZE}"
echo "  kernel_gens=${KERNEL_GENS} mutations=${KERNEL_MUTATIONS} compositions=${KERNEL_COMPOSITIONS}"
echo "  population=${POPULATION_SIZE} max_workers=${MAX_WORKERS} max_retry=${MAX_RETRY}"
echo "  model_selection=${MODEL_SELECTION}"
if [[ "${MODEL_SELECTION}" == "softmax_crps" ]]; then
    echo "  softmax_temperature=${SOFTMAX_TEMPERATURE}"
fi
echo "  simplicity_prompt=${SIMPLICITY_PROMPT:-off}"
echo "  score_subsample=${SCORE_SUBSAMPLE}"
echo "  codegen_temperature=${CODEGEN_TEMP}"
echo "  Output: ${OUT_ROOT}"
echo "======================================================================"

get_slot() {
    while true; do
        for (( i = 0; i < TOTAL_SLOTS; i++ )); do
            local pid="${JOB_PIDS[$i]}"
            if [[ "$pid" -eq 0 ]]; then
                echo "$i"; return 0
            fi
            if ! kill -0 "$pid" 2>/dev/null; then
                if ! wait "$pid"; then
                    echo "WARNING: job in slot $i (GPU ${GPU_SLOTS[$i]}) failed. Continuing..." >&2
                fi
                JOB_PIDS[$i]=0
                echo "$i"; return 0
            fi
        done
        sleep 1
    done
}

# --- Main sweep ---
for benchmark in "${BENCHMARKS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        slot=$(get_slot)
        gpu="${GPU_SLOTS[$slot]}"

        echo "--------------------------------------------------------------------"
        echo "Launching: benchmark=${benchmark} seed=${seed} GPU=${gpu} (slot ${slot})"
        echo "--------------------------------------------------------------------"

        SIMPLICITY_FLAG=""
        if [[ -n "${SIMPLICITY_PROMPT}" ]]; then
            SIMPLICITY_FLAG="--simplicity_prompt"
        fi

        SOFTMAX_FLAG=""
        if [[ "${MODEL_SELECTION}" == "softmax_crps" ]]; then
            SOFTMAX_FLAG="--softmax_temperature ${SOFTMAX_TEMPERATURE}"
        fi

        WANDB_FLAG=""
        if [[ -n "${WANDB}" ]]; then
            WANDB_FLAG="--wandb"
        fi

        CUDA_VISIBLE_DEVICES="${gpu}" python -m hdbo.run_bo_llm \
            --benchmark        "${benchmark}" \
            --n_init           "${N_INIT}" \
            --n_iter           "${N_ITER}" \
            --batch_size       "${BATCH_SIZE}" \
            --acquisition      qlognei \
            --model_selection  "${MODEL_SELECTION}" \
            --model_name       "${MODEL_NAME}" \
            --kernel_gens      "${KERNEL_GENS}" \
            --kernel_mutations "${KERNEL_MUTATIONS}" \
            --kernel_compositions "${KERNEL_COMPOSITIONS}" \
            --seed             "${seed}" \
            --max_workers      "${MAX_WORKERS}" \
            --population_size  "${POPULATION_SIZE}" \
            --data_fname \
            --max_retry        "${MAX_RETRY}" \
            --codegen_temperature "${CODEGEN_TEMP}" \
            --output_dir       "${OUT_ROOT}" \
            --score_subsample  "${SCORE_SUBSAMPLE}" \
            --max_eval_time    "${MAX_EVAL_TIME}" \
            --fit_backend      "${FIT_BACKEND}" \
            --acqf_backend     "${ACQF_BACKEND}" \
            --prompt_highdim   "${PROMPT_HIGHDIM}" \
            --prompt_psd       "${PROMPT_PSD}" \
            ${SIMPLICITY_FLAG} \
            ${SOFTMAX_FLAG} \
            ${WANDB_FLAG} \
            &

        JOB_PIDS[$slot]=$!
        echo "  PID: ${JOB_PIDS[$slot]}"
    done
done

# --- Drain remaining jobs ---
echo "All jobs launched. Waiting for completion..."
FAIL_COUNT=0
for (( i = 0; i < TOTAL_SLOTS; i++ )); do
    if [[ "${JOB_PIDS[$i]}" -ne 0 ]]; then
        if ! wait "${JOB_PIDS[$i]}"; then
            echo "WARNING: job in slot $i (GPU ${GPU_SLOTS[$i]}) failed." >&2
            FAIL_COUNT=$(( FAIL_COUNT + 1 ))
        fi
        JOB_PIDS[$i]=0
    fi
done

echo "======================================================================"
echo "Done. Output: ${OUT_ROOT}/<benchmark>/<run_dir>/"
TOTAL_RUNS=$(( ${#BENCHMARKS[@]} * ${#SEEDS[@]} ))
echo "Total: ${#BENCHMARKS[@]} benchmarks × ${#SEEDS[@]} seeds = ${TOTAL_RUNS} runs"
if [[ "$FAIL_COUNT" -gt 0 ]]; then
    echo "WARNING: ${FAIL_COUNT} job(s) failed."
fi
