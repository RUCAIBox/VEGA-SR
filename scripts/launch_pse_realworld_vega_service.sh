#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VLLM_BIN="${VLLM_BIN:-vllm}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-VL-32B-Instruct}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3-VL-32B-Instruct}"
GPU_DEVICES="${GPU_DEVICES:-0,1}"
PORT="${PORT:-18001}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.72}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/results/pse_realworld_validation/logs/vega_service}"

mkdir -p "${LOG_DIR}"
export CUDA_VISIBLE_DEVICES="${GPU_DEVICES}"
export HF_HOME="${HF_HOME:-${ROOT_DIR}/.cache/huggingface}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

exec "${VLLM_BIN}" serve "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --host 127.0.0.1 \
  --port "${PORT}" \
  --tensor-parallel-size 2 \
  --dtype bfloat16 \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --max-num-seqs 8 \
  --limit-mm-per-prompt '{"image":16,"video":0}' \
  --trust-remote-code \
  2>&1 | tee "${LOG_DIR}/vllm_qwen3vl32b.log"
