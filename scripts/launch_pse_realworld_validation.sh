#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${CONFIG_PATH:-${ROOT_DIR}/pse_realworld_vega_sr.yaml}"
METHOD="${METHOD:-pse}"
DATASET="${DATASET:?set DATASET to emps or roughpipe}"
GPU_INDEX="${GPU_INDEX:?set GPU_INDEX to the physical GPU index}"
PYTHON_BIN="${PYTHON_BIN:?set PYTHON_BIN to the method environment python}"
RESULTS_DIR="${RESULTS_DIR:-${ROOT_DIR}/results/pse_realworld_validation}"
CASE_BUDGET_SEC="${CASE_BUDGET_SEC:-90}"
TIMEOUT_GRACE_SEC="${TIMEOUT_GRACE_SEC:-30}"
HARD_TIMEOUT_SEC="${HARD_TIMEOUT_SEC:-600}"
SEEDS="${SEEDS:-0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19}"

CASE_DIR="${RESULTS_DIR}/cases/${METHOD}/${DATASET}"
LOG_DIR="${RESULTS_DIR}/logs/${METHOD}/${DATASET}"
mkdir -p "${CASE_DIR}" "${LOG_DIR}"
export NO_PROXY="127.0.0.1,localhost,${NO_PROXY:-}"
export no_proxy="127.0.0.1,localhost,${no_proxy:-}"

for seed in ${SEEDS}; do
  output_path="${CASE_DIR}/seed_${seed}.json"
  log_path="${LOG_DIR}/seed_${seed}.log"
  if [[ -f "${output_path}" ]] && "${PYTHON_BIN}" - "${output_path}" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as handle:
    result = json.load(handle)
raise SystemExit(0 if result.get("status") == "ok" else 1)
PY
  then
    echo "[SKIP] ${METHOD} ${DATASET} seed=${seed}"
    continue
  fi
  echo "[RUN] ${METHOD} ${DATASET} seed=${seed} gpu=${GPU_INDEX}"
  set +e
  # The upstream PSE timer limits enumeration, but constant fitting happens
  # after that timer.  Keep a separate watchdog so slow post-processing does
  # not erase an otherwise valid 90-second search result.
  timeout --signal=TERM --kill-after=10s "${HARD_TIMEOUT_SEC}s" \
    "${PYTHON_BIN}" "${ROOT_DIR}/scripts/run_pse_realworld_case.py" \
      --config "${CONFIG_PATH}" \
      --method "${METHOD}" \
      --dataset "${DATASET}" \
      --seed "${seed}" \
      --gpu-index "${GPU_INDEX}" \
      --budget-sec "${CASE_BUDGET_SEC}" \
      --output "${output_path}" \
      2>&1 | tee "${log_path}"
  case_status=${PIPESTATUS[0]}
  set -e
  if [[ ${case_status} -ne 0 ]]; then
    "${PYTHON_BIN}" - "${output_path}" "${METHOD}" "${DATASET}" "${seed}" "${CASE_BUDGET_SEC}" "${case_status}" <<'PY'
import json
import os
import sys
import time
path, method, dataset, seed, budget, exit_code = sys.argv[1:]
existing = {}
if os.path.exists(path):
    try:
        with open(path, encoding="utf-8") as handle:
            existing = json.load(handle)
    except Exception:
        existing = {}
if existing.get("status") != "ok":
    existing.update({
        "method": method,
        "dataset": dataset,
        "repeat_seed": int(seed),
        "search_budget_sec": float(budget),
        "status": "timed_out" if int(exit_code) == 124 else "error",
        "timed_out": int(exit_code) == 124,
        "process_exit_code": int(exit_code),
        "finished_at_unix": time.time(),
    })
    temporary = path + ".launcher.tmp"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(existing, handle, ensure_ascii=False, indent=2)
    os.replace(temporary, path)
PY
    echo "[FAIL] ${METHOD} ${DATASET} seed=${seed} exit=${case_status}" | tee -a "${log_path}"
  else
    echo "[DONE] ${METHOD} ${DATASET} seed=${seed}" | tee -a "${log_path}"
  fi
done
