#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="${RESULTS_DIR:-${ROOT_DIR}/results/pse_realworld_validation}"
VEGA_PYTHON="${VEGA_PYTHON:-${ROOT_DIR}/.venv-vega/bin/python}"
PSE_PYTHON="${PSE_PYTHON:-${ROOT_DIR}/.venv-pse/bin/python}"
EMPS_GPU_INDEX="${EMPS_GPU_INDEX:-0}"
ROUGHPIPE_GPU_INDEX="${ROUGHPIPE_GPU_INDEX:-1}"
CONTROL_DIR="${RESULTS_DIR}/control"
mkdir -p "${CONTROL_DIR}"

while tmux has-session -t pse_realworld_emps 2>/dev/null || tmux has-session -t pse_realworld_roughpipe 2>/dev/null; do
  sleep 30
done

# Re-run only missing/error/timeout rows. Successful JSON files are skipped by
# the launcher. The search budget stays 90 seconds; only the outer watchdog is
# enlarged to allow PSE's post-search constant optimization to finish.
env METHOD=pse DATASET=emps GPU_INDEX="${EMPS_GPU_INDEX}" PYTHON_BIN="${PSE_PYTHON}" RESULTS_DIR="${RESULTS_DIR}" HARD_TIMEOUT_SEC=600 \
  "${ROOT_DIR}/scripts/launch_pse_realworld_validation.sh" \
  > "${CONTROL_DIR}/pse_emps_recovery.log" 2>&1 &
emps_recovery_pid=$!
env METHOD=pse DATASET=roughpipe GPU_INDEX="${ROUGHPIPE_GPU_INDEX}" PYTHON_BIN="${PSE_PYTHON}" RESULTS_DIR="${RESULTS_DIR}" HARD_TIMEOUT_SEC=600 \
  "${ROOT_DIR}/scripts/launch_pse_realworld_validation.sh" \
  > "${CONTROL_DIR}/pse_roughpipe_recovery.log" 2>&1 &
roughpipe_recovery_pid=$!
wait "${emps_recovery_pid}"
wait "${roughpipe_recovery_pid}"

"${VEGA_PYTHON}" "${ROOT_DIR}/scripts/summarize_pse_realworld_validation.py" --results-dir "${RESULTS_DIR}" \
  > "${CONTROL_DIR}/pse_summary_after_completion.log" 2>&1

service_ready=0
if curl -fsS --max-time 5 http://127.0.0.1:18001/v1/models >/dev/null; then
  service_ready=1
elif ! tmux has-session -t pse_realworld_vega_service 2>/dev/null; then
  tmux new-session -d -s pse_realworld_vega_service \
    "cd '${ROOT_DIR}' && '${ROOT_DIR}/scripts/launch_pse_realworld_vega_service.sh'"
fi

# Loading the 32B checkpoint from shared storage can take 10-15 minutes even
# when the GPUs are healthy. Allow 30 minutes and keep checking service health.
for _ in $(seq 1 180); do
  if [[ ${service_ready} -eq 1 ]]; then
    break
  fi
  if curl -fsS --max-time 5 http://127.0.0.1:18001/v1/models >/dev/null; then
    service_ready=1
    break
  fi
  if ! tmux has-session -t pse_realworld_vega_service 2>/dev/null; then
    break
  fi
  sleep 10
done
if [[ ${service_ready} -ne 1 ]]; then
  echo "VEGA model service did not become ready" >&2
  exit 3
fi

common_env="METHOD=vega_sr PYTHON_BIN=${VEGA_PYTHON} RESULTS_DIR=${RESULTS_DIR} LLMSR_AGENT_API_BASE=http://127.0.0.1:18001/v1 LLMSR_AGENT_API_KEY=EMPTY LLMSR_AGENT_MODEL=Qwen3-VL-32B-Instruct"
tmux new-session -d -s pse_realworld_vega_emps \
  "cd '${ROOT_DIR}' && env ${common_env} DATASET=emps GPU_INDEX='${EMPS_GPU_INDEX}' '${ROOT_DIR}/scripts/launch_pse_realworld_validation.sh' 2>&1 | tee '${CONTROL_DIR}/vega_emps_driver.log'"
tmux new-session -d -s pse_realworld_vega_roughpipe \
  "cd '${ROOT_DIR}' && env ${common_env} DATASET=roughpipe GPU_INDEX='${ROUGHPIPE_GPU_INDEX}' '${ROOT_DIR}/scripts/launch_pse_realworld_validation.sh' 2>&1 | tee '${CONTROL_DIR}/vega_roughpipe_driver.log'"

while tmux has-session -t pse_realworld_vega_emps 2>/dev/null || tmux has-session -t pse_realworld_vega_roughpipe 2>/dev/null; do
  sleep 30
done
"${VEGA_PYTHON}" "${ROOT_DIR}/scripts/summarize_pse_realworld_validation.py" --results-dir "${RESULTS_DIR}" \
  > "${CONTROL_DIR}/final_summary.log" 2>&1
