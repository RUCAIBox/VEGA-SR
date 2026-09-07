#!/usr/bin/env bash
# Create a self-contained EMPS E1/E5/E6 artifact directory and run one arm.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARM="${ARM:?set ARM to emps_full_no_template, emps_numeric_only_no_template, or emps_no_feedback_no_template}"
GPU_INDEX="${GPU_INDEX:?set GPU_INDEX to a GPU with the frozen VEGA-SR model service}"
PYTHON_BIN="${PYTHON_BIN:?set PYTHON_BIN to the VEGA-SR environment python}"
RESULTS_DIR="${RESULTS_DIR:?set RESULTS_DIR to a new, empty output directory}"
SEEDS="${SEEDS:-0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19}"
CASE_BUDGET_SEC="${CASE_BUDGET_SEC:-90}"
CONFIG_PATH="${CONFIG_PATH:-${ROOT_DIR}/pse_realworld_vega_sr.yaml}"
AGENT_API_BASE="${LLMSR_AGENT_API_BASE:-http://127.0.0.1:18001/v1}"
AGENT_MODEL="${LLMSR_AGENT_MODEL:-Qwen3-VL-32B-Instruct}"

case "${ARM}" in
  emps_full_no_template|emps_numeric_only_no_template|emps_no_feedback_no_template) ;;
  *) echo "unsupported fair EMPS arm: ${ARM}" >&2; exit 2 ;;
esac

if [[ -e "${RESULTS_DIR}" ]] && [[ -n "$(find "${RESULTS_DIR}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "RESULTS_DIR must be new or empty; refusing to mix artifacts: ${RESULTS_DIR}" >&2
  exit 2
fi
mkdir -p "${RESULTS_DIR}" "${RESULTS_DIR}/per_run" "${RESULTS_DIR}/logs"
cp "${CONFIG_PATH}" "${RESULTS_DIR}/resolved_config.yaml"
git -C "${ROOT_DIR}" rev-parse HEAD > "${RESULTS_DIR}/git_commit.txt"
git -C "${ROOT_DIR}" status --short > "${RESULTS_DIR}/git_status.txt"
git -C "${ROOT_DIR}" diff --binary > "${RESULTS_DIR}/git_diff.patch"

"${PYTHON_BIN}" - "${RESULTS_DIR}/run_manifest.json" "${ARM}" "${CONFIG_PATH}" "${AGENT_API_BASE}" "${AGENT_MODEL}" "${CASE_BUDGET_SEC}" "${SEEDS}" <<'PY'
import hashlib, json, platform, sys, time
from pathlib import Path
output, arm, config, api_base, model, budget, seeds = sys.argv[1:]
config_path = Path(config).resolve()
manifest = {
    "created_at_unix": time.time(),
    "arm": arm,
    "dataset": "emps",
    "repeat_seeds": [int(value) for value in seeds.split()],
    "case_budget_sec": float(budget),
    "agent_api_base": api_base,
    "agent_model_requested": model,
    "config_path": str(config_path),
    "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
    "python_version": platform.python_version(),
}
Path(output).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
PY

export LLMSR_AGENT_API_BASE="${AGENT_API_BASE}"
export LLMSR_AGENT_MODEL="${AGENT_MODEL}"
CONFIG_PATH="${CONFIG_PATH}" \
METHOD=vega_sr DATASET=emps VEGA_PROFILE="${ARM}" GPU_INDEX="${GPU_INDEX}" \
PYTHON_BIN="${PYTHON_BIN}" RESULTS_DIR="${RESULTS_DIR}" CASE_BUDGET_SEC="${CASE_BUDGET_SEC}" \
SEEDS="${SEEDS}" \
bash "${ROOT_DIR}/scripts/launch_pse_realworld_validation.sh"

"${PYTHON_BIN}" "${ROOT_DIR}/scripts/summarize_emps_fair_rerun.py" \
  --results-dir "${RESULTS_DIR}" --method vega_sr --dataset emps
