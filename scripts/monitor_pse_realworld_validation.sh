#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="${RESULTS_DIR:-${ROOT_DIR}/results/pse_realworld_validation}"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv-vega/bin/python}"
CONTROL_DIR="${RESULTS_DIR}/control"
mkdir -p "${CONTROL_DIR}"

"${PYTHON_BIN}" - "${RESULTS_DIR}" <<'PY'
import json
import os
import sys
import time
from pathlib import Path

path = Path(sys.argv[1]) / "completion_audit.json"
temporary = path.with_suffix(path.suffix + ".tmp")
temporary.write_text(json.dumps({
    "state": "running",
    "complete": False,
    "monitor_started_at_unix": time.time(),
}, indent=2), encoding="utf-8")
os.replace(temporary, path)
PY

while tmux has-session -t pse_realworld_vega_emps 2>/dev/null || tmux has-session -t pse_realworld_vega_roughpipe 2>/dev/null; do
  "${PYTHON_BIN}" "${ROOT_DIR}/scripts/summarize_pse_realworld_validation.py" --results-dir "${RESULTS_DIR}" \
    > "${CONTROL_DIR}/live_summary.log" 2>&1 || true
  sleep 30
done

"${PYTHON_BIN}" "${ROOT_DIR}/scripts/summarize_pse_realworld_validation.py" --results-dir "${RESULTS_DIR}" \
  > "${CONTROL_DIR}/final_summary.log" 2>&1

"${PYTHON_BIN}" - "${RESULTS_DIR}" <<'PY'
import json
import math
import os
import re
import sys
import time
from pathlib import Path

root = Path(sys.argv[1])
expected = {(method, dataset, seed) for method in ("pse", "vega_sr") for dataset in ("emps", "roughpipe") for seed in range(20)}
observed = {}
issues = []
expected_split_sizes = {
    "emps": (4000, 1000, 5001),
    "roughpipe": (218, 72, 72),
}
for path in (root / "cases").glob("*/*/seed_*.json"):
    with path.open(encoding="utf-8") as handle:
        row = json.load(handle)
    path_method, path_dataset = path.parent.parent.name, path.parent.name
    path_seed = int(path.stem.removeprefix("seed_"))
    key = (row.get("method"), row.get("dataset"), row.get("repeat_seed"))
    observed[key] = row.get("status")
    case_issues = []
    if key != (path_method, path_dataset, path_seed):
        case_issues.append("identity_does_not_match_path")
    if row.get("test_used_for_selection") is not False:
        case_issues.append("test_used_for_selection")
    if tuple(row.get(name) for name in ("n_train", "n_val", "n_test")) != expected_split_sizes.get(path_dataset):
        case_issues.append("unexpected_split_sizes")
    if not math.isclose(float(row.get("search_budget_sec", -1)), 90.0, rel_tol=0.0, abs_tol=1e-9):
        case_issues.append("unexpected_search_budget")
    if not str(row.get("best_expr") or "").strip():
        case_issues.append("missing_expression")
    for metric in ("test_mse", "test_nmse", "test_r2"):
        try:
            if not math.isfinite(float(row.get(metric))):
                raise ValueError
        except (TypeError, ValueError):
            case_issues.append(f"nonfinite_{metric}")
    if path_method == "vega_sr":
        if row.get("selected_expression_uses_allowed_operators") is not True:
            case_issues.append("operator_constraint_failed")
        budget_config = row.get("vega_sr_budget_config") or {}
        if budget_config != {
            "full_budget": True,
            "allow_full_calls_under_180": True,
            "image_loop_max_rounds": 3,
            "restore_text_proposer": True,
        }:
            case_issues.append("incomplete_vega_budget_profile")
        if row.get("llm_seed_policy") != "role_call_v1" or row.get("llm_repeat_seed") != path_seed:
            case_issues.append("vega_llm_seed_policy_failed")
        seed_audit = row.get("llm_seed_audit") or []
        if not seed_audit or any(item.get("dataset") != path_dataset or item.get("repeat_seed") != path_seed for item in seed_audit):
            case_issues.append("vega_llm_seed_audit_failed")
        if path_dataset == "emps":
            prior = row.get("pse_aligned_domain_prior") or {}
            if "sign(qdot)" not in " ".join(prior.get("protected_templates", [])):
                case_issues.append("missing_pse_aligned_emps_prior")
            if row.get("pse_aligned_selection_metric") != "validation_reward_eta_0.99":
                case_issues.append("missing_pse_aligned_validation_selection")
        try:
            leakage = json.loads(row.get("no_leakage_audit") or "{}")
        except (TypeError, json.JSONDecodeError):
            leakage = {}
        if leakage.get("selection_uses_validation_only") is not True or leakage.get("test_split_used_for_selection") is not False:
            case_issues.append("vega_no_leakage_audit_failed")
    if case_issues:
        issues.append({"path": str(path), "issues": case_issues})
missing = sorted(expected - set(observed))
not_ok = sorted((key, status) for key, status in observed.items() if key in expected and status != "ok")
postprocessing_overruns = sorted(
    key for key, row in (
        ((item.get("method"), item.get("dataset"), item.get("repeat_seed")), item)
        for path in (root / "cases").glob("*/*/seed_*.json")
        for item in [json.loads(path.read_text(encoding="utf-8"))]
    )
    if row.get("postprocessing_overrun") is True
)
historical_failures = set()
failure_pattern = re.compile(r"\[FAIL\]\s+(\S+)\s+(\S+)\s+seed=(\d+)\s+exit=(\d+)")
for log_path in (root / "control").glob("*_driver.log"):
    for match in failure_pattern.finditer(log_path.read_text(encoding="utf-8", errors="replace")):
        historical_failures.add((match.group(1), match.group(2), int(match.group(3)), int(match.group(4))))
recovered_failures = sorted(
    failure for failure in historical_failures
    if observed.get(failure[:3]) == "ok"
)
unresolved_historical_failures = sorted(set(historical_failures) - set(recovered_failures))
audit = {
    "state": "complete" if not missing and not not_ok and not issues and not unresolved_historical_failures else "incomplete",
    "audited_at_unix": time.time(),
    "expected_cases": len(expected),
    "observed_cases": len(observed),
    "missing": missing,
    "not_ok": not_ok,
    "invariant_issues": issues,
    "postprocessing_overrun_cases": postprocessing_overruns,
    "historical_launcher_failures_recovered": recovered_failures,
    "historical_launcher_failures_unresolved": unresolved_historical_failures,
    "complete": not missing and not not_ok and not issues and not unresolved_historical_failures,
}
path = root / "completion_audit.json"
temporary = path.with_suffix(path.suffix + ".tmp")
temporary.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
os.replace(temporary, path)
print(json.dumps(audit, ensure_ascii=False))
raise SystemExit(0 if audit["complete"] else 1)
PY
