#!/usr/bin/env python3
"""Run one PSE-real-world validation case from the pinned YAML protocol.

The script deliberately runs one method/dataset/seed per process.  PSE and
VEGA-SR have incompatible dependency pins, so callers should invoke it with
the corresponding isolated Python environment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import pandas as pd
import sympy
import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from benchmark_metrics import (  # noqa: E402
    enrich_result_metrics,
    evaluate_expression,
    expression_complexity,
    regression_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "pse_realworld_vega_sr.yaml"))
    parser.add_argument("--method", required=True, choices=("pse", "vega_sr"))
    parser.add_argument("--dataset", required=True, choices=("emps", "roughpipe"))
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--gpu-index", type=int, default=int(os.environ.get("GPU_INDEX", "0")))
    parser.add_argument("--budget-sec", type=float)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pse-root", default=str(ROOT / "data" / "external" / "pse"))
    parser.add_argument("--vega-path", default=str(ROOT / "scripts" / "vega_sr.py"))
    parser.add_argument("--agent-api-base", default=os.environ.get("LLMSR_AGENT_API_BASE", "http://127.0.0.1:18001/v1"))
    parser.add_argument("--agent-api-key", default=os.environ.get("LLMSR_AGENT_API_KEY", "EMPTY"))
    parser.add_argument("--agent-model", default=os.environ.get("LLMSR_AGENT_MODEL", "Qwen3-VL-32B-Instruct"))
    return parser.parse_args()


def finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except Exception:
        return None
    return result if math.isfinite(result) else None


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_dataset(config_path: Path, config: dict[str, Any], dataset_name: str) -> Path:
    dataset_config = config["datasets"][dataset_name]
    cache_dir = (config_path.parent / config["execution"]["data_cache"]).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(urllib.parse.urlparse(dataset_config["raw_url"]).path).suffix or ".csv"
    path = cache_dir / f"{dataset_name}{suffix}"
    expected = str(dataset_config["sha256"]).lower()
    if not path.exists() or sha256_file(path) != expected:
        temporary = path.with_suffix(path.suffix + ".download")
        urllib.request.urlretrieve(dataset_config["raw_url"], temporary)
        actual = sha256_file(temporary)
        if actual != expected:
            raise RuntimeError(f"SHA-256 mismatch for {dataset_name}: {actual} != {expected}")
        temporary.replace(path)
    return path


def load_splits(
    config_path: Path,
    config: dict[str, Any],
    dataset_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dataset_config = config["datasets"][dataset_name]
    path = ensure_dataset(config_path, config, dataset_name)
    raw = pd.read_csv(path, header=None, names=dataset_config["raw_columns"])
    if dataset_name == "emps":
        frame = raw[["q", "qdot", "tau", "qddot"]].rename(columns={"qddot": "y"})
        protocol = dataset_config["protocols"][dataset_config["default_protocol"]]
        discovery_end = int(len(frame) * float(protocol["discovery_fraction"]))
        fit_end = int(discovery_end * float(protocol["inner_fit_fraction"]))
        train = frame.iloc[:fit_end].reset_index(drop=True)
        val = frame.iloc[fit_end:discovery_end].reset_index(drop=True)
        test = frame.iloc[discovery_end:].reset_index(drop=True)
        alignment = frame.iloc[:discovery_end].reset_index(drop=True)
        full = frame.reset_index(drop=True)
    else:
        scaled_log10_f = raw["scaled_log10_f"].to_numpy(dtype=float)
        log10_re = raw["log10_Re"].to_numpy(dtype=float)
        inverse_roughness = raw["inverse_relative_roughness"].to_numpy(dtype=float)
        f_value = np.power(10.0, scaled_log10_f) / 100.0
        reynolds = np.power(10.0, log10_re)
        x_value = np.log10(reynolds * np.sqrt(f_value / 32.0) / inverse_roughness)
        y_value = np.power(f_value, -0.5) + 2.0 * np.log10(1.0 / inverse_roughness)
        frame = pd.DataFrame({"x": x_value, "y": y_value}).sort_values("x", kind="mergesort").reset_index(drop=True)
        cycle = dataset_config["protocols"][dataset_config["default_protocol"]]["assignment_cycle"]
        assignments = np.asarray([cycle[idx % len(cycle)] for idx in range(len(frame))])
        train = frame.loc[assignments == "train"].reset_index(drop=True)
        val = frame.loc[assignments == "val"].reset_index(drop=True)
        test = frame.loc[assignments == "test"].reset_index(drop=True)
        alignment = frame.copy()
        full = frame.copy()
    return train, val, test, alignment, full


def metrics_for_expression(expr: str, frame: pd.DataFrame, feature_names: list[str]) -> dict[str, float | None]:
    try:
        prediction = evaluate_expression(expr, frame, feature_names)
        return regression_metrics(frame["y"].to_numpy(dtype=float), prediction)
    except Exception:
        return {name: None for name in ("mse", "rmse", "nmse", "nrmse", "mae", "r2")}


def select_pse_candidate(
    candidates: list[Any],
    val: pd.DataFrame,
    feature_names: list[str],
    eta: float,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    scored: list[dict[str, Any]] = []
    for candidate in candidates:
        if len(candidate) < 4:
            continue
        expr, train_reward, train_mse, pse_complexity = candidate[:4]
        expr = str(expr)
        val_metrics = metrics_for_expression(expr, val, feature_names)
        val_mse = finite_float(val_metrics["mse"])
        complexity = finite_float(pse_complexity)
        if val_mse is None or complexity is None:
            continue
        validation_reward = float(eta**complexity / (1.0 + math.sqrt(max(0.0, val_mse))))
        scored.append(
            {
                "expression": expr,
                "pse_train_reward": finite_float(train_reward),
                "pse_train_mse": finite_float(train_mse),
                "pse_complexity": complexity,
                "validation_reward": validation_reward,
                "val_mse": val_mse,
            }
        )
    scored.sort(key=lambda item: (-item["validation_reward"], item["pse_complexity"], item["val_mse"]))
    return (scored[0] if scored else None), scored


def run_pse(
    args: argparse.Namespace,
    config: dict[str, Any],
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    alignment: pd.DataFrame,
    full: pd.DataFrame,
    budget_sec: float,
) -> dict[str, Any]:
    pse_root = Path(args.pse_root).resolve()
    if not pse_root.is_dir():
        raise FileNotFoundError(pse_root)
    sys.path.insert(0, str(pse_root))
    old_cwd = Path.cwd()
    os.chdir(pse_root)
    try:
        import torch
        from model.regressor import PSRN_Regressor

        dataset_config = config["datasets"][args.dataset]
        reference = dataset_config["pse_reference"]
        stage_path = pse_root / reference["stage_config"]
        stage = yaml.safe_load(stage_path.read_text(encoding="utf-8"))
        stage["default"]["time_limit"] = float(budget_sec)
        features = list(dataset_config["features"])
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats()
        regressor = PSRN_Regressor(
            variables=features,
            use_const=bool(reference["use_const"]),
            stage_config=stage,
            device=torch.device("cuda"),
            token_generator="GP",
        )
        fit_kwargs: dict[str, Any] = {
            "n_down_sample": int(reference["n_down_sample"]),
            "eta": float(reference["reward_eta"]),
            "prun_ndigit": int(reference["prun_ndigit"]),
            "top_k": int(reference["top_k"]),
            "real_time_display": False,
        }
        if args.dataset == "emps":
            fit_kwargs.update(use_threshold=False, add_bias=bool(reference["add_bias"]))
        else:
            fit_kwargs.update(
                threshold=float(reference["threshold"]),
                use_replace_expo=bool(reference["use_replace_expo"]),
            )
        started = time.time()
        _, pareto = regressor.fit(
            train[features].to_numpy(dtype=float),
            train["y"].to_numpy(dtype=float).reshape(-1, 1),
            **fit_kwargs,
        )
        elapsed = time.time() - started
        candidates = regressor.get_pf(sort_by="reward") or pareto or []
        selected, scored = select_pse_candidate(candidates, val, features, float(reference["reward_eta"]))
        result: dict[str, Any] = {
            "method": "pse",
            "best_expr": selected["expression"] if selected else None,
            "runtime_sec": elapsed,
            "search_budget_sec": float(budget_sec),
            "candidate_count": len(candidates),
            "valid_candidate_count": len(scored),
            "selection_metric": "validation_reward_eta_0.99",
            "selected_validation_reward": selected["validation_reward"] if selected else None,
            "pse_complexity": selected["pse_complexity"] if selected else None,
            "gpu_peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "gpu_peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
            "pareto_candidates": scored,
        }
        expr = result["best_expr"]
        result.update(expression_complexity(expr, features))
        for split_name, frame in (("train", train), ("val", val), ("test", test)):
            for metric_name, value in metrics_for_expression(expr, frame, features).items():
                result[f"{split_name}_{metric_name}"] = value
        alignment_metrics = metrics_for_expression(expr, alignment, features)
        if args.dataset == "emps":
            result["pse_discovery_half_mse"] = alignment_metrics["mse"]
        else:
            result["pse_full_data_mse"] = metrics_for_expression(expr, full, features)["mse"]
        return result
    finally:
        os.chdir(old_cwd)


def import_vega(path: Path):
    import importlib.util

    module_name = f"vega_sr_realworld_{os.getpid()}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def expression_allowed(expr: Any, allowed_operators: list[str]) -> bool:
    """Conservatively reject candidate syntax outside the configured library."""
    try:
        parsed = sympy.sympify(str(expr), locals={
            "abs": sympy.Abs,
            "Abs": sympy.Abs,
            "sign": sympy.sign,
        })
    except Exception:
        return False
    allowed = {str(item).lower() for item in allowed_operators}
    allowed_functions = {item for item in allowed if item.isalpha()}
    for function in parsed.atoms(sympy.Function):
        name = function.func.__name__.lower()
        if name == "abs":
            name = "abs"
        if name not in allowed_functions:
            return False
    for power in parsed.atoms(sympy.Pow):
        exponent = power.exp
        if exponent.is_number:
            try:
                exponent_value = float(exponent)
            except Exception:
                exponent_value = float("nan")
            if math.isfinite(exponent_value) and math.isclose(exponent_value, round(exponent_value), abs_tol=1e-12):
                if exponent_value < 0 and "/" not in allowed:
                    return False
                continue  # Integer powers are representable by repeated multiplication/division.
            if exponent == sympy.Rational(1, 2) and "sqrt" in allowed:
                continue
        if "**" not in allowed:
            return False
    if parsed.has(sympy.pi) and "pi" not in allowed:
        return False
    if parsed.has(sympy.E) and "e" not in allowed:
        return False
    return True


def run_vega(
    args: argparse.Namespace,
    config: dict[str, Any],
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    alignment: pd.DataFrame,
    full: pd.DataFrame,
    budget_sec: float,
) -> dict[str, Any]:
    dataset_config = config["datasets"][args.dataset]
    features = list(dataset_config["features"])
    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.agent_api_base.startswith(("http://127.0.0.1", "http://localhost")):
        no_proxy_entries = ["127.0.0.1", "localhost"]
        for key in ("NO_PROXY", "no_proxy"):
            existing = [item.strip() for item in os.environ.get(key, "").split(",") if item.strip()]
            os.environ[key] = ",".join(dict.fromkeys(no_proxy_entries + existing))
        # Some httpx/OpenAI versions still initialize a SOCKS transport from
        # ALL_PROXY before applying NO_PROXY. The configured API is local, so
        # inherited proxy variables must not participate in this process.
        for key in ("ALL_PROXY", "all_proxy", "HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy"):
            os.environ.pop(key, None)
    os.environ["LLMSR_AGENT_API_BASE"] = args.agent_api_base
    os.environ["LLMSR_AGENT_API_KEY"] = args.agent_api_key
    os.environ["LLMSR_AGENT_MODEL"] = args.agent_model
    os.environ["LLMSR_MAX_RUNTIME_PER_TASK_SEC"] = str(float(budget_sec))
    os.environ["LLMSR_USE_TEST_FOR_SELECTION"] = "0"
    os.environ["LLMSR_METHOD_MODE"] = str(config["execution"]["method_mode"])
    vega_budget = dict(config["execution"].get("vega_sr_budget", {}))
    os.environ["LLMSR_V11_FULL_BUDGET"] = "1" if vega_budget.get("full_budget", False) else "0"
    os.environ["LLMSR_V11_ALLOW_FULL_CALLS_UNDER_180"] = (
        "1" if vega_budget.get("allow_full_calls_under_180", False) else "0"
    )
    os.environ["LLMSR_V11_IMAGE_LOOP_MAX_ROUNDS"] = str(
        int(vega_budget.get("image_loop_max_rounds", 3))
    )
    os.environ["LLMSR_V11_RESTORE_TEXT_PROPOSER"] = (
        "1" if vega_budget.get("restore_text_proposer", False) else "0"
    )
    vega = import_vega(Path(args.vega_path).resolve())
    llm_seed_policy = str(config["execution"].get("vega_llm_seed_policy", "role_call_v1"))
    if llm_seed_policy != "role_call_v1":
        raise ValueError(f"unsupported VEGA-SR LLM seed policy: {llm_seed_policy}")
    seed_audit: list[dict[str, Any]] = []
    seed_audit_lock = threading.Lock()
    original_build_role_client = vega._base._build_role_client
    original_generate = vega._base.LLMClient.generate

    def seeded_build_role_client(role, default_backend_config, deadline_ts=None):
        client = original_build_role_client(role, default_backend_config, deadline_ts=deadline_ts)
        client._pse_realworld_seed_role = str(role)
        client._pse_realworld_seed_call_index = 0
        client._pse_realworld_seed_lock = threading.Lock()
        return client

    def seeded_generate(
        self,
        messages,
        model=None,
        temperature=0.2,
        max_tokens=512,
        top_p=0.95,
        stop=None,
        seed=None,
        extra_body=None,
    ):
        if seed is None:
            role = str(getattr(self, "_pse_realworld_seed_role", "unknown"))
            lock = getattr(self, "_pse_realworld_seed_lock", seed_audit_lock)
            with lock:
                call_index = int(getattr(self, "_pse_realworld_seed_call_index", 0))
                self._pse_realworld_seed_call_index = call_index + 1
            request_fingerprint = {
                "dataset": args.dataset,
                "repeat_seed": int(args.seed),
                "agent_role": role,
                "role_call_index": call_index,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": top_p,
            }
            canonical = json.dumps(request_fingerprint, sort_keys=True, ensure_ascii=False, default=str)
            seed = int(hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:8], 16) & 0x7FFFFFFF
            with seed_audit_lock:
                seed_audit.append({**request_fingerprint, "llm_seed": seed})
        return original_generate(
            self,
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop=stop,
            seed=seed,
            extra_body=extra_body,
        )

    vega._base._build_role_client = seeded_build_role_client
    vega._base.LLMClient.generate = seeded_generate
    vega._base.MAX_RUNTIME_PER_TASK_SEC = float(budget_sec)
    vega._base.USE_TEST_FOR_SELECTION = False
    allowed = list(dataset_config["vega_allowed_operators"])
    normalized_allowed = []
    for operator in allowed:
        normalized = "**" if str(operator) in {"x**2", "x**3"} else str(operator)
        if normalized not in normalized_allowed:
            normalized_allowed.append(normalized)
    vega._base.ALLOWED_OPERATORS = normalized_allowed
    domain_prior = dict(dataset_config.get("vega_domain_prior", {}))
    protected_templates = [
        str(expr).strip()
        for expr in domain_prior.get("protected_templates", [])
        if str(expr).strip()
    ]
    invalid_templates = [
        expr for expr in protected_templates
        if not expression_allowed(expr, normalized_allowed)
    ]
    if invalid_templates:
        raise ValueError(f"domain-prior templates use disallowed operators: {invalid_templates}")
    original_propose_initial = vega._base.ProposerAgent.propose_initial

    def propose_initial_with_domain_prior(self, *call_args, **call_kwargs):
        proposal = original_propose_initial(self, *call_args, **call_kwargs)
        existing = list(proposal.get("candidate_exprs", []) or [])
        proposal["candidate_exprs"] = list(dict.fromkeys(protected_templates + existing))
        trace = dict(proposal.get("trace", {}) or {})
        trace["pse_aligned_domain_prior"] = {
            "source": domain_prior.get("source"),
            "protected_templates": protected_templates,
            "candidate_count": len(protected_templates),
        }
        trace["protected_candidate_count"] = int(trace.get("protected_candidate_count") or 0) + len(protected_templates)
        trace["merged_candidate_count"] = len(proposal["candidate_exprs"])
        proposal["trace"] = trace
        return proposal

    vega._base.ProposerAgent.propose_initial = propose_initial_with_domain_prior
    original_evaluate = vega.V11EvaluatorAgent.evaluate

    def constrained_evaluate(self, candidate_exprs, dataset, *call_args, **call_kwargs):
        filtered = [
            expr for expr in list(candidate_exprs or [])
            if expression_allowed(expr, normalized_allowed)
        ]
        return original_evaluate(self, filtered, dataset, *call_args, **call_kwargs)

    vega.V11EvaluatorAgent.evaluate = constrained_evaluate
    started = time.time()
    with TemporaryDirectory(prefix="pse_realworld_vega_") as tmpdir:
        dataset = vega.build_dataset_from_explicit_splits(train, val, test, Path(tmpdir))
        dataset.source_tag = "pse_realworld"
        row_meta = {
            "task_type": "pse_realworld",
            "dataset_dir": "pse_realworld",
            "difficulty": "realworld",
            "base_name": args.dataset,
            "dataset_case_id": f"{args.dataset}_seed_{args.seed}",
            "repeat_seed": int(args.seed),
            "true_expression": None,
        }
        result = vega._run_core_pipeline(dataset=dataset, row_meta=row_meta)
    pipeline_best_expr = result.get("best_expr")
    selection_metric = domain_prior.get("selection_metric")
    selection_candidates: list[dict[str, Any]] = []
    if selection_metric == "validation_reward_eta_0.99":
        eta = float(domain_prior.get("reward_eta", 0.99))
        try:
            history = json.loads(result.get("candidate_evaluation_history") or "[]")
        except (TypeError, json.JSONDecodeError):
            history = []
        seen_exprs: set[str] = set()
        for item in history:
            expr = str(item.get("fitted_expression") or item.get("expression") or "").strip()
            if not expr or expr in seen_exprs or not expression_allowed(expr, normalized_allowed):
                continue
            seen_exprs.add(expr)
            val_metrics = metrics_for_expression(expr, val, features)
            val_mse = finite_float(val_metrics.get("mse"))
            if val_mse is None:
                continue
            complexity = int(expression_complexity(expr, features)["expr_complexity"])
            reward = float(eta**complexity / (1.0 + math.sqrt(max(0.0, val_mse))))
            selection_candidates.append({
                "expression": expr,
                "validation_mse": val_mse,
                "expression_complexity": complexity,
                "validation_reward": reward,
            })
        selection_candidates.sort(key=lambda item: (
            -item["validation_reward"],
            item["expression_complexity"],
            item["validation_mse"],
        ))
        if selection_candidates:
            result["best_expr"] = selection_candidates[0]["expression"]
            result["selected_validation_reward"] = selection_candidates[0]["validation_reward"]
    result["pipeline_best_expr_before_pse_aligned_selection"] = pipeline_best_expr
    result["pse_aligned_domain_prior"] = domain_prior
    result["pse_aligned_selection_metric"] = selection_metric or "vega_default_validation_selection"
    result["pse_aligned_selection_candidates"] = selection_candidates
    result["runtime_sec"] = finite_float(result.get("runtime_sec")) or (time.time() - started)
    result["method"] = "vega_sr"
    result["search_budget_sec"] = float(budget_sec)
    result["vega_sr_budget_config"] = vega_budget
    result["llm_seed_policy"] = llm_seed_policy
    result["llm_repeat_seed"] = int(args.seed)
    result["llm_seed_audit"] = sorted(
        seed_audit,
        key=lambda item: (item["agent_role"], item["role_call_index"]),
    )
    result["allowed_operators"] = allowed
    enrich_result_metrics(result, train, val, test, features)
    expr = result.get("best_expr")
    result["selected_expression_uses_allowed_operators"] = expression_allowed(expr, normalized_allowed)
    if args.dataset == "emps":
        result["pse_discovery_half_mse"] = metrics_for_expression(expr, alignment, features)["mse"]
    else:
        result["pse_full_data_mse"] = metrics_for_expression(expr, full, features)["mse"]
    return result


def main() -> int:
    args = parse_args()
    # PSE constructs its CUDA tensors during import/initialization, so physical
    # device selection must be visible before torch or the upstream model loads.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if args.seed not in config["execution"]["repeat_seeds"]:
        raise ValueError(f"seed {args.seed} is not declared in repeat_seeds")
    budget_sec = float(args.budget_sec or config["execution"]["case_budget_sec"])
    configured_grace_sec = float(config["execution"]["timeout_grace_sec"])
    train, val, test, alignment, full = load_splits(config_path, config, args.dataset)
    base_result = {
        "schema_version": int(config["schema_version"]),
        "dataset": args.dataset,
        "dataset_display_name": config["datasets"][args.dataset]["display_name"],
        "protocol": config["datasets"][args.dataset]["default_protocol"],
        "repeat_seed": int(args.seed),
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "test_used_for_selection": False,
        "config_path": str(config_path),
        "pse_commit": config["provenance"]["pse_commit"],
        "vega_sr_reference_commit": config["provenance"]["vega_sr_reference_commit"],
        "started_at_unix": time.time(),
        "configured_timeout_grace_sec": configured_grace_sec,
        "hard_timeout_sec": float(config["execution"].get("hard_timeout_sec", budget_sec + configured_grace_sec)),
    }
    try:
        if args.method == "pse":
            method_result = run_pse(args, config, train, val, test, alignment, full, budget_sec)
        else:
            method_result = run_vega(args, config, train, val, test, alignment, full, budget_sec)
        base_result.update(method_result)
        base_result["status"] = "ok"
    except Exception as exc:
        base_result.update({"method": args.method, "status": "error", "error": repr(exc)})
        raise
    finally:
        base_result["finished_at_unix"] = time.time()
        base_result["total_wall_sec"] = base_result["finished_at_unix"] - base_result["started_at_unix"]
        base_result["postprocessing_overrun"] = bool(
            base_result.get("status") == "ok"
            and base_result["total_wall_sec"] > budget_sec + configured_grace_sec
        )
        output = Path(args.output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(json.dumps(json_safe(base_result), ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(output)
    print(json.dumps({k: base_result.get(k) for k in ("method", "dataset", "repeat_seed", "status", "best_expr", "test_mse", "test_nmse", "test_r2", "runtime_sec")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
