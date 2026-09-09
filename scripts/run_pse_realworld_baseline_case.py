#!/usr/bin/env python3
"""Run one strict-split baseline for the PSE real-world comparison.

PySR and Operon follow the operator libraries and 90-second settings released
with PSE, but candidates are selected on our validation split rather than on
the final test split. ``physics_ls`` is an EMPS-only prior ablation: it fits the
same Newton/friction templates supplied to VEGA-SR without doing symbolic
search. DSO reuses the same official adapter and released hyperparameter family
used by the paper's 895-task comparison.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import random
import re
import sys
import time
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

from benchmark_metrics import expression_complexity  # noqa: E402
from run_pse_realworld_case import (  # noqa: E402
    json_safe,
    load_splits,
    metrics_for_expression,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "pse_realworld_vega_sr.yaml"))
    parser.add_argument(
        "--method",
        required=True,
        choices=("pysr", "operon", "gplearn", "dso", "llm_sr", "linear_ls", "physics_ls"),
    )
    parser.add_argument("--dataset", required=True, choices=("emps", "roughpipe"))
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--budget-sec", type=float)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def score_candidates(
    expressions: list[str],
    val: pd.DataFrame,
    features: list[str],
    eta: float,
    native: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Deduplicate and rank candidates using validation data only."""
    scored: list[dict[str, Any]] = []
    seen: set[str] = set()
    native = native or [{} for _ in expressions]
    for expression, metadata in zip(expressions, native):
        expression = str(expression).strip()
        if not expression or expression in seen:
            continue
        seen.add(expression)
        val_metrics = metrics_for_expression(expression, val, features)
        val_mse = val_metrics.get("mse")
        if val_mse is None or not math.isfinite(float(val_mse)):
            continue
        complexity = expression_complexity(expression, features).get("expr_complexity")
        if complexity is None:
            continue
        reward = float(eta ** int(complexity) / (1.0 + math.sqrt(max(0.0, float(val_mse)))))
        scored.append(
            {
                "expression": expression,
                "validation_mse": float(val_mse),
                "expression_complexity": int(complexity),
                "validation_reward": reward,
                **metadata,
            }
        )
    scored.sort(
        key=lambda item: (
            -item["validation_reward"],
            item["expression_complexity"],
            item["validation_mse"],
        )
    )
    return (scored[0] if scored else None), scored


def run_pysr(
    dataset: str,
    seed: int,
    budget_sec: float,
    train: pd.DataFrame,
    val: pd.DataFrame,
    features: list[str],
    eta: float,
) -> dict[str, Any]:
    from pysr import PySRRegressor

    binary = ["+", "-", "*", "/"]
    if dataset == "emps":
        unary = ["sin", "cos", "exp", "log", "sign", "abs", "cosh", "tanh"]
        released_niterations = 100
        mappings: dict[str, Any] = {}
    else:
        unary = ["cos", "sin", "exp", "log", "tanh", "cosh", "square", "cube"]
        released_niterations = 50
        mappings = {"square": lambda x: x**2, "cube": lambda x: x**3}
    released_cycles_per_iteration = 380
    # Preserve the released maximum number of mutation cycles while making
    # each cycle its own interruptible iteration. This makes PySR 1.5 check
    # the same 90-second deadline at the finest safe granularity.
    niterations = released_niterations * released_cycles_per_iteration
    with TemporaryDirectory(prefix=f"pse_baseline_pysr_{dataset}_{seed}_") as output_dir:
        model = PySRRegressor(
            timeout_in_seconds=float(budget_sec),
            random_state=seed,
            deterministic=True,
            parallelism="serial",
            niterations=niterations,
            # PySR checks its wall-clock deadline between mutation-cycle
            # batches. A smaller batch prevents seed-dependent multi-minute
            # overruns while preserving the same 90-second search deadline.
            ncycles_per_iteration=1,
            binary_operators=binary,
            unary_operators=unary,
            extra_sympy_mappings=mappings,
            output_directory=output_dir,
            progress=False,
            verbosity=0,
        )
        started = time.time()
        model.fit(
            train[features].to_numpy(dtype=float),
            train["y"].to_numpy(dtype=float),
            variable_names=features,
        )
        runtime = time.time() - started
        expressions = [str(item) for item in model.sympy(list(range(len(model.equations_))))]
        native = [
            {
                "native_complexity": int(row["complexity"]),
                "native_train_loss": float(row["loss"]),
            }
            for _, row in model.equations_.iterrows()
        ]
    selected, scored = score_candidates(expressions, val, features, eta, native)
    return {
        "method": "pysr",
        "best_expr": selected["expression"] if selected else None,
        "runtime_sec": runtime,
        "candidate_count": len(expressions),
        "valid_candidate_count": len(scored),
        "pareto_candidates": scored,
        "allowed_operators": binary + unary,
        "baseline_source": "PSE released PySR real-world configuration",
        "baseline_hyperparameters": {
            "niterations": niterations,
            "released_niterations": released_niterations,
            "released_ncycles_per_iteration": released_cycles_per_iteration,
            "timeout_in_seconds": float(budget_sec),
            "deterministic": True,
            "parallelism": "serial",
            "ncycles_per_iteration": 1,
            "julia_threads": 1,
            "blas_threads": 1,
        },
    }


def canonicalize_operon(expression: str, features: list[str]) -> str:
    locals_map: dict[str, Any] = {
        "abs": sympy.Abs,
        "square": lambda x: x**2,
        "cube": lambda x: x**3,
    }
    for index, name in enumerate(features, start=1):
        locals_map[f"X{index}"] = sympy.Symbol(name)
    parsed = sympy.sympify(str(expression).replace("^", "**"), locals=locals_map)
    return str(parsed)


def canonicalize_gplearn(expression: str, features: list[str]) -> str:
    locals_map: dict[str, Any] = {
        "add": lambda x, y: x + y,
        "sub": lambda x, y: x - y,
        "mul": lambda x, y: x * y,
        "div": lambda x, y: x / y,
        "sin": sympy.sin,
        "cos": sympy.cos,
        "log": lambda x: sympy.log(sympy.Abs(x)),
        "abs": sympy.Abs,
        "tanh": sympy.tanh,
        "sign": sympy.sign,
        "exp": sympy.exp,
    }
    for index, name in enumerate(features):
        locals_map[f"X{index}"] = sympy.Symbol(name)
    return str(sympy.sympify(str(expression), locals=locals_map))


def run_operon(
    dataset: str,
    seed: int,
    budget_sec: float,
    train: pd.DataFrame,
    val: pd.DataFrame,
    features: list[str],
    eta: float,
) -> dict[str, Any]:
    from pyoperon.sklearn import SymbolicRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error

    x_train = train[features].to_numpy(dtype=float)
    y_train = train["y"].to_numpy(dtype=float)
    rf = RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=1)
    uncertainty = float(math.sqrt(mean_squared_error(y_train, rf.fit(x_train, y_train).predict(x_train))))
    if dataset == "emps":
        allowed = "add,sub,mul,div,sin,cos,exp,logabs,tanh,cosh,abs,constant,variable"
    else:
        allowed = "add,sub,mul,div,sin,cos,exp,logabs,tanh,cosh,square,constant,variable"
    regressor = SymbolicRegressor(
        allowed_symbols=allowed,
        brood_size=10,
        comparison_factor=0,
        crossover_internal_probability=0.9,
        crossover_probability=1.0,
        epsilon=1e-5,
        female_selector="tournament",
        generations=10_000_000_000_000,
        initialization_max_depth=5,
        initialization_max_length=10,
        initialization_method="btc",
        irregularity_bias=0.0,
        local_search_probability=1.0,
        lamarckian_probability=1.0,
        optimizer_iterations=1,
        optimizer="lm",
        male_selector="tournament",
        max_depth=10,
        max_evaluations=100_000_000_000_000,
        max_length=50,
        max_selection_pressure=100,
        model_selection_criterion="minimum_description_length",
        mutation_probability=0.25,
        n_threads=4,
        objectives=["mse", "length"],
        offspring_generator="os",
        pool_size=1000,
        population_size=1000,
        random_state=seed,
        reinserter="keep-best",
        max_time=max(1, int(round(budget_sec))),
        tournament_size=3,
        uncertainty=[uncertainty],
        add_model_intercept_term=True,
        add_model_scale_term=True,
    )
    started = time.time()
    regressor.fit(x_train, y_train)
    runtime = time.time() - started
    expressions: list[str] = []
    native: list[dict[str, Any]] = []
    for solution in regressor.pareto_front_:
        expressions.append(canonicalize_operon(regressor.get_model_string(solution["tree"], 12), features))
        objectives = solution["objective_values"]
        native.append(
            {
                "native_complexity": int(objectives[1]),
                "native_train_loss": float(objectives[0]),
                "native_mdl": float(solution["minimum_description_length"]),
            }
        )
    selected, scored = score_candidates(expressions, val, features, eta, native)
    return {
        "method": "operon",
        "best_expr": selected["expression"] if selected else None,
        "runtime_sec": runtime,
        "candidate_count": len(expressions),
        "valid_candidate_count": len(scored),
        "pareto_candidates": scored,
        "allowed_operators": allowed.split(","),
        "baseline_source": "PSE released PyOperon real-world configuration",
        "baseline_hyperparameters": {
            "objectives": ["mse", "length"],
            "population_size": 1000,
            "pool_size": 1000,
            "n_threads": 4,
            "max_time_seconds": max(1, int(round(budget_sec))),
        },
    }


def run_gplearn(
    dataset: str,
    seed: int,
    budget_sec: float,
    train: pd.DataFrame,
    val: pd.DataFrame,
    features: list[str],
    eta: float,
) -> dict[str, Any]:
    """Classic genetic-programming baseline with validation-only selection."""
    from gplearn.functions import make_function
    from gplearn.genetic import SymbolicRegressor
    from sklearn.utils.validation import check_X_y
    from types import MethodType

    def protected_exp(x):
        return np.exp(np.clip(x, -20.0, 20.0))

    function_set: list[Any] = [
        "add", "sub", "mul", "div", "sin", "cos", "log", "abs",
        make_function(function=np.tanh, name="tanh", arity=1, wrap=False),
        make_function(function=np.sign, name="sign", arity=1, wrap=False),
        make_function(function=protected_exp, name="exp", arity=1, wrap=False),
    ]
    model = SymbolicRegressor(
        population_size=1000,
        generations=1,
        tournament_size=20,
        stopping_criteria=0.0,
        const_range=(-5.0, 5.0),
        init_depth=(2, 6),
        init_method="half and half",
        function_set=function_set,
        metric="mse",
        parsimony_coefficient=0.001,
        p_crossover=0.7,
        p_subtree_mutation=0.1,
        p_hoist_mutation=0.05,
        p_point_mutation=0.1,
        max_samples=1.0,
        warm_start=True,
        low_memory=False,
        n_jobs=1,
        verbose=0,
        random_state=seed,
    )
    if not hasattr(model, "_validate_data"):
        # gplearn 0.4.2 predates sklearn 1.7, where BaseEstimator's private
        # helper was removed. Keep the compatibility shim local to this
        # instance and preserve the validation semantics gplearn expects.
        def compat_validate_data(self, x, y, y_numeric=True):
            checked_x, checked_y = check_X_y(x, y, y_numeric=y_numeric)
            self.n_features_in_ = checked_x.shape[1]
            return checked_x, checked_y

        model._validate_data = MethodType(compat_validate_data, model)
    x_train = train[features].to_numpy(dtype=float)
    y_train = train["y"].to_numpy(dtype=float)
    started = time.time()
    generation = 0
    expressions: list[str] = []
    native: list[dict[str, Any]] = []
    while generation == 0 or time.time() - started < float(budget_sec):
        generation += 1
        model.set_params(generations=generation)
        generation_started = time.time()
        model.fit(x_train, y_train)
        programs = [program for program in (getattr(model, "_best_programs", None) or []) if program]
        best_program = getattr(model, "_program", None)
        if best_program is not None:
            programs.append(best_program)
        populations = getattr(model, "_programs", None) or []
        if populations:
            population = [program for program in populations[-1] if program is not None]
            population.sort(key=lambda program: float(program.raw_fitness_))
            programs.extend(population[:50])
        for program in programs:
            expression = canonicalize_gplearn(str(program), features)
            expressions.append(expression)
            native.append({"native_fitness": float(program.raw_fitness_), "generation": generation})
        generation_sec = time.time() - generation_started
        remaining = float(budget_sec) - (time.time() - started)
        if remaining <= max(0.25, generation_sec * 1.1):
            break
    runtime = time.time() - started
    selected, scored = score_candidates(expressions, val, features, eta, native)
    return {
        "method": "gplearn",
        "best_expr": selected["expression"] if selected else None,
        "runtime_sec": runtime,
        "candidate_count": len(expressions),
        "valid_candidate_count": len(scored),
        "pareto_candidates": scored,
        "allowed_operators": ["+", "-", "*", "/", "sin", "cos", "log", "abs", "tanh", "sign", "exp"],
        "baseline_source": "gplearn 0.4.2 SymbolicRegressor",
        "baseline_hyperparameters": {
            "population_size": 1000,
            "completed_generations": generation,
            "parsimony_coefficient": 0.001,
            "time_budget_seconds": float(budget_sec),
        },
    }


def canonicalize_dso(expression: str, features: list[str]) -> str:
    """Map DSO's one-based x1..xN symbols to the dataset feature names."""
    def replace(match: re.Match[str]) -> str:
        index = int(match.group(1)) - 1
        return features[index] if 0 <= index < len(features) else match.group(0)

    return re.sub(r"\bx(\d+)\b", replace, str(expression))


def run_dso(
    seed: int,
    budget_sec: float,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
) -> dict[str, Any]:
    """Run the paper's DSO adapter while preserving protected-op predictions."""
    dso_root = Path(
        os.environ.get(
            "DSO_ROOT",
            "/home/liuyihong/deep-symbolic-optimization/dso",
        )
    ).resolve()
    if not (dso_root / "dso" / "__init__.py").is_file():
        raise FileNotFoundError(f"DSO source tree not found: {dso_root}")
    if str(dso_root) not in sys.path:
        sys.path.insert(0, str(dso_root))
    import run_cpu_baseline_benchmarks as cpu_baselines

    source_config = ROOT / "evaluation_suites" / "cpu_symbolic_regression_fourbench" / "configs" / "dso_100s.yaml"
    dso_config = yaml.safe_load(source_config.read_text(encoding="utf-8"))
    dso_config.setdefault("runtime", {})["max_fit_seconds"] = float(budget_sec)
    with TemporaryDirectory(prefix=f"pse_baseline_dso_{seed}_") as temporary_dir:
        runtime_config = Path(temporary_dir) / "dso_realworld.yaml"
        runtime_config.write_text(yaml.safe_dump(dso_config, sort_keys=False), encoding="utf-8")
        native = cpu_baselines.fit_dso(
            train,
            val,
            test,
            config_path=runtime_config,
            random_state=seed,
        )

    expression = canonicalize_dso(native.get("best_expr"), features)
    split_metrics = {
        "train_mse": native.get("best_train_mse"),
        "val_mse": native.get("best_val_mse"),
        "test_mse": native.get("best_test_mse"),
        "train_r2": native.get("train_r2"),
        "val_r2": native.get("val_r2"),
        "test_r2": native.get("test_r2"),
    }
    for split_name, frame in (("train", train), ("val", val), ("test", test)):
        mse = split_metrics.get(f"{split_name}_mse")
        variance = float(np.var(frame["y"].to_numpy(dtype=float)))
        split_metrics[f"{split_name}_nmse"] = (
            float(mse) / variance
            if mse is not None and math.isfinite(float(mse)) and variance > 0
            else None
        )
    return {
        "method": "dso",
        "best_expr": expression,
        "runtime_sec": native.get("fit_runtime_sec"),
        "candidate_count": native.get("num_candidate_exprs"),
        "valid_candidate_count": native.get("validation_search_unique_evaluations"),
        "allowed_operators": ["+", "-", "*", "/", "sin", "cos", "exp", "log"],
        "baseline_source": "DSO configuration used by the VEGA-SR 895-task comparison",
        "baseline_hyperparameters": {
            "source_config": str(source_config),
            "source_revision": "8348d5b08d1eef6170fdbfd222a492ded990ea12",
            "time_budget_seconds": float(budget_sec),
            "protected_operators": True,
            "n_samples": dso_config.get("dso", {}).get("training", {}).get("n_samples"),
            "batch_size": dso_config.get("dso", {}).get("training", {}).get("batch_size"),
        },
        "native_validation_selection": True,
        "native_metrics": native,
        "_precomputed_split_metrics": split_metrics,
    }


def physical_llmsr_expression(
    expression: str | None,
    parameters: np.ndarray,
    features: list[str],
    normalization: dict[str, Any],
) -> str | None:
    """Convert an LLM-SR normalized expression back to physical variables."""
    if not expression:
        return None
    substituted = str(expression)
    for index, value in enumerate(parameters):
        substituted = re.sub(
            rf"\bparams\s*\[\s*{index}\s*\]",
            f"({float(value):.17g})",
            substituted,
        )
    substituted = substituted.replace("np.", "").replace("numpy.", "")
    locals_map: dict[str, Any] = {
        "abs": sympy.Abs,
        "Abs": sympy.Abs,
        "sign": sympy.sign,
        "minimum": sympy.Min,
        "maximum": sympy.Max,
    }
    x_symbols = [sympy.Symbol(f"x{i}") for i in range(len(features))]
    locals_map.update({str(symbol): symbol for symbol in x_symbols})
    parsed = sympy.sympify(substituted, locals=locals_map)
    replacements = {
        symbol: (sympy.Symbol(feature) - float(normalization["x_mu"][index]))
        / float(normalization["x_scale"][index])
        for index, (symbol, feature) in enumerate(zip(x_symbols, features))
    }
    physical = float(normalization["y_mu"]) + float(normalization["y_scale"]) * parsed.xreplace(replacements)
    return str(physical)


def run_llm_sr(
    dataset: str,
    seed: int,
    budget_sec: float,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    eta: float,
) -> dict[str, Any]:
    """Run the official LLM-SR search and select candidates on validation."""
    official_root = Path(os.environ.get("OFFICIAL_LLMSR_ROOT", "/home/liuyihong/LLM-SR")).resolve()
    adapter_path = ROOT / "evaluation_suites" / "official_llmsr_fourbench" / "run_official_llmsr_fourbench.py"
    if not (official_root / "llmsr").is_dir():
        raise FileNotFoundError(f"official LLM-SR source tree not found: {official_root}")
    os.environ["OFFICIAL_LLMSR_ROOT"] = str(official_root)
    if str(official_root) not in sys.path:
        sys.path.insert(0, str(official_root))
    module_spec = importlib.util.spec_from_file_location("pse_realworld_official_llmsr", adapter_path)
    if module_spec is None or module_spec.loader is None:
        raise ImportError(f"cannot load LLM-SR adapter: {adapter_path}")
    adapter = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(adapter)

    normalized_train, normalized_val, normalized_test, normalization = adapter.normalize_splits(train, val, test)
    specification = adapter.make_case_spec(
        "realworld",
        {"case_name": dataset},
        normalized_train,
        max_params=10,
    )
    x_train = normalized_train[features].to_numpy(dtype=float)
    y_train = normalized_train["y"].to_numpy(dtype=float)
    inputs = {"data": {"inputs": x_train, "outputs": y_train}}
    class DeadlineVLLMChatLLM(adapter.OfficialVLLMChatLLM):
        """Official API sampler with deterministic seeds and a call-boundary deadline."""

        deadline = math.inf
        repeat_seed = int(seed)
        request_index = 0

        def __init__(self, samples_per_prompt: int) -> None:
            adapter.official_sampler.LLM.__init__(self, samples_per_prompt)
            from openai import OpenAI

            self._client = OpenAI(base_url=self.api_base, api_key=self.api_key, max_retries=0)

        @staticmethod
        def clean_body(text: str, config: Any) -> str:
            text = str(text or "").strip()
            fenced = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
            if fenced:
                text = fenced.group(1).strip()
            body = adapter.official_sampler._extract_body(text, config)
            if re.search(r"^\s*def\s+", text, flags=re.MULTILINE):
                return body
            lines = body.splitlines()
            first_return = next((i for i, line in enumerate(lines) if line.lstrip().startswith("return ")), None)
            if first_return is not None:
                lines = lines[first_return:]
            return "\n".join(line if line.startswith("    ") else "    " + line for line in lines if line.strip()) + "\n"

        def draw_samples(self, prompt: str, config: Any) -> list[str]:
            system = (
                "You are running the official LLM-SR algorithm. Complete only the Python body "
                "of the equation function. Use numpy operations, provided variables, and params "
                "for constants. Start directly with return and do not use Markdown fences."
            )
            user_prompt = "Complete the function below with only its Python body.\n" + prompt
            samples: list[str] = []
            for _ in range(self._samples_per_prompt):
                remaining = self.deadline - time.time()
                if remaining <= 0:
                    raise TimeoutError(f"LLM-SR search exceeded {budget_sec:.1f} seconds")
                request_seed = self.repeat_seed * 100000 + self.request_index
                self.request_index += 1
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    seed=request_seed,
                    timeout=max(1.0, min(self.request_timeout_sec, remaining)),
                )
                samples.append(self.clean_body(response.choices[0].message.content or "", config))
            return samples

    DeadlineVLLMChatLLM.api_base = os.environ.get("OFFICIAL_LLMSR_API_BASE", "http://127.0.0.1:18080/v1")
    DeadlineVLLMChatLLM.api_key = os.environ.get("OFFICIAL_LLMSR_API_KEY", "EMPTY")
    DeadlineVLLMChatLLM.model = os.environ.get("OFFICIAL_LLMSR_MODEL", "llm-baseline-qwen2.5-32b")
    DeadlineVLLMChatLLM.temperature = float(os.environ.get("OFFICIAL_LLMSR_TEMPERATURE", "0.8"))
    DeadlineVLLMChatLLM.max_tokens = int(os.environ.get("OFFICIAL_LLMSR_MAX_TOKENS", "512"))
    DeadlineVLLMChatLLM.request_timeout_sec = min(float(budget_sec), 60.0)
    llmsr_config = adapter.official_config.Config(
        num_samplers=1,
        num_evaluators=1,
        samples_per_prompt=int(os.environ.get("OFFICIAL_LLMSR_SAMPLES_PER_PROMPT", "4")),
        evaluate_timeout_seconds=min(20, max(2, int(budget_sec // 4))),
        use_api=False,
    )
    class_config = adapter.official_config.ClassConfig(
        llm_class=DeadlineVLLMChatLLM,
        sandbox_class=adapter.official_evaluator.LocalSandbox,
    )

    timed_out = False
    started = time.time()
    DeadlineVLLMChatLLM.deadline = started + float(budget_sec)
    with TemporaryDirectory(prefix=f"pse_realworld_llmsr_{dataset}_{seed}_") as temporary_dir:
        log_dir = Path(temporary_dir) / "official_log"
        try:
            adapter.official_pipeline.main(
                specification=specification,
                inputs=inputs,
                config=llmsr_config,
                max_sample_nums=100000,
                class_config=class_config,
                log_dir=str(log_dir),
            )
        except TimeoutError:
            timed_out = True
        except Exception as exc:
            if time.time() >= DeadlineVLLMChatLLM.deadline or type(exc).__name__ in {
                "APITimeoutError",
                "ReadTimeout",
            }:
                timed_out = True
            else:
                raise

        samples = adapter.load_samples_in_generation_order(log_dir / "samples")
        scored: list[dict[str, Any]] = []
        for sample in samples:
            try:
                predictions = adapter.fit_predict(
                    sample["function"],
                    normalized_train,
                    normalized_val,
                    normalized_test,
                    train,
                    val,
                    test,
                    normalization,
                    max_params=10,
                )
                val_mse = float(np.mean((predictions["y_val"] - predictions["pred_val"]) ** 2))
                raw_expression = adapter.extract_return_expression(sample["function"])
                expression = physical_llmsr_expression(
                    raw_expression,
                    predictions["optimized_params"],
                    features,
                    normalization,
                )
                complexity = expression_complexity(expression, features).get("expr_complexity")
                if complexity is None or not math.isfinite(val_mse):
                    continue
                scored.append({
                    "expression": expression,
                    "official_function": sample["function"],
                    "validation_mse": val_mse,
                    "expression_complexity": int(complexity),
                    "validation_reward": float(eta ** int(complexity) / (1.0 + math.sqrt(max(0.0, val_mse)))),
                    "sample_order": sample.get("sample_order"),
                    "official_score": sample.get("official_score"),
                })
            except Exception as exc:
                scored.append({
                    "expression": adapter.extract_return_expression(sample.get("function", "")),
                    "official_function": sample.get("function"),
                    "status": "invalid",
                    "error": repr(exc),
                    "sample_order": sample.get("sample_order"),
                })

    valid_scored = [candidate for candidate in scored if candidate.get("validation_reward") is not None]
    valid_scored.sort(key=lambda item: (-item["validation_reward"], item["expression_complexity"], item["validation_mse"]))
    selected = valid_scored[0] if valid_scored else None
    if selected is None:
        raise RuntimeError(f"LLM-SR produced no valid candidate from {len(samples)} samples")
    return {
        "method": "llm_sr",
        "best_expr": selected["expression"],
        "runtime_sec": time.time() - started,
        "candidate_count": len(samples),
        "valid_candidate_count": len(valid_scored),
        "pareto_candidates": scored,
        "allowed_operators": "model-generated NumPy expressions",
        "baseline_source": "official LLM-SR adapter used by the VEGA-SR 895-task comparison",
        "baseline_hyperparameters": {
            "official_root": str(official_root),
            "official_revision": "41c212312df6c16d936c9cb395356a62774c47e3",
            "model": DeadlineVLLMChatLLM.model,
            "samples_per_prompt": llmsr_config.samples_per_prompt,
            "temperature": DeadlineVLLMChatLLM.temperature,
            "max_tokens": DeadlineVLLMChatLLM.max_tokens,
            "request_seed_rule": "repeat_seed*100000 + request_index",
            "time_budget_seconds": float(budget_sec),
        },
        "generated_specification": specification,
        "normalization": {
            "x_mu": normalization["x_mu"].tolist(),
            "x_scale": normalization["x_scale"].tolist(),
            "y_mu": float(normalization["y_mu"]),
            "y_scale": float(normalization["y_scale"]),
        },
        "search_timed_out_at_budget": timed_out,
    }
def run_linear_ls(
    train: pd.DataFrame,
    val: pd.DataFrame,
    features: list[str],
    eta: float,
) -> dict[str, Any]:
    """Ordinary linear model as a deterministic lower-bound ablation."""
    started = time.time()
    columns = [(name, train[name].to_numpy(dtype=float)) for name in features]
    expression = fitted_linear_expression(columns, train["y"].to_numpy(dtype=float))
    selected, scored = score_candidates([expression], val, features, eta)
    return {
        "method": "linear_ls",
        "best_expr": selected["expression"] if selected else None,
        "runtime_sec": time.time() - started,
        "candidate_count": 1,
        "valid_candidate_count": len(scored),
        "pareto_candidates": scored,
        "allowed_operators": ["+", "*"],
        "baseline_source": "ordinary least squares sanity baseline",
        "baseline_hyperparameters": {"coefficient_fitter": "numpy.linalg.lstsq"},
    }


def fitted_linear_expression(columns: list[tuple[str, np.ndarray]], y: np.ndarray) -> str:
    matrix = np.column_stack([values for _, values in columns] + [np.ones(len(y))])
    coefficients, *_ = np.linalg.lstsq(matrix, y, rcond=None)
    terms = [f"({float(coefficient):.17g})*({name})" for (name, _), coefficient in zip(columns, coefficients[:-1])]
    terms.append(f"({float(coefficients[-1]):.17g})")
    return " + ".join(terms)


def run_physics_ls(
    dataset: str,
    train: pd.DataFrame,
    val: pd.DataFrame,
    features: list[str],
    eta: float,
) -> dict[str, Any]:
    if dataset != "emps":
        raise ValueError("physics_ls is defined only for EMPS")
    q = train["q"].to_numpy(dtype=float)
    qdot = train["qdot"].to_numpy(dtype=float)
    tau = train["tau"].to_numpy(dtype=float)
    y = train["y"].to_numpy(dtype=float)
    started = time.time()
    expressions = [
        fitted_linear_expression([("qdot", qdot), ("tau", tau), ("sign(qdot)", np.sign(qdot))], y),
        fitted_linear_expression([("q", q), ("qdot", qdot), ("tau", tau), ("sign(qdot)", np.sign(qdot))], y),
        fitted_linear_expression([("qdot", qdot), ("tau", tau), ("sign(qdot)", np.sign(qdot)), ("Abs(qdot)", np.abs(qdot))], y),
    ]
    runtime = time.time() - started
    selected, scored = score_candidates(expressions, val, features, eta)
    return {
        "method": "physics_ls",
        "best_expr": selected["expression"] if selected else None,
        "runtime_sec": runtime,
        "candidate_count": len(expressions),
        "valid_candidate_count": len(scored),
        "pareto_candidates": scored,
        "allowed_operators": ["+", "*", "sign", "abs"],
        "baseline_source": "PSE Newton/friction prior, linear least-squares ablation",
        "baseline_hyperparameters": {
            "templates": 3,
            "coefficient_fitter": "numpy.linalg.lstsq",
        },
    }


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if args.seed not in config["execution"]["repeat_seeds"]:
        raise ValueError(f"seed {args.seed} is not declared in repeat_seeds")
    budget_sec = float(args.budget_sec or config["execution"]["case_budget_sec"])
    train, val, test, alignment, full = load_splits(config_path, config, args.dataset)
    features = list(config["datasets"][args.dataset]["features"])
    eta = float(config["datasets"][args.dataset]["pse_reference"]["reward_eta"])
    random.seed(args.seed)
    np.random.seed(args.seed)
    result: dict[str, Any] = {
        "schema_version": int(config["schema_version"]),
        "method": args.method,
        "dataset": args.dataset,
        "dataset_display_name": config["datasets"][args.dataset]["display_name"],
        "protocol": config["datasets"][args.dataset]["default_protocol"],
        "repeat_seed": int(args.seed),
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "test_used_for_selection": False,
        "selection_metric": "validation_reward_eta_0.99",
        "search_budget_sec": budget_sec,
        "started_at_unix": time.time(),
        "config_path": str(config_path),
    }
    try:
        if args.method == "pysr":
            method_result = run_pysr(args.dataset, args.seed, budget_sec, train, val, features, eta)
        elif args.method == "operon":
            method_result = run_operon(args.dataset, args.seed, budget_sec, train, val, features, eta)
        elif args.method == "gplearn":
            method_result = run_gplearn(args.dataset, args.seed, budget_sec, train, val, features, eta)
        elif args.method == "dso":
            method_result = run_dso(args.seed, budget_sec, train, val, test, features)
        elif args.method == "llm_sr":
            method_result = run_llm_sr(args.dataset, args.seed, budget_sec, train, val, test, features, eta)
        elif args.method == "linear_ls":
            method_result = run_linear_ls(train, val, features, eta)
        else:
            method_result = run_physics_ls(args.dataset, train, val, features, eta)
        result.update(method_result)
        expression = result.get("best_expr")
        result.update(expression_complexity(expression, features))
        precomputed = result.pop("_precomputed_split_metrics", None)
        if precomputed is not None:
            result.update(precomputed)
        else:
            for split_name, frame in (("train", train), ("val", val), ("test", test)):
                for metric_name, value in metrics_for_expression(expression, frame, features).items():
                    result[f"{split_name}_{metric_name}"] = value
        if args.dataset == "emps":
            result["pse_discovery_half_mse"] = metrics_for_expression(expression, alignment, features)["mse"]
        else:
            result["pse_full_data_mse"] = metrics_for_expression(expression, full, features)["mse"]
        # A validation-selected expression can be outside its numerical domain
        # on held-out test inputs (for example log(q) when test q < 0). Do not
        # use the test split to fall back to another candidate. Record the
        # invalid prediction explicitly and keep the selection protocol intact.
        result["test_prediction_valid"] = result.get("test_mse") is not None
        result["test_prediction_failure"] = (
            None if result["test_prediction_valid"] else "non_finite_or_unevaluable_prediction"
        )
        result["status"] = "ok"
    except Exception as exc:
        result.update({"status": "error", "error": repr(exc)})
        raise
    finally:
        result["finished_at_unix"] = time.time()
        result["total_wall_sec"] = result["finished_at_unix"] - result["started_at_unix"]
        output = Path(args.output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(json.dumps(json_safe(result), ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(output)
    print(json.dumps({key: result.get(key) for key in ("method", "dataset", "repeat_seed", "status", "best_expr", "test_mse", "test_r2", "runtime_sec")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
