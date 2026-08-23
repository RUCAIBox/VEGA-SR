#!/usr/bin/env python3
"""Run one strict-split baseline for the PSE real-world comparison.

PySR and Operon follow the operator libraries and 90-second settings released
with PSE, but candidates are selected on our validation split rather than on
the final test split. ``physics_ls`` is an EMPS-only prior ablation: it fits the
same Newton/friction templates supplied to VEGA-SR without doing symbolic
search.
"""

from __future__ import annotations

import argparse
import json
import math
import random
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
    parser.add_argument("--method", required=True, choices=("pysr", "operon", "physics_ls"))
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
        else:
            method_result = run_physics_ls(args.dataset, train, val, features, eta)
        result.update(method_result)
        expression = result.get("best_expr")
        result.update(expression_complexity(expression, features))
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
