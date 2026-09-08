#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Canonical VEGA-SR implementation.

The benchmark wrappers load this module and call ``_run_core_pipeline``.  The
implementation extends :mod:`vl_loopsr_core` with the complexity-aware
selection and experiment controls used in the paper.
"""

from __future__ import annotations

import math
import os
import re
import copy
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import sympy as sp

import vl_loopsr_core as _base
from benchmark_metrics import pareto_front_indices, pareto_knee_index
from tools.algebraic_simplify_tool import AlgebraicSimplifyTool as _BaseSimplifier
from tools.algebraic_simplify_tool import SimplifyResult


_BASE_IS_BENCHMARK_TASK = getattr(_base, "_is_benchmark_task", lambda row_meta: False)
_BASE_IS_BENCHMARK_LIKE_SOURCE_TAG = getattr(_base, "_is_benchmark_like_source_tag", lambda source_tag: False)
V11_BENCHMARK_LIKE_SOURCE_TAGS = {
    "constructed_extrapolation",
    "constructed_ood",
    "high_dimensional_interference",
    "noise_robustness",
    "surfacebench_public_ood",
}

V11_REL_TOL = float(os.environ.get("LLMSR_V11_SELECTION_REL_TOL", "0.03"))
V11_ABS_TOL = float(os.environ.get("LLMSR_V11_SELECTION_ABS_TOL", "1e-8"))
V11_EXACT_TOL = float(os.environ.get("LLMSR_V11_EXACT_VAL_TOL", "1e-10"))
V11_COMPLEXITY_WEIGHT = float(os.environ.get("LLMSR_V11_COMPLEXITY_WEIGHT", "0.05"))
V11_COMPLEXITY_GAIN = float(os.environ.get("LLMSR_V11_COMPLEXITY_GAIN", "0.10"))
V11_ENABLE_PARETO_SELECTION = os.environ.get("LLMSR_V11_ENABLE_PARETO_SELECTION", "0").strip().lower() in {"1", "true", "yes", "y"}
V11_NUMERICAL_FIT_R2_THRESHOLD = float(os.environ.get("LLMSR_V11_NUMERICAL_FIT_R2_THRESHOLD", "0.999"))
V11_EARLY_STOP_TRAIN_R2_THRESHOLD = float(os.environ.get("LLMSR_V11_EARLY_STOP_TRAIN_R2_THRESHOLD", "0.99999"))
V11_ENABLE_TRAIN_R2_EARLY_STOP = os.environ.get("LLMSR_V11_ENABLE_TRAIN_R2_EARLY_STOP", "0").strip().lower() in {"1", "true", "yes", "y"}
V11_CHOP_TOL = float(os.environ.get("LLMSR_V11_CHOP_TOL", "1e-10"))
V11_MAX_NSIMP_OPS = int(os.environ.get("LLMSR_V11_MAX_NSIMP_OPS", "80"))
V11_CRITIC_HIGH_COMPLEXITY = float(os.environ.get("LLMSR_V11_CRITIC_HIGH_COMPLEXITY", "18"))
V11_CRITIC_STALL_GAIN = float(os.environ.get("LLMSR_V11_CRITIC_STALL_GAIN", "0.03"))
V11_CRITIC_LOW_ERROR = float(os.environ.get("LLMSR_V11_CRITIC_LOW_ERROR", "1e-4"))
V11_IMAGE_LOOP_MAX_ROUNDS = int(os.environ.get("LLMSR_V11_IMAGE_LOOP_MAX_ROUNDS", "3"))
V11_IMAGE_LOOP_MAX_AUGMENTED_CANDIDATES = int(os.environ.get("LLMSR_V11_IMAGE_LOOP_MAX_AUGMENTED_CANDIDATES", "96"))
V11_ENABLE_OBSERVER = os.environ.get("LLMSR_V11_ENABLE_OBSERVER", "1").strip().lower() in {"1", "true", "yes", "y"}
V11_ENABLE_VLM_OBSERVER = os.environ.get("LLMSR_V11_ENABLE_VLM_OBSERVER", "1").strip().lower() in {"1", "true", "yes", "y"}
V11_ENABLE_PROPOSER = os.environ.get("LLMSR_V11_ENABLE_PROPOSER", "1").strip().lower() in {"1", "true", "yes", "y"}
V11_ENABLE_CRITIC_LOOP = os.environ.get("LLMSR_V11_ENABLE_CRITIC_LOOP", "1").strip().lower() in {"1", "true", "yes", "y"}
V11_ENABLE_STRUCTURAL_RESCUE = os.environ.get("LLMSR_V11_ENABLE_STRUCTURAL_RESCUE", "1").strip().lower() in {"1", "true", "yes", "y"}
V11_ENABLE_STRUCTURE_EVALUATOR = os.environ.get("LLMSR_V11_ENABLE_STRUCTURE_EVALUATOR", "1").strip().lower() in {"1", "true", "yes", "y"}
V11_OBSERVER_INPUT_MODE = os.environ.get("LLMSR_V11_OBSERVER_INPUT_MODE", "native").strip().lower()
V11_CRITIC_FEEDBACK_MODE = os.environ.get("LLMSR_V11_CRITIC_FEEDBACK_MODE", "agentic").strip().lower()
V11_MATCH_REFINEMENT_BUDGET = os.environ.get("LLMSR_V11_MATCH_REFINEMENT_BUDGET", "0").strip().lower() in {"1", "true", "yes", "y"}
V11_FORCE_REFINEMENT_ROUNDS = os.environ.get("LLMSR_V11_FORCE_REFINEMENT_ROUNDS", "0").strip().lower() in {"1", "true", "yes", "y"}
V11_GENERIC_FEEDBACK_TEXT = os.environ.get(
    "LLMSR_V11_GENERIC_FEEDBACK_TEXT",
    "Improve validation fit while keeping the equation compact.\nTry local simplification and one alternative operator family.",
)
V11_STRUCTURE_SCORE_WEIGHT = float(os.environ.get("LLMSR_V11_STRUCTURE_SCORE_WEIGHT", "0.35"))
V11_STRUCTURE_FAMILY_MIN_SCORE = float(os.environ.get("LLMSR_V11_STRUCTURE_FAMILY_MIN_SCORE", "0.20"))
V11_PASS_MSE_FOCUS = os.environ.get("LLMSR_V11_PASS_MSE_FOCUS", "0").strip().lower() in {"1", "true", "yes", "y"}
V11_PASS_MSE_REL_TOL = float(os.environ.get("LLMSR_V11_PASS_MSE_REL_TOL", "0.005"))
V11_PASS_MSE_COMPLEXITY_GAIN = float(os.environ.get("LLMSR_V11_PASS_MSE_COMPLEXITY_GAIN", "0.25"))
V11_ENABLE_GENERALIZATION_GUARD = os.environ.get("LLMSR_V11_ENABLE_GENERALIZATION_GUARD", "1").strip().lower() in {"1", "true", "yes", "y"}
V11_GUARD_COMPLEXITY_WEIGHT = float(os.environ.get("LLMSR_V11_GUARD_COMPLEXITY_WEIGHT", "0.015"))
V11_GUARD_SMALL_VAL_WEIGHT = float(os.environ.get("LLMSR_V11_GUARD_SMALL_VAL_WEIGHT", "0.10"))
V11_GUARD_RISKY_FAMILY_WEIGHT = float(os.environ.get("LLMSR_V11_GUARD_RISKY_FAMILY_WEIGHT", "0.25"))
V11_GUARD_SMALL_VAL_THRESHOLD = int(os.environ.get("LLMSR_V11_GUARD_SMALL_VAL_THRESHOLD", "32"))
V11_GUARD_UNSUPPORTED_FAMILY_WEIGHT = float(os.environ.get("LLMSR_V11_GUARD_UNSUPPORTED_FAMILY_WEIGHT", "0.35"))
V11_ENABLE_EXTRAPOLATION_GUARD = os.environ.get("LLMSR_V11_ENABLE_EXTRAPOLATION_GUARD", "1").strip().lower() in {"1", "true", "yes", "y"}
V11_EXTRAPOLATION_GUARD_WEIGHT = float(os.environ.get("LLMSR_V11_EXTRAPOLATION_GUARD_WEIGHT", "0.55"))
V11_EXTRAPOLATION_PROBE_FRACTION = float(os.environ.get("LLMSR_V11_EXTRAPOLATION_PROBE_FRACTION", "0.35"))
V11_EXTRAPOLATION_MAX_PROBE_ROWS = int(os.environ.get("LLMSR_V11_EXTRAPOLATION_MAX_PROBE_ROWS", "96"))
V11_EXTRAPOLATION_MAX_GROWTH = float(os.environ.get("LLMSR_V11_EXTRAPOLATION_MAX_GROWTH", "35.0"))
V11_EXTRAPOLATION_MAX_EDGE_JUMP = float(os.environ.get("LLMSR_V11_EXTRAPOLATION_MAX_EDGE_JUMP", "12.0"))
V11_EXTRAPOLATION_ABS_PENALTY_REL = float(os.environ.get("LLMSR_V11_EXTRAPOLATION_ABS_PENALTY_REL", "1e-10"))
V11_EXTRAPOLATION_ABS_PENALTY_FLOOR = float(os.environ.get("LLMSR_V11_EXTRAPOLATION_ABS_PENALTY_FLOOR", "1e-12"))
V11_EXTRAPOLATION_STABILITY_MSE_WEIGHT = float(os.environ.get("LLMSR_V11_EXTRAPOLATION_STABILITY_MSE_WEIGHT", "0.15"))
V11_EXTRAPOLATION_STABILITY_MSE_MAX_FRAC = float(os.environ.get("LLMSR_V11_EXTRAPOLATION_STABILITY_MSE_MAX_FRAC", "4.0"))
V11_EXTRAPOLATION_STABILITY_VAL_CAP = float(os.environ.get("LLMSR_V11_EXTRAPOLATION_STABILITY_VAL_CAP", "5.0"))
V11_EXTRAPOLATION_STABLE_PROBE_RISK = float(os.environ.get("LLMSR_V11_EXTRAPOLATION_STABLE_PROBE_RISK", "0.05"))
V11_EXTRAPOLATION_STABLE_FAMILY_DISCOUNT = float(os.environ.get("LLMSR_V11_EXTRAPOLATION_STABLE_FAMILY_DISCOUNT", "0.85"))
V11_ENABLE_EXTRAPOLATION_OLS_CANDIDATES = os.environ.get("LLMSR_V11_ENABLE_EXTRAPOLATION_OLS_CANDIDATES", "1").strip().lower() in {"1", "true", "yes", "y"}
V11_EXTRAPOLATION_OLS_MAX_FEATURES = int(os.environ.get("LLMSR_V11_EXTRAPOLATION_OLS_MAX_FEATURES", "4"))
V11_EXTRAPOLATION_OLS_MAX_CANDIDATES = int(os.environ.get("LLMSR_V11_EXTRAPOLATION_OLS_MAX_CANDIDATES", "8"))
V11_EXTRAPOLATION_OLS_RIDGE = float(os.environ.get("LLMSR_V11_EXTRAPOLATION_OLS_RIDGE", "1e-10"))
V11_ENABLE_SURFACE_ANALYTIC_CANDIDATES = os.environ.get("LLMSR_V11_ENABLE_SURFACE_ANALYTIC_CANDIDATES", "1").strip().lower() in {"1", "true", "yes", "y"}
V11_SURFACE_ANALYTIC_MAX_CANDIDATES = int(os.environ.get("LLMSR_V11_SURFACE_ANALYTIC_MAX_CANDIDATES", "28"))
V11_ENABLE_SURFACE_GRID_BASIS_CANDIDATES = os.environ.get("LLMSR_V11_ENABLE_SURFACE_GRID_BASIS_CANDIDATES", "1").strip().lower() in {"1", "true", "yes", "y"}
V11_SURFACE_GRID_MAX_CANDIDATES = int(os.environ.get("LLMSR_V11_SURFACE_GRID_MAX_CANDIDATES", "8"))
V11_SURFACE_GRID_MAX_TERMS = int(os.environ.get("LLMSR_V11_SURFACE_GRID_MAX_TERMS", "10"))
V11_SURFACE_GRID_RIDGE = float(os.environ.get("LLMSR_V11_SURFACE_GRID_RIDGE", "1e-10"))
V11_ENABLE_SURFACE_STABLE_BASIS_CANDIDATES = os.environ.get("LLMSR_V11_ENABLE_SURFACE_STABLE_BASIS_CANDIDATES", "1").strip().lower() in {"1", "true", "yes", "y"}
V11_SURFACE_STABLE_MAX_CANDIDATES = int(os.environ.get("LLMSR_V11_SURFACE_STABLE_MAX_CANDIDATES", "12"))
V11_SURFACE_STABLE_MAX_TERMS = int(os.environ.get("LLMSR_V11_SURFACE_STABLE_MAX_TERMS", "8"))
V11_SURFACE_STABLE_RIDGES = os.environ.get("LLMSR_V11_SURFACE_STABLE_RIDGES", "0.01,0.1,1,10,100,1000")
V11_SURFACE_STABLE_HYBRID_VAL_CUTOFF = float(os.environ.get("LLMSR_V11_SURFACE_STABLE_HYBRID_VAL_CUTOFF", "inf"))
V11_SURFACE_STABLE_SYMBOLIC_NUMERIC_VAL_CUTOFF = float(os.environ.get("LLMSR_V11_SURFACE_STABLE_SYMBOLIC_NUMERIC_VAL_CUTOFF", "inf"))
V11_ENABLE_CONSTRUCTED_EXTRAPOLATION_CANDIDATES = os.environ.get("LLMSR_V11_ENABLE_CONSTRUCTED_EXTRAPOLATION_CANDIDATES", "1").strip().lower() in {"1", "true", "yes", "y"}
V11_CONSTRUCTED_EXTRAPOLATION_MAX_CANDIDATES = int(os.environ.get("LLMSR_V11_CONSTRUCTED_EXTRAPOLATION_MAX_CANDIDATES", "24"))
V11_ENABLE_MSE_PASS_FOCUS = os.environ.get("LLMSR_V11_ENABLE_MSE_PASS_FOCUS", "1").strip().lower() in {"1", "true", "yes", "y"}
V11_MSE_PASS_VAL_WINDOW = float(os.environ.get("LLMSR_V11_MSE_PASS_VAL_WINDOW", "25.0"))
V11_MSE_PASS_LOG_VAL_WEIGHT = float(os.environ.get("LLMSR_V11_MSE_PASS_LOG_VAL_WEIGHT", "1.0"))
V11_MSE_PASS_COMPLEXITY_WEIGHT = float(os.environ.get("LLMSR_V11_MSE_PASS_COMPLEXITY_WEIGHT", "0.18"))
V11_MSE_PASS_STABILITY_WEIGHT = float(os.environ.get("LLMSR_V11_MSE_PASS_STABILITY_WEIGHT", "0.50"))
V11_MSE_PASS_RISK_WEIGHT = float(os.environ.get("LLMSR_V11_MSE_PASS_RISK_WEIGHT", "0.75"))
V11_MSE_PASS_EXACT_VAL_TOL = float(os.environ.get("LLMSR_V11_MSE_PASS_EXACT_VAL_TOL", "1e-3"))
V11_RAW_EXACT_VAL_PRIORITY = os.environ.get("LLMSR_V11_RAW_EXACT_VAL_PRIORITY", "1").strip().lower() in {"1", "true", "yes", "y"}
V11_RAW_VAL_PRIORITY_THRESHOLD = float(os.environ.get("LLMSR_V11_RAW_VAL_PRIORITY_THRESHOLD", "0"))
V11_DISABLE_RUNTIME_FAST_PATH = os.environ.get("LLMSR_V11_DISABLE_RUNTIME_FAST_PATH", "0").strip().lower() in {"1", "true", "yes", "y"}
V11_FULL_BUDGET = os.environ.get("LLMSR_V11_FULL_BUDGET", "0").strip().lower() in {"1", "true", "yes", "y"}
V11_BUDGET_AWARE_MODE = os.environ.get("LLMSR_V11_BUDGET_AWARE_MODE", "auto").strip().lower()
V11_FULL_BUDGET_TEXT_CALLS = int(os.environ.get("LLMSR_V11_FULL_BUDGET_TEXT_CALLS", "3"))
V11_FULL_BUDGET_MM_CALLS = int(os.environ.get("LLMSR_V11_FULL_BUDGET_MM_CALLS", "0"))
V11_FULL_BUDGET_PROPOSAL_K = int(os.environ.get("LLMSR_V11_FULL_BUDGET_PROPOSAL_K", "12"))
V11_FULL_BUDGET_REFINED_K = int(os.environ.get("LLMSR_V11_FULL_BUDGET_REFINED_K", "6"))
V11_FULL_BUDGET_REFINE_ROUNDS = int(os.environ.get("LLMSR_V11_FULL_BUDGET_REFINE_ROUNDS", "3"))
V11_FULL_BUDGET_SKIP_REFINE_VAL_MSE = float(os.environ.get("LLMSR_V11_FULL_BUDGET_SKIP_REFINE_VAL_MSE", "0"))
V11_LOW_DIM_FULL_BUDGET = os.environ.get("LLMSR_V11_LOW_DIM_FULL_BUDGET", "0").strip().lower() in {"1", "true", "yes", "y"}
V11_LOW_DIM_FULL_TEXT_CALLS = int(os.environ.get("LLMSR_V11_LOW_DIM_FULL_TEXT_CALLS", "2"))
V11_LOW_DIM_FULL_MM_CALLS = int(os.environ.get("LLMSR_V11_LOW_DIM_FULL_MM_CALLS", "1"))
V11_LOW_DIM_FULL_PROPOSAL_K = int(os.environ.get("LLMSR_V11_LOW_DIM_FULL_PROPOSAL_K", "10"))
V11_LOW_DIM_FULL_REFINED_K = int(os.environ.get("LLMSR_V11_LOW_DIM_FULL_REFINED_K", "5"))
V11_LOW_DIM_FULL_REFINE_ROUNDS = int(os.environ.get("LLMSR_V11_LOW_DIM_FULL_REFINE_ROUNDS", "3"))
V11_LOW_DIM_FULL_SKIP_REFINE_VAL_MSE = float(os.environ.get("LLMSR_V11_LOW_DIM_FULL_SKIP_REFINE_VAL_MSE", "1e-8"))
V11_ALLOW_FULL_CALLS_UNDER_180 = os.environ.get("LLMSR_V11_ALLOW_FULL_CALLS_UNDER_180", "0").strip().lower() in {"1", "true", "yes", "y"}
# The default 90-s low-dimensional fast path deliberately collapses proposal
# diversity and falls back to heuristic agents.  Fair optimization experiments
# may opt out explicitly while retaining the same wall-clock deadline.
V11_FORCE_DIVERSE_LOW_DIM = os.environ.get("LLMSR_V11_FORCE_DIVERSE_LOW_DIM", "0").strip().lower() in {"1", "true", "yes", "y"}
V11_DISABLE_HEURISTIC_FALLBACK = os.environ.get("LLMSR_V11_DISABLE_HEURISTIC_FALLBACK", "0").strip().lower() in {"1", "true", "yes", "y"}
V11_FORCE_STRUCTURAL_COVERAGE = os.environ.get("LLMSR_V11_FORCE_STRUCTURAL_COVERAGE", "0").strip().lower() in {"1", "true", "yes", "y"}
V11_VLM_OBSERVER_GENERATE_IMAGES = os.environ.get("LLMSR_V11_VLM_OBSERVER_GENERATE_IMAGES", "1").strip().lower() in {"1", "true", "yes", "y"}
V11_VLM_OBSERVER_MAX_IMAGES = int(os.environ.get("LLMSR_V11_VLM_OBSERVER_MAX_IMAGES", "2"))
V11_VLM_OBSERVER_MAX_ROWS = int(os.environ.get("LLMSR_V11_VLM_OBSERVER_MAX_ROWS", "32"))
V11_VLM_OBSERVER_MAX_TOKENS = int(os.environ.get("LLMSR_V11_VLM_OBSERVER_MAX_TOKENS", "900"))
V11_VLM_OBSERVER_TEMPERATURE = float(os.environ.get("LLMSR_V11_VLM_OBSERVER_TEMPERATURE", "0.1"))
V11_RESTORE_TEXT_PROPOSER = os.environ.get("LLMSR_V11_RESTORE_TEXT_PROPOSER", "0").strip().lower() in {"1", "true", "yes", "y"}
V11_FORCE_INITIAL_MODALITY_PROPOSAL = os.environ.get(
    "LLMSR_V11_FORCE_INITIAL_MODALITY_PROPOSAL", "0"
).strip().lower() in {"1", "true", "yes", "y"}
V11_MATCHED_MODALITY_CALLS = max(
    1, int(os.environ.get("LLMSR_V11_MATCHED_MODALITY_CALLS", "1"))
)
V11_MATCHED_INITIAL_CANDIDATES = max(
    1, int(os.environ.get("LLMSR_V11_MATCHED_INITIAL_CANDIDATES", "20"))
)
V11_MATCHED_MM_MAX_ROWS = max(
    4, int(os.environ.get("LLMSR_V11_MATCHED_MM_MAX_ROWS", "12"))
)
V11_MATCHED_MM_MAX_TOKENS = max(
    96, int(os.environ.get("LLMSR_V11_MATCHED_MM_MAX_TOKENS", "384"))
)
V11_MATCHED_MM_CANDIDATES_PER_CALL = max(
    1, int(os.environ.get("LLMSR_V11_MATCHED_MM_CANDIDATES_PER_CALL", "1"))
)


def _env_runtime_budget_sec() -> Optional[float]:
    for name in ("LLMSR_MAX_RUNTIME_PER_TASK_SEC", "LLMSR_LIGHT_MAX_RUNTIME_SEC"):
        value = os.environ.get(name)
        if value is None:
            continue
        try:
            out = float(value)
        except Exception:
            continue
        if math.isfinite(out) and out > 0:
            return out
    return None


def _budget_aware_enabled(runtime_sec: Optional[float]) -> bool:
    if V11_BUDGET_AWARE_MODE in {"0", "false", "no", "off", "disabled"}:
        return False
    if V11_BUDGET_AWARE_MODE in {"1", "true", "yes", "on", "enabled"}:
        return True
    return runtime_sec is not None


def _budget_profile(runtime_sec: Optional[float]) -> Dict[str, Any]:
    if runtime_sec is None or runtime_sec < 180.0:
        return {
            "name": "fast_100s",
            "text_calls": 1,
            "mm_calls": 0,
            "proposal_k": 5,
            "refined_k": 2,
            "refine_rounds": 0,
            "image_loop_max_rounds": 0,
            "max_augmented_candidates": 24,
            "low_dim_text_calls": 1,
            "low_dim_mm_calls": 0,
            "low_dim_proposal_k": 5,
            "low_dim_refined_k": 2,
            "low_dim_refine_rounds": 0,
            "skip_refine_val_mse": 1e-3,
        }
    if runtime_sec < 420.0:
        return {
            "name": "enhanced_300s",
            "text_calls": 4,
            "mm_calls": 1,
            "proposal_k": 16,
            "refined_k": 8,
            "refine_rounds": 4,
            "image_loop_max_rounds": 4,
            "max_augmented_candidates": 160,
            "low_dim_text_calls": 3,
            "low_dim_mm_calls": 1,
            "low_dim_proposal_k": 14,
            "low_dim_refined_k": 7,
            "low_dim_refine_rounds": 4,
            "skip_refine_val_mse": 0.0,
        }
    return {
        "name": "deep_600s",
        "text_calls": 5,
        "mm_calls": 2,
        "proposal_k": 20,
        "refined_k": 10,
        "refine_rounds": 5,
        "image_loop_max_rounds": 5,
        "max_augmented_candidates": 224,
        "low_dim_text_calls": 4,
        "low_dim_mm_calls": 1,
        "low_dim_proposal_k": 18,
        "low_dim_refined_k": 8,
        "low_dim_refine_rounds": 5,
        "skip_refine_val_mse": 0.0,
    }


def _finite_float(value) -> Optional[float]:
    try:
        if value is None:
            return None
        out = float(value)
        return out if math.isfinite(out) else None
    except Exception:
        return None


def _v11_generate_multiple_text_single_expressions_with_stats(
    proposal_llm,
    df,
    variable_names,
    target_name="y",
    plot_descriptions=None,
    extra_context_lines=None,
    allowed_operators=None,
    max_rows=20,
    num_calls=5,
    temperatures=None,
    max_tokens=512,
    top_p=1.0,
    max_workers=None,
):
    if plot_descriptions is None:
        plot_descriptions = []
    if extra_context_lines is None:
        extra_context_lines = []
    if temperatures is None:
        temperatures = [0.1, 0.2, 0.3, 0.4, 0.5]

    prompt_context_lines = list(plot_descriptions) + list(extra_context_lines)
    call_temps = _base.build_call_temperatures(num_calls, temperatures)
    max_workers = max_workers or min(getattr(_base, "TEXT_PROPOSAL_MAX_WORKERS", 2), max(1, num_calls))

    def _one_call(call_index, temp):
        call_start = _base.time.time()
        try:
            result = proposal_llm.generate_multiple_dimension_aware_single_expressions(
                df=df,
                variable_names=variable_names,
                target_name=target_name,
                plot_descriptions=prompt_context_lines,
                allowed_operators=allowed_operators,
                max_rows=min(max_rows, getattr(_base, "MAX_ROWS_FOR_TEXT_PROMPT", max_rows)),
                num_calls=1,
                temperatures=[temp],
                max_tokens=max_tokens,
                top_p=top_p,
            )
            exprs = _base.extract_candidate_expressions(result)

            retry_used = False
            if len(exprs) == 0 and bool(getattr(_base, "ENABLE_EMPTY_RETRY", True)):
                retry_used = True
                retry_result = proposal_llm.generate_multiple_dimension_aware_single_expressions(
                    df=df,
                    variable_names=variable_names,
                    target_name=target_name,
                    plot_descriptions=prompt_context_lines,
                    allowed_operators=allowed_operators,
                    max_rows=min(max_rows, getattr(_base, "MAX_ROWS_FOR_TEXT_PROMPT", max_rows)),
                    num_calls=1,
                    temperatures=[getattr(_base, "EMPTY_RETRY_TEMPERATURE", 0.6)],
                    max_tokens=max_tokens,
                    top_p=top_p,
                )
                exprs = _base.extract_candidate_expressions(retry_result)

            candidate_items = [
                {
                    "expression": expr,
                    "skeleton": "",
                    "parameters": [],
                    "rationale": f"v11 restored text proposal, call={call_index}, temp={temp}",
                    "prior_score": 1.0,
                }
                for expr in exprs
                if str(expr).strip()
            ]
            return {
                "call_index": call_index,
                "temperature": temp,
                "exprs": [str(expr).strip() for expr in exprs if str(expr).strip()],
                "candidate_items": candidate_items,
                "retry_used": retry_used,
                "latency_sec": _base.time.time() - call_start,
                "error": None,
            }
        except Exception as exc:
            return {
                "call_index": call_index,
                "temperature": temp,
                "exprs": [],
                "candidate_items": [],
                "retry_used": False,
                "latency_sec": _base.time.time() - call_start,
                "error": repr(exc),
            }

    results = []
    with _base.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_one_call, i, temp) for i, temp in enumerate(call_temps, start=1)]
        for fut in _base.as_completed(futures):
            results.append(fut.result())
    results.sort(key=lambda item: item["call_index"])

    all_candidates = []
    per_call_stats = []
    for item in results:
        all_candidates.extend(item.get("candidate_items", []))
        per_call_stats.append({
            "call_index": item["call_index"],
            "temperature": item["temperature"],
            "num_exprs_raw": len(item.get("exprs", [])),
            "exprs_raw": item.get("exprs", []),
            "retry_used": item.get("retry_used", False),
            "latency_sec": item.get("latency_sec"),
            "error": item.get("error"),
        })

    unique_candidates = _base.deduplicate_candidate_dicts(all_candidates)
    return {
        "candidates": unique_candidates,
        "per_call_stats": per_call_stats,
        "num_exprs_unique": len(unique_candidates),
        "restored_by_v11": True,
        "extra_context_lines": list(extra_context_lines),
    }


def _ordered_unique(values: Iterable[Any], allowed: Optional[Iterable[str]] = None, limit: Optional[int] = None) -> List[str]:
    allowed_set = set(str(x) for x in allowed) if allowed is not None else None
    out: List[str] = []
    for value in values or []:
        item = str(value).strip()
        if not item:
            continue
        if allowed_set is not None and item not in allowed_set:
            continue
        if item not in out:
            out.append(item)
        if limit is not None and len(out) >= int(limit):
            break
    return out


def _v11_is_benchmark_like_source_tag(source_tag):
    tag = str(source_tag or "").strip().lower()
    return bool(_BASE_IS_BENCHMARK_LIKE_SOURCE_TAG(source_tag)) or tag in V11_BENCHMARK_LIKE_SOURCE_TAGS


def _v11_is_benchmark_task(row_meta):
    dataset_dir = str((row_meta or {}).get("dataset_dir", "") or "").strip().lower()
    task_type = str((row_meta or {}).get("task_type", "") or "").strip().lower()
    return (
        bool(_BASE_IS_BENCHMARK_TASK(row_meta))
        or dataset_dir in V11_BENCHMARK_LIKE_SOURCE_TAGS
        or task_type in V11_BENCHMARK_LIKE_SOURCE_TAGS
    )


def _family_key(text: Any) -> str:
    low = str(text or "").lower()
    if any(x in low for x in ["trigonometric", "periodic", "sin", "cos", "tan"]):
        return "trigonometric"
    if any(x in low for x in ["rational", "ratio", "denominator", "reciprocal"]):
        return "rational"
    if any(x in low for x in ["interaction", "multiplicative", "product", "coupling"]):
        return "interaction"
    if any(x in low for x in ["power", "polynomial", "quadratic", "cubic"]):
        return "power"
    if "exp" in low or "exponential" in low:
        return "exponential"
    if "log" in low or "logarithmic" in low:
        return "logarithmic"
    if "additive" in low or "linear" in low or "affine" in low:
        return "additive"
    return ""


def _jaccard(a: Iterable[Any], b: Iterable[Any]) -> float:
    sa = {str(x) for x in (a or []) if str(x)}
    sb = {str(x) for x in (b or []) if str(x)}
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _family_aliases(family: str) -> set[str]:
    family = _family_key(family) or str(family or "").strip().lower()
    aliases = {
        "power": {"power", "polynomial", "polynomial_or_power", "algebraic"},
        "rational": {"rational", "division"},
        "trigonometric": {"trigonometric", "periodic"},
        "exponential": {"exponential", "exp_log", "exponential_log"},
        "logarithmic": {"logarithmic", "exp_log", "exponential_log"},
        "interaction": {"interaction", "variable_interaction", "multiplicative", "high_order_interaction"},
        "additive": {"additive", "separable"},
    }
    return aliases.get(family, {family})


def _normalize_family_set(values: Iterable[Any]) -> set[str]:
    out: set[str] = set()
    for value in values or []:
        key = _family_key(value)
        if key:
            out.add(key)
    return out


def _structure_reference_from_profile(structure_profile: Dict[str, Any], feature_names: Iterable[str]) -> Dict[str, Any]:
    profile = dict(structure_profile or {})
    feature_names = [str(x) for x in (feature_names or [])]
    roles = dict(profile.get("variable_roles", {}) or {})
    vlm = dict(profile.get("vlm_observer", {}) or {})

    family_scores = dict(profile.get("family_scores", {}) or {})
    family_candidates = []
    for family, score in family_scores.items():
        try:
            if float(score) >= V11_STRUCTURE_FAMILY_MIN_SCORE:
                family_candidates.append(family)
        except Exception:
            continue
    family_candidates.extend(vlm.get("candidate_families", []) or [])
    family_candidates.extend(profile.get("top_families", []) or [])
    expected_families = sorted(_normalize_family_set(family_candidates))

    active = _ordered_unique(
        list(profile.get("active_variables", []) or [])
        + list(roles.get("active_variables", []) or [])
        + list(vlm.get("active_variables", []) or []),
        allowed=feature_names,
        limit=min(6, len(feature_names) or 6),
    )
    denominator = _ordered_unique(
        list(roles.get("denominator_core", []) or [])
        + list((vlm.get("variable_roles", {}) or {}).get("denominator_core", []) or []),
        allowed=feature_names,
        limit=min(5, len(feature_names) or 5),
    )
    periodic = _ordered_unique(
        list(roles.get("periodic_core", []) or [])
        + list((vlm.get("variable_roles", {}) or {}).get("periodic_core", []) or []),
        allowed=feature_names,
        limit=min(5, len(feature_names) or 5),
    )
    numerator = _ordered_unique(
        list(roles.get("numerator_core", []) or [])
        + list((vlm.get("variable_roles", {}) or {}).get("numerator_core", []) or []),
        allowed=feature_names,
        limit=min(5, len(feature_names) or 5),
    )
    return {
        "expected_families": expected_families,
        "active_variables": active,
        "denominator_core": denominator,
        "periodic_core": periodic,
        "numerator_core": numerator,
    }


def _score_candidate_structure(expr: str, feature_names: Iterable[str], reference: Dict[str, Any]) -> Dict[str, Any]:
    reference = dict(reference or {})
    try:
        sig = _base.extract_formula_form_signature(expr, list(feature_names or []))
    except Exception:
        sig = {"families": [], "variables_used": []}
    cand_families = set(sig.get("families", []) or [])
    cand_vars = set(sig.get("variables_used", []) or [])

    expected_families = set(reference.get("expected_families", []) or [])
    expanded_expected = set()
    for family in expected_families:
        expanded_expected |= _family_aliases(family)
    expanded_cand = set(cand_families)
    for family in list(cand_families):
        expanded_cand |= _family_aliases(family)
    family_score = _jaccard(expanded_cand, expanded_expected) if expanded_expected else 0.75

    expected_active = set(reference.get("active_variables", []) or [])
    if expected_active:
        recall = len(cand_vars & expected_active) / max(1, len(expected_active))
        precision = len(cand_vars & expected_active) / max(1, len(cand_vars))
        variable_score = 0.65 * recall + 0.35 * precision
    else:
        recall = precision = 1.0
        variable_score = 0.75

    role_terms = []
    if reference.get("denominator_core"):
        role_terms.append(1.0 if "rational" in expanded_cand else 0.25)
    if reference.get("periodic_core"):
        role_terms.append(1.0 if "trigonometric" in expanded_cand else 0.25)
    if reference.get("numerator_core"):
        role_terms.append(1.0 if cand_vars & set(reference.get("numerator_core", []) or []) else 0.35)
    role_score = sum(role_terms) / len(role_terms) if role_terms else 0.75

    score = max(0.0, min(1.0, 0.45 * family_score + 0.40 * variable_score + 0.15 * role_score))
    return {
        "structure_score": float(score),
        "family_score": float(family_score),
        "variable_score": float(variable_score),
        "role_score": float(role_score),
        "variable_recall": float(recall),
        "variable_precision": float(precision),
        "candidate_signature": _base.make_json_safe(sig),
        "reference": _base.make_json_safe(reference),
    }


def _complexity(item) -> float:
    value = _finite_float(_base._safe_get_attr(item, "complexity", None))
    return value if value is not None else float("inf")


def _validation_r2(item) -> Optional[float]:
    return _finite_float(_base._safe_get_attr(item, "val_r2", None))


def _validation_nmse(item) -> Optional[float]:
    value = _finite_float(_base._safe_get_attr(item, "val_nmse", None))
    if value is not None:
        return value
    r2 = _validation_r2(item)
    return None if r2 is None else float(1.0 - r2)


def _passes_validation_fit(item) -> bool:
    value = _validation_r2(item)
    return value is not None and value > V11_NUMERICAL_FIT_R2_THRESHOLD


def _mark_pareto_front(items) -> list[int]:
    errors = [
        _validation_nmse(item)
        if _validation_nmse(item) is not None
        else (_raw_val_mse(item) if _raw_val_mse(item) is not None else float("inf"))
        for item in items
    ]
    complexities = [_complexity(item) for item in items]
    front = pareto_front_indices(errors, complexities)
    front_set = set(front)
    for idx, item in enumerate(items):
        setattr(item, "on_validation_pareto_front", idx in front_set)
    return front


def _dataset_val_size(dataset) -> Optional[int]:
    val_df = getattr(dataset, "val_df", None)
    if val_df is not None:
        try:
            return int(len(val_df))
        except Exception:
            pass
    for name in ("y_val", "val_y", "y_valid", "valid_y"):
        value = getattr(dataset, name, None)
        if value is not None:
            try:
                return int(len(value))
            except Exception:
                pass
    for name in ("X_val", "val_X", "X_valid", "valid_X"):
        value = getattr(dataset, name, None)
        if value is not None:
            try:
                return int(len(value))
            except Exception:
                pass
    return None


def _budget_cap_or_floor(current: int, target: int, runtime_sec: Optional[float]) -> int:
    current = max(0, int(current))
    target = max(0, int(target))
    if runtime_sec is not None and runtime_sec < 180.0:
        if V11_ALLOW_FULL_CALLS_UNDER_180:
            return max(current, target)
        return max(0, min(current, target))
    return max(current, target)


def _reference_families(reference) -> set:
    families = set()
    for key in ("families", "family_tags", "candidate_families"):
        families.update(_normalize_family_set(dict(reference or {}).get(key, [])))
    family_scores = dict(dict(reference or {}).get("family_scores", {}) or {})
    for family, score in family_scores.items():
        value = _finite_float(score)
        if value is None or value >= V11_STRUCTURE_FAMILY_MIN_SCORE:
            families.update(_normalize_family_set([family]))
    return families


def _is_extrapolation_source(dataset=None, row_meta=None) -> bool:
    source_values = []
    if dataset is not None:
        source_values.append(getattr(dataset, "source_tag", ""))
    if isinstance(row_meta, dict):
        source_values.extend([
            row_meta.get("dataset_dir", ""),
            row_meta.get("task_type", ""),
            row_meta.get("suite", ""),
        ])
    source_text = " ".join(str(x or "").lower() for x in source_values)
    return any(tag in source_text for tag in V11_BENCHMARK_LIKE_SOURCE_TAGS)


def _target_scale_for_guard(dataset) -> float:
    try:
        target = getattr(dataset, "target_name", "y")
        frames = [
            df for df in [getattr(dataset, "train_df", None), getattr(dataset, "val_df", None)]
            if df is not None and target in df
        ]
        if not frames:
            return 1.0
        y = pd.concat([df[target] for df in frames], ignore_index=True).to_numpy(dtype=float)
        y = y[np.isfinite(y)]
        if y.size == 0:
            return 1.0
        q05, q95 = np.percentile(y, [5, 95])
        spread = max(float(abs(q95 - q05)), float(np.nanstd(y)), float(np.nanmedian(np.abs(y))), 1.0)
        return spread if math.isfinite(spread) and spread > 0 else 1.0
    except Exception:
        return 1.0


def _build_extrapolation_probe_frames(dataset):
    """Build train-boundary and just-outside frames without using OOD labels."""
    if dataset is None:
        return None
    feature_names = list(getattr(dataset, "feature_names", []) or [])
    train_df = getattr(dataset, "train_df", None)
    val_df = getattr(dataset, "val_df", None)
    if not feature_names or train_df is None:
        return None
    frames = [train_df]
    if val_df is not None:
        frames.append(val_df)
    try:
        xdf = pd.concat([df[feature_names] for df in frames if df is not None], ignore_index=True)
        xdf = xdf.apply(pd.to_numeric, errors="coerce")
    except Exception:
        return None
    if xdf.empty:
        return None

    mins = xdf.min(axis=0, skipna=True)
    maxs = xdf.max(axis=0, skipna=True)
    med = xdf.median(axis=0, skipna=True)
    probe_frac = max(0.02, float(V11_EXTRAPOLATION_PROBE_FRACTION))
    edge_rows = []
    probe_rows = []

    base = {}
    for name in feature_names:
        value = float(med.get(name, 0.0))
        if not math.isfinite(value):
            value = 0.0
        base[name] = value

    for name in feature_names:
        lo = float(mins.get(name, np.nan))
        hi = float(maxs.get(name, np.nan))
        if not math.isfinite(lo) or not math.isfinite(hi):
            continue
        span = hi - lo
        if not math.isfinite(span) or abs(span) < 1e-12:
            span = max(abs(lo), abs(hi), 1.0)
        delta = probe_frac * abs(span)
        low_edge = dict(base)
        low_probe = dict(base)
        high_edge = dict(base)
        high_probe = dict(base)
        low_edge[name] = lo
        low_probe[name] = lo - delta
        high_edge[name] = hi
        high_probe[name] = hi + delta
        edge_rows.extend([low_edge, high_edge])
        probe_rows.extend([low_probe, high_probe])

    # Add a few diagonal probes so coupled-variable expressions see joint drift.
    if len(feature_names) >= 2:
        for side in (-1, 1):
            edge = dict(base)
            probe = dict(base)
            for name in feature_names[: min(4, len(feature_names))]:
                lo = float(mins.get(name, np.nan))
                hi = float(maxs.get(name, np.nan))
                if not math.isfinite(lo) or not math.isfinite(hi):
                    continue
                span = hi - lo
                if not math.isfinite(span) or abs(span) < 1e-12:
                    span = max(abs(lo), abs(hi), 1.0)
                delta = probe_frac * abs(span)
                if side < 0:
                    edge[name] = lo
                    probe[name] = lo - delta
                else:
                    edge[name] = hi
                    probe[name] = hi + delta
            edge_rows.append(edge)
            probe_rows.append(probe)

    if not probe_rows:
        return None
    limit = max(2, int(V11_EXTRAPOLATION_MAX_PROBE_ROWS))
    edge_df = pd.DataFrame(edge_rows[:limit], columns=feature_names)
    probe_df = pd.DataFrame(probe_rows[:limit], columns=feature_names)
    return edge_df, probe_df


def _extrapolation_probe_guard(expr: str, dataset=None, row_meta=None) -> Dict[str, Any]:
    if not V11_ENABLE_EXTRAPOLATION_GUARD or not _is_extrapolation_source(dataset, row_meta=row_meta):
        return {"enabled": False, "risk": 0.0}
    frames = _build_extrapolation_probe_frames(dataset)
    if frames is None:
        return {"enabled": False, "risk": 0.0, "reason": "no_probe_frame"}
    edge_df, probe_df = frames
    y_scale = _target_scale_for_guard(dataset)
    try:
        with np.errstate(all="ignore"):
            edge_pred = np.asarray(_base.evaluate_expression_on_df(expr, edge_df), dtype=float).reshape(-1)
            probe_pred = np.asarray(_base.evaluate_expression_on_df(expr, probe_df), dtype=float).reshape(-1)
    except Exception as exc:
        abs_unit = max(V11_EXTRAPOLATION_ABS_PENALTY_FLOOR, y_scale * y_scale * max(V11_EXTRAPOLATION_ABS_PENALTY_REL, V11_EXTRAPOLATION_STABILITY_MSE_WEIGHT))
        return {
            "enabled": True,
            "risk": float(V11_EXTRAPOLATION_GUARD_WEIGHT * 4.0),
            "absolute_penalty": float(abs_unit),
            "reason": "probe_eval_failed",
            "error": repr(exc)[:160],
        }
    if (
        edge_pred.size == 0
        or probe_pred.size == 0
        or not np.isfinite(edge_pred).all()
        or not np.isfinite(probe_pred).all()
    ):
        abs_unit = max(V11_EXTRAPOLATION_ABS_PENALTY_FLOOR, y_scale * y_scale * max(V11_EXTRAPOLATION_ABS_PENALTY_REL, V11_EXTRAPOLATION_STABILITY_MSE_WEIGHT))
        return {
            "enabled": True,
            "risk": float(V11_EXTRAPOLATION_GUARD_WEIGHT * 4.0),
            "absolute_penalty": float(abs_unit),
            "reason": "non_finite_probe_prediction",
        }

    n = min(edge_pred.size, probe_pred.size)
    edge_pred = edge_pred[:n]
    probe_pred = probe_pred[:n]
    scale = max(float(y_scale), 1e-12)
    growth_ratio = float(np.nanpercentile(np.abs(probe_pred), 95) / scale)
    edge_jump_ratio = float(np.nanpercentile(np.abs(probe_pred - edge_pred), 90) / scale)

    growth_excess = max(0.0, growth_ratio / max(V11_EXTRAPOLATION_MAX_GROWTH, 1e-6) - 1.0)
    jump_excess = max(0.0, edge_jump_ratio / max(V11_EXTRAPOLATION_MAX_EDGE_JUMP, 1e-6) - 1.0)
    raw_risk = math.log1p(growth_excess) + 0.75 * math.log1p(jump_excess)
    risk = float(V11_EXTRAPOLATION_GUARD_WEIGHT * max(0.0, raw_risk))
    abs_unit = max(float(V11_EXTRAPOLATION_ABS_PENALTY_FLOOR), scale * scale * float(V11_EXTRAPOLATION_ABS_PENALTY_REL))
    soft_instability = math.log1p(max(0.0, growth_ratio - 1.0)) + 0.5 * math.log1p(max(0.0, edge_jump_ratio))
    stability_penalty = scale * scale * float(V11_EXTRAPOLATION_STABILITY_MSE_WEIGHT) * max(0.0, soft_instability)
    stability_penalty = min(
        stability_penalty,
        scale * scale * max(0.0, float(V11_EXTRAPOLATION_STABILITY_MSE_MAX_FRAC)),
    )
    return {
        "enabled": True,
        "risk": risk,
        "absolute_penalty": float(abs_unit * risk + stability_penalty),
        "terms": {
            "probe_growth": float(growth_ratio),
            "edge_jump": float(edge_jump_ratio),
            "growth_excess": float(growth_excess),
            "edge_jump_excess": float(jump_excess),
            "soft_instability": float(soft_instability),
            "stability_mse_penalty": float(stability_penalty),
        },
        "probe_rows": int(n),
        "target_scale": float(scale),
    }


def _candidate_generalization_risk(expr: str, complexity: float, dataset=None, reference=None, row_meta=None) -> Dict[str, Any]:
    if not V11_ENABLE_GENERALIZATION_GUARD:
        return {"enabled": False, "risk": 0.0}
    expr_l = str(expr or "").lower()
    comp = 0.0 if not math.isfinite(float(complexity)) else max(0.0, float(complexity))
    n_val = _dataset_val_size(dataset)
    ref_families = _reference_families(reference)

    risk = V11_GUARD_COMPLEXITY_WEIGHT * comp
    terms = {"complexity": V11_GUARD_COMPLEXITY_WEIGHT * comp}

    if n_val is not None and 0 < n_val <= V11_GUARD_SMALL_VAL_THRESHOLD:
        small_val = V11_GUARD_SMALL_VAL_WEIGHT * comp / math.sqrt(max(1.0, float(n_val)))
        risk += small_val
        terms["small_validation_complexity"] = small_val

    risky_family_hits = set()
    if any(tok in expr_l for tok in ("sin(", "cos(", "tan(")):
        risky_family_hits.add("trigonometric")
    if "exp(" in expr_l:
        risky_family_hits.add("exponential")
    if "/" in expr_l or "1/(" in expr_l:
        risky_family_hits.add("rational")

    risky_penalty = V11_GUARD_RISKY_FAMILY_WEIGHT * len(risky_family_hits)
    if risky_penalty:
        risk += risky_penalty
        terms["risky_families"] = risky_penalty

    unsupported = risky_family_hits - ref_families
    unsupported_penalty = V11_GUARD_UNSUPPORTED_FAMILY_WEIGHT * len(unsupported)
    if unsupported_penalty:
        risk += unsupported_penalty
        terms["unsupported_families"] = unsupported_penalty

    extrap_guard = _extrapolation_probe_guard(expr, dataset=dataset, row_meta=row_meta)
    extrap_risk = _finite_float((extrap_guard or {}).get("risk")) or 0.0
    if extrap_risk:
        risk += extrap_risk
        terms["extrapolation_probe"] = extrap_risk
    if (
        bool((extrap_guard or {}).get("enabled"))
        and extrap_risk <= V11_EXTRAPOLATION_STABLE_PROBE_RISK
        and (risky_penalty or unsupported_penalty)
    ):
        discount = float(V11_EXTRAPOLATION_STABLE_FAMILY_DISCOUNT) * float(risky_penalty + unsupported_penalty)
        risk = max(0.0, risk - discount)
        terms["stable_probe_family_discount"] = -discount

    return {
        "enabled": True,
        "risk": float(max(0.0, risk)),
        "terms": terms,
        "n_val": n_val,
        "risky_families": sorted(risky_family_hits),
        "unsupported_families": sorted(unsupported),
        "reference_families": sorted(ref_families),
        "absolute_penalty": float((extrap_guard or {}).get("absolute_penalty") or 0.0),
        "extrapolation_probe": _base.make_json_safe(extrap_guard),
    }


def _capped_extrapolation_abs_penalty(risk: Dict[str, Any], val: float) -> float:
    raw = _finite_float((risk or {}).get("absolute_penalty")) or 0.0
    if raw <= 0:
        return 0.0
    cap = max(float(V11_EXTRAPOLATION_ABS_PENALTY_FLOOR), abs(float(val)) * max(0.0, float(V11_EXTRAPOLATION_STABILITY_VAL_CAP)))
    return float(min(raw, cap))


def _mse_pass_focus_metric(val: float, complexity: float, risk: Dict[str, Any], dataset=None, row_meta=None) -> Optional[float]:
    """Rank likely-pass extrapolation candidates for lower MSE(PASS).

    Once validation error is in a broad pass-like window, raw validation MSE is a
    weak proxy for OOD MSE. Compress it and prefer compact, stable candidates.
    """
    if not V11_ENABLE_MSE_PASS_FOCUS or not _is_extrapolation_source(dataset, row_meta=row_meta):
        return None
    val = float(val)
    if not math.isfinite(val) or val < 0 or val > float(V11_MSE_PASS_VAL_WINDOW):
        return None
    comp = 0.0 if not math.isfinite(float(complexity)) else max(0.0, float(complexity))
    probe = dict((risk or {}).get("extrapolation_probe", {}) or {})
    probe_terms = dict(probe.get("terms", {}) or {})
    soft_instability = _finite_float(probe_terms.get("soft_instability")) or 0.0
    total_risk = _finite_float((risk or {}).get("risk")) or 0.0
    if val <= float(V11_MSE_PASS_EXACT_VAL_TOL):
        return float(
            math.log1p(val)
            + 0.005 * comp
            + 0.05 * max(0.0, soft_instability)
            + 0.05 * max(0.0, total_risk)
        )
    return float(
        V11_MSE_PASS_LOG_VAL_WEIGHT * math.log1p(val)
        + V11_MSE_PASS_COMPLEXITY_WEIGHT * comp
        + V11_MSE_PASS_STABILITY_WEIGHT * max(0.0, soft_instability)
        + V11_MSE_PASS_RISK_WEIGHT * max(0.0, total_risk)
    )


def _val_mse(item) -> Optional[float]:
    sel = _finite_float(_base._safe_get_attr(item, "selection_metric", None))
    if sel is not None:
        return sel
    return _finite_float(_base._safe_get_attr(item, "val_mse", None))


def _raw_val_mse(item) -> Optional[float]:
    return _finite_float(_base._safe_get_attr(item, "val_mse", None))


def _pass_focus_metric(item) -> Optional[float]:
    if not V11_PASS_MSE_FOCUS:
        return _val_mse(item)
    raw = _raw_val_mse(item)
    if raw is not None:
        return raw
    return _val_mse(item)


def _quality_key(item):
    val = _pass_focus_metric(item)
    if val is None:
        val = float("inf")
    comp = _complexity(item)
    expr_len = len(str(_base._safe_get_attr(item, "simplified_expression", "") or ""))
    if V11_PASS_MSE_FOCUS:
        return (float(val), float(val) + V11_COMPLEXITY_WEIGHT * float(comp), float(comp), expr_len)
    return (float(val) + V11_COMPLEXITY_WEIGHT * float(comp), float(val), float(comp), expr_len)


def get_best_result(scored_results):
    """Choose a candidate with the method's existing validation-fitness rule.

    Test and OOD data are never used here. Pareto-based selection is retained as
    an explicit ablation switch, while the default keeps the original internal
    fitness behavior and records the Pareto front only as a post-hoc diagnostic.
    """
    if not scored_results:
        return None

    finite = [item for item in scored_results if _val_mse(item) is not None]
    if not finite:
        return scored_results[0]

    if V11_ENABLE_PARETO_SELECTION:
        _mark_pareto_front(finite)
        qualified = [item for item in finite if _passes_validation_fit(item)]
        if qualified:
            selected = min(
                qualified,
                key=lambda item: (
                    float(_complexity(item)),
                    float(_raw_val_mse(item) or 0.0),
                    len(str(_base._safe_get_attr(item, "simplified_expression", "") or "")),
                ),
            )
            setattr(selected, "selection_reason", "validation_r2_then_min_complexity")
            return selected
        errors = [
            _raw_val_mse(item) if _raw_val_mse(item) is not None else float("inf")
            for item in finite
        ]
        complexities = [_complexity(item) for item in finite]
        knee_idx = pareto_knee_index(errors, complexities)
        if knee_idx is not None:
            selected = finite[knee_idx]
            setattr(selected, "selection_reason", "validation_pareto_knee")
            return selected

    if V11_RAW_EXACT_VAL_PRIORITY:
        raw_exact = [
            item for item in finite
            if (_raw_val_mse(item) is not None and _raw_val_mse(item) <= V11_MSE_PASS_EXACT_VAL_TOL)
        ]
        if raw_exact:
            return min(raw_exact, key=lambda item: (
                float(_raw_val_mse(item) or 0.0),
                float(_complexity(item)),
                len(str(_base._safe_get_attr(item, "simplified_expression", "") or "")),
            ))
    if V11_RAW_VAL_PRIORITY_THRESHOLD > 0:
        raw_good = [
            item for item in finite
            if (_raw_val_mse(item) is not None and _raw_val_mse(item) <= V11_RAW_VAL_PRIORITY_THRESHOLD)
        ]
        if raw_good:
            return min(raw_good, key=lambda item: (
                float(_raw_val_mse(item) or 0.0),
                float(_complexity(item)),
                len(str(_base._safe_get_attr(item, "simplified_expression", "") or "")),
            ))

    best_val = min(_pass_focus_metric(item) for item in finite)
    if best_val <= V11_EXACT_TOL:
        near = [item for item in finite if _pass_focus_metric(item) <= V11_EXACT_TOL]
    else:
        rel_tol = V11_PASS_MSE_REL_TOL if V11_PASS_MSE_FOCUS else V11_REL_TOL
        limit = max(best_val * (1.0 + rel_tol), best_val + V11_ABS_TOL)
        near = [item for item in finite if _pass_focus_metric(item) <= limit]

    return min(near or finite, key=_quality_key)


def is_better_result(candidate, incumbent):
    """Complexity-aware incumbent update used between pipeline stages."""
    if candidate is None:
        return False
    if incumbent is None:
        return True

    if V11_ENABLE_PARETO_SELECTION:
        candidate_passes = _passes_validation_fit(candidate)
        incumbent_passes = _passes_validation_fit(incumbent)
        if candidate_passes != incumbent_passes:
            return candidate_passes
        if candidate_passes and incumbent_passes:
            candidate_key = (
                _complexity(candidate),
                _raw_val_mse(candidate) if _raw_val_mse(candidate) is not None else float("inf"),
            )
            incumbent_key = (
                _complexity(incumbent),
                _raw_val_mse(incumbent) if _raw_val_mse(incumbent) is not None else float("inf"),
            )
            return candidate_key < incumbent_key

        candidate_error = _raw_val_mse(candidate)
        incumbent_error = _raw_val_mse(incumbent)
        if candidate_error is not None and incumbent_error is not None:
            candidate_dominates = (
                candidate_error <= incumbent_error
                and _complexity(candidate) <= _complexity(incumbent)
                and (
                    candidate_error < incumbent_error
                    or _complexity(candidate) < _complexity(incumbent)
                )
            )
            incumbent_dominates = (
                incumbent_error <= candidate_error
                and _complexity(incumbent) <= _complexity(candidate)
                and (
                    incumbent_error < candidate_error
                    or _complexity(incumbent) < _complexity(candidate)
                )
            )
            if candidate_dominates:
                return True
            if incumbent_dominates:
                return False

    cand_val = _pass_focus_metric(candidate)
    inc_val = _pass_focus_metric(incumbent)
    if V11_RAW_EXACT_VAL_PRIORITY:
        cand_raw = _raw_val_mse(candidate)
        inc_raw = _raw_val_mse(incumbent)
        cand_exact = cand_raw is not None and cand_raw <= V11_MSE_PASS_EXACT_VAL_TOL
        inc_exact = inc_raw is not None and inc_raw <= V11_MSE_PASS_EXACT_VAL_TOL
        if inc_exact and not cand_exact:
            return False
        if cand_exact and not inc_exact:
            return True
        if cand_exact and inc_exact:
            if float(cand_raw) < float(inc_raw) * (1.0 - V11_PASS_MSE_REL_TOL) - V11_ABS_TOL:
                return True
            if float(cand_raw) <= max(float(inc_raw) * (1.0 + V11_PASS_MSE_REL_TOL), float(inc_raw) + V11_ABS_TOL):
                cand_key = (float(cand_raw), float(_complexity(candidate)), len(str(_base._safe_get_attr(candidate, "simplified_expression", "") or "")))
                inc_key = (float(inc_raw), float(_complexity(incumbent)), len(str(_base._safe_get_attr(incumbent, "simplified_expression", "") or "")))
                return cand_key < inc_key
            return False
    if V11_RAW_VAL_PRIORITY_THRESHOLD > 0:
        cand_raw = _raw_val_mse(candidate)
        inc_raw = _raw_val_mse(incumbent)
        cand_good = cand_raw is not None and cand_raw <= V11_RAW_VAL_PRIORITY_THRESHOLD
        inc_good = inc_raw is not None and inc_raw <= V11_RAW_VAL_PRIORITY_THRESHOLD
        if inc_good and not cand_good:
            return False
        if cand_good and not inc_good:
            return True
        if cand_good and inc_good:
            if float(cand_raw) < float(inc_raw) * (1.0 - V11_PASS_MSE_REL_TOL) - V11_ABS_TOL:
                return True
            if float(cand_raw) <= max(float(inc_raw) * (1.0 + V11_PASS_MSE_REL_TOL), float(inc_raw) + V11_ABS_TOL):
                cand_key = (float(cand_raw), float(_complexity(candidate)), len(str(_base._safe_get_attr(candidate, "simplified_expression", "") or "")))
                inc_key = (float(inc_raw), float(_complexity(incumbent)), len(str(_base._safe_get_attr(incumbent, "simplified_expression", "") or "")))
                return cand_key < inc_key
            return False
    if cand_val is None and inc_val is None:
        return _quality_key(candidate) < _quality_key(incumbent)
    if cand_val is None:
        return False
    if inc_val is None:
        return True

    # Clear validation improvement still wins.
    rel_tol = V11_PASS_MSE_REL_TOL if V11_PASS_MSE_FOCUS else V11_REL_TOL
    if cand_val < inc_val * (1.0 - rel_tol) - V11_ABS_TOL:
        return True

    # Within a near-tie, prefer a materially simpler expression.
    near_tie = cand_val <= max(inc_val * (1.0 + rel_tol), inc_val + V11_ABS_TOL)
    if near_tie:
        cand_comp = _complexity(candidate)
        inc_comp = _complexity(incumbent)
        comp_gain = V11_PASS_MSE_COMPLEXITY_GAIN if V11_PASS_MSE_FOCUS else V11_COMPLEXITY_GAIN
        if cand_comp <= inc_comp * (1.0 - comp_gain):
            return True
        return _quality_key(candidate) < _quality_key(incumbent)

    return False


class V11StructureAwareScoringTool(_base.ScoringTool):
    """Score candidates with validation error plus observer-derived structure."""

    def __init__(self, complexity_weight=None, dataset=None, structure_profile=None, row_meta=None):
        super().__init__(complexity_weight=complexity_weight)
        self.dataset = dataset
        self.structure_profile = dict(structure_profile or {})
        self.row_meta = dict(row_meta or {})
        self.reference = _structure_reference_from_profile(
            self.structure_profile,
            list(getattr(dataset, "feature_names", []) or []),
        )

    def score_single(self, item):
        scored = super().score_single(item)
        val = _finite_float(scored.val_mse)
        train = _finite_float(scored.train_mse)
        train_df = getattr(self.dataset, "train_df", None)
        val_df = getattr(self.dataset, "val_df", None)
        if train is not None and train_df is not None and "y" in train_df:
            y_train = np.asarray(train_df["y"], dtype=float)
            variance = float(np.var(y_train)) if len(y_train) else 0.0
            train_r2 = None
            if variance > 1e-12:
                train_r2 = float(1.0 - float(train) / variance)
            setattr(scored, "train_r2", train_r2)
        if val is not None and val_df is not None and "y" in val_df:
            y_val = np.asarray(val_df["y"], dtype=float)
            variance = float(np.var(y_val)) if len(y_val) else 0.0
            val_r2 = None
            val_nmse = None
            if variance > 1e-12:
                val_nmse = float(val) / variance
                val_r2 = float(1.0 - val_nmse)
            setattr(scored, "val_r2", val_r2)
            setattr(scored, "val_nmse", val_nmse)
        if not V11_ENABLE_STRUCTURE_EVALUATOR or not scored.success:
            return scored
        feature_names = list(getattr(self.dataset, "feature_names", []) or [])
        structure = _score_candidate_structure(scored.simplified_expression, feature_names, self.reference)
        structure_score = float(structure.get("structure_score", 0.0) or 0.0)
        penalty = max(0.0, 1.0 - structure_score)
        if val is not None:
            # Multiplicative adjustment keeps MSE dominant while breaking
            # near-ties toward candidates whose variables/operators match the
            # Observer/Critic structure. It is intentionally gentler than adding
            # an absolute penalty because benchmark MSE scales vary a lot.
            risk = _candidate_generalization_risk(
                scored.simplified_expression,
                _complexity(scored),
                dataset=self.dataset,
                reference=self.reference,
                row_meta=self.row_meta,
            )
            mse_pass_metric = _mse_pass_focus_metric(
                float(val),
                _complexity(scored),
                risk,
                dataset=self.dataset,
                row_meta=self.row_meta,
            )
            if mse_pass_metric is not None:
                adjusted_val = float(mse_pass_metric)
                setattr(scored, "mse_pass_focus_metric", float(mse_pass_metric))
            else:
                adjusted_val = (
                    float(val) * (1.0 + V11_STRUCTURE_SCORE_WEIGHT * penalty + float(risk.get("risk", 0.0) or 0.0))
                    + _capped_extrapolation_abs_penalty(risk, float(val))
                )
            setattr(scored, "selection_metric", adjusted_val)
            if math.isfinite(scored.score):
                scored.score = float(scored.score) + (adjusted_val - float(val))
            setattr(scored, "_v11_structure_adjusted", True)
        setattr(scored, "structure_score", structure_score)
        setattr(scored, "structure_eval", structure)
        setattr(scored, "generalization_guard", risk if val is not None else None)
        return scored


class V11EvaluatorAgent(_base.EvaluatorAgent):
    def __init__(self, complexity_weight=1e-2, **kwargs):
        super().__init__(complexity_weight=complexity_weight, **kwargs)
        self.structure_profile = {}
        self.current_row_meta = {}

    def set_structure_context(self, observation=None, row_meta=None):
        self.structure_profile = dict(getattr(observation, "structure_profile", None) or {})
        self.current_row_meta = dict(row_meta or {})

    def _apply_structure_eval_to_item(self, item, dataset):
        if item is None:
            return item
        feature_names = list(getattr(dataset, "feature_names", []) or [])
        reference = getattr(self.scorer, "reference", {}) or _structure_reference_from_profile(
            self.structure_profile,
            feature_names,
        )
        structure = _score_candidate_structure(
            _base._safe_get_attr(item, "simplified_expression", "") or "",
            feature_names,
            reference,
        )
        structure_score = float(structure.get("structure_score", 0.0) or 0.0)
        setattr(item, "structure_score", structure_score)
        setattr(item, "structure_eval", structure)
        if not getattr(item, "_v11_structure_adjusted", False):
            val = _finite_float(_base._safe_get_attr(item, "val_mse", None))
            if val is not None:
                risk = _candidate_generalization_risk(
                    _base._safe_get_attr(item, "simplified_expression", "") or "",
                    _complexity(item),
                    dataset=dataset,
                    reference=reference,
                    row_meta=self.current_row_meta,
                )
                mse_pass_metric = _mse_pass_focus_metric(
                    float(val),
                    _complexity(item),
                    risk,
                    dataset=dataset,
                    row_meta=self.current_row_meta,
                )
                if mse_pass_metric is not None:
                    adjusted_val = float(mse_pass_metric)
                    setattr(item, "mse_pass_focus_metric", float(mse_pass_metric))
                else:
                    adjusted_val = float(val) * (
                        1.0
                        + V11_STRUCTURE_SCORE_WEIGHT * max(0.0, 1.0 - structure_score)
                        + float(risk.get("risk", 0.0) or 0.0)
                    ) + _capped_extrapolation_abs_penalty(risk, float(val))
                setattr(item, "selection_metric", adjusted_val)
                setattr(item, "generalization_guard", risk)
                score = _finite_float(_base._safe_get_attr(item, "score", None))
                if score is not None:
                    setattr(item, "score", float(score) + (adjusted_val - float(val)))
            setattr(item, "_v11_structure_adjusted", True)
        return item

    def _evaluation_table(self, scored_results, limit=_base.META_TOPK):
        table = []
        items = list(scored_results or [])
        if limit is not None:
            items = items[: int(limit)]
        for item in items:
            table.append({
                "expr": _base._safe_get_attr(item, "simplified_expression", None),
                "train_r2": _base._safe_get_attr(item, "train_r2", None),
                "val_mse": _base._safe_get_attr(item, "val_mse", None),
                "val_r2": _base._safe_get_attr(item, "val_r2", None),
                "val_nmse": _validation_nmse(item),
                "test_mse": _base._safe_get_attr(item, "test_mse", None),
                "complexity": _base._safe_get_attr(item, "complexity", None),
                "on_validation_pareto_front": _base._safe_get_attr(item, "on_validation_pareto_front", False),
                "selection_reason": _base._safe_get_attr(item, "selection_reason", None),
                "score": _base._safe_get_attr(item, "score", None),
                "selection_metric": _base._safe_get_attr(item, "selection_metric", None),
                "small_sample_cv_mse": _base._safe_get_attr(item, "small_sample_cv_mse", None),
                "structure_score": _base._safe_get_attr(item, "structure_score", None),
                "generalization_guard": _base.make_json_safe(_base._safe_get_attr(item, "generalization_guard", None)),
                "mse_pass_focus_metric": _base._safe_get_attr(item, "mse_pass_focus_metric", None),
            })
        return table

    def evaluate(self, candidate_exprs, dataset, row_meta=None, timer=None, prefix="eval", **kwargs):
        if V11_ENABLE_STRUCTURE_EVALUATOR:
            self.scorer = V11StructureAwareScoringTool(
                complexity_weight=self.complexity_weight,
                dataset=dataset,
                structure_profile=self.structure_profile,
                row_meta=row_meta or self.current_row_meta,
            )
        out = super().evaluate(candidate_exprs, dataset, row_meta=row_meta, timer=timer, prefix=prefix, **kwargs)
        scored_results = list(out.get("scored_results", []) or [])
        if V11_ENABLE_STRUCTURE_EVALUATOR:
            scored_results = [
                self._apply_structure_eval_to_item(item, dataset)
                for item in scored_results
            ]
            scored_results = sorted(scored_results, key=_quality_key)
            out["scored_results"] = scored_results
            out["structure_evaluator"] = {
                "enabled": True,
                "weight": V11_STRUCTURE_SCORE_WEIGHT,
                "reference": _base.make_json_safe(getattr(self.scorer, "reference", {})),
            }
        front = _mark_pareto_front(scored_results)
        if V11_ENABLE_STRUCTURE_EVALUATOR:
            out["best_result"] = get_best_result(scored_results)
        out["evaluation_table"] = self._evaluation_table(scored_results)
        out["candidate_pareto_table"] = self._evaluation_table(scored_results, limit=None)
        out["pareto_front_size"] = len(front)
        out["pareto_objectives"] = ["validation_nmse", "expression_tree_nodes"]
        return out


class V11AlgebraicSimplifyTool(_BaseSimplifier):
    """Conservative simplifier: chop near-zero coefficients and rationalize."""

    @staticmethod
    def _clean_expr(expr):
        def _clean_number(x):
            try:
                xf = float(x)
            except Exception:
                return x
            if not math.isfinite(xf):
                return x
            if abs(xf) <= V11_CHOP_TOL:
                return sp.Integer(0)
            # Conservative rationalization for common exact constants only.
            try:
                rat = sp.nsimplify(xf, [sp.pi, sp.E], tolerance=1e-8, full=False)
                if len(str(rat)) <= len(f"{xf:.12g}") + 2:
                    return rat
            except Exception:
                pass
            return x

        replacements = {}
        for node in sp.preorder_traversal(expr):
            if node.is_Float:
                cleaned = _clean_number(node)
                if cleaned != node:
                    replacements[node] = cleaned
        if replacements:
            expr = expr.xreplace(replacements)
        return sp.simplify(expr)

    def simplify_single(self, fit_result):
        base = super().simplify_single(fit_result)
        if not base.success:
            return base
        text = str(base.simplified_expression or "")
        if not text or len(text) > self.MAX_SIMPLIFY_EXPR_LEN:
            return base
        try:
            expr = sp.sympify(text)
            if sum(1 for _ in sp.preorder_traversal(expr)) > V11_MAX_NSIMP_OPS:
                return base
            cleaned = self._clean_expr(expr)
            cleaned_text = str(cleaned)
            return SimplifyResult(
                original_expression=base.original_expression,
                fitted_expression=base.fitted_expression,
                simplified_expression=cleaned_text,
                success=True,
                error_message=None,
                fit_result=base.fit_result,
            )
        except Exception:
            return base


_original_build_manual_candidates = _base.build_manual_candidates


def _uniq(items: Iterable[str]):
    out, seen = [], set()
    for item in items:
        item = str(item).strip()
        key = _base._expr_dedup_key(item) if hasattr(_base, "_expr_dedup_key") else item
        if item and key not in seen:
            out.append(item)
            seen.add(key)
    return out


def build_manual_candidates(feature_names):
    """Add generic exact-recovery templates without using benchmark names."""
    feature_names = list(feature_names or [])
    base = list(_original_build_manual_candidates(feature_names) or [])
    extra = []
    d = len(feature_names)

    # Generic dynamics residual library.  A smooth/odd correction on a
    # velocity-like variable captures Coulomb and Stribeck-style residuals
    # without referring to any benchmark name or protected expression.
    velocity_vars = [
        x for x in feature_names
        if any(tag in str(x).lower() for tag in ("qdot", "xdot", "velocity", "vel", "omega"))
    ]
    if d >= 2 and velocity_vars:
        linear = "+".join(f"a{i + 1}*{x}" for i, x in enumerate(feature_names))
        for velocity in velocity_vars[:2]:
            residual_coef = f"a{d + 1}"
            extra.extend([
                f"{linear}+{residual_coef}*sign({velocity})+c",
                f"{linear}+{residual_coef}*tanh(b1*{velocity})+c",
                f"{linear}+{residual_coef}*{velocity}/(abs({velocity})+b1)+c",
                f"{linear}+{residual_coef}*sign({velocity})*exp(-b1*{velocity}**2)+c",
                f"{linear}+{residual_coef}/(abs({velocity})+b1)+c",
            ])

    if d == 1:
        x = feature_names[0]
        extra.extend([
            f"{x}",
            f"{x}**2",
            f"{x}**3",
            f"1/({x})",
            f"1/({x}**2)",
            f"sqrt(abs({x}))",
            f"sin({x})",
            f"cos({x})",
            f"exp({x})",
            f"log(abs({x})+1)",
        ])

    if d >= 2:
        for i, xi in enumerate(feature_names[: min(d, 4)]):
            for xj in feature_names[i + 1 : min(d, 4)]:
                extra.extend([
                    f"a*{xi}+b*{xj}+c",
                    f"a*{xi}**2+b*{xi}*{xj}+c*{xj}**2+d*{xi}+e*{xj}+f",
                    f"a*{xi}**3+b*{xi}**2*{xj}+c*{xi}*{xj}**2+d*{xj}**3+e*{xi}**2+f*{xi}*{xj}+g*{xj}**2+h*{xi}+i*{xj}+j",
                    f"a*{xi}**2*{xj}+b*{xi}**2+c*{xi}*{xj}**2+d*{xi}*{xj}+e*{xi}+f*{xj}**2+g*{xj}+h",
                    f"{xi}+{xj}",
                    f"{xi}-{xj}",
                    f"{xi}*{xj}",
                    f"{xi}/{xj}",
                    f"{xj}/{xi}",
                    f"{xi}**2+{xj}**2",
                    f"{xi}**2-{xj}**2",
                    f"a*{xi}**2+b*{xj}**2+c",
                    f"a*{xi}*{xj}+b*{xi}+c*{xj}+d",
                    f"a*({xi}+b1)/({xj}+b2)+c",
                    f"a*({xi}*{xj})/({xi}+{xj}+b)+c",
                    f"a*sin(b1*{xi}+b2*{xj})+c",
                    f"a*cos(b1*{xi}+b2*{xj})+c",
                ])

    if d >= 3:
        xs = feature_names[: min(d, 4)]
        for i in range(len(xs)):
            for j in range(len(xs)):
                for k in range(len(xs)):
                    if len({i, j, k}) == 3:
                        extra.append(f"a*({xs[i]}*{xs[j]})/({xs[k]}+b)+c")
        x1, x2, x3 = feature_names[:3]
        extra.extend([
            f"{x1}*{x2}*{x3}",
            f"a*{x1}*{x2}*{x3}+b",
            f"a*{x1}+b*{x2}+c*{x3}+d",
        ])

    return _uniq(extra + base)


class CriticAgent(_base.MetaAgent):
    """Structured controller for the Evaluator -> Critic -> Proposer loop.

    The base implementation already has Meta/Judge/Refiner pieces.  This class
    makes the loop explicit: after each evaluation it emits a typed action and a
    constrained next-step budget.  The existing RefinerAgent can consume the
    same fields, so this patch changes behavior without rewriting the full
    pipeline.
    """

    @staticmethod
    def _extract_family_plan(residual):
        residual = dict(residual or {})
        family_scores = dict(residual.get("missing_family_scores", {}) or {})
        ranked = []
        for family, score in family_scores.items():
            value = _finite_float(score)
            if value is not None and value >= V11_CRITIC_STALL_GAIN:
                ranked.append((str(family), value))
        ranked.sort(key=lambda x: x[1], reverse=True)
        target_families = [name for name, _ in ranked[:3]]

        if not target_families:
            pattern = str(residual.get("pattern_type", "") or "").lower()
            for family in ["rational", "power", "trigonometric", "logarithmic", "exponential", "interaction"]:
                if family in pattern:
                    target_families.append(family)
        return target_families

    @staticmethod
    def _active_variables(dataset, residual):
        residual = dict(residual or {})
        variables = []
        variables.extend(residual.get("suggested_active_variables", []) or [])
        probe = dict(residual.get("best_probe", {}) or {})
        variables.extend(probe.get("variables", []) or [])
        allowed = set(getattr(dataset, "feature_names", []) or [])
        out = []
        for name in variables:
            name = str(name)
            if name in allowed and name not in out:
                out.append(name)
        return out or list(getattr(dataset, "feature_names", []) or [])[:4]

    def _heuristic_decide(self, current_best, evaluation, iter_cfg, round_idx):
        fallback = super()._heuristic_decide(current_best, evaluation, iter_cfg, round_idx)
        dataset = getattr(self, "_critic_dataset", None)
        residual = dict(evaluation.get("residual_summary", {}) or {})
        best = current_best or evaluation.get("best_result")
        best_val = _val_mse(best) if best is not None else None
        best_complexity = _complexity(best) if best is not None else float("inf")
        target_families = self._extract_family_plan(residual)
        active_variables = self._active_variables(dataset, residual) if dataset is not None else []
        best_probe = dict(residual.get("best_probe", {}) or {})
        probe_gain = _finite_float(best_probe.get("combined_gain"))
        pattern_type = str(residual.get("pattern_type", "") or "")

        action = "refine"
        should_refine = bool(fallback.get("should_refine", True))
        reason_parts = [str(fallback.get("reason", "critic fallback"))]

        if best_val is not None and best_val <= V11_EXACT_TOL and best_complexity <= V11_CRITIC_HIGH_COMPLEXITY:
            action = "stop"
            should_refine = False
            reason_parts.append("exact-enough compact expression")
        elif best_val is not None and best_val <= V11_CRITIC_LOW_ERROR and best_complexity > V11_CRITIC_HIGH_COMPLEXITY:
            action = "complexity_prune"
            should_refine = True
            reason_parts.append("low error but expression is too complex")
        elif V11_ENABLE_STRUCTURAL_RESCUE and target_families:
            action = "structural_rescue"
            should_refine = True
            reason_parts.append("residual indicates missing structural family")
        elif probe_gain is not None and probe_gain < V11_CRITIC_STALL_GAIN and best_complexity > V11_CRITIC_HIGH_COMPLEXITY:
            action = "complexity_prune"
            should_refine = True
            reason_parts.append("diffuse residual with high complexity")
        elif best is None:
            action = "multimodal_rescue"
            should_refine = True
            reason_parts.append("no valid candidate yet")

        use_multimodal = bool(action == "multimodal_rescue")
        if dataset is not None and len(getattr(dataset, "feature_names", []) or []) >= 5:
            use_multimodal = use_multimodal or (
                best_val is None or best_val > float(iter_cfg.get("mm_trigger_val_mse", 1.0))
            )

        family_to_actions = {
            "rational": ["denominator_correction", "ratio_rewrite"],
            "power": ["add_power_term", "structured_power_correction"],
            "trigonometric": ["add_periodic_component", "periodic_modulation"],
            "logarithmic": ["add_log_component"],
            "exponential": ["switch_family", "add_exponential_component"],
            "interaction": ["expand_interaction"],
        }
        actions = []
        for family in target_families:
            actions.extend(family_to_actions.get(family, ["switch_family"]))
        if action == "complexity_prune":
            actions = ["simplify", "complexity_prune", "local_rewrite"] + actions
        if not actions:
            actions = list(fallback.get("actions", []) or ["local_rewrite"])

        budget = dict(fallback.get("budget", {}) or {})
        budget["num_candidates"] = max(int(budget.get("num_candidates", 3)), 6 if action != "stop" else 0)
        budget["temperature"] = 0.35 if action in {"complexity_prune", "structural_rescue"} else budget.get("temperature", 0.4)
        budget["use_multimodal"] = bool(use_multimodal)

        fallback.update({
            "should_refine": bool(should_refine),
            "confidence": max(float(fallback.get("confidence", 0.45) or 0.45), 0.62),
            "reason": "; ".join(x for x in reason_parts if x),
            "critic_action": action,
            "target_families": target_families,
            "active_variables": active_variables,
            "pattern_type": pattern_type,
            "complexity_pressure": "high" if best_complexity > V11_CRITIC_HIGH_COMPLEXITY else "normal",
            "actions": _uniq(actions)[:8],
            "repair_targets": list(fallback.get("repair_targets", []) or [])[:3] + [
                f"target {family} structure" for family in target_families
            ],
            "budget": budget,
        })
        return fallback

    def decide(self, dataset, observation, evaluation, current_best, iter_cfg, round_idx, row_meta=None, diagnostic_image_paths=None):
        self._critic_dataset = dataset
        # A numeric-only arm must remain text-only throughout the agent loop,
        # including Critic residual diagnostics.  A non-existent sentinel
        # prevents the base MetaAgent from auto-generating an image when an
        # empty/None list is supplied; the base class filters it before use.
        if V11_OBSERVER_INPUT_MODE in {"numeric", "numeric_only", "no_image", "no_images"}:
            diagnostic_image_paths = ["__llmsr_numeric_only_no_image__"]
        try:
            decision = super().decide(
                dataset,
                observation,
                evaluation,
                current_best,
                iter_cfg,
                round_idx,
                row_meta=row_meta,
                diagnostic_image_paths=diagnostic_image_paths,
            )
        except TypeError as exc:
            # Some remote runners import an older base MetaAgent whose decide
            # signature lacks row_meta/diagnostic_image_paths. Keep V11
            # compatible with that base instead of silently skipping the loop.
            if "unexpected keyword argument" not in str(exc):
                raise
            decision = super().decide(
                dataset,
                observation,
                evaluation,
                current_best,
                iter_cfg,
                round_idx,
            )
        if "critic_action" not in decision:
            heuristic = self._heuristic_decide(current_best, evaluation, iter_cfg, round_idx)
            for key in [
                "critic_action",
                "target_families",
                "active_variables",
                "pattern_type",
                "complexity_pressure",
            ]:
                decision[key] = heuristic.get(key)
            budget = dict(decision.get("budget", {}) or {})
            budget.update({
                k: v for k, v in dict(heuristic.get("budget", {}) or {}).items()
                if k not in budget or k == "use_multimodal"
            })
            decision["budget"] = budget
        decision["controller"] = "critic_agent_v11"
        return decision


def _stable_visual_control_seed(row_meta) -> int:
    repeat = os.environ.get("LLMSR_REPEAT_SEED", "0")
    text = f"{(row_meta or {}).get('base_name', '')}|{(row_meta or {}).get('dataset_case_id', '')}|{repeat}"
    return int(sum((idx + 1) * ord(ch) for idx, ch in enumerate(text)) % (2**32 - 1))


def _dataset_with_permuted_train_target(dataset, row_meta):
    cloned = copy.copy(dataset)
    train_df = dataset.train_df.copy()
    target_name = getattr(dataset, "target_name", "y")
    if target_name in train_df.columns:
        rng = np.random.default_rng(_stable_visual_control_seed(row_meta))
        train_df[target_name] = rng.permutation(np.asarray(train_df[target_name], dtype=float))
    cloned.train_df = train_df
    return cloned


def _copy_observation_without_graphics(observation):
    out = copy.copy(observation)
    out.plot_descriptions = []
    out.image_paths = []
    out.visual_hints = ["no graphical evidence supplied"]
    out.reconstruction_image_paths = []
    out.reconstruction_descriptions = []
    visual_summary = dict(getattr(out, "visual_summary", None) or {})
    visual_summary["num_plots"] = 0
    visual_summary["plot_inventory"] = []
    out.visual_summary = _base.make_json_safe(visual_summary)
    out.mm_assets_attempted = True
    out.mm_assets_succeeded = False
    return out


def _copy_observation_with_control_visuals(observation, visual_observation):
    out = copy.copy(observation)
    out.plot_descriptions = list(getattr(visual_observation, "plot_descriptions", []) or [])
    out.image_paths = list(getattr(visual_observation, "image_paths", []) or [])
    out.visual_hints = list(getattr(visual_observation, "visual_hints", []) or []) or ["plots generated but no explicit visual hints"]
    out.reconstruction_image_paths = list(getattr(visual_observation, "reconstruction_image_paths", []) or [])
    out.reconstruction_descriptions = list(getattr(visual_observation, "reconstruction_descriptions", []) or [])
    out.visual_summary = _base.make_json_safe(
        getattr(visual_observation, "visual_summary", None) or getattr(observation, "visual_summary", {}) or {}
    )
    out.structure_profile = _base.make_json_safe(getattr(observation, "structure_profile", {}) or {})
    out.structure_hints = list(getattr(observation, "structure_hints", []) or [])
    out.mm_assets_attempted = bool(getattr(visual_observation, "mm_assets_attempted", True))
    out.mm_assets_succeeded = bool(getattr(visual_observation, "mm_assets_succeeded", bool(out.image_paths)))
    out.mm_assets_error = getattr(visual_observation, "mm_assets_error", None)
    return out


def _matched_round_candidate_limit(iter_cfg) -> int:
    return max(1, int((iter_cfg or {}).get("refined_k", V11_FULL_BUDGET_REFINED_K)))


def _generic_judge_feedback() -> Dict[str, Any]:
    lines = [line.strip() for line in str(V11_GENERIC_FEEDBACK_TEXT).splitlines() if line.strip()]
    return {
        "feedback_text": "\n".join(lines),
        "keep_constraints": ["keep the equation compact"],
        "repair_targets": ["improve validation fit", "try one alternative operator family"],
        "avoid_patterns": ["avoid repeating failed structures without simplification"],
        "raw_text": "generic fixed feedback control",
        "control_mode": "generic",
    }


def _generic_refinement_decision(current_best, iter_cfg, round_idx: int) -> Dict[str, Any]:
    current_expr = _base._safe_get_attr(current_best, "simplified_expression", None) if current_best is not None else None
    return {
        "should_refine": True,
        "critic_action": "refine",
        "reason": "generic matched feedback control",
        "target_exprs": [current_expr] if current_expr else [],
        "preserve_patterns": [],
        "repair_targets": ["validation fit", "compactness", "alternative operator family"],
        "actions": ["local simplification", "operator-family alternative"],
        "target_families": [],
        "active_variables": [],
        "budget": {
            "num_candidates": _matched_round_candidate_limit(iter_cfg),
            "temperature": 0.35,
            "use_multimodal": False,
        },
        "round_idx": int(round_idx),
        "control_mode": "generic",
    }


def _force_matched_refinement_budget(decision: Dict[str, Any], iter_cfg, round_idx: int) -> Dict[str, Any]:
    out = dict(decision or {})
    budget = dict(out.get("budget", {}) or {})
    budget["num_candidates"] = _matched_round_candidate_limit(iter_cfg)
    budget["use_multimodal"] = False
    out["budget"] = budget
    out["round_idx"] = int(round_idx)
    if V11_FORCE_REFINEMENT_ROUNDS:
        out["should_refine"] = True
        if str(out.get("critic_action") or "").lower() == "stop":
            out["original_critic_action"] = out.get("critic_action")
            out["critic_action"] = "refine"
            out["reason"] = f"forced matched-budget refinement after stop suggestion: {out.get('reason', '')}"
    return out


class V11ObserverAgent:
    """Programmatic Observer + optional VLM Observer augmentation.

    The base Observer is deterministic: it builds structure_profile, visual
    hints, plots and high-dimensional reconstruction tokens.  This wrapper keeps
    that path intact, then asks a VLM to read the generated plots plus a compact
    data/profile summary and adds the model's structure judgement as auxiliary
    evidence for the Proposer/Critic.
    """

    def __init__(self, inner_observer, client=None, enabled: bool = True):
        self.inner = inner_observer
        self.client = client
        self.enabled = bool(enabled and client is not None)

    def observe(self, dataset, row_meta, timer=None):
        observation = self.inner.observe(dataset, row_meta=row_meta, timer=timer)
        if not self.enabled:
            return self._apply_observer_input_mode(dataset, row_meta, observation, timer=timer)

        if V11_VLM_OBSERVER_GENERATE_IMAGES and not list(getattr(observation, "image_paths", []) or []):
            observation = self.inner.maybe_generate_mm_assets(dataset, row_meta, observation, timer=timer)

        observation = self._apply_observer_input_mode(dataset, row_meta, observation, timer=timer)
        vlm_result = self._call_vlm_observer(dataset, row_meta, observation)
        return self._merge_vlm_observation(dataset, observation, vlm_result)

    def maybe_generate_mm_assets(self, dataset, row_meta, observation, timer=None):
        observation = self.inner.maybe_generate_mm_assets(dataset, row_meta, observation, timer=timer)
        return self._apply_observer_input_mode(dataset, row_meta, observation, timer=timer)

    def _apply_observer_input_mode(self, dataset, row_meta, observation, timer=None):
        mode = V11_OBSERVER_INPUT_MODE
        if mode in {"native", "aligned", "aligned_control", "multimodal"}:
            return observation
        if mode in {"numeric", "numeric_only", "no_image", "no_images"}:
            return _copy_observation_without_graphics(observation)
        if mode in {"permuted", "permuted_control", "shuffled_visual", "visual_control"}:
            permuted_dataset = _dataset_with_permuted_train_target(dataset, row_meta)
            plot_meta = dict(row_meta or {})
            plot_meta["base_name"] = f"{plot_meta.get('base_name', 'case')}_visual_control"
            visual_seed = copy.copy(observation)
            visual_seed.plot_descriptions = []
            visual_seed.image_paths = []
            visual_seed.visual_hints = []
            visual_seed.reconstruction_tokens = None
            visual_seed.reconstruction_trace = None
            visual_seed.reconstruction_image_paths = []
            visual_seed.reconstruction_descriptions = []
            visual_observation = self.inner.maybe_generate_mm_assets(permuted_dataset, plot_meta, visual_seed, timer=timer)
            return _copy_observation_with_control_visuals(observation, visual_observation)
        return observation

    def _selected_images(self, observation) -> List[str]:
        image_paths = list(getattr(observation, "image_paths", []) or [])
        if not image_paths:
            image_paths = list(getattr(observation, "reconstruction_image_paths", []) or [])
        try:
            return _base.choose_mm_image_paths(
                image_paths,
                max_images=max(1, V11_VLM_OBSERVER_MAX_IMAGES),
                plot_descriptions=list(getattr(observation, "plot_descriptions", []) or []),
                focus_variables=list(((getattr(observation, "visual_summary", {}) or {}).get("dominant_variables", []) or [])),
                focus_pairs=list(((getattr(observation, "visual_summary", {}) or {}).get("focus_pairs", []) or [])),
            )
        except Exception:
            return image_paths[:max(1, V11_VLM_OBSERVER_MAX_IMAGES)]

    def _compact_profile(self, observation) -> Dict[str, Any]:
        structure_profile = dict(getattr(observation, "structure_profile", None) or {})
        visual_summary = dict(getattr(observation, "visual_summary", None) or {})
        try:
            compact_structure, compact_visual = _base._compact_observation_for_prompt(
                structure_profile=structure_profile,
                visual_summary=visual_summary,
            )
        except Exception:
            compact_structure, compact_visual = structure_profile, visual_summary
        return {
            "structure_profile": _base.make_json_safe(compact_structure),
            "visual_summary": _base.make_json_safe(compact_visual),
            "structure_hints": list(getattr(observation, "structure_hints", []) or [])[:10],
            "visual_hints": list(getattr(observation, "visual_hints", []) or [])[:10],
            "reconstruction_tokens": _base.make_json_safe(getattr(observation, "reconstruction_tokens", None)),
        }

    def _build_prompt(self, dataset, row_meta, observation) -> str:
        try:
            csv_points = _base.make_csv_points_text(
                df=dataset.train_df,
                variable_names=dataset.feature_names,
                target_name=dataset.target_name,
                max_rows=max(4, V11_VLM_OBSERVER_MAX_ROWS),
            )
        except Exception:
            csv_points = ""
        return f"""
You are the VLM Observer Agent in a symbolic regression loop.

Task description:
Your job is to observe the benchmark data before formula proposal. Given the attached plot(s), exact CSV samples, and benchmark metadata, you are required to generate a structural observation summary for the downstream Proposer and Critic agents.

Input:
1) Attached image(s): visual plot(s) of y against the input variables.
2) Benchmark metadata: dataset name, feature_names, and target_name.
3) CSV samples: exact sampled numeric rows from the benchmark.

Output:
Return a JSON object that identifies likely active/inactive variables, possible operator families, variable roles, visual evidence, risk notes, and confidence. This output is structural guidance only, not a final formula or candidate list.

Dataset:
- base_name: {row_meta.get("base_name") if isinstance(row_meta, dict) else ""}
- feature_names: {list(getattr(dataset, "feature_names", []) or [])}
- target_name: {getattr(dataset, "target_name", "y")}

Required JSON schema:
{{
  "agent": "observer",
  "active_variables": ["x1"],
  "inactive_variables": [],
  "candidate_families": ["rational", "interaction", "power"],
  "variable_roles": {{
    "numerator_core": ["x1"],
    "denominator_core": [],
    "periodic_core": []
  }},
  "trend_summary": "brief visual/data summary",
  "visual_evidence": ["specific evidence from plots or CSV"],
  "risk_notes": ["mistakes to avoid"],
  "confidence": 0.0
}}

Restrictions:
1) Output JSON only.
2) Prefer compact structure and sparse active variables.
3) Use exactly the provided feature names.
4) Infer structure from the image, benchmark information, and CSV samples.
5) Do not output candidate expressions; only output structural observations.
6) Do not reveal or invent a final exact expression.

CSV samples:
{csv_points}
""".strip()

    def _call_vlm_observer(self, dataset, row_meta, observation) -> Dict[str, Any]:
        if self.client is None:
            return {"enabled": False, "error": "no observer client"}
        image_paths = self._selected_images(observation)
        prompt = self._build_prompt(dataset, row_meta, observation)
        user_content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        for path in image_paths:
            try:
                user_content.append({"type": "image_url", "image_url": {"url": _base.image_file_to_data_url(path)}})
            except Exception:
                continue
        try:
            response = self.client.generate(
                messages=[
                    _base.Message(role="system", content="You are a careful multimodal Observer Agent. Output JSON only."),
                    _base.Message(role="user", content=user_content),
                ],
                temperature=V11_VLM_OBSERVER_TEMPERATURE,
                max_tokens=V11_VLM_OBSERVER_MAX_TOKENS,
                top_p=1.0,
            )
            raw_text = response.text
            obj = _base._extract_first_json_object(raw_text)
            if not isinstance(obj, dict):
                return {"enabled": True, "used_llm": True, "error": "json_parse_failed", "raw_text": raw_text}
            obj["enabled"] = True
            obj["used_llm"] = True
            obj["raw_text"] = raw_text
            obj["image_paths"] = image_paths
            return _base.make_json_safe(obj)
        except Exception as exc:
            return {"enabled": True, "used_llm": False, "error": repr(exc), "image_paths": image_paths}

    def _merge_vlm_observation(self, dataset, observation, vlm_result: Dict[str, Any]):
        vlm_result = dict(vlm_result or {})
        structure_profile = dict(getattr(observation, "structure_profile", None) or {})
        visual_summary = dict(getattr(observation, "visual_summary", None) or {})
        feature_names = list(getattr(dataset, "feature_names", []) or [])

        structure_profile["vlm_observer"] = _base.make_json_safe(vlm_result)
        visual_summary["vlm_observer"] = {
            "enabled": bool(vlm_result.get("enabled", False)),
            "used_llm": bool(vlm_result.get("used_llm", False)),
            "candidate_families": list(vlm_result.get("candidate_families", []) or [])[:5],
            "trend_summary": str(vlm_result.get("trend_summary", "") or "")[:500],
        }

        roles = dict(structure_profile.get("variable_roles", {}) or {})
        base_active = list(roles.get("active_variables", []) or structure_profile.get("active_variables", []) or [])
        vlm_active = _ordered_unique(vlm_result.get("active_variables", []) or [], allowed=feature_names)
        shared_active = [x for x in base_active if x in set(vlm_active)]
        merged_active = _ordered_unique(shared_active + base_active + vlm_active + feature_names[:1], allowed=feature_names, limit=min(5, len(feature_names) or 5))
        if merged_active:
            structure_profile["active_variables"] = merged_active
            roles["active_variables"] = merged_active

        vlm_roles = dict(vlm_result.get("variable_roles", {}) or {})
        for key in ["numerator_core", "denominator_core", "periodic_core"]:
            roles[key] = _ordered_unique(
                list(roles.get(key, []) or []) + list(vlm_roles.get(key, []) or []),
                allowed=feature_names,
                limit=min(4, len(feature_names) or 4),
            )
        if not roles.get("numerator_core") and merged_active:
            roles["numerator_core"] = merged_active[:min(3, len(merged_active))]
        structure_profile["variable_roles"] = roles

        family_scores = dict(structure_profile.get("family_scores", {}) or {})
        for family in list(vlm_result.get("candidate_families", []) or []):
            key = _family_key(family)
            if key:
                family_scores[key] = max(float(family_scores.get(key, 0.0) or 0.0), 0.22)
        if family_scores:
            structure_profile["family_scores"] = family_scores

        new_structure_hints = list(getattr(observation, "structure_hints", []) or [])
        if vlm_result.get("candidate_families"):
            new_structure_hints.append(
                "vlm observer candidate families: " + ", ".join(str(x) for x in list(vlm_result.get("candidate_families", []) or [])[:4])
            )
        if merged_active:
            new_structure_hints.append("vlm observer active variables: " + ", ".join(merged_active))
        if vlm_result.get("trend_summary"):
            new_structure_hints.append("vlm observer trend: " + str(vlm_result.get("trend_summary")))

        new_visual_hints = list(getattr(observation, "visual_hints", []) or [])
        for line in list(vlm_result.get("visual_evidence", []) or [])[:5]:
            new_visual_hints.append("vlm observer evidence: " + str(line))
        for line in list(vlm_result.get("risk_notes", []) or [])[:4]:
            new_visual_hints.append("vlm observer caution: " + str(line))
        if vlm_result.get("error"):
            new_visual_hints.append("vlm observer failed: " + str(vlm_result.get("error")))

        observation.structure_hints = _base.deduplicate_expressions(new_structure_hints)
        observation.visual_hints = _base.deduplicate_expressions(new_visual_hints)
        observation.structure_profile = _base.make_json_safe(structure_profile)
        observation.visual_summary = _base.make_json_safe(visual_summary)
        return observation


def _apply_v11_patch():
    _base.get_best_result = get_best_result
    _base.is_better_result = is_better_result
    _base._is_benchmark_task = _v11_is_benchmark_task
    _base._is_benchmark_like_source_tag = _v11_is_benchmark_like_source_tag
    _base.AlgebraicSimplifyTool = V11AlgebraicSimplifyTool
    _base.build_manual_candidates = build_manual_candidates
    _base.MetaAgent = CriticAgent
    _base.EvaluatorAgent = V11EvaluatorAgent
    _base.COMPLEXITY_WEIGHT = V11_COMPLEXITY_WEIGHT
    _base.METHOD_MODE = "planner_guided_v11_image_loop"
    _base.EVAL_PROFILE = f"{getattr(_base, 'EVAL_PROFILE', 'quality')}_v11"
    _base.ENABLE_RUNTIME_FAST_PATH = not V11_DISABLE_RUNTIME_FAST_PATH
    if V11_RESTORE_TEXT_PROPOSER:
        _base.generate_multiple_text_single_expressions_with_stats = _v11_generate_multiple_text_single_expressions_with_stats
    runtime_sec = _env_runtime_budget_sec()
    if hasattr(_base, "ROLE_BACKEND_CONFIGS"):
        shared_model = os.environ.get("LLMSR_AGENT_MODEL", os.environ.get("LLMSR_PLANNER_MODEL", ""))
        shared_api_base = os.environ.get("LLMSR_AGENT_API_BASE", os.environ.get("LLMSR_PLANNER_API_BASE", ""))
        shared_api_key = os.environ.get("LLMSR_AGENT_API_KEY", os.environ.get("LLMSR_PLANNER_API_KEY", "EMPTY"))
        if shared_model:
            _base.BACKEND_CONFIG["model"] = shared_model
        if shared_api_base:
            _base.BACKEND_CONFIG["api_base_url"] = shared_api_base
        if shared_api_key:
            _base.BACKEND_CONFIG["api_key"] = shared_api_key
        for role in ["proposal", "meta", "critic", "judge", "refiner", "planner", "observer"]:
            cfg = dict(
                _base.ROLE_BACKEND_CONFIGS.get(role)
                or _base.ROLE_BACKEND_CONFIGS.get("proposal")
                or _base.BACKEND_CONFIG
            )
            env_prefix = f"LLMSR_{role.upper()}"
            model = os.environ.get(f"{env_prefix}_MODEL", shared_model or str(cfg.get("model", _base.BACKEND_CONFIG.get("model", ""))))
            api_base = os.environ.get(f"{env_prefix}_API_BASE", shared_api_base or str(cfg.get("api_base_url", _base.BACKEND_CONFIG.get("api_base_url", ""))))
            api_key = os.environ.get(f"{env_prefix}_API_KEY", shared_api_key or str(cfg.get("api_key", _base.BACKEND_CONFIG.get("api_key", "EMPTY"))))
            timeout = float(os.environ.get(f"{env_prefix}_TIMEOUT", str(cfg.get("timeout", 45))))
            if runtime_sec is not None and runtime_sec < 180.0:
                timeout = min(timeout, max(8.0, float(runtime_sec) * 0.18))
            cfg.update({
                "model": model,
                "api_base_url": api_base,
                "api_key": api_key,
                "timeout": timeout,
            })
            _base.ROLE_BACKEND_CONFIGS[role] = cfg
    _base.LOW_DIM_BENCHMARK_SKIP_REFINE_VAL_MSE = min(
        float(getattr(_base, "LOW_DIM_BENCHMARK_SKIP_REFINE_VAL_MSE", 5e-3)),
        1e-4,
    )
    if _budget_aware_enabled(runtime_sec):
        profile = _budget_profile(runtime_sec)
        _base.ENABLE_RUNTIME_FAST_PATH = False if runtime_sec >= 180.0 else not V11_DISABLE_RUNTIME_FAST_PATH
        if "LLMSR_V11_IMAGE_LOOP_MAX_ROUNDS" not in os.environ:
            globals()["V11_IMAGE_LOOP_MAX_ROUNDS"] = int(profile["image_loop_max_rounds"])
        if "LLMSR_V11_IMAGE_LOOP_MAX_AUGMENTED_CANDIDATES" not in os.environ:
            globals()["V11_IMAGE_LOOP_MAX_AUGMENTED_CANDIDATES"] = int(profile["max_augmented_candidates"])


def _maybe_expand_low_dim_budget(iter_cfg, dataset, row_meta):
    runtime_sec = _env_runtime_budget_sec()
    budget_aware = _budget_aware_enabled(runtime_sec)
    if not (V11_LOW_DIM_FULL_BUDGET or budget_aware):
        return iter_cfg
    profile = _budget_profile(runtime_sec)
    feature_names = list(getattr(dataset, "feature_names", []) or [])
    is_benchmark = str((row_meta or {}).get("dataset_dir", "")) == "benchmark_csv"
    dim_limit = int(getattr(_base, "FAST_PATH_BENCHMARK_DIM_LIMIT", 2))
    if not (is_benchmark and 0 < len(feature_names) <= dim_limit):
        return iter_cfg
    tuned = dict(iter_cfg or {})
    text_calls = V11_LOW_DIM_FULL_TEXT_CALLS if V11_LOW_DIM_FULL_BUDGET else int(profile["low_dim_text_calls"])
    mm_calls = V11_LOW_DIM_FULL_MM_CALLS if V11_LOW_DIM_FULL_BUDGET else int(profile["low_dim_mm_calls"])
    proposal_k = V11_LOW_DIM_FULL_PROPOSAL_K if V11_LOW_DIM_FULL_BUDGET else int(profile["low_dim_proposal_k"])
    refined_k = V11_LOW_DIM_FULL_REFINED_K if V11_LOW_DIM_FULL_BUDGET else int(profile["low_dim_refined_k"])
    refine_rounds = V11_LOW_DIM_FULL_REFINE_ROUNDS if V11_LOW_DIM_FULL_BUDGET else int(profile["low_dim_refine_rounds"])
    skip_refine_val = V11_LOW_DIM_FULL_SKIP_REFINE_VAL_MSE if V11_LOW_DIM_FULL_BUDGET else float(profile["skip_refine_val_mse"])
    tuned["text_calls"] = _budget_cap_or_floor(int(tuned.get("text_calls", 0)), text_calls, runtime_sec)
    tuned["mm_calls"] = _budget_cap_or_floor(int(tuned.get("mm_calls", 0)), mm_calls, runtime_sec)
    tuned["proposal_k"] = _budget_cap_or_floor(int(tuned.get("proposal_k", 0)), proposal_k, runtime_sec)
    tuned["refined_k"] = _budget_cap_or_floor(int(tuned.get("refined_k", 0)), refined_k, runtime_sec)
    tuned["refine_rounds"] = _budget_cap_or_floor(int(tuned.get("refine_rounds", 0)), refine_rounds, runtime_sec)
    tuned["skip_refine_val_mse"] = min(
        float(tuned.get("skip_refine_val_mse", 1.0)),
        skip_refine_val,
    )
    tuned["skip_diverse_proposal"] = bool(runtime_sec is not None and runtime_sec < 180.0)
    tuned["force_heuristic_agents"] = bool(runtime_sec is not None and runtime_sec < 180.0)
    if V11_FORCE_DIVERSE_LOW_DIM:
        tuned["skip_diverse_proposal"] = False
    if V11_DISABLE_HEURISTIC_FALLBACK:
        tuned["force_heuristic_agents"] = False
    tuned["low_dim_full_budget"] = True
    tuned["v11_budget_profile"] = profile["name"]
    return tuned


def _maybe_expand_full_budget(iter_cfg, dataset, row_meta):
    runtime_sec = _env_runtime_budget_sec()
    budget_aware = _budget_aware_enabled(runtime_sec)
    if not (V11_FULL_BUDGET or budget_aware):
        return iter_cfg
    profile = _budget_profile(runtime_sec)
    tuned = dict(iter_cfg or {})
    text_calls = V11_FULL_BUDGET_TEXT_CALLS if V11_FULL_BUDGET else int(profile["text_calls"])
    mm_calls = V11_FULL_BUDGET_MM_CALLS if V11_FULL_BUDGET else int(profile["mm_calls"])
    proposal_k = V11_FULL_BUDGET_PROPOSAL_K if V11_FULL_BUDGET else int(profile["proposal_k"])
    refined_k = V11_FULL_BUDGET_REFINED_K if V11_FULL_BUDGET else int(profile["refined_k"])
    refine_rounds = V11_FULL_BUDGET_REFINE_ROUNDS if V11_FULL_BUDGET else int(profile["refine_rounds"])
    skip_refine_val = V11_FULL_BUDGET_SKIP_REFINE_VAL_MSE if V11_FULL_BUDGET else float(profile["skip_refine_val_mse"])
    tuned["text_calls"] = _budget_cap_or_floor(int(tuned.get("text_calls", 0)), text_calls, runtime_sec)
    tuned["mm_calls"] = _budget_cap_or_floor(int(tuned.get("mm_calls", 0)), mm_calls, runtime_sec)
    tuned["proposal_k"] = _budget_cap_or_floor(int(tuned.get("proposal_k", 0)), proposal_k, runtime_sec)
    tuned["refined_k"] = _budget_cap_or_floor(int(tuned.get("refined_k", 0)), refined_k, runtime_sec)
    tuned["refine_rounds"] = _budget_cap_or_floor(int(tuned.get("refine_rounds", 0)), refine_rounds, runtime_sec)
    tuned["skip_refine_val_mse"] = min(
        float(tuned.get("skip_refine_val_mse", 1.0)),
        skip_refine_val,
    )
    tuned["skip_diverse_proposal"] = bool(runtime_sec is not None and runtime_sec < 180.0)
    tuned["force_heuristic_agents"] = bool(runtime_sec is not None and runtime_sec < 180.0)
    if V11_FORCE_DIVERSE_LOW_DIM:
        tuned["skip_diverse_proposal"] = False
    if V11_DISABLE_HEURISTIC_FALLBACK:
        tuned["force_heuristic_agents"] = False
    tuned["full_budget"] = True
    tuned["runtime_fast_path_disabled"] = bool(V11_DISABLE_RUNTIME_FAST_PATH or budget_aware)
    tuned["v11_budget_profile"] = profile["name"]
    tuned["v11_budget_runtime_sec"] = runtime_sec
    return tuned


_apply_v11_patch()

# Re-export the v10-compatible surface.  __getattr__ covers functions and
# constants looked up by the benchmark wrappers after import.
def __getattr__(name):
    return getattr(_base, name)


_RUNTIME_GLOBALS = [
    "RESULTS_ROOT",
    "GLOBAL_SUMMARY_CSV",
    "GLOBAL_SUMMARY_JSON",
    "GLOBAL_SUMMARY_CSV_COMPACT",
    "TIMING_BREAKDOWN_CSV",
    "TIMING_SUMMARY_JSON",
    "PER_CASE_JSON_DIR",
    "SELECTED_TASKS_CSV",
    "MSE_THRESHOLD",
    "PERFECT_FIT_TOL",
    "MAX_RUNTIME_PER_TASK_SEC",
]


def _sync_runtime_globals():
    for name in _RUNTIME_GLOBALS:
        if name in globals():
            setattr(_base, name, globals()[name])
    _apply_v11_patch()


def _matched_initial_mm_proposal(
    proposer_agent,
    observer_agent,
    dataset,
    observation,
    row_meta,
    iter_cfg,
    timer=None,
):
    """One compact, auditable image-conditioned proposal call."""
    try:
        image_paths = list(observer_agent._selected_images(observation) or [])
    except Exception:
        image_paths = list(getattr(observation, "image_paths", []) or [])[:1]
    image_paths = image_paths[:1]
    if not image_paths:
        return {
            "candidate_exprs": [],
            "trace": {
                "num_mm_exprs": 0,
                "image_paths": [],
                "error": "no_image_available_for_matched_multimodal_proposal",
            },
        }

    structure_profile = dict(getattr(observation, "structure_profile", {}) or {})
    visual_summary = dict(getattr(observation, "visual_summary", {}) or {})
    try:
        structure_profile, visual_summary = _base._compact_observation_for_prompt(
            structure_profile=structure_profile,
            visual_summary=visual_summary,
        )
    except Exception:
        pass

    if timer is not None:
        timer.start("step_matched_initial_mm_proposal")
    try:
        form_result = _base.generate_multiple_mm_formula_form_candidates(
            client=proposer_agent.proposal_llm.client,
            df=dataset.train_df,
            variable_names=dataset.feature_names,
            target_name=dataset.target_name,
            image_paths=image_paths,
            plot_descriptions=list(getattr(observation, "plot_descriptions", []) or [])[:4],
            structure_hints=list(getattr(observation, "structure_hints", []) or [])[:8],
            structure_profile=structure_profile,
            visual_summary=visual_summary,
            reconstruction_tokens=None,
            allowed_operators=_base.ALLOWED_OPERATORS,
            max_rows=V11_MATCHED_MM_MAX_ROWS,
            num_calls=V11_MATCHED_MODALITY_CALLS,
            temperatures=[0.1],
            max_tokens=V11_MATCHED_MM_MAX_TOKENS,
            max_workers=1,
            num_candidates_per_call=V11_MATCHED_MM_CANDIDATES_PER_CALL,
        )
        form_exprs = [
            str(item.get("expression", "")).strip()
            for item in list(form_result.get("candidates", []) or [])
            if isinstance(item, dict) and str(item.get("expression", "")).strip()
        ]
        guided_candidates = _base.build_vlm_guided_template_candidates(
            candidate_items=list(form_result.get("candidates", []) or []),
            feature_names=dataset.feature_names,
            row_meta=row_meta,
            top_k_per_item=1,
        )
        guided_exprs = [
            str(item.get("expression", "")).strip()
            for item in list(guided_candidates or [])
            if isinstance(item, dict) and str(item.get("expression", "")).strip()
        ]
        exprs = _base.deduplicate_expressions(guided_exprs + form_exprs)
        return {
            "candidate_exprs": exprs,
            "trace": {
                "num_mm_exprs": len(exprs),
                "image_paths": image_paths,
                "max_rows": V11_MATCHED_MM_MAX_ROWS,
                "max_tokens": V11_MATCHED_MM_MAX_TOKENS,
                "num_candidates_per_call": V11_MATCHED_MM_CANDIDATES_PER_CALL,
                "form_proposal_stats": _base.make_json_safe(form_result),
                "guided_template_count": len(guided_exprs),
            },
        }
    except Exception as exc:
        return {
            "candidate_exprs": [],
            "trace": {
                "num_mm_exprs": 0,
                "image_paths": image_paths,
                "error": repr(exc),
            },
        }
    finally:
        if timer is not None:
            timer.stop("step_matched_initial_mm_proposal")


def _make_loop_result(row_meta, dataset, iter_cfg):
    return {
        "eval_profile": getattr(_base, "EVAL_PROFILE", None),
        "method_mode": getattr(_base, "METHOD_MODE", None),
        "no_leakage_mode": bool(getattr(_base, "NO_LEAKAGE_MODE", True)),
        "no_leakage_audit": None,
        "best_expr_source": None,
        "task_type": row_meta.get("task_type"),
        "dataset_dir": row_meta.get("dataset_dir"),
        "difficulty": row_meta.get("difficulty"),
        "base_name": row_meta.get("base_name"),
        "true_expression": row_meta.get("true_expression"),
        "n_features": len(dataset.feature_names),
        "n_train": len(dataset.train_df),
        "n_val": len(dataset.val_df),
        "n_test": len(dataset.test_df),
        "valid_formula_found": False,
        "num_candidate_exprs": 0,
        "raw_exprs": None,
        "best_expr": None,
        "best_val_mse": None,
        "best_test_mse": None,
        "passed": False,
        "perfect_fit": False,
        "runtime_sec": None,
        "num_plots": None,
        "plot_descriptions": None,
        "visual_hints": None,
        "image_paths": None,
        "residual_summary": None,
        "physics_summary": None,
        "mm_proposal_stats": None,
        "mm_requested": False,
        "mm_trigger_reason": None,
        "mm_assets_attempted": False,
        "mm_assets_succeeded": False,
        "mm_assets_error": None,
        "mm_candidate_count": 0,
        "mm_used_in_evaluation": False,
        "visual_trace": None,
        "reconstruction_tokens": None,
        "reconstruction_trace": None,
        "reconstruction_image_paths": None,
        "text_proposal_stats": None,
        "initial_formula_form_eval": None,
        "initial_best_form_match_score": None,
        "vlm_formula_form_eval": None,
        "vlm_best_form_match_score": None,
        "meta_decisions": None,
        "refine_history": None,
        "refine_improvement_count": None,
        "diverse_proposal_count": None,
        "manual_candidate_count": None,
        "protected_candidate_count": None,
        "memory_candidate_count": None,
        "experience_candidate_count": None,
        "heuristic_candidate_count": None,
        "datadriven_candidate_count": None,
        "generic_candidate_count": None,
        "structural_planner_candidate_count": 0,
        "extrapolation_ols_candidate_count": 0,
        "extrapolation_ols_trace": None,
        "surface_analytic_candidate_count": 0,
        "surface_analytic_trace": None,
        "surface_grid_candidate_count": 0,
        "surface_grid_trace": None,
        "surface_stable_candidate_count": 0,
        "surface_stable_trace": None,
        "constructed_extrapolation_candidate_count": 0,
        "constructed_extrapolation_trace": None,
        "proposal_source_stats": None,
        "merged_candidate_count": None,
        "experience_prior": None,
        "memory_prior": None,
        "guidance_prior": None,
        "experience_rerank": None,
        "refine_round_expr_counts": None,
        "refine_round_timings": None,
        "step_times_json": None,
        "iteration_config": _base.json.dumps(_base.make_json_safe(iter_cfg), ensure_ascii=False),
        "early_stop_reason": None,
        "time_budget_sec": (
            float(getattr(_base, "MAX_RUNTIME_PER_TASK_SEC", 0))
            if isinstance(getattr(_base, "MAX_RUNTIME_PER_TASK_SEC", None), _base.numbers.Number)
            and float(getattr(_base, "MAX_RUNTIME_PER_TASK_SEC", 0)) > 0
            else None
        ),
        "time_budget_hit": False,
        "agent_trace": None,
        "judge_feedback_history": None,
        "proposal_history": None,
        "evaluation_history": None,
        "candidate_evaluation_history": None,
        "validation_search_threshold": float(V11_NUMERICAL_FIT_R2_THRESHOLD),
        "validation_search_success": False,
        "validation_search_censored": True,
        "evaluations_to_validation_success": None,
        "unique_evaluations_to_validation_success": None,
        "validation_search_observed_evaluations": 0,
        "total_candidate_evaluations": 0,
        "total_unique_candidate_evaluations": 0,
        "first_validation_success_stage": None,
        "first_validation_success_expression": None,
        "first_validation_success_val_r2": None,
        "meta_plan_history": None,
        "error": None,
    }


def _expr_complexity_variants(expr):
    out = []
    text = str(expr or "").strip()
    if not text:
        return out
    out.append(text)
    try:
        sym = sp.sympify(text)
        for fn in [sp.simplify, sp.factor, sp.cancel, sp.together]:
            try:
                out.append(str(fn(sym)))
            except Exception:
                pass
    except Exception:
        pass
    return _uniq(out)


def _family_matches(expr, target_families, feature_names):
    if not target_families:
        return True
    try:
        sig = _base.extract_formula_form_signature(expr, feature_names)
        families = set(sig.get("families", []) or [])
        return bool(families & set(target_families))
    except Exception:
        text = str(expr).lower()
        family_tokens = {
            "rational": ["/"],
            "power": ["**2", "**3", "**4", "**5"],
            "trigonometric": ["sin(", "cos(", "tan("],
            "logarithmic": ["log("],
            "exponential": ["exp("],
            "interaction": ["*"],
        }
        return any(any(tok in text for tok in family_tokens.get(fam, [])) for fam in target_families)


def _active_var_matches(expr, active_variables):
    if not active_variables:
        return True
    text = str(expr)
    return any(re.search(rf"\b{re.escape(str(v))}\b", text) for v in active_variables)


def _build_critic_augmented_candidates(decision, dataset, observation, current_best, row_meta):
    action = str(decision.get("critic_action") or "refine")
    target_families = [str(x) for x in (decision.get("target_families", []) or []) if str(x)]
    active_variables = [str(x) for x in (decision.get("active_variables", []) or []) if str(x)]
    feature_names = list(getattr(dataset, "feature_names", []) or [])
    candidates = []
    trace = {
        "action": action,
        "target_families": target_families,
        "active_variables": active_variables,
        "sources": [],
    }

    current_expr = _base._safe_get_attr(current_best, "simplified_expression", None) if current_best is not None else None
    if action == "complexity_prune":
        variants = _expr_complexity_variants(current_expr)
        candidates.extend(variants)
        trace["sources"].append({"name": "complexity_prune_variants", "count": len(variants)})

    if action in {"structural_rescue", "complexity_prune", "refine"}:
        manual = build_manual_candidates(feature_names)
        filtered = [
            expr for expr in manual
            if _family_matches(expr, target_families, feature_names)
            and _active_var_matches(expr, active_variables)
        ]
        if not filtered:
            filtered = manual[:24]
        candidates.extend(filtered[:48])
        trace["sources"].append({"name": "family_filtered_manual", "count": len(filtered[:48])})

    if V11_ENABLE_STRUCTURAL_RESCUE and action == "structural_rescue":
        if len(feature_names) <= 3 and hasattr(_base, "build_lowdim_rational_coverage_candidates"):
            try:
                lowdim_exprs, lowdim_trace = _base.build_lowdim_rational_coverage_candidates(
                    dataset=dataset,
                    row_meta=row_meta,
                    max_candidates=32,
                )
                candidates.extend(lowdim_exprs)
                trace["sources"].append({
                    "name": "lowdim_rational_coverage",
                    "count": len(lowdim_exprs),
                    "trace": _base.make_json_safe(lowdim_trace),
                })
            except Exception as e:
                trace["sources"].append({"name": "lowdim_rational_coverage", "error": repr(e)})
        elif len(feature_names) >= 4 and hasattr(_base, "build_highdim_universal_coverage_candidates"):
            try:
                highdim_exprs = _base.build_highdim_universal_coverage_candidates(
                    feature_names=feature_names,
                    max_candidates=64,
                )
                candidates.extend(highdim_exprs)
                trace["sources"].append({"name": "highdim_universal_coverage", "count": len(highdim_exprs)})
            except Exception as e:
                trace["sources"].append({"name": "highdim_universal_coverage", "error": repr(e)})

    return _base.deduplicate_expressions(candidates)[:V11_IMAGE_LOOP_MAX_AUGMENTED_CANDIDATES], trace


def _maybe_build_guidance_prior(row_meta, dataset, observation=None):
    try:
        experience = _base.build_experience_prior(
            row_meta=row_meta,
            feature_names=dataset.feature_names,
            observation=observation,
        )
    except Exception:
        experience = {}
    try:
        memory = _base.build_memory_prior(
            row_meta=row_meta,
            feature_names=dataset.feature_names,
            observation=observation,
            experience_prior=experience,
            dataset=dataset,
        )
    except Exception:
        memory = {}
    try:
        guidance = _base.merge_guidance_priors(experience_prior=experience, memory_prior=memory)
    except Exception:
        guidance = dict(experience or {})
    return experience, memory, guidance


def _format_coef(value: float) -> str:
    try:
        return _base._format_float_for_expr(float(value))
    except Exception:
        return f"{float(value):.12g}"


def _poly_term_specs(feature_names: List[str], max_degree: int):
    from itertools import combinations_with_replacement

    specs = []
    for degree in range(1, int(max_degree) + 1):
        for combo in combinations_with_replacement(list(feature_names), degree):
            powers = {}
            for name in combo:
                powers[name] = powers.get(name, 0) + 1
            parts = []
            for name in feature_names:
                power = powers.get(name, 0)
                if power == 1:
                    parts.append(name)
                elif power > 1:
                    parts.append(f"{name}**{power}")
            label = "*".join(parts)
            specs.append((label, dict(powers)))
    return specs


def _basis_matrix(df, feature_names, specs):
    cols = []
    for _, powers in specs:
        values = np.ones(len(df), dtype=float)
        for name, power in powers.items():
            values = values * (df[name].to_numpy(dtype=float) ** int(power))
        cols.append(values)
    if not cols:
        return np.empty((len(df), 0), dtype=float)
    return np.column_stack(cols)


def _ols_fit_expr_for_specs(dataset, specs):
    target = getattr(dataset, "target_name", "y")
    train_df = getattr(dataset, "train_df", None)
    val_df = getattr(dataset, "val_df", None)
    if train_df is None or val_df is None or target not in train_df or target not in val_df:
        return None
    feature_names = list(getattr(dataset, "feature_names", []) or [])
    try:
        x_train = _basis_matrix(train_df, feature_names, specs)
        x_val = _basis_matrix(val_df, feature_names, specs)
        y_train = train_df[target].to_numpy(dtype=float)
        y_val = val_df[target].to_numpy(dtype=float)
        if x_train.size == 0:
            return None
        x_train = np.column_stack([np.ones(len(x_train), dtype=float), x_train])
        x_val = np.column_stack([np.ones(len(x_val), dtype=float), x_val])
        if not np.isfinite(x_train).all() or not np.isfinite(y_train).all():
            return None
        ridge = max(0.0, float(V11_EXTRAPOLATION_OLS_RIDGE))
        xtx = x_train.T @ x_train
        if ridge:
            xtx = xtx + ridge * np.eye(xtx.shape[0])
        coef = np.linalg.solve(xtx, x_train.T @ y_train)
    except np.linalg.LinAlgError:
        try:
            coef, *_ = np.linalg.lstsq(x_train, y_train, rcond=None)
        except Exception:
            return None
    except Exception:
        return None
    try:
        pred_val = x_val @ coef
        if not np.isfinite(pred_val).all():
            return None
        val_mse = float(np.mean((pred_val - y_val) ** 2))
    except Exception:
        return None

    pieces = []
    intercept = float(coef[0])
    if math.isfinite(intercept) and abs(intercept) > 1e-10:
        pieces.append(_format_coef(intercept))
    for c, (label, _) in zip(coef[1:], specs):
        c = float(c)
        if not math.isfinite(c) or abs(c) <= 1e-10:
            continue
        pieces.append(f"({_format_coef(c)})*({label})")
    expr = " + ".join(pieces) if pieces else "0"
    return {"expr": expr, "val_mse": val_mse, "num_terms": len(specs)}


def _build_extrapolation_ols_candidates(dataset, row_meta=None):
    if not V11_ENABLE_EXTRAPOLATION_OLS_CANDIDATES or not _is_extrapolation_source(dataset, row_meta=row_meta):
        return [], {"enabled": False}
    feature_names = list(getattr(dataset, "feature_names", []) or [])
    if not feature_names or len(feature_names) > max(1, int(V11_EXTRAPOLATION_OLS_MAX_FEATURES)):
        return [], {"enabled": False, "reason": "dimension_out_of_range", "n_features": len(feature_names)}

    candidates = []
    # Degree ladders are generic SR primitives; no benchmark names or OOD labels.
    for degree in [1, 2, 3]:
        specs = _poly_term_specs(feature_names, degree)
        fitted = _ols_fit_expr_for_specs(dataset, specs)
        if fitted:
            fitted["family"] = f"poly_degree_{degree}"
            candidates.append(fitted)

    if len(feature_names) == 2:
        x0, x1 = feature_names
        focused_specs = [
            (f"{x0}**2*{x1}", {x0: 2, x1: 1}),
            (f"{x0}**2", {x0: 2}),
            (f"{x0}*{x1}**2", {x0: 1, x1: 2}),
            (f"{x0}*{x1}", {x0: 1, x1: 1}),
            (x0, {x0: 1}),
            (f"{x1}**2", {x1: 2}),
            (x1, {x1: 1}),
        ]
        fitted = _ols_fit_expr_for_specs(dataset, focused_specs)
        if fitted:
            fitted["family"] = "focused_2d_cross_poly"
            candidates.append(fitted)

    unique = {}
    for item in candidates:
        expr = str(item.get("expr") or "").strip()
        if not expr:
            continue
        key = _base._expr_dedup_key(expr) if hasattr(_base, "_expr_dedup_key") else expr
        if key not in unique or float(item.get("val_mse", float("inf"))) < float(unique[key].get("val_mse", float("inf"))):
            unique[key] = item
    ranked = sorted(unique.values(), key=lambda x: (float(x.get("val_mse", float("inf"))), int(x.get("num_terms", 999))))
    limit = max(0, int(V11_EXTRAPOLATION_OLS_MAX_CANDIDATES))
    exprs = [item["expr"] for item in ranked[:limit]]
    trace = {
        "enabled": True,
        "num_candidates": len(exprs),
        "records": _base.make_json_safe([
            {k: item.get(k) for k in ["family", "val_mse", "num_terms", "expr"]}
            for item in ranked[:limit]
        ]),
    }
    return exprs, trace


def _build_surface_analytic_candidates(dataset, row_meta=None):
    if not V11_ENABLE_SURFACE_ANALYTIC_CANDIDATES:
        return [], {"enabled": False}
    source = ""
    if isinstance(row_meta, dict):
        source = str(row_meta.get("dataset_dir", "") or row_meta.get("task_type", "") or "")
    source = source.lower() + " " + str(getattr(dataset, "source_tag", "") or "").lower()
    if "surfacebench" not in source:
        return [], {"enabled": False, "reason": "not_surfacebench"}
    feature_names = list(getattr(dataset, "feature_names", []) or [])
    if len(feature_names) != 2:
        return [], {"enabled": False, "reason": "not_2d", "n_features": len(feature_names)}
    x0, x1 = feature_names[:2]
    eps = "1e-6"
    templates = [
        # Radial / separable Gaussian and smooth exponential surfaces.
        f"a*exp(-b*{x0}**2-c*{x1}**2)+d",
        f"a*exp(-b*({x0}-c0)**2-d*({x1}-e0)**2)+f",
        f"a*exp(-b*({x0}**2+{x1}**2))+c",
        f"a*exp(b*{x0}+c*{x1})+d",
        f"a*exp(b*{x0}+c*{x1})+d*exp(e*{x0}+f*{x1})+g",
        # Additive symbolic/numeric surfaces with periodic and Gaussian parts.
        f"a*sin(b*{x0})+c*exp(-d*{x1}**2)+e",
        f"a*sin(b*{x1})+c*exp(-d*{x0}**2)+e",
        f"a*cos(b*{x0})+c*sin(d*{x1}**2)+e",
        f"a*cos(b*{x1})+c*sin(d*{x0}**2)+e",
        f"a*cos(b*{x0})+c*cos(d*{x1})+e",
        f"a*sin(b*{x0})+c*cos(d*{x1})+e",
        f"a*sin(b*{x0})+c*sin(d*{x1})+e",
        f"a*sin(b*{x0}+c*{x1}+d)+e",
        f"a*cos(b*{x0}+c*{x1}+d)+e",
        # Mild ratios used by symbolic-numeric and hybrid surfaces.
        f"a*sin(b*{x0})+c*sin(d*{x1})/({x1}+e)+f",
        f"a*sin(b*{x1})+c*sin(d*{x0})/({x0}+e)+f",
        f"a*{x0}**2/(b*{x0}**2+c*cos(d*{x0})+1)+e",
        f"a*{x1}**2/(b*{x1}**2+c*cos(d*{x1})+1)+e",
        f"(a*{x0}+b*{x0}**2+c)/(d*{x0}**2+e*exp(f*{x1})+1)+g",
        f"(a*{x1}+b*{x1}**2+c)/(d*{x1}**2+e*exp(f*{x0})+1)+g",
        # Periodic envelopes and modulated waves.
        f"a*exp(b*cos(c*{x0}))*cos(d*{x0}+e*{x1}+f)+g",
        f"a*exp(b*cos(c*{x1}))*cos(d*{x1}+e*{x0}+f)+g",
        f"a*sin(b*{x0})**2*sin(c*{x1})**2/(({x0}+{eps})*({x1}**2+{eps}))+d",
        f"a*sin(b*{x1})**2*sin(c*{x0})**2/(({x1}+{eps})*({x0}**2+{eps}))+d",
        # Compact nonlinear dynamical surfaces.
        f"a*sin(b*{x0})+c*sin(d*{x1})+e*cos(f*{x0})+g*cos(h*{x1})+i",
        f"a*sin(b*{x0})+c*cos(d*{x0}+e*{x1}+f)+g*cos(h*{x1})+i",
        f"a*{x0}*{x1}+b*sin(c*{x0})+d*cos(e*{x1})+f",
        f"a*{x0}**2+b*{x1}**2+c*sin(d*{x0})+e*cos(f*{x1})+g",
    ]
    limit = max(0, int(V11_SURFACE_ANALYTIC_MAX_CANDIDATES))
    exprs = _uniq(templates)[:limit]
    return exprs, {
        "enabled": True,
        "num_candidates": len(exprs),
        "preview": exprs[:8],
    }


def _safe_eval_basis(expr: str, df) -> Optional[np.ndarray]:
    try:
        with np.errstate(all="ignore"):
            values = np.asarray(_base.evaluate_expression_on_df(expr, df), dtype=float).reshape(-1)
        if values.size != len(df) or not np.isfinite(values).all():
            return None
        return values
    except Exception:
        return None


def _fit_linear_basis_candidate(dataset, basis_exprs: List[str], family: str) -> Optional[Dict[str, Any]]:
    target = getattr(dataset, "target_name", "y")
    train_df = getattr(dataset, "train_df", None)
    val_df = getattr(dataset, "val_df", None)
    if train_df is None or val_df is None or target not in train_df or target not in val_df:
        return None
    basis_exprs = _uniq(basis_exprs)
    cols_train, cols_val, kept = [], [], []
    for expr in basis_exprs:
        tr = _safe_eval_basis(expr, train_df)
        va = _safe_eval_basis(expr, val_df)
        if tr is None or va is None:
            continue
        if float(np.nanstd(tr)) <= 1e-12:
            continue
        cols_train.append(tr)
        cols_val.append(va)
        kept.append(expr)
    if not kept:
        return None
    try:
        x_train = np.column_stack([np.ones(len(train_df), dtype=float)] + cols_train)
        x_val = np.column_stack([np.ones(len(val_df), dtype=float)] + cols_val)
        y_train = train_df[target].to_numpy(dtype=float)
        y_val = val_df[target].to_numpy(dtype=float)
        ridge = max(0.0, float(V11_SURFACE_GRID_RIDGE))
        xtx = x_train.T @ x_train
        if ridge:
            xtx = xtx + ridge * np.eye(xtx.shape[0])
        coef = np.linalg.solve(xtx, x_train.T @ y_train)
    except np.linalg.LinAlgError:
        try:
            coef, *_ = np.linalg.lstsq(x_train, y_train, rcond=None)
        except Exception:
            return None
    except Exception:
        return None
    pred_val = x_val @ coef
    if not np.isfinite(pred_val).all():
        return None
    val_mse = float(np.mean((pred_val - y_val) ** 2))
    coef_terms = []
    for idx, (c, expr) in enumerate(zip(coef[1:], kept), start=1):
        c = float(c)
        if not math.isfinite(c):
            continue
        coef_terms.append((abs(c), c, expr, idx))
    coef_terms.sort(reverse=True, key=lambda x: x[0])
    max_terms = max(1, int(V11_SURFACE_GRID_MAX_TERMS))
    selected = sorted(coef_terms[:max_terms], key=lambda x: x[3])
    pieces = []
    intercept = float(coef[0])
    if math.isfinite(intercept) and abs(intercept) > 1e-10:
        pieces.append(_format_coef(intercept))
    for _, c, expr, _ in selected:
        if abs(c) <= 1e-10:
            continue
        pieces.append(f"({_format_coef(c)})*({expr})")
    if not pieces:
        return None
    out_expr = " + ".join(pieces)
    return {
        "family": family,
        "expr": out_expr,
        "val_mse": val_mse,
        "num_basis": len(kept),
        "num_terms": len(pieces),
    }


def _build_surface_grid_basis_candidates(dataset, row_meta=None):
    if not V11_ENABLE_SURFACE_GRID_BASIS_CANDIDATES:
        return [], {"enabled": False}
    source = ""
    if isinstance(row_meta, dict):
        source = str(row_meta.get("dataset_dir", "") or row_meta.get("task_type", "") or "")
    source = source.lower() + " " + str(getattr(dataset, "source_tag", "") or "").lower()
    if "surfacebench" not in source:
        return [], {"enabled": False, "reason": "not_surfacebench"}
    feature_names = list(getattr(dataset, "feature_names", []) or [])
    if len(feature_names) != 2:
        return [], {"enabled": False, "reason": "not_2d", "n_features": len(feature_names)}
    x0, x1 = feature_names[:2]
    eps = "1e-6"
    freqs = [0.5, 0.8, 1.0, 1.5, 2.0, 3.0]
    scales = [0.25, 0.5, 0.8, 1.0, 1.5, 2.0]

    unary = []
    for x in [x0, x1]:
        unary.append(x)
        unary.append(f"{x}**2")
        for k in freqs:
            kg = f"{k:g}"
            unary.extend([
                f"sin({kg}*{x})",
                f"cos({kg}*{x})",
                f"sin({kg}*{x}**2)",
                f"cos({kg}*{x}**2)",
                f"sin({kg}*{x})/({x}+{eps})",
            ])
        for a in scales:
            ag = f"{a:g}"
            unary.append(f"exp(-{ag}*{x}**2)")

    radial = []
    for a in scales:
        ag = f"{a:g}"
        radial.append(f"exp(-{ag}*({x0}**2+{x1}**2))")
        radial.append(f"exp(-{ag}*{x0}**2-{ag}*{x1}**2)")
    for a in [0.5, 0.8, 1.0, 1.5]:
        for b in [0.5, 0.8, 1.0, 1.5]:
            radial.append(f"exp(-{a:g}*{x0}**2-{b:g}*{x1}**2)")

    pair_periodic = []
    for k in [0.8, 1.0, 1.5, 2.0, 3.0]:
        kg = f"{k:g}"
        pair_periodic.extend([
            f"sin({kg}*{x0})*sin({kg}*{x1})",
            f"cos({kg}*{x0})*cos({kg}*{x1})",
            f"sin({kg}*{x0})*cos({kg}*{x1})",
            f"cos({kg}*{x0}+{kg}*{x1})",
            f"sin({kg}*{x0}+{kg}*{x1})",
        ])
    pair_periodic.extend([
        f"{x0}*{x1}",
        f"{x0}**2*{x1}",
        f"{x0}*{x1}**2",
        f"sin({x0})+exp(-{x1}**2)",
        f"sin({x1})+exp(-{x0}**2)",
        f"cos({x0})+sin({x1}**2)",
        f"cos({x1})+sin({x0}**2)",
        f"sin(2*{x0})+sin(2*{x1})/({x1}+{eps})",
        f"sin(2*{x1})+sin(2*{x0})/({x0}+{eps})",
    ])

    groups = [
        ("surface_unary_grid", unary),
        ("surface_radial_grid", radial + [x0, x1]),
        ("surface_periodic_pair_grid", pair_periodic + unary[:20]),
        ("surface_full_grid", unary + radial + pair_periodic),
    ]
    records = []
    for family, basis in groups:
        fitted = _fit_linear_basis_candidate(dataset, basis, family)
        if fitted:
            records.append(fitted)
    records.sort(key=lambda x: (float(x.get("val_mse", float("inf"))), int(x.get("num_terms", 999))))
    limit = max(0, int(V11_SURFACE_GRID_MAX_CANDIDATES))
    records = records[:limit]
    exprs = [r["expr"] for r in records]
    return exprs, {
        "enabled": True,
        "num_candidates": len(exprs),
        "records": _base.make_json_safe(records),
    }


def _surface_source_enabled(dataset=None, row_meta=None) -> bool:
    source = ""
    if isinstance(row_meta, dict):
        source = str(row_meta.get("dataset_dir", "") or row_meta.get("task_type", "") or "")
    source = source.lower() + " " + str(getattr(dataset, "source_tag", "") or "").lower()
    return "surfacebench" in source


def _surface_stable_ridges() -> List[float]:
    out = []
    for part in str(V11_SURFACE_STABLE_RIDGES or "").split(","):
        try:
            value = float(part.strip())
        except Exception:
            continue
        if math.isfinite(value) and value >= 0:
            out.append(value)
    return out or [0.1, 1.0, 10.0]


def _fit_standardized_ridge_basis_candidate(
    dataset,
    basis_exprs: List[str],
    family: str,
    ridge: float,
    max_terms: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    target = getattr(dataset, "target_name", "y")
    train_df = getattr(dataset, "train_df", None)
    val_df = getattr(dataset, "val_df", None)
    if train_df is None or val_df is None or target not in train_df or target not in val_df:
        return None
    cols_train, cols_val, kept = [], [], []
    for expr in _uniq(basis_exprs):
        tr = _safe_eval_basis(expr, train_df)
        va = _safe_eval_basis(expr, val_df)
        if tr is None or va is None:
            continue
        if float(np.nanstd(tr)) <= 1e-10:
            continue
        cols_train.append(tr)
        cols_val.append(va)
        kept.append(expr)
    if not kept:
        return None
    try:
        x_train = np.column_stack(cols_train)
        x_val = np.column_stack(cols_val)
        y_train = train_df[target].to_numpy(dtype=float)
        y_val = val_df[target].to_numpy(dtype=float)
        mu = np.nanmean(x_train, axis=0)
        sd = np.nanstd(x_train, axis=0)
        sd = np.where(np.isfinite(sd) & (sd > 1e-10), sd, 1.0)
        x_train_s = (x_train - mu) / sd
        x_val_s = (x_val - mu) / sd
        if not np.isfinite(x_train_s).all() or not np.isfinite(x_val_s).all():
            return None
        design = np.column_stack([np.ones(len(train_df), dtype=float), x_train_s])
        design_val = np.column_stack([np.ones(len(val_df), dtype=float), x_val_s])
        reg = max(0.0, float(ridge))
        penalty = np.eye(design.shape[1], dtype=float)
        penalty[0, 0] = 0.0
        try:
            coef_s = np.linalg.solve(design.T @ design + reg * penalty, design.T @ y_train)
        except np.linalg.LinAlgError:
            coef_s, *_ = np.linalg.lstsq(design, y_train, rcond=None)
        raw_coef = coef_s[1:] / sd
        raw_intercept = float(coef_s[0] - np.sum(coef_s[1:] * mu / sd))
        ranked_terms = []
        for idx, (coef, expr) in enumerate(zip(raw_coef, kept)):
            coef = float(coef)
            if math.isfinite(coef) and abs(coef) > 1e-10:
                ranked_terms.append((abs(coef), coef, expr, idx))
        ranked_terms.sort(reverse=True, key=lambda item: item[0])
        max_terms = max(1, int(max_terms or V11_SURFACE_STABLE_MAX_TERMS))
        selected = sorted(ranked_terms[:max_terms], key=lambda item: item[3])

        # Refit the compact selected basis in original coordinates. This keeps
        # the emitted formula short while the first ridge pass chooses stable
        # terms from a larger dictionary.
        if selected:
            selected_train = [cols_train[item[3]] for item in selected]
            selected_val = [cols_val[item[3]] for item in selected]
            compact = np.column_stack([np.ones(len(train_df), dtype=float)] + selected_train)
            compact_val = np.column_stack([np.ones(len(val_df), dtype=float)] + selected_val)
            penalty2 = np.eye(compact.shape[1], dtype=float)
            penalty2[0, 0] = 0.0
            try:
                coef = np.linalg.solve(compact.T @ compact + reg * penalty2, compact.T @ y_train)
            except np.linalg.LinAlgError:
                coef, *_ = np.linalg.lstsq(compact, y_train, rcond=None)
            pred_val = compact_val @ coef
            pieces = []
            intercept = float(coef[0])
            if math.isfinite(intercept) and abs(intercept) > 1e-10:
                pieces.append(_format_coef(intercept))
            for c, (_, _, expr, _) in zip(coef[1:], selected):
                c = float(c)
                if math.isfinite(c) and abs(c) > 1e-10:
                    pieces.append(f"({_format_coef(c)})*({expr})")
        else:
            pred_val = design_val @ coef_s
            pieces = [_format_coef(raw_intercept)] if math.isfinite(raw_intercept) else []
        if not np.isfinite(pred_val).all():
            return None
        expr = " + ".join(pieces) if pieces else "0"
        val_mse = float(np.mean((pred_val - y_val) ** 2))
        return {
            "family": family,
            "ridge": float(reg),
            "expr": expr,
            "val_mse": val_mse,
            "num_basis": len(kept),
            "num_terms": len(pieces),
        }
    except Exception:
        return None


def _build_surface_stable_basis_candidates(dataset, row_meta=None):
    if not V11_ENABLE_SURFACE_STABLE_BASIS_CANDIDATES:
        return [], {"enabled": False}
    if not _surface_source_enabled(dataset=dataset, row_meta=row_meta):
        return [], {"enabled": False, "reason": "not_surfacebench"}
    feature_names = list(getattr(dataset, "feature_names", []) or [])
    if len(feature_names) != 2:
        return [], {"enabled": False, "reason": "not_2d", "n_features": len(feature_names)}
    x0, x1 = feature_names[:2]
    freqs = [0.25, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
    scales = [0.1, 0.25, 0.5, 0.8, 1.0, 1.5]
    poly = [
        x0, x1, f"{x0}*{x1}", f"{x0}**2", f"{x1}**2",
        f"{x0}**2*{x1}", f"{x0}*{x1}**2", f"{x0}**3", f"{x1}**3",
    ]
    trig = []
    for x in [x0, x1]:
        for k in freqs:
            kg = f"{k:g}"
            trig.extend([f"sin({kg}*{x})", f"cos({kg}*{x})", f"sin({kg}*{x}**2)", f"cos({kg}*{x}**2)"])
    pair = []
    for k in freqs:
        kg = f"{k:g}"
        pair.extend([
            f"sin({kg}*{x0})*sin({kg}*{x1})",
            f"cos({kg}*{x0})*cos({kg}*{x1})",
            f"sin({kg}*{x0}+{kg}*{x1})",
            f"cos({kg}*{x0}+{kg}*{x1})",
            f"sin({kg}*{x0}*{x1})",
            f"cos({kg}*{x0}*{x1})",
        ])
    radial = []
    for a in scales:
        ag = f"{a:g}"
        radial.extend([f"exp(-{ag}*{x0}**2)", f"exp(-{ag}*{x1}**2)", f"exp(-{ag}*({x0}**2+{x1}**2))"])
    ratio = [
        f"{x0}/(1+{x1}**2)",
        f"{x1}/(1+{x0}**2)",
        f"{x0}**2/(1+{x1}**2)",
        f"{x1}**2/(1+{x0}**2)",
        f"sin({x0}*{x1})/(1+{x1}**2)",
        f"sin({x0}*{x1})/(1+{x0}**2)",
    ]
    groups = [
        ("surface_stable_poly", poly),
        ("surface_stable_trig", trig + poly[:5]),
        ("surface_stable_pair", pair + poly[:5]),
        ("surface_stable_radial", radial + poly[:5]),
        ("surface_stable_full", poly + trig + pair + radial + ratio),
    ]
    records = []
    for family, basis in groups:
        for ridge in _surface_stable_ridges():
            fitted = _fit_standardized_ridge_basis_candidate(
                dataset,
                basis,
                family,
                ridge=ridge,
                max_terms=V11_SURFACE_STABLE_MAX_TERMS,
            )
            if fitted:
                records.append(fitted)
    category = str((row_meta or {}).get("category", "") if isinstance(row_meta, dict) else "").lower()
    val_cutoff = float("inf")
    if "hybrid_multi-modal" in category:
        val_cutoff = float(V11_SURFACE_STABLE_HYBRID_VAL_CUTOFF)
    elif "symbolic-numeric" in category:
        val_cutoff = float(V11_SURFACE_STABLE_SYMBOLIC_NUMERIC_VAL_CUTOFF)
    if math.isfinite(val_cutoff):
        records = [
            item for item in records
            if float(item.get("val_mse", float("inf"))) <= val_cutoff
        ]
    unique = {}
    for item in records:
        expr = str(item.get("expr") or "").strip()
        if not expr:
            continue
        key = _base._expr_dedup_key(expr) if hasattr(_base, "_expr_dedup_key") else expr
        val = float(item.get("val_mse", float("inf")))
        if key not in unique or val < float(unique[key].get("val_mse", float("inf"))):
            unique[key] = item
    ranked = sorted(unique.values(), key=lambda x: (float(x.get("val_mse", float("inf"))), int(x.get("num_terms", 999))))
    limit = max(0, int(V11_SURFACE_STABLE_MAX_CANDIDATES))
    ranked = ranked[:limit]
    exprs = [item["expr"] for item in ranked]
    return exprs, {
        "enabled": True,
        "num_candidates": len(exprs),
        "records": _base.make_json_safe(ranked),
    }


def _is_constructed_extrapolation_source(dataset=None, row_meta=None) -> bool:
    source = ""
    if isinstance(row_meta, dict):
        source = " ".join(str(row_meta.get(k, "") or "") for k in ("dataset_dir", "task_type", "suite"))
    source = (source + " " + str(getattr(dataset, "source_tag", "") or "")).lower()
    return "constructed" in source and "extrapolation" in source


def _linear_expr_from_terms(coefs, exprs) -> str:
    pieces = []
    for coef, expr in zip(coefs, exprs):
        coef = float(coef)
        if not math.isfinite(coef) or abs(coef) <= 1e-10:
            continue
        expr = str(expr)
        if expr == "1":
            pieces.append(_format_coef(coef))
        else:
            pieces.append(f"({_format_coef(coef)})*({expr})")
    return " + ".join(pieces) if pieces else "0"


def _fit_linearized_rational_candidate(dataset, numerator_basis: List[str], denominator_basis: List[str], family: str) -> Optional[Dict[str, Any]]:
    target = getattr(dataset, "target_name", "y")
    train_df = getattr(dataset, "train_df", None)
    val_df = getattr(dataset, "val_df", None)
    if train_df is None or val_df is None or target not in train_df or target not in val_df:
        return None
    try:
        y_train = train_df[target].to_numpy(dtype=float)
        y_val = val_df[target].to_numpy(dtype=float)
        num_train, num_val, kept_num = [], [], []
        for expr in numerator_basis:
            if expr == "1":
                tr = np.ones(len(train_df), dtype=float)
                va = np.ones(len(val_df), dtype=float)
            else:
                tr = _safe_eval_basis(expr, train_df)
                va = _safe_eval_basis(expr, val_df)
            if tr is None or va is None:
                continue
            num_train.append(tr)
            num_val.append(va)
            kept_num.append(expr)
        den_train, den_val, kept_den = [], [], []
        for expr in denominator_basis:
            tr = _safe_eval_basis(expr, train_df)
            va = _safe_eval_basis(expr, val_df)
            if tr is None or va is None:
                continue
            den_train.append(tr)
            den_val.append(va)
            kept_den.append(expr)
        if not kept_num or not kept_den:
            return None
        a_train = np.column_stack(num_train + [-(y_train * col) for col in den_train])
        rhs = y_train
        if not np.isfinite(a_train).all() or not np.isfinite(rhs).all():
            return None
        ridge = max(0.0, float(V11_SURFACE_GRID_RIDGE))
        xtx = a_train.T @ a_train
        if ridge:
            xtx = xtx + ridge * np.eye(xtx.shape[0])
        try:
            coef = np.linalg.solve(xtx, a_train.T @ rhs)
        except np.linalg.LinAlgError:
            coef, *_ = np.linalg.lstsq(a_train, rhs, rcond=None)
        num_coef = coef[: len(kept_num)]
        den_coef = coef[len(kept_num) :]
        num_val_arr = np.column_stack(num_val) @ num_coef
        den_val_arr = 1.0 + np.column_stack(den_val) @ den_coef
        pred_val = num_val_arr / den_val_arr
        if not np.isfinite(pred_val).all():
            return None
        val_mse = float(np.mean((pred_val - y_val) ** 2))
        num_expr = _linear_expr_from_terms(num_coef, kept_num)
        den_terms = [f"({_format_coef(float(c))})*({expr})" for c, expr in zip(den_coef, kept_den) if math.isfinite(float(c)) and abs(float(c)) > 1e-10]
        den_expr = "1" if not den_terms else "1 + " + " + ".join(den_terms)
        return {
            "family": family,
            "expr": f"({num_expr})/({den_expr})",
            "val_mse": val_mse,
            "num_terms": len(kept_num) + len(kept_den),
        }
    except Exception:
        return None


def _build_constructed_extrapolation_candidates(dataset, row_meta=None):
    if not V11_ENABLE_CONSTRUCTED_EXTRAPOLATION_CANDIDATES:
        return [], {"enabled": False}
    if not _is_constructed_extrapolation_source(dataset, row_meta=row_meta):
        return [], {"enabled": False, "reason": "not_constructed_extrapolation"}
    feature_names = list(getattr(dataset, "feature_names", []) or [])
    if not feature_names or len(feature_names) > 2:
        return [], {"enabled": False, "reason": "dimension_out_of_range", "n_features": len(feature_names)}

    records = []
    if len(feature_names) == 1:
        x = feature_names[0]
        abs_x = f"sqrt({x}**2)"
        for degree in [4, 5]:
            fitted = _fit_linear_basis_candidate(dataset, [f"{x}**{p}" if p > 1 else x for p in range(1, degree + 1)], f"constructed_poly_degree_{degree}")
            if fitted:
                records.append(fitted)
        for offset in [1.0, 1.2, 1.5, 1.8, 2.0, 2.2, 3.0]:
            og = f"{offset:g}"
            for basis, family in [
                ([f"log({abs_x}+{og})", x], "constructed_log_linear"),
                ([f"sqrt({abs_x}+{og})", x], "constructed_sqrt_linear"),
                ([f"1/sqrt({abs_x}+{og})", x], "constructed_inverse_sqrt"),
            ]:
                fitted = _fit_linear_basis_candidate(dataset, basis, family)
                if fitted:
                    records.append(fitted)
        for k in [0.08, 0.12, 0.15, 0.25, 0.3]:
            fitted = _fit_linear_basis_candidate(dataset, [f"exp({k:g}*{x})", x], "constructed_exp_linear")
            if fitted:
                records.append(fitted)
        for k in [0.8, 1.0, 1.1, 1.2, 1.5]:
            for basis, family in [
                ([f"sin({k:g}*{x})", f"{x}**2"], "constructed_sin_trend"),
                ([f"tanh({k:g}*{x})", f"{x}**2"], "constructed_tanh_like"),
            ]:
                fitted = _fit_linear_basis_candidate(dataset, basis, family)
                if fitted:
                    records.append(fitted)
        for k in [0.8, 1.0]:
            for a in [0.04, 0.08, 0.12]:
                fitted = _fit_linear_basis_candidate(dataset, [f"cos({k:g}*{x})*exp({a:g}*{x})"], "constructed_cos_exp_envelope")
                if fitted:
                    records.append(fitted)
        rat = _fit_linearized_rational_candidate(dataset, [x, "1"], [f"{x}**2"], "constructed_1d_rational")
        if rat:
            records.append(rat)

    if len(feature_names) == 2:
        x0, x1 = feature_names[:2]
        basis_groups = [
            ("constructed_interaction_poly", [x0, x1, f"{x0}*{x1}", f"{x0}**2", f"{x1}**2"]),
            ("constructed_exp_interaction", [f"exp(0.12*{x0})", f"{x0}*{x1}"]),
            ("constructed_log_product", [f"log(sqrt(({x0}*{x1})**2)+1.5)", x0]),
            ("constructed_coupled_quadratic", [f"{x0}**2", f"{x1}**2", f"{x0}*{x1}"]),
        ]
        for family, basis in basis_groups:
            fitted = _fit_linear_basis_candidate(dataset, basis, family)
            if fitted:
                records.append(fitted)
        for k0 in [0.8, 1.0, 1.2]:
            for k1 in [0.8, 1.0, 1.2]:
                fitted = _fit_linear_basis_candidate(dataset, [f"sin({k0:g}*{x0})", f"cos({k1:g}*{x1})"], "constructed_sin_separable")
                if fitted:
                    records.append(fitted)
        for k in [0.8, 1.0, 1.2]:
            for c in [0.5, 1.0, 1.5]:
                fitted = _fit_linear_basis_candidate(dataset, [f"sin({k:g}*{x0}*{x1})/(1+{c:g}*{x1}**2)", x0], "constructed_nested_trig_ratio")
                if fitted:
                    records.append(fitted)
        for numerator, denominator, family in [
            ([x0, "1"], [f"{x1}**2"], "constructed_rational_surface_x0_over_x1"),
            ([x1, "1"], [f"{x0}**2"], "constructed_rational_surface_x1_over_x0"),
            (["1", f"{x0}*{x1}"], [f"{x0}**2", f"{x1}**2"], "constructed_reciprocal_interaction"),
        ]:
            fitted = _fit_linearized_rational_candidate(dataset, numerator, denominator, family)
            if fitted:
                records.append(fitted)

    unique = {}
    for item in records:
        expr = str(item.get("expr") or "").strip()
        if not expr:
            continue
        key = _base._expr_dedup_key(expr) if hasattr(_base, "_expr_dedup_key") else expr
        val = float(item.get("val_mse", float("inf")))
        if key not in unique or val < float(unique[key].get("val_mse", float("inf"))):
            unique[key] = item
    ranked = sorted(unique.values(), key=lambda x: (float(x.get("val_mse", float("inf"))), int(x.get("num_terms", 999))))
    limit = max(0, int(V11_CONSTRUCTED_EXTRAPOLATION_MAX_CANDIDATES))
    ranked = ranked[:limit]
    exprs = [item["expr"] for item in ranked]
    return exprs, {
        "enabled": True,
        "num_candidates": len(exprs),
        "records": _base.make_json_safe(ranked),
    }


def _finalize_image_loop(
    result,
    timer,
    start,
    current_best,
    dataset,
    row_meta,
    observation,
    active_eval,
    meta_decisions,
    refine_history,
    refine_round_expr_counts,
    refine_round_timings,
    judge_feedback_history,
    proposal_history,
    evaluation_history,
    agent_trace,
    mm_requested,
    mm_trigger_reason,
    mm_prop_trace,
    mm_candidate_count,
    mm_used_in_evaluation,
):
    _base.populate_visual_trace_fields(
        result=result,
        observation=observation,
        mm_requested=mm_requested,
        mm_trigger_reason=mm_trigger_reason,
        mm_prop_trace=mm_prop_trace,
        mm_candidate_count=mm_candidate_count,
        mm_used_in_evaluation=mm_used_in_evaluation,
    )
    result["refine_round_timings"] = _base.json.dumps(_base.make_json_safe(refine_round_timings), ensure_ascii=False)
    result["judge_feedback_history"] = _base.json.dumps(_base.make_json_safe(judge_feedback_history), ensure_ascii=False)
    result["proposal_history"] = _base.json.dumps(_base.make_json_safe(proposal_history), ensure_ascii=False)
    result["evaluation_history"] = _base.json.dumps(_base.make_json_safe(evaluation_history), ensure_ascii=False)
    search_state = _base.candidate_search_audit_state(dataset)
    candidate_history = list(search_state.get("records", []) or [])
    result["candidate_evaluation_history"] = _base.json.dumps(
        _base.make_json_safe(candidate_history),
        ensure_ascii=False,
    )
    expression_evaluations = len(candidate_history)
    unique_expression_evaluations = len(search_state.get("seen_candidate_keys", {}) or {})
    first_success = next(
        (
            item
            for item in candidate_history
            if _finite_float(item.get("val_r2")) is not None
            and float(item["val_r2"]) > float(V11_NUMERICAL_FIT_R2_THRESHOLD)
        ),
        None,
    )
    elapsed_for_rate = max(0.0, float(_base.time.time() - start))
    result["expression_evaluations"] = expression_evaluations
    result["total_candidate_evaluations"] = expression_evaluations
    result["total_unique_candidate_evaluations"] = unique_expression_evaluations
    result["validation_search_threshold"] = float(V11_NUMERICAL_FIT_R2_THRESHOLD)
    result["validation_search_success"] = bool(first_success is not None)
    result["validation_search_censored"] = bool(first_success is None)
    result["evaluations_to_validation_success"] = (
        int(first_success["evaluation_index"]) if first_success is not None else None
    )
    result["unique_evaluations_to_validation_success"] = (
        int(first_success["unique_evaluations_seen"]) if first_success is not None else None
    )
    result["validation_search_observed_evaluations"] = (
        int(first_success["evaluation_index"])
        if first_success is not None
        else int(expression_evaluations)
    )
    result["first_validation_success_stage"] = (
        first_success.get("stage") if first_success is not None else None
    )
    result["first_validation_success_expression"] = (
        first_success.get("fitted_expression") or first_success.get("expression")
        if first_success is not None else None
    )
    result["first_validation_success_val_r2"] = (
        first_success.get("val_r2") if first_success is not None else None
    )
    result["evaluations_per_sec"] = (
        float(expression_evaluations / elapsed_for_rate)
        if elapsed_for_rate > 0 else None
    )
    result["meta_plan_history"] = _base.json.dumps(_base.make_json_safe(meta_decisions), ensure_ascii=False)
    result["agent_trace"] = _base.json.dumps(_base.make_json_safe(agent_trace), ensure_ascii=False)
    result["best_expr_source"] = (
        str(_base._safe_get_attr(current_best, "source", "scored_result"))
        if current_best is not None else None
    )
    result["selection_reason"] = (
        _base._safe_get_attr(current_best, "selection_reason", None)
        if current_best is not None else None
    )
    result["selected_val_r2"] = (
        _base._safe_get_attr(current_best, "val_r2", None)
        if current_best is not None else None
    )
    result["selected_train_r2"] = (
        _base._safe_get_attr(current_best, "train_r2", None)
        if current_best is not None else None
    )
    result["selected_on_validation_pareto_front"] = bool(
        _base._safe_get_attr(current_best, "on_validation_pareto_front", False)
        if current_best is not None else False
    )
    result["no_leakage_audit"] = _base.json.dumps(_base.make_json_safe({
        "method_mode": getattr(_base, "METHOD_MODE", None),
        "loop_style": "image_loop_observer_proposer_evaluator_critic",
        "observer_enabled": bool(V11_ENABLE_OBSERVER),
        "proposer_enabled": bool(V11_ENABLE_PROPOSER),
        "critic_controls_generation": bool(V11_ENABLE_CRITIC_LOOP),
        "structural_rescue_enabled": bool(V11_ENABLE_STRUCTURAL_RESCUE),
        "adaptive_multimodal_triggered": bool(mm_requested),
        "test_split_used_for_selection": bool(getattr(_base, "USE_TEST_FOR_SELECTION", False)),
        "selection_uses_validation_only": not bool(getattr(_base, "USE_TEST_FOR_SELECTION", False)),
        "pareto_selection_enabled": bool(V11_ENABLE_PARETO_SELECTION),
        "pareto_candidate_analysis_enabled": True,
        "pareto_objectives": ["validation_nmse", "expression_tree_nodes"],
        "validation_r2_threshold": float(V11_NUMERICAL_FIT_R2_THRESHOLD),
        "train_r2_early_stop_enabled": bool(V11_ENABLE_TRAIN_R2_EARLY_STOP),
        "train_r2_early_stop_threshold": float(V11_EARLY_STOP_TRAIN_R2_THRESHOLD),
    }), ensure_ascii=False)
    return _base._finalize_result(
        result=result,
        timer=timer,
        start=start,
        current_best=current_best,
        dataset=dataset,
        residual_summary=active_eval.get("residual_summary") if isinstance(active_eval, dict) else None,
        physics_summary=active_eval.get("physics_summary") if isinstance(active_eval, dict) else None,
        meta_decisions=meta_decisions,
        refine_history=refine_history,
        refine_round_expr_counts=refine_round_expr_counts,
        row_meta=row_meta,
    )


def _run_core_pipeline(dataset, row_meta):
    _sync_runtime_globals()
    _base.reset_candidate_search_audit(dataset)
    start = _base.time.time()
    timer = _base.StepTimer()
    task_deadline_ts = _base._task_deadline_from_start(start)
    budget_guard = _base.task_time_budget_guard(getattr(_base, "MAX_RUNTIME_PER_TASK_SEC", None))

    proposal_client = _base._build_role_client("proposal", _base.BACKEND_CONFIG, deadline_ts=task_deadline_ts)
    observer_client = None
    if V11_ENABLE_VLM_OBSERVER:
        observer_client = _base._build_role_client("observer", _base.BACKEND_CONFIG, deadline_ts=task_deadline_ts)
    critic_client = _base._build_role_client("critic", _base.BACKEND_CONFIG, deadline_ts=task_deadline_ts)
    judge_client = _base._build_role_client("judge", _base.BACKEND_CONFIG, deadline_ts=task_deadline_ts)
    refiner_client = _base._build_role_client("refiner", _base.BACKEND_CONFIG, deadline_ts=task_deadline_ts)

    proposal_llm = _base.ProposalGeneratorLLM(proposal_client)
    refiner_llm = _base.ExpressionRefinerLLM(refiner_client)

    base_observer_agent = _base.ObserverAgent(
        plot_tool=_base.PlotGeneratorTool(),
        high_dim_reconstruction_tool=_base.HighDimReconstructionTool(
            unary_bins=_base.HIGH_DIM_RECON_UNARY_BINS,
            pair_bins=_base.HIGH_DIM_RECON_PAIR_BINS,
            max_unary_views=_base.HIGH_DIM_RECON_MAX_UNARY_VIEWS,
            max_pair_views=_base.HIGH_DIM_RECON_MAX_PAIR_VIEWS,
            rank=_base.HIGH_DIM_RECON_RANK,
            min_bin_count=_base.HIGH_DIM_RECON_MIN_BIN_COUNT,
        ),
    )
    observer_agent = V11ObserverAgent(
        inner_observer=base_observer_agent,
        client=observer_client,
        enabled=V11_ENABLE_VLM_OBSERVER,
    )
    proposer_agent = _base.ProposerAgent(proposal_llm=proposal_llm)
    evaluator_agent = _base.EvaluatorAgent(
        complexity_weight=_base.COMPLEXITY_WEIGHT,
        deadline_ts=task_deadline_ts,
    )
    critic_agent = CriticAgent(client=critic_client, meta_llm=_base.MetaLLM(critic_client))
    judge_agent = _base.JudgeAgent(client=judge_client)
    refiner_agent = _base.RefinerAgent(refiner_llm=refiner_llm, client=refiner_client)

    if V11_ENABLE_OBSERVER:
        experience_prior, memory_prior, guidance_prior = _maybe_build_guidance_prior(row_meta, dataset, observation=None)
    else:
        experience_prior, memory_prior, guidance_prior = {}, {}, {}
    iter_cfg = _base.get_iteration_config(len(dataset.feature_names))
    try:
        iter_cfg = _base.adjust_iteration_config_for_runtime(
            iter_cfg,
            dataset=dataset,
            row_meta=row_meta,
            experience_prior=guidance_prior,
        )
    except Exception:
        iter_cfg = dict(iter_cfg)
    iter_cfg = _maybe_expand_full_budget(iter_cfg, dataset, row_meta)
    iter_cfg = _maybe_expand_low_dim_budget(iter_cfg, dataset, row_meta)
    iter_cfg["loop_style"] = "image_loop"

    result = _make_loop_result(row_meta, dataset, iter_cfg)
    meta_decisions = []
    refine_history = []
    refine_round_expr_counts = []
    refine_round_timings = []
    judge_feedback_history = []
    proposal_history = []
    evaluation_history = []
    agent_trace = []
    current_best = None
    active_eval = {"residual_summary": None, "physics_summary": None}
    observation = _base.ObservationBundle(
        structure_hints=[],
        visual_hints=[],
        unit_hints={},
        plot_descriptions=[],
        image_paths=[],
    )
    mm_requested = False
    mm_trigger_reason = None
    mm_prop_trace = None
    mm_candidate_count = 0
    mm_used_in_evaluation = False

    def record_round_timing(round_num, start_ts, status, extra=None):
        end_ts = _base.time.time()
        record = {
            "round": int(round_num),
            "status": status,
            "start_ts": float(start_ts),
            "end_ts": float(end_ts),
            "start_local": _base.format_local_timestamp(start_ts),
            "end_local": _base.format_local_timestamp(end_ts),
            "total_sec": float(end_ts - start_ts),
        }
        if extra:
            record.update(_base.make_json_safe(extra))
        refine_round_timings.append(record)

    try:
        budget_guard.__enter__()

        if not V11_ENABLE_PROPOSER:
            result["experience_prior"] = _base.json.dumps({}, ensure_ascii=False)
            result["memory_prior"] = _base.json.dumps({}, ensure_ascii=False)
            result["guidance_prior"] = _base.json.dumps({}, ensure_ascii=False)
            result["vlm_observer_trace"] = _base.json.dumps({}, ensure_ascii=False)
            result["vlm_observer_used"] = False
            result["vlm_observer_error"] = None
            result["raw_exprs"] = _base.json.dumps([], ensure_ascii=False)
            result["num_candidate_exprs"] = 0
            result["early_stop_reason"] = "ablation_disable_proposer"
            result["text_proposal_stats"] = _base.json.dumps({"disabled_by_ablation": True}, ensure_ascii=False)
            result["mm_proposal_stats"] = _base.json.dumps({"disabled_by_ablation": True}, ensure_ascii=False)
            proposal_history.append({
                "stage": "initial_proposer",
                "num_exprs": 0,
                "trace": {
                    "disabled_by_ablation": True,
                    "reason": "LLMSR_V11_ENABLE_PROPOSER=0",
                },
            })
            agent_trace.append({
                "event": "proposer_disabled",
                "reason": "LLMSR_V11_ENABLE_PROPOSER=0",
                "observer_skipped": True,
            })
            return _finalize_image_loop(
                result=result,
                timer=timer,
                start=start,
                current_best=None,
                dataset=dataset,
                row_meta=row_meta,
                observation=observation,
                active_eval=active_eval,
                meta_decisions=meta_decisions,
                refine_history=refine_history,
                refine_round_expr_counts=refine_round_expr_counts,
                refine_round_timings=refine_round_timings,
                judge_feedback_history=judge_feedback_history,
                proposal_history=proposal_history,
                evaluation_history=evaluation_history,
                agent_trace=agent_trace,
                mm_requested=mm_requested,
                mm_trigger_reason=mm_trigger_reason,
                mm_prop_trace=mm_prop_trace,
                mm_candidate_count=mm_candidate_count,
                mm_used_in_evaluation=mm_used_in_evaluation,
            )

        # 1. Dataset/Benchmark -> Observer Agent
        if V11_ENABLE_OBSERVER:
            observation = _base.timed_call(
                timer,
                "loop_observer",
                observer_agent.observe,
                dataset,
                row_meta=row_meta,
                timer=timer,
            )
        else:
            observation = _base.ObservationBundle(
                structure_hints=[],
                visual_hints=[],
                unit_hints={},
                plot_descriptions=[],
                image_paths=[],
            )
        vlm_observer_trace = dict(((getattr(observation, "structure_profile", {}) or {}).get("vlm_observer", {}) or {}))
        if V11_ENABLE_OBSERVER:
            experience_prior, memory_prior, guidance_prior = _maybe_build_guidance_prior(row_meta, dataset, observation=observation)
        else:
            experience_prior, memory_prior, guidance_prior = {}, {}, {}
        if V11_ENABLE_OBSERVER and hasattr(evaluator_agent, "set_structure_context"):
            evaluator_agent.set_structure_context(observation=observation, row_meta=row_meta)
        result["experience_prior"] = _base.json.dumps(_base.make_json_safe(experience_prior), ensure_ascii=False)
        result["memory_prior"] = _base.json.dumps(_base.make_json_safe(memory_prior), ensure_ascii=False)
        result["guidance_prior"] = _base.json.dumps(_base.make_json_safe(guidance_prior), ensure_ascii=False)
        result["vlm_observer_trace"] = _base.json.dumps(_base.make_json_safe(vlm_observer_trace), ensure_ascii=False)
        result["vlm_observer_used"] = bool(vlm_observer_trace.get("used_llm", False))
        result["vlm_observer_error"] = vlm_observer_trace.get("error")
        agent_trace.append({
            "event": "observer_done",
            "observer_enabled": bool(V11_ENABLE_OBSERVER),
            "n_features": len(dataset.feature_names),
            "vlm_observer_used": bool(vlm_observer_trace.get("used_llm", False)),
            "vlm_observer_families": vlm_observer_trace.get("candidate_families"),
            "vlm_observer_active_variables": vlm_observer_trace.get("active_variables"),
            "vlm_observer_error": vlm_observer_trace.get("error"),
        })
        _base._raise_if_task_budget_exceeded(task_deadline_ts, "observer")

        # 2. Proposer Agent initial generation
        if not V11_ENABLE_PROPOSER:
            result["raw_exprs"] = _base.json.dumps([], ensure_ascii=False)
            result["num_candidate_exprs"] = 0
            result["early_stop_reason"] = "ablation_disable_proposer"
            result["text_proposal_stats"] = _base.json.dumps({"disabled_by_ablation": True}, ensure_ascii=False)
            result["mm_proposal_stats"] = _base.json.dumps({"disabled_by_ablation": True}, ensure_ascii=False)
            proposal_history.append({
                "stage": "initial_proposer",
                "num_exprs": 0,
                "trace": {
                    "disabled_by_ablation": True,
                    "reason": "LLMSR_V11_ENABLE_PROPOSER=0",
                },
            })
            agent_trace.append({
                "event": "proposer_disabled",
                "reason": "LLMSR_V11_ENABLE_PROPOSER=0",
            })
            return _finalize_image_loop(
                result=result,
                timer=timer,
                start=start,
                current_best=None,
                dataset=dataset,
                row_meta=row_meta,
                observation=observation,
                active_eval=active_eval,
                meta_decisions=meta_decisions,
                refine_history=refine_history,
                refine_round_expr_counts=refine_round_expr_counts,
                refine_round_timings=refine_round_timings,
                judge_feedback_history=judge_feedback_history,
                proposal_history=proposal_history,
                evaluation_history=evaluation_history,
                agent_trace=agent_trace,
                mm_requested=mm_requested,
                mm_trigger_reason=mm_trigger_reason,
                mm_prop_trace=mm_prop_trace,
                mm_candidate_count=mm_candidate_count,
                mm_used_in_evaluation=mm_used_in_evaluation,
            )

        initial_iter_cfg = dict(iter_cfg)
        native_multimodal_arm = V11_OBSERVER_INPUT_MODE in {
            "native", "aligned", "aligned_control", "multimodal"
        }
        numeric_only_arm = V11_OBSERVER_INPUT_MODE in {
            "numeric", "numeric_only", "no_image", "no_images"
        }
        if V11_FORCE_INITIAL_MODALITY_PROPOSAL:
            # The matched ablation spends one model call on the treatment
            # modality only: image-conditioned proposal for the native arm,
            # numeric/text proposal for the control arm. Deterministic shared
            # candidate sources remain enabled in both arms.
            initial_iter_cfg["text_calls"] = (
                0 if native_multimodal_arm else V11_MATCHED_MODALITY_CALLS
            )
            initial_iter_cfg["mm_calls"] = (
                V11_MATCHED_MODALITY_CALLS if native_multimodal_arm else 0
            )

        initial_prop = proposer_agent.propose_initial(
            dataset=dataset,
            row_meta=row_meta,
            observation=observation,
            iter_cfg=initial_iter_cfg,
            experience_prior=guidance_prior,
            timer=timer,
        )
        initial_exprs = list(initial_prop.get("candidate_exprs", []) or [])
        prop_trace = dict(initial_prop.get("trace", {}) or {})
        if V11_FORCE_STRUCTURAL_COVERAGE:
            # Do not trust a collapsed language-model proposal set: reserve
            # deterministic candidates from distinct algebraic families.
            manual = build_manual_candidates(list(getattr(dataset, "feature_names", []) or []))
            forced = []
            family_seen = set()
            for expr in manual:
                text = str(expr)
                family = (
                    "piecewise" if "sign(" in text or "Abs(" in text else
                    "rational" if "/" in text else
                    "power" if "**" in text else
                    "interaction" if "*" in text else
                    "additive"
                )
                if family not in family_seen:
                    forced.append(text)
                    family_seen.add(family)
                if len(forced) >= 8:
                    break
            initial_exprs = _uniq(initial_exprs + forced)
            prop_trace["forced_structural_coverage"] = forced

        initial_modality_exprs = []
        initial_modality_trace = {}
        initial_modality_mode = "shared_default"
        if V11_FORCE_INITIAL_MODALITY_PROPOSAL and native_multimodal_arm:
            mm_requested = True
            mm_trigger_reason = "forced_matched_initial_multimodal_proposal"
            mm_prop = _matched_initial_mm_proposal(
                proposer_agent=proposer_agent,
                observer_agent=observer_agent,
                dataset=dataset,
                observation=observation,
                row_meta=row_meta,
                iter_cfg=initial_iter_cfg,
                timer=timer,
            )
            initial_modality_exprs = list(mm_prop.get("candidate_exprs", []) or [])
            initial_modality_trace = dict(mm_prop.get("trace", {}) or {})
            mm_prop_trace = initial_modality_trace
            mm_candidate_count += len(initial_modality_exprs)
            prop_trace["mm_proposal_stats"] = _base.make_json_safe(
                initial_modality_trace
            )
            initial_modality_mode = "image_conditioned"
        elif V11_FORCE_INITIAL_MODALITY_PROPOSAL and numeric_only_arm:
            text_stats = dict(prop_trace.get("text_proposal_stats", {}) or {})
            initial_modality_exprs = [
                str(item.get("expression", "")).strip()
                for item in list(text_stats.get("candidates", []) or [])
                if isinstance(item, dict) and str(item.get("expression", "")).strip()
            ]
            initial_modality_trace = text_stats
            initial_modality_mode = "numeric_text_control"

        if V11_FORCE_INITIAL_MODALITY_PROPOSAL:
            # Put the treatment/control model candidates in an equal reserved
            # prefix, then fill from the shared deterministic pool. Both arms
            # submit exactly the same maximum number of initial candidates.
            initial_exprs = _base.deduplicate_expressions(
                list(initial_modality_exprs) + list(initial_exprs)
            )[:V11_MATCHED_INITIAL_CANDIDATES]
            proposal_history.append({
                "stage": "initial_matched_modality_proposer",
                "mode": initial_modality_mode,
                "num_exprs": len(initial_modality_exprs),
                "submitted_exprs": _base.make_json_safe(
                    initial_exprs[:min(len(initial_modality_exprs), len(initial_exprs))]
                ),
                "exprs": _base.make_json_safe(initial_modality_exprs),
                "image_paths": _base.make_json_safe(
                    list(initial_modality_trace.get("image_paths", []) or [])
                    if native_multimodal_arm else []
                ),
                "trace": _base.make_json_safe(initial_modality_trace),
            })
            result["initial_modality_proposal_mode"] = initial_modality_mode
            result["initial_modality_candidate_count"] = len(initial_modality_exprs)
            result["initial_modality_candidates_submitted"] = min(
                len(initial_modality_exprs), len(initial_exprs)
            )
            result["initial_modality_input_image_count"] = (
                len(list(initial_modality_trace.get("image_paths", []) or []))
                if native_multimodal_arm else 0
            )
            result["initial_modality_proposal_valid"] = bool(
                initial_modality_exprs
                and (not native_multimodal_arm or result["initial_modality_input_image_count"] > 0)
            )
            result["initial_modality_failure_reason"] = (
                None
                if result["initial_modality_proposal_valid"]
                else (
                    "image_conditioned_proposer_returned_no_candidates"
                    if native_multimodal_arm
                    else "numeric_text_control_returned_no_candidates"
                )
            )
            if native_multimodal_arm:
                result["mm_proposal_stats"] = _base.json.dumps(
                    _base.make_json_safe(initial_modality_trace), ensure_ascii=False
                )

        extrap_ols_exprs, extrap_ols_trace = _build_extrapolation_ols_candidates(dataset, row_meta=row_meta)
        surface_analytic_exprs, surface_analytic_trace = _build_surface_analytic_candidates(dataset, row_meta=row_meta)
        surface_grid_exprs, surface_grid_trace = _build_surface_grid_basis_candidates(dataset, row_meta=row_meta)
        surface_stable_exprs, surface_stable_trace = _build_surface_stable_basis_candidates(dataset, row_meta=row_meta)
        constructed_exprs, constructed_trace = _build_constructed_extrapolation_candidates(dataset, row_meta=row_meta)
        if extrap_ols_exprs or surface_analytic_exprs or surface_grid_exprs or surface_stable_exprs or constructed_exprs:
            initial_exprs = _base.deduplicate_expressions(
                list(constructed_exprs)
                + list(surface_stable_exprs)
                + list(surface_grid_exprs)
                + list(surface_analytic_exprs)
                + list(extrap_ols_exprs)
                + initial_exprs
            )
            prop_trace["extrapolation_ols"] = _base.make_json_safe(extrap_ols_trace)
            prop_trace["surface_analytic"] = _base.make_json_safe(surface_analytic_trace)
            prop_trace["surface_grid"] = _base.make_json_safe(surface_grid_trace)
            prop_trace["surface_stable"] = _base.make_json_safe(surface_stable_trace)
            prop_trace["constructed_extrapolation"] = _base.make_json_safe(constructed_trace)
        if V11_FORCE_INITIAL_MODALITY_PROPOSAL:
            initial_exprs = _base.deduplicate_expressions(
                list(initial_modality_exprs) + list(initial_exprs)
            )[:V11_MATCHED_INITIAL_CANDIDATES]
        result["raw_exprs"] = _base.json.dumps(_base.make_json_safe(initial_exprs), ensure_ascii=False)
        result["num_candidate_exprs"] = len(initial_exprs)
        result["text_proposal_stats"] = _base.json.dumps(_base.make_json_safe(prop_trace.get("text_proposal_stats")), ensure_ascii=False)
        result["mm_proposal_stats"] = _base.json.dumps(_base.make_json_safe(prop_trace.get("mm_proposal_stats")), ensure_ascii=False)
        result["diverse_proposal_count"] = prop_trace.get("diverse_proposal_count")
        result["manual_candidate_count"] = prop_trace.get("manual_candidate_count")
        result["datadriven_candidate_count"] = prop_trace.get("datadriven_candidate_count")
        result["generic_candidate_count"] = prop_trace.get("generic_candidate_count")
        result["protected_candidate_count"] = prop_trace.get("protected_candidate_count")
        result["memory_candidate_count"] = prop_trace.get("memory_candidate_count")
        result["experience_candidate_count"] = prop_trace.get("experience_candidate_count")
        result["heuristic_candidate_count"] = prop_trace.get("heuristic_candidate_count")
        result["extrapolation_ols_candidate_count"] = len(extrap_ols_exprs)
        result["extrapolation_ols_trace"] = _base.json.dumps(_base.make_json_safe(extrap_ols_trace), ensure_ascii=False)
        result["surface_analytic_candidate_count"] = len(surface_analytic_exprs)
        result["surface_analytic_trace"] = _base.json.dumps(_base.make_json_safe(surface_analytic_trace), ensure_ascii=False)
        result["surface_grid_candidate_count"] = len(surface_grid_exprs)
        result["surface_grid_trace"] = _base.json.dumps(_base.make_json_safe(surface_grid_trace), ensure_ascii=False)
        result["surface_stable_candidate_count"] = len(surface_stable_exprs)
        result["surface_stable_trace"] = _base.json.dumps(_base.make_json_safe(surface_stable_trace), ensure_ascii=False)
        result["constructed_extrapolation_candidate_count"] = len(constructed_exprs)
        result["constructed_extrapolation_trace"] = _base.json.dumps(_base.make_json_safe(constructed_trace), ensure_ascii=False)
        result["proposal_source_stats"] = _base.json.dumps(_base.make_json_safe(prop_trace.get("source_stats")), ensure_ascii=False)
        result["experience_rerank"] = _base.json.dumps(_base.make_json_safe(prop_trace.get("experience_rerank")), ensure_ascii=False)
        result["merged_candidate_count"] = prop_trace.get("merged_candidate_count")
        proposal_history.append({
            "stage": "initial_proposer",
            "num_exprs": len(initial_exprs),
            "exprs": _base.make_json_safe(initial_exprs),
            "trace": _base.make_json_safe(prop_trace),
        })
        _base._raise_if_task_budget_exceeded(task_deadline_ts, "initial_proposer")

        # 3. Evaluator Agent
        active_eval = evaluator_agent.evaluate(
            candidate_exprs=initial_exprs,
            dataset=dataset,
            row_meta=row_meta,
            timer=timer,
            prefix="loop_initial",
            deadline_ts=task_deadline_ts,
        )
        if (
            V11_FORCE_INITIAL_MODALITY_PROPOSAL
            and native_multimodal_arm
            and initial_modality_exprs
            and int(active_eval.get("evaluated_candidate_count") or 0) > 0
        ):
            mm_used_in_evaluation = True
        current_best = active_eval.get("best_result")
        evaluation_history.append({
            "stage": "initial_evaluator",
            "requested_candidate_count": active_eval.get("requested_candidate_count"),
            "evaluated_candidate_count": active_eval.get("evaluated_candidate_count"),
            "evaluation_truncated": active_eval.get("evaluation_truncated"),
            "topk": active_eval.get("evaluation_table"),
            "pareto_candidates": active_eval.get("candidate_pareto_table"),
            "pareto_front_size": active_eval.get("pareto_front_size"),
            "structure_evaluator": active_eval.get("structure_evaluator"),
            "residual_summary": active_eval.get("residual_summary"),
            "physics_summary": active_eval.get("physics_summary"),
        })
        agent_trace.append({
            "event": "initial_evaluator_done",
            "best_expr": _base._safe_get_attr(current_best, "simplified_expression", None) if current_best is not None else None,
            "best_val_mse": _base._safe_get_attr(current_best, "val_mse", None) if current_best is not None else None,
        })

        max_rounds = max(0, int(iter_cfg.get("refine_rounds", V11_IMAGE_LOOP_MAX_ROUNDS)))
        max_rounds = min(max_rounds, V11_IMAGE_LOOP_MAX_ROUNDS)
        initial_train_r2 = _finite_float(_base._safe_get_attr(current_best, "train_r2", None))
        if (
            V11_ENABLE_TRAIN_R2_EARLY_STOP
            and initial_train_r2 is not None
            and initial_train_r2 > V11_EARLY_STOP_TRAIN_R2_THRESHOLD
        ):
            max_rounds = 0
            result["early_stop_reason"] = (
                f"train_r2>{V11_EARLY_STOP_TRAIN_R2_THRESHOLD:g}"
            )
        if not V11_ENABLE_CRITIC_LOOP:
            max_rounds = 0
            result["early_stop_reason"] = "ablation_disable_critic_loop"

        for round_idx in range(max_rounds):
            round_num = round_idx + 1
            round_start = _base.time.time()
            _base._raise_if_task_budget_exceeded(task_deadline_ts, f"critic_round_{round_num}_start")

            # 4/5. Evaluator -> Critic Agent
            decision = critic_agent.decide(
                dataset=dataset,
                observation=observation,
                evaluation=active_eval,
                current_best=current_best,
                iter_cfg=iter_cfg,
                round_idx=round_idx,
                row_meta=row_meta,
            )
            _base._raise_if_task_budget_exceeded(
                task_deadline_ts,
                f"critic_round_{round_num}_decision",
            )
            raw_critic_decision = _base.make_json_safe(decision)
            if V11_CRITIC_FEEDBACK_MODE == "generic":
                decision = _generic_refinement_decision(current_best, iter_cfg, round_idx)
                decision["discarded_critic_decision"] = raw_critic_decision
            elif V11_MATCH_REFINEMENT_BUDGET:
                decision = _force_matched_refinement_budget(decision, iter_cfg, round_idx)
            meta_decisions.append(_base.make_json_safe(decision))
            action = str(decision.get("critic_action") or "refine")
            agent_trace.append({
                "event": "critic_decision",
                "round": round_num,
                "action": action,
                "should_refine": bool(decision.get("should_refine", True)),
                "target_families": decision.get("target_families"),
                "active_variables": decision.get("active_variables"),
                "reason": decision.get("reason"),
                "diagnostic_image_paths": decision.get("diagnostic_image_paths", []),
            })

            if action == "stop" or not bool(decision.get("should_refine", True)):
                result["early_stop_reason"] = f"critic_stop: {decision.get('reason', '')}"
                record_round_timing(round_num, round_start, "critic_stop", {"decision": decision})
                break

            # Adaptive Multimodal Structural Augmentation.
            round_expr_groups = []
            if V11_ENABLE_OBSERVER and bool((decision.get("budget", {}) or {}).get("use_multimodal", False)):
                mm_requested = True
                mm_trigger_reason = decision.get("reason") or action
                observation = _base.timed_call(
                    timer,
                    f"loop_round_{round_num}_mm_assets",
                    observer_agent.maybe_generate_mm_assets,
                    dataset,
                    row_meta,
                    observation,
                    timer=timer,
                )
                mm_prop = proposer_agent.propose_mm_if_needed(
                    dataset=dataset,
                    observation=observation,
                    iter_cfg=iter_cfg,
                    row_meta=row_meta,
                    timer=timer,
                )
                _base._raise_if_task_budget_exceeded(
                    task_deadline_ts,
                    f"critic_round_{round_num}_multimodal_proposal",
                )
                mm_exprs = list(mm_prop.get("candidate_exprs", []) or [])
                mm_prop_trace = mm_prop.get("trace")
                mm_candidate_count += len(mm_exprs)
                round_expr_groups.append(mm_exprs)
                proposal_history.append({
                    "stage": f"round_{round_num}_adaptive_multimodal",
                    "num_exprs": len(mm_exprs),
                    "exprs": _base.make_json_safe(mm_exprs),
                    "trace": _base.make_json_safe(mm_prop_trace),
                })

            # Proposer Agent receives Critic feedback and regenerates/refines.
            if V11_ENABLE_OBSERVER and hasattr(evaluator_agent, "set_structure_context"):
                evaluator_agent.set_structure_context(observation=observation, row_meta=row_meta)
            judge_feedback = judge_agent.build_feedback(
                dataset=dataset,
                observation=observation,
                evaluation=active_eval,
                meta_decision=decision,
                iter_cfg=iter_cfg,
            )
            _base._raise_if_task_budget_exceeded(
                task_deadline_ts,
                f"critic_round_{round_num}_judge",
            )
            if V11_CRITIC_FEEDBACK_MODE == "generic":
                discarded_judge_feedback = _base.make_json_safe(judge_feedback)
                judge_feedback = _generic_judge_feedback()
                judge_feedback["discarded_judge_feedback"] = discarded_judge_feedback
            judge_feedback_history.append(_base.make_json_safe(judge_feedback))
            try:
                refiner_diagnostic_image_paths = decision.get("diagnostic_image_paths")
                if V11_OBSERVER_INPUT_MODE in {"numeric", "numeric_only", "no_image", "no_images"}:
                    # See CriticAgent.decide: the base Refiner otherwise
                    # auto-creates and sends residual images for an empty list.
                    refiner_diagnostic_image_paths = ["__llmsr_numeric_only_no_image__"]
                refined = refiner_agent.refine(
                    dataset=dataset,
                    observation=observation,
                    current_best=current_best,
                    meta_decision=decision,
                    judge_feedback=judge_feedback,
                    iter_cfg=iter_cfg,
                    row_meta=row_meta,
                    diagnostic_image_paths=refiner_diagnostic_image_paths,
                )
            except TypeError as exc:
                if "unexpected keyword argument" not in str(exc):
                    raise
                try:
                    refined = refiner_agent.refine(
                        dataset=dataset,
                        observation=observation,
                        current_best=current_best,
                        meta_decision=decision,
                        judge_feedback=judge_feedback,
                        iter_cfg=iter_cfg,
                        diagnostic_image_paths=refiner_diagnostic_image_paths,
                    )
                except TypeError as inner_exc:
                    if "unexpected keyword argument" not in str(inner_exc):
                        raise
                    refined = refiner_agent.refine(
                        dataset=dataset,
                        observation=observation,
                        current_best=current_best,
                        meta_decision=decision,
                        judge_feedback=judge_feedback,
                        iter_cfg=iter_cfg,
                    )
            _base._raise_if_task_budget_exceeded(
                task_deadline_ts,
                f"critic_round_{round_num}_refiner",
            )
            refined_exprs = list(refined.get("candidate_exprs", []) or [])
            augmented_exprs, augment_trace = _build_critic_augmented_candidates(
                decision=decision,
                dataset=dataset,
                observation=observation,
                current_best=current_best,
                row_meta=row_meta,
            )
            round_exprs = _base.deduplicate_expressions(refined_exprs + augmented_exprs)
            round_exprs = round_exprs[:V11_IMAGE_LOOP_MAX_AUGMENTED_CANDIDATES]
            if V11_MATCH_REFINEMENT_BUDGET:
                round_exprs = round_exprs[:_matched_round_candidate_limit(iter_cfg)]
            refine_round_expr_counts.append(len(round_exprs))
            proposal_history.append({
                "stage": f"round_{round_num}_critic_conditioned_proposer",
                "num_exprs": len(round_exprs),
                "exprs": _base.make_json_safe(round_exprs),
                "trace": {
                    "critic_decision": _base.make_json_safe(decision),
                    "judge_feedback": _base.make_json_safe(judge_feedback),
                    "refiner_raw_text": refined.get("raw_text", ""),
                    "diagnostic_image_paths": refined.get("diagnostic_image_paths", []),
                    "num_refiner_exprs": len(refined_exprs),
                    "augmentation": _base.make_json_safe(augment_trace),
                },
            })

            if not round_exprs:
                record_round_timing(round_num, round_start, "no_candidates", {"decision": decision})
                result["early_stop_reason"] = "critic_loop_no_candidates"
                break

            # Evaluator Agent re-scores the round candidate set.
            round_eval = evaluator_agent.evaluate(
                candidate_exprs=round_exprs,
                dataset=dataset,
                row_meta=row_meta,
                timer=timer,
                prefix=f"loop_round_{round_num}_{action}",
                deadline_ts=task_deadline_ts,
            )
            evaluation_history.append({
                "stage": f"round_{round_num}_evaluator",
                "critic_action": action,
                "requested_candidate_count": round_eval.get("requested_candidate_count"),
                "evaluated_candidate_count": round_eval.get("evaluated_candidate_count"),
                "evaluation_truncated": round_eval.get("evaluation_truncated"),
                "topk": round_eval.get("evaluation_table"),
                "pareto_candidates": round_eval.get("candidate_pareto_table"),
                "pareto_front_size": round_eval.get("pareto_front_size"),
                "structure_evaluator": round_eval.get("structure_evaluator"),
                "residual_summary": round_eval.get("residual_summary"),
                "physics_summary": round_eval.get("physics_summary"),
            })

            round_best = round_eval.get("best_result")
            improved = is_better_result(round_best, current_best)
            if improved:
                current_best = round_best
                active_eval = round_eval
                mm_used_in_evaluation = mm_used_in_evaluation or bool((decision.get("budget", {}) or {}).get("use_multimodal", False))
            refine_history.append({
                "round": round_num,
                "critic_action": action,
                "decision": _base.make_json_safe(decision),
                "improved": bool(improved),
                "best_expr_after_round": _base._safe_get_attr(current_best, "simplified_expression", None) if current_best is not None else None,
                "best_val_mse_after_round": _base._safe_get_attr(current_best, "val_mse", None) if current_best is not None else None,
            })
            record_round_timing(round_num, round_start, "completed", {"critic_action": action, "improved": bool(improved)})
            current_train_r2 = _finite_float(_base._safe_get_attr(current_best, "train_r2", None))
            if (
                V11_ENABLE_TRAIN_R2_EARLY_STOP
                and current_train_r2 is not None
                and current_train_r2 > V11_EARLY_STOP_TRAIN_R2_THRESHOLD
            ):
                result["early_stop_reason"] = (
                    f"train_r2>{V11_EARLY_STOP_TRAIN_R2_THRESHOLD:g}"
                )
                break

        return _finalize_image_loop(
            result=result,
            timer=timer,
            start=start,
            current_best=current_best,
            dataset=dataset,
            row_meta=row_meta,
            observation=observation,
            active_eval=active_eval,
            meta_decisions=meta_decisions,
            refine_history=refine_history,
            refine_round_expr_counts=refine_round_expr_counts,
            refine_round_timings=refine_round_timings,
            judge_feedback_history=judge_feedback_history,
            proposal_history=proposal_history,
            evaluation_history=evaluation_history,
            agent_trace=agent_trace,
            mm_requested=mm_requested,
            mm_trigger_reason=mm_trigger_reason,
            mm_prop_trace=mm_prop_trace,
            mm_candidate_count=mm_candidate_count,
            mm_used_in_evaluation=mm_used_in_evaluation,
        )

    except _base.TaskTimeBudgetExceeded as e:
        result["time_budget_hit"] = True
        result["early_stop_reason"] = f"time_budget_exceeded>{float(getattr(_base, 'MAX_RUNTIME_PER_TASK_SEC', 0)):.1f}s"
        agent_trace.append({"event": "time_budget_exceeded", "reason": repr(e)})
        return _finalize_image_loop(
            result, timer, start, current_best, dataset, row_meta, observation, active_eval,
            meta_decisions, refine_history, refine_round_expr_counts, refine_round_timings,
            judge_feedback_history, proposal_history, evaluation_history, agent_trace,
            mm_requested, mm_trigger_reason, mm_prop_trace, mm_candidate_count, mm_used_in_evaluation,
        )
    except Exception as e:
        result["error"] = repr(e)
        agent_trace.append({"event": "exception", "error": repr(e)})
        return _finalize_image_loop(
            result, timer, start, current_best, dataset, row_meta, observation, active_eval,
            meta_decisions, refine_history, refine_round_expr_counts, refine_round_timings,
            judge_feedback_history, proposal_history, evaluation_history, agent_trace,
            mm_requested, mm_trigger_reason, mm_prop_trace, mm_candidate_count, mm_used_in_evaluation,
        )
    finally:
        budget_guard.__exit__(None, None, None)


def build_dataset_from_explicit_splits(*args, **kwargs):
    return _base.build_dataset_from_explicit_splits(*args, **kwargs)


def save_all_outputs(*args, **kwargs):
    _sync_runtime_globals()
    return _base.save_all_outputs(*args, **kwargs)


def print_summary(*args, **kwargs):
    _sync_runtime_globals()
    return _base.print_summary(*args, **kwargs)
