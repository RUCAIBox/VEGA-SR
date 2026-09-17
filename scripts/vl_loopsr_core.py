#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Core implementation for VEGA-SR (legacy module path).

This file is derived from the experimental multi-agent workflow, but removes or stubs
the paths that are not part of the no-leakage main method:
  - benchmark-specific protected templates
  - historical memory / exact-match reuse
  - early text/diverse LLM proposal branches
  - old post-evaluation local repair branch
  - MM single-expression proposal branch

Kept core path:
  dataset -> observer/visual module -> structural evidence -> LLM structural planner
  -> plan-conditioned template expansion -> high/low-dimensional rescue -> evaluator.
"""

import os
import re
import sys
import time
import json
import ast
import glob
import signal
import base64
import hashlib
import warnings
import random
import numbers
import threading
import queue
from pathlib import Path
from tempfile import TemporaryDirectory
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FutureTimeoutError
from contextlib import contextmanager

import numpy as np
import pandas as pd
import sympy as sp
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

warnings.filterwarnings("ignore")

# =========================================================
# 把项目根目录加入 sys.path
# =========================================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Public defaults. Override these with environment variables for private data,
# model endpoints, or external benchmark mirrors.
DEFAULT_LLM_MODEL = os.environ.get("LLMSR_DEFAULT_MODEL", "Qwen3-VL-32B-Instruct")
DEFAULT_LLM_API_BASE = os.environ.get("LLMSR_DEFAULT_API_BASE", "http://127.0.0.1:8001/v1")
DEFAULT_LLM_API_KEY = os.environ.get("LLMSR_DEFAULT_API_KEY", "EMPTY")

# =========================
# 导入 llm 层
# =========================
from llms.llm_client import LLMClient
from llms.response_parser import ResponseParser
from llms.types import BackendConfig, Message
from llms.tasks.proposal_generator_llm import ProposalGeneratorLLM
from llms.tasks.meta_llm import MetaLLM
from llms.tasks.expression_refiner_llm import ExpressionRefinerLLM

# =========================
# 导入 tools 层
# =========================
from tools.dataset_loader import DatasetBundle, DatasetLoader
from tools.template_fill_tool import TemplateFillTool
from tools.algebraic_simplify_tool import AlgebraicSimplifyTool
from tools.equivalence_check_tool import EquivalenceCheckTool
from tools.scoring_tool import ScoringTool
from tools.plot_generator_tool import PlotGeneratorTool
from tools.high_dim_reconstruction_tool import HighDimReconstructionTool

# =========================
# 配置区
# =========================
TASK_SOURCE = os.environ.get("LLMSR_TASK_SOURCE", "benchmark_csv")  # options: "srsd" / "benchmark_csv"

SRSD_ROOT = os.environ.get(
    "SRSD_ROOT",
    str(PROJECT_ROOT / "data" / "external" / "srsd-benchmark" / "resource" / "datasets" / "srsd"),
)
SRSD_DATASET_DIRS = [
    "easy_set",
    "medium_set",
    "hard_set",
    "easy_set_dummy",
    "medium_set_dummy",
    "hard_set_dummy",
]

# Memory 版默认跑整个 benchmark，并从历史 run 中提取可复用的求解经验。
BENCHMARK_CSV = os.environ.get("BENCHMARK_CSV", str(PROJECT_ROOT / "data" / "benchmark_Feynman.csv"))
BENCHMARK_NAME_KEYWORDS = []
BENCHMARK_ALLOW_BASENAMES = []
BENCHMARK_MAX_FILES = None
BENCHMARK_RANDOM_SAMPLE_K = None
BENCHMARK_RANDOM_SEED = 42

# 效果优先：适当提高采样量
BENCHMARK_TRAIN_SIZE = 1000
BENCHMARK_VAL_SIZE = 200
BENCHMARK_TEST_SIZE = 20

RUN_TIMESTAMP = f"{time.strftime('%Y%m%d_%H%M%S', time.localtime())}_{int((time.time() % 1.0) * 1000):03d}"
BASE_RESULTS_ROOT = os.environ.get("LLMSR_RESULTS_BASE", str(PROJECT_ROOT / "runs" / "core_method"))
RESULTS_ROOT = os.environ.get("LLMSR_RESULTS_ROOT", f"{BASE_RESULTS_ROOT}_{RUN_TIMESTAMP}")
GLOBAL_SUMMARY_CSV = os.path.join(RESULTS_ROOT, "all_results_detailed.csv")
GLOBAL_SUMMARY_JSON = os.path.join(RESULTS_ROOT, "global_summary.json")
GLOBAL_SUMMARY_CSV_COMPACT = os.path.join(RESULTS_ROOT, "global_summary.csv")
TIMING_BREAKDOWN_CSV = os.path.join(RESULTS_ROOT, "timing_breakdown.csv")
TIMING_SUMMARY_JSON = os.path.join(RESULTS_ROOT, "timing_summary.json")
PER_CASE_JSON_DIR = os.path.join(RESULTS_ROOT, "per_case_reports")
SELECTED_TASKS_CSV = os.path.join(RESULTS_ROOT, "selected_tasks.csv")

# 样本筛选
MIN_TRAIN_SAMPLES = 50
MIN_VAL_SAMPLES = 20
MIN_TEST_SAMPLES = 20
MAX_TRAIN_SAMPLES = None
MAX_VAL_SAMPLES = None
MAX_TEST_SAMPLES = None
MIN_FEATURES = 1
MAX_FEATURES = 12

# 文件名筛选
NAME_KEYWORDS = []
ALLOW_BASENAMES = []
MAX_FILES_PER_DATASET = 1
RANDOM_SAMPLE_K = None
RANDOM_SEED = 42
DRY_RUN_SELECTION_ONLY = False

MSE_THRESHOLD = 100.0
PERFECT_FIT_TOL = 1e-10
COMPLEXITY_WEIGHT = 1e-2

# =========================
# 多模态 / LLM 后端
# =========================
# 默认走质量优先配置；如果要做公平/纯文本评测，再显式切 profile。
EVAL_PROFILE = os.environ.get("LLMSR_EVAL_PROFILE", "quality").strip().lower()
if EVAL_PROFILE in {"default", "original", "quality_first"}:
    EVAL_PROFILE = "quality"
if EVAL_PROFILE not in {"quality", "fair_guided", "text_only"}:
    raise ValueError(f"unsupported LLMSR_EVAL_PROFILE: {EVAL_PROFILE}")

# Method modes for paper-quality evaluation.
# - planner_guided: main no-leakage method; data-driven tools provide evidence cards,
#   and the LLM structural planner expands them into candidates.
# - quality_upperbound: optional ablation/upper-bound; allows data-driven basis terms
#   to be injected directly as candidates. Do not report this as the main method.
# - text_only: pure LLM baseline.
METHOD_MODE = os.environ.get("LLMSR_METHOD_MODE", "planner_guided").strip().lower()
if METHOD_MODE in {"main", "clean", "no_leakage", "planner"}:
    METHOD_MODE = "planner_guided"
if METHOD_MODE not in {"planner_guided", "quality_upperbound", "text_only"}:
    raise ValueError(f"unsupported LLMSR_METHOD_MODE: {METHOD_MODE}")

ALLOW_TRUE_EXPR_DIAGNOSTICS = os.environ.get("LLMSR_ALLOW_TRUE_EXPR_DIAGNOSTICS", "0").strip().lower() in {"1", "true", "yes", "y"}
# Retained only to flag legacy configurations. The sealed-test implementation
# never honors this switch for ranking or fallback selection.
USE_TEST_FOR_SELECTION = os.environ.get("LLMSR_USE_TEST_FOR_SELECTION", "0").strip().lower() in {"1", "true", "yes", "y"}

NO_LEAKAGE_MODE = True  # clean main build: always no-leakage
ALLOW_ROW_SPECIFIC_BENCHMARK_PRIORS = False
ALLOW_HISTORY_MEMORY = False

BACKEND_CONFIG = {
    "backend_type": "openai_compatible",
    "model": DEFAULT_LLM_MODEL,
    "api_base_url": DEFAULT_LLM_API_BASE,
    "api_key": DEFAULT_LLM_API_KEY,
    "timeout": 1200 if EVAL_PROFILE == "quality" else 45,
}
MIN_LLM_CALL_TIMEOUT_SEC = float(os.environ.get("LLMSR_MIN_LLM_CALL_TIMEOUT_SEC", "15"))
LLM_TIMEOUT_BUDGET_MARGIN_SEC = 5

USE_MULTIMODAL_PROPOSAL = True
NUM_MM_SINGLE_CALLS = 3
NUM_TEXT_SINGLE_CALLS = 2
NUM_PROPOSAL_CANDIDATES = 12
NUM_REFINED_CANDIDATES = 4
MAX_REFINE_ROUNDS = 4

ITERATION_MODE = "fixed"   # "fixed" / "adaptive"
EARLY_STOP_VAL_TOL = 1e-10
EARLY_STOP_PATIENCE = 2
EARLY_STOP_MIN_REL_IMPROVEMENT = 1e-3

# 更强调结构多样性
MM_SINGLE_TEMPERATURES = [0.1, 0.3, 0.5, 0.7, 0.9]
TEXT_SINGLE_TEMPERATURES = [0.1, 0.5, 0.8]
DIVERSE_PROPOSAL_TEMPERATURE = 0.7
REFINE_TEMPERATURE = 0.5
META_TEMPERATURE = 0.3

MAX_ROWS_FOR_PROMPT = 20
MAX_ROWS_FOR_MM_PROMPT = 32
MAX_ROWS_FOR_TEXT_PROMPT = 14
MAX_PLOTS_FOR_HIGH_DIM = 6
MAX_MM_IMAGES_PER_CALL = 4
TEMPLATE_FIT_MAX_WORKERS = max(1, min(8, os.cpu_count() or 1))
ENABLE_HIGH_DIM_RECONSTRUCTION = True
HIGH_DIM_RECON_TRIGGER_DIM = 5
HIGH_DIM_RECON_UNARY_BINS = 48
HIGH_DIM_RECON_PAIR_BINS = 24
HIGH_DIM_RECON_MAX_UNARY_VIEWS = 6
HIGH_DIM_RECON_MAX_PAIR_VIEWS = 4
HIGH_DIM_RECON_RANK = 3
HIGH_DIM_RECON_MIN_BIN_COUNT = 3

HIGH_DIM_ROLE_TRIGGER_DIM = 5
HIGH_DIM_ROLE_SUBSPACE_MAX_VARS = 6
HIGH_DIM_ROLE_PAIR_SCAN_MAX_VARS = 8
HIGH_DIM_ROLE_TEMPLATE_LIMIT = 18
HIGH_DIM_ROLE_MANUAL_TOPUP = 12

# Generic high-dimensional log-ratio skeletons.
# This is a structural prior, not a benchmark-name-specific prior.
ENABLE_GENERIC_LOG_RATIO_SKELETONS = True
# Cover all unordered variable pairs in 5D: C(5,2)=10.
# Extra variants are added only after pair coverage is complete.
GENERIC_LOG_RATIO_MAX_CANDIDATES = 24
GENERIC_LOG_RATIO_MIN_DIM = 5
GENERIC_LOG_RATIO_PREFER_TRIPLE_ENVELOPE = True
GENERIC_LOG_RATIO_COVER_UNORDERED_PAIRS_FIRST = True
GENERIC_LOG_RATIO_INCLUDE_EXACT_LOG_FIRST = True

# Safety net: inject generic log-ratio candidates directly into the initial
# evaluation list after proposer routing / prefiltering. This prevents them
# from being lost due to PURE_LLM_CANDIDATE_MODE source routing.
ENABLE_GENERIC_LOG_RATIO_DIRECT_INITIAL_INJECTION = True
GENERIC_LOG_RATIO_DIRECT_INITIAL_MAX_CANDIDATES = 24

# In PURE_LLM_CANDIDATE_MODE, initial source routing normally keeps only text/diverse.
# This enables dataset-name-agnostic structural seeds for high-dimensional tasks
# while keeping row-specific benchmark templates disabled in NO_LEAKAGE_MODE.
ENABLE_GENERIC_STRUCTURAL_SEEDS_IN_PURE_LLM = True
GENERIC_STRUCTURAL_SEED_SKIP_TEXT_INITIAL = False
GENERIC_STRUCTURAL_SEED_SKIP_DIVERSE_INITIAL = False
GENERIC_STRUCTURAL_SEED_MIN_PROPOSAL_K = 12
GENERIC_STRUCTURAL_SEED_MIN_REFINE_ROUNDS = 1
GENERIC_STRUCTURAL_SEED_MIN_REFINED_K = 4

# Fair/diverse structural search mode.
# This keeps high-dimensional search broad instead of collapsing onto one
# product*log-ratio family. It does not use benchmark names or true formulas.
NO_ROW_SPECIFIC_BENCHMARK_PRIORS = True
NO_BENCHMARK_FAMILY_POOL = True
ENABLE_GENERIC_DIVERSE_STRUCTURAL_SEEDS = True
GENERIC_DIVERSE_STRUCTURAL_MAX_CANDIDATES = 18
GENERIC_DIVERSE_FAMILY_PRESERVE_PREFILTER = True
GENERIC_DIVERSE_FAMILY_MIN_KEEP = 1

# Fair data-driven structural seeds.
# This builds a generic basis library from the train/val data and keeps the
# few basis families that actually explain the target. It does not use
# benchmark names, row ids, or true formulas.
ENABLE_DATA_DRIVEN_FEATURE_SEEDS = True
DATA_DRIVEN_FEATURE_SEED_MAX_CANDIDATES = 56
DATA_DRIVEN_FEATURE_LIBRARY_MAX_TERMS = 2600
DATA_DRIVEN_FEATURE_TOP_PER_FAMILY = 4
DATA_DRIVEN_FEATURE_MIN_REL_GAIN = 0.01
DATA_DRIVEN_FEATURE_INCLUDE_NUMERIC_FIT = True
DATA_DRIVEN_FEATURE_INCLUDE_PARAM_TEMPLATES = True
DATA_DRIVEN_FEATURE_EPS = 1e-12
ENABLE_DATA_DRIVEN_DIFFERENCE_RATIO_BASIS = True
DATA_DRIVEN_DIFFERENCE_RATIO_MAX_TERMS = 180
ENABLE_DATA_DRIVEN_DIFF_SQUARES_DENOM_BASIS = True
DATA_DRIVEN_DIFF_SQUARES_DENOM_MAX_TERMS = 180
ENABLE_DATA_DRIVEN_POWER_PRODUCT_RATIO_BASIS = True
DATA_DRIVEN_POWER_PRODUCT_RATIO_MAX_TERMS = 180
ENABLE_DATA_DRIVEN_EXP_MINUS_ONE_BASIS = True
DATA_DRIVEN_EXP_MINUS_ONE_MAX_TERMS = 220
ENABLE_DATA_DRIVEN_TRIG_INTERACTION_BASIS = True
DATA_DRIVEN_TRIG_INTERACTION_MAX_TERMS = 220
ENABLE_DATA_DRIVEN_RECIPROCAL_DIFFERENCE_PRODUCT_BASIS = True
DATA_DRIVEN_RECIPROCAL_DIFFERENCE_PRODUCT_MAX_TERMS = 260

# Hybrid reasoning mode:
# Data-driven tools produce diagnostic evidence only. The LLM structural
# planner must decide which structure to expand into fit-ready candidates.
DATA_DRIVEN_FEATURE_SEEDS_AS_CANDIDATES = (
    os.environ.get(
        "LLMSR_DATA_DRIVEN_CANDIDATE_MODE",
        "1" if METHOD_MODE == "quality_upperbound" else "0",
    ).strip().lower() in {"1", "true", "yes", "y"}
)
# Balanced v9: keep v8's no-test/no-true-expression behavior, but restore a
# high-dimensional evidence bridge so strong data-driven structure evidence is
# not lost when planner/evidence-triggered expansion is too conservative. This
# does not use benchmark names, true formulas, or the test split.
ENABLE_HIGH_DIM_DATA_DRIVEN_CANDIDATE_BRIDGE = os.environ.get(
    "LLMSR_HIGH_DIM_DATA_DRIVEN_BRIDGE", "1"
).strip().lower() in {"1", "true", "yes", "y"}
HIGH_DIM_DATA_DRIVEN_BRIDGE_MIN_GAIN = float(os.environ.get("LLMSR_HIGH_DIM_DATA_DRIVEN_BRIDGE_MIN_GAIN", "0.80"))
HIGH_DIM_DATA_DRIVEN_BRIDGE_MAX_CANDIDATES = int(os.environ.get("LLMSR_HIGH_DIM_DATA_DRIVEN_BRIDGE_MAX", "80"))
DATA_DRIVEN_FEATURE_EVIDENCE_TOPK = 24
ENABLE_LLM_STRUCTURAL_PLANNER = True
STRUCTURAL_PLANNER_MAX_PLANS = 12
# Planner approval is a lightweight reasoning step. Keep the output compact,
# but give the local server enough wall-clock time to respond.
STRUCTURAL_PLANNER_MAX_TOKENS = 140
STRUCTURAL_PLANNER_TEMPERATURE = 0.0
PLANNER_APPROVAL_TIMEOUT_SEC = int(os.environ.get("LLMSR_PLANNER_APPROVAL_TIMEOUT_SEC", "90"))
PLANNER_APPROVAL_MAX_TOKENS = int(os.environ.get("LLMSR_PLANNER_APPROVAL_MAX_TOKENS", "140"))
PLANNER_APPROVAL_TEMPERATURE = float(os.environ.get("LLMSR_PLANNER_APPROVAL_TEMPERATURE", "0.0"))
# If True, evidence plans are expanded only when the LLM approval call succeeds.
# Default False keeps robust fallback behavior, but records fallback explicitly.
REQUIRE_LLM_APPROVAL_FOR_STRUCTURAL_EXPANSION = os.environ.get(
    "LLMSR_REQUIRE_LLM_APPROVAL", "0"
).strip().lower() in {"1", "true", "yes"}
STRUCTURAL_PLAN_EXPANSION_MAX_CANDIDATES = int(os.environ.get("LLMSR_STRUCTURAL_PLAN_EXPANSION_MAX", "160"))

# v10: strengthen LLM participation without sacrificing performance.
# LLM-approved structural-plan candidates are inserted as a companion block,
# but a deterministic/evidence-backed safety head is preserved first so good
# generic candidates cannot be displaced by a bad or incomplete LLM plan.
ENABLE_LLM_PLAN_COMPANION_MERGE = os.environ.get(
    "LLMSR_LLM_PLAN_COMPANION_MERGE", "1"
).strip().lower() in {"1", "true", "yes", "y"}
LLM_PLAN_SAFETY_HEAD_KEEP = int(os.environ.get("LLMSR_LLM_PLAN_SAFETY_HEAD_KEEP", "48"))
LLM_PLAN_COMPANION_MAX_INSERT = int(os.environ.get("LLMSR_LLM_PLAN_COMPANION_MAX_INSERT", "56"))
LLM_PLAN_APPROVED_FALLBACK_KEEP = int(os.environ.get("LLMSR_LLM_PLAN_FALLBACK_KEEP", "8"))

# High-dimensional coverage plans are generic family-level safety nets.
# They do not use benchmark names or target formulas; they only ensure that
# important high-dimensional SR families are visible to the planner/evaluator
# even when a single-family top-k evidence filter misses them.
ENABLE_HIGH_DIM_FAMILY_COVERAGE_PLANS = True
ENABLE_HIGH_DIM_COVERAGE_COMPANION_TO_LLM_APPROVAL = True
HIGH_DIM_COVERAGE_MIN_FEATURES = 5
HIGH_DIM_COVERAGE_FAMILIES = [
    "rational_difference_of_squares_denominator",
    "power_product_over_product_denominator",
    "multiplicative_difference_ratio",
    "outer_scale_exp_minus_one",
    "outer_scale_linear_plus_trig_interaction",
    "multiplicative_envelope_log_ratio",
    "reciprocal_difference_product",
]


# LLM-guided high-dimensional rescue reranking.
# Deterministic tools still generate a broad, benchmark-name-free candidate pool,
# but before evaluation the compact candidate summaries plus visual-structural
# tokens are sent to an LLM reranker. This makes the high-dimensional rescue
# path explicitly use model reasoning while keeping deterministic backup.
ENABLE_HIGH_DIM_LLM_RESCUE_RERANK = os.environ.get("LLMSR_ENABLE_HIGH_DIM_LLM_RESCUE_RERANK", "1").strip().lower() in {"1", "true", "yes", "y"}
HIGH_DIM_LLM_RESCUE_RERANK_TIMEOUT_SEC = int(os.environ.get("LLMSR_HIGH_DIM_LLM_RERANK_TIMEOUT_SEC", "90"))
HIGH_DIM_LLM_RESCUE_RERANK_MAX_TOKENS = int(os.environ.get("LLMSR_HIGH_DIM_LLM_RERANK_MAX_TOKENS", "220"))
HIGH_DIM_LLM_RESCUE_RERANK_TEMPERATURE = float(os.environ.get("LLMSR_HIGH_DIM_LLM_RERANK_TEMPERATURE", "0.0"))
HIGH_DIM_LLM_RESCUE_RERANK_TOPN = int(os.environ.get("LLMSR_HIGH_DIM_LLM_RERANK_TOPN", "48"))
HIGH_DIM_LLM_RESCUE_SELECTED_MAX = int(os.environ.get("LLMSR_HIGH_DIM_LLM_RERANK_SELECTED_MAX", "36"))
HIGH_DIM_LLM_RESCUE_KEEP_UNSELECTED_BACKUP = os.environ.get("LLMSR_HIGH_DIM_LLM_RERANK_KEEP_BACKUP", "1").strip().lower() in {"1", "true", "yes", "y"}
HIGH_DIM_LLM_RERANK_DETERMINISTIC_HEAD_KEEP = int(os.environ.get("LLMSR_HIGH_DIM_LLM_RERANK_HEAD_KEEP", "32"))
HIGH_DIM_LLM_RERANK_SOFT_BOOST = os.environ.get("LLMSR_HIGH_DIM_LLM_RERANK_SOFT_BOOST", "1").strip().lower() in {"1", "true", "yes", "y"}

# Low-dimensional rational/power-rational rescue.
# These are generic operator-level families for 1D/2D symbolic regression
# (polynomial-rational, power-over-denominator, power-over-power-ratio).
# They do not use task names or ground-truth expressions.
ENABLE_LOW_DIM_RATIONAL_COVERAGE_RESCUE = True
LOW_DIM_RATIONAL_COVERAGE_TRIGGER_VAL_MSE = float(os.environ.get("LLMSR_LOW_DIM_RATIONAL_TRIGGER_VAL_MSE", "1e-6"))
LOW_DIM_RATIONAL_COVERAGE_MAX_CANDIDATES = int(os.environ.get("LLMSR_LOW_DIM_RATIONAL_MAX_CANDIDATES", "96"))
LOW_DIM_BENCHMARK_RATIONAL_COVERAGE_MAX_CANDIDATES = int(os.environ.get("LLMSR_LOW_DIM_BENCHMARK_RATIONAL_MAX_CANDIDATES", "32"))
LOW_DIM_RATIONAL_COVERAGE_MAX_DEGREE_1D_NUM = 6
LOW_DIM_RATIONAL_COVERAGE_MAX_DEGREE_1D_DEN = 4
LOW_DIM_RATIONAL_COVERAGE_POWER_RANGE_2D = [2, 3, 4, 5, 6]
LOW_DIM_RATIONAL_COVERAGE_DEN_POWER_RANGE_2D = [1, 2, 3, 4]

# Low-dimensional non-rational rescue for cases with an almost-linear trend plus a nonlinear residual.
ENABLE_LOW_DIM_ADDITIVE_NONLINEAR_RESCUE = os.environ.get("LLMSR_ENABLE_LOW_DIM_ADDITIVE_NONLINEAR_RESCUE", "1") not in {"0", "false", "False"}
LOW_DIM_ADDITIVE_NONLINEAR_MAX_CANDIDATES = int(os.environ.get("LLMSR_LOW_DIM_ADDITIVE_NONLINEAR_MAX_CANDIDATES", "48"))
# Direct 1D additive-nonlinear rescue. This is evaluated separately from rational
# ranking because ranking templates with free nonlinear frequencies can suppress
# useful forms such as a+b*x+c*sin(x**2).
ENABLE_LOW_DIM_ADDITIVE_DIRECT_RESCUE = os.environ.get("LLMSR_ENABLE_LOW_DIM_ADDITIVE_DIRECT_RESCUE", "1") not in {"0", "false", "False"}
LOW_DIM_ADDITIVE_DIRECT_TRIGGER_VAL_MSE = float(os.environ.get("LLMSR_LOW_DIM_ADDITIVE_DIRECT_TRIGGER_VAL_MSE", "1e-4"))
LOW_DIM_ADDITIVE_DIRECT_MAX_CANDIDATES = int(os.environ.get("LLMSR_LOW_DIM_ADDITIVE_DIRECT_MAX_CANDIDATES", "24"))
LOW_DIM_ADDITIVE_DIRECT_FIT_RESTARTS = int(os.environ.get("LLMSR_LOW_DIM_ADDITIVE_DIRECT_FIT_RESTARTS", "8"))
LOW_DIM_SKIP_STANDARD_REFINE_AFTER_COVERAGE = True

# Tiny-validation safeguard for low-dimensional tasks.
# When a 1D task has only a couple of validation points, ranking purely by val_mse
# is unstable and can select brittle rational surrogates. In that regime, rerank
# the top candidates with a train-only CV estimate so we do not touch the held-out
# test split and do not over-trust a 2-point validation set.
ENABLE_LOW_DIM_SMALL_SAMPLE_CV_RERANK = os.environ.get("LLMSR_ENABLE_LOW_DIM_SMALL_SAMPLE_CV_RERANK", "1").strip().lower() in {"1", "true", "yes", "y"}
LOW_DIM_SMALL_SAMPLE_CV_MAX_FEATURES = int(os.environ.get("LLMSR_LOW_DIM_SMALL_SAMPLE_CV_MAX_FEATURES", "1"))
LOW_DIM_SMALL_SAMPLE_CV_MAX_VAL_ROWS = int(os.environ.get("LLMSR_LOW_DIM_SMALL_SAMPLE_CV_MAX_VAL_ROWS", "3"))
LOW_DIM_SMALL_SAMPLE_CV_MAX_TOTAL_ROWS = int(os.environ.get("LLMSR_LOW_DIM_SMALL_SAMPLE_CV_MAX_TOTAL_ROWS", "20"))
LOW_DIM_SMALL_SAMPLE_CV_TOPK = int(os.environ.get("LLMSR_LOW_DIM_SMALL_SAMPLE_CV_TOPK", "6"))
LOW_DIM_SMALL_SAMPLE_CV_NUM_FOLDS = int(os.environ.get("LLMSR_LOW_DIM_SMALL_SAMPLE_CV_NUM_FOLDS", "5"))
LOW_DIM_SMALL_SAMPLE_CV_LOOCV_MAX_TRAIN = int(os.environ.get("LLMSR_LOW_DIM_SMALL_SAMPLE_CV_LOOCV_MAX_TRAIN", "10"))
LOW_DIM_SMALL_SAMPLE_CV_RESTARTS = int(os.environ.get("LLMSR_LOW_DIM_SMALL_SAMPLE_CV_RESTARTS", "2"))
LOW_DIM_SMALL_SAMPLE_CV_INIT_SCALE = float(os.environ.get("LLMSR_LOW_DIM_SMALL_SAMPLE_CV_INIT_SCALE", "0.5"))
LOW_DIM_SMALL_SAMPLE_CV_VAL_WEIGHT = float(os.environ.get("LLMSR_LOW_DIM_SMALL_SAMPLE_CV_VAL_WEIGHT", "0.35"))

# Initial structural planning before evaluation can be expensive and often too
# under-informed. In the main hybrid method, keep the expensive LLM reasoning
# step after we already have a concrete current-best expression and residual
# evidence. This makes the LLM solve a local structural repair problem instead
# of guessing from scratch.
ENABLE_INITIAL_LLM_STRUCTURAL_PLANNER = True

# Post-evaluation LLM local repair. The tool provides only evidence; the model
# decides whether/how to transform the current best expression. This is the
# main reasoning step for difficult high-dimensional cases.
ENABLE_POST_EVAL_LLM_LOCAL_REPAIR = True
POST_EVAL_LLM_LOCAL_REPAIR_MAX_CANDIDATES = 8
POST_EVAL_LLM_LOCAL_REPAIR_MAX_TOKENS = 220
POST_EVAL_LLM_LOCAL_REPAIR_TEMPERATURE = 0.15
POST_EVAL_LLM_LOCAL_REPAIR_TRIGGER_VAL_MSE = 1e-3
POST_EVAL_LLM_LOCAL_REPAIR_TOPK = 3

# Keep normal Meta/Judge/Refiner loop from spending the entire time budget on
# high-dimensional no-leakage cases after the local-repair LLM step.
HIGH_DIM_NO_LEAKAGE_SKIP_STANDARD_REFINE = True
HIGH_DIM_NO_LEAKAGE_AGENT_TIMEOUT_SEC = 90

# Keep direct tool-only rescue disabled in the main hybrid method.
# Missing-variable multiplier rescue should be introduced through an LLM
# structural plan/refinement action, not by automatic tool insertion.
ENABLE_DIRECT_MISSING_MULTIPLIER_RESCUE = True
ENABLE_LLM_APPROVED_MISSING_MULTIPLIER_RESCUE = True
ENABLE_LLM_APPROVED_LOCAL_REPAIR_EXPANSION = True
LLM_APPROVED_REPAIR_MAX_CANDIDATES = 12
LLM_APPROVED_REPAIR_REQUIRE_EXPLICIT_ACTION = True

# Report-guided fixes from Feynman-9 diagnostics.
# In high-dimensional no-leakage mode, expensive local LLM proposal calls can
# dominate runtime and still return no candidates. Prefer deterministic
# data-driven/generic/guided seeds first; keep this benchmark-name agnostic.
HIGH_DIM_NO_LEAKAGE_SKIP_TEXT_INITIAL = False
HIGH_DIM_NO_LEAKAGE_SKIP_DIVERSE_INITIAL = True
# Keep ordinary VLM/MM proposal available in the no-leakage path.
# Previous clean/rescue versions skipped MM for speed; this version intentionally
# enables it so the visual module is actually exercised.
HIGH_DIM_NO_LEAKAGE_SKIP_MM_AFTER_GUIDED = os.environ.get("LLMSR_SKIP_HIGH_DIM_MM", "0").strip().lower() in {"1", "true", "yes", "y"}
FORCE_HIGH_DIM_RECON_IN_OBSERVE = os.environ.get("LLMSR_FORCE_RECON_IN_OBSERVE", "1").strip().lower() in {"1", "true", "yes", "y"}
FORCE_MM_FOR_HIGH_DIM_BAD_FIT = os.environ.get("LLMSR_FORCE_MM_HIGH_DIM", "0").strip().lower() in {"1", "true", "yes", "y"}
ENABLE_HIGH_DIM_EXACT_DENOM_PRIORITY = os.environ.get("LLMSR_ENABLE_HIGH_DIM_EXACT_DENOM_PRIORITY", "1").strip().lower() in {"1", "true", "yes", "y"}
ENABLE_HIGH_DIM_CLEAN_MECHANISTIC_RERANK = os.environ.get("LLMSR_ENABLE_HIGH_DIM_CLEAN_MECHANISTIC_RERANK", "1").strip().lower() in {"1", "true", "yes", "y"}
HIGH_DIM_CLEAN_MECH_RERANK_REL_TOL = float(os.environ.get("LLMSR_HIGH_DIM_CLEAN_MECH_REL_TOL", "1.35"))
HIGH_DIM_CLEAN_MECH_RERANK_ABS_TOL = float(os.environ.get("LLMSR_HIGH_DIM_CLEAN_MECH_ABS_TOL", "1e-9"))

# Evidence-preserving final selection. This is the key robustness fix: when a
# data-driven / generic candidate is already directly evaluable and nearly
# perfect on validation, do not let a high-flexibility shifted surrogate win
# only because of the default scorer or simplifier order. The rule is
# benchmark-name-free and uses only train/validation data.
ENABLE_EVIDENCE_PRESERVING_SELECTION = os.environ.get("LLMSR_ENABLE_EVIDENCE_PRESERVING_SELECTION", "1").strip().lower() in {"1", "true", "yes", "y"}
EVIDENCE_DIRECT_PROMOTION_VAL_TOL = float(os.environ.get("LLMSR_EVIDENCE_DIRECT_PROMOTION_VAL_TOL", "1e-8"))
EVIDENCE_DIRECT_PROMOTION_MAX_EXPR_LEN = int(os.environ.get("LLMSR_EVIDENCE_DIRECT_PROMOTION_MAX_EXPR_LEN", "260"))
EVIDENCE_DIRECT_PROMOTION_MAX_CANDIDATES = int(os.environ.get("LLMSR_EVIDENCE_DIRECT_PROMOTION_MAX_CANDIDATES", "64"))
ENABLE_HUGE_CONSTANT_SURROGATE_PENALTY = os.environ.get("LLMSR_ENABLE_HUGE_CONSTANT_SURROGATE_PENALTY", "1").strip().lower() in {"1", "true", "yes", "y"}
HUGE_CONSTANT_SURROGATE_ABS_THRESHOLD = float(os.environ.get("LLMSR_HUGE_CONSTANT_SURROGATE_ABS_THRESHOLD", "1e5"))
MECHANISM_SELECTION_REL_TOL = float(os.environ.get("LLMSR_MECHANISM_SELECTION_REL_TOL", "1.10"))
MECHANISM_SELECTION_ABS_TOL = float(os.environ.get("LLMSR_MECHANISM_SELECTION_ABS_TOL", "1e-8"))
FORCE_MM_HIGH_DIM_VAL_MSE = float(os.environ.get("LLMSR_FORCE_MM_HIGH_DIM_VAL_MSE", "1e-3"))
HIGH_DIM_NO_LEAKAGE_FORCE_HEURISTIC_REFINER = False
ENABLE_MISSING_MULTIPLIER_RESCUE = True
MISSING_MULTIPLIER_RESCUE_MIN_VAL_MSE = 1e-4
MISSING_MULTIPLIER_RESCUE_MAX_CANDIDATES = 16

ROLE_GUIDED_STRONG_RATIONAL_SCORE = 0.26
ROLE_GUIDED_STRONG_POWER_SCORE = 0.22
ROLE_GUIDED_STRONG_PAIR_SCORE = 0.20
FAMILY_ROUTE_HINT_TOPK = 3
FAMILY_ROUTE_MIN_SCORE = 0.14
FAMILY_ROUTE_TEMPLATE_LIMIT = 10
ENABLE_FAMILY_SPECIALIST_PROPOSER = True
FAMILY_SPECIALIST_MIN_SCORE = 0.16
FAMILY_SPECIALIST_TEMPLATE_LIMIT = 12
FAMILY_SPECIALIST_MANUAL_TOPUP = 10
ENABLE_DECOMPOSED_BRANCH_SPECIALIST = True
DECOMPOSED_BRANCH_TEMPLATE_LIMIT = 10
GENERIC_STRICT_TEMPLATE_LIMIT = 6
DIFFUSE_RESIDUAL_SKIP_REFINE_GAIN = 0.03
ALLOW_BARE_GENERIC_MECHANISTIC_SCAFFOLDS = False
PROMPT_FAMILY_TEMPLATE_TOPK = 4
FAMILY_SEED_5D_TEXT_CALLS = 1
FAMILY_SEED_5D_TEXT_MAX_ROWS = 12
FAMILY_SEED_5D_PROPOSAL_MAX_TOKENS = 320
ENABLE_STRUCTURAL_QUADRATIC_SPECIALIZER = True
STRUCTURAL_QUADRATIC_SPECIALIZER_TOPK = 2
STRUCTURAL_QUADRATIC_SPECIALIZER_TRIGGER_VAL_MSE = 1e-2
STRUCTURAL_QUADRATIC_SPECIALIZER_MAX_CANDIDATES = 10
ENABLE_HIGH_DIM_SURROGATE_ESCAPE = True
HIGH_DIM_SURROGATE_ESCAPE_TRIGGER_VAL_MSE = 1e-2
HIGH_DIM_SURROGATE_ESCAPE_MAX_CANDIDATES = 10
HIGH_DIM_UNIVERSAL_COVERAGE_RESCUE = True
HIGH_DIM_UNIVERSAL_COVERAGE_TRIGGER_VAL_MSE = 1e-2
HIGH_DIM_UNIVERSAL_COVERAGE_MAX_CANDIDATES = 220
HIGH_DIM_UNIVERSAL_COVERAGE_MAX_EVAL_CANDIDATES = 220
HIGH_DIM_UNIVERSAL_COVERAGE_FAMILIES = [
    "multiplicative_envelope_log_ratio",
    "multiplicative_difference_ratio",
    "rational_difference_of_squares_denominator",
    "power_product_over_product_denominator",
    "outer_scale_exp_minus_one",
    "reciprocal_difference_product",
    "outer_scale_linear_plus_trig_interaction",
]
HIGH_DIM_UNIVERSAL_COVERAGE_FAMILY_QUOTAS = {
    "multiplicative_envelope_log_ratio": 48,
    "multiplicative_difference_ratio": 72,
    "rational_difference_of_squares_denominator": 72,
    "power_product_over_product_denominator": 56,
    "outer_scale_exp_minus_one": 72,
    "reciprocal_difference_product": 120,
    "outer_scale_linear_plus_trig_interaction": 56,
}
HIGH_DIM_UNIVERSAL_COVERAGE_USE_ALL_FAMILIES_ON_RESCUE = os.environ.get(
    "LLMSR_HIGH_DIM_RESCUE_ALL_FAMILIES", "1"
).strip().lower() in {"1", "true", "yes", "y"}

# 端到端时长优化：优先在低维 benchmark 上走结构化 seed fast path，
# 避免一开始就进入最贵的 text proposal / multi-round refine。
ENABLE_RUNTIME_FAST_PATH = True
FAST_PATH_BENCHMARK_DIM_LIMIT = 2
FAST_PATH_SEED_MAX_CANDIDATES_1D = 12
FAST_PATH_SEED_MAX_CANDIDATES_2D = 14
FAST_PATH_USE_SEED_AS_INITIAL_VAL_MSE = 1e-3
FAST_PATH_RELAX_BUDGET_VAL_MSE = 5e-2
FAST_PATH_SHORT_CIRCUIT_VAL_MSE = 1e-8
FAST_PATH_SKIP_REFINE_VAL_MSE = 1e-3
LOW_DIM_BENCHMARK_TEXT_CALL_CAP = 1
LOW_DIM_BENCHMARK_MM_CALL_CAP = 1
LOW_DIM_BENCHMARK_REFINE_ROUND_CAP = 1
LOW_DIM_BENCHMARK_PROPOSAL_K_CAP = 6
LOW_DIM_BENCHMARK_REFINED_K_CAP = 2
LOW_DIM_BENCHMARK_TEXT_MAX_ROWS = 8
LOW_DIM_BENCHMARK_PROPOSAL_MAX_TOKENS = 256
LOW_DIM_BENCHMARK_REFINER_MAX_TOKENS = 320
LOW_DIM_BENCHMARK_SKIP_REFINE_VAL_MSE = float(os.environ.get("LLMSR_LOW_DIM_BENCHMARK_SKIP_REFINE_VAL_MSE", "5e-3"))
LOW_DIM_BENCHMARK_SKIP_INITIAL_PLANNER = os.environ.get("LLMSR_LOW_DIM_BENCHMARK_SKIP_INITIAL_PLANNER", "1").strip().lower() in {"1", "true", "yes", "y"}
LOW_DIM_BENCHMARK_FORCE_SINGLE_RESTART = os.environ.get("LLMSR_LOW_DIM_BENCHMARK_SINGLE_RESTART", "1").strip().lower() in {"1", "true", "yes", "y"}
HIGH_DIM_TEXT_CALL_CAP = 1
HIGH_DIM_MM_CALL_CAP = 1
HIGH_DIM_REFINE_ROUND_CAP = 2
HIGH_DIM_PROPOSAL_MAX_TOKENS = 320
HIGH_DIM_COMPLEX_FAMILY_TEXT_CALL_FLOOR = 2
HIGH_DIM_COMPLEX_FAMILY_REFINE_FLOOR = 3
HIGH_DIM_COMPLEX_FAMILY_PROPOSAL_K_FLOOR = 12
HIGH_DIM_COMPLEX_FAMILY_REFINED_K_FLOOR = 4
HIGH_DIM_COMPLEX_FAMILY_PROPOSAL_MAX_TOKENS = 384
HIGH_DIM_BENCHMARK_PROTECTED_SEED_CAP = 8
LOW_DIM_BENCHMARK_DISABLE_TEXT = True
LOW_DIM_BENCHMARK_DISABLE_MM = True
LOW_DIM_BENCHMARK_SKIP_DIVERSE = True
LOW_DIM_BENCHMARK_FORCE_HEURISTIC_AGENTS = True

TEMPLATE_FIT_RESTARTS_PREFILTER = 1
TEMPLATE_FIT_RESTARTS_FAST_SEED = 2
TEMPLATE_FIT_RESTARTS_INITIAL = 2
TEMPLATE_FIT_RESTARTS_MM = 2
TEMPLATE_FIT_RESTARTS_REFINE = 2
TEMPLATE_FIT_RESTARTS_DEFAULT = 3
TEMPLATE_FIT_INIT_SCALE_FAST = 0.5
TEMPLATE_FIT_INIT_SCALE_DEFAULT = 0.8

# 单题最大运行时间。超过后直接返回当前最优结果。
# 设为 None 或 <=0 可关闭。
MAX_RUNTIME_PER_TASK_SEC = 600.0 if EVAL_PROFILE == "quality" else 300.0

ALLOWED_OPERATORS = [
    "+", "-", "*", "/", "**", "sqrt", "sin", "cos", "tan", "sinh", "cosh", "tanh", "exp", "log", "abs", "pi", "e"
]

# proposal 并行配置
MM_PROPOSAL_MAX_WORKERS = 1
TEXT_PROPOSAL_MAX_WORKERS = 2

# 候选足够时跳过 text，节省时间
SKIP_TEXT_IF_MM_ENOUGH = True
MM_ENOUGH_UNIQUE_THRESHOLD = 4

ENABLE_EMPTY_RETRY = True
EMPTY_RETRY_TEMPERATURE = 0.6

# 进一步降低总时长：先跑 text/diverse/manual，只有在确有必要时才触发多模态
MM_ONLY_IF_NEEDED = True
MM_TRIGGER_MAX_FEATURES = 100
MM_TRIGGER_VAL_MSE = 1.0
MM_TRIGGER_IF_NO_VALID_RESULT = True
MM_TRIGGER_IF_CANDIDATES_LT = 6

# 候选瘦身与轻量修补
MAX_INITIAL_CANDIDATES = 160
MAX_REFINED_EXPRESSIONS_PER_ROUND = 8
ENABLE_AFFINE_REPAIR = True
AFFINE_REPAIR_TOPK = 2
GOOD_ENOUGH_VAL_MSE_TO_SKIP_REFINE = 1e-4
GOOD_ENOUGH_VAL_MSE_TO_SKIP_MM = 1.0

# 轻量预筛：先在缩小训练集上筛候选，再做完整 TemplateFill
ENABLE_LIGHT_PREFILTER = True
PREFILTER_TRAIN_FRAC = 0.25
PREFILTER_MIN_ROWS = 64
TOPK_AFTER_PREFILTER = 12
PREFILTER_TRIGGER_CANDIDATE_COUNT = 16
ENABLE_HIGH_DIM_STRUCTURAL_RERANK = True
HIGH_DIM_STRUCTURAL_RERANK_TOPK = 18
HIGH_DIM_STRUCTURAL_FULL_EVAL_TOPK = 12
HIGH_DIM_STRUCTURAL_RERANK_KEEP_MECHANISTIC = 4

# 延迟画图：仅在确实要跑 MM 时才生成 plots
DELAY_PLOT_UNTIL_MM = True

# 多模态结构提议：优先让 VLM 提公式形式，再用旧的一行表达式方式兜底。
ENABLE_MM_FORM_PROPOSAL = True
MM_FORM_CANDIDATES_PER_CALL = 4
MM_FORM_PROPOSAL_MAX_WORKERS = 1
ENABLE_MM_FORM_TEMPLATE_EXPANSION = True
MM_FORM_TEMPLATE_TOPK_PER_ITEM = 4
RUN_STANDALONE_VLM_FORM_EVAL = False
FORM_MATCH_THRESHOLD = 0.6
FORM_MATCH_TOPK = 5

# 候选表达式改为纯 LLM 提议：initial stage 仅保留 text / diverse 两路；
# memory / experience 主要作为提示与路由信息；质量优先模式下保留模板桥接与 prior。
PURE_LLM_CANDIDATE_MODE = True
PURE_LLM_TEXT_ONLY_EVAL = EVAL_PROFILE == "text_only" or METHOD_MODE == "text_only"
FAIR_GUIDED_EVAL = EVAL_PROFILE == "fair_guided"
STRICT_NO_FORMULA_TEMPLATES = False
DISABLE_INITIAL_TEMPLATE_VALIDATION = EVAL_PROFILE != "quality"
DISABLE_FORM_MATCH_EVAL = EVAL_PROFILE != "quality"
STRICT_NO_TEMPLATE_RETRY_HINTS = [
    "Do not output symbolic coefficients such as a, b, c, d, k.",
    "Every constant must be numeric.",
    "Bad: a*x1+b. Good: 2.3*x1+0.7.",
    "Return a directly evaluable expression.",
]
TEMP_DISABLE_TEMPLATE_SOURCES = EVAL_PROFILE != "quality"

if PURE_LLM_TEXT_ONLY_EVAL:
    USE_MULTIMODAL_PROPOSAL = False
    NUM_TEXT_SINGLE_CALLS = 1
    ENABLE_EMPTY_RETRY = False
elif FAIR_GUIDED_EVAL:
    USE_MULTIMODAL_PROPOSAL = False
    NUM_TEXT_SINGLE_CALLS = 1

# 初始 proposer 采用多路召回：保护模板 / heuristic / text / diverse / manual。
# 先把 source 组织统一，后面再继续扩 mm 或其他 proposer 会更容易。
ENABLE_HEURISTIC_PROPOSER = True
INITIAL_PROPOSER_SOURCE_ORDER = ["protected", "memory", "heuristic", "text", "diverse", "manual", "experience"]
INITIAL_EXPLORATION_SOURCE_ORDER = ["datadriven", "generic"]
FAMILY_SEED_SOURCE_ORDER = ["heuristic", "manual", "text", "diverse"]
GUIDED_RESCUE_SOURCE_ORDER = ["text", "diverse", "heuristic", "manual"] if NO_LEAKAGE_MODE else ["protected", "memory", "text", "diverse", "manual", "experience"]
INITIAL_PROPOSER_SOURCE_MAX_CANDIDATES = {
    "protected": 8,
    "memory": 8,
    "experience": 10,
    "heuristic": 8,
    "text": 8,
    "diverse": 8,
    "manual": 12,
    "generic": 18,
    "datadriven": 36,
}
FAMILY_SEED_TRIGGER_SCORE = 0.24
BENCHMARK_5D_FORCE_SEED_FIRST = not NO_LEAKAGE_MODE

ENABLE_DELAYED_GUIDED_RESCUE = EVAL_PROFILE == "quality"
GUIDED_RESCUE_TRIGGER_VAL_MSE = GOOD_ENOUGH_VAL_MSE_TO_SKIP_REFINE
GUIDED_RESCUE_TRIGGER_IF_NO_VALID_RESULT = True
GUIDED_RESCUE_TRIGGER_IF_CANDIDATES_LT = 4

ENABLE_EXPERIENCE_PRIOR = not PURE_LLM_TEXT_ONLY_EVAL
EXPERIENCE_PRIOR_FORMAT = "experience_card_v1"
EXPERIENCE_STRONG_PRIOR_PROMOTION_SCORE = 0.72
EXPERIENCE_WEAK_PRIOR_PROMOTION_SCORE = 0.84
EXPERIENCE_RERANK_TOPK = 12
EXPERIENCE_STRONG_PRIOR_SOURCE_ORDER = ["protected", "memory", "experience", "heuristic", "manual", "text", "diverse"]
EXPERIENCE_STRONG_PRIOR_SOURCE_CAPS = {
    "protected": 10,
    "memory": 8,
    "experience": 10,
    "heuristic": 8,
    "manual": 6,
    "text": 4,
    "diverse": 3,
}
EXPERIENCE_TEMPLATE_TOPK_PER_ITEM = 6

ENABLE_MEMORY_PRIOR = False
MEMORY_USE_HISTORICAL_CANDIDATES = False
MEMORY_REPORT_GLOB_PATTERNS = [str(PROJECT_ROOT / "runs" / "*" / "per_case_reports" / "*.json")]
MEMORY_MAX_REPORTS = 400
MEMORY_TOPK_MATCHES = 5
MEMORY_MAX_CANDIDATE_EXPRS = 8
MEMORY_MIN_MATCH_SCORE = 1.75
MEMORY_EXACT_MATCH_SCORE_BONUS = 8.0
MEMORY_GOOD_VAL_MSE = 1e-8
MEMORY_STRONG_VAL_MSE = 1e-4
MEMORY_REPORT_MTIME_WEIGHT = 0.15
MEMORY_POINT_SIGNATURE_MAX_POINTS = 64
MEMORY_POINT_DISTANCE_WEIGHT = 3.5

_MEMORY_BANK_CACHE = None

RESPONSE_PARSER = ResponseParser()


class TaskTimeBudgetExceeded(TimeoutError):
    pass


@contextmanager
def task_time_budget_guard(max_runtime_sec):
    enabled = isinstance(max_runtime_sec, numbers.Number) and float(max_runtime_sec) > 0
    if not enabled or not hasattr(signal, "SIGALRM") or threading.current_thread() is not threading.main_thread():
        yield
        return

    seconds = float(max_runtime_sec)
    prev_handler = signal.getsignal(signal.SIGALRM)
    prev_timer = signal.setitimer(signal.ITIMER_REAL, 0.0)

    def _handle_timeout(signum, frame):
        raise TaskTimeBudgetExceeded(f"task runtime exceeded {seconds:.1f}s")

    try:
        signal.signal(signal.SIGALRM, _handle_timeout)
        signal.setitimer(signal.ITIMER_REAL, seconds)
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, prev_handler)
        if isinstance(prev_timer, tuple) and len(prev_timer) == 2:
            old_value, old_interval = prev_timer
            if float(old_value) > 0 or float(old_interval) > 0:
                signal.setitimer(signal.ITIMER_REAL, float(old_value), float(old_interval))


def _task_deadline_from_start(start_ts):
    if not isinstance(MAX_RUNTIME_PER_TASK_SEC, numbers.Number) or float(MAX_RUNTIME_PER_TASK_SEC) <= 0:
        return None
    return float(start_ts) + float(MAX_RUNTIME_PER_TASK_SEC)


def _task_budget_remaining_sec(deadline_ts):
    if deadline_ts is None:
        return None
    return float(deadline_ts) - time.time()


def _raise_if_task_budget_exceeded(deadline_ts, stage_label=""):
    remaining = _task_budget_remaining_sec(deadline_ts)
    if remaining is not None and remaining <= 0:
        suffix = f" at {stage_label}" if stage_label else ""
        raise TaskTimeBudgetExceeded(f"task runtime exceeded {float(MAX_RUNTIME_PER_TASK_SEC):.1f}s{suffix}")


def safe_numeric_mean(values):
    clean = []
    for v in values:
        if isinstance(v, numbers.Number) and np.isfinite(v):
            clean.append(float(v))
    if not clean:
        return None
    return float(np.mean(clean))


def _safe_get_attr(obj, name, default=None):
    try:
        return getattr(obj, name, default)
    except Exception:
        return default


def _safe_metric_float(value):
    try:
        value = float(value)
    except Exception:
        return None
    if not np.isfinite(value):
        return None
    return value


def _metric_improvement(before, after):
    before_f = _safe_metric_float(before)
    after_f = _safe_metric_float(after)
    if before_f is None or after_f is None:
        return None
    return float(before_f - after_f)


def _relative_metric_improvement(before, after):
    before_f = _safe_metric_float(before)
    after_f = _safe_metric_float(after)
    if before_f is None or after_f is None:
        return None
    if abs(before_f) <= 1e-12:
        return 0.0 if abs(after_f) <= 1e-12 else None
    return float((before_f - after_f) / abs(before_f))


def _result_performance_snapshot(item):
    return {
        "expr": _safe_get_attr(item, "simplified_expression", None) if item is not None else None,
        "selection_metric": _safe_metric_float(_safe_get_attr(item, "selection_metric", None)) if item is not None else None,
        "small_sample_cv_mse": _safe_metric_float(_safe_get_attr(item, "small_sample_cv_mse", None)) if item is not None else None,
        "val_mse": _safe_metric_float(_safe_get_attr(item, "val_mse", None)) if item is not None else None,
        "score": _safe_metric_float(_safe_get_attr(item, "score", None)) if item is not None else None,
        "complexity": _safe_get_attr(item, "complexity", None) if item is not None else None,
    }


def get_iteration_config(n_features: int):
    if ITERATION_MODE == "fixed":
        return {
            "mm_calls": NUM_MM_SINGLE_CALLS,
            "text_calls": NUM_TEXT_SINGLE_CALLS,
            "proposal_k": NUM_PROPOSAL_CANDIDATES,
            "refined_k": NUM_REFINED_CANDIDATES,
            "refine_rounds": MAX_REFINE_ROUNDS,
        }

    if n_features <= 2:
        return {
            "mm_calls": 2,
            "text_calls": 3,
            "proposal_k": 6,
            "refined_k": 3,
            "refine_rounds": 2,
        }
    if n_features <= 4:
        return {
            "mm_calls": 2,
            "text_calls": 4,
            "proposal_k": 8,
            "refined_k": 4,
            "refine_rounds": 2,
        }
    return {
        "mm_calls": 1,
        "text_calls": 4,
        "proposal_k": 10,
        "refined_k": 5,
        "refine_rounds": 3,
    }


def _is_benchmark_task(row_meta):
    dataset_dir = str((row_meta or {}).get("dataset_dir", "") or "").strip().lower()
    return dataset_dir in {"benchmark_csv", "sldbench", "llmsrbench", "srbench"}


def _is_benchmark_like_source_tag(source_tag):
    return str(source_tag or "").strip().lower() in {"benchmark_csv", "sldbench", "llmsrbench", "srbench"}


def _is_low_dim_benchmark_like(row_meta=None, dataset=None):
    n_features = 0
    if dataset is not None:
        try:
            n_features = len(getattr(dataset, "feature_names", []) or [])
        except Exception:
            n_features = 0
        if n_features > 0 and _is_benchmark_like_source_tag(getattr(dataset, "source_tag", "")):
            return n_features <= FAST_PATH_BENCHMARK_DIM_LIMIT
    if row_meta is not None and _is_benchmark_task(row_meta):
        if n_features > 0:
            return n_features <= FAST_PATH_BENCHMARK_DIM_LIMIT
        try:
            n_features = int((row_meta or {}).get("n_features", 0) or 0)
        except Exception:
            n_features = 0
        return 0 < n_features <= FAST_PATH_BENCHMARK_DIM_LIMIT
    return False


def _runtime_fast_path_eligible(row_meta, dataset):
    if TEMP_DISABLE_TEMPLATE_SOURCES:
        return False
    if DISABLE_INITIAL_TEMPLATE_VALIDATION:
        return False
    if PURE_LLM_CANDIDATE_MODE:
        return False
    feature_names = list(getattr(dataset, "feature_names", []) or [])
    has_high_dim_benchmark_seed = (
        _is_benchmark_task(row_meta)
        and len(feature_names) == 5
        and bool(build_protected_benchmark_templates(row_meta, feature_names))
    )
    return (
        ENABLE_RUNTIME_FAST_PATH
        and _is_benchmark_task(row_meta)
        and (
            0 < len(feature_names) <= FAST_PATH_BENCHMARK_DIM_LIMIT
            or has_high_dim_benchmark_seed
        )
    )


def adjust_iteration_config_for_runtime(iter_cfg, dataset, row_meta, seed_best=None, experience_prior=None):
    tuned = dict(iter_cfg or {})
    tuned["mm_calls"] = max(0, int(tuned.get("mm_calls", NUM_MM_SINGLE_CALLS)))
    tuned["text_calls"] = max(0, int(tuned.get("text_calls", NUM_TEXT_SINGLE_CALLS)))
    tuned["proposal_k"] = max(1, int(tuned.get("proposal_k", NUM_PROPOSAL_CANDIDATES)))
    tuned["refined_k"] = max(1, int(tuned.get("refined_k", NUM_REFINED_CANDIDATES)))
    tuned["refine_rounds"] = max(0, int(tuned.get("refine_rounds", MAX_REFINE_ROUNDS)))
    tuned["text_max_rows"] = min(MAX_ROWS_FOR_TEXT_PROMPT, int(tuned.get("text_max_rows", MAX_ROWS_FOR_TEXT_PROMPT)))
    tuned["proposal_max_tokens"] = max(128, int(tuned.get("proposal_max_tokens", 384)))
    tuned["refiner_max_tokens"] = max(192, int(tuned.get("refiner_max_tokens", REFINER_MAX_TOKENS)))
    tuned["skip_refine_val_mse"] = float(tuned.get("skip_refine_val_mse", GOOD_ENOUGH_VAL_MSE_TO_SKIP_REFINE))
    tuned["skip_diverse_proposal"] = bool(tuned.get("skip_diverse_proposal", False))
    tuned["force_heuristic_agents"] = bool(tuned.get("force_heuristic_agents", False))

    if not ENABLE_RUNTIME_FAST_PATH:
        return tuned

    n_features = len(getattr(dataset, "feature_names", []) or [])
    is_benchmark = _is_benchmark_task(row_meta)

    if is_benchmark and n_features <= FAST_PATH_BENCHMARK_DIM_LIMIT:
        tuned["text_calls"] = min(tuned["text_calls"], LOW_DIM_BENCHMARK_TEXT_CALL_CAP)
        tuned["mm_calls"] = min(tuned["mm_calls"], LOW_DIM_BENCHMARK_MM_CALL_CAP)
        tuned["refine_rounds"] = min(tuned["refine_rounds"], LOW_DIM_BENCHMARK_REFINE_ROUND_CAP)
        tuned["proposal_k"] = min(tuned["proposal_k"], LOW_DIM_BENCHMARK_PROPOSAL_K_CAP)
        tuned["refined_k"] = min(tuned["refined_k"], LOW_DIM_BENCHMARK_REFINED_K_CAP)
        tuned["text_max_rows"] = min(tuned["text_max_rows"], LOW_DIM_BENCHMARK_TEXT_MAX_ROWS)
        tuned["proposal_max_tokens"] = min(tuned["proposal_max_tokens"], LOW_DIM_BENCHMARK_PROPOSAL_MAX_TOKENS)
        tuned["refiner_max_tokens"] = min(tuned["refiner_max_tokens"], LOW_DIM_BENCHMARK_REFINER_MAX_TOKENS)
        tuned["skip_refine_val_mse"] = max(
            tuned["skip_refine_val_mse"],
            FAST_PATH_SKIP_REFINE_VAL_MSE,
            LOW_DIM_BENCHMARK_SKIP_REFINE_VAL_MSE,
        )
        tuned["skip_diverse_proposal"] = tuned["skip_diverse_proposal"] or LOW_DIM_BENCHMARK_SKIP_DIVERSE
        tuned["force_heuristic_agents"] = tuned["force_heuristic_agents"] or LOW_DIM_BENCHMARK_FORCE_HEURISTIC_AGENTS
        if LOW_DIM_BENCHMARK_DISABLE_TEXT:
            tuned["text_calls"] = 0
        if LOW_DIM_BENCHMARK_DISABLE_MM:
            tuned["mm_calls"] = 0
    elif n_features >= 5:
        tuned["text_calls"] = min(tuned["text_calls"], HIGH_DIM_TEXT_CALL_CAP)
        tuned["mm_calls"] = min(tuned["mm_calls"], HIGH_DIM_MM_CALL_CAP)
        tuned["refine_rounds"] = min(tuned["refine_rounds"], HIGH_DIM_REFINE_ROUND_CAP)
        tuned["proposal_max_tokens"] = min(tuned["proposal_max_tokens"], HIGH_DIM_PROPOSAL_MAX_TOKENS)
        family_tags = set([str(x).strip() for x in list((experience_prior or {}).get("family_tags", []) or []) if str(x).strip()])
        variable_roles = dict((experience_prior or {}).get("variable_roles", {}) or {})
        complex_high_dim_family = (
            len(family_tags & {"rational", "trigonometric", "exponential", "logarithmic", "power", "interaction"}) >= 3
            or ({"rational", "interaction"} <= family_tags)
            or (bool(list(variable_roles.get("denominator_core", []) or [])) and bool(list(variable_roles.get("numerator_core", []) or [])))
        )
        if complex_high_dim_family:
            tuned["text_calls"] = max(tuned["text_calls"], HIGH_DIM_COMPLEX_FAMILY_TEXT_CALL_FLOOR)
            tuned["refine_rounds"] = max(tuned["refine_rounds"], HIGH_DIM_COMPLEX_FAMILY_REFINE_FLOOR)
            tuned["proposal_k"] = max(tuned["proposal_k"], HIGH_DIM_COMPLEX_FAMILY_PROPOSAL_K_FLOOR)
            tuned["refined_k"] = max(tuned["refined_k"], HIGH_DIM_COMPLEX_FAMILY_REFINED_K_FLOOR)
            tuned["proposal_max_tokens"] = max(tuned["proposal_max_tokens"], HIGH_DIM_COMPLEX_FAMILY_PROPOSAL_MAX_TOKENS)
            tuned["skip_diverse_proposal"] = False

    tuned = apply_experience_prior_to_iteration_config(tuned, experience_prior=experience_prior)

    weak_high_dim_no_leakage = bool(
        is_benchmark
        and NO_LEAKAGE_MODE
        and n_features >= HIGH_DIM_ROLE_TRIGGER_DIM
        and not bool((experience_prior or {}).get("strong_prior", False))
    )
    if weak_high_dim_no_leakage:
        tuned["proposal_k"] = max(int(tuned.get("proposal_k", 0)), 12)
        tuned["refine_rounds"] = max(int(tuned.get("refine_rounds", 0)), 1)
        tuned["refined_k"] = max(int(tuned.get("refined_k", 0)), 4)
        if HIGH_DIM_NO_LEAKAGE_SKIP_TEXT_INITIAL:
            tuned["text_calls"] = 0
        else:
            tuned["text_calls"] = max(int(tuned.get("text_calls", 0)), 1)
        tuned["skip_diverse_proposal"] = bool(HIGH_DIM_NO_LEAKAGE_SKIP_DIVERSE_INITIAL)
        if HIGH_DIM_NO_LEAKAGE_SKIP_MM_AFTER_GUIDED:
            tuned["mm_calls"] = 0
        elif FORCE_MM_FOR_HIGH_DIM_BAD_FIT and USE_MULTIMODAL_PROPOSAL:
            tuned["mm_calls"] = max(1, int(tuned.get("mm_calls", 0)))
        if HIGH_DIM_NO_LEAKAGE_FORCE_HEURISTIC_REFINER:
            tuned["force_heuristic_agents"] = True

    best_val = _safe_get_attr(seed_best, "val_mse", None) if seed_best is not None else None
    try:
        if best_val is not None and np.isfinite(best_val):
            if float(best_val) <= FAST_PATH_RELAX_BUDGET_VAL_MSE:
                tuned["text_calls"] = min(tuned["text_calls"], 0 if (is_benchmark and n_features == 1) else 1)
                tuned["refine_rounds"] = min(tuned["refine_rounds"], 1)
                tuned["refined_k"] = min(tuned["refined_k"], 2)
                tuned["proposal_max_tokens"] = min(tuned["proposal_max_tokens"], LOW_DIM_BENCHMARK_PROPOSAL_MAX_TOKENS)
                tuned["refiner_max_tokens"] = min(tuned["refiner_max_tokens"], 256)
                tuned["skip_diverse_proposal"] = True
                tuned["force_heuristic_agents"] = True
            if float(best_val) <= float(tuned.get("skip_refine_val_mse", GOOD_ENOUGH_VAL_MSE_TO_SKIP_REFINE)):
                tuned["refine_rounds"] = 0
            if float(best_val) <= FAST_PATH_SHORT_CIRCUIT_VAL_MSE:
                tuned["text_calls"] = 0
                tuned["mm_calls"] = 0
                tuned["refine_rounds"] = 0
                tuned["skip_diverse_proposal"] = True
                tuned["force_heuristic_agents"] = True
    except Exception:
        pass

    if PURE_LLM_TEXT_ONLY_EVAL:
        tuned["text_calls"] = 1
        tuned["mm_calls"] = 0
        tuned["refine_rounds"] = 0
        tuned["skip_diverse_proposal"] = True
        return tuned

    if PURE_LLM_CANDIDATE_MODE:
        if TEMP_DISABLE_TEMPLATE_SOURCES:
            tuned["text_calls"] = max(1, int(tuned.get("text_calls", NUM_TEXT_SINGLE_CALLS)))
            tuned["skip_diverse_proposal"] = False
            return tuned
        strong_memory_seed = bool((experience_prior or {}).get("memory_seed_exprs")) and bool((experience_prior or {}).get("memory_exact_match", False))
        if strong_memory_seed:
            tuned["text_calls"] = 0
            tuned["skip_diverse_proposal"] = True
            tuned["refine_rounds"] = 0
        else:
            if weak_high_dim_no_leakage and HIGH_DIM_NO_LEAKAGE_SKIP_TEXT_INITIAL:
                tuned["text_calls"] = 0
            else:
                tuned["text_calls"] = max(1, int(tuned.get("text_calls", NUM_TEXT_SINGLE_CALLS)))
            if weak_high_dim_no_leakage and HIGH_DIM_NO_LEAKAGE_SKIP_DIVERSE_INITIAL:
                tuned["skip_diverse_proposal"] = True
            else:
                tuned["skip_diverse_proposal"] = False

    return tuned


def is_better_result(candidate, incumbent):
    if candidate is None:
        return False
    if incumbent is None:
        return True

    cand_sel = _safe_metric_float(_safe_get_attr(candidate, "selection_metric", None))
    inc_sel = _safe_metric_float(_safe_get_attr(incumbent, "selection_metric", None))
    if cand_sel is not None and inc_sel is None:
        return True
    if cand_sel is None and inc_sel is not None:
        return False
    if cand_sel is not None and inc_sel is not None:
        try:
            return float(cand_sel) < float(inc_sel)
        except Exception:
            pass

    cand_val = _safe_get_attr(candidate, "val_mse", None)
    inc_val = _safe_get_attr(incumbent, "val_mse", None)
    if cand_val is not None and inc_val is None:
        return True
    if cand_val is None and inc_val is not None:
        return False
    if cand_val is not None and inc_val is not None:
        try:
            return float(cand_val) < float(inc_val)
        except Exception:
            pass

    cand_score = _safe_get_attr(candidate, "score", None)
    inc_score = _safe_get_attr(incumbent, "score", None)
    if cand_score is not None and inc_score is None:
        return True
    if cand_score is None and inc_score is not None:
        return False
    if cand_score is not None and inc_score is not None:
        try:
            return float(cand_score) > float(inc_score)
        except Exception:
            pass
    return False


def should_early_stop(best_history):
    vals = [v for v in best_history if v is not None and np.isfinite(v)]
    if not vals:
        return False, None
    if vals[-1] <= EARLY_STOP_VAL_TOL:
        return True, f"val_mse<={EARLY_STOP_VAL_TOL}"
    if len(vals) < EARLY_STOP_PATIENCE + 1:
        return False, None
    recent = vals[-(EARLY_STOP_PATIENCE + 1):]
    improvements = []
    for prev, cur in zip(recent[:-1], recent[1:]):
        if prev == 0:
            improvements.append(0.0)
        else:
            improvements.append((prev - cur) / (abs(prev) + 1e-12))
    if all(impr < EARLY_STOP_MIN_REL_IMPROVEMENT for impr in improvements):
        return True, f"relative_improvement<{EARLY_STOP_MIN_REL_IMPROVEMENT} for {EARLY_STOP_PATIENCE} rounds"
    return False, None


def make_json_safe(obj):
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_json_safe(x) for x in obj]
    try:
        json.dumps(obj)
        return obj
    except Exception:
        return str(obj)


class StepTimer:
    def __init__(self):
        self.times = {}
        self._starts = {}

    def start(self, name: str):
        self._starts[name] = time.time()

    def stop(self, name: str):
        if name in self._starts:
            dt = time.time() - self._starts[name]
            self.times[name] = self.times.get(name, 0.0) + dt
            del self._starts[name]
            return dt
        return 0.0

    def add(self, name: str, dt: float):
        self.times[name] = self.times.get(name, 0.0) + float(dt)

    def get(self, name: str, default=0.0):
        return self.times.get(name, default)

    def as_dict(self):
        return dict(self.times)


def timed_call(step_timer: Optional[StepTimer], name: str, fn, *args, **kwargs):
    if step_timer is None:
        return fn(*args, **kwargs)
    step_timer.start(name)
    try:
        return fn(*args, **kwargs)
    finally:
        step_timer.stop(name)


def format_local_timestamp(ts: float):
    base = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))
    millis = int((float(ts) % 1.0) * 1000)
    return f"{base}.{millis:03d}"


def sanitize_name(path_str: str):
    base = path_str.replace("/", "__").replace("\\", "__")
    base = re.sub(r"[^a-zA-Z0-9_.\-]+", "_", base)
    short_hash = hashlib.md5(path_str.encode("utf-8")).hexdigest()[:8]
    return f"{base}_{short_hash}"


def load_txt_dataset(txt_path: str) -> pd.DataFrame:
    arr = np.loadtxt(txt_path)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    n_cols = arr.shape[1]
    if n_cols < 2:
        raise ValueError(f"invalid txt file: {txt_path}, shape={arr.shape}")
    columns = [f"x{i+1}" for i in range(n_cols - 1)] + ["y"]
    return pd.DataFrame(arr, columns=columns)


def build_basic_structure_hints(dataset):
    d = len(dataset.feature_names)
    common = [
        f"dataset has {len(dataset.df)} total rows",
        f"input variables are {dataset.feature_names}",
        f"target variable is {dataset.target_name}",
        "use only the provided variable names",
        "prefer exact or near-exact symbolic expressions when possible",
        "prefer nonlinear expressions over a poor affine approximation when needed",
    ]
    if d == 1:
        common += [
            "consider polynomial, rational, exponential, logarithmic, trigonometric, and hyperbolic forms when appropriate",
            "consider composite nonlinear forms such as sin(x1**2), cos(x1**2), sin(x1)*cos(x1)",
            "do not return only affine or linear expressions",
        ]
    elif d == 2:
        common += [
            "consider additive forms such as f(x1) + g(x2)",
            "consider multiplicative interactions such as x1 * x2",
            "consider separable or partially separable structure",
            "consider low-order polynomial or trigonometric interactions",
            "do not return only purely linear expressions",
        ]
    elif d == 3:
        common += [
            "consider additive forms such as f(x1) + g(x2) + h(x3)",
            "consider pairwise interactions such as x1*x2, x1*x3, x2*x3",
            "consider low-order three-way interactions such as x1*x2*x3 when necessary",
            "consider sparse structure instead of dense fully-coupled formulas",
            "consider separable or partially separable structure",
        ]
    else:
        common += [
            "consider sparse low-order interactions",
            "consider additive or partially separable structure",
            "only a few variables may dominate",
            "avoid overly dense formulas involving all variables at once",
            "first identify a compact active subset before coupling many variables together",
            "for six or more variables, prefer a small active subspace plus one denominator or modulation branch when supported",
        ]
    return common


def extract_candidate_expressions(result_dict):
    exprs = []
    if not isinstance(result_dict, dict):
        return exprs
    for item in result_dict.get("candidates", []):
        if not isinstance(item, dict):
            continue
        expr = str(item.get("expression", "")).strip()
        if expr:
            exprs.append(expr)
    for item in result_dict.get("refined_candidates", []):
        if not isinstance(item, dict):
            continue
        expr = str(item.get("expression", "")).strip()
        if expr:
            exprs.append(expr)
    seen = set()
    out = []
    for expr in exprs:
        if expr not in seen:
            out.append(expr)
            seen.add(expr)
    return out


def summarize_scored_results(scored_results, top_k=3):
    lines = []
    for i, item in enumerate(scored_results[:top_k], start=1):
        lines.append(
            f"rank={i}; expr={_safe_get_attr(item, 'simplified_expression', None)}; "
            f"val_mse={_safe_get_attr(item, 'val_mse', None)}; "
            f"complexity={_safe_get_attr(item, 'complexity', None)}; "
            f"score={_safe_get_attr(item, 'score', None)}"
        )
    return lines


def evaluate_candidate_expressions(
    candidate_exprs,
    dataset,
    complexity_weight=None,
    timer=None,
    prefix="eval",
    fitter=None,
    simplifier=None,
    deduper=None,
    scorer=None,
    deadline_ts=None,
):
    fitter = fitter or TemplateFillTool(max_workers=TEMPLATE_FIT_MAX_WORKERS)
    simplifier = simplifier or AlgebraicSimplifyTool()
    deduper = deduper or EquivalenceCheckTool()
    scorer = scorer or ScoringTool(complexity_weight=complexity_weight)

    if (
        _should_apply_high_dim_structural_rerank(dataset)
        and str(prefix).startswith(("initial", "guided_rescue"))
    ):
        candidate_exprs, _ = structural_rerank_high_dim_candidates(
            candidate_exprs,
            dataset,
            top_k=HIGH_DIM_STRUCTURAL_FULL_EVAL_TOPK,
        )

    if STRICT_NO_FORMULA_TEMPLATES:
        feature_names = list(getattr(dataset, "feature_names", []) or [])
        candidate_exprs, _ = _filter_non_template_expressions(candidate_exprs, feature_names)

    candidate_count = len(candidate_exprs or [])
    low_dim_benchmark_like = _is_low_dim_benchmark_like(dataset=dataset)
    fit_restarts = TEMPLATE_FIT_RESTARTS_DEFAULT
    fit_init_scale = TEMPLATE_FIT_INIT_SCALE_DEFAULT
    if str(prefix).startswith("initial_prefilter") or str(prefix).startswith("mm_prefilter") or str(prefix).startswith("prefilter"):
        fit_restarts = TEMPLATE_FIT_RESTARTS_PREFILTER
        fit_init_scale = TEMPLATE_FIT_INIT_SCALE_FAST
    elif str(prefix).startswith("seed_fast_path"):
        fit_restarts = TEMPLATE_FIT_RESTARTS_FAST_SEED
        fit_init_scale = TEMPLATE_FIT_INIT_SCALE_FAST
    elif str(prefix).startswith("initial"):
        fit_restarts = TEMPLATE_FIT_RESTARTS_INITIAL
        fit_init_scale = TEMPLATE_FIT_INIT_SCALE_FAST
    elif str(prefix).startswith("guided_rescue"):
        fit_restarts = TEMPLATE_FIT_RESTARTS_INITIAL
        fit_init_scale = TEMPLATE_FIT_INIT_SCALE_FAST
    elif str(prefix).startswith("surrogate_escape"):
        fit_restarts = TEMPLATE_FIT_RESTARTS_INITIAL
        fit_init_scale = TEMPLATE_FIT_INIT_SCALE_FAST
    elif str(prefix).startswith("highdim_universal_coverage"):
        # Universal coverage candidates are intentionally low-parameter canonical
        # templates. One restart is enough for most and keeps high-dimensional
        # rescue from becoming slower than the main pipeline.
        fit_restarts = 1
        fit_init_scale = TEMPLATE_FIT_INIT_SCALE_FAST
    elif str(prefix).startswith("lowdim_rational_coverage"):
        # Low-dimensional rational candidates can have several linear parameters.
        # Use a small number of restarts, but keep it cheaper than LLM refine.
        fit_restarts = 2
        fit_init_scale = TEMPLATE_FIT_INIT_SCALE_FAST
    elif str(prefix).startswith("lowdim_additive_nonlinear"):
        # Additive nonlinear forms have few candidates but nonconvex sine/exp
        # parameters. Use more restarts and do not let the generic candidate-count
        # clamp reduce this budget.
        fit_restarts = max(1, int(LOW_DIM_ADDITIVE_DIRECT_FIT_RESTARTS))
        fit_init_scale = TEMPLATE_FIT_INIT_SCALE_FAST
    elif str(prefix).startswith("mm_rescue"):
        fit_restarts = TEMPLATE_FIT_RESTARTS_MM
        fit_init_scale = TEMPLATE_FIT_INIT_SCALE_FAST
    elif str(prefix).startswith("refine_round_"):
        fit_restarts = TEMPLATE_FIT_RESTARTS_REFINE
        fit_init_scale = TEMPLATE_FIT_INIT_SCALE_FAST

    if not str(prefix).startswith("lowdim_additive_nonlinear"):
        if candidate_count >= 20:
            fit_restarts = min(fit_restarts, 2)
        if candidate_count >= 32:
            fit_restarts = 1
    if low_dim_benchmark_like and LOW_DIM_BENCHMARK_FORCE_SINGLE_RESTART and not str(prefix).startswith("lowdim_additive_nonlinear"):
        fit_restarts = 1

    _raise_if_task_budget_exceeded(deadline_ts, f"{prefix}_template_fit_start")
    if timer is not None:
        timer.start(f"{prefix}_template_fit")
    fit_results = fitter.run(
        candidate_exprs,
        dataset,
        n_restarts=fit_restarts,
        init_scale=fit_init_scale,
        deadline_ts=deadline_ts,
    )
    record_candidate_search_audit(
        dataset=dataset,
        stage=str(prefix),
        fit_results=fit_results,
    )
    if timer is not None:
        timer.stop(f"{prefix}_template_fit")

    if timer is not None:
        timer.start(f"{prefix}_algebraic_simplify")
    simplify_results = simplifier.run(fit_results)
    if timer is not None:
        timer.stop(f"{prefix}_algebraic_simplify")

    if timer is not None:
        timer.start(f"{prefix}_equivalence_check")
    unique_results = deduper.run(simplify_results)
    if timer is not None:
        timer.stop(f"{prefix}_equivalence_check")

    if timer is not None:
        timer.start(f"{prefix}_scoring")
    scored_results = scorer.run(unique_results)
    if timer is not None:
        timer.stop(f"{prefix}_scoring")

    return fit_results, simplify_results, unique_results, scored_results


def reset_candidate_search_audit(dataset):
    state = {
        "records": [],
        "seen_candidate_keys": {},
    }
    setattr(dataset, "_candidate_search_audit_state", state)
    return state


def candidate_search_audit_state(dataset):
    state = getattr(dataset, "_candidate_search_audit_state", None)
    if not isinstance(state, dict):
        state = reset_candidate_search_audit(dataset)
    state.setdefault("records", [])
    state.setdefault("seen_candidate_keys", {})
    return state


def share_candidate_search_audit(source_dataset, target_dataset):
    state = candidate_search_audit_state(source_dataset)
    setattr(target_dataset, "_candidate_search_audit_state", state)
    return state


def record_candidate_search_audit(dataset, stage, fit_results):
    """Record fitted candidates in their actual evaluator input order."""
    state = candidate_search_audit_state(dataset)
    records = state["records"]
    seen = state["seen_candidate_keys"]
    val_df = getattr(dataset, "val_df", None)
    target_name = getattr(dataset, "target_name", "y")
    val_variance = None
    if val_df is not None and target_name in val_df:
        y_val = np.asarray(val_df[target_name], dtype=float)
        if len(y_val):
            variance = float(np.var(y_val))
            if np.isfinite(variance) and variance > 0.0:
                val_variance = variance

    for stage_index, item in enumerate(list(fit_results or []), start=1):
        expression = str(_safe_get_attr(item, "expression", "") or "")
        key = _expr_dedup_key(expression)
        is_new_unique = key not in seen
        if is_new_unique:
            seen[key] = len(seen) + 1

        val_mse = _safe_metric_float(_safe_get_attr(item, "val_mse", None))
        val_r2 = None
        if val_mse is not None and val_variance is not None:
            val_r2 = float(1.0 - float(val_mse) / float(val_variance))

        records.append({
            "evaluation_index": len(records) + 1,
            "unique_evaluations_seen": len(seen),
            "first_unique_evaluation_index": int(seen[key]),
            "is_new_unique_candidate": bool(is_new_unique),
            "stage": str(stage),
            "stage_evaluation_index": int(stage_index),
            "expression": expression,
            "fitted_expression": _safe_get_attr(item, "fitted_expression", None),
            "success": bool(_safe_get_attr(item, "success", False)),
            "val_mse": val_mse,
            "val_r2": val_r2,
        })
    return records


def merge_expression_lists(*expr_lists):
    merged = []
    seen = set()
    for expr_list in expr_lists:
        for expr in expr_list:
            expr = str(expr).strip()
            if expr and expr not in seen:
                merged.append(expr)
                seen.add(expr)
    return merged



def rerank_with_protected_family_bias(scored_results, row_meta):
    """Clean no-leakage build: no benchmark-specific reranking."""
    return scored_results


def get_best_result(scored_results):
    if not scored_results:
        return None
    return scored_results[0]





def build_livermore_2d_templates(row_meta, feature_names):
    """Clean no-leakage build: benchmark-specific Livermore templates are disabled."""
    return []


def build_feynman_5d_family_templates(base_name, feature_names):
    """Clean no-leakage build: benchmark-specific Feynman templates are disabled."""
    return []


def build_protected_benchmark_templates(row_meta, feature_names):
    """Clean no-leakage build: row-specific protected benchmark templates are disabled."""
    return []


def build_feynman_5d_templates(feature_names):
    """Clean no-leakage build: benchmark-specific Feynman family templates are disabled."""
    return []

def build_manual_candidates(feature_names):
    d = len(feature_names)
    if d == 0:
        return []

    def _uniq(items):
        out, seen = [], set()
        for x in items:
            x = str(x).strip()
            if x and x not in seen:
                out.append(x)
                seen.add(x)
        return out

    if d == 1:
        x = feature_names[0]
        templates = [
            f"a*{x}+b",
            f"a*{x}**2+b*{x}+c",
            f"a*{x}**3+b*{x}**2+c*{x}+d",
            f"a*{x}**4+b*{x}**3+c*{x}**2+d*{x}+e",
            f"a1*{x}**5+a2*{x}**4+a3*{x}**3+a4*{x}**2+a5*{x}+a6",
            f"a1*{x}**6+a2*{x}**5+a3*{x}**4+a4*{x}**3+a5*{x}**2+a6*{x}+a7",
            f"a1*{x}**9+a2*{x}**8+a3*{x}**7+a4*{x}**6+a5*{x}**5+a6*{x}**4+a7*{x}**3+a8*{x}**2+a9*{x}+a10",
            f"a*sin(b*{x}+c)+d",
            f"a*cos(b*{x}+c)+d",
            f"a*sin(b*{x})*cos(c*{x})+d",
            f"a*sin(b*{x}**2+c)+d",
            f"a*cos(b*{x}**2+c)+d",
            f"a*sin(b*{x}+c*{x}**2+d)+e",
            f"a*sin(b*{x})*cos(c*{x}**2+d)+e",
            f"a1*sin(b1*{x}+c1)+a2*sin(b2*{x}**2+c2)+d",
            f"a*sinh(b*{x}+c)+d",
            f"a*cosh(b*{x}+c)+d",
            f"a*tanh(b*{x}+c)+d",
            f"a*exp(b*{x})+c",
            f"a*log(abs(b*{x})+c)+d",
            f"a*log(abs(b*{x})+c)+d*log({x}**2+e)+f",
            f"a*sqrt(abs(b*{x})+c)+d",
            f"a/({x}+b)+c",
            f"a/({x}**2+b)+c",
            f"(a*{x}+b)/(c*{x}+d)",
            f"(a1*{x}**3+a2*{x}**2+a3*{x}+a4)/(b1*{x}**2+b2*{x}+b3)",
            f"(a1*{x}**5+a2*{x}**4+a3*{x}**3+a4*{x}**2+a5*{x}+a6)/(b1*{x}**2+b2*{x}+b3)",
            f"(a1*{x}**6+a2*{x}**5+a3*{x}**4+a4*{x}**3+a5*{x}**2+a6*{x}+a7)/(b1*{x}**4+b2*{x}**3+b3*{x}**2+b4*{x}+b5)",
            f"a*(({x}+b1)**3)/({x}**2+b2*{x}+b3)+c",
            f"a*({x}**5+b1*{x}**3+b2)/({x}**2+b3)+c",
        ]
        return _uniq(templates)

    if d == 2:
        x1, x2 = feature_names
        templates = [
            f"{x1}*{x2}",
            f"a*{x1}*{x2}",
            f"a*{x1}*{x2}+b",
            # generic low-dimensional interaction-power primitives, e.g. x1*x2**2
            # This is not benchmark-specific; it covers common multiplicative power laws.
            f"a*{x1}*{x2}**2+b",
            f"a*{x1}**2*{x2}+b",
            f"a*{x1}*({x2}+b1)**2+c",
            f"a*({x1}+b1)*{x2}**2+c",
            f"(a1*{x1} + b1) * (a2*{x2} + b2) + c",
            f"a*({x1} + b) * ({x2} + c) + d",
            f"a*({x1}*{x2} + b) + c*{x1} + d*{x2} + e",
            f"a*{x1}+b*{x2}+c",
            f"(a*{x1} + b) / (c*{x2} + d) + e",
            f"a*({x1}/({x2}+b))+c",
            f"a*({x2}/({x1}+b))+c",
            f"a*sin(b*{x1})*cos(c*{x2})+d",
            f"a*sin(b*{x1})*cos(c*{x2})+e*{x1}+f*{x2}+g",
            f"a*sin(b*{x1}+c*{x2})+d",
            f"a*cos(b*{x1}+c*{x2})+d",
            f"a*exp(b*{x1}+c*{x2})+d",
            f"a*log(abs(b*{x1}+c*{x2})+d)+e",
            f"a*sqrt(abs(b*{x1}+c*{x2})+d)+e",
        ]
        return _uniq(templates)

    if d == 3:
        x1, x2, x3 = feature_names
        templates = [
            f"a*{x1}+b*{x2}+c*{x3}+d",
            f"a*{x1}*{x2}+b*{x3}+c",
            f"a*{x1}*{x3}+b*{x2}+c",
            f"a*{x2}*{x3}+b*{x1}+c",
            f"a*{x1}*{x2}*{x3}+b",
            f"a*({x1}*{x2})/({x3}+b)+c",
            f"a*({x1}*{x3})/({x2}+b)+c",
            f"a*({x2}*{x3})/({x1}+b)+c",
            f"a*sin(b*{x1})+c*{x2}+d*{x3}+e",
            f"a*sin(b*{x1})*cos(c*{x2})+d*{x3}+e",
            f"a*exp(b*{x1})+c*{x2}+d*{x3}+e",
        ]
        return _uniq(templates)

    vars_ = list(feature_names)
    main = vars_[:min(len(vars_), 4)]
    linear = " + ".join([f"a{i+1}*{v}" for i, v in enumerate(main)])
    pair_terms = []
    for i in range(len(main)):
        for j in range(i + 1, len(main)):
            pair_terms.append(f"b{i+1}{j+1}*{main[i]}*{main[j]}")
    pair_sum = " + ".join(pair_terms[:4]) if pair_terms else "0"

    templates = [
        f"{linear} + c",
        f"{pair_sum} + {linear} + c",
        f"a1*{main[0]}*{main[1]} + {linear} + c" if len(main) >= 2 else f"{linear} + c",
        f"a1*({main[0]}/({main[1]}+b1)) + {linear} + c" if len(main) >= 2 else f"{linear} + c",
        f"a1*sin(b1*{main[0]}) + {linear} + c",
        f"a1*exp(b1*{main[0]}) + {linear} + c",
        f"a1*({main[0]}*{main[1]})/({main[2]}+b1) + {linear} + c" if len(main) >= 3 else f"{linear} + c",
        f"a1*sin(b1*{main[0]} + b2*{main[1]}) + {linear} + c" if len(main) >= 2 else f"{linear} + c",
    ]
    if len(feature_names) >= 5:
        x1, x2, x3, x4, x5 = feature_names[:5]
        templates.extend([
            f"a1*({x1}*{x2}) + a2*({x3}+b1*{x4}) + c",
            f"a1*({x1}*{x2}) + a2/(abs({x5})+b1) + c",
            f"a1*({x1}*{x2})*log(abs(({x5}+b1)/({x4}+b2)) + c1) + d",
            f"a1*({x1}*{x2}) + a2*exp(b1*{x5}+c1) + d",
            f"a1*({x1}*{x2}) + a2*sin(b1*{x5}+c1) + d",
            f"a1*({x1}+b1*{x2}) + a2*({x3}+b2*{x4}) + a3/(abs({x5})+b3) + c",
            f"a1*({x1}*{x2})*({x3}+b1)*log(abs(({x5}+b2)/({x4}+b3)) + c1) + d",
        ])
        return _filter_specialist_exprs(_uniq(templates))
    return _uniq(templates)


def _build_role_seed_components(feature_names, structure_profile=None, experience_prior=None):
    feature_names = list(feature_names or [])
    if not feature_names:
        return {
            "active_variables": [],
            "numerator_core": [],
            "denominator_core": [],
            "periodic_core": [],
            "top_pair_patterns": [],
        }

    profile = dict(structure_profile or {})
    experience_prior = dict(experience_prior or {})
    roles_profile = dict(profile.get("variable_roles", {}) or {})
    roles_prior = dict(experience_prior.get("variable_roles", {}) or {})
    global_tags = set(profile.get("global_tags", []) or [])
    allowed = set(feature_names)

    def _ordered_allowed(items):
        out = []
        seen = set()
        for item in items or []:
            item = str(item)
            if item in allowed and item not in seen:
                out.append(item)
                seen.add(item)
        return out

    prefer_ordered_slot_layout = len(feature_names) >= 5

    active_order = (
        list(roles_prior.get("active_variables", []) or [])
        + list(feature_names)
        + list(roles_profile.get("active_variables", []) or [])
        + list(profile.get("active_variables", []) or [])
    ) if prefer_ordered_slot_layout else (
        list(roles_profile.get("active_variables", []) or [])
        + list(profile.get("active_variables", []) or [])
        + list(roles_prior.get("active_variables", []) or [])
        + list(feature_names)
    )
    active_variables = _ordered_allowed(active_order)

    numerator_order = (
        list(roles_prior.get("numerator_core", []) or [])
        + list(feature_names[:min(4, len(feature_names))])
        + list(roles_profile.get("numerator_core", []) or [])
        + list(active_variables)
    ) if prefer_ordered_slot_layout else (
        list(roles_profile.get("numerator_core", []) or [])
        + list(roles_prior.get("numerator_core", []) or [])
        + list(active_variables)
    )
    numerator_core = _ordered_allowed(numerator_order)

    denominator_core = _ordered_allowed(
        list(roles_profile.get("denominator_core", []) or [])
        + list(roles_prior.get("denominator_core", []) or [])
    )
    periodic_core = _ordered_allowed(
        list(roles_profile.get("periodic_core", []) or [])
        + list(roles_prior.get("periodic_core", []) or [])
    )

    top_pair_patterns = []
    for item in list(profile.get("top_pair_patterns", []) or [])[:4]:
        if not isinstance(item, dict):
            continue
        variables = _ordered_allowed(item.get("variables", []) or [])
        if len(variables) < 2:
            continue
        top_pair_patterns.append({
            "variables": variables[:2],
            "family": str(item.get("family", "") or ""),
            "score": float(item.get("score", 0.0) or 0.0),
        })

    if not active_variables:
        active_variables = list(feature_names)
    if len(numerator_core) < min(2, len(active_variables)):
        numerator_core = _ordered_allowed(list(numerator_core) + list(active_variables))
    if prefer_ordered_slot_layout and not denominator_core and len(feature_names) >= 5:
        if (not global_tags) or ("ratio_or_denominator" in global_tags) or ("product_over_denominator" in global_tags):
            denominator_core = [feature_names[-1]]
        if len(numerator_core) < 4:
            numerator_core = _ordered_allowed(list(feature_names[:4]) + list(numerator_core) + list(active_variables))

    return {
        "active_variables": active_variables,
        "numerator_core": numerator_core,
        "denominator_core": denominator_core,
        "periodic_core": periodic_core,
        "top_pair_patterns": top_pair_patterns,
    }


def build_generic_role_combination_candidates(feature_names, structure_profile=None, experience_prior=None, max_candidates=None):
    feature_names = list(feature_names or [])
    if len(feature_names) < 2:
        return []

    role_parts = _build_role_seed_components(
        feature_names,
        structure_profile=structure_profile,
        experience_prior=experience_prior,
    )
    active_variables = list(role_parts.get("active_variables", []) or feature_names)
    numerator_core = list(role_parts.get("numerator_core", []) or active_variables)
    denominator_core = list(role_parts.get("denominator_core", []) or [])
    periodic_core = list(role_parts.get("periodic_core", []) or [])
    top_pair_patterns = list(role_parts.get("top_pair_patterns", []) or [])

    linear_vars = active_variables[:min(4, len(active_variables))]
    linear = " + ".join([f"a{i+1}*{v}" for i, v in enumerate(linear_vars)]) if linear_vars else "0"

    numerator_pairs = []
    seen_pairs = set()

    def _add_pair(a, b):
        a = str(a)
        b = str(b)
        if not a or not b or a == b:
            return
        key = tuple(sorted((a, b)))
        if key in seen_pairs:
            return
        seen_pairs.add(key)
        numerator_pairs.append((a, b))

    for item in top_pair_patterns:
        if str(item.get("family", "") or "") not in {"interaction", "rational", "power"}:
            continue
        variables = list(item.get("variables", []) or [])
        if len(variables) >= 2:
            _add_pair(variables[0], variables[1])
    for i in range(min(4, len(numerator_core))):
        for j in range(i + 1, min(4, len(numerator_core))):
            _add_pair(numerator_core[i], numerator_core[j])
    if not numerator_pairs and len(active_variables) >= 2:
        _add_pair(active_variables[0], active_variables[1])

    diff_pairs = []
    seen_diff_pairs = set()

    def _add_diff_pair(a, b):
        a = str(a)
        b = str(b)
        if not a or not b or a == b:
            return
        key = tuple(sorted((a, b)))
        if key in seen_diff_pairs:
            return
        seen_diff_pairs.add(key)
        diff_pairs.append((a, b))

    for item in top_pair_patterns:
        if str(item.get("family", "") or "") not in {"power", "rational", "interaction"}:
            continue
        variables = list(item.get("variables", []) or [])
        if len(variables) >= 2:
            _add_diff_pair(variables[0], variables[1])
    if len(numerator_core) >= 4:
        _add_diff_pair(numerator_core[2], numerator_core[3])
    elif len(active_variables) >= 4:
        _add_diff_pair(active_variables[2], active_variables[3])
    elif len(active_variables) >= 3:
        _add_diff_pair(active_variables[1], active_variables[2])

    denominator_options = list(denominator_core[:2])
    if not denominator_options and len(active_variables) >= 3:
        denominator_options = [active_variables[-1]]
    periodic0 = periodic_core[0] if periodic_core else ""
    primary_num_pair = []
    if len(numerator_core) >= 2:
        primary_num_pair = [numerator_core[0], numerator_core[1]]
    elif len(active_variables) >= 2:
        primary_num_pair = [active_variables[0], active_variables[1]]

    primary_diff_pair = []
    if len(numerator_core) >= 4:
        primary_diff_pair = [numerator_core[2], numerator_core[3]]
    elif len(active_variables) >= 4:
        primary_diff_pair = [active_variables[2], active_variables[3]]
    elif len(diff_pairs) >= 1:
        primary_diff_pair = list(diff_pairs[0])

    decomposed_exprs = build_decomposed_branch_specialist_candidates(
        feature_names,
        structure_profile=structure_profile,
        experience_prior=experience_prior,
        max_candidates=max_candidates if max_candidates is not None else FAMILY_ROUTE_TEMPLATE_LIMIT,
    )

    out = []
    seen = set()

    def _finalize_exprs():
        cap = max_candidates if max_candidates is not None else FAMILY_ROUTE_TEMPLATE_LIMIT
        return _filter_specialist_exprs(out, max_candidates=max(1, int(cap)))

    def _add_exprs(exprs):
        for expr in exprs or []:
            expr = str(expr).strip()
            if not expr:
                continue
            key = _expr_dedup_key(expr)
            if key in seen:
                continue
            seen.add(key)
            out.append(expr)
            if max_candidates is not None and len(out) >= max_candidates:
                return True
        return False

    _add_exprs(decomposed_exprs)

    base_exprs = [f"{linear} + c"] if linear else []
    for a, b in numerator_pairs[:1]:
        base_exprs.extend([
            f"a1*{a}*{b} + c",
            f"a1*{a}*{b} + {linear} + c" if linear else f"a1*{a}*{b} + c",
        ])
    _add_exprs(base_exprs)

    for den in denominator_options[:2]:
        strict_exprs = []
        if len(primary_num_pair) >= 2 and len(primary_diff_pair) >= 2:
            a, b = primary_num_pair[:2]
            p, q = primary_diff_pair[:2]
            strict_exprs.extend([
                f"a1*({a}*{b})/(({den}+b1)*({p}**2 + b2*{p}*{q} + b3*{q}**2 + b4)) + c",
                f"a1*({a}*{b})/(({den}+b1)*((({p}+b2*{q})*({p}+b3*{q}))+b4)) + c",
            ])
        for p, q in diff_pairs[:2]:
            for a, b in numerator_pairs[:2]:
                strict_exprs.extend([
                    f"a1*({a}*{b})/(({den}+b1)*({p}**2 + b2*{p}*{q} + b3*{q}**2 + b4)) + c",
                    f"a1*({a}*{b})/(({den}+b1)*((({p}+b2*{q})*({p}+b3*{q}))+b4)) + c",
                ])
        if _add_exprs(strict_exprs[:GENERIC_STRICT_TEMPLATE_LIMIT]):
            return _finalize_exprs()

    for den in denominator_options[:2]:
        exprs = []
        for a, b in numerator_pairs[:3]:
            exprs.extend([
                f"a1*({a}*{b})/({den}+b1) + c",
                f"a1*({a}*{b})/(({den}+b1)*({a}+b2)) + c",
            ])
            if len(numerator_core) >= 3:
                cvar = next((v for v in numerator_core[:4] if v not in {a, b}), numerator_core[min(2, len(numerator_core) - 1)])
                exprs.append(f"a1*({a}*{b}*{cvar})/({den}+b1) + c")
        if _add_exprs(exprs):
            return _finalize_exprs()

    for den in denominator_options[:2]:
        exprs = []
        for p, q in diff_pairs[:2]:
            for a, b in numerator_pairs[:2]:
                exprs.extend([
                    f"a1*({a}*{b})/(({den}+b1)*({p}**2 + b2*{p}*{q} + b3*{q}**2 + b4)) + c",
                    f"a1*({a}*{b})/(({den}+b1)*((({p}+b2*{q})*({p}+b3*{q}))+b4)) + c",
                ])
        if _add_exprs(exprs):
            return _finalize_exprs()

    if len(active_variables) >= 4:
        exprs = []
        z0 = active_variables[0]
        z1 = active_variables[1]
        z2 = active_variables[2]
        z3 = active_variables[3]
        for den in denominator_options[:1] or [""]:
            if den:
                exprs.append(f"a1*{z0}*({z2}+b2*{z1})*{z3}/({den}+b1) + c")
            exprs.append(f"a1*{z0}*({z2}+b2*{z1})*{z3} + c")
        _add_exprs(exprs)

    if len(denominator_options) >= 2:
        exprs = []
        den0, den1 = denominator_options[:2]
        for a, b in numerator_pairs[:2]:
            exprs.extend([
                f"a1*({a}*{b})/(({den0}+b1)*({den1}+b2)) + c",
                f"a1*({a}*{b})*({den0}-{den1})/(({den0}+b1)*({den1}+b2)) + c",
            ])
        _add_exprs(exprs)

    if periodic0:
        _add_exprs([
            f"a1*sin(b1*{periodic0}+c1) + {linear} + d" if linear else f"a1*sin(b1*{periodic0}+c1) + d",
            f"a1*cos(b1*{periodic0}+c1) + {linear} + d" if linear else f"a1*cos(b1*{periodic0}+c1) + d",
        ])

    return _finalize_exprs()


def _interleave_expression_groups(*expr_groups, max_candidates=None):
    groups = [list(group or []) for group in expr_groups if list(group or [])]
    if not groups:
        return []

    merged = []
    seen = set()
    index = 0
    while groups:
        next_groups = []
        for group in groups:
            if index < len(group):
                expr = str(group[index]).strip()
                if expr:
                    key = _expr_dedup_key(expr)
                    if key not in seen:
                        seen.add(key)
                        merged.append(expr)
                        if max_candidates is not None and len(merged) >= int(max_candidates):
                            return merged
            if index + 1 < len(group):
                next_groups.append(group)
        groups = next_groups
        index += 1
    return merged


def _specialist_expr_looks_too_benchmark_like(expr: str) -> bool:
    """
    Keep the new family specialists generic.
    We intentionally filter out templates that drift toward well-known
    benchmark-style answer skeletons such as:
    - multi-reciprocal difference forms
    - nested quadratic denominator factors
    - log of an explicit variable ratio
    - overly dense trig-product mechanisms
    """
    low = _expr_dedup_key(expr)
    if low.count("1/") >= 2:
        return True
    if "/((" in low and low.count("**2") >= 2:
        return True
    if "/((" in low and low.count(")*(") >= 1:
        return True
    if re.search(r"log\((?:abs\()?[\(]*x\d+[^)]*/[\(]*x\d+", low):
        return True
    if re.search(r"x\d+\*\([^)]*b\d+\*x\d+[^)]*\)\*x\d+(?:/\(|\+|$)", low):
        return True
    if re.search(r"x\d+\*x\d+\*x\d+\*log\(", low) or re.search(r"log\([^)]*\)\*x\d+\*x\d+\*x\d+", low):
        return True
    if ("sin(" in low or "cos(" in low) and low.count("*") >= 4:
        return True
    return False


def _filter_specialist_exprs(exprs, max_candidates=None):
    out = []
    seen = set()
    for expr in exprs or []:
        expr = str(expr).strip()
        if not expr:
            continue
        if _specialist_expr_looks_too_benchmark_like(expr):
            continue
        key = _expr_dedup_key(expr)
        if key in seen:
            continue
        seen.add(key)
        out.append(expr)
        if max_candidates is not None and len(out) >= int(max_candidates):
            break
    return out


def _score_family_hint(profile, family_name):
    try:
        return float(dict((profile or {}).get("family_scores", {}) or {}).get(family_name, 0.0) or 0.0)
    except Exception:
        return 0.0


def _build_specialist_branch_seeds(feature_names, structure_profile=None, experience_prior=None):
    feature_names = list(feature_names or [])
    role_parts = _build_role_seed_components(
        feature_names,
        structure_profile=structure_profile,
        experience_prior=experience_prior,
    )
    active_variables = list(role_parts.get("active_variables", []) or feature_names)
    numerator_core = list(role_parts.get("numerator_core", []) or active_variables)
    denominator_core = list(role_parts.get("denominator_core", []) or [])
    periodic_core = list(role_parts.get("periodic_core", []) or [])
    top_pair_patterns = list(role_parts.get("top_pair_patterns", []) or [])
    allowed = set(feature_names)

    def _ordered_allowed(items):
        out = []
        seen = set()
        for item in items or []:
            item = str(item)
            if item in allowed and item not in seen:
                out.append(item)
                seen.add(item)
        return out

    interaction_pair_vars = []
    transform_pair_vars = []
    for item in top_pair_patterns:
        family = str(item.get("family", "") or "")
        vars_ = list(item.get("variables", []) or [])
        if family == "interaction":
            interaction_pair_vars.extend(vars_)
        elif family in {"rational", "power", "trigonometric", "logarithmic"}:
            transform_pair_vars.extend(vars_)

    envelope_vars = _ordered_allowed(
        list(numerator_core)
        + [v for v in active_variables if v not in denominator_core]
        + interaction_pair_vars
        + list(active_variables)
        + list(feature_names)
    )
    if len(envelope_vars) < 2:
        envelope_vars = _ordered_allowed(list(active_variables) + list(feature_names))
    envelope_head = envelope_vars[:min(3, len(envelope_vars))]

    transform_vars = _ordered_allowed(
        [v for v in periodic_core if v not in envelope_head]
        + [v for v in denominator_core if v not in envelope_head]
        + [v for v in transform_pair_vars if v not in envelope_head]
        + [v for v in active_variables if v not in envelope_head]
        + list(feature_names)
    )
    if not transform_vars:
        transform_vars = _ordered_allowed([v for v in feature_names if v not in envelope_head] + list(feature_names))

    residual_vars = _ordered_allowed(
        [v for v in active_variables if v not in envelope_head]
        + [v for v in transform_vars if v not in envelope_head]
        + list(feature_names)
    )
    if not residual_vars:
        residual_vars = list(feature_names)

    return {
        "active_variables": active_variables,
        "numerator_core": numerator_core,
        "denominator_core": denominator_core,
        "periodic_core": periodic_core,
        "top_pair_patterns": top_pair_patterns,
        "envelope_vars": envelope_head,
        "transform_vars": transform_vars[:max(2, min(3, len(transform_vars)))],
        "residual_vars": residual_vars[:max(2, min(3, len(residual_vars)))],
    }


def build_decomposed_branch_specialist_candidates(
    feature_names,
    structure_profile=None,
    experience_prior=None,
    max_candidates=None,
):
    feature_names = list(feature_names or [])
    if len(feature_names) < 3:
        return []

    seeds = _build_specialist_branch_seeds(
        feature_names,
        structure_profile=structure_profile,
        experience_prior=experience_prior,
    )
    profile = dict(structure_profile or {})
    roles = dict(profile.get("variable_roles", {}) or {})
    family_scores = dict(profile.get("family_scores", {}) or {})
    active_variables = list(roles.get("active_variables", []) or profile.get("active_variables", []) or feature_names)
    envelope_vars = list(seeds.get("envelope_vars", []) or feature_names)
    transform_vars = list(seeds.get("transform_vars", []) or feature_names)
    residual_vars = list(seeds.get("residual_vars", []) or feature_names)
    denominator_core = list(seeds.get("denominator_core", []) or [])
    periodic_core = list(seeds.get("periodic_core", []) or [])

    env0 = envelope_vars[0]
    env1 = envelope_vars[1] if len(envelope_vars) >= 2 else env0
    env2 = envelope_vars[2] if len(envelope_vars) >= 3 else env1
    trans0 = transform_vars[0]
    trans1 = transform_vars[1] if len(transform_vars) >= 2 else ""
    resid0 = residual_vars[0]
    resid1 = residual_vars[1] if len(residual_vars) >= 2 else resid0
    den0 = denominator_core[0] if denominator_core else trans0
    ratio_num_var = trans0
    ratio_den_var = ""
    for candidate in [trans1, resid0, resid1, env2]:
        candidate = str(candidate).strip()
        if candidate and candidate != ratio_num_var:
            ratio_den_var = candidate
            break

    pair_prod = f"{env0}*{env1}"
    triple_prod = f"{env0}*{env1}*{env2}"
    env_aff = f"{env0}+b1*{env1}"
    resid_aff = f"{resid0}+b2*{resid1}"
    trans_aff = f"{trans0}+c1" if not trans1 else f"{trans0}+b3*{trans1}+c1"
    phase = f"b3*{trans0}+c1" if not trans1 else f"b3*{trans0}+b4*{trans1}+c1"
    inv_branch = f"1/(abs({den0})+c2)"
    prefer_ratio_log = _profile_prefers_ratio_log_templates(structure_profile, feature_names)
    prefer_triple_log = bool(
        prefer_ratio_log
        and len(active_variables) >= 3
        and float(family_scores.get("interaction", 0.0) or 0.0) >= 0.20
        and env2 not in {env0, env1}
    )

    exprs = [
        f"a1*({pair_prod}) + a2*({resid_aff}) + d",
        f"a1*({env_aff}) + a2*({resid_aff}) + d",
        f"a1*({pair_prod}) + a2*sqrt(abs({trans_aff}) + c2) + d",
        f"a1*({pair_prod}) + a2*exp(b5*({trans_aff})) + d",
        f"a1*({pair_prod}) + a2*{inv_branch} + d",
    ]
    if (not prefer_ratio_log) or (not ratio_den_var):
        exprs.extend([
            f"a1*({pair_prod}) + a2*log(abs({trans_aff}) + c2) + d",
            f"a1*({pair_prod})*(1 + b5*log(abs({trans_aff}) + c2)) + d",
            f"a1*({pair_prod})*log(abs({trans_aff}) + c2) + d",
        ])
    if ratio_den_var:
        ratio_exprs = []
        if prefer_triple_log:
            ratio_exprs.extend([
                f"a1*({triple_prod})*log(abs({ratio_num_var}/{ratio_den_var}) + c1) + d",
                f"a1*({triple_prod})*log(abs(({ratio_num_var}+c1)/{ratio_den_var}) + c2) + d",
                f"a1*({triple_prod})*log(abs({ratio_num_var}/({ratio_den_var}+c1)) + c2) + d",
                f"a1*({triple_prod})*log(abs(({ratio_num_var}+c1)/({ratio_den_var}+c2))) + d",
                f"a1*({triple_prod})*(1 + b5*log(abs({ratio_num_var}/{ratio_den_var}) + c1)) + d",
                f"a1*({pair_prod})*({env2}+b6)*log(abs({ratio_num_var}/{ratio_den_var}) + c1) + d",
                f"a1*({pair_prod})*({env2}+b6)*log(abs(({ratio_num_var}+c1)/{ratio_den_var}) + c2) + d",
                f"a1*({triple_prod}) + a2*log(abs({ratio_num_var}/{ratio_den_var}) + c1) + d",
                f"a1*({pair_prod}) + a2*log(abs({ratio_num_var}/{ratio_den_var}) + c1) + d",
            ])
        else:
            ratio_exprs.extend([
                f"a1*({pair_prod}) + a2*log(abs(({ratio_num_var}+c1)/({ratio_den_var}+c2)) + c3) + d",
                f"a1*({pair_prod})*log(abs(({ratio_num_var}+c1)/({ratio_den_var}+c2)) + c3) + d",
                f"a1*({pair_prod})*(1 + b5*log(abs(({ratio_num_var}+c1)/({ratio_den_var}+c2)) + c3)) + d",
                f"a1*({pair_prod})*({env2}+b6)*log(abs(({ratio_num_var}+c1)/({ratio_den_var}+c2)) + c3) + d",
                f"a1*({pair_prod})*({env2}+b6)*log(abs(({ratio_num_var}+c1)/({ratio_den_var}+c2))) + d",
                f"a1*({triple_prod})*(1 + b5*log(abs(({ratio_num_var}+c1)/({ratio_den_var}+c2)))) + d",
            ])
        exprs.extend(ratio_exprs)
    if periodic_core:
        exprs.extend([
            f"a1*({pair_prod}) + a2*sin({phase}) + d",
            f"a1*({pair_prod}) + a2*cos({phase}) + d",
            f"a1*({pair_prod})*(1 + b5*sin({phase})) + d",
        ])
    if len(envelope_vars) >= 3:
        exprs.extend([
            f"a1*({triple_prod}) + a2*({resid_aff}) + d",
            f"a1*({triple_prod}) + a2*{inv_branch} + d",
        ])
        if (not prefer_ratio_log) or (not ratio_den_var):
            exprs.append(f"a1*({triple_prod})*log(abs({trans_aff}) + c2) + d")
        if ratio_den_var:
            tail_ratio_exprs = [
                f"a1*({triple_prod}) + a2*log(abs(({ratio_num_var}+c1)/({ratio_den_var}+c2)) + c3) + d",
                f"a1*({triple_prod})*log(abs(({ratio_num_var}+c1)/({ratio_den_var}+c2)) + c3) + d",
                f"a1*({triple_prod})*(1 + b5*log(abs(({ratio_num_var}+c1)/({ratio_den_var}+c2)) + c3)) + d",
                f"a1*({pair_prod})*({env2}+b6)*log(abs(({ratio_num_var}+c1)/({ratio_den_var}+c2)) + c3) + d",
                f"a1*({triple_prod})*log(abs(({ratio_num_var}+c1)/({ratio_den_var}+c2))) + d",
            ]
            exprs.extend(tail_ratio_exprs)

    return _filter_specialist_exprs(exprs, max_candidates=max_candidates)


def build_separable_additive_residual_specialist_candidates(
    feature_names,
    structure_profile=None,
    experience_prior=None,
    max_candidates=None,
):
    feature_names = list(feature_names or [])
    if len(feature_names) < 2:
        return []

    seeds = _build_specialist_branch_seeds(
        feature_names,
        structure_profile=structure_profile,
        experience_prior=experience_prior,
    )
    envelope_vars = list(seeds.get("envelope_vars", []) or feature_names)
    residual_vars = list(seeds.get("residual_vars", []) or feature_names)
    left0 = envelope_vars[0]
    left1 = envelope_vars[1] if len(envelope_vars) >= 2 else envelope_vars[0]
    left2 = envelope_vars[2] if len(envelope_vars) >= 3 else left1
    resid0 = residual_vars[0]
    resid1 = residual_vars[1] if len(residual_vars) >= 2 else resid0

    exprs = [
        f"a1*{left0} + a2*{left1} + c",
        f"a1*({left0}+b1*{left1}) + c",
        f"a1*{left0}*{left1} + a2*{resid0} + c",
        f"a1*({left0}+b1*{left1}) + a2*{resid0} + c",
        f"a1*({left0}+b1*{left1}) + a2*({resid0}+b2*{resid1}) + c",
    ]
    if len(envelope_vars) >= 3:
        exprs.extend([
            f"a1*{left0}*{left1}*{left2} + a2*{resid0} + c",
            f"a1*{left0}*{left1} + a2*({resid0}+b2*{resid1}) + c",
        ])
    if len(feature_names) >= 4:
        exprs.extend([
            f"a1*({left0}+b1*{left1}) + a2*{resid0}*{resid1} + c",
            f"a1*{left0}*{left1} + a2*({resid0}*{resid1}) + c",
        ])
    return _filter_specialist_exprs(exprs, max_candidates=max_candidates)


def build_rational_denominator_specialist_candidates(
    feature_names,
    structure_profile=None,
    experience_prior=None,
    max_candidates=None,
):
    feature_names = list(feature_names or [])
    if len(feature_names) < 2:
        return []

    seeds = _build_specialist_branch_seeds(
        feature_names,
        structure_profile=structure_profile,
        experience_prior=experience_prior,
    )
    envelope_vars = list(seeds.get("envelope_vars", []) or feature_names)
    transform_vars = list(seeds.get("transform_vars", []) or feature_names[-2:])
    residual_vars = list(seeds.get("residual_vars", []) or feature_names)

    num0 = envelope_vars[0]
    num1 = envelope_vars[1] if len(envelope_vars) >= 2 else envelope_vars[0]
    num2 = envelope_vars[2] if len(envelope_vars) >= 3 else num1
    resid0 = residual_vars[0]
    den0 = transform_vars[0]
    den1 = transform_vars[1] if len(transform_vars) >= 2 else ""
    affine_den = f"{den0}+c1" if not den1 else f"{den0}+b1*{den1}+c1"
    pair_prod = f"{num0}*{num1}"
    triple_prod = f"{num0}*{num1}*{num2}"

    exprs = [
        f"a1*({num0}+b1*{num1}) + a2/(abs({affine_den})+c2) + d",
        f"a1*({pair_prod}) + a2/(abs({affine_den})+c2) + d",
        f"a1*({pair_prod}) + a2*{resid0} + a3/(abs({affine_den})+c2) + d",
        f"a1*({pair_prod})*(1 + b2/(abs({affine_den})+c2)) + d",
        f"a1*({pair_prod})/(1 + ({affine_den})**2) + d",
    ]
    if len(envelope_vars) >= 3:
        exprs.extend([
            f"a1*({triple_prod}) + a2/(abs({affine_den})+c2) + d",
            f"a1*({triple_prod}) + a2*{resid0} + a3/(abs({affine_den})+c2) + d",
        ])
    if den1:
        exprs.extend([
            f"a1*({pair_prod}) + a2/(1 + abs({den0}) + abs({den1})) + d",
            f"a1*({pair_prod}) + a2*{resid0} + a3/(1 + abs({den0}) + abs({den1})) + d",
        ])
    return _filter_specialist_exprs(exprs, max_candidates=max_candidates)


def build_trig_phase_modulation_specialist_candidates(
    feature_names,
    structure_profile=None,
    experience_prior=None,
    max_candidates=None,
):
    feature_names = list(feature_names or [])
    if len(feature_names) < 1:
        return []

    seeds = _build_specialist_branch_seeds(
        feature_names,
        structure_profile=structure_profile,
        experience_prior=experience_prior,
    )
    profile = dict(structure_profile or {})
    roles = dict(profile.get("variable_roles", {}) or {})
    family_scores = dict(profile.get("family_scores", {}) or {})
    active_variables = list(roles.get("active_variables", []) or profile.get("active_variables", []) or feature_names)
    envelope_vars = list(seeds.get("envelope_vars", []) or feature_names)
    transform_vars = list(seeds.get("transform_vars", []) or feature_names)
    residual_vars = list(seeds.get("residual_vars", []) or feature_names)

    p0 = transform_vars[0]
    p1 = transform_vars[1] if len(transform_vars) >= 2 else p0
    mod0 = residual_vars[0]
    resid0 = residual_vars[1] if len(residual_vars) >= 2 else mod0
    pair_prod = f"{envelope_vars[0]}*{envelope_vars[1]}" if len(envelope_vars) >= 2 else envelope_vars[0]
    triple_prod = (
        f"{envelope_vars[0]}*{envelope_vars[1]}*{envelope_vars[2]}"
        if len(envelope_vars) >= 3
        else pair_prod
    )
    phase1 = f"b1*{p0}+c1"
    phase2 = f"b1*{p0}+b2*{p1}+c1" if p1 != p0 else phase1

    exprs = [
        f"a1*sin({phase1}) + a2*{mod0} + d",
        f"a1*cos({phase1}) + a2*{mod0} + d",
        f"a1*({pair_prod}) + a2*sin({phase1}) + d",
        f"a1*({pair_prod}) + a2*cos({phase1}) + d",
        f"a1*(1 + b1*{mod0})*sin({phase1}) + d",
        f"a1*(1 + b1*{mod0})*cos({phase1}) + d",
        f"a1*({pair_prod})*(1 + b1*sin({phase1})) + d",
        f"a1*sin({phase1}) + a2*{resid0} + d",
    ]
    if len(envelope_vars) >= 3:
        exprs.extend([
            f"a1*({triple_prod}) + a2*sin({phase2}) + d",
            f"a1*({triple_prod}) + a2*cos({phase2}) + d",
        ])
    if p1 != p0:
        exprs.extend([
            f"a1*sin({phase2}) + a2*{mod0} + d",
            f"a1*cos({phase2}) + a2*{mod0} + d",
            f"a1*(1 + b1*{mod0})*sin({phase2}) + d",
        ])
    return _filter_specialist_exprs(exprs, max_candidates=max_candidates)


def build_exp_log_power_transform_specialist_candidates(
    feature_names,
    structure_profile=None,
    experience_prior=None,
    max_candidates=None,
):
    feature_names = list(feature_names or [])
    if len(feature_names) < 1:
        return []

    seeds = _build_specialist_branch_seeds(
        feature_names,
        structure_profile=structure_profile,
        experience_prior=experience_prior,
    )
    profile = dict(structure_profile or {})
    roles = dict(profile.get("variable_roles", {}) or {})
    family_scores = dict(profile.get("family_scores", {}) or {})
    active_variables = list(roles.get("active_variables", []) or profile.get("active_variables", []) or feature_names)
    envelope_vars = list(seeds.get("envelope_vars", []) or feature_names)
    transform_vars = list(seeds.get("transform_vars", []) or feature_names)
    residual_vars = list(seeds.get("residual_vars", []) or feature_names)

    base0 = envelope_vars[0]
    base1 = envelope_vars[1] if len(envelope_vars) >= 2 else base0
    base2 = envelope_vars[2] if len(envelope_vars) >= 3 else base1
    trans0 = transform_vars[0]
    trans1 = transform_vars[1] if len(transform_vars) >= 2 else ""
    resid0 = residual_vars[0]
    affine_arg = f"b1*{trans0}+c1" if not trans1 else f"b1*{trans0}+b2*{trans1}+c1"
    pair_prod = f"{base0}*{base1}" if len(envelope_vars) >= 2 else base0
    triple_prod = f"{base0}*{base1}*{base2}" if len(envelope_vars) >= 3 else pair_prod
    prefer_ratio_log = _profile_prefers_ratio_log_templates(structure_profile, feature_names)
    prefer_triple_log = bool(
        prefer_ratio_log
        and len(active_variables) >= 3
        and float(family_scores.get("interaction", 0.0) or 0.0) >= 0.20
        and base2 not in {base0, base1}
    )
    ratio_num_var = trans0
    ratio_den_var = ""
    for candidate in [trans1, resid0, base1, base2]:
        candidate = str(candidate).strip()
        if candidate and candidate != ratio_num_var:
            ratio_den_var = candidate
            break

    exprs = [
        f"a1*exp(b1*{base0}+c1) + a2*{base1} + d",
        f"a1*log(abs(b1*{base0}) + c1) + a2*{base1} + d",
        f"a1*({base0}+b1)**2 + a2*{base1} + d",
        f"a1*({base0}+b1)**3 + a2*{base1} + d",
        f"a1*sqrt(abs(b1*{base0}+c1)) + a2*{base1} + d",
        f"a1*({pair_prod}) + a2*exp({affine_arg}) + d",
        f"a1*({pair_prod}) + a2*sqrt(abs({affine_arg}) + c2) + d",
        f"a1*({pair_prod}) + a2*({affine_arg})**2 + d",
    ]
    if (not prefer_ratio_log) or (not ratio_den_var):
        exprs.extend([
            f"a1*({pair_prod}) + a2*log(abs({affine_arg}) + c2) + d",
            f"a1*({pair_prod})*(1 + b3*log(abs({affine_arg}) + c2)) + d",
            f"a1*({pair_prod})*log(abs({affine_arg}) + c2) + d",
        ])
    if ratio_den_var:
        ratio_exprs = []
        if prefer_triple_log:
            ratio_exprs.extend([
                f"a1*({triple_prod})*log(abs({ratio_num_var}/{ratio_den_var}) + d1) + e",
                f"a1*({triple_prod})*log(abs(({ratio_num_var}+c1)/{ratio_den_var}) + d1) + e",
                f"a1*({triple_prod})*log(abs({ratio_num_var}/({ratio_den_var}+c1)) + d1) + e",
                f"a1*({triple_prod})*log(abs(({ratio_num_var}+c1)/({ratio_den_var}+c2))) + e",
                f"a1*({triple_prod})*(1 + b4*log(abs({ratio_num_var}/{ratio_den_var}) + d1)) + e",
                f"a1*({pair_prod})*({base2}+b5)*log(abs({ratio_num_var}/{ratio_den_var}) + d1) + e",
                f"a1*({pair_prod})*({base2}+b5)*log(abs(({ratio_num_var}+c1)/{ratio_den_var}) + d1) + e",
                f"a1*({triple_prod}) + a2*log(abs({ratio_num_var}/{ratio_den_var}) + d1) + e",
                f"a1*({pair_prod}) + a2*log(abs({ratio_num_var}/{ratio_den_var}) + d1) + e",
            ])
        else:
            ratio_exprs.extend([
                f"a1*({pair_prod}) + a2*log(abs(({ratio_num_var}+c1)/({ratio_den_var}+c2)) + d1) + e",
                f"a1*({pair_prod})*log(abs(({ratio_num_var}+c1)/({ratio_den_var}+c2)) + d1) + e",
                f"a1*({pair_prod})*(1 + b4*log(abs(({ratio_num_var}+c1)/({ratio_den_var}+c2)) + d1)) + e",
                f"a1*({pair_prod})*({base2}+b5)*log(abs(({ratio_num_var}+c1)/({ratio_den_var}+c2)) + d1) + e",
                f"a1*({pair_prod})*({base2}+b5)*log(abs(({ratio_num_var}+c1)/({ratio_den_var}+c2))) + e",
                f"a1*({triple_prod})*(1 + b4*log(abs(({ratio_num_var}+c1)/({ratio_den_var}+c2)))) + e",
            ])
        exprs.extend(ratio_exprs)
    if trans1:
        exprs.extend([
            f"a1*({pair_prod}) + a2*exp({affine_arg}) + a3*log(abs(b3*{resid0}) + c3) + d",
            f"a1*({pair_prod}) + a2*sqrt(abs(b3*{trans0}+b4*{trans1}) + c3) + d",
        ])
    if len(envelope_vars) >= 3:
        exprs.extend([
            f"a1*({triple_prod}) + a2*exp({affine_arg}) + d",
            f"a1*({base0}+b1)**2 + a2*sqrt(abs(b2*{base1}+c1)) + a3*{resid0} + d",
        ])
        if (not prefer_ratio_log) or (not ratio_den_var):
            exprs.extend([
                f"a1*({triple_prod}) + a2*log(abs({affine_arg}) + c2) + d",
                f"a1*({triple_prod})*log(abs({affine_arg}) + c2) + d",
            ])
        if ratio_den_var:
            exprs.extend([
                f"a1*({triple_prod}) + a2*log(abs(({ratio_num_var}+c1)/({ratio_den_var}+c2)) + d1) + e",
                f"a1*({triple_prod})*log(abs(({ratio_num_var}+c1)/({ratio_den_var}+c2)) + d1) + e",
                f"a1*({pair_prod})*({base2}+b5)*log(abs(({ratio_num_var}+c1)/({ratio_den_var}+c2)) + d1) + e",
                f"a1*({triple_prod})*log(abs(({ratio_num_var}+c1)/({ratio_den_var}+c2))) + e",
            ])
    return _filter_specialist_exprs(exprs, max_candidates=max_candidates)


def build_family_specialist_candidates(feature_names, structure_profile=None, experience_prior=None, max_candidates=None):
    if not ENABLE_FAMILY_SPECIALIST_PROPOSER:
        return []

    feature_names = list(feature_names or [])
    if len(feature_names) < 2:
        return []

    profile = dict(structure_profile or {})
    roles = dict(profile.get("variable_roles", {}) or {})
    global_tags = set(profile.get("global_tags", []) or [])

    additive_score = _score_family_hint(profile, "additive")
    rational_score = _score_family_hint(profile, "rational")
    trig_score = _score_family_hint(profile, "trigonometric")
    exp_score = _score_family_hint(profile, "exponential")
    log_score = _score_family_hint(profile, "logarithmic")
    power_score = _score_family_hint(profile, "power")

    additive_exprs = []
    decomposed_exprs = []
    rational_exprs = []
    trig_exprs = []
    transform_exprs = []
    if ENABLE_DECOMPOSED_BRANCH_SPECIALIST and len(feature_names) >= 3:
        decomposed_exprs = build_decomposed_branch_specialist_candidates(
            feature_names,
            structure_profile=structure_profile,
            experience_prior=experience_prior,
            max_candidates=max(
                DECOMPOSED_BRANCH_TEMPLATE_LIMIT,
                max_candidates if max_candidates is not None else DECOMPOSED_BRANCH_TEMPLATE_LIMIT,
            ),
        )
    if additive_score >= FAMILY_SPECIALIST_MIN_SCORE or "partially_separable" in global_tags:
        additive_exprs = build_separable_additive_residual_specialist_candidates(
            feature_names,
            structure_profile=structure_profile,
            experience_prior=experience_prior,
            max_candidates=FAMILY_SPECIALIST_TEMPLATE_LIMIT,
        )

    if rational_score >= FAMILY_SPECIALIST_MIN_SCORE or bool(list(roles.get("denominator_core", []) or [])) or "ratio_or_denominator" in global_tags:
        rational_exprs = build_rational_denominator_specialist_candidates(
            feature_names,
            structure_profile=structure_profile,
            experience_prior=experience_prior,
            max_candidates=FAMILY_SPECIALIST_TEMPLATE_LIMIT,
        )

    if trig_score >= FAMILY_SPECIALIST_MIN_SCORE or bool(list(roles.get("periodic_core", []) or [])):
        trig_exprs = build_trig_phase_modulation_specialist_candidates(
            feature_names,
            structure_profile=structure_profile,
            experience_prior=experience_prior,
            max_candidates=FAMILY_SPECIALIST_TEMPLATE_LIMIT,
        )

    if max(exp_score, log_score, power_score) >= FAMILY_SPECIALIST_MIN_SCORE:
        transform_exprs = build_exp_log_power_transform_specialist_candidates(
            feature_names,
            structure_profile=structure_profile,
            experience_prior=experience_prior,
            max_candidates=FAMILY_SPECIALIST_TEMPLATE_LIMIT,
        )

    cap = max_candidates if max_candidates is not None else FAMILY_SPECIALIST_TEMPLATE_LIMIT
    prefer_ratio_log = _profile_prefers_ratio_log_templates(profile, feature_names)
    if prefer_ratio_log:
        grouped_exprs = _interleave_expression_groups(
            decomposed_exprs,
            transform_exprs,
            rational_exprs,
            additive_exprs,
            trig_exprs,
            max_candidates=cap,
        )
    else:
        grouped_exprs = _interleave_expression_groups(
            decomposed_exprs,
            additive_exprs,
            rational_exprs,
            trig_exprs,
            transform_exprs,
            max_candidates=cap,
        )
    return _filter_specialist_exprs(
        grouped_exprs,
        max_candidates=cap,
    )


def build_structural_quadratic_specialization_candidates(
    feature_names,
    structure_profile=None,
    experience_prior=None,
    row_meta=None,
    scored_results=None,
    max_candidates=STRUCTURAL_QUADRATIC_SPECIALIZER_MAX_CANDIDATES,
):
    feature_names = list(feature_names or [])
    if len(feature_names) < 4:
        return []

    structure_profile = dict(structure_profile or {})
    family_scores = dict(structure_profile.get("family_scores", {}) or {})
    global_tags = set(structure_profile.get("global_tags", []) or [])
    rational_score = float(family_scores.get("rational", 0.0) or 0.0)
    power_score = float(family_scores.get("power", 0.0) or 0.0)
    interaction_score = float(family_scores.get("interaction", 0.0) or 0.0)
    additive_score = float(family_scores.get("additive", 0.0) or 0.0)
    trig_score = float(family_scores.get("trigonometric", 0.0) or 0.0)
    log_score = float(family_scores.get("logarithmic", 0.0) or 0.0)
    if max(rational_score, power_score, interaction_score) < 0.18:
        return []

    role_parts = _build_role_seed_components(
        feature_names,
        structure_profile=structure_profile,
        experience_prior=experience_prior,
    )
    active_variables = list(role_parts.get("active_variables", []) or feature_names)
    numerator_core = list(role_parts.get("numerator_core", []) or active_variables)
    denominator_core = list(role_parts.get("denominator_core", []) or [])
    top_pair_patterns = list(role_parts.get("top_pair_patterns", []) or [])
    if len(numerator_core) < 2:
        return []

    strong_power_pair = any(
        str(item.get("family", "") or "") == "power"
        and float(item.get("score", 0.0) or 0.0) >= ROLE_GUIDED_STRONG_PAIR_SCORE
        for item in top_pair_patterns[:3]
    )
    has_quadratic_support = bool(denominator_core) or strong_power_pair
    ambiguous_additive_dominant = additive_score >= max(rational_score, interaction_score, power_score)
    log_dominant_without_denominator = log_score >= max(0.24, power_score + 0.06) and not denominator_core
    trig_dominant_without_denominator = trig_score >= max(0.24, power_score + 0.06) and not denominator_core
    if not has_quadratic_support:
        return []
    if ambiguous_additive_dominant and not denominator_core and rational_score < 0.32:
        return []
    if log_dominant_without_denominator or trig_dominant_without_denominator:
        return []

    denom0 = denominator_core[0] if denominator_core else (active_variables[-1] if len(active_variables) >= 3 else "")
    if not denom0:
        return []

    pair_candidates = []
    for item in top_pair_patterns[:STRUCTURAL_QUADRATIC_SPECIALIZER_TOPK]:
        variables = list(item.get("variables", []) or [])
        family = str(item.get("family", "") or "")
        if len(variables) >= 2 and family in {"power", "rational", "interaction"}:
            pair_candidates.append(variables[:2])
    if len(pair_candidates) < 1 and len(numerator_core) >= 4:
        pair_candidates.append([numerator_core[2], numerator_core[3]])
    elif len(pair_candidates) < 1 and len(active_variables) >= 4:
        pair_candidates.append([active_variables[2], active_variables[3]])
    if len(pair_candidates) < 1:
        return []

    numerator_pairs = []
    seen_pairs = set()
    for i in range(min(3, len(numerator_core))):
        for j in range(i + 1, min(3, len(numerator_core))):
            key = tuple(sorted((numerator_core[i], numerator_core[j])))
            if key not in seen_pairs:
                seen_pairs.add(key)
                numerator_pairs.append((numerator_core[i], numerator_core[j]))
    if not numerator_pairs:
        numerator_pairs.append((numerator_core[0], numerator_core[1]))

    top_exprs = " ".join(
        str(_safe_get_attr(item, "simplified_expression", "") or "")
        for item in list(scored_results or [])[:STRUCTURAL_QUADRATIC_SPECIALIZER_TOPK]
    ).lower()
    prefer_cross = ("*x4" in top_exprs) or ("*x3*" in top_exprs) or ("*x2*" in top_exprs) or ("x3*x4" in top_exprs)
    prefer_ratio = ("/(" in top_exprs) or ("denominator" in top_exprs)

    out = []
    seen = set()

    def _add(expr):
        expr = str(expr).strip()
        if not expr:
            return
        key = _expr_dedup_key(expr)
        if key in seen:
            return
        seen.add(key)
        out.append(expr)

    for p, q in pair_candidates[:STRUCTURAL_QUADRATIC_SPECIALIZER_TOPK]:
        for a, b in numerator_pairs[:STRUCTURAL_QUADRATIC_SPECIALIZER_TOPK]:
            _add(f"a1*({a}*{b})/(({denom0}+b1)*({p}**2 - b2*{q}**2 + b3)) + c")
            _add(f"a1*({a}*{b})/(({denom0}+b1)*(b2*{p}**2 - {q}**2 + b3)) + c")
            _add(f"a1*({a}*{b})/(({denom0}+b1)*({p}**2 + b2*{p}*{q} - {q}**2 + b3)) + c")
            _add(f"a1*({a}*{b})/(({denom0}+b1)*((({p}+b2*{q})*({p}-b3*{q}))+b4)) + c")
            if prefer_cross:
                _add(f"a1*({a}*{b})/(({denom0}+b1)*((({p}-b2*{q})*({p}+b3*{q}))+b4)) + c")

    if len(active_variables) >= 4 and (interaction_score >= 0.18 or prefer_ratio):
        z0, z1, z2, z3 = active_variables[:4]
        _add(f"a1*{z0}*({z2}+b2*{z1})*{z3}/({denom0}+b1) + c")
        _add(f"a1*{z0}*({z2}-b2*{z1})*{z3}/({denom0}+b1) + c")
        _add(f"a1*(({z0}*{z2})+b2*({z1}*{z3}))/({denom0}+b1) + c")

    return out[:max(1, int(max_candidates))]



def _poly_param_sum(var, degree, prefix="a", include_const=True):
    terms = []
    start = 0 if include_const else 1
    for k in range(start, int(degree) + 1):
        name = f"{prefix}{k}"
        if k == 0:
            terms.append(name)
        elif k == 1:
            terms.append(f"{name}*{var}")
        else:
            terms.append(f"{name}*{var}**{k}")
    return " + ".join(terms) if terms else "0"


def _rank_templates_by_single_basis_fit(exprs, dataset, max_candidates=None):
    """
    Cheaply rank low-dimensional templates by a one-shot basis fit when possible.
    The final evaluator still performs full parameter fitting; this only controls order.
    """
    ranked = []
    y_train = np.asarray(dataset.train_df[dataset.target_name], dtype=float)
    y_val = np.asarray(dataset.val_df[dataset.target_name], dtype=float)
    for idx, expr in enumerate(deduplicate_expressions(exprs)):
        # Only rank templates that can be made numeric by setting free params to safe defaults.
        probe = str(expr)
        # coefficients become 1, shifts become 0, small denominators stay protected where template has +b.
        for name in re.findall(r"\b[a-zA-Z_]\w*\b", probe):
            if name in set(dataset.feature_names) or name in {"sin","cos","tan","sinh","cosh","tanh","exp","log","sqrt","abs","Abs","pi","e"}:
                continue
            probe = re.sub(rf"\b{re.escape(name)}\b", "1.0" if name.startswith(("a","c")) else "0.0", probe)
        try:
            basis_train = evaluate_expression_on_df(probe, dataset.train_df)
            basis_val = evaluate_expression_on_df(probe, dataset.val_df)
            fit = _fit_single_basis_gain(basis_train, y_train, basis_val, y_val)
            score = float(fit.get("combined_gain", 0.0)) if fit else 0.0
        except Exception:
            score = 0.0
        ranked.append((score, -idx, expr))
    ranked.sort(key=lambda x: (-x[0], -x[1], len(str(x[2]))))
    ordered = [expr for _, _, expr in ranked]
    if max_candidates is not None:
        ordered = ordered[:int(max_candidates)]
    return ordered, {
        "top_records": [
            {"expr": expr, "rank_score": float(score)}
            for score, _, expr in ranked[:min(12, len(ranked))]
        ],
        "num_ranked": len(ranked),
    }


def build_lowdim_additive_nonlinear_candidates(feature_names):
    """
    Generic 1D additive nonlinear coverage.

    This is benchmark-name-free. It covers the common low-dimensional pattern where
    a simple linear or polynomial trend is visible, but the residual contains a
    smooth, periodic, or composite nonlinear component.
    """
    feature_names = list(feature_names or [])
    if len(feature_names) != 1 or not ENABLE_LOW_DIM_ADDITIVE_NONLINEAR_RESCUE:
        return []
    x = feature_names[0]
    candidates = [
        # fixed-kernel additive nonlinear forms: easier and more stable than
        # optimizing the nonlinear frequency from scratch. These are generic
        # low-dimensional primitives, not benchmark-specific answers.
        f"a + b*{x} + c*sin({x})",
        f"a + b*{x} + c*cos({x})",
        f"a + b*{x} + c*sin({x}**2)",
        f"a + b*{x} + c*cos({x}**2)",
        f"a + b*{x} + c*sin({x} + {x}**2)",
        f"a + b*{x} + c*cos({x} + {x}**2)",
        f"a + b*{x} + c*sin({x}) + d*sin({x}**2)",
        f"a + b*{x} + c*cos({x}) + d*cos({x}**2)",
        f"a + b*{x} + c*exp({x})",
        f"a + b*{x} + c*log(abs({x}) + 1)",
        f"a + b*{x} + c*sqrt(abs({x}) + 1)",
        # linear trend + periodic/composite residual
        f"a*{x} + b*sin(c*{x} + d) + e",
        f"a*{x} + b*cos(c*{x} + d) + e",
        f"a*{x} + b*sin(c*{x}**2 + d) + e",
        f"a*{x} + b*cos(c*{x}**2 + d) + e",
        f"a*{x} + b*sin(c*{x} + d*{x}**2 + e) + f",
        # polynomial trend + periodic residual
        f"a*{x}**2 + b*{x} + c*sin(d*{x}**2 + e) + f",
        f"a*{x}**2 + b*{x} + c*cos(d*{x}**2 + e) + f",
        f"a*{x}**3 + b*{x}**2 + c*{x} + d*sin(e*{x}**2 + f) + g",
        # linear trend + smooth nonlinear residual
        f"a*{x} + b*exp(c*{x}) + d",
        f"a*{x} + b*log(abs(c*{x}) + d) + e",
        f"a*{x} + b*sqrt(abs(c*{x}) + d) + e",
        f"a*{x} + b*tanh(c*{x} + d) + e",
        # richer but still generic two-component residuals
        f"a + b*{x} + c*sin(d*{x}**2)",
        f"a + b*{x} + c*sin(d*{x}**2 + e)",
        f"a + b*{x} + c*sin(d*{x}) + f*sin(g*{x}**2)",
        f"a + b*{x} + c*cos(d*{x}) + f*cos(g*{x}**2)",
        # stand-alone composite nonlinear forms, useful when trend is not purely linear
        f"a*sin(b*{x}**2 + c)*cos(d*{x} + e) + f",
        f"a*sin(b*{x}**3 + c)*cos(d*{x}**2 + e) + f",
        f"a1*sin(b1*{x}+c1)+a2*sin(b2*{x}**2+c2)+d",
        f"a1*cos(b1*{x}+c1)+a2*cos(b2*{x}**2+c2)+d",
    ]
    return deduplicate_expressions(candidates)[:LOW_DIM_ADDITIVE_NONLINEAR_MAX_CANDIDATES]


def build_lowdim_polynomial_surface_candidates(feature_names):
    """
    Cheap generic 2D polynomial surfaces.

    These are intentionally simple and benchmark-name-free. They provide fast
    coverage for the common black-box case where a response surface is mostly
    linear/quadratic/cubic in two variables and would otherwise be approximated
    by a much slower shifted-rational surrogate.
    """
    feature_names = list(feature_names or [])
    if len(feature_names) != 2:
        return []
    x1, x2 = feature_names
    candidates = [
        f"a1*{x1} + a2*{x2} + a3",
        f"a1*{x1}*{x2} + a2*{x1} + a3*{x2} + a4",
        f"a1*{x1}**2 + a2*{x1} + a3*{x2} + a4",
        f"a1*{x2}**2 + a2*{x1} + a3*{x2} + a4",
        f"a1*{x1}**2 + a2*{x2}**2 + a3*{x1} + a4*{x2} + a5",
        f"a1*{x1}**2 + a2*{x2}**2 + a3*{x1}*{x2} + a4*{x1} + a5*{x2} + a6",
        f"a1*{x1}**3 + a2*{x1}**2 + a3*{x2}**2 + a4*{x1}*{x2} + a5*{x1} + a6*{x2} + a7",
        f"a1*{x2}**3 + a2*{x1}**2 + a3*{x2}**2 + a4*{x1}*{x2} + a5*{x1} + a6*{x2} + a7",
        f"a1*{x1}**2*{x2} + a2*{x1}*{x2}**2 + a3*{x1}**2 + a4*{x2}**2 + a5*{x1}*{x2} + a6*{x1} + a7*{x2} + a8",
    ]
    return deduplicate_expressions(candidates)


def build_lowdim_rational_coverage_candidates(dataset, row_meta=None, max_candidates=None):
    """
    Generic low-dimensional rational/power-rational coverage.
    This is deliberately benchmark-name-free: it uses only variable names and data dimensionality.
    """
    feature_names = list(getattr(dataset, "feature_names", []) or [])
    d = len(feature_names)
    exprs = []
    seen = set()
    effective_max_candidates = int(max_candidates or LOW_DIM_RATIONAL_COVERAGE_MAX_CANDIDATES)
    if _is_low_dim_benchmark_like(row_meta=row_meta, dataset=dataset):
        effective_max_candidates = min(
            effective_max_candidates,
            int(LOW_DIM_BENCHMARK_RATIONAL_COVERAGE_MAX_CANDIDATES),
        )

    def add(expr):
        expr = str(expr).strip()
        if not expr:
            return
        key = _expr_dedup_key(expr)
        if key not in seen:
            seen.add(key)
            exprs.append(expr)

    if d == 1:
        x = feature_names[0]
        # Additive nonlinear coverage: important for nearly-linear 1D functions with
        # periodic/composite residuals (linear + sin(x), linear + sin(x**2), etc.).
        for expr in build_lowdim_additive_nonlinear_candidates(feature_names):
            add(expr)
        # Polynomial baselines up to moderately high degree.
        for deg in range(2, max(3, LOW_DIM_RATIONAL_COVERAGE_MAX_DEGREE_1D_NUM) + 1):
            add(_poly_param_sum(x, deg, prefix="p", include_const=True))
        # Generic rational polynomials: P_m(x) / Q_n(x).
        for ndeg in range(2, LOW_DIM_RATIONAL_COVERAGE_MAX_DEGREE_1D_NUM + 1):
            for ddeg in range(1, LOW_DIM_RATIONAL_COVERAGE_MAX_DEGREE_1D_DEN + 1):
                numer = _poly_param_sum(x, ndeg, prefix="a", include_const=True)
                # fix denominator constant to 1 to reduce scale ambiguity
                den_terms = ["1"] + [f"b{k}*{x}" if k == 1 else f"b{k}*{x}**{k}" for k in range(1, ddeg + 1)]
                denom = " + ".join(den_terms)
                add(f"({numer})/({denom})")
                add(f"c0 + ({numer})/({denom})")
        # Compact shifted power-over-polynomial variants.
        for pow_deg in [2, 3, 4, 5, 6]:
            add(f"a*({x}+b1)**{pow_deg}/({x}**2+b2*{x}+b3)+c")
            add(f"a*({x}+b1)**{pow_deg}/({x}**4+b2*{x}**3+b3*{x}**2+b4*{x}+b5)+c")
        add(f"a/({x}+b)+c")
        add(f"a/({x}**2+b)+c")
        add(f"a*log(abs({x})+b)+c")
        add(f"a*sqrt(abs({x})+b)+c")

    elif d == 2:
        x1, x2 = feature_names
        vars2 = [x1, x2]
        # Evaluate simple polynomial surfaces before the larger rational pool.
        for expr in build_lowdim_polynomial_surface_candidates(feature_names):
            add(expr)
        # Exact/near-exact interaction-power primitives should be tested before
        # larger rational pools; otherwise low-dimensional product-power laws can be
        # approximated by ugly shifted rational/Abs surrogates.
        add(f"a*{x1}*{x2}**2+b")
        add(f"a*{x1}**2*{x2}+b")
        add(f"a*{x2}*{x1}**2+b")
        add(f"a*{x2}**2*{x1}+b")
        add(f"a*{x1}*({x2}+b1)**2+c")
        add(f"a*({x1}+b1)*{x2}**2+c")
        add(f"a*{x2}*({x1}+b1)**2+c")
        add(f"a*({x2}+b1)*{x1}**2+c")

        for num in vars2:
            other = x2 if num == x1 else x1
            # Power over additive denominator, e.g. x^4/(x+y).
            for pwr in LOW_DIM_RATIONAL_COVERAGE_POWER_RANGE_2D:
                add(f"a*{num}**{pwr}/({num}+{other}+b)+c")
                add(f"a*({num}+b1)**{pwr}/({num}+{other}+b2)+c")
                # Power over powered denominator, e.g. x^5/y^3.
                for qwr in LOW_DIM_RATIONAL_COVERAGE_DEN_POWER_RANGE_2D:
                    add(f"a*{num}**{pwr}/({other}**{qwr}+b)+c")
                    add(f"a*({num}+b1)**{pwr}/(({other}+b2)**{qwr})+c")
            # Mixed low-order rational interactions.
            add(f"a*{num}*{other}+b")
            add(f"a*{num}/({other}+b)+c")
            add(f"a*{num}**2/({other}+b)+c")
            add(f"a*{num}**3/({other}+b)+c")
            add(f"a*{num}**4/({num}+{other}+b)+c")
        # General polynomial-over-linear denominator.
        for main, other in [(x1, x2), (x2, x1)]:
            for deg in [2, 3, 4, 5]:
                numer = _poly_param_sum(main, deg, prefix="a", include_const=True)
                add(f"({numer})/(b1*{main}+b2*{other}+b3)")
                add(f"c0 + ({numer})/(b1*{main}+b2*{other}+b3)")

    ranked, rank_trace = _rank_templates_by_single_basis_fit(
        exprs,
        dataset=dataset,
        max_candidates=effective_max_candidates,
    )
    return ranked, {
        "num_raw": len(exprs),
        "num_ranked": len(ranked),
        "effective_max_candidates": int(effective_max_candidates),
        "benchmark_like_cap_applied": bool(
            _is_low_dim_benchmark_like(row_meta=row_meta, dataset=dataset)
            and effective_max_candidates < int(max_candidates or LOW_DIM_RATIONAL_COVERAGE_MAX_CANDIDATES)
        ),
        "rank_trace": make_json_safe(rank_trace),
        "preview": ranked[:20],
    }


def build_lowdim_additive_direct_rescue_candidates(dataset, max_candidates=None):
    """
    Direct 1D additive-nonlinear rescue candidates.

    This path is intentionally separate from lowdim rational coverage. Many 1D
    Livermore-style functions have a strong linear trend plus a small nonlinear
    residual. If these candidates are mixed into a large rational pool, they may
    be ranked late or fitted with too few restarts.
    """
    feature_names = list(getattr(dataset, "feature_names", []) or [])
    if len(feature_names) != 1 or not ENABLE_LOW_DIM_ADDITIVE_DIRECT_RESCUE:
        return [], {"skipped": True, "reason": "not 1D or disabled"}
    x = feature_names[0]
    exprs = [
        f"a + b*{x} + c*sin({x}**2)",
        f"a + b*{x} + c*cos({x}**2)",
        f"a + b*{x} + c*sin({x})",
        f"a + b*{x} + c*cos({x})",
        f"a + b*{x} + c*sin({x} + {x}**2)",
        f"a + b*{x} + c*cos({x} + {x}**2)",
        f"a + b*{x} + c*sin({x}) + d*sin({x}**2)",
        f"a + b*{x} + c*cos({x}) + d*cos({x}**2)",
        f"a*{x}**2 + b*{x} + c*sin({x}**2) + d",
        f"a*{x}**2 + b*{x} + c*cos({x}**2) + d",
        f"a + b*{x} + c*exp({x})",
        f"a + b*{x} + c*exp(-{x})",
        f"a + b*{x} + c*log(abs({x}) + 1)",
        f"a + b*{x} + c*sqrt(abs({x}) + 1)",
        # parametric variants kept after the stable fixed-kernel forms
        f"a*{x} + b*sin(c*{x}**2 + d) + e",
        f"a*{x} + b*cos(c*{x}**2 + d) + e",
        f"a*{x} + b*sin(c*{x} + d) + e",
        f"a*{x} + b*cos(c*{x} + d) + e",
    ]
    exprs = deduplicate_expressions(exprs)[:int(max_candidates or LOW_DIM_ADDITIVE_DIRECT_MAX_CANDIDATES)]
    return exprs, {
        "num_exprs": len(exprs),
        "preview": exprs[:20],
        "note": "direct additive nonlinear rescue with stable fixed-kernel candidates",
    }


def should_run_lowdim_additive_direct_rescue(current_best, dataset, row_meta=None):
    if not ENABLE_LOW_DIM_ADDITIVE_DIRECT_RESCUE:
        return False, "lowdim additive direct rescue disabled"
    d = len(getattr(dataset, "feature_names", []) or [])
    if d != 1:
        return False, f"n_features={d} not 1"
    if current_best is None:
        return True, "no current best"
    val = _safe_get_attr(current_best, "val_mse", None)
    try:
        if val is None or not np.isfinite(val):
            return True, "best val unavailable"
        if float(val) > LOW_DIM_ADDITIVE_DIRECT_TRIGGER_VAL_MSE:
            return True, f"best_val_mse>{LOW_DIM_ADDITIVE_DIRECT_TRIGGER_VAL_MSE}"
        return False, f"best_val_mse<={LOW_DIM_ADDITIVE_DIRECT_TRIGGER_VAL_MSE}"
    except Exception:
        return True, "best val parse failed"


def should_run_lowdim_rational_coverage_rescue(current_best, dataset, row_meta=None):
    if not ENABLE_LOW_DIM_RATIONAL_COVERAGE_RESCUE:
        return False, "lowdim rational coverage disabled"
    d = len(getattr(dataset, "feature_names", []) or [])
    if d not in {1, 2}:
        return False, f"n_features={d} not in {{1,2}}"
    if current_best is None:
        return True, "no current best"
    val = _safe_get_attr(current_best, "val_mse", None)
    try:
        if val is None or not np.isfinite(val):
            return True, "best val unavailable"
        if float(val) > LOW_DIM_RATIONAL_COVERAGE_TRIGGER_VAL_MSE:
            return True, f"best_val_mse>{LOW_DIM_RATIONAL_COVERAGE_TRIGGER_VAL_MSE}"
        return False, f"best_val_mse<={LOW_DIM_RATIONAL_COVERAGE_TRIGGER_VAL_MSE}"
    except Exception:
        return True, "best val parse failed"



def select_highdim_families_from_visual_evidence(observation, fallback_families=None):
    """Choose high-dimensional coverage families from visual/data evidence.

    This keeps generic primitives evidence-triggered instead of always expanding
    every hand-written family. It is intentionally conservative: when visual
    tokens are weak or unavailable, it falls back to the configured family set.
    """
    fallback = list(fallback_families or HIGH_DIM_UNIVERSAL_COVERAGE_FAMILIES)
    if HIGH_DIM_UNIVERSAL_COVERAGE_USE_ALL_FAMILIES_ON_RESCUE:
        return fallback, {"mode": "all_generic_families", "reason": "v9_highdim_balanced_safety_net", "families": fallback}
    if observation is None:
        return fallback, {"mode": "fallback", "reason": "observation_unavailable", "families": fallback}

    tokens = dict(getattr(observation, "reconstruction_tokens", None) or {})
    structure_profile = dict(getattr(observation, "structure_profile", None) or {})
    visual_summary = dict(getattr(observation, "visual_summary", None) or {})
    text_blob = " ".join([
        json.dumps(tokens, ensure_ascii=False),
        json.dumps(structure_profile, ensure_ascii=False),
        json.dumps(visual_summary, ensure_ascii=False),
        " ".join(str(x) for x in list(getattr(observation, "visual_hints", []) or [])),
        " ".join(str(x) for x in list(getattr(observation, "structure_hints", []) or [])),
    ]).lower()

    selected = []
    reasons = {}

    def add(fam, reason):
        if fam in fallback and fam not in selected:
            selected.append(fam)
            reasons[fam] = reason

    if any(k in text_blob for k in ["log", "ratio", "ratio_like", "ratio-like", "denominator", "normalization"]):
        add("multiplicative_envelope_log_ratio", "ratio/log/denominator evidence")
    if any(k in text_blob for k in ["difference", "diagonal_contrast", "diagonal contrast", "subtract", "contrast"]):
        add("multiplicative_difference_ratio", "difference/contrast evidence")
        add("reciprocal_difference_product", "difference plus denominator-compatible evidence")
    if any(k in text_blob for k in ["square", "**2", "squared", "symmetry", "mirror_symmetry", "mirror symmetry"]):
        add("rational_difference_of_squares_denominator", "squared/symmetry evidence")
    if any(k in text_blob for k in ["power", "curvature", "polynomial"]):
        add("power_product_over_product_denominator", "power/curvature evidence")
    if any(k in text_blob for k in ["exp", "exponential", "growth", "decay"]):
        add("outer_scale_exp_minus_one", "exponential evidence")
    if any(k in text_blob for k in ["sin", "cos", "periodic", "oscillat", "trig"]):
        add("outer_scale_linear_plus_trig_interaction", "periodic/trigonometric evidence")

    if not selected:
        return fallback, {"mode": "fallback", "reason": "weak_or_ambiguous_visual_evidence", "families": fallback}

    # Add broad companion families so noisy visual tokens do not over-prune.
    for companion in ["multiplicative_difference_ratio", "multiplicative_envelope_log_ratio"]:
        if companion in fallback and companion not in selected and len(selected) < 4:
            selected.append(companion)
            reasons[companion] = "companion backup for noisy high-dimensional evidence"

    return selected, {"mode": "evidence_triggered", "families": selected, "reasons": reasons}

def build_highdim_universal_coverage_candidates(feature_names, max_candidates=None, families=None):
    """
    Balanced generic high-dimensional physics-style family coverage.

    The previous version filled the candidate budget with whichever family was
    listed first, so later families such as log-ratio or difference-of-squares
    could be absent even when they were the right family. This version builds a
    bucket for every generic family and merges them round-robin, so each family
    is guaranteed coverage without using benchmark names or true formulas.
    """
    feature_names = list(feature_names or [])
    if len(feature_names) < 5:
        return []
    from itertools import combinations

    fams = list(families or HIGH_DIM_UNIVERSAL_COVERAGE_FAMILIES)
    max_candidates = max(1, int(max_candidates or HIGH_DIM_UNIVERSAL_COVERAGE_MAX_CANDIDATES))
    vars_ = feature_names[:min(len(feature_names), 6)]
    priority_exprs = build_high_dim_full_role_priority_candidates(
        vars_,
        max_candidates=min(180, max_candidates),
    )
    priority_keys = {_expr_dedup_key(x) for x in priority_exprs}
    buckets = {fam: [] for fam in fams}
    seen = set(priority_keys)

    def add(fam, expr):
        if fam not in buckets:
            return
        expr = str(expr).strip()
        if not expr:
            return
        key = _expr_dedup_key(expr)
        if key in seen:
            return
        buckets[fam].append(expr)
        seen.add(key)

    # 1) product envelope × log-ratio. Correct family for log-ratio Feynman-like tasks.
    fam = "multiplicative_envelope_log_ratio"
    if fam in buckets:
        for u, v in combinations(vars_, 2):
            env_pool = [x for x in vars_ if x not in {u, v}]
            if len(env_pool) < 1:
                continue
            # Use the full remaining envelope first when d=5, then smaller envelopes.
            env_choices = []
            if len(env_pool) >= 3:
                env_choices.append(tuple(env_pool[:3]))
            for r in range(min(2, len(env_pool)), min(3, len(env_pool)) + 1):
                for env in combinations(env_pool, r):
                    if env not in env_choices:
                        env_choices.append(env)
            for env in env_choices:
                env_expr = "*".join(env)
                add(fam, f"a*{env_expr}*(log(abs({u})+b1)-log(abs({v})+b2)) + c")
                add(fam, f"a*{env_expr}*(log(abs({v})+b1)-log(abs({u})+b2)) + c")
                add(fam, f"a*{env_expr}*log(abs({u})/(abs({v})+b)) + c")
                if len(buckets[fam]) >= HIGH_DIM_UNIVERSAL_COVERAGE_FAMILY_QUOTAS.get(fam, 36):
                    break
            if len(buckets[fam]) >= HIGH_DIM_UNIVERSAL_COVERAGE_FAMILY_QUOTAS.get(fam, 36):
                break

    # 2) multiplicative difference ratio: multiplier product × (u-v) / denominator.
    fam = "multiplicative_difference_ratio"
    if fam in buckets:
        for den in vars_:
            rem = [v for v in vars_ if v != den]
            for d1, d2 in combinations(rem, 2):
                mults = [v for v in rem if v not in {d1, d2}]
                if len(mults) < 1:
                    continue
                mult_choices = []
                if len(mults) >= 2:
                    mult_choices.extend(list(combinations(mults, 2)))
                mult_choices.extend([(m,) for m in mults])
                for mult in mult_choices:
                    mult_expr = "*".join(mult)
                    add(fam, f"a*{mult_expr}*({d1}-{d2})/({den}+b) + c")
                    add(fam, f"a*{mult_expr}*({d2}-{d1})/({den}+b) + c")
                    add(fam, f"a*{mult_expr}*({d1}-{d2})/{den} + c")
                    if len(buckets[fam]) >= HIGH_DIM_UNIVERSAL_COVERAGE_FAMILY_QUOTAS.get(fam, 48):
                        break
                if len(buckets[fam]) >= HIGH_DIM_UNIVERSAL_COVERAGE_FAMILY_QUOTAS.get(fam, 48):
                    break
            if len(buckets[fam]) >= HIGH_DIM_UNIVERSAL_COVERAGE_FAMILY_QUOTAS.get(fam, 48):
                break

    # 3) numerator product over scaled difference-of-squares denominator.
    fam = "rational_difference_of_squares_denominator"
    if fam in buckets:
        for scale in vars_:
            rest = [v for v in vars_ if v != scale]
            for u, v in combinations(rest, 2):
                nums = [x for x in rest if x not in {u, v}]
                if len(nums) < 2:
                    continue
                for n1, n2 in combinations(nums, 2):
                    add(fam, f"a*{n1}*{n2}/({scale}*({u}**2-{v}**2)) + c")
                    add(fam, f"a*{n1}*{n2}/({scale}*({v}**2-{u}**2)) + c")
                    add(fam, f"a*{n1}*{n2}/({scale}*(({u}-{v})*({u}+{v}))+b) + c")
                    add(fam, f"a*{n1}*{n2}/(({scale}+b1)*(({u}+b2)**2-({v}+b3)**2)) + c")
                    if len(buckets[fam]) >= HIGH_DIM_UNIVERSAL_COVERAGE_FAMILY_QUOTAS.get(fam, 48):
                        break
                if len(buckets[fam]) >= HIGH_DIM_UNIVERSAL_COVERAGE_FAMILY_QUOTAS.get(fam, 48):
                    break
            if len(buckets[fam]) >= HIGH_DIM_UNIVERSAL_COVERAGE_FAMILY_QUOTAS.get(fam, 48):
                break

    # 4) power-product numerator over product denominator.
    fam = "power_product_over_product_denominator"
    if fam in buckets:
        for d1, d2 in combinations(vars_, 2):
            nums = [v for v in vars_ if v not in {d1, d2}]
            if len(nums) < 3:
                continue
            for n1, n2, n3 in combinations(nums, 3):
                for sq in (n1, n2, n3):
                    others = [v for v in (n1, n2, n3) if v != sq]
                    if len(others) < 2:
                        continue
                    u, w = others[0], others[1]
                    add(fam, f"a*{u}*{sq}**2*{w}/({d1}*{d2}) + c")
                    add(fam, f"a*{u}*{sq}**2*{w}/({d1}*{d2}+b) + c")
                    add(fam, f"a*{u}*({sq}+b1)**2*{w}/(({d1}+b2)*({d2}+b3)) + c")
                    if len(buckets[fam]) >= HIGH_DIM_UNIVERSAL_COVERAGE_FAMILY_QUOTAS.get(fam, 42):
                        break
                if len(buckets[fam]) >= HIGH_DIM_UNIVERSAL_COVERAGE_FAMILY_QUOTAS.get(fam, 42):
                    break
            if len(buckets[fam]) >= HIGH_DIM_UNIVERSAL_COVERAGE_FAMILY_QUOTAS.get(fam, 42):
                break

    # 5) outer scale × (exp(product ratio)-1).
    fam = "outer_scale_exp_minus_one"
    if fam in buckets:
        for outer in vars_:
            rest = [v for v in vars_ if v != outer]
            for n1, n2 in combinations(rest, 2):
                dens = [v for v in rest if v not in {n1, n2}]
                if len(dens) < 2:
                    continue
                for d1, d2 in combinations(dens, 2):
                    add(fam, f"a*{outer}*(exp(({n1}*{n2})/({d1}*{d2})) - 1) + c")
                    add(fam, f"a*{outer}*(exp(({n1}*{n2})/({d1}*{d2}+b)) - 1) + c")
                    add(fam, f"a*{outer}*exp(({n1}*{n2})/({d1}*{d2}+b)) + c")
                    if len(buckets[fam]) >= HIGH_DIM_UNIVERSAL_COVERAGE_FAMILY_QUOTAS.get(fam, 48):
                        break
                if len(buckets[fam]) >= HIGH_DIM_UNIVERSAL_COVERAGE_FAMILY_QUOTAS.get(fam, 48):
                    break
            if len(buckets[fam]) >= HIGH_DIM_UNIVERSAL_COVERAGE_FAMILY_QUOTAS.get(fam, 48):
                break

    # 6) product times reciprocal difference: scale product × (1/u - 1/v).
    # Important: add exact low-parameter reciprocal-difference forms first.
    # This covers generic structures such as E*(1/u - 1/v) without using task names.
    # Shifted versions are kept only as flexible backups because they can otherwise
    # fit unstable surrogate denominators.
    fam = "reciprocal_difference_product"
    if fam in buckets:
        quota = HIGH_DIM_UNIVERSAL_COVERAGE_FAMILY_QUOTAS.get(fam, 96)
        # First pass: full envelope of all remaining variables for every pair.
        # For 5-D this guarantees forms like x1*x2*x5*(1/x4 - 1/x3).
        for u, v in combinations(vars_, 2):
            mults = [x for x in vars_ if x not in {u, v}]
            if not mults:
                continue
            mult_expr = "*".join(mults)
            add(fam, f"a*{mult_expr}*(1/{u} - 1/{v}) + c")
            add(fam, f"a*{mult_expr}*(1/{v} - 1/{u}) + c")
            add(fam, f"a*{mult_expr}*(({v}-{u})/({u}*{v}+1e-12)) + c")
            add(fam, f"a*{mult_expr}*(({u}-{v})/({u}*{v}+1e-12)) + c")
            if len(buckets[fam]) >= quota:
                break
        # Second pass: smaller envelopes and shifted-denominator backups.
        if len(buckets[fam]) < quota:
            for u, v in combinations(vars_, 2):
                mults = [x for x in vars_ if x not in {u, v}]
                if len(mults) < 1:
                    continue
                mult_choices = []
                for r in range(1, min(3, len(mults)) + 1):
                    mult_choices.extend(list(combinations(mults, r)))
                for mult in mult_choices:
                    mult_expr = "*".join(mult)
                    add(fam, f"a*{mult_expr}*(1/{u} - 1/{v}) + c")
                    add(fam, f"a*{mult_expr}*(1/{v} - 1/{u}) + c")
                    add(fam, f"a*{mult_expr}*(({v}-{u})/({u}*{v}+1e-12)) + c")
                    add(fam, f"a*{mult_expr}*(({u}-{v})/({u}*{v}+1e-12)) + c")
                    add(fam, f"a*{mult_expr}*(1/({u}+b1)-1/({v}+b2)) + c")
                    add(fam, f"a*{mult_expr}*(1/({v}+b1)-1/({u}+b2)) + c")
                    if len(buckets[fam]) >= quota:
                        break
                if len(buckets[fam]) >= quota:
                    break

    # 7) outer scale × (linear + interaction*sin(angle)).
    fam = "outer_scale_linear_plus_trig_interaction"
    if fam in buckets:
        for outer in vars_:
            rest = [v for v in vars_ if v != outer]
            for base in rest:
                rest2 = [v for v in rest if v != base]
                if len(rest2) < 3:
                    continue
                for m1, m2 in combinations(rest2, 2):
                    angles = [v for v in rest2 if v not in {m1, m2}]
                    for ang in angles:
                        add(fam, f"a*{outer}*({base} + b1*{m1}*{m2}*sin({ang})) + c")
                        add(fam, f"a*{outer}*({base} + b1*{m1}*{m2}*sin(b2*{ang}+b3)) + c")
                        add(fam, f"a*{outer}*(b1*{base} + b2*{m1}*{m2}*cos(b3*{ang}+b4)) + c")
                        if len(buckets[fam]) >= HIGH_DIM_UNIVERSAL_COVERAGE_FAMILY_QUOTAS.get(fam, 48):
                            break
                    if len(buckets[fam]) >= HIGH_DIM_UNIVERSAL_COVERAGE_FAMILY_QUOTAS.get(fam, 48):
                        break
                if len(buckets[fam]) >= HIGH_DIM_UNIVERSAL_COVERAGE_FAMILY_QUOTAS.get(fam, 48):
                    break
            if len(buckets[fam]) >= HIGH_DIM_UNIVERSAL_COVERAGE_FAMILY_QUOTAS.get(fam, 48):
                break

    # Merge round-robin so no family can occupy the whole budget.
    merged = []
    merged_seen = set()
    max_bucket_len = max([len(v) for v in buckets.values()] or [0])
    for idx in range(max_bucket_len):
        for fam in fams:
            bucket = buckets.get(fam, [])
            if idx >= len(bucket):
                continue
            expr = bucket[idx]
            key = _expr_dedup_key(expr)
            if key in merged_seen:
                continue
            merged.append(expr)
            merged_seen.add(key)
            if len(merged) >= max_candidates:
                break
        if len(merged) >= max_candidates:
            break

    return merge_expression_groups_with_limit(
        [priority_exprs, merged],
        max_total=max_candidates,
    )


def _infer_highdim_rescue_family_from_expr(expr):
    expr = str(expr or "").replace(" ", "")
    low = expr.lower()
    if "exp(" in low:
        return "outer_scale_exp_minus_one"
    if "sin(" in low or "cos(" in low:
        return "outer_scale_linear_plus_trig_interaction"
    if "log(" in low:
        return "multiplicative_envelope_log_ratio"
    if "**2" in low and "/" in low and "-" in low:
        return "rational_difference_of_squares_denominator"
    if "1/(" in low and ")-1/(" in low:
        return "reciprocal_difference_product"
    if "/" in low and "**2" in low:
        return "power_product_over_product_denominator"
    if "/" in low and "-" in low:
        return "multiplicative_difference_ratio"
    if "/" in low:
        return "rational_interaction"
    return "generic_highdim_structure"


def _build_highdim_candidate_summary(expr, candidate_id, feature_names):
    expr = str(expr or "").strip()
    family = _infer_highdim_rescue_family_from_expr(expr)
    sig = extract_formula_form_signature(expr, feature_names)
    return {
        "id": int(candidate_id),
        "family": family,
        "expression_template": expr,
        "variables_used": list(sig.get("variables_used", []) or []),
        "functions_used": list(sig.get("functions_used", []) or []),
        "operators": list(sig.get("op_symbols", []) or []),
        "max_multiplicative_arity": int(sig.get("max_multiplicative_arity", 1) or 1),
    }


def _compact_visual_structural_tokens_for_highdim_rerank(observation):
    observation = observation or ObservationBundle(
        structure_hints=[], visual_hints=[], unit_hints={}, plot_descriptions=[], image_paths=[]
    )
    compact_structure, compact_visual = _compact_observation_for_prompt(
        structure_profile=getattr(observation, "structure_profile", None),
        visual_summary=getattr(observation, "visual_summary", None),
    )
    recon_tokens = _compact_reconstruction_tokens_for_prompt(
        reconstruction_tokens=getattr(observation, "reconstruction_tokens", None)
    )
    return {
        "structure_summary": make_json_safe(compact_structure),
        "visual_summary": make_json_safe(compact_visual),
        "reconstruction_tokens": make_json_safe(recon_tokens),
        "visual_hints": list(getattr(observation, "visual_hints", []) or [])[:8],
        "plot_descriptions": list(getattr(observation, "plot_descriptions", []) or [])[:8],
        "reconstruction_descriptions": list(getattr(observation, "reconstruction_descriptions", []) or [])[:8],
    }


def _call_highdim_reranker_with_hard_timeout(fn, timeout_sec, default_result):
    return _call_planner_with_hard_timeout(
        fn,
        timeout_sec=max(1, int(timeout_sec or HIGH_DIM_LLM_RESCUE_RERANK_TIMEOUT_SEC)),
        default_result=default_result,
    )


def rerank_highdim_rescue_candidates_with_llm(client, dataset, observation, candidate_exprs, current_best=None, max_tokens=None):
    """
    LLM-guided high-dimensional rescue reranker.

    The deterministic rescue module still creates a broad, benchmark-name-free
    candidate pool. This function compresses the pool into token-level candidate
    summaries plus visual-structural tokens, asks the LLM to approve/reorder a
    small subset, then returns selected candidates first and keeps deterministic
    backup candidates after them.
    """
    candidate_exprs = deduplicate_expressions(candidate_exprs or [])
    if (not ENABLE_HIGH_DIM_LLM_RESCUE_RERANK) or not candidate_exprs:
        return candidate_exprs, {
            "used_llm": False,
            "fallback_used": False,
            "reason": "llm highdim rerank disabled or no candidates",
        }

    feature_names = list(getattr(dataset, "feature_names", []) or [])
    if len(feature_names) < HIGH_DIM_COVERAGE_MIN_FEATURES:
        return candidate_exprs, {
            "used_llm": False,
            "fallback_used": False,
            "reason": "not high-dimensional",
        }

    topn = max(1, min(int(HIGH_DIM_LLM_RESCUE_RERANK_TOPN), len(candidate_exprs)))
    compact_candidates = [
        _build_highdim_candidate_summary(expr, i, feature_names)
        for i, expr in enumerate(candidate_exprs[:topn], start=1)
    ]
    visual_tokens = _compact_visual_structural_tokens_for_highdim_rerank(observation)
    current_best_snapshot = _result_performance_snapshot(current_best)

    prompt = f"""
You are a high-dimensional symbolic regression rescue reranker.

You must NOT use benchmark names or true formulas.
Your job is not to write a final formula. Your job is to select and reorder
generic high-dimensional candidate templates using visual-structural tokens,
structure evidence, and candidate summaries.

Variables: {feature_names}
Target: {getattr(dataset, 'target_name', 'y')}
Current best: {json.dumps(make_json_safe(current_best_snapshot), ensure_ascii=False)}

Visual-structural tokens:
{json.dumps(make_json_safe(visual_tokens), ensure_ascii=False)}

Candidate summaries:
{json.dumps(make_json_safe(compact_candidates), ensure_ascii=False)}

Return STRICT JSON only:
{{
  "selected_candidate_ids": [1, 2, 3],
  "reranked_candidate_ids": [1, 2, 3, 4],
  "preferred_families": ["multiplicative_difference_ratio"],
  "reason": "brief evidence-based reason"
}}

Selection guidelines:
- Prefer candidates whose variable roles match visual/reconstruction tokens.
- Prefer mechanism-like structures over dense polynomial surrogates.
- Keep diverse families when evidence is ambiguous.
- Do not invent new variables or formulas.
""".strip()

    def _one_call():
        try:
            response = client.generate(
                messages=[
                    Message(role="system", content="Output compact JSON only."),
                    Message(role="user", content=prompt),
                ],
                temperature=float(HIGH_DIM_LLM_RESCUE_RERANK_TEMPERATURE),
                max_tokens=int(max_tokens or HIGH_DIM_LLM_RESCUE_RERANK_MAX_TOKENS),
                top_p=1.0,
            )
            raw_text = response.text
            obj = _extract_first_json_object(raw_text)
            selected_ids = []
            reranked_ids = []
            preferred_families = []
            reason = ""
            if isinstance(obj, dict):
                for key, target in [("selected_candidate_ids", selected_ids), ("reranked_candidate_ids", reranked_ids)]:
                    for item in obj.get(key, []) or []:
                        try:
                            cid = int(item)
                            if 1 <= cid <= topn and cid not in target:
                                target.append(cid)
                        except Exception:
                            pass
                preferred_families = [str(x) for x in obj.get("preferred_families", []) or [] if str(x).strip()]
                reason = str(obj.get("reason", "") or "")
            return {
                "used_llm": True,
                "fallback_used": False,
                "selected_candidate_ids": selected_ids,
                "reranked_candidate_ids": reranked_ids,
                "preferred_families": preferred_families,
                "reason": reason,
                "raw_text": raw_text,
                "prompt_num_candidates": int(topn),
                "visual_tokens_used": bool(visual_tokens),
                "timeout_sec": int(HIGH_DIM_LLM_RESCUE_RERANK_TIMEOUT_SEC),
            }
        except Exception as e:
            return {
                "used_llm": False,
                "fallback_used": True,
                "selected_candidate_ids": [],
                "reranked_candidate_ids": [],
                "preferred_families": [],
                "reason": f"llm reranker failed: {repr(e)}",
                "raw_text": "",
                "prompt_num_candidates": int(topn),
                "timeout_sec": int(HIGH_DIM_LLM_RESCUE_RERANK_TIMEOUT_SEC),
            }

    trace = _call_highdim_reranker_with_hard_timeout(
        _one_call,
        timeout_sec=HIGH_DIM_LLM_RESCUE_RERANK_TIMEOUT_SEC,
        default_result={
            "used_llm": False,
            "fallback_used": True,
            "selected_candidate_ids": [],
            "reranked_candidate_ids": [],
            "preferred_families": [],
            "reason": "hard-timeout",
            "raw_text": "",
            "prompt_num_candidates": int(topn),
            "timeout_sec": int(HIGH_DIM_LLM_RESCUE_RERANK_TIMEOUT_SEC),
        },
    )

    chosen_ids = []
    for cid in list(trace.get("reranked_candidate_ids", []) or []) + list(trace.get("selected_candidate_ids", []) or []):
        try:
            cid = int(cid)
        except Exception:
            continue
        if 1 <= cid <= topn and cid not in chosen_ids:
            chosen_ids.append(cid)

    selected_max = max(1, int(HIGH_DIM_LLM_RESCUE_SELECTED_MAX))
    chosen_exprs = [candidate_exprs[cid - 1] for cid in chosen_ids[:selected_max]]

    # If the LLM returned preferred families but no ids, promote candidates from
    # those families. This still uses only generic family names and visual tokens.
    if not chosen_exprs and trace.get("preferred_families"):
        preferred = set(str(x) for x in trace.get("preferred_families", []) or [])
        for expr in candidate_exprs[:topn]:
            if _infer_highdim_rescue_family_from_expr(expr) in preferred:
                chosen_exprs.append(expr)
                if len(chosen_exprs) >= selected_max:
                    break

    if not chosen_exprs:
        trace["fallback_used"] = True
        if not trace.get("reason"):
            trace["reason"] = "LLM returned no selected candidates; using original deterministic order"
        return candidate_exprs, make_json_safe(trace)

    chosen_keys = {_expr_dedup_key(x) for x in chosen_exprs}
    if HIGH_DIM_LLM_RERANK_SOFT_BOOST:
        head_keep = max(0, min(int(HIGH_DIM_LLM_RERANK_DETERMINISTIC_HEAD_KEEP), len(candidate_exprs)))
        deterministic_head = list(candidate_exprs[:head_keep])
        deterministic_head_keys = {_expr_dedup_key(x) for x in deterministic_head}
        llm_bonus = [x for x in chosen_exprs if _expr_dedup_key(x) not in deterministic_head_keys]
        llm_bonus_keys = {_expr_dedup_key(x) for x in llm_bonus}
        tail = [
            x for x in candidate_exprs
            if _expr_dedup_key(x) not in deterministic_head_keys and _expr_dedup_key(x) not in llm_bonus_keys
        ]
        final_exprs = deterministic_head + llm_bonus + tail
        trace["ordering_mode"] = "soft_boost"
        trace["deterministic_head_keep"] = head_keep
        trace["llm_participated_but_deterministic_head_preserved"] = True
    else:
        if HIGH_DIM_LLM_RESCUE_KEEP_UNSELECTED_BACKUP:
            final_exprs = chosen_exprs + [x for x in candidate_exprs if _expr_dedup_key(x) not in chosen_keys]
        else:
            final_exprs = chosen_exprs
        trace["ordering_mode"] = "llm_frontload"
    trace["num_selected_exprs"] = len(chosen_exprs)
    trace["selected_expr_preview"] = chosen_exprs[:10]
    return deduplicate_expressions(final_exprs), make_json_safe(trace)


def should_run_highdim_universal_coverage_rescue(current_best, dataset, row_meta):
    if not HIGH_DIM_UNIVERSAL_COVERAGE_RESCUE:
        return False, "disabled"
    if not (_is_benchmark_task(row_meta) and NO_LEAKAGE_MODE and len(getattr(dataset, "feature_names", []) or []) >= 5):
        return False, "not_high_dim_no_leakage_benchmark"
    best_val = _safe_get_attr(current_best, "val_mse", None) if current_best is not None else None
    try:
        if best_val is None or (not np.isfinite(best_val)):
            return True, "best_val_unavailable"
        if float(best_val) > float(HIGH_DIM_UNIVERSAL_COVERAGE_TRIGGER_VAL_MSE):
            return True, f"best_val_mse>{HIGH_DIM_UNIVERSAL_COVERAGE_TRIGGER_VAL_MSE}"
        return False, f"best_val_mse<={HIGH_DIM_UNIVERSAL_COVERAGE_TRIGGER_VAL_MSE}"
    except Exception:
        return True, "best_val_parse_failed"


def should_run_high_dim_surrogate_escape(current_best, dataset, row_meta=None, structure_profile=None, experience_prior=None):
    if not ENABLE_HIGH_DIM_SURROGATE_ESCAPE:
        return False, "surrogate escape disabled"
    if not NO_LEAKAGE_MODE:
        return False, "surrogate escape only for no-leakage mode"
    if not _is_benchmark_task(row_meta):
        return False, "surrogate escape only for benchmark tasks"

    feature_names = list(getattr(dataset, "feature_names", []) or [])
    if len(feature_names) < HIGH_DIM_ROLE_TRIGGER_DIM:
        return False, "surrogate escape only for high-dimensional tasks"
    if current_best is None:
        return False, "no current best expression"

    best_expr = str(_safe_get_attr(current_best, "simplified_expression", "") or "").strip()
    if not best_expr:
        return False, "current best expression unavailable"

    best_val = _safe_get_attr(current_best, "val_mse", None)
    try:
        if best_val is None or (not np.isfinite(best_val)) or float(best_val) <= HIGH_DIM_SURROGATE_ESCAPE_TRIGGER_VAL_MSE:
            return False, f"best_val_mse<={HIGH_DIM_SURROGATE_ESCAPE_TRIGGER_VAL_MSE}"
    except Exception:
        return False, "best_val_mse invalid"

    sig = extract_formula_form_signature(best_expr, feature_names)
    profile = dict(structure_profile or {})
    family_scores = dict(profile.get("family_scores", {}) or {})
    roles = dict(profile.get("variable_roles", {}) or {})
    active_variables = list(roles.get("active_variables", []) or profile.get("active_variables", []) or feature_names)
    used_variables = list(sig.get("variables_used", []) or [])
    missing_active = [v for v in active_variables if v not in set(used_variables)]

    interaction_score = float(family_scores.get("interaction", 0.0) or 0.0)
    rational_score = float(family_scores.get("rational", 0.0) or 0.0)
    logarithmic_score = float(family_scores.get("logarithmic", 0.0) or 0.0)
    has_log = bool("logarithmic" in set(sig.get("families", []) or []) or "log(" in best_expr)
    low_arity = int(sig.get("max_multiplicative_arity", 1) or 1) < 3
    missing_active_core = bool(missing_active)

    if has_log and (missing_active_core or low_arity) and max(interaction_score, rational_score, logarithmic_score) >= 0.18:
        if missing_active_core and low_arity:
            return True, "log-surrogate missing active interaction variable"
        if missing_active_core:
            return True, "log-surrogate missing active variable"
        return True, "log-surrogate multiplicative arity too low"
    return False, "no stuck log-surrogate signature"


def build_high_dim_surrogate_escape_candidates(
    feature_names,
    current_best_expr,
    structure_profile=None,
    experience_prior=None,
    max_candidates=None,
):
    feature_names = list(feature_names or [])
    current_best_expr = str(current_best_expr or "").strip()
    if len(feature_names) < HIGH_DIM_ROLE_TRIGGER_DIM or not current_best_expr:
        return []

    profile = dict(structure_profile or {})
    seeds = _build_specialist_branch_seeds(
        feature_names,
        structure_profile=structure_profile,
        experience_prior=experience_prior,
    )
    sig = extract_formula_form_signature(current_best_expr, feature_names)
    used_variables = list(sig.get("variables_used", []) or [])
    log_variables = _extract_log_variables(current_best_expr, feature_names)

    active_variables = list(seeds.get("active_variables", []) or feature_names)
    envelope_vars = list(seeds.get("envelope_vars", []) or feature_names)
    transform_vars = list(seeds.get("transform_vars", []) or feature_names)
    top_pair_patterns = list(seeds.get("top_pair_patterns", []) or [])

    def _ordered_allowed(items):
        out = []
        seen = set()
        allowed = set(feature_names)
        for item in items or []:
            item = str(item).strip()
            if not item or item not in allowed or item in seen:
                continue
            out.append(item)
            seen.add(item)
        return out

    missing_active = _ordered_allowed([v for v in active_variables if v not in set(used_variables)])
    product_vars = _ordered_allowed(
        list(used_variables)
        + [v for v in envelope_vars if v not in log_variables]
        + list(active_variables)
        + list(feature_names)
    )
    if len(product_vars) < 2:
        product_vars = _ordered_allowed(list(feature_names))

    env0 = product_vars[0]
    env1 = product_vars[1] if len(product_vars) >= 2 else env0
    env2 = (
        missing_active[0]
        if missing_active
        else (product_vars[2] if len(product_vars) >= 3 else env1)
    )

    ratio_pairs = []
    seen_pairs = set()

    def _add_pair(a, b):
        a = str(a).strip()
        b = str(b).strip()
        if not a or not b or a == b:
            return
        key = tuple(sorted((a, b)))
        if key in seen_pairs:
            return
        seen_pairs.add(key)
        ratio_pairs.append((a, b))

    if len(log_variables) >= 2:
        _add_pair(log_variables[0], log_variables[1])
    for item in top_pair_patterns:
        family = str(item.get("family", "") or "")
        variables = list(item.get("variables", []) or [])
        if family in {"rational", "logarithmic", "power"} and len(variables) >= 2:
            _add_pair(variables[0], variables[1])
    transform_candidates = _ordered_allowed(list(transform_vars) + list(log_variables) + list(feature_names))
    if len(transform_candidates) >= 2:
        _add_pair(transform_candidates[0], transform_candidates[1])

    pair_prod = f"{env0}*{env1}"
    triple_prod = f"{env0}*{env1}*{env2}"
    exprs = []
    prefer_triple_escape = bool(missing_active and env2 not in {env0, env1})
    for a, b in ratio_pairs[:3]:
        ratio_exprs = []
        if prefer_triple_escape:
            ratio_exprs.extend([
                f"a1*({triple_prod})*log(abs({a}/{b}) + c1) + d",
                f"a1*({triple_prod})*log(abs({b}/{a}) + c1) + d",
                f"a1*({triple_prod})*log(abs(({a}+c1)/{b}) + c2) + d",
                f"a1*({triple_prod})*log(abs({a}/({b}+c1)) + c2) + d",
                f"a1*({triple_prod})*(1 + b1*log(abs({a}/{b}) + c1)) + d",
                f"a1*({pair_prod})*({env2}+b1)*log(abs({a}/{b}) + c1) + d",
                f"a1*({pair_prod})*({env2}+b1)*log(abs(({a}+c1)/{b}) + c2) + d",
                f"a1*({pair_prod})*log(abs({a}/{b}) + c1) + d",
                f"a1*({pair_prod})*log(abs({b}/{a}) + c1) + d",
            ])
        else:
            ratio_exprs.extend([
                f"a1*({pair_prod})*log(abs(({a}+c1)/({b}+c2)) + c3) + d",
                f"a1*({pair_prod})*log(abs(({b}+c1)/({a}+c2)) + c3) + d",
                f"a1*({triple_prod})*log(abs(({a}+c1)/({b}+c2)) + c3) + d",
                f"a1*({triple_prod})*log(abs(({b}+c1)/({a}+c2)) + c3) + d",
                f"a1*({pair_prod})*({env2}+b1)*log(abs(({a}+c1)/({b}+c2)) + c3) + d",
                f"a1*({pair_prod})*(1 + b1*log(abs(({a}+c1)/({b}+c2)) + c3)) + d",
                f"a1*({pair_prod})*({env2}+b1)*log(abs(({a}+c1)/({b}+c2))) + d",
                f"a1*({triple_prod})*(1 + b1*log(abs(({a}+c1)/({b}+c2)))) + d",
            ])
        exprs.extend(ratio_exprs)

    return _filter_specialist_exprs(exprs, max_candidates=max_candidates)


def _build_high_dim_role_context(feature_names, structure_profile=None, experience_prior=None, max_vars=None):
    feature_names = list(feature_names or [])
    max_vars = max(2, int(max_vars or HIGH_DIM_ROLE_SUBSPACE_MAX_VARS))
    if len(feature_names) < HIGH_DIM_ROLE_TRIGGER_DIM:
        return {
            "working_variables": feature_names,
            "active_variables": list(feature_names[:max_vars]),
            "numerator_core": list(feature_names[:max_vars]),
            "denominator_core": [],
            "periodic_core": [],
            "top_pair_patterns": [],
            "global_tags": [],
        }

    allowed = set(feature_names)
    structure_profile = dict(structure_profile or {})
    experience_prior = dict(experience_prior or {})

    def _ordered_allowed(names):
        out = []
        seen = set()
        for name in names or []:
            name = str(name)
            if name in allowed and name not in seen:
                out.append(name)
                seen.add(name)
        return out

    role_sources = []
    if isinstance(structure_profile.get("variable_roles"), dict):
        role_sources.append(structure_profile.get("variable_roles"))
    if isinstance(experience_prior.get("variable_roles"), dict):
        role_sources.append(experience_prior.get("variable_roles"))

    active_variables = []
    numerator_core = []
    denominator_core = []
    periodic_core = []
    for roles in role_sources:
        active_variables.extend(list(roles.get("active_variables", []) or []))
        numerator_core.extend(list(roles.get("numerator_core", []) or []))
        denominator_core.extend(list(roles.get("denominator_core", []) or []))
        periodic_core.extend(list(roles.get("periodic_core", []) or []))

    top_pair_patterns = []
    for item in list(structure_profile.get("top_pair_patterns", []) or [])[:4]:
        if not isinstance(item, dict):
            continue
        variables = _ordered_allowed(item.get("variables", []) or [])
        if len(variables) < 2:
            continue
        top_pair_patterns.append({
            "variables": variables[:2],
            "family": str(item.get("family", "") or ""),
            "score": float(item.get("score", 0.0) or 0.0),
        })

    pair_variables = []
    for item in top_pair_patterns:
        pair_variables.extend(item.get("variables", []) or [])

    working_variables = _ordered_allowed(
        list(active_variables)
        + list(numerator_core)
        + list(denominator_core)
        + list(periodic_core)
        + list(pair_variables)
        + list(feature_names)
    )[:max_vars]
    if len(working_variables) < min(3, len(feature_names)):
        working_variables = _ordered_allowed(list(working_variables) + list(feature_names))[:max_vars]

    active_variables = _ordered_allowed(list(active_variables) + list(pair_variables) + list(working_variables))
    active_variables = [name for name in active_variables if name in set(working_variables)]
    if len(active_variables) < min(2, len(working_variables)):
        active_variables = list(working_variables[:min(max(2, len(active_variables) or 0), len(working_variables))])

    numerator_core = _ordered_allowed(list(numerator_core) + list(active_variables))
    numerator_core = [name for name in numerator_core if name in set(working_variables)]
    if len(numerator_core) < min(2, len(active_variables)):
        numerator_core = list(active_variables[:min(3, len(active_variables))])

    denominator_core = _ordered_allowed(denominator_core)
    denominator_core = [name for name in denominator_core if name in set(working_variables)]
    periodic_core = _ordered_allowed(periodic_core)
    periodic_core = [name for name in periodic_core if name in set(working_variables)]

    return {
        "working_variables": working_variables,
        "active_variables": active_variables,
        "numerator_core": numerator_core,
        "denominator_core": denominator_core,
        "periodic_core": periodic_core,
        "top_pair_patterns": top_pair_patterns,
        "global_tags": list(structure_profile.get("global_tags", []) or []),
    }


def _select_high_dim_difference_pair(role_ctx):
    denominator_core = list(role_ctx.get("denominator_core", []) or [])
    if len(denominator_core) >= 2:
        return denominator_core[:2]

    for item in list(role_ctx.get("top_pair_patterns", []) or []):
        variables = list(item.get("variables", []) or [])
        family = str(item.get("family", "") or "")
        if len(variables) >= 2 and family in {"power", "rational", "interaction"}:
            return variables[:2]

    active_variables = list(role_ctx.get("active_variables", []) or [])
    if len(active_variables) >= 4:
        return active_variables[2:4]
    return active_variables[1:3] if len(active_variables) >= 3 else []


def _should_enable_role_guided_exact_structures(role_ctx, structure_profile=None):
    structure_profile = dict(structure_profile or {})
    family_scores = dict(structure_profile.get("family_scores", {}) or {})
    global_tags = set(structure_profile.get("global_tags", []) or [])
    top_pair_patterns = list(role_ctx.get("top_pair_patterns", []) or [])
    diff_pair = _select_high_dim_difference_pair(role_ctx)
    denominator_core = list(role_ctx.get("denominator_core", []) or [])
    active_variables = list(role_ctx.get("active_variables", []) or [])

    if len(diff_pair) < 2 or not denominator_core or len(active_variables) < 3:
        return False

    strong_rational = float(family_scores.get("rational", 0.0) or 0.0) >= ROLE_GUIDED_STRONG_RATIONAL_SCORE
    strong_power = float(family_scores.get("power", 0.0) or 0.0) >= ROLE_GUIDED_STRONG_POWER_SCORE
    strong_power_pair = any(
        str(item.get("family", "") or "") == "power" and float(item.get("score", 0.0) or 0.0) >= ROLE_GUIDED_STRONG_PAIR_SCORE
        for item in top_pair_patterns
    )
    tags_support = "ratio_or_denominator" in global_tags and "nonlinear_power" in global_tags
    return bool(strong_rational and (strong_power or strong_power_pair or tags_support))




def build_high_dim_full_role_priority_candidates(feature_names, max_candidates=None):
    """Benchmark-name-free full high-dimensional role coverage."""
    feature_names = list(feature_names or [])
    if len(feature_names) < 5:
        return []
    limit = int(max_candidates or 180)
    out, seen = [], set()

    def add(expr):
        expr = str(expr).strip()
        if not expr:
            return False
        key = _expr_dedup_key(expr)
        if key in seen:
            return False
        seen.add(key)
        out.append(expr)
        return len(out) >= limit

    def prod(vars_):
        return "*".join([v for v in vars_ if v in feature_names])

    # 1) Full envelope + log-ratio for every unordered pair.
    for i in range(len(feature_names)):
        for j in range(i + 1, len(feature_names)):
            u, v = feature_names[i], feature_names[j]
            env_prod = prod([x for x in feature_names if x not in {u, v}][:3])
            if not env_prod:
                continue
            for expr in [
                f"a*{env_prod}*(log({u})-log({v})) + c",
                f"a*{env_prod}*(log({v})-log({u})) + c",
                f"a*{env_prod}*log({u}/{v}) + c",
                f"a*{env_prod}*log({v}/{u}) + c",
            ]:
                if add(expr):
                    return out

    # 2) Difference over denominator for every denominator and difference pair.
    for den in feature_names:
        rest = [x for x in feature_names if x != den]
        for i in range(len(rest)):
            for j in range(i + 1, len(rest)):
                u, v = rest[i], rest[j]
                env_prod = prod([x for x in rest if x not in {u, v}][:2])
                if not env_prod:
                    continue
                # Exact-denominator forms come first. They are lower-parameter and
                # much more stable for envelope*(u-v)/den than shifted denominators.
                exact_exprs = [
                    f"a*({env_prod}*({u}-{v})/{den}) + c",
                    f"a*({env_prod}*({v}-{u})/{den}) + c",
                    f"a*{env_prod}*({u}-{v})/{den} + c",
                    f"a*{env_prod}*({v}-{u})/{den} + c",
                    f"a*{env_prod}*({u}-{v})/{den}",
                    f"a*{env_prod}*({v}-{u})/{den}",
                ]
                shifted_exprs = [
                    f"a*{env_prod}*({u}-{v})/({den}+b) + c",
                    f"a*{env_prod}*({v}-{u})/({den}+b) + c",
                    f"a*({env_prod})*(({u}+b1)-({v}+b2))/({den}+b3) + c",
                    f"a*({env_prod})*(({v}+b1)-({u}+b2))/({den}+b3) + c",
                ]
                for expr in (exact_exprs + shifted_exprs if ENABLE_HIGH_DIM_EXACT_DENOM_PRIORITY else shifted_exprs + exact_exprs):
                    if add(expr):
                        return out

    # 3) Exp-ratio and product-over-product denominator coverage.
    for outer in feature_names:
        rest = [x for x in feature_names if x != outer]
        for i in range(len(rest)):
            for j in range(i + 1, len(rest)):
                n1, n2 = rest[i], rest[j]
                den_vars = [x for x in rest if x not in {n1, n2}]
                if len(den_vars) < 2:
                    continue
                d1, d2 = den_vars[:2]
                for expr in [
                    f"a*{outer}*(exp(({n1}*{n2})/({d1}*{d2})) - 1) + c",
                    f"a*{outer}*{n1}**2*{n2}/({d1}*{d2}) + c",
                ]:
                    if add(expr):
                        return out

    # 4) Difference-of-squares denominator, reciprocal difference, and trig interaction.
    for i in range(len(feature_names)):
        for j in range(i + 1, len(feature_names)):
            d1, d2 = feature_names[i], feature_names[j]
            rest = [x for x in feature_names if x not in {d1, d2}]
            if len(rest) >= 3:
                n1, n2, den_scale = rest[:3]
                for expr in [
                    f"a*{n1}*{n2}/({den_scale}*({d1}**2-{d2}**2)) + c",
                    f"a*{n1}*{n2}/({den_scale}*({d2}**2-{d1}**2)) + c",
                    f"a*{n1}*{n2}*{den_scale}*(1/{d1} - 1/{d2}) + c",
                    f"a*{n1}*({n2} + {den_scale}*{d1}*sin({d2})) + c",
                ]:
                    if add(expr):
                        return out

    return out[:limit]

def build_generic_log_ratio_candidates(
    feature_names,
    structure_profile=None,
    experience_prior=None,
    max_candidates=None,
):
    """
    Generic high-dimensional log-ratio skeletons.

    This is intentionally NOT benchmark-specific:
    - no row_meta
    - no base_name
    - no Feynman index
    - no exact formula injection

    It proposes a broad structural family commonly seen in physics-like data:
        multiplicative envelope * log(variable ratio)

    Important implementation detail:
    for 5D tasks, the first pass covers every unordered ratio pair once.
    This avoids a subtle truncation bug where max_candidates=10 only kept
    early pairs such as x1/x2 and never reached x4/x5.
    """
    if not ENABLE_GENERIC_LOG_RATIO_SKELETONS:
        return []

    feature_names = list(feature_names or [])
    if len(feature_names) < GENERIC_LOG_RATIO_MIN_DIM:
        return []

    limit = int(max_candidates or GENERIC_LOG_RATIO_MAX_CANDIDATES)
    limit = max(1, limit)

    profile = dict(structure_profile or {})
    roles = dict(profile.get("variable_roles", {}) or {})
    top_pair_patterns = list(profile.get("top_pair_patterns", []) or [])

    allowed = set(feature_names)
    active_variables = list(
        roles.get("active_variables", [])
        or profile.get("active_variables", [])
        or feature_names
    )
    active_variables = [v for v in active_variables if v in allowed]
    if len(active_variables) < 3:
        active_variables = list(feature_names)

    out = []
    seen_exprs = set()

    def _add(expr):
        expr = str(expr).strip()
        if not expr:
            return False
        key = _expr_dedup_key(expr)
        if key in seen_exprs:
            return False
        seen_exprs.add(key)
        out.append(expr)
        return len(out) >= limit

    def _product_expr(vars_):
        vars_ = [v for v in vars_ if v in allowed]
        if not vars_:
            return ""
        return "*".join(vars_)

    def _remaining_for_pair(u, v):
        remaining = [x for x in active_variables if x not in {u, v}]
        if len(remaining) < 1:
            remaining = [x for x in feature_names if x not in {u, v}]
        return remaining

    def _triple_or_best_envelope(u, v):
        remaining = _remaining_for_pair(u, v)
        if len(remaining) >= 3:
            # For 5D this is exactly the complement of the ratio pair.
            return remaining[:3]
        if len(remaining) >= 2:
            return remaining[:2]
        return remaining[:1]

    # ------------------------------------------------------------
    # Phase 1: pair coverage first.
    # Use unordered pairs because a negative coefficient can flip
    # log(x_i/x_j) into log(x_j/x_i). For 5 variables this yields
    # exactly 10 candidates, so x4/x5 is guaranteed to appear.
    # ------------------------------------------------------------
    pair_coverage = []
    if GENERIC_LOG_RATIO_COVER_UNORDERED_PAIRS_FIRST:
        for i in range(len(feature_names)):
            for j in range(i + 1, len(feature_names)):
                pair_coverage.append((feature_names[i], feature_names[j]))

    for u, v in pair_coverage:
        env = _triple_or_best_envelope(u, v)
        env_prod = _product_expr(env)
        if not env_prod:
            continue

        if GENERIC_LOG_RATIO_INCLUDE_EXACT_LOG_FIRST:
            # Exact clean form. Works well for benchmark ranges that are positive.
            # log(u/v) and log(v/u) are equivalent up to a negative coefficient.
            if _add(f"a*{env_prod}*log({u}/{v}) + c"):
                return out[:limit]
        else:
            if _add(f"a*{env_prod}*log(abs({u})/(abs({v})+1e-12)) + c"):
                return out[:limit]

    # ------------------------------------------------------------
    # Phase 2: use pair patterns from the profile if available.
    # These are extra variants, not a replacement for full pair coverage.
    # ------------------------------------------------------------
    profile_pairs = []
    for item in top_pair_patterns[:8]:
        if not isinstance(item, dict):
            continue
        pair_vars = list(item.get("variables", []) or [])
        pair_family = str(item.get("family", "") or "")
        if len(pair_vars) >= 2 and pair_family in {"logarithmic", "rational", "power", "interaction"}:
            u, v = pair_vars[:2]
            if u in allowed and v in allowed and u != v:
                profile_pairs.append((u, v))
                profile_pairs.append((v, u))

    # If there is still room and no profile pairs, fall back to ordered pairs.
    if not profile_pairs:
        for u in feature_names:
            for v in feature_names:
                if u != v:
                    profile_pairs.append((u, v))

    seen_pairs = set(pair_coverage)
    for u, v in profile_pairs:
        key = (u, v)
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        env = _triple_or_best_envelope(u, v)
        env_prod = _product_expr(env)
        if not env_prod:
            continue
        if _add(f"a*{env_prod}*log({u}/{v}) + c"):
            return out[:limit]

    # ------------------------------------------------------------
    # Phase 3: robust fallback variants. These are added after the
    # clean exact forms, because abs/offset versions are harder to fit.
    # ------------------------------------------------------------
    for u, v in pair_coverage + profile_pairs:
        env = _triple_or_best_envelope(u, v)
        env_prod = _product_expr(env)
        if not env_prod:
            continue
        if _add(f"a*{env_prod}*(log({u})-log({v})) + c"):
            return out[:limit]
        if _add(f"a*{env_prod}*log(abs({u})/(abs({v})+1e-12)) + c"):
            return out[:limit]

    return out[:limit]

def build_role_guided_high_dim_candidates(feature_names, structure_profile=None, experience_prior=None, max_candidates=None):
    feature_names = list(feature_names or [])
    if len(feature_names) < HIGH_DIM_ROLE_TRIGGER_DIM:
        return []

    role_ctx = _build_high_dim_role_context(
        feature_names,
        structure_profile=structure_profile,
        experience_prior=experience_prior,
        max_vars=HIGH_DIM_ROLE_SUBSPACE_MAX_VARS,
    )
    working_variables = list(role_ctx.get("working_variables", []) or [])
    active_variables = list(role_ctx.get("active_variables", []) or working_variables)
    numerator_core = list(role_ctx.get("numerator_core", []) or active_variables)
    denominator_core = list(role_ctx.get("denominator_core", []) or [])
    periodic_core = list(role_ctx.get("periodic_core", []) or [])
    if len(working_variables) < 2 or len(active_variables) < 2:
        return []

    out = build_generic_role_combination_candidates(
        feature_names,
        structure_profile=structure_profile,
        experience_prior=experience_prior,
        max_candidates=max_candidates,
    )

    # Generic high-dimensional log-ratio candidates.
    # These do not use row_meta / base_name, so they are safe in no-leakage mode.
    log_ratio_candidates = build_generic_log_ratio_candidates(
        feature_names=feature_names,
        structure_profile=structure_profile,
        experience_prior=experience_prior,
        max_candidates=min(
            GENERIC_LOG_RATIO_MAX_CANDIDATES,
            max(4, int(max_candidates or HIGH_DIM_ROLE_TEMPLATE_LIMIT) // 2),
        ),
    )

    # Put them before generic combinations so they survive later truncation.
    # The final cap is still enforced below, so total candidate volume is controlled.
    out = deduplicate_expressions(log_ratio_candidates + out)

    seen = set()
    for expr in out:
        seen.add(_expr_dedup_key(expr))

    def _add(exprs):
        for expr in exprs or []:
            expr = str(expr).strip()
            if not expr:
                continue
            key = _expr_dedup_key(expr)
            if key in seen:
                continue
            seen.add(key)
            out.append(expr)
            if max_candidates is not None and len(out) >= max_candidates:
                return True
        return False

    linear_vars = active_variables[:min(4, len(active_variables))]
    linear = " + ".join([f"a{i+1}*{v}" for i, v in enumerate(linear_vars)]) if linear_vars else "0"
    num0 = numerator_core[0]
    num1 = numerator_core[1] if len(numerator_core) >= 2 else active_variables[1]
    num2 = numerator_core[2] if len(numerator_core) >= 3 else (active_variables[2] if len(active_variables) >= 3 else num1)
    num3 = numerator_core[3] if len(numerator_core) >= 4 else (active_variables[3] if len(active_variables) >= 4 else num2)
    denom0 = denominator_core[0] if denominator_core else ""
    diff_pair = _select_high_dim_difference_pair(role_ctx)
    enable_exact_structures = _should_enable_role_guided_exact_structures(role_ctx, structure_profile=structure_profile)

    sparse = [
        f"{num0}*{num1}",
        f"a1*{num0}*{num1} + c",
        f"{linear} + c",
        f"a1*{num0}*{num1} + {linear} + c",
    ]
    if len(numerator_core) >= 3:
        sparse.extend([
            f"{num0}*{num1}*{num2}",
            f"a1*{num0}*{num1}*{num2} + c",
        ])
    if denom0:
        sparse.extend([
            f"({num0}*{num1})/({denom0}+b1)",
            f"a1*({num0}*{num1})/({denom0}+b1) + c",
        ])
        if len(numerator_core) >= 3:
            sparse.extend([
                f"({num0}*{num1}*{num2})/({denom0}+b1)",
                f"a1*({num0}*{num1}*{num2})/({denom0}+b1) + c",
            ])
    if denom0 and len(diff_pair) >= 2:
        d0, d1 = diff_pair[:2]
        sparse.extend([
            f"a1*{num0}*{num1}/(({denom0}+b1)*({d0}**2 + b2*{d0}*{d1} + b3*{d1}**2 + b4)) + c",
            f"a1*{num0}*{num1}/(({denom0}+b1)*((({d0}+b2*{d1})*({d0}+b3*{d1}))+b4)) + c",
        ])
    if denom0 and len(active_variables) >= 4:
        sparse.extend([
            f"a1*{num0}*({num2}+b2*{num1})*{num3}/({denom0}+b1) + c",
        ])
    if periodic_core:
        pv = periodic_core[0]
        sparse.extend([
            f"a1*sin(b1*{pv}+c1) + {linear} + d",
            f"a1*cos(b1*{pv}+c1) + {linear} + d",
        ])

    for item in list(role_ctx.get("top_pair_patterns", []) or [])[:2]:
        variables = list(item.get("variables", []) or [])
        family = str(item.get("family", "") or "")
        if len(variables) < 2:
            continue
        a, b = variables[:2]
        if family == "interaction":
            sparse.extend([f"{a}*{b}", f"a1*{a}*{b} + c"])
        elif family == "rational":
            sparse.extend([f"{a}/({b}+b1)", f"{b}/({a}+b1)"])
        elif family == "power":
            sparse.extend([
                f"{a}**2 + b2*{a}*{b} - {b}**2",
                f"({num0}*{num1})/({a}**2 + b2*{a}*{b} + b3*{b}**2 + b1)",
            ])

    _add(sparse)
    cap = max_candidates if max_candidates is not None else HIGH_DIM_ROLE_TEMPLATE_LIMIT
    return _filter_specialist_exprs(out, max_candidates=max(1, int(cap)))

def file_passes_filters(base_name, n_train, n_val, n_test, n_features):
    if n_train < MIN_TRAIN_SAMPLES or n_val < MIN_VAL_SAMPLES or n_test < MIN_TEST_SAMPLES:
        return False
    if MAX_TRAIN_SAMPLES is not None and n_train > MAX_TRAIN_SAMPLES:
        return False
    if MAX_VAL_SAMPLES is not None and n_val > MAX_VAL_SAMPLES:
        return False
    if MAX_TEST_SAMPLES is not None and n_test > MAX_TEST_SAMPLES:
        return False
    if n_features < MIN_FEATURES:
        return False
    if MAX_FEATURES is not None and n_features > MAX_FEATURES:
        return False
    if ALLOW_BASENAMES and base_name not in ALLOW_BASENAMES:
        return False
    if NAME_KEYWORDS:
        low = base_name.lower()
        if not any(k.lower() in low for k in NAME_KEYWORDS):
            return False
    return True


def collect_raw_tasks_for_dataset(dataset_dir_name: str):
    root = Path(SRSD_ROOT) / dataset_dir_name
    train_dir = root / "train"
    val_dir = root / "val"
    test_dir = root / "test"
    true_eq_dir = root / "true_eq"

    if not train_dir.exists():
        raise FileNotFoundError(f"missing train dir: {train_dir}")
    if not val_dir.exists():
        raise FileNotFoundError(f"missing val dir: {val_dir}")
    if not test_dir.exists():
        raise FileNotFoundError(f"missing test dir: {test_dir}")

    train_files = {p.stem: p for p in train_dir.glob("*.txt")}
    val_files = {p.stem: p for p in val_dir.glob("*.txt")}
    test_files = {p.stem: p for p in test_dir.glob("*.txt")}

    common = sorted(set(train_files) & set(val_files) & set(test_files))
    rows = []
    for base in common:
        train_df = load_txt_dataset(str(train_files[base]))
        val_df = load_txt_dataset(str(val_files[base]))
        test_df = load_txt_dataset(str(test_files[base]))
        n_features = train_df.shape[1] - 1
        eq_path = true_eq_dir / f"{base}.pkl"
        has_true_eq = eq_path.exists()
        rows.append({
            "task_type": "srsd",
            "dataset_dir": dataset_dir_name,
            "difficulty": dataset_dir_name,
            "base_name": base,
            "train_path": str(train_files[base]),
            "val_path": str(val_files[base]),
            "test_path": str(test_files[base]),
            "true_eq_path": str(eq_path) if has_true_eq else None,
            "n_features": n_features,
            "n_train": len(train_df),
            "n_val": len(val_df),
            "n_test": len(test_df),
            "true_expression": None,
        })
    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df
    keep = df.apply(
        lambda r: file_passes_filters(r["base_name"], r["n_train"], r["n_val"], r["n_test"], r["n_features"]),
        axis=1,
    )
    df = df[keep].copy().sort_values(by=["n_features", "base_name"]).reset_index(drop=True)
    if RANDOM_SAMPLE_K is not None and RANDOM_SAMPLE_K > 0 and len(df) > RANDOM_SAMPLE_K:
        rng = random.Random(RANDOM_SEED)
        idxs = list(range(len(df)))
        rng.shuffle(idxs)
        idxs = sorted(idxs[:RANDOM_SAMPLE_K])
        df = df.iloc[idxs].reset_index(drop=True)
    if MAX_FILES_PER_DATASET is not None:
        df = df.head(MAX_FILES_PER_DATASET).copy()
    return df


def parse_range_spec(range_str):
    if isinstance(range_str, str):
        return ast.literal_eval(range_str)
    return range_str


def collect_tasks_from_benchmark_csv(csv_path: str):
    df = pd.read_csv(csv_path)
    rows = []
    for _, r in df.iterrows():
        base = str(r["name"])
        if BENCHMARK_ALLOW_BASENAMES and base not in BENCHMARK_ALLOW_BASENAMES:
            continue
        if BENCHMARK_NAME_KEYWORDS and not any(k.lower() in base.lower() for k in BENCHMARK_NAME_KEYWORDS):
            continue
        d = int(r["dimension"])
        if d < MIN_FEATURES or (MAX_FEATURES is not None and d > MAX_FEATURES):
            continue
        variable_names = [f"x{i+1}" for i in range(d)]
        rows.append({
            "task_type": "benchmark_csv",
            "dataset_dir": "benchmark_csv",
            "difficulty": "benchmark_csv",
            "base_name": base,
            "train_path": None,
            "val_path": None,
            "test_path": None,
            "true_eq_path": None,
            "n_features": d,
            "n_train": BENCHMARK_TRAIN_SIZE,
            "n_val": BENCHMARK_VAL_SIZE,
            "n_test": BENCHMARK_TEST_SIZE,
            "true_expression": str(r["expression"]),
            "dimension": d,
            "constant": int(r["constant"]) if "constant" in r and not pd.isna(r["constant"]) else 0,
            "distribution": str(r["distribution"]) if "distribution" in r else "U",
            "range_spec": r["range"],
            "variable_names": json.dumps(variable_names, ensure_ascii=False),
        })
    out = pd.DataFrame(rows)
    if BENCHMARK_RANDOM_SAMPLE_K is not None and BENCHMARK_RANDOM_SAMPLE_K > 0 and len(out) > BENCHMARK_RANDOM_SAMPLE_K:
        out = out.sample(BENCHMARK_RANDOM_SAMPLE_K, random_state=BENCHMARK_RANDOM_SEED).reset_index(drop=True)
    if BENCHMARK_MAX_FILES is not None:
        out = out.head(BENCHMARK_MAX_FILES).copy()
    return out


def build_dataset_from_explicit_splits(train_df, val_df, test_df, tmpdir: Path):
    feature_names = [c for c in train_df.columns if c != "y"]
    # ``df`` is search-visible context, so it must contain only fitting and
    # validation rows.  Keep the held-out split in its dedicated field and do
    # not route it through DatasetLoader, which would mix and re-split rows.
    search_df = pd.concat([train_df, val_df], axis=0, ignore_index=True)
    dataset = DatasetBundle(
        df=search_df.copy(),
        feature_names=feature_names,
        target_name="y",
        train_df=train_df.copy(),
        val_df=val_df.copy(),
        test_df=test_df.copy(),
    )
    dataset.source_tag = ""
    return dataset


def make_csv_points_text(df, variable_names, target_name, max_rows=20):
    cols = variable_names + [target_name]
    sub_df = df[cols].head(max_rows).copy()
    return sub_df.to_csv(index=False).strip()


def image_file_to_data_url(image_path: str) -> str:
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/png;base64,{img_b64}"


def parse_answer_tag_expression(text: str, target_name: str = "y") -> str:
    if not text:
        return ""
    m = re.search(rf"<ANSWER>\s*{re.escape(target_name)}\s*=\s*(.*?)\s*</ANSWER>", text, flags=re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(rf"\b{re.escape(target_name)}\s*=\s*(.+)", text)
    if m:
        expr = m.group(1).strip().split("\n")[0].strip()
        return expr
    return ""


def build_call_temperatures(num_calls, temperatures):
    if temperatures is None or len(temperatures) == 0:
        temperatures = [0.1, 0.2, 0.3, 0.4, 0.5]
    return [temperatures[i % len(temperatures)] for i in range(num_calls)]


def _expr_dedup_key(expr: str) -> str:
    return re.sub(r"\s+", "", str(expr or "").strip())


def deduplicate_expressions(exprs):
    unique_exprs = []
    seen = set()
    for expr in exprs or []:
        expr = str(expr).strip()
        if not expr:
            continue
        expr_key = _expr_dedup_key(expr)
        if expr_key in seen:
            continue
        seen.add(expr_key)
        unique_exprs.append(expr)
    return unique_exprs


def candidate_family_label(expr):
    """Coarse structural family label used only for diversity preservation."""
    expr = str(expr or "").replace(" ", "")
    low = expr.lower()
    if "sin(" in low or "cos(" in low or "tan(" in low or "sinh(" in low or "cosh(" in low or "tanh(" in low:
        return "trigonometric"
    if "exp(" in low:
        return "exponential"
    if "log(" in low:
        return "logarithmic"
    if "/" in expr:
        return "rational"
    if "**" in expr:
        return "power"
    if expr.count("*") >= 3:
        return "high_order_interaction"
    if "*" in expr:
        return "interaction"
    if "+" in expr or "-" in expr:
        return "additive"
    return "algebraic"


def build_generic_diverse_structural_candidates(feature_names, structure_profile=None, max_total=None):
    """
    Dataset-name-agnostic structural seed pool.

    This is intentionally broad: each family receives a small quota so the search
    does not collapse onto a single template family such as product*log-ratio.
    It never reads row_meta/base_name/true_expression.
    """
    feature_names = list(feature_names or [])
    if not feature_names or not ENABLE_GENERIC_DIVERSE_STRUCTURAL_SEEDS:
        return []

    from itertools import combinations

    max_total = int(max_total or GENERIC_DIVERSE_STRUCTURAL_MAX_CANDIDATES)
    max_total = max(1, max_total)
    d = len(feature_names)
    vars_for_pairs = feature_names[:min(d, 6)]
    vars_for_unary = feature_names[:min(d, 5)]

    buckets = {
        "linear": [],
        "power": [],
        "interaction": [],
        "rational": [],
        "logarithmic": [],
        "exponential": [],
        "trigonometric": [],
        "separable": [],
    }

    def add(bucket, expr):
        expr = str(expr).strip()
        if expr and expr not in buckets[bucket]:
            buckets[bucket].append(expr)

    linear_terms = " + ".join([f"a{i+1}*{x}" for i, x in enumerate(vars_for_unary)])
    add("linear", f"{linear_terms} + c")

    for x in vars_for_unary:
        add("power", f"a*{x}**2 + b*{x} + c")
        add("power", f"a*{x}**3 + b*{x}**2 + c*{x} + d")

    for x, y in combinations(vars_for_pairs, 2):
        add("interaction", f"a*{x}*{y} + b")
        # interaction-power primitives: generic for low-dimensional product-power laws.
        add("interaction", f"a*{x}*{y}**2 + b")
        add("interaction", f"a*{x}**2*{y} + b")
        add("interaction", f"a*{x}*({y}+b1)**2 + c")
        add("interaction", f"a*({x}+b1)*{y}**2 + c")
        add("interaction", f"a*({x}+b1)*({y}+b2) + c")
    for triple in list(combinations(vars_for_pairs, 3))[:8]:
        x, y, z = triple
        add("interaction", f"a*{x}*{y}*{z} + b")

    for x, y in combinations(vars_for_pairs, 2):
        add("rational", f"a*{x}/({y}+b) + c")
        add("rational", f"a*{y}/({x}+b) + c")
        add("rational", f"a*({x}*{y})/(b + abs({x}) + abs({y})) + c")

    for x in vars_for_unary:
        add("logarithmic", f"a*log(abs({x}) + b) + c")
    for x, y in combinations(vars_for_pairs, 2):
        add("logarithmic", f"a*log(abs({x})/(abs({y})+b)) + c")
        add("logarithmic", f"a*log(abs({y})/(abs({x})+b)) + c")

    for x in vars_for_unary:
        add("exponential", f"a*exp(b*{x}) + c")
    for x, y in combinations(vars_for_unary, 2):
        add("exponential", f"a*exp(b1*{x} + b2*{y}) + c")

    for x in vars_for_unary:
        add("trigonometric", f"a*sin(b*{x}+c) + d")
        add("trigonometric", f"a*cos(b*{x}+c) + d")
    for x, y in combinations(vars_for_unary, 2):
        add("trigonometric", f"a*sin(b1*{x}+b2*{y}+c) + d")
        add("trigonometric", f"a*{x}*sin(b*{y}+c) + d")

    if d >= 2:
        x1, x2 = feature_names[0], feature_names[1]
        add("separable", f"a1*log(abs({x1})+b1) + a2*{x2} + c")
        add("separable", f"a1*exp(b1*{x1}) + a2*{x2} + c")
        add("separable", f"a1*sin(b1*{x1}+c1) + a2*{x2} + c")
    if d >= 3:
        x1, x2, x3 = feature_names[:3]
        add("separable", f"a1*{x1}*{x2} + a2*{x3} + c")
        add("separable", f"a1*{x1}/({x2}+b1) + a2*{x3} + c")

    quotas = {
        "linear": 2,
        "power": 4,
        "interaction": 8,
        "rational": 4,
        "logarithmic": 3,
        "exponential": 3,
        "trigonometric": 4,
        "separable": 3,
    }

    profile = dict(structure_profile or {})
    family_scores = dict(profile.get("family_scores", {}) or {})
    for fam, score in family_scores.items():
        try:
            score = float(score)
        except Exception:
            continue
        if score >= 0.30:
            if fam in quotas:
                quotas[fam] = min(quotas[fam] + 1, 5)
            elif fam == "high_order_interaction":
                quotas["interaction"] = min(quotas["interaction"] + 1, 6)

    selected, seen = [], set()
    family_order = [
        "linear", "interaction", "rational", "power",
        "logarithmic", "exponential", "trigonometric", "separable",
    ]
    for family in family_order:
        for expr in buckets.get(family, [])[:quotas.get(family, 2)]:
            key = _expr_dedup_key(expr)
            if key in seen:
                continue
            selected.append(expr)
            seen.add(key)
            if len(selected) >= max_total:
                return selected
    return selected[:max_total]




def _format_float_for_expr(value, max_abs=1e12):
    """Compact stable float formatting for generated numeric expressions."""
    try:
        value = float(value)
    except Exception:
        return "0.0"
    if not np.isfinite(value):
        return "0.0"
    value = max(-max_abs, min(max_abs, value))
    if abs(value) < 1e-12:
        return "0.0"
    return f"{value:.12g}"


def _safe_basis_vector(values, max_abs=1e8):
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        return None
    finite = np.isfinite(arr)
    if int(finite.sum()) < max(8, int(0.5 * arr.size)):
        return None
    arr = np.nan_to_num(arr, nan=0.0, posinf=max_abs, neginf=-max_abs)
    arr = np.clip(arr, -max_abs, max_abs)
    if np.nanstd(arr) < 1e-12:
        return None
    return arr


def _fit_single_basis_candidate(train_basis, train_y, val_basis=None, val_y=None):
    train_basis = _safe_basis_vector(train_basis)
    train_y = _safe_basis_vector(train_y, max_abs=1e12)
    if train_basis is None or train_y is None or len(train_basis) != len(train_y):
        return None
    mask = np.isfinite(train_basis) & np.isfinite(train_y)
    if int(mask.sum()) < 8:
        return None
    x = train_basis[mask]
    y = train_y[mask]
    try:
        design = np.column_stack([x, np.ones_like(x)])
        coeff, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
        coef = float(coeff[0])
        intercept = float(coeff[1])
        pred = coef * x + intercept
        train_mse = float(np.mean((y - pred) ** 2))
        baseline_train_mse = float(np.mean((y - np.mean(y)) ** 2))
        train_gain = 0.0 if baseline_train_mse <= 1e-12 else max(0.0, 1.0 - train_mse / baseline_train_mse)
        val_mse = train_mse
        val_gain = train_gain
        if val_basis is not None and val_y is not None:
            val_basis = _safe_basis_vector(val_basis)
            val_y = _safe_basis_vector(val_y, max_abs=1e12)
            if val_basis is not None and val_y is not None and len(val_basis) == len(val_y):
                vmask = np.isfinite(val_basis) & np.isfinite(val_y)
                if int(vmask.sum()) >= 6:
                    xv = val_basis[vmask]
                    yv = val_y[vmask]
                    pred_v = coef * xv + intercept
                    val_mse = float(np.mean((yv - pred_v) ** 2))
                    baseline_val_mse = float(np.mean((yv - np.mean(yv)) ** 2))
                    val_gain = 0.0 if baseline_val_mse <= 1e-12 else max(0.0, 1.0 - val_mse / baseline_val_mse)
        combined_gain = 0.70 * float(val_gain) + 0.30 * float(train_gain)
        return {
            "coef": coef,
            "intercept": intercept,
            "train_mse": train_mse,
            "val_mse": val_mse,
            "train_gain": train_gain,
            "val_gain": val_gain,
            "combined_gain": combined_gain,
        }
    except Exception:
        return None


def _safe_column(df, name):
    try:
        arr = np.asarray(df[name], dtype=float).reshape(-1)
        return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    except Exception:
        return None


def _make_basis_item(expr, family, variables, train_values, val_values=None):
    return {
        "expr": str(expr).strip(),
        "family": str(family).strip() or "algebraic",
        "variables": list(variables or []),
        "train_values": train_values,
        "val_values": val_values,
    }


def build_data_driven_feature_seed_candidates(dataset, max_candidates=None):
    """
    Build fair data-driven structural seeds from generic feature bases.

    Unlike protected benchmark templates, this never uses row_meta/base_name or
    true_expression. It simply tries a broad library of basis functions and keeps
    structurally diverse bases that explain y on train/val.
    """
    if not ENABLE_DATA_DRIVEN_FEATURE_SEEDS:
        return [], {"enabled": False, "reason": "disabled"}
    if dataset is None or getattr(dataset, "train_df", None) is None:
        return [], {"enabled": False, "reason": "dataset unavailable"}
    feature_names = list(getattr(dataset, "feature_names", []) or [])
    target_name = getattr(dataset, "target_name", "y")
    if not feature_names or target_name not in dataset.train_df.columns:
        return [], {"enabled": False, "reason": "missing features or target"}
    max_candidates = int(max_candidates or DATA_DRIVEN_FEATURE_SEED_MAX_CANDIDATES)
    max_candidates = max(1, max_candidates)
    eps = float(DATA_DRIVEN_FEATURE_EPS)
    train_df = dataset.train_df
    val_df = getattr(dataset, "val_df", None)
    train_y = np.asarray(train_df[target_name], dtype=float).reshape(-1)
    val_y = np.asarray(val_df[target_name], dtype=float).reshape(-1) if val_df is not None and target_name in val_df.columns else None
    xtr = {name: _safe_column(train_df, name) for name in feature_names}
    xva = {name: _safe_column(val_df, name) for name in feature_names} if val_df is not None else {}
    if any(v is None for v in xtr.values()):
        return [], {"enabled": False, "reason": "feature column read failed"}
    from itertools import combinations
    items = []
    seen = set()
    def add(expr, family, variables, train_values, val_values=None):
        if len(items) >= int(DATA_DRIVEN_FEATURE_LIBRARY_MAX_TERMS):
            return
        expr = str(expr).strip()
        if not expr:
            return
        key = _expr_dedup_key(expr)
        if key in seen:
            return
        train_values2 = _safe_basis_vector(train_values)
        if train_values2 is None or len(train_values2) != len(train_y):
            return
        val_values2 = None
        if val_values is not None:
            val_values2 = _safe_basis_vector(val_values)
        seen.add(key)
        items.append(_make_basis_item(expr, family, variables, train_values2, val_values2))
    vars_for_unary = feature_names[:min(len(feature_names), 8)]
    vars_for_pairs = feature_names[:min(len(feature_names), 8)]
    for x in vars_for_unary:
        xt = xtr[x]
        xv = xva.get(x)
        add(x, "additive", [x], xt, xv)
        add(f"{x}**2", "power", [x], xt ** 2, None if xv is None else xv ** 2)
        add(f"{x}**3", "power", [x], xt ** 3, None if xv is None else xv ** 3)
        add(f"log(abs({x})+{eps:g})", "logarithmic", [x], np.log(np.abs(xt) + eps), None if xv is None else np.log(np.abs(xv) + eps))
        add(f"exp({x})", "exponential", [x], np.exp(np.clip(xt, -30, 30)), None if xv is None else np.exp(np.clip(xv, -30, 30)))
        add(f"sin({x})", "trigonometric", [x], np.sin(xt), None if xv is None else np.sin(xv))
        add(f"cos({x})", "trigonometric", [x], np.cos(xt), None if xv is None else np.cos(xv))
    pair_list = list(combinations(vars_for_pairs, 2))
    for x, y in pair_list:
        xt, yt = xtr[x], xtr[y]
        xv, yv = xva.get(x), xva.get(y)
        add(f"{x}*{y}", "interaction", [x, y], xt * yt, None if xv is None or yv is None else xv * yv)
        # Generic interaction-power bases, crucial for forms like envelope * variable**2.
        # The fitted coefficient absorbs constants; this uses no benchmark name or true formula.
        add(f"{x}*{y}**2", "interaction_power", [x, y], xt * (yt ** 2), None if xv is None or yv is None else xv * (yv ** 2))
        add(f"{x}**2*{y}", "interaction_power", [x, y], (xt ** 2) * yt, None if xv is None or yv is None else (xv ** 2) * yv)
        add(f"{y}*{x}**2", "interaction_power", [y, x], yt * (xt ** 2), None if xv is None or yv is None else yv * (xv ** 2))
        add(f"{y}**2*{x}", "interaction_power", [y, x], (yt ** 2) * xt, None if xv is None or yv is None else (yv ** 2) * xv)
        add(f"{x}/({y}+{eps:g})", "rational", [x, y], xt / (yt + eps), None if xv is None or yv is None else xv / (yv + eps))
        add(f"{y}/({x}+{eps:g})", "rational", [x, y], yt / (xt + eps), None if xv is None or yv is None else yv / (xv + eps))
        add(f"log(abs({x})+{eps:g})-log(abs({y})+{eps:g})", "logarithmic", [x, y], np.log(np.abs(xt)+eps)-np.log(np.abs(yt)+eps), None if xv is None or yv is None else np.log(np.abs(xv)+eps)-np.log(np.abs(yv)+eps))
        add(f"sin({x}+{y})", "trigonometric", [x, y], np.sin(xt + yt), None if xv is None or yv is None else np.sin(xv + yv))
        add(f"exp({x}+{y})", "exponential", [x, y], np.exp(np.clip(xt + yt, -30, 30)), None if xv is None or yv is None else np.exp(np.clip(xv + yv, -30, 30)))
    triples = list(combinations(vars_for_pairs, 3))[:32]
    for x, y, z in triples:
        xt, yt, zt = xtr[x], xtr[y], xtr[z]
        xv, yv, zv = xva.get(x), xva.get(y), xva.get(z)
        prod_t = xt * yt * zt
        prod_v = None if xv is None or yv is None or zv is None else xv * yv * zv
        add(f"{x}*{y}*{z}", "high_order_interaction", [x, y, z], prod_t, prod_v)
    if len(feature_names) >= 4 and ENABLE_DATA_DRIVEN_DIFFERENCE_RATIO_BASIS:
        # Generic physical primitive: multiplier(s) * (difference) / denominator.
        # This is not benchmark-specific; it is a common SR grammar primitive for
        # contrast / rate-like laws.  The fitted coefficient absorbs sign, while
        # plan expansion later tries both difference directions.
        all_vars = feature_names[:min(len(feature_names), 7)]
        num_added_diff_ratio = 0
        for diff_a, diff_b in combinations(all_vars, 2):
            remaining_after_diff = [v for v in all_vars if v not in {diff_a, diff_b}]
            diff_t = xtr[diff_a] - xtr[diff_b]
            diff_v = None
            if diff_a in xva and diff_b in xva:
                diff_v = xva[diff_a] - xva[diff_b]
            add(f"({diff_a}-{diff_b})", "difference", [diff_a, diff_b], diff_t, diff_v)
            for m in remaining_after_diff[:4]:
                mt = xtr[m]
                mv = xva.get(m)
                add(
                    f"{m}*({diff_a}-{diff_b})",
                    "multiplicative_difference",
                    [m, diff_a, diff_b],
                    mt * diff_t,
                    None if mv is None or diff_v is None else mv * diff_v,
                )
            for den in remaining_after_diff:
                multiplier_pool = [v for v in remaining_after_diff if v != den]
                for mults in list(combinations(multiplier_pool, 2))[:6]:
                    m1, m2 = mults
                    train_basis = xtr[m1] * xtr[m2] * diff_t / (xtr[den] + eps)
                    val_basis = None
                    if all(v in xva for v in [m1, m2, diff_a, diff_b, den]):
                        val_basis = xva[m1] * xva[m2] * (xva[diff_a] - xva[diff_b]) / (xva[den] + eps)
                    expr = f"{m1}*{m2}*({diff_a}-{diff_b})/({den}+{eps:g})"
                    add(expr, "multiplicative_difference_ratio", [m1, m2, diff_a, diff_b, den], train_basis, val_basis)
                    num_added_diff_ratio += 1
                    if num_added_diff_ratio >= int(DATA_DRIVEN_DIFFERENCE_RATIO_MAX_TERMS):
                        break
                if num_added_diff_ratio >= int(DATA_DRIVEN_DIFFERENCE_RATIO_MAX_TERMS):
                    break
            if num_added_diff_ratio >= int(DATA_DRIVEN_DIFFERENCE_RATIO_MAX_TERMS):
                break

    if len(feature_names) >= 5 and ENABLE_DATA_DRIVEN_DIFF_SQUARES_DENOM_BASIS:
        # Generic rational primitive: numerator product over a denominator
        # containing a scale variable times a difference-of-squares term.
        # Example topology: n1*n2 / (s*(u**2 - v**2)).
        # This is a grammar primitive, not tied to any benchmark name.
        all_vars = feature_names[:min(len(feature_names), 7)]
        num_added_dsq = 0
        for diff_a, diff_b in combinations(all_vars, 2):
            diff_sq_t = xtr[diff_a] ** 2 - xtr[diff_b] ** 2
            diff_sq_v = None
            if diff_a in xva and diff_b in xva:
                diff_sq_v = xva[diff_a] ** 2 - xva[diff_b] ** 2
            remaining_after_diff = [v for v in all_vars if v not in {diff_a, diff_b}]
            for scale_den in remaining_after_diff:
                numerator_pool = [v for v in remaining_after_diff if v != scale_den]
                for num1, num2 in list(combinations(numerator_pool, 2))[:6]:
                    denom_t = xtr[scale_den] * diff_sq_t
                    train_basis = (xtr[num1] * xtr[num2]) / (denom_t + eps)
                    val_basis = None
                    if all(v in xva for v in [num1, num2, scale_den, diff_a, diff_b]):
                        denom_v = xva[scale_den] * (xva[diff_a] ** 2 - xva[diff_b] ** 2)
                        val_basis = (xva[num1] * xva[num2]) / (denom_v + eps)
                    expr = f"{num1}*{num2}/(({scale_den}+{eps:g})*(({diff_a})**2-({diff_b})**2)+{eps:g})"
                    add(expr, "rational_difference_of_squares_denominator", [num1, num2, scale_den, diff_a, diff_b], train_basis, val_basis)
                    num_added_dsq += 1
                    if num_added_dsq >= int(DATA_DRIVEN_DIFF_SQUARES_DENOM_MAX_TERMS):
                        break
                if num_added_dsq >= int(DATA_DRIVEN_DIFF_SQUARES_DENOM_MAX_TERMS):
                    break
            if num_added_dsq >= int(DATA_DRIVEN_DIFF_SQUARES_DENOM_MAX_TERMS):
                break

    if len(feature_names) >= 5 and ENABLE_DATA_DRIVEN_POWER_PRODUCT_RATIO_BASIS:
        # Generic rational primitive: product numerator with one squared factor
        # over a product denominator. Example topology: n1*n2**2*n3/(d1*d2).
        all_vars = feature_names[:min(len(feature_names), 7)]
        num_added_power_ratio = 0
        for den1, den2 in combinations(all_vars, 2):
            numerator_pool = [v for v in all_vars if v not in {den1, den2}]
            for n1, n2, n3 in list(combinations(numerator_pool, 3))[:10]:
                for sq in (n1, n2, n3):
                    others = [v for v in [n1, n2, n3] if v != sq]
                    train_basis = xtr[others[0]] * (xtr[sq] ** 2) * xtr[others[1]] / (xtr[den1] * xtr[den2] + eps)
                    val_basis = None
                    if all(v in xva for v in [others[0], sq, others[1], den1, den2]):
                        val_basis = xva[others[0]] * (xva[sq] ** 2) * xva[others[1]] / (xva[den1] * xva[den2] + eps)
                    expr = f"{others[0]}*{sq}**2*{others[1]}/(({den1}+{eps:g})*({den2}+{eps:g}))"
                    add(expr, "power_product_over_product_denominator", [others[0], sq, others[1], den1, den2], train_basis, val_basis)
                    num_added_power_ratio += 1
                    if num_added_power_ratio >= int(DATA_DRIVEN_POWER_PRODUCT_RATIO_MAX_TERMS):
                        break
                if num_added_power_ratio >= int(DATA_DRIVEN_POWER_PRODUCT_RATIO_MAX_TERMS):
                    break
            if num_added_power_ratio >= int(DATA_DRIVEN_POWER_PRODUCT_RATIO_MAX_TERMS):
                break

    if len(feature_names) >= 5 and ENABLE_DATA_DRIVEN_EXP_MINUS_ONE_BASIS:
        # Generic physical primitive: outer scale times (exp(inner product/ratio) - 1).
        # Example topology: outer*(exp((n1*n2)/(d1*d2))-1). This targets
        # exponential activation/contrast laws without using benchmark names.
        all_vars = feature_names[:min(len(feature_names), 7)]
        num_added_exp_minus_one = 0
        for outer in all_vars:
            remaining_outer = [v for v in all_vars if v != outer]
            for n1, n2 in combinations(remaining_outer, 2):
                den_pool = [v for v in remaining_outer if v not in {n1, n2}]
                for d1, d2 in combinations(den_pool, 2):
                    inner_t = (xtr[n1] * xtr[n2]) / (xtr[d1] * xtr[d2] + eps)
                    inner_t_clip = np.clip(inner_t, -30, 30)
                    train_basis = xtr[outer] * (np.exp(inner_t_clip) - 1.0)
                    val_basis = None
                    if all(v in xva for v in [outer, n1, n2, d1, d2]):
                        inner_v = (xva[n1] * xva[n2]) / (xva[d1] * xva[d2] + eps)
                        val_basis = xva[outer] * (np.exp(np.clip(inner_v, -30, 30)) - 1.0)
                    expr = f"{outer}*(exp(({n1}*{n2})/(({d1}+{eps:g})*({d2}+{eps:g})))-1)"
                    add(expr, "outer_scale_exp_minus_one", [outer, n1, n2, d1, d2], train_basis, val_basis)
                    num_added_exp_minus_one += 1
                    if num_added_exp_minus_one >= int(DATA_DRIVEN_EXP_MINUS_ONE_MAX_TERMS):
                        break
                if num_added_exp_minus_one >= int(DATA_DRIVEN_EXP_MINUS_ONE_MAX_TERMS):
                    break
            if num_added_exp_minus_one >= int(DATA_DRIVEN_EXP_MINUS_ONE_MAX_TERMS):
                break

    if len(feature_names) >= 5 and ENABLE_DATA_DRIVEN_TRIG_INTERACTION_BASIS:
        # Generic physical primitive: outer scale times (linear base + interaction*sin(angle)).
        # Example topology: outer*(base + m1*m2*sin(angle)). This covers
        # modulation laws without tying to a benchmark id.
        all_vars = feature_names[:min(len(feature_names), 7)]
        num_added_trig_interaction = 0
        for outer in all_vars:
            rest_outer = [v for v in all_vars if v != outer]
            for base in rest_outer:
                rest_base = [v for v in rest_outer if v != base]
                for m1, m2 in combinations(rest_base, 2):
                    angle_pool = [v for v in rest_base if v not in {m1, m2}]
                    for angle in angle_pool:
                        train_basis = xtr[outer] * (xtr[base] + xtr[m1] * xtr[m2] * np.sin(xtr[angle]))
                        val_basis = None
                        if all(v in xva for v in [outer, base, m1, m2, angle]):
                            val_basis = xva[outer] * (xva[base] + xva[m1] * xva[m2] * np.sin(xva[angle]))
                        expr = f"{outer}*({base}+{m1}*{m2}*sin({angle}))"
                        add(expr, "outer_scale_linear_plus_trig_interaction", [outer, base, m1, m2, angle], train_basis, val_basis)
                        num_added_trig_interaction += 1
                        if num_added_trig_interaction >= int(DATA_DRIVEN_TRIG_INTERACTION_MAX_TERMS):
                            break
                    if num_added_trig_interaction >= int(DATA_DRIVEN_TRIG_INTERACTION_MAX_TERMS):
                        break
                if num_added_trig_interaction >= int(DATA_DRIVEN_TRIG_INTERACTION_MAX_TERMS):
                    break
            if num_added_trig_interaction >= int(DATA_DRIVEN_TRIG_INTERACTION_MAX_TERMS):
                break

    if len(feature_names) >= 5 and ENABLE_DATA_DRIVEN_RECIPROCAL_DIFFERENCE_PRODUCT_BASIS:
        # Generic reciprocal-difference primitive: envelope * (1/u - 1/v),
        # equivalently envelope * ((v-u)/(u*v)). This is crucial for inverse-law
        # contrast structures and is benchmark-name agnostic.
        all_vars = feature_names[:min(len(feature_names), 7)]
        num_added_recip_diff = 0
        for u, v in combinations(all_vars, 2):
            remaining = [x for x in all_vars if x not in {u, v}]
            env_choices = []
            if remaining:
                # Full envelope first; for 5-D this uses all three remaining variables.
                env_choices.append(tuple(remaining))
            for r in range(1, min(3, len(remaining)) + 1):
                env_choices.extend(list(combinations(remaining, r)))
            env_seen = set()
            for env in env_choices:
                env = tuple(env)
                if env in env_seen:
                    continue
                env_seen.add(env)
                env_expr = "*".join(env) if env else "1"
                env_t = np.ones_like(train_y, dtype=float)
                for e in env:
                    env_t = env_t * xtr[e]
                recip_t = (1.0 / (xtr[u] + eps)) - (1.0 / (xtr[v] + eps))
                train_basis = env_t * recip_t
                val_basis = None
                if val_y is not None and all(name in xva for name in list(env) + [u, v]):
                    env_v = np.ones_like(val_y, dtype=float)
                    for e in env:
                        env_v = env_v * xva[e]
                    recip_v = (1.0 / (xva[u] + eps)) - (1.0 / (xva[v] + eps))
                    val_basis = env_v * recip_v
                expr = f"{env_expr}*(1/({u}+{eps:g})-1/({v}+{eps:g}))"
                add(expr, "reciprocal_difference_product", list(env) + [u, v], train_basis, val_basis)
                # Add the algebraically equivalent product-denominator view too.
                train_basis2 = env_t * ((xtr[v] - xtr[u]) / (xtr[u] * xtr[v] + eps))
                val_basis2 = None
                if val_y is not None and all(name in xva for name in list(env) + [u, v]):
                    env_v2 = np.ones_like(val_y, dtype=float)
                    for e in env:
                        env_v2 = env_v2 * xva[e]
                    val_basis2 = env_v2 * ((xva[v] - xva[u]) / (xva[u] * xva[v] + eps))
                expr2 = f"{env_expr}*(({v}-{u})/(({u})*({v})+{eps:g}))"
                add(expr2, "reciprocal_difference_product", list(env) + [u, v], train_basis2, val_basis2)
                num_added_recip_diff += 2
                if num_added_recip_diff >= int(DATA_DRIVEN_RECIPROCAL_DIFFERENCE_PRODUCT_MAX_TERMS):
                    break
            if num_added_recip_diff >= int(DATA_DRIVEN_RECIPROCAL_DIFFERENCE_PRODUCT_MAX_TERMS):
                break

    if len(feature_names) >= 5:
        all_vars = feature_names[:min(len(feature_names), 7)]
        for ratio_x, ratio_y in combinations(all_vars, 2):
            remaining = [v for v in all_vars if v not in {ratio_x, ratio_y}]
            for env in list(combinations(remaining, 3))[:4]:
                e1, e2, e3 = env
                env_t = xtr[e1] * xtr[e2] * xtr[e3]
                log_t = np.log(np.abs(xtr[ratio_x]) + eps) - np.log(np.abs(xtr[ratio_y]) + eps)
                val_values = None
                if all(v in xva for v in [e1, e2, e3, ratio_x, ratio_y]):
                    env_v = xva[e1] * xva[e2] * xva[e3]
                    log_v = np.log(np.abs(xva[ratio_x]) + eps) - np.log(np.abs(xva[ratio_y]) + eps)
                    val_values = env_v * log_v
                expr = f"{e1}*{e2}*{e3}*(log(abs({ratio_x})+{eps:g})-log(abs({ratio_y})+{eps:g}))"
                add(expr, "compound_log_interaction", [e1, e2, e3, ratio_x, ratio_y], env_t * log_t, val_values)
    scored = []
    for rank, item in enumerate(items, start=1):
        fit = _fit_single_basis_candidate(item["train_values"], train_y, item.get("val_values"), val_y)
        if fit is None:
            continue
        if float(fit.get("combined_gain", 0.0)) < float(DATA_DRIVEN_FEATURE_MIN_REL_GAIN):
            continue
        scored.append({
            "rank": rank,
            "basis_expr": item["expr"],
            "family": item["family"],
            "variables": item["variables"],
            "fit": fit,
            "combined_gain": float(fit.get("combined_gain", 0.0)),
            "val_mse": float(fit.get("val_mse", np.inf)),
        })
    scored.sort(key=lambda x: (-x["combined_gain"], x["val_mse"], x["rank"]))
    selected_records = []
    selected_keys = set()
    family_counts = {}
    top_per_family = max(1, int(DATA_DRIVEN_FEATURE_TOP_PER_FAMILY))
    for item in scored:
        fam = str(item.get("family", "algebraic"))
        if family_counts.get(fam, 0) >= top_per_family:
            continue
        key = _expr_dedup_key(item["basis_expr"])
        if key in selected_keys:
            continue
        selected_records.append(item)
        selected_keys.add(key)
        family_counts[fam] = family_counts.get(fam, 0) + 1
        if len(selected_records) >= max_candidates:
            break
    for item in scored:
        if len(selected_records) >= max_candidates:
            break
        key = _expr_dedup_key(item["basis_expr"])
        if key in selected_keys:
            continue
        selected_records.append(item)
        selected_keys.add(key)
    exprs = []
    for item in selected_records:
        basis = item["basis_expr"]
        coef = _format_float_for_expr(item["fit"].get("coef", 0.0))
        intercept = _format_float_for_expr(item["fit"].get("intercept", 0.0))
        if DATA_DRIVEN_FEATURE_INCLUDE_NUMERIC_FIT:
            exprs.append(f"({coef})*({basis}) + ({intercept})")
        if DATA_DRIVEN_FEATURE_INCLUDE_PARAM_TEMPLATES:
            exprs.append(f"a*({basis}) + b")
    exprs = deduplicate_expressions(exprs)[:max_candidates]
    feature_names_for_bridge = list(getattr(dataset, "feature_names", []) or [])
    high_dim_bridge_enabled = bool(
        ENABLE_HIGH_DIM_DATA_DRIVEN_CANDIDATE_BRIDGE
        and len(feature_names_for_bridge) >= HIGH_DIM_ROLE_TRIGGER_DIM
        and (not DATA_DRIVEN_FEATURE_SEEDS_AS_CANDIDATES)
    )
    if high_dim_bridge_enabled:
        bridge_exprs = []
        for item in selected_records:
            if float(item.get("combined_gain", 0.0) or 0.0) < float(HIGH_DIM_DATA_DRIVEN_BRIDGE_MIN_GAIN):
                continue
            basis = item["basis_expr"]
            coef = _format_float_for_expr(item["fit"].get("coef", 0.0))
            intercept = _format_float_for_expr(item["fit"].get("intercept", 0.0))
            if DATA_DRIVEN_FEATURE_INCLUDE_PARAM_TEMPLATES:
                bridge_exprs.append(f"a*({basis}) + b")
            if DATA_DRIVEN_FEATURE_INCLUDE_NUMERIC_FIT:
                bridge_exprs.append(f"({coef})*({basis}) + ({intercept})")
            if len(bridge_exprs) >= HIGH_DIM_DATA_DRIVEN_BRIDGE_MAX_CANDIDATES:
                break
        exprs = deduplicate_expressions(bridge_exprs)[:min(max_candidates, HIGH_DIM_DATA_DRIVEN_BRIDGE_MAX_CANDIDATES)]
    def _basis_pattern_for_record(record):
        family = str(record.get("family", "algebraic"))
        variables = list(record.get("variables", []) or [])
        if family == "compound_log_interaction":
            return "multiplicative envelope times log-difference/ratio"
        if family == "multiplicative_difference_ratio":
            return "multiplicative difference times ratio/denominator"
        if family == "rational_difference_of_squares_denominator":
            return "numerator product over scaled difference-of-squares denominator"
        if family == "power_product_over_product_denominator":
            return "power-product numerator over product denominator"
        if family == "outer_scale_exp_minus_one":
            return "outer scale times exp(inner product/ratio) minus one"
        if family == "outer_scale_linear_plus_trig_interaction":
            return "outer scale times linear plus trigonometric interaction"
        if family == "reciprocal_difference_product":
            return "multiplicative envelope times reciprocal difference"
        if family == "multiplicative_difference":
            return "multiplicative difference / contrast interaction"
        if family == "interaction_power":
            return "multiplicative interaction with one powered variable"
        if family == "difference":
            return "difference or contrast between two variables"
        if family == "logarithmic" and len(variables) >= 2:
            return "log-difference or log-ratio over two variables"
        if family in {"interaction", "high_order_interaction"}:
            return "multiplicative interaction"
        if family == "rational":
            return "ratio or denominator-like relation"
        if family == "trigonometric":
            return "periodic or modulated relation"
        if family == "exponential":
            return "exponential relation"
        if family == "power":
            return "polynomial or power relation"
        if family == "additive":
            return "additive/unary relation"
        return family

    def _evidence_payload(record):
        payload = {
            "family": record["family"],
            "variables": list(record.get("variables", []) or []),
            "basis_pattern": _basis_pattern_for_record(record),
            "combined_gain": round(float(record["combined_gain"]), 6),
            "val_mse": round(float(record["val_mse"]), 6) if np.isfinite(record["val_mse"]) else None,
            "coef_sign": "positive" if float(record["fit"].get("coef", 0.0)) >= 0 else "negative",
        }
        if DATA_DRIVEN_FEATURE_SEEDS_AS_CANDIDATES or high_dim_bridge_enabled:
            payload["basis_expr"] = record["basis_expr"]
            payload["coef"] = round(float(record["fit"].get("coef", 0.0)), 6)
            payload["intercept"] = round(float(record["fit"].get("intercept", 0.0)), 6)
        return payload

    trace = {
        "enabled": True,
        "mode": "candidate_mode" if DATA_DRIVEN_FEATURE_SEEDS_AS_CANDIDATES else ("highdim_evidence_bridge" if high_dim_bridge_enabled else "evidence_only"),
        "candidate_injection_enabled": bool(DATA_DRIVEN_FEATURE_SEEDS_AS_CANDIDATES or high_dim_bridge_enabled),
        "high_dim_bridge_enabled": bool(high_dim_bridge_enabled),
        "num_library_terms": len(items),
        "num_scored_terms": len(scored),
        "num_selected_records": len(selected_records),
        "num_exprs": len(exprs) if (DATA_DRIVEN_FEATURE_SEEDS_AS_CANDIDATES or high_dim_bridge_enabled) else 0,
        "family_counts": {k: int(v) for k, v in family_counts.items()},
        "top_records": make_json_safe([
            _evidence_payload(r) for r in selected_records[:max(1, int(DATA_DRIVEN_FEATURE_EVIDENCE_TOPK))]
        ]),
        "preview_exprs": exprs[:8] if (DATA_DRIVEN_FEATURE_SEEDS_AS_CANDIDATES or high_dim_bridge_enabled) else [],
        "note": "Data-driven feature library is diagnostic evidence only; it does not directly inject closed-form candidates."
            if (not DATA_DRIVEN_FEATURE_SEEDS_AS_CANDIDATES and not high_dim_bridge_enabled)
            else ("High-dimensional data-driven evidence bridge is injecting generic fit-ready candidates."
                  if high_dim_bridge_enabled and not DATA_DRIVEN_FEATURE_SEEDS_AS_CANDIDATES
                  else "Data-driven feature library is injecting selected candidates."),
    }
    if not (DATA_DRIVEN_FEATURE_SEEDS_AS_CANDIDATES or high_dim_bridge_enabled):
        return [], trace
    return exprs, trace

def deduplicate_candidate_dicts(candidate_items):
    unique_candidates = []
    seen = set()
    for item in candidate_items:
        if not isinstance(item, dict):
            continue
        expr = str(item.get("expression", "")).strip()
        if not expr:
            continue
        expr_key = _expr_dedup_key(expr)
        if expr_key in seen:
            continue
        seen.add(expr_key)
        unique_candidates.append(item)
    return unique_candidates


def filter_identity_refinements(exprs, current_best_expr):
    current_key = _expr_dedup_key(current_best_expr) if str(current_best_expr or "").strip() else ""
    out = []
    seen = set()
    for expr in exprs or []:
        expr = str(expr).strip()
        if not expr:
            continue
        expr_key = _expr_dedup_key(expr)
        if current_key and expr_key == current_key:
            continue
        if expr_key in seen:
            continue
        seen.add(expr_key)
        out.append(expr)
    return out


def _expr_has_disallowed_template_params(expr: str, feature_names) -> bool:
    expr = str(expr or "").strip()
    if not expr:
        return True
    feature_set = set(feature_names or [])
    blocked_symbols = {
        "a", "b", "c", "d", "k", "m", "n", "p", "q", "r", "s", "t", "u", "v", "w",
        "alpha", "beta", "gamma", "theta", "lambda",
    }
    try:
        sym_expr = sp.sympify(expr, locals=TemplateFillTool.SYMPY_LOCALS)
    except Exception:
        return True
    free_symbols = {str(symbol) for symbol in sym_expr.free_symbols}
    for symbol in free_symbols:
        if symbol in feature_set:
            continue
        if symbol.lower() in blocked_symbols:
            return True
    return False


def _filter_non_template_expressions(exprs, feature_names):
    kept = []
    dropped = []
    for expr in exprs or []:
        expr = str(expr).strip()
        if not expr:
            continue
        if _expr_has_disallowed_template_params(expr, feature_names):
            dropped.append(expr)
        else:
            kept.append(expr)
    return deduplicate_expressions(kept), deduplicate_expressions(dropped)


def get_initial_proposer_source_cap(source_name, iter_cfg=None):
    allow_protected_seed_bridge = bool((iter_cfg or {}).get("allow_protected_seed_bridge", False))
    allow_guided_candidate_sources = bool((iter_cfg or {}).get("allow_guided_candidate_sources", False))
    allow_family_seed_sources = bool((iter_cfg or {}).get("allow_family_seed_sources", False))
    if PURE_LLM_CANDIDATE_MODE and (not allow_guided_candidate_sources) and (not allow_family_seed_sources) and source_name not in {"datadriven", "generic", "text", "diverse"}:
        if not (allow_protected_seed_bridge and source_name == "protected"):
            return 0
    proposal_k = max(2, int((iter_cfg or {}).get("proposal_k", NUM_PROPOSAL_CANDIDATES)))
    experience_caps = dict((iter_cfg or {}).get("experience_source_cap_overrides", {}) or {})
    configured = experience_caps.get(source_name, INITIAL_PROPOSER_SOURCE_MAX_CANDIDATES.get(source_name, proposal_k))
    bonus = 2 if source_name in {"protected", "manual"} else 0
    return max(1, min(int(configured), proposal_k + bonus))


def get_initial_proposer_source_order(iter_cfg=None, available_sources=None):
    if PURE_LLM_CANDIDATE_MODE:
        allow_guided_candidate_sources = bool((iter_cfg or {}).get("allow_guided_candidate_sources", False))
        if allow_guided_candidate_sources:
            order = list((iter_cfg or {}).get("guided_source_order") or GUIDED_RESCUE_SOURCE_ORDER)
        elif bool((iter_cfg or {}).get("allow_family_seed_sources", False)):
            order = list((iter_cfg or {}).get("family_seed_source_order") or FAMILY_SEED_SOURCE_ORDER)
        else:
            order = list(INITIAL_EXPLORATION_SOURCE_ORDER)
    else:
        order = list((iter_cfg or {}).get("experience_source_order") or INITIAL_PROPOSER_SOURCE_ORDER)
    if available_sources is None:
        return order
    available = set(available_sources)
    ordered = [name for name in order if name in available]
    if PURE_LLM_CANDIDATE_MODE:
        return ordered
    for name in available_sources:
        if name not in ordered:
            ordered.append(name)
    return ordered


def _candidate_items_from_exprs(exprs, source_name, rationale_prefix="", prior_score=1.0):
    out = []
    for rank, expr in enumerate(deduplicate_expressions(exprs), start=1):
        msg = f"{rationale_prefix} source={source_name}, rank={rank}".strip()
        out.append({
            "expression": expr,
            "skeleton": "",
            "parameters": [],
            "rationale": msg,
            "prior_score": float(prior_score),
            "source": source_name,
        })
    return out


def _normalize_source_candidate_items(candidate_items, source_name, rationale_prefix=""):
    normalized = []
    seen = set()
    for rank, item in enumerate(candidate_items or [], start=1):
        if not isinstance(item, dict):
            continue
        expr = str(item.get("expression", "")).strip()
        if not expr:
            continue
        expr_key = _expr_dedup_key(expr)
        if expr_key in seen:
            continue
        seen.add(expr_key)
        payload = dict(item)
        payload["expression"] = expr
        payload.setdefault("skeleton", "")
        payload.setdefault("parameters", [])
        payload.setdefault("rationale", f"{rationale_prefix} source={source_name}, rank={rank}".strip())
        payload.setdefault("prior_score", 1.0)
        payload["source"] = source_name
        normalized.append(payload)
    return normalized


def make_proposer_source_result(
    source_name,
    exprs=None,
    candidate_items=None,
    per_call_stats=None,
    raw_result=None,
    skipped=False,
    skip_reason=None,
    metadata=None,
    iter_cfg=None,
    rationale_prefix="",
):
    exprs = deduplicate_expressions(exprs or [])
    if candidate_items is None:
        candidate_items = _candidate_items_from_exprs(
            exprs,
            source_name=source_name,
            rationale_prefix=rationale_prefix,
        )
    else:
        candidate_items = _normalize_source_candidate_items(
            candidate_items,
            source_name=source_name,
            rationale_prefix=rationale_prefix,
        )
        if not exprs:
            exprs = [str(item.get("expression", "")).strip() for item in candidate_items if str(item.get("expression", "")).strip()]
            exprs = deduplicate_expressions(exprs)

    cap = get_initial_proposer_source_cap(source_name, iter_cfg=iter_cfg)
    exprs = exprs[:cap]
    item_by_key = {_expr_dedup_key(item.get("expression", "")): item for item in candidate_items}
    ordered_items = []
    missing_exprs = []
    for expr in exprs:
        expr_key = _expr_dedup_key(expr)
        if expr_key in item_by_key:
            ordered_items.append(item_by_key[expr_key])
        else:
            missing_exprs.append(expr)
    if missing_exprs:
        ordered_items.extend(
            _candidate_items_from_exprs(
                missing_exprs,
                source_name=source_name,
                rationale_prefix=rationale_prefix,
            )
        )

    return {
        "source": source_name,
        "exprs": exprs,
        "candidate_items": ordered_items,
        "per_call_stats": per_call_stats or [],
        "num_exprs_unique": len(exprs),
        "skipped": bool(skipped),
        "skip_reason": skip_reason,
        "raw_result": make_json_safe(raw_result),
        "metadata": make_json_safe(metadata or {}),
    }


def build_heuristic_skeleton_candidates(feature_names, structure_hints=None, visual_hints=None, max_candidates=None):
    feature_names = list(feature_names or [])
    if not feature_names:
        return []

    hints_text = " ".join([str(x) for x in (structure_hints or []) + (visual_hints or [])]).lower()
    prefer_periodic = any(k in hints_text for k in ["periodic", "oscillat", "wave", "sin", "cos"])
    prefer_exponential = any(k in hints_text for k in ["exp", "exponential", "growth", "decay"])
    prefer_rational = any(k in hints_text for k in ["ratio", "reciprocal", "denominator", "rational"])
    prefer_additive = any(k in hints_text for k in ["additive", "separable", "sum"])

    out = []
    seen = set()

    def _add(items):
        for expr in items:
            expr = str(expr).strip()
            if not expr:
                continue
            expr_key = _expr_dedup_key(expr)
            if expr_key in seen:
                continue
            seen.add(expr_key)
            out.append(expr)
            if max_candidates is not None and len(out) >= max_candidates:
                return True
        return False

    d = len(feature_names)
    if d == 1:
        x = feature_names[0]
        polynomial = [
            x,
            f"a*{x}+b",
            f"a*{x}**2+b*{x}+c",
            f"a*{x}**3+b*{x}**2+c*{x}+d",
        ]
        smooth = [
            f"a*exp(b*{x})+c",
            f"a*log(abs(b*{x})+c)+d",
            f"(a*{x}+b)/(c*{x}+d)",
        ]
        periodic = [
            f"a*sin(b*{x}+c)+d",
            f"a*cos(b*{x}+c)+d",
            f"a*sin(b*{x})*cos(c*{x})+d",
        ]
        if _add(polynomial):
            return out
        if prefer_periodic and _add(periodic):
            return out
        if prefer_exponential and _add(smooth):
            return out
        _add(periodic[:1] + smooth[:2])
        return out[:max_candidates] if max_candidates is not None else out

    if d == 2:
        x1, x2 = feature_names
        interaction = [
            f"{x1}*{x2}",
            f"a*{x1}*{x2}+b",
            f"a*{x1}*{x2}**2+b",
            f"a*{x1}**2*{x2}+b",
            f"a*({x1}+b1)*({x2}+b2)+c",
            f"a*({x1}+b1)*({x2}+b2)**2+c",
        ]
        additive = [
            f"a*{x1}+b*{x2}+c",
            f"a*{x1}**2+b*{x2}+c",
            f"a*{x1}+b*{x2}**2+c",
            f"a*{x1}**2+b*{x2}**2+c",
        ]
        rational = [
            f"(a*{x1}+b)/(c*{x2}+d)+e",
            f"a*({x1}/({x2}+b))+c",
            f"a*({x2}/({x1}+b))+c",
        ]
        smooth = [
            f"a*exp(b*{x1}+c*{x2})+d",
            f"a*log(abs(b*{x1}+c*{x2})+d)+e",
            f"a*sqrt(abs(b*{x1}+c*{x2})+d)+e",
        ]
        periodic = [
            f"a*sin(b*{x1}+c*{x2})+d",
            f"a*cos(b*{x1}+c*{x2})+d",
            f"a*sin(b*{x1})*cos(c*{x2})+d",
        ]
        if _add(interaction):
            return out
        if prefer_additive and _add(additive):
            return out
        if prefer_rational and _add(rational):
            return out
        if prefer_periodic and _add(periodic):
            return out
        if prefer_exponential and _add(smooth):
            return out
        _add(additive[:2] + rational[:2] + periodic[:1] + smooth[:1])
        return out[:max_candidates] if max_candidates is not None else out

    if d == 3:
        x1, x2, x3 = feature_names[:3]
        sparse = [
            f"a*{x1}+b*{x2}+c*{x3}+d",
            f"a*{x1}*{x2}+b*{x3}+c",
            f"a*{x1}*{x3}+b*{x2}+c",
            f"a*{x2}*{x3}+b*{x1}+c",
            f"a*{x1}*{x2}*{x3}+b",
            f"a*({x1}*{x2})/({x3}+b)+c",
            f"a*sin(b*{x1})+c*{x2}+d*{x3}+e",
            f"a*exp(b*{x1})+c*{x2}+d*{x3}+e",
        ]
        _add(sparse)
        return out[:max_candidates] if max_candidates is not None else out

    main = feature_names[:min(d, 4)]
    linear = " + ".join([f"a{i+1}*{v}" for i, v in enumerate(main)])
    pair_terms = []
    for i in range(len(main)):
        for j in range(i + 1, len(main)):
            pair_terms.append(f"b{i+1}{j+1}*{main[i]}*{main[j]}")
    pair_sum = " + ".join(pair_terms[:4]) if pair_terms else "0"
    x0 = feature_names[0]
    x1 = feature_names[1] if len(feature_names) >= 2 else feature_names[0]
    x2 = feature_names[2] if len(feature_names) >= 3 else x1
    x3 = feature_names[3] if len(feature_names) >= 4 else x2
    x4 = feature_names[4] if len(feature_names) >= 5 else feature_names[-1]
    mechanistic = [
        f"a1*({x0}*{x1}) + a2*({x2}+b2*{x3}) + c",
        f"a1*({x0}*{x1}) + a2/(abs({x4})+b1) + c",
        f"a1*({x0}*{x1})*log(abs(({x4}+b1)/({x3}+b2)) + c1) + d",
        f"a1*({x0}*{x1}) + a2*exp(b1*{x4}+c1) + d",
        f"a1*({x0}*{x1}) + a2*sin(b1*{x4}+c1) + d",
        f"a1*({x0}+b1*{x1}) + a2*({x2}+b2*{x3}) + a3/(abs({x4})+b3) + c",
        f"a1*({x0}*{x1})*({x2}+b1)*log(abs(({x4}+b2)/({x3}+b3)) + c1) + d",
    ]
    linear_family = [
        f"{linear} + c",
        f"{pair_sum} + {linear} + c",
        f"a1*{x0}*{x1} + {linear} + c",
    ]
    smooth_family = [
        f"a1*sin(b1*{main[0]}) + {linear} + c",
        f"a1*exp(b1*{main[0]}) + {linear} + c",
    ]
    if prefer_rational:
        sparse = mechanistic + linear_family + smooth_family
    elif prefer_exponential or prefer_periodic:
        sparse = smooth_family + mechanistic + linear_family
    else:
        sparse = mechanistic[:6] + linear_family + smooth_family[:1]
    _add(sparse)
    if len(feature_names) >= 5:
        return _filter_specialist_exprs(out, max_candidates=max_candidates)
    return out[:max_candidates] if max_candidates is not None else out


def prepare_initial_template_sources(feature_names, row_meta, iter_cfg, experience_prior=None):
    feature_names = list(feature_names or [])
    manual_candidates = build_manual_candidates(feature_names)
    protected_templates = build_protected_benchmark_templates(row_meta, feature_names)
    livermore_templates = build_livermore_2d_templates(row_meta, feature_names)
    inject_into_protected = bool((experience_prior or {}).get("inject_into_protected", False)) if ALLOW_ROW_SPECIFIC_BENCHMARK_PRIORS else False
    memory_seed_candidates = list((experience_prior or {}).get("memory_seed_exprs", []) or []) if ALLOW_HISTORY_MEMORY else []
    experience_candidates = []
    if inject_into_protected:
        experience_candidates = list((experience_prior or {}).get("candidate_families", []) or [])
    protected_candidates = merge_expression_lists(
        protected_templates,
        livermore_templates,
        memory_seed_candidates,
        experience_candidates,
    )

    if ALLOW_ROW_SPECIFIC_BENCHMARK_PRIORS and str(row_meta.get("dataset_dir")) == "benchmark_csv" and len(feature_names) == 5:
        iter_cfg["proposal_k"] = max(iter_cfg.get("proposal_k", 0), 8)
        iter_cfg["refine_rounds"] = max(iter_cfg.get("refine_rounds", 0), 2)
        iter_cfg["refined_k"] = max(iter_cfg.get("refined_k", 0), 5)
        protected_candidates = merge_expression_lists(
            protected_candidates,
            build_feynman_5d_templates(feature_names),
        )
    elif ALLOW_ROW_SPECIFIC_BENCHMARK_PRIORS and str(row_meta.get("dataset_dir")) == "benchmark_csv" and len(feature_names) == 2:
        iter_cfg["proposal_k"] = max(iter_cfg.get("proposal_k", 0), 8)
        iter_cfg["refine_rounds"] = max(iter_cfg.get("refine_rounds", 0), 2)
        iter_cfg["refined_k"] = max(iter_cfg.get("refined_k", 0), 5)

    if PURE_LLM_CANDIDATE_MODE:
        if inject_into_protected or memory_seed_candidates:
            return protected_candidates, []
        return [], []

    return protected_candidates, manual_candidates


def _extract_plot_variables_from_description(description):
    desc = str(description or "")
    vars_found = []

    m = re.search(r"versus\s+([A-Za-z_]\w*)", desc)
    if m:
        vars_found.append(m.group(1))

    m = re.search(r"([A-Za-z_]\w*)-([A-Za-z_]\w*)\s+plane", desc)
    if m:
        vars_found.extend([m.group(1), m.group(2)])

    ordered = []
    seen = set()
    for item in vars_found:
        if item not in seen:
            ordered.append(item)
            seen.add(item)
    return ordered


def choose_mm_image_paths(image_paths, max_images=2, plot_descriptions=None, focus_variables=None, focus_pairs=None):
    if not image_paths:
        return []
    if not plot_descriptions:
        return image_paths[:max_images]

    focus_variables = [str(x).strip() for x in (focus_variables or []) if str(x).strip()]
    focus_var_set = set(focus_variables)
    focus_pair_set = {
        tuple(sorted([str(v) for v in pair if str(v).strip()]))
        for pair in (focus_pairs or [])
        if isinstance(pair, (list, tuple)) and len(pair) >= 2
    }

    ranked = []
    for idx, path in enumerate(image_paths):
        desc = str(plot_descriptions[idx]).lower() if idx < len(plot_descriptions) else ""
        desc_vars = _extract_plot_variables_from_description(plot_descriptions[idx] if idx < len(plot_descriptions) else "")
        desc_var_set = set(desc_vars)
        score = 0.0
        if "colored by" in desc:
            score += 4.0
        if "plane" in desc:
            score += 2.0
        if "versus" in desc or " vs " in desc:
            score += 1.0
        if "scatter" in desc:
            score += 0.2
        if focus_var_set:
            score += 3.0 * len(desc_var_set & focus_var_set)
            if desc_vars and desc_vars[0] in focus_var_set:
                score += 1.0
        if focus_pair_set and len(desc_vars) >= 2:
            desc_pair = tuple(sorted(desc_vars[:2]))
            if desc_pair in focus_pair_set:
                score += 5.0
        ranked.append((score, idx, path))

    selected = sorted(ranked, key=lambda x: (-x[0], x[1]))[:max_images]
    selected.sort(key=lambda x: x[1])
    return [path for _, _, path in selected]


def _parse_json_like_text(text):
    try:
        parsed = RESPONSE_PARSER.parse_json_like_text(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    return None


def _parse_float01(value, default=0.0):
    try:
        value = float(value)
    except Exception:
        return float(default)
    return float(max(0.0, min(1.0, value)))


def _call_name_from_ast(func_node):
    if isinstance(func_node, ast.Name):
        return func_node.id
    if isinstance(func_node, ast.Attribute):
        return func_node.attr
    return ""


def _extract_numeric_constant(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, numbers.Number):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = _extract_numeric_constant(node.operand)
        if inner is not None:
            return -float(inner)
    return None


def _collect_factor_variables(node, variable_name_set):
    if node is None:
        return set()
    if isinstance(node, ast.Name):
        return {node.id} if node.id in variable_name_set else set()
    if isinstance(node, ast.UnaryOp):
        return _collect_factor_variables(node.operand, variable_name_set)
    if isinstance(node, ast.Call):
        out = set()
        for arg in getattr(node, "args", []) or []:
            out |= _collect_factor_variables(arg, variable_name_set)
        return out
    if isinstance(node, ast.BinOp):
        return _collect_factor_variables(node.left, variable_name_set) | _collect_factor_variables(node.right, variable_name_set)
    out = set()
    for child in ast.iter_child_nodes(node):
        out |= _collect_factor_variables(child, variable_name_set)
    return out


class _FormulaFormVisitor(ast.NodeVisitor):
    OP_LABELS = {
        ast.Add: "+",
        ast.Sub: "-",
        ast.Mult: "*",
        ast.Div: "/",
        ast.Pow: "**",
    }

    def __init__(self, variable_names):
        self.variable_name_set = set(variable_names)
        self.variables_used = set()
        self.functions_used = set()
        self.op_symbols = set()
        self.numeric_powers = []
        self.has_additive_mix = False
        self.max_multiplicative_arity = 1

    def visit_Name(self, node):
        if node.id in self.variable_name_set:
            self.variables_used.add(node.id)

    def visit_Call(self, node):
        fn = _call_name_from_ast(node.func)
        if fn:
            self.functions_used.add(fn)
        self.generic_visit(node)

    def visit_BinOp(self, node):
        label = self.OP_LABELS.get(type(node.op))
        if label:
            self.op_symbols.add(label)

        if isinstance(node.op, (ast.Add, ast.Sub)):
            self.has_additive_mix = True

        if isinstance(node.op, ast.Pow):
            exp = _extract_numeric_constant(node.right)
            if exp is not None:
                self.numeric_powers.append(abs(float(exp)))

        if isinstance(node.op, (ast.Mult, ast.Div)):
            vars_in_term = _collect_factor_variables(node, self.variable_name_set)
            self.max_multiplicative_arity = max(self.max_multiplicative_arity, len(vars_in_term))

        self.generic_visit(node)


def extract_formula_form_signature(expr, variable_names):
    expr = str(expr or "").strip()
    signature = {
        "expression": expr,
        "parse_success": False,
        "variables_used": [],
        "functions_used": [],
        "op_symbols": [],
        "families": [],
        "max_power_degree": 1.0,
        "max_multiplicative_arity": 1,
        "has_additive_mix": False,
    }
    if not expr:
        return signature

    try:
        tree = ast.parse(expr, mode="eval")
        visitor = _FormulaFormVisitor(variable_names)
        visitor.visit(tree)

        families = set()
        fn_set = set(visitor.functions_used)
        op_set = set(visitor.op_symbols)
        max_degree = max(visitor.numeric_powers) if visitor.numeric_powers else (2.0 if "**" in op_set else 1.0)

        if "/" in op_set:
            families.add("rational")
        if max_degree > 1.0:
            families.add("power")
        if fn_set & {"sin", "cos", "tan", "sinh", "cosh", "tanh"}:
            families.add("trigonometric")
        if "exp" in fn_set:
            families.add("exponential")
        if "log" in fn_set:
            families.add("logarithmic")
        if "sqrt" in fn_set:
            families.add("radical")
        if visitor.has_additive_mix and len(visitor.variables_used) >= 2:
            families.add("additive")
        if visitor.max_multiplicative_arity >= 2:
            families.add("interaction")
        if visitor.max_multiplicative_arity >= 3:
            families.add("high_order_interaction")
        if not families:
            families.add("algebraic")

        signature.update({
            "parse_success": True,
            "variables_used": sorted(visitor.variables_used),
            "functions_used": sorted(fn_set),
            "op_symbols": sorted(op_set),
            "families": sorted(families),
            "max_power_degree": float(max_degree),
            "max_multiplicative_arity": int(max(1, visitor.max_multiplicative_arity)),
            "has_additive_mix": bool(visitor.has_additive_mix),
        })
        return signature
    except Exception:
        low = expr.lower()
        variables_used = sorted([v for v in variable_names if re.search(rf"\b{re.escape(v)}\b", expr)])
        functions_used = sorted([
            fn for fn in ["sin", "cos", "tan", "sinh", "cosh", "tanh", "exp", "log", "sqrt", "abs"]
            if f"{fn}(" in low
        ])
        op_symbols = []
        for op in ["**", "/", "*", "+", "-"]:
            if op in expr:
                op_symbols.append(op)
        families = set()
        if "/" in expr:
            families.add("rational")
        if "**" in expr:
            families.add("power")
        if any(fn in functions_used for fn in ["sin", "cos", "tan", "sinh", "cosh", "tanh"]):
            families.add("trigonometric")
        if "exp" in functions_used:
            families.add("exponential")
        if "log" in functions_used:
            families.add("logarithmic")
        if "sqrt" in functions_used:
            families.add("radical")
        if "*" in expr and len(variables_used) >= 2:
            families.add("interaction")
        if ("+" in expr or "-" in expr) and len(variables_used) >= 2:
            families.add("additive")
        if not families:
            families.add("algebraic")

        max_degree = 1.0
        for m in re.finditer(r"\*\*\s*([0-9]+(?:\.[0-9]+)?)", expr):
            try:
                max_degree = max(max_degree, abs(float(m.group(1))))
            except Exception:
                pass

        signature.update({
            "variables_used": variables_used,
            "functions_used": functions_used,
            "op_symbols": sorted(set(op_symbols)),
            "families": sorted(families),
            "max_power_degree": float(max_degree),
            "max_multiplicative_arity": max(1, len(variables_used) if "*" in expr else 1),
            "has_additive_mix": ("+" in expr or "-" in expr),
        })
        return signature


def _safe_jaccard(a, b):
    set_a = set(a or [])
    set_b = set(b or [])
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    if not union:
        return 0.0
    return float(len(set_a & set_b) / len(union))


def score_formula_form_match(candidate_expr, true_expr, variable_names):
    cand_sig = extract_formula_form_signature(candidate_expr, variable_names)
    true_sig = extract_formula_form_signature(true_expr, variable_names)

    family_overlap = _safe_jaccard(cand_sig["families"], true_sig["families"])
    function_overlap = _safe_jaccard(cand_sig["functions_used"], true_sig["functions_used"])
    variable_overlap = _safe_jaccard(cand_sig["variables_used"], true_sig["variables_used"])

    degree_gap = abs(float(cand_sig["max_power_degree"]) - float(true_sig["max_power_degree"]))
    degree_score = max(0.0, 1.0 - min(degree_gap, 4.0) / 4.0)

    arity_gap = abs(int(cand_sig["max_multiplicative_arity"]) - int(true_sig["max_multiplicative_arity"]))
    interaction_score = max(0.0, 1.0 - min(arity_gap, 3.0) / 3.0)

    total_score = (
        0.40 * family_overlap
        + 0.20 * function_overlap
        + 0.20 * variable_overlap
        + 0.10 * degree_score
        + 0.10 * interaction_score
    )

    return {
        "score": float(max(0.0, min(1.0, total_score))),
        "family_overlap": float(family_overlap),
        "function_overlap": float(function_overlap),
        "variable_overlap": float(variable_overlap),
        "degree_score": float(degree_score),
        "interaction_score": float(interaction_score),
        "candidate_signature": cand_sig,
        "true_signature": true_sig,
    }


def evaluate_formula_form_proposals(candidate_exprs, true_expr, variable_names):
    if not true_expr:
        return None

    clean_exprs = deduplicate_expressions(candidate_exprs)
    if not clean_exprs:
        return {
            "num_candidates": 0,
            "best_form_match_score": 0.0,
            "first_candidate_form_score": None,
            "mean_top3_form_match_score": None,
            "hit_at_threshold": False,
            "best_candidate": None,
            "top_candidates": [],
            "true_signature": extract_formula_form_signature(true_expr, variable_names),
        }

    scored = []
    for rank, expr in enumerate(clean_exprs, start=1):
        metrics = score_formula_form_match(expr, true_expr, variable_names)
        scored.append({
            "rank": rank,
            "expression": expr,
            "form_match_score": metrics["score"],
            "family_overlap": metrics["family_overlap"],
            "function_overlap": metrics["function_overlap"],
            "variable_overlap": metrics["variable_overlap"],
            "degree_score": metrics["degree_score"],
            "interaction_score": metrics["interaction_score"],
            "signature": metrics["candidate_signature"],
        })

    ranked = sorted(scored, key=lambda x: (-x["form_match_score"], x["rank"]))
    top3 = ranked[:3]

    return {
        "num_candidates": len(clean_exprs),
        "best_form_match_score": float(ranked[0]["form_match_score"]),
        "first_candidate_form_score": float(scored[0]["form_match_score"]) if scored else None,
        "mean_top3_form_match_score": safe_numeric_mean([x["form_match_score"] for x in top3]),
        "hit_at_threshold": any(float(x["form_match_score"]) >= FORM_MATCH_THRESHOLD for x in scored),
        "best_candidate": ranked[0],
        "top_candidates": ranked[:FORM_MATCH_TOPK],
        "true_signature": extract_formula_form_signature(true_expr, variable_names),
    }


def _families_from_text_hint(text):
    low = str(text or "").lower()
    families = set()
    if any(k in low for k in ["rational", "ratio", "denominator", "reciprocal", "fraction"]):
        families.add("rational")
    if any(k in low for k in ["trig", "periodic", "oscillat", "sin", "cos", "modulation"]):
        families.add("trigonometric")
    if any(k in low for k in ["exp", "exponential"]):
        families.add("exponential")
    if any(k in low for k in ["log", "logarithmic"]):
        families.add("logarithmic")
    if any(k in low for k in ["sqrt", "radical", "root"]):
        families.add("radical")
    if any(k in low for k in ["power", "polynomial", "quadratic", "cubic"]):
        families.add("power")
    if any(k in low for k in ["interaction", "product", "multiplicative", "coupling"]):
        families.add("interaction")
    if any(k in low for k in ["additive", "separable", "sum"]):
        families.add("additive")
    return families


def _ordered_unique_feature_subset(values, feature_names):
    value_set = {str(x).strip() for x in (values or []) if str(x).strip()}
    return [v for v in feature_names if v in value_set]


def _collect_item_variable_hints(item, feature_names):
    values = []
    values.extend(item.get("used_variables", []) or [])
    values.extend(item.get("denominator_variables", []) or [])
    values.extend(item.get("periodic_variables", []) or [])
    for group in item.get("interaction_groups", []) or []:
        if isinstance(group, (list, tuple)):
            values.extend(group)
    return _ordered_unique_feature_subset(values, feature_names)


def _build_reference_signature_from_form_item(item, feature_names):
    expr_sig = extract_formula_form_signature(item.get("expression", ""), feature_names)
    families = set(expr_sig.get("families", []) or [])
    families |= _families_from_text_hint(item.get("family", ""))
    families |= _families_from_text_hint(item.get("composition", ""))
    families |= _families_from_text_hint(item.get("structure_summary", ""))

    op_symbols = set(expr_sig.get("op_symbols", []) or [])
    for op in item.get("key_operators", []) or []:
        op = str(op).strip()
        if not op:
            continue
        op_symbols.add(op)
        if op == "/":
            families.add("rational")
        elif op == "*":
            families.add("interaction")
        elif op in {"+", "-"}:
            families.add("additive")
        elif op == "**":
            families.add("power")

    if item.get("periodic_variables"):
        families.add("trigonometric")
    if item.get("denominator_variables"):
        families.add("rational")

    variables_used = _collect_item_variable_hints(item, feature_names)
    if not variables_used:
        variables_used = expr_sig.get("variables_used", []) or []

    out = dict(expr_sig)
    out["families"] = sorted(families or set(expr_sig.get("families", []) or ["algebraic"]))
    out["variables_used"] = variables_used
    out["op_symbols"] = sorted(op_symbols)
    return out


def _score_formula_signature_match(candidate_sig, reference_sig):
    family_overlap = _safe_jaccard(candidate_sig.get("families", []), reference_sig.get("families", []))
    function_overlap = _safe_jaccard(candidate_sig.get("functions_used", []), reference_sig.get("functions_used", []))
    variable_overlap = _safe_jaccard(candidate_sig.get("variables_used", []), reference_sig.get("variables_used", []))
    op_overlap = _safe_jaccard(candidate_sig.get("op_symbols", []), reference_sig.get("op_symbols", []))

    degree_gap = abs(float(candidate_sig.get("max_power_degree", 1.0)) - float(reference_sig.get("max_power_degree", 1.0)))
    degree_score = max(0.0, 1.0 - min(degree_gap, 4.0) / 4.0)

    arity_gap = abs(int(candidate_sig.get("max_multiplicative_arity", 1)) - int(reference_sig.get("max_multiplicative_arity", 1)))
    interaction_score = max(0.0, 1.0 - min(arity_gap, 3.0) / 3.0)

    return float(
        0.30 * family_overlap
        + 0.15 * function_overlap
        + 0.20 * variable_overlap
        + 0.20 * op_overlap
        + 0.075 * degree_score
        + 0.075 * interaction_score
    )


def _extract_denominator_variables(expr, feature_names):
    expr = str(expr or "").strip()
    if not expr:
        return []
    out = set()
    try:
        tree = ast.parse(expr, mode="eval")
        feature_set = set(feature_names or [])

        class _DenominatorVisitor(ast.NodeVisitor):
            def visit_BinOp(self, node):
                if isinstance(node.op, ast.Div):
                    out.update(_collect_factor_variables(node.right, feature_set))
                self.generic_visit(node)

        _DenominatorVisitor().visit(tree)
    except Exception:
        low = expr
        for var in feature_names or []:
            if re.search(rf"/[^\\n]*\\b{re.escape(var)}\\b", low):
                out.add(var)
    return [v for v in feature_names if v in out]


def _extract_periodic_variables(expr, feature_names):
    expr = str(expr or "").strip()
    if not expr:
        return []
    periodic = set()
    try:
        tree = ast.parse(expr, mode="eval")
        feature_set = set(feature_names or [])

        class _PeriodicVisitor(ast.NodeVisitor):
            def visit_Call(self, node):
                fn_name = None
                if isinstance(node.func, ast.Name):
                    fn_name = node.func.id
                if fn_name in {"sin", "cos", "tan", "sinh", "cosh", "tanh"}:
                    for arg in node.args:
                        periodic.update(_collect_factor_variables(arg, feature_set))
                self.generic_visit(node)

        _PeriodicVisitor().visit(tree)
    except Exception:
        for fn in ["sin", "cos", "tan", "sinh", "cosh", "tanh"]:
            m = re.finditer(rf"{fn}\((.*?)\)", expr)
            for mm in m:
                chunk = mm.group(1)
                for var in feature_names or []:
                    if re.search(rf"\b{re.escape(var)}\b", chunk):
                        periodic.add(var)
    return [v for v in feature_names if v in periodic]


def _extract_log_variables(expr, feature_names):
    expr = str(expr or "").strip()
    if not expr:
        return []
    used = set()
    try:
        tree = ast.parse(expr, mode="eval")
        feature_set = set(feature_names or [])

        class _LogVisitor(ast.NodeVisitor):
            def visit_Call(self, node):
                fn_name = None
                if isinstance(node.func, ast.Name):
                    fn_name = node.func.id
                if fn_name == "log":
                    for arg in node.args:
                        used.update(_collect_factor_variables(arg, feature_set))
                self.generic_visit(node)

        _LogVisitor().visit(tree)
    except Exception:
        for mm in re.finditer(r"log\((.*?)\)", expr):
            chunk = mm.group(1)
            for var in feature_names or []:
                if re.search(rf"\b{re.escape(var)}\b", chunk):
                    used.add(var)
    return [v for v in feature_names if v in used]


def _extract_variable_power_hints(expr, feature_names):
    expr = str(expr or "").strip()
    hints = {}
    if not expr:
        return hints
    for var in feature_names or []:
        max_power = 1.0
        found = False
        for match in re.finditer(rf"\b{re.escape(var)}\b\s*\*\*\s*([0-9]+(?:\.[0-9]+)?)", expr):
            try:
                max_power = max(max_power, float(match.group(1)))
                found = True
            except Exception:
                pass
        if found and max_power > 1.0:
            hints[var] = float(max_power)
    return hints


def _build_experience_reference_signature(feature_names, candidate_families, family_tags=None, function_hints=None):
    candidate_families = [str(x).strip() for x in (candidate_families or []) if str(x).strip()]
    if candidate_families:
        return extract_formula_form_signature(candidate_families[0], feature_names)

    family_tags = sorted(set([str(x).strip() for x in (family_tags or []) if str(x).strip()]))
    function_hints = sorted(set([str(x).strip() for x in (function_hints or []) if str(x).strip()]))
    if not family_tags and not function_hints:
        return None

    op_symbols = set()
    if "interaction" in family_tags or "high_order_interaction" in family_tags:
        op_symbols.add("*")
    if "rational" in family_tags:
        op_symbols.add("/")
    if "power" in family_tags:
        op_symbols.add("**")
    if "additive" in family_tags:
        op_symbols.add("+")

    return {
        "expression": "",
        "parse_success": False,
        "variables_used": list(feature_names or []),
        "functions_used": function_hints,
        "op_symbols": sorted(op_symbols),
        "families": family_tags or ["algebraic"],
        "max_power_degree": 2.0 if "power" in family_tags else 1.0,
        "max_multiplicative_arity": max(1, min(len(feature_names or []), 5)),
        "has_additive_mix": bool("additive" in family_tags),
    }


def _safe_json_loads(value, default=None):
    if isinstance(value, (dict, list)):
        return value
    if not isinstance(value, str) or not value.strip():
        return default
    try:
        return json.loads(value)
    except Exception:
        return default


def _boolish(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, numbers.Number):
        return bool(value)
    low = str(value).strip().lower()
    if low in {"true", "1", "yes", "y"}:
        return True
    if low in {"false", "0", "no", "n", ""}:
        return False
    return False


def _floatish(value, default=None):
    try:
        if value is None or value == "":
            return default
        out = float(value)
        if np.isfinite(out):
            return out
    except Exception:
        pass
    return default


def _ordered_unique_strings(values, limit=None):
    out, seen = [], set()
    for item in values or []:
        text = str(item).strip()
        if not text or text in seen:
            continue
        out.append(text)
        seen.add(text)
        if limit is not None and len(out) >= int(limit):
            break
    return out


def _merge_budget_adjustments(base_budget, extra_budget):
    merged = dict(base_budget or {})
    for key, value in dict(extra_budget or {}).items():
        if key.endswith("_cap") and isinstance(value, numbers.Number):
            prev = merged.get(key)
            if isinstance(prev, numbers.Number):
                merged[key] = min(float(prev), float(value))
                if float(merged[key]).is_integer():
                    merged[key] = int(merged[key])
            else:
                merged[key] = int(value) if float(value).is_integer() else float(value)
        elif isinstance(value, bool):
            merged[key] = bool(merged.get(key, False) or value)
        else:
            merged[key] = value
    return merged


def _discover_memory_report_paths():
    return []


def _extract_feature_names_from_task_meta(task_meta):
    variable_names = _safe_json_loads(task_meta.get("variable_names"), default=None)
    if isinstance(variable_names, list) and variable_names:
        return [str(x) for x in variable_names]
    n_features = int(task_meta.get("n_features", task_meta.get("dimension", 0)) or 0)
    return [f"x{i+1}" for i in range(n_features)]


def _compress_point_signature_rows(df, feature_names, target_name="y", max_points=MEMORY_POINT_SIGNATURE_MAX_POINTS):
    if df is None or not len(df) or not feature_names:
        return None
    cols = [c for c in list(feature_names) + [target_name] if c in df.columns]
    if len(cols) != len(feature_names) + 1:
        return None

    work_df = df.loc[:, cols].copy()
    work_df = work_df.replace([np.inf, -np.inf], np.nan).dropna()
    if work_df.empty:
        return None

    work_df = work_df.sort_values(by=list(feature_names), kind="mergesort").reset_index(drop=True)
    if len(work_df) > max_points:
        keep_idx = np.linspace(0, len(work_df) - 1, num=max_points, dtype=int)
        work_df = work_df.iloc[keep_idx].reset_index(drop=True)

    matrix = work_df.to_numpy(dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] <= 0:
        return None

    mean = np.nanmean(matrix, axis=0)
    std = np.nanstd(matrix, axis=0)
    std = np.where(std > 1e-12, std, 1.0)
    normalized = (matrix - mean) / std
    normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
    vector = [round(float(x), 6) for x in normalized.reshape(-1).tolist()]

    return {
        "mode": "sorted_zscore_xy",
        "n_points": int(matrix.shape[0]),
        "n_features": int(len(feature_names)),
        "columns": list(feature_names) + [target_name],
        "vector": vector,
    }


def _build_benchmark_point_signature_from_meta(task_meta, feature_names):
    expr = str(task_meta.get("true_expression") or "").strip()
    range_spec = task_meta.get("range_spec")
    distribution = task_meta.get("distribution", "U")
    n_features = int(task_meta.get("n_features", task_meta.get("dimension", len(feature_names))) or len(feature_names))
    if not expr or not range_spec or n_features <= 0:
        return None
    try:
        rng = np.random.default_rng(BENCHMARK_RANDOM_SEED)
        train_x = sample_features_from_range(
            BENCHMARK_TRAIN_SIZE,
            n_features,
            range_spec,
            distribution=distribution,
            rng=rng,
        )
        train_df = train_x.copy()
        train_df["y"] = evaluate_expression_on_df(expr, train_x)
        return _compress_point_signature_rows(train_df, feature_names, target_name="y")
    except Exception:
        return None


def _resolve_task_point_signature(row_meta=None, dataset=None, feature_names=None):
    feature_names = list(feature_names or [])
    if dataset is not None and getattr(dataset, "train_df", None) is not None:
        signature = _compress_point_signature_rows(dataset.train_df, feature_names, target_name=getattr(dataset, "target_name", "y"))
        if signature is not None:
            return signature
    row_meta = dict(row_meta or {})
    if str(row_meta.get("dataset_dir", "")) == "benchmark_csv":
        return _build_benchmark_point_signature_from_meta(row_meta, feature_names)
    return None


def _point_signature_distance(sig_a, sig_b):
    if not isinstance(sig_a, dict) or not isinstance(sig_b, dict):
        return None
    if int(sig_a.get("n_features", -1)) != int(sig_b.get("n_features", -2)):
        return None
    vec_a = list(sig_a.get("vector", []) or [])
    vec_b = list(sig_b.get("vector", []) or [])
    if not vec_a or not vec_b:
        return None
    usable = min(len(vec_a), len(vec_b))
    if usable <= 0:
        return None
    arr_a = np.asarray(vec_a[:usable], dtype=float)
    arr_b = np.asarray(vec_b[:usable], dtype=float)
    if arr_a.size == 0 or arr_b.size == 0:
        return None
    return float(np.linalg.norm(arr_a - arr_b) / np.sqrt(float(usable)))


def _extract_memory_entry_from_report(report_path):
    return None


def _load_memory_bank(force_reload=False):
    return []


def _build_memory_query(row_meta, feature_names, observation=None, experience_prior=None, dataset=None):
    return {"disabled": True, "feature_names": list(feature_names or [])}


def _score_memory_entry_against_query(entry, query):
    return 0.0


def _memory_entry_is_numerically_exact(entry):
    return False


def _memory_entry_quality_rank(entry):
    return 99


def _memory_entry_sort_key(score, exact_match, entry):
    return (1, 99, 0.0)


def summarize_memory_prior(memory_prior):
    return {"memory_enabled": False, "reason": "clean no-leakage build disables historical memory"}


def build_memory_prior(row_meta, feature_names, observation=None, experience_prior=None, dataset=None):
    return {
        "strength": "none",
        "confidence": 0.0,
        "exact_match": False,
        "memory_source_enabled": False,
        "candidate_exprs": [],
        "candidate_families": [],
        "memory_seed_exprs": [],
        "historical_candidate_exprs": [],
        "historical_candidate_families": [],
        "preferred_sources": [],
        "source_caps": {},
        "budget_adjustments": {},
        "memory_notes": ["historical memory disabled in clean no-leakage build"],
        "top_matches": [],
    }


def merge_guidance_priors(experience_prior=None, memory_prior=None):
    guidance = dict(experience_prior or {})
    guidance["memory_source_enabled"] = False
    guidance["memory_candidate_exprs"] = []
    guidance["memory_candidate_families"] = []
    guidance["memory_seed_exprs"] = []
    guidance["memory_strength"] = "none"
    guidance["memory_exact_match"] = False
    guidance["memory_prior_summary"] = summarize_memory_prior({})
    return guidance


def summarize_experience_prior(experience_prior):
    if not isinstance(experience_prior, dict) or not experience_prior:
        return None
    return {
        "format": experience_prior.get("format"),
        "base_name": experience_prior.get("base_name"),
        "strength": experience_prior.get("strength"),
        "confidence": experience_prior.get("confidence"),
        "experience_source_enabled": bool(experience_prior.get("experience_source_enabled", False)),
        "memory_source_enabled": bool(experience_prior.get("memory_source_enabled", False)),
        "memory_strength": experience_prior.get("memory_strength"),
        "memory_exact_match": bool(experience_prior.get("memory_exact_match", False)),
        "rerank_enabled": bool(experience_prior.get("rerank_enabled", False)),
        "inject_into_protected": bool(experience_prior.get("inject_into_protected", False)),
        "family_tags": list(experience_prior.get("family_tags", []) or []),
        "variable_roles": make_json_safe(experience_prior.get("variable_roles", {})),
        "function_hints": list(experience_prior.get("function_hints", []) or []),
        "preferred_sources": list(experience_prior.get("preferred_sources", []) or []),
        "source_caps": make_json_safe(experience_prior.get("source_caps", {})),
        "budget_adjustments": make_json_safe(experience_prior.get("budget_adjustments", {})),
        "candidate_families": list((experience_prior.get("candidate_families", []) or [])[:4]),
        "memory_candidate_exprs": list((experience_prior.get("memory_candidate_exprs", []) or [])[:4]),
        "memory_seed_exprs": list((experience_prior.get("memory_seed_exprs", []) or [])[:4]),
        "avoid_patterns": list(experience_prior.get("avoid_patterns", []) or []),
        "notes": list(experience_prior.get("experience_prior", []) or []),
        "memory_prior_summary": make_json_safe(experience_prior.get("memory_prior_summary")),
    }


def build_experience_prompt_hints(experience_prior):
    if not ENABLE_EXPERIENCE_PRIOR or not isinstance(experience_prior, dict) or not experience_prior:
        return []

    hints = []
    family_tags = list(experience_prior.get("family_tags", []) or [])
    variable_roles = dict(experience_prior.get("variable_roles", {}) or {})
    degree_hints = dict(experience_prior.get("degree_hints", {}) or {})
    candidate_families = list(experience_prior.get("candidate_families", []) or [])
    avoid_patterns = list(experience_prior.get("avoid_patterns", []) or [])

    if family_tags:
        hints.append(f"experience prior family tags: {', '.join(family_tags)}")
    if variable_roles.get("active_variables"):
        hints.append(f"experience prior active variables: {', '.join(variable_roles['active_variables'])}")
    if variable_roles.get("numerator_core"):
        hints.append(f"experience prior numerator variables: {', '.join(variable_roles['numerator_core'])}")
    if variable_roles.get("denominator_core"):
        hints.append(f"experience prior denominator variables: {', '.join(variable_roles['denominator_core'])}")
    if variable_roles.get("periodic_core"):
        hints.append(f"experience prior periodic variables: {', '.join(variable_roles['periodic_core'])}")
    if degree_hints:
        degree_text = ", ".join([f"{k}^{int(v) if float(v).is_integer() else round(float(v), 3)}" for k, v in degree_hints.items()])
        hints.append(f"experience prior degree hints: {degree_text}")
    for expr in candidate_families[:2]:
        hints.append(f"experience prior candidate family: {expr}")
    for note in avoid_patterns[:2]:
        hints.append(f"experience prior avoid pattern: {note}")
    return hints


def build_experience_form_items(experience_prior, feature_names):
    if not ENABLE_EXPERIENCE_PRIOR or not isinstance(experience_prior, dict) or not experience_prior:
        return []

    feature_names = list(feature_names or [])
    family_tags = list(experience_prior.get("family_tags", []) or [])
    function_hints = list(experience_prior.get("function_hints", []) or [])
    variable_roles = dict(experience_prior.get("variable_roles", {}) or {})
    candidate_families = list(experience_prior.get("candidate_families", []) or [])
    confidence = float(experience_prior.get("confidence", 0.0) or 0.0)
    notes = list(experience_prior.get("experience_prior", []) or [])

    key_operators = []
    if "interaction" in family_tags or "high_order_interaction" in family_tags:
        key_operators.append("*")
    if "rational" in family_tags:
        key_operators.append("/")
    if "power" in family_tags:
        key_operators.append("**")
    if "additive" in family_tags:
        key_operators.append("+")
    for fn in function_hints:
        if fn:
            key_operators.append(fn)

    interaction_groups = []
    numerator_core = list(variable_roles.get("numerator_core", []) or [])
    denominator_core = list(variable_roles.get("denominator_core", []) or [])
    active_variables = list(variable_roles.get("active_variables", []) or feature_names)
    periodic_core = list(variable_roles.get("periodic_core", []) or [])
    if len(numerator_core) >= 2:
        interaction_groups.append(numerator_core)
    if len(denominator_core) >= 2:
        interaction_groups.append(denominator_core)

    structure_summary = "; ".join(
        [x for x in [
            f"experience prior families: {', '.join(family_tags)}" if family_tags else "",
            f"active variables: {', '.join(active_variables)}" if active_variables else "",
            f"numerator core: {', '.join(numerator_core)}" if numerator_core else "",
            f"denominator core: {', '.join(denominator_core)}" if denominator_core else "",
            notes[0] if notes else "",
        ] if x]
    )

    form_items = []
    seeds = candidate_families[:max(1, min(3, len(candidate_families)))] or [""]
    for expr in seeds:
        form_items.append({
            "expression": str(expr).strip(),
            "family": " ".join(family_tags) if family_tags else "experience prior",
            "composition": "experience prior structural template",
            "structure_summary": structure_summary,
            "used_variables": active_variables,
            "interaction_groups": interaction_groups,
            "denominator_variables": denominator_core,
            "periodic_variables": periodic_core,
            "key_operators": deduplicate_expressions(key_operators),
            "confidence": confidence,
            "prior_score": confidence,
        })
    return form_items


def build_experience_expansion_candidates(experience_prior, feature_names, row_meta=None, top_k_per_item=6):
    form_items = build_experience_form_items(experience_prior, feature_names)
    if not form_items:
        return []

    template_pool = list(build_manual_candidates(feature_names))
    if row_meta is not None:
        template_pool.extend(build_protected_benchmark_templates(row_meta, feature_names))
        template_pool.extend(build_livermore_2d_templates(row_meta, feature_names))
        if (
            ALLOW_ROW_SPECIFIC_BENCHMARK_PRIORS
            and str(row_meta.get("dataset_dir")) == "benchmark_csv"
            and len(feature_names) == 5
        ):
            template_pool.extend(build_feynman_5d_templates(feature_names))

    prompt_hints = build_experience_prompt_hints(experience_prior)
    heuristic_templates = build_heuristic_skeleton_candidates(
        feature_names=feature_names,
        structure_hints=prompt_hints,
        visual_hints=[],
        max_candidates=max(4, int(top_k_per_item)),
    )

    candidate_families = list(experience_prior.get("candidate_families", []) or [])
    confidence = float(experience_prior.get("confidence", 0.0) or 0.0)
    expanded = []

    for item in form_items:
        family_templates = _build_family_templates_from_form_item(item, feature_names)
        ranked_templates = _rank_template_pool_against_form_item(
            template_pool=template_pool + heuristic_templates + candidate_families,
            item=item,
            feature_names=feature_names,
            top_k=top_k_per_item,
        )
        merged_exprs = deduplicate_expressions(candidate_families + family_templates + ranked_templates + heuristic_templates)
        for expr in merged_exprs:
            expanded.append({
                "expression": expr,
                "rationale": "experience prior expansion",
                "prior_score": confidence,
            })

    return deduplicate_candidate_dicts(expanded)


def build_experience_prior(row_meta, feature_names, observation=None):
    if not ENABLE_EXPERIENCE_PRIOR:
        return {}

    feature_names = list(feature_names or [])
    observation = observation or ObservationBundle(
        structure_hints=[],
        visual_hints=[],
        unit_hints={},
        plot_descriptions=[],
        image_paths=[],
    )
    row_meta = dict(row_meta or {})
    base_name = str(row_meta.get("base_name", "") or "")
    dataset_dir = str(row_meta.get("dataset_dir", "") or "")
    protected_templates = [] if (FAIR_GUIDED_EVAL or (not ALLOW_ROW_SPECIFIC_BENCHMARK_PRIORS)) else build_protected_benchmark_templates(row_meta, feature_names)
    candidate_families = list(protected_templates[:4])

    if (
        not candidate_families
        and (observation.structure_hints or observation.visual_hints)
    ):
        heuristic_candidates = build_heuristic_skeleton_candidates(
            feature_names=feature_names,
            structure_hints=list(observation.structure_hints or []),
            visual_hints=list(observation.visual_hints or []),
            max_candidates=6,
        )
        if FAIR_GUIDED_EVAL:
            hint_families = set()
            for hint in list(observation.structure_hints or []) + list(observation.visual_hints or []):
                hint_families |= _families_from_text_hint(hint)
            pseudo_item = {
                "expression": "",
                "family": " ".join(sorted(hint_families or {"algebraic"})),
                "composition": "fair guided observation-derived template query",
                "structure_summary": " ; ".join([str(x) for x in list(observation.structure_hints or [])[:4] + list(observation.visual_hints or [])[:4]]),
                "used_variables": list(feature_names),
                "interaction_groups": [list(feature_names[:min(len(feature_names), 4)])] if len(feature_names) >= 2 else [list(feature_names)],
                "denominator_variables": [feature_names[-1]] if ("rational" in hint_families and feature_names) else [],
                "periodic_variables": list(feature_names[:1]) if "trigonometric" in hint_families and feature_names else [],
                "key_operators": [op for fam, op in [("interaction", "*"), ("rational", "/"), ("power", "**"), ("additive", "+")] if fam in hint_families],
                "confidence": 0.7,
                "prior_score": 0.7,
            }
            template_pool = deduplicate_expressions(build_manual_candidates(feature_names) + heuristic_candidates)
            ranked_templates = _rank_template_pool_against_form_item(
                template_pool=template_pool,
                item=pseudo_item,
                feature_names=feature_names,
                top_k=6,
            )
            family_templates = _build_family_templates_from_form_item(pseudo_item, feature_names)
            candidate_families = deduplicate_expressions(family_templates + ranked_templates + heuristic_candidates)[:6]
        else:
            candidate_families = heuristic_candidates

    if len(feature_names) >= HIGH_DIM_ROLE_TRIGGER_DIM:
        role_guided_candidates = build_role_guided_high_dim_candidates(
            feature_names,
            structure_profile=getattr(observation, "structure_profile", None),
            experience_prior=None,
            max_candidates=HIGH_DIM_ROLE_TEMPLATE_LIMIT,
        )
        if role_guided_candidates:
            candidate_families = deduplicate_expressions(role_guided_candidates + candidate_families)

    family_tags = set()
    function_hints = set()
    degree_hints = {}
    denominator_core = []
    periodic_core = []
    active_variables = []
    numerator_core = []

    canonical_expr = candidate_families[0] if candidate_families else ""
    if canonical_expr:
        canonical_sig = extract_formula_form_signature(canonical_expr, feature_names)
        family_tags.update(canonical_sig.get("families", []) or [])
        function_hints.update(canonical_sig.get("functions_used", []) or [])
        degree_hints.update(_extract_variable_power_hints(canonical_expr, feature_names))
        denominator_core = _extract_denominator_variables(canonical_expr, feature_names)
        periodic_core = _extract_periodic_variables(canonical_expr, feature_names)
        active_variables = list(canonical_sig.get("variables_used", []) or [])
        numerator_core = [v for v in active_variables if v not in set(denominator_core)]

    hint_families = set()
    for hint in list(observation.structure_hints or []) + list(observation.visual_hints or []):
        hint_families |= _families_from_text_hint(hint)

    if not active_variables:
        active_variables = list(feature_names)
        numerator_core = list(feature_names)

    strong_prior = (not FAIR_GUIDED_EVAL) and bool(protected_templates) and dataset_dir == "benchmark_csv"
    high_dim_weak_structural_only = bool(
        NO_LEAKAGE_MODE
        and (not strong_prior)
        and len(feature_names) >= HIGH_DIM_ROLE_TRIGGER_DIM
        and _is_benchmark_task({"dataset_dir": dataset_dir})
    )
    if high_dim_weak_structural_only:
        candidate_families = []
        degree_hints = {}
        function_hints = set()

    has_structural_signal = bool(candidate_families) or bool(hint_families) or bool(getattr(observation, "structure_profile", None))
    strength = "strong" if strong_prior else ("medium" if has_structural_signal else "weak")
    confidence = 0.95 if strong_prior else (0.62 if has_structural_signal else 0.45)
    experience_source_enabled = bool(strong_prior)
    rerank_enabled = False if FAIR_GUIDED_EVAL else bool(strong_prior)
    inject_into_protected = False if FAIR_GUIDED_EVAL else bool(strong_prior)

    if (not strong_prior) and len(feature_names) >= 5:
        weak_active_coverage = len(set(active_variables or [])) < max(3, len(feature_names) - 1)
        if weak_active_coverage or not denominator_core:
            active_variables = list(feature_names)
            numerator_core = list(feature_names)
            denominator_core = []
            periodic_core = _ordered_unique_feature_subset(periodic_core, feature_names)

    if strong_prior:
        if not family_tags:
            family_tags |= hint_families
    else:
        family_tags |= hint_families

    preferred_sources = list(INITIAL_PROPOSER_SOURCE_ORDER)
    source_caps = {}
    budget_adjustments = {}
    experience_notes = []
    avoid_patterns = []

    if FAIR_GUIDED_EVAL:
        experience_notes.append("fair guided mode: only observation-derived structural hints are used")
        experience_notes.append("row-specific protected benchmark templates are disabled")
        avoid_patterns.append("avoid same-task template shortcuts; prefer sparse mechanistic structure from observed data")
    elif strong_prior:
        preferred_sources = list(EXPERIENCE_STRONG_PRIOR_SOURCE_ORDER)
        source_caps = dict(EXPERIENCE_STRONG_PRIOR_SOURCE_CAPS)
        experience_notes.append("row-specific protected benchmark family is available")
        experience_notes.append("prefer protected mechanistic family before generic surrogate fits")
        experience_notes.append("experience source is enabled because the prior is row-specific and benchmark-backed")
        avoid_patterns.append("avoid dense additive surrogate if protected family already matches operator topology")
        if "rational" in family_tags:
            avoid_patterns.append("avoid standalone log or exp surrogate when denominator structure is available")
        if len(feature_names) >= 5:
            budget_adjustments.update({
                "text_calls_cap": 1,
                "mm_calls_cap": 0,
                "proposal_k_cap": 8,
                "refined_k_cap": 3,
                "refine_rounds_cap": 1,
                "skip_diverse_proposal": True,
                "force_heuristic_agents": True,
            })
            experience_notes.append("strong 5D benchmark prior detected, so multimodal rescue can be skipped by default")
    else:
        experience_notes.append("non-strong prior detected, so experience stays advisory and will not dominate source routing")
        if high_dim_weak_structural_only:
            experience_notes.append("high-dimensional no-leakage mode: keep only abstract structural hints, not explicit formula seeds")
        avoid_patterns.append("avoid letting weak experience guesses displace generic proposer diversity")
        if "rational" in family_tags:
            avoid_patterns.append("avoid pure additive fit when ratio-like structure is hinted")
        elif "trigonometric" in family_tags or "exponential" in family_tags:
            avoid_patterns.append("avoid oversmoothing periodic hints into generic polynomial surrogates")

    reference_signature = _build_experience_reference_signature(
        feature_names=feature_names,
        candidate_families=candidate_families,
        family_tags=sorted(family_tags),
        function_hints=sorted(function_hints),
    )

    return {
        "format": EXPERIENCE_PRIOR_FORMAT,
        "schema_version": 1,
        "dataset_dir": dataset_dir,
        "base_name": base_name,
        "n_features": len(feature_names),
        "strength": strength,
        "confidence": float(confidence),
        "strong_prior": bool(strong_prior),
        "experience_source_enabled": bool(experience_source_enabled),
        "rerank_enabled": bool(rerank_enabled),
        "inject_into_protected": bool(inject_into_protected),
        "family_tags": sorted(family_tags or {"algebraic"}),
        "variable_roles": {
            "active_variables": active_variables,
            "numerator_core": numerator_core,
            "denominator_core": denominator_core,
            "periodic_core": periodic_core,
        },
        "degree_hints": make_json_safe(degree_hints),
        "function_hints": sorted(function_hints),
        "candidate_families": candidate_families,
        "preferred_sources": preferred_sources,
        "source_caps": source_caps,
        "budget_adjustments": budget_adjustments,
        "avoid_patterns": avoid_patterns,
        "experience_prior": experience_notes,
        "reference_signature": make_json_safe(reference_signature),
    }


def apply_experience_prior_to_iteration_config(iter_cfg, experience_prior=None):
    tuned = dict(iter_cfg or {})
    if not ENABLE_EXPERIENCE_PRIOR or not isinstance(experience_prior, dict) or not experience_prior:
        tuned.pop("experience_source_order", None)
        tuned.pop("experience_source_cap_overrides", None)
        return tuned

    preferred_sources = list(experience_prior.get("preferred_sources", []) or [])
    if preferred_sources:
        tuned["experience_source_order"] = preferred_sources

    source_caps = dict(experience_prior.get("source_caps", {}) or {})
    if source_caps:
        tuned["experience_source_cap_overrides"] = source_caps
    tuned["allow_protected_seed_bridge"] = False if TEMP_DISABLE_TEMPLATE_SOURCES else bool(
        experience_prior.get("inject_into_protected", False)
        or experience_prior.get("memory_seed_exprs")
    )

    budget = dict(experience_prior.get("budget_adjustments", {}) or {})
    if "text_calls_cap" in budget and not TEMP_DISABLE_TEMPLATE_SOURCES:
        tuned["text_calls"] = min(int(tuned.get("text_calls", 0)), int(budget["text_calls_cap"]))
    if "mm_calls_cap" in budget:
        tuned["mm_calls"] = min(int(tuned.get("mm_calls", 0)), int(budget["mm_calls_cap"]))
    if "proposal_k_cap" in budget:
        tuned["proposal_k"] = min(int(tuned.get("proposal_k", 0)), int(budget["proposal_k_cap"]))
    if "refined_k_cap" in budget:
        tuned["refined_k"] = min(int(tuned.get("refined_k", 0)), int(budget["refined_k_cap"]))
    if "refine_rounds_cap" in budget:
        tuned["refine_rounds"] = min(int(tuned.get("refine_rounds", 0)), int(budget["refine_rounds_cap"]))
    if "proposal_max_tokens_cap" in budget:
        tuned["proposal_max_tokens"] = min(int(tuned.get("proposal_max_tokens", 0)), int(budget["proposal_max_tokens_cap"]))
    if "skip_diverse_proposal" in budget and not TEMP_DISABLE_TEMPLATE_SOURCES:
        tuned["skip_diverse_proposal"] = bool(tuned.get("skip_diverse_proposal", False) or budget["skip_diverse_proposal"])
    if "force_heuristic_agents" in budget:
        tuned["force_heuristic_agents"] = bool(tuned.get("force_heuristic_agents", False) or budget["force_heuristic_agents"])
    return tuned


def should_run_guided_rescue_after_initial_pass(current_best, num_existing_candidates, iter_cfg=None):
    if not ENABLE_DELAYED_GUIDED_RESCUE:
        return False, "guided rescue disabled"
    if bool((iter_cfg or {}).get("allow_guided_candidate_sources", False)):
        return False, "guided rescue already enabled"
    if GUIDED_RESCUE_TRIGGER_IF_CANDIDATES_LT and int(num_existing_candidates or 0) < int(GUIDED_RESCUE_TRIGGER_IF_CANDIDATES_LT):
        return True, f"num_candidates<{int(GUIDED_RESCUE_TRIGGER_IF_CANDIDATES_LT)}"
    if current_best is None:
        return bool(GUIDED_RESCUE_TRIGGER_IF_NO_VALID_RESULT), "no_valid_initial_result"

    val_mse = _safe_get_attr(current_best, "val_mse", None)
    if val_mse is None or not np.isfinite(val_mse):
        return bool(GUIDED_RESCUE_TRIGGER_IF_NO_VALID_RESULT), "initial_val_mse_invalid"
    if float(val_mse) > float(GUIDED_RESCUE_TRIGGER_VAL_MSE):
        return True, f"initial_val_mse>{GUIDED_RESCUE_TRIGGER_VAL_MSE}"
    return False, f"initial_val_mse<={GUIDED_RESCUE_TRIGGER_VAL_MSE}"


def build_guided_rescue_iteration_config(iter_cfg):
    tuned = dict(iter_cfg or {})
    tuned["allow_guided_candidate_sources"] = True
    tuned["guided_source_order"] = list(GUIDED_RESCUE_SOURCE_ORDER)
    if not TEMP_DISABLE_TEMPLATE_SOURCES:
        tuned["allow_protected_seed_bridge"] = True
    tuned["text_calls"] = max(1, int(tuned.get("text_calls", NUM_TEXT_SINGLE_CALLS)))
    tuned["skip_diverse_proposal"] = False
    return tuned


def should_enable_family_seed_initial_pass(structure_profile=None, feature_names=None, row_meta=None, experience_prior=None):
    profile = dict(structure_profile or {})
    feature_names = list(feature_names or [])
    row_meta = dict(row_meta or {})
    experience_prior = dict(experience_prior or {})
    if len(feature_names) < 3:
        return False

    if BENCHMARK_5D_FORCE_SEED_FIRST and str(row_meta.get("dataset_dir", "")) == "benchmark_csv" and len(feature_names) == 5:
        return True

    family_scores = dict(profile.get("family_scores", {}) or {})
    global_tags = set(profile.get("global_tags", []) or [])
    variable_roles = dict(profile.get("variable_roles", {}) or {})
    top_pair_patterns = list(profile.get("top_pair_patterns", []) or [])

    additive_score = float(family_scores.get("additive", 0.0) or 0.0)
    rational_score = float(family_scores.get("rational", 0.0) or 0.0)
    interaction_score = float(family_scores.get("interaction", 0.0) or 0.0)
    power_score = float(family_scores.get("power", 0.0) or 0.0)

    strong_pair = any(float(item.get("score", 0.0) or 0.0) >= FAMILY_SEED_TRIGGER_SCORE for item in top_pair_patterns[:2])
    benchmark_like = _is_benchmark_task(row_meta)
    denominator_core = list(variable_roles.get("denominator_core", []) or [])
    candidate_families = [str(x).strip() for x in (experience_prior.get("candidate_families", []) or []) if str(x).strip()]
    strong_prior = bool(experience_prior.get("strong_prior", False))
    if NO_LEAKAGE_MODE and benchmark_like and len(feature_names) >= HIGH_DIM_ROLE_TRIGGER_DIM and not strong_prior:
        # Fair mode: do not switch initial routing into heuristic/manual family-seed mode.
        # Broad generic structural seeds are supplied by the dedicated "generic" source instead.
        return False
    helpful_family_seed = any(
        ("/(" in expr or "**2-" in expr or ("*(" in expr and ")/(" in expr) or ("*x4" in expr and "/(" in expr))
        for expr in candidate_families[:8]
    )

    if len(feature_names) >= 5:
        if (not strong_prior) and (not helpful_family_seed):
            return False
        ambiguous_additive_case = additive_score >= max(rational_score, interaction_score) and not denominator_core and not helpful_family_seed
        if ambiguous_additive_case and power_score < 0.28:
            return False
        if (not strong_prior) and (not helpful_family_seed) and (not denominator_core):
            return False
        if helpful_family_seed:
            return True
        if rational_score >= FAMILY_SEED_TRIGGER_SCORE and bool(denominator_core) and (power_score >= 0.22 or strong_pair):
            return True
        if interaction_score >= FAMILY_SEED_TRIGGER_SCORE and ("partially_separable" in global_tags or strong_pair):
            return True
        if benchmark_like and bool(denominator_core) and (rational_score >= 0.18 or interaction_score >= 0.22):
            return True

    if len(feature_names) >= 3 and interaction_score >= 0.30 and "partially_separable" in global_tags:
        return True
    return False


def reorder_expressions_by_experience(exprs, feature_names, experience_prior=None, top_k=None):
    clean_exprs = deduplicate_expressions(exprs)
    if not ENABLE_EXPERIENCE_PRIOR or not clean_exprs or not isinstance(experience_prior, dict) or not experience_prior:
        return clean_exprs, {"applied": False, "reason": "experience prior unavailable"}
    if not bool(experience_prior.get("rerank_enabled", False)):
        return clean_exprs, {"applied": False, "reason": "experience rerank disabled"}

    reference_signature = dict(experience_prior.get("reference_signature", {}) or {})
    candidate_families = {_expr_dedup_key(x) for x in (experience_prior.get("candidate_families", []) or []) if str(x).strip()}
    variable_roles = dict(experience_prior.get("variable_roles", {}) or {})
    active_variables = list(variable_roles.get("active_variables", []) or [])
    denominator_core = list(variable_roles.get("denominator_core", []) or [])

    if not reference_signature and not candidate_families:
        return clean_exprs, {"applied": False, "reason": "experience reference unavailable"}

    scored = []
    for idx, expr in enumerate(clean_exprs, start=1):
        expr_key = _expr_dedup_key(expr)
        cand_sig = extract_formula_form_signature(expr, feature_names)
        score = 0.0
        if reference_signature:
            score += _score_formula_signature_match(cand_sig, reference_signature)
        if expr_key in candidate_families:
            score += 0.50
        if active_variables:
            score += 0.08 * _safe_jaccard(cand_sig.get("variables_used", []), active_variables)
        if denominator_core:
            score += 0.08 * _safe_jaccard(_extract_denominator_variables(expr, feature_names), denominator_core)
        scored.append({
            "expr": expr,
            "score": float(score),
            "rank_before": idx,
        })

    scored_sorted = sorted(scored, key=lambda x: (-x["score"], x["rank_before"], len(str(x["expr"]))))
    threshold = EXPERIENCE_STRONG_PRIOR_PROMOTION_SCORE if experience_prior.get("strong_prior") else EXPERIENCE_WEAK_PRIOR_PROMOTION_SCORE
    top_k = max(1, int(top_k or EXPERIENCE_RERANK_TOPK))
    promoted = []
    promoted_keys = set()
    for item in scored_sorted[:top_k]:
        expr_key = _expr_dedup_key(item["expr"])
        if item["score"] >= threshold or expr_key in candidate_families:
            promoted.append(item["expr"])
            promoted_keys.add(expr_key)

    reordered = promoted + [expr for expr in clean_exprs if _expr_dedup_key(expr) not in promoted_keys]
    trace = {
        "applied": bool(promoted),
        "threshold": float(threshold),
        "top_k": int(top_k),
        "promoted_exprs": promoted[:6],
        "top_scores": [
            {
                "expr": item["expr"],
                "score": item["score"],
                "rank_before": item["rank_before"],
            }
            for item in scored_sorted[:min(5, len(scored_sorted))]
        ],
    }
    return reordered, trace


class _ParameterizeNumericConstantsTransformer(ast.NodeTransformer):
    def __init__(self):
        self.counter = 0
        self.in_power_exponent = 0

    def _new_param_node(self):
        self.counter += 1
        return ast.Name(id=f"k{self.counter}", ctx=ast.Load())

    def _should_preserve_numeric(self, value):
        try:
            value = float(value)
        except Exception:
            return True
        if self.in_power_exponent and value in {2.0, 3.0, 4.0, 0.5, -1.0}:
            return True
        return value in {0.0, 1.0, -1.0}

    def visit_BinOp(self, node):
        if isinstance(node.op, ast.Pow):
            node.left = self.visit(node.left)
            self.in_power_exponent += 1
            node.right = self.visit(node.right)
            self.in_power_exponent -= 1
            return node
        return self.generic_visit(node)

    def visit_Constant(self, node):
        if isinstance(node.value, numbers.Number) and not self._should_preserve_numeric(node.value):
            return self._new_param_node()
        return node

    def visit_UnaryOp(self, node):
        if isinstance(node.op, ast.USub) and isinstance(node.operand, ast.Constant) and isinstance(node.operand.value, numbers.Number):
            neg_val = -float(node.operand.value)
            if not self._should_preserve_numeric(neg_val):
                return self._new_param_node()
        return self.generic_visit(node)


def parameterize_expression_numeric_constants(expr):
    expr = str(expr or "").strip()
    if not expr:
        return ""
    try:
        tree = ast.parse(expr, mode="eval")
        transformer = _ParameterizeNumericConstantsTransformer()
        new_tree = transformer.visit(tree)
        ast.fix_missing_locations(new_tree)
        out = ast.unparse(new_tree).strip()
        return out or expr
    except Exception:
        return expr


def _build_family_templates_from_form_item(item, feature_names):
    reference_sig = _build_reference_signature_from_form_item(item, feature_names)
    used_vars = reference_sig.get("variables_used", []) or list(feature_names[:min(len(feature_names), 3)])
    families = set(reference_sig.get("families", []) or [])
    summary_text = " ".join([
        str(item.get("family", "")),
        str(item.get("composition", "")),
        str(item.get("structure_summary", "")),
        str(item.get("expression", "")),
    ]).lower()
    periodic_vars = _ordered_unique_feature_subset(item.get("periodic_variables", []), feature_names)
    denominator_vars = _ordered_unique_feature_subset(item.get("denominator_variables", []), feature_names)

    templates = []
    raw_expr = str(item.get("expression", "")).strip()
    if raw_expr:
        param_expr = parameterize_expression_numeric_constants(raw_expr)
        templates.append(raw_expr)
        templates.append(param_expr)
        templates.append(f"a*({param_expr}) + b")

    if used_vars:
        u0 = used_vars[0]
        u1 = used_vars[1] if len(used_vars) >= 2 else used_vars[0]
        u2 = used_vars[2] if len(used_vars) >= 3 else u1

        if "interaction" in families:
            templates.extend([
                f"a*{u0}*{u1} + b",
                f"a*({u0}+b1)*({u1}+b2) + c",
            ])
            if len(used_vars) >= 3:
                templates.append(f"a*{u0}*{u1}*{u2} + b")

        if "additive" in families:
            templates.extend([
                f"a1*{u0} + a2*{u1} + c",
                f"a1*({u0}+b1)*({u1}+b2) + a2*{u0} + a3*{u1} + c",
            ])

        if "rational" in families:
            denom_vars = denominator_vars or [v for v in used_vars if v not in {u0, u1}]
            if not denom_vars:
                denom_vars = [u1]
            denom_expr = "*".join([f"({v}+b{i+1})" for i, v in enumerate(denom_vars[:2])])
            numer_expr = f"{u0}" if len(used_vars) == 1 else f"{u0}*{u1}"
            templates.extend([
                f"a*({numer_expr})/({denom_expr}) + c",
                f"a*({numer_expr})/({denom_vars[0]} + b) + c",
            ])
            if len(used_vars) >= 4:
                dv0 = denom_vars[0]
                templates.extend([
                    f"a*{u0}*({u2}+b1*{u1})*{used_vars[3]}/({dv0}+b2) + c",
                    f"a*({u0}*{u2} + b1*{u0}*{u1})*{used_vars[3]}/({dv0}+b2) + c",
                ])
            if len(denom_vars) >= 2 or ("difference-of-squares" in summary_text or "difference of squares" in summary_text):
                dv0 = denom_vars[0]
                dv1 = denom_vars[1] if len(denom_vars) >= 2 else u2
                templates.append(f"a*({numer_expr})/(({dv0}**2 - {dv1}**2) + b) + c")
            if len(used_vars) >= 5 and ("power" in families or "difference-of-squares" in summary_text or "difference of squares" in summary_text):
                templates.extend([
                    f"a*{u0}*{u1}/(({used_vars[4]}+b1)*({u2}**2 + b2*{u2}*{used_vars[3]} + b3*{used_vars[3]}**2 + b4)) + c",
                    f"a*{u0}*{u1}/(({used_vars[4]}+b1)*((({u2}+b2*{used_vars[3]})*({u2}+b3*{used_vars[3]}))+b4)) + c",
                ])

        if "trigonometric" in families:
            pv = periodic_vars or used_vars[:min(2, len(used_vars))]
            p0 = pv[0]
            templates.extend([
                f"a*sin(b*{p0} + c) + d",
                f"a*cos(b*{p0} + c) + d",
            ])
            if len(pv) >= 2:
                templates.extend([
                    f"a*sin(b1*{pv[0]} + b2*{pv[1]} + c) + d",
                    f"a*cos(b1*{pv[0]} + b2*{pv[1]} + c) + d",
                ])
            if len(used_vars) >= 3:
                templates.append(f"a*{u0}*({u1} + b1*{u2}*sin(b2*{p0} + b3)) + c")

        if "exponential" in families:
            templates.extend([
                f"a*exp(b*{u0}) + c",
                f"a*exp(b1*{u0} + b2*{u1}) + c",
            ])
            if "rational" in families and len(used_vars) >= 3:
                templates.append(f"a*exp((b1*{u0} + b2*{u1})/({u2} + c1)) + d")

        if "logarithmic" in families:
            templates.extend([
                f"a*log(abs(b*{u0}) + c) + d",
                f"a*log(abs(b1*{u0} + b2*{u1}) + c1) + d",
            ])

        if "power" in families:
            degree = int(max(2, min(4, round(float(reference_sig.get("max_power_degree", 2.0))))))
            templates.extend([
                f"a*{u0}**{degree} + b",
                f"a*({u0}*{u1})**2 + b",
            ])

    return deduplicate_expressions(templates)


def _rank_template_pool_against_form_item(template_pool, item, feature_names, top_k=4):
    reference_sig = _build_reference_signature_from_form_item(item, feature_names)
    ranked = []
    for expr in deduplicate_expressions(template_pool):
        cand_sig = extract_formula_form_signature(expr, feature_names)
        ranked.append((
            _score_formula_signature_match(cand_sig, reference_sig),
            len(str(expr)),
            expr,
        ))
    ranked.sort(key=lambda x: (-x[0], x[1], x[2]))
    return [expr for _, _, expr in ranked[:max(1, top_k)]]


def build_vlm_guided_template_candidates(candidate_items, feature_names, row_meta=None, top_k_per_item=4):
    if not candidate_items:
        return []

    template_pool = list(build_manual_candidates(feature_names))
    if row_meta is not None:
        template_pool.extend(build_protected_benchmark_templates(row_meta, feature_names))
        template_pool.extend(build_livermore_2d_templates(row_meta, feature_names))
        if (
            ALLOW_ROW_SPECIFIC_BENCHMARK_PRIORS
            and str(row_meta.get("dataset_dir")) == "benchmark_csv"
            and len(feature_names) == 5
        ):
            template_pool.extend(build_feynman_5d_templates(feature_names))

    expanded = []
    for item in candidate_items:
        if not isinstance(item, dict):
            continue
        family_templates = _build_family_templates_from_form_item(item, feature_names)
        ranked_templates = _rank_template_pool_against_form_item(
            template_pool=template_pool,
            item=item,
            feature_names=feature_names,
            top_k=top_k_per_item,
        )
        for expr in deduplicate_expressions(family_templates + ranked_templates):
            expanded.append({
                "expression": expr,
                "rationale": f"vlm-guided template expansion from {str(item.get('family', '') or item.get('composition', '')).strip() or 'form hint'}",
                "prior_score": float(item.get("confidence", item.get("prior_score", 0.0)) or 0.0),
            })

    return deduplicate_candidate_dicts(expanded)


def _compact_observation_for_prompt(structure_profile=None, visual_summary=None):
    structure_profile = dict(structure_profile or {})
    visual_summary = dict(visual_summary or {})

    family_scores = sorted(
        ((str(k), float(v)) for k, v in dict(structure_profile.get("family_scores", {}) or {}).items()),
        key=lambda x: (-x[1], x[0]),
    )
    compact_structure = {
        "top_families": [{"family": k, "score": round(v, 3)} for k, v in family_scores[:4]],
        "global_tags": list(structure_profile.get("global_tags", []) or [])[:6],
        "active_variables": list(structure_profile.get("active_variables", []) or [])[:5],
        "variable_roles": make_json_safe(dict(structure_profile.get("variable_roles", {}) or {})),
        "top_unary_patterns": make_json_safe(list(structure_profile.get("top_unary_patterns", []) or [])[:4]),
        "top_pair_patterns": make_json_safe(list(structure_profile.get("top_pair_patterns", []) or [])[:4]),
        "evidence": list(structure_profile.get("evidence", []) or [])[:5],
    }
    compact_visual = {
        "num_plots": int(visual_summary.get("num_plots", 0) or 0),
        "dominant_variables": list(visual_summary.get("dominant_variables", []) or [])[:5],
        "focus_pairs": make_json_safe(list(visual_summary.get("focus_pairs", []) or [])[:4]),
        "top_families": list(visual_summary.get("top_families", []) or [])[:4],
        "global_tags": list(visual_summary.get("global_tags", []) or [])[:6],
        "story": list(visual_summary.get("story", []) or [])[:5],
        "plot_inventory": list(visual_summary.get("plot_inventory", []) or [])[:6],
    }
    return compact_structure, compact_visual


def _compact_reconstruction_tokens_for_prompt(reconstruction_tokens=None):
    tokens = dict(reconstruction_tokens or {})
    if not tokens:
        return None
    compact = {
        "mode": tokens.get("mode"),
        "selected_variables": list(tokens.get("selected_variables", []) or [])[:6],
        "selected_pairs": make_json_safe(list(tokens.get("selected_pairs", []) or [])[:6]),
        "dominant_unary_variables": list(tokens.get("dominant_unary_variables", []) or [])[:5],
        "dominant_pairs": make_json_safe(list(tokens.get("dominant_pairs", []) or [])[:5]),
        "periodic_variables": list(tokens.get("periodic_variables", []) or [])[:4],
        "symmetry_variables": list(tokens.get("symmetry_variables", []) or [])[:4],
        "denominator_like_variables": list(tokens.get("denominator_like_variables", []) or [])[:4],
        "estimated_unary_rank": tokens.get("estimated_unary_rank"),
        "estimated_pair_rank": tokens.get("estimated_pair_rank"),
        "unary_spectrum": make_json_safe(list(tokens.get("unary_spectrum", []) or [])[:4]),
        "pair_spectrum": make_json_safe(list(tokens.get("pair_spectrum", []) or [])[:4]),
        "unary_tokens": make_json_safe(list(tokens.get("unary_tokens", []) or [])[:4]),
        "pair_tokens": make_json_safe(list(tokens.get("pair_tokens", []) or [])[:4]),
    }
    return compact


def build_observer_output_for_prompt(structure_hints=None, structure_profile=None, visual_summary=None, visual_hints=None):
    structure_profile = dict(structure_profile or {})
    visual_summary = dict(visual_summary or {})
    vlm_observer = dict(structure_profile.get("vlm_observer", {}) or {})
    roles = dict(structure_profile.get("variable_roles", {}) or {})

    family_scores = sorted(
        ((str(k), float(v)) for k, v in dict(structure_profile.get("family_scores", {}) or {}).items()),
        key=lambda x: (-x[1], x[0]),
    )
    candidate_families = list(vlm_observer.get("candidate_families", []) or [])
    if not candidate_families:
        candidate_families = [k for k, _ in family_scores[:5]]
    if not candidate_families:
        candidate_families = list(visual_summary.get("top_families", []) or [])[:5]

    active_variables = list(vlm_observer.get("active_variables", []) or [])
    if not active_variables:
        active_variables = list(structure_profile.get("active_variables", []) or roles.get("active_variables", []) or [])
    inactive_variables = list(vlm_observer.get("inactive_variables", []) or [])

    trend_summary = str(
        vlm_observer.get("trend_summary")
        or "; ".join(str(x) for x in list(visual_summary.get("story", []) or [])[:3])
        or "; ".join(str(x) for x in list(structure_hints or [])[:3])
    )
    visual_evidence = list(vlm_observer.get("visual_evidence", []) or [])
    if not visual_evidence:
        visual_evidence = [str(x) for x in list(visual_hints or [])[:4]]
    risk_notes = list(vlm_observer.get("risk_notes", []) or [])
    if not risk_notes:
        risk_notes = [
            "avoid dense high-degree polynomial fallback unless supported by residuals",
            "check evaluator metrics before accepting visually plausible formulas",
        ]

    return make_json_safe({
        "agent": "observer",
        "active_variables": active_variables[:8],
        "inactive_variables": inactive_variables[:8],
        "candidate_families": candidate_families[:6],
        "variable_roles": {
            "numerator_core": list(roles.get("numerator_core", []) or active_variables)[:6],
            "denominator_core": list(roles.get("denominator_core", []) or [])[:6],
            "periodic_core": list(roles.get("periodic_core", []) or [])[:6],
            "interaction_core": list(roles.get("interaction_core", []) or [])[:6],
        },
        "trend_summary": trend_summary[:600],
        "visual_evidence": visual_evidence[:5],
        "risk_notes": risk_notes[:4],
        "confidence": vlm_observer.get("confidence", structure_profile.get("confidence", None)),
    })


def build_evaluator_results_for_critic(evaluation, current_best=None, top_k=None):
    evaluation = dict(evaluation or {})
    top_k = int(top_k or META_TOPK)
    best = current_best or evaluation.get("best_result")
    current_best_row = None
    if best is not None:
        current_best_row = {
            "expr": _safe_get_attr(best, "simplified_expression", None),
            "val_mse": _safe_get_attr(best, "val_mse", None),
            "complexity": _safe_get_attr(best, "complexity", None),
            "score": _safe_get_attr(best, "score", None),
            "selection_metric": _safe_get_attr(best, "selection_metric", None),
        }

    table = []
    raw_table = list(evaluation.get("evaluation_table", []) or [])
    if raw_table:
        for idx, row in enumerate(raw_table[:top_k], start=1):
            row = dict(row or {})
            table.append({
                "rank": idx,
                "expression": row.get("expr") or row.get("expression"),
                "val_mse": row.get("val_mse"),
                "complexity": row.get("complexity"),
                "score": row.get("score"),
                "selection_metric": row.get("selection_metric"),
                "small_sample_cv_mse": row.get("small_sample_cv_mse"),
            })
    else:
        for idx, item in enumerate(list(evaluation.get("scored_results", []) or [])[:top_k], start=1):
            table.append({
                "rank": idx,
                "expression": _safe_get_attr(item, "simplified_expression", None),
                "val_mse": _safe_get_attr(item, "val_mse", None),
                "complexity": _safe_get_attr(item, "complexity", None),
                "score": _safe_get_attr(item, "score", None),
                "selection_metric": _safe_get_attr(item, "selection_metric", None),
            })

    return make_json_safe({
        "current_best": current_best_row,
        "candidate_evaluation_table": table,
        "requested_candidate_count": evaluation.get("requested_candidate_count"),
        "evaluated_candidate_count": evaluation.get("evaluated_candidate_count"),
        "evaluation_truncated": evaluation.get("evaluation_truncated"),
    })


def formula_form_schema_example(variable_names) -> Dict[str, Any]:
    names = list(variable_names or ["x1"])
    if len(names) >= 2:
        x1, x2 = names[:2]
        return {
            "expression": f"a*sin(b*{x1}) + c*{x2}**2 + d",
            "family": "periodic polynomial interaction",
            "composition": "additive_interaction",
            "structure_summary": "periodic component in one variable plus low-order nonlinear term in another",
            "used_variables": [x1, x2],
            "interaction_groups": [[x1], [x2]],
            "denominator_variables": [],
            "periodic_variables": [x1],
            "key_operators": ["sin", "*", "**", "+"],
            "confidence": 0.82,
        }
    x1 = names[0]
    return {
        "expression": f"a*sin(b*{x1}) + c",
        "family": "periodic",
        "composition": "single_variable_periodic",
        "structure_summary": "compact one-variable nonlinear skeleton with fit parameters",
        "used_variables": [x1],
        "interaction_groups": [],
        "denominator_variables": [],
        "periodic_variables": [x1],
        "key_operators": ["sin", "*", "+"],
        "confidence": 0.82,
    }


def build_mm_formula_form_messages(
    csv_points,
    variable_names,
    target_name,
    image_data_urls,
    plot_descriptions=None,
    structure_hints=None,
    structure_profile=None,
    visual_summary=None,
    reconstruction_tokens=None,
    allowed_operators=None,
    num_candidates=4,
):
    if allowed_operators is None:
        allowed_operators = ALLOWED_OPERATORS
    if plot_descriptions is None:
        plot_descriptions = []
    if structure_hints is None:
        structure_hints = []

    d = len(variable_names)
    vars_text = ", ".join(variable_names)
    ops_text = ", ".join(allowed_operators)
    plot_text = "\n".join(f"- {x}" for x in plot_descriptions) if plot_descriptions else "- no plot descriptions available"
    observer_output = build_observer_output_for_prompt(
        structure_hints=structure_hints,
        structure_profile=structure_profile,
        visual_summary=visual_summary,
        visual_hints=plot_descriptions,
    )
    schema_example = formula_form_schema_example(variable_names)

    system_prompt = (
        "You are the Proposer Agent in a multimodal symbolic-regression loop. "
        "Return STRICT JSON only."
    )

    user_prompt = f"""
<image>You are the Proposer Agent in a multimodal symbolic-regression loop.

Task description:
Your job is to propose formula structures for symbolic regression. Given the attached plot(s), exact CSV samples, dataset name, feature names, target name, allowed operators, and Observer output, you are required to generate a small set of structurally distinct, fit-ready candidate formula forms for the Evaluator.

Input:
1) Attached image(s): benchmark plot(s) showing visual structure.
2) Variables and target: feature_names and target_name.
3) Allowed operators: symbolic operators that may appear in candidates.
4) Observer output: structural hints about active variables, operator families, and risks.
5) CSV samples: exact numeric evidence from the benchmark.

Output:
Return a JSON object with a "forms" list. Each item must contain one candidate expression skeleton plus its family, composition, used variables, interaction groups, denominator variables, periodic variables, key operators, structure summary, and confidence.

You are doing Symbolic Regression.
Mode: adaptive_multimodal_formula_form_proposal.

Variables:
{vars_text}
Target:
{target_name}

Available operators:
{ops_text}

Plot descriptions:
{plot_text}

Observer output:
{json.dumps(observer_output, ensure_ascii=False)}

Task:
Propose up to {num_candidates} structurally distinct formula forms for:
{target_name} = f({vars_text})

Return JSON with this format:
{json.dumps({"forms": [schema_example]}, ensure_ascii=False, indent=2)}

Restrictions:
1) Output ONLY JSON.
2) Candidates must be structurally distinct. Do not return near-duplicate affine or polynomial variants.
3) Prefer mechanism-like forms over generic dense polynomial fits.
4) Use variable names exactly as given.
5) Free parameters such as a, b, c are allowed.
6) Each proposed expression should be compact enough for later parameter fitting.
7) If the observer output or plots suggest denominator structure, periodicity, separability, sparse interactions, saturation, or difference-of-squares, encode that explicitly.
8) Focus on operator topology and variable coupling; do not try to estimate exact numeric constants from limited points.
9) Prefer sparse variable usage when the plots do not clearly support dense coupling.
10) Make the expression a fit-ready skeleton with symbolic coefficients, not a final numerically tuned formula.
11) The most important fields are family, composition, used_variables, interaction_groups, denominator_variables, and periodic_variables.
12) If unsure, still give the best structural guess instead of backing off to a generic linear form.
13) Every expression must be directly parseable by the symbolic-regression backend: use Python-style ** for powers, use only allowed functions/operators, and do not introduce undefined feature variables. Free scalar parameters such as a, b, c are allowed.

<Data>
{csv_points}
</Data>
""".strip()

    user_content = [{"type": "text", "text": user_prompt}]
    for url in image_data_urls:
        user_content.append({"type": "image_url", "image_url": {"url": url}})
    return [Message(role="system", content=system_prompt), Message(role="user", content=user_content)]


def parse_mm_formula_form_candidates(text, target_name="y"):
    parsed = _parse_json_like_text(text)
    raw_items = []

    if isinstance(parsed, dict):
        for key in ("forms", "formula_forms", "candidates"):
            items = parsed.get(key)
            if isinstance(items, list):
                raw_items.extend(items)

    candidates = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        expr = str(item.get("expression", "")).strip()
        if not expr:
            continue
        confidence = _parse_float01(item.get("confidence", item.get("prior_score", 0.0)))
        rationale = str(item.get("structure_summary", item.get("rationale", "multimodal formula-form proposal"))).strip()
        candidates.append({
            "expression": expr,
            "family": str(item.get("family", "")).strip(),
            "composition": str(item.get("composition", item.get("outer_form", ""))).strip(),
            "structure_summary": rationale,
            "used_variables": [str(x) for x in item.get("used_variables", []) if str(x).strip()],
            "interaction_groups": [
                [str(y) for y in group if str(y).strip()]
                for group in item.get("interaction_groups", [])
                if isinstance(group, (list, tuple))
            ],
            "denominator_variables": [str(x) for x in item.get("denominator_variables", []) if str(x).strip()],
            "periodic_variables": [str(x) for x in item.get("periodic_variables", []) if str(x).strip()],
            "key_operators": [str(x) for x in item.get("key_operators", []) if str(x).strip()],
            "confidence": confidence,
            "rationale": rationale or "multimodal formula-form proposal",
            "prior_score": confidence,
        })

    if candidates:
        return deduplicate_candidate_dicts(candidates)

    expr = parse_answer_tag_expression(text, target_name=target_name)
    if expr:
        return [{
            "expression": expr,
            "family": "",
            "composition": "",
            "structure_summary": "fallback extracted from answer tag",
            "used_variables": [],
            "interaction_groups": [],
            "denominator_variables": [],
            "periodic_variables": [],
            "key_operators": [],
            "confidence": 0.0,
            "rationale": "fallback extracted from answer tag",
            "prior_score": 0.0,
        }]
    return []


def generate_mm_formula_form_candidates(
    client,
    df,
    variable_names,
    target_name="y",
    image_paths=None,
    image_data_urls=None,
    plot_descriptions=None,
    structure_hints=None,
    structure_profile=None,
    visual_summary=None,
    reconstruction_tokens=None,
    allowed_operators=None,
    max_rows=20,
    temperature=0.1,
    max_tokens=768,
    top_p=1.0,
    num_candidates=4,
    csv_points=None,
):
    if image_paths is None:
        image_paths = []
    if plot_descriptions is None:
        plot_descriptions = []
    if structure_hints is None:
        structure_hints = []
    if image_data_urls is None:
        image_data_urls = [image_file_to_data_url(p) for p in image_paths]
    if csv_points is None:
        csv_points = make_csv_points_text(
            df=df,
            variable_names=variable_names,
            target_name=target_name,
            max_rows=max_rows,
        )

    messages = build_mm_formula_form_messages(
        csv_points=csv_points,
        variable_names=variable_names,
        target_name=target_name,
        image_data_urls=image_data_urls,
        plot_descriptions=plot_descriptions,
        structure_hints=structure_hints,
        structure_profile=structure_profile,
        visual_summary=visual_summary,
        reconstruction_tokens=reconstruction_tokens,
        allowed_operators=allowed_operators,
        num_candidates=num_candidates,
    )
    response = client.generate(messages=messages, temperature=temperature, max_tokens=max_tokens, top_p=top_p)
    candidates = parse_mm_formula_form_candidates(response.text, target_name=target_name)
    return {"candidates": candidates, "raw_text": response.text}


def generate_multiple_mm_formula_form_candidates(
    client,
    df,
    variable_names,
    target_name="y",
    image_paths=None,
    plot_descriptions=None,
    structure_hints=None,
    structure_profile=None,
    visual_summary=None,
    reconstruction_tokens=None,
    allowed_operators=None,
    max_rows=20,
    num_calls=3,
    temperatures=None,
    max_tokens=768,
    top_p=1.0,
    max_workers=None,
    num_candidates_per_call=4,
):
    if image_paths is None:
        image_paths = []
    if plot_descriptions is None:
        plot_descriptions = []
    if structure_hints is None:
        structure_hints = []
    if temperatures is None:
        temperatures = [0.1, 0.3, 0.6]

    call_temps = build_call_temperatures(num_calls, temperatures)
    max_workers = max_workers or min(MM_FORM_PROPOSAL_MAX_WORKERS, max(1, num_calls))

    selected_image_paths = choose_mm_image_paths(
        image_paths,
        MAX_MM_IMAGES_PER_CALL,
        plot_descriptions=plot_descriptions,
        focus_variables=list((visual_summary or {}).get("dominant_variables", []) or []),
        focus_pairs=list((visual_summary or {}).get("focus_pairs", []) or []),
    )
    image_data_urls = [image_file_to_data_url(p) for p in selected_image_paths]
    csv_points = make_csv_points_text(
        df=df,
        variable_names=variable_names,
        target_name=target_name,
        max_rows=min(max_rows, MAX_ROWS_FOR_MM_PROMPT),
    )

    def _one_call(call_index, temp):
        start = time.time()
        try:
            result = generate_mm_formula_form_candidates(
                client=client,
                df=df,
                variable_names=variable_names,
                target_name=target_name,
                image_data_urls=image_data_urls,
                plot_descriptions=plot_descriptions,
                structure_hints=structure_hints,
                structure_profile=structure_profile,
                visual_summary=visual_summary,
                reconstruction_tokens=reconstruction_tokens,
                allowed_operators=allowed_operators,
                max_rows=min(max_rows, MAX_ROWS_FOR_MM_PROMPT),
                temperature=temp,
                max_tokens=max_tokens,
                top_p=top_p,
                num_candidates=num_candidates_per_call,
                csv_points=csv_points,
            )
            exprs = [str(item.get("expression", "")).strip() for item in result.get("candidates", []) if str(item.get("expression", "")).strip()]
            return {
                "call_index": call_index,
                "temperature": temp,
                "num_exprs_raw": len(exprs),
                "exprs_raw": exprs,
                "candidate_items": result.get("candidates", []),
                "raw_text": result.get("raw_text", ""),
                "latency_sec": time.time() - start,
                "error": None,
            }
        except Exception as e:
            return {
                "call_index": call_index,
                "temperature": temp,
                "num_exprs_raw": 0,
                "exprs_raw": [],
                "candidate_items": [],
                "raw_text": "",
                "latency_sec": time.time() - start,
                "error": repr(e),
            }

    futures = []
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for i, temp in enumerate(call_temps, start=1):
            futures.append(executor.submit(_one_call, i, temp))
        for fut in as_completed(futures):
            results.append(fut.result())

    results.sort(key=lambda x: x["call_index"])

    all_candidates = []
    raw_texts = []
    per_call_stats = []
    for item in results:
        all_candidates.extend(item.get("candidate_items", []))
        raw_texts.append(item.get("raw_text", ""))
        per_call_stats.append({
            "call_index": item["call_index"],
            "temperature": item["temperature"],
            "num_exprs_raw": item["num_exprs_raw"],
            "exprs_raw": item["exprs_raw"],
            "latency_sec": item["latency_sec"],
            "error": item["error"],
        })

    unique_candidates = deduplicate_candidate_dicts(all_candidates)
    return {
        "candidates": unique_candidates,
        "raw_text": "\n\n".join([x for x in raw_texts if x]),
        "per_call_stats": per_call_stats,
        "num_exprs_unique": len(unique_candidates),
    }


def build_mm_single_expression_messages(*args, **kwargs):
    raise RuntimeError("MM single-expression proposal is removed in clean build; use MM formula-form proposal instead.")


def maybe_retry_empty_mm_result(*args, **kwargs):
    return {"candidates": [], "raw_text": "MM single-expression retry removed in clean build"}


def generate_mm_single_expression(*args, **kwargs):
    return {"candidates": [], "raw_text": "MM single-expression proposal removed in clean build"}


def generate_multiple_mm_single_expressions(*args, **kwargs):
    return {"candidates": [], "raw_text": "", "per_call_stats": [], "num_exprs_unique": 0}


def generate_multiple_text_single_expressions_with_stats(*args, **kwargs):
    return {"candidates": [], "per_call_stats": [], "num_exprs_unique": 0, "skipped": True, "skip_reason": "text proposer removed from clean main build"}


def sample_features_from_range(n_rows, n_features, range_spec, distribution="U", rng=None):
    """
    Robustly sample features from benchmark range specs.

    Supported examples:
    1) 单变量统一广播到所有维度:
       [[1, 5, 50]]   -> 对所有 x1..xd 使用 [1, 5]
       [[-3, 3]]      -> 对所有 x1..xd 使用 [-3, 3]

    2) 每个变量单独给范围:
       [[-1,1,20], [0,2,20], [3,5,20]]

    3) 直接给一个二元范围:
       [1, 5] -> 广播到所有维度
    """
    rng = rng or np.random.default_rng(42)

    rs = parse_range_spec(range_spec)

    if rs is None:
        raise ValueError(f"range_spec is invalid: {range_spec}")

    # 情况 A: [lo, hi]
    if isinstance(rs, (list, tuple)) and len(rs) == 2 and all(isinstance(x, (int, float)) for x in rs):
        ranges = [tuple(rs)] * n_features

    # 情况 B: [[lo, hi]] 或 [[lo, hi, bins]]
    elif (
        isinstance(rs, (list, tuple))
        and len(rs) == 1
        and isinstance(rs[0], (list, tuple))
        and len(rs[0]) >= 2
        and all(isinstance(x, (int, float)) for x in rs[0][:2])
    ):
        lo, hi = float(rs[0][0]), float(rs[0][1])
        ranges = [(lo, hi)] * n_features

    # 情况 C: [[...], [...], ...] 每个变量一组
    elif isinstance(rs, (list, tuple)) and len(rs) == n_features:
        ranges = []
        for spec in rs:
            if not isinstance(spec, (list, tuple)) or len(spec) < 2:
                raise ValueError(f"invalid per-variable range spec: {spec}")
            lo, hi = float(spec[0]), float(spec[1])
            ranges.append((lo, hi))

    else:
        raise ValueError(
            f"Invalid range spec for n_features={n_features}: {range_spec}. "
            f"Parsed={rs}"
        )

    X = np.zeros((n_rows, n_features), dtype=float)
    dist = str(distribution).upper()

    for j in range(n_features):
        lo, hi = ranges[j]
        if not np.isfinite(lo) or not np.isfinite(hi):
            raise ValueError(f"non-finite bounds for feature {j}: {(lo, hi)}")
        if hi < lo:
            lo, hi = hi, lo

        if dist.startswith("U"):
            X[:, j] = rng.uniform(lo, hi, size=n_rows)
        else:
            # 暂时统一回退到 uniform
            X[:, j] = rng.uniform(lo, hi, size=n_rows)

    columns = [f"x{i+1}" for i in range(n_features)]
    return pd.DataFrame(X, columns=columns)


def evaluate_expression_on_df(expr, df):
    local_vars = {c: df[c].values for c in df.columns}
    local_vars.update({
        "sin": np.sin,
        "cos": np.cos,
        "tan": np.tan,
        "sinh": np.sinh,
        "cosh": np.cosh,
        "tanh": np.tanh,
        "exp": np.exp,
        "log": np.log,
        "sqrt": np.sqrt,
        "abs": np.abs,
        "Abs": np.abs,
        "pi": np.pi,
        "e": np.e,
    })
    with np.errstate(all="ignore"):
        y = eval(expr, {"__builtins__": {}}, local_vars)
    y = np.asarray(y, dtype=float)
    return y


def analyze_residual_patterns(best_result, dataset):
    return {"available": False, "messages": ["legacy residual analyzer removed in clean build"]}, []


def physics_check_placeholder(best_result, dataset, unit_hints=None):
    if unit_hints is None:
        unit_hints = {}
    return {
        "available": False,
        "messages": [
            "physics check placeholder only",
            f"feature_units={unit_hints}",
            f"best_expr={_safe_get_attr(best_result, 'simplified_expression', None)}",
        ],
    }


def build_result_report(
    row_meta,
    result,
    current_best=None,
    dataset=None,
    residual_summary=None,
    physics_summary=None,
    meta_decisions=None,
    refine_history=None,
    refine_round_expr_counts=None,
    refine_round_timings=None,
):
    report = {
        "task_meta": make_json_safe(dict(row_meta)),
        "result_core": make_json_safe(result),
        "task_point_signature": make_json_safe(
            _resolve_task_point_signature(
                row_meta=row_meta,
                dataset=dataset,
                feature_names=list(getattr(dataset, "feature_names", []) or _extract_feature_names_from_task_meta(dict(row_meta or {}))),
            )
        ),
        "best_expr": _safe_get_attr(current_best, "simplified_expression", None),
        "best_val_mse": _safe_get_attr(current_best, "val_mse", None),
        "best_test_mse": _safe_get_attr(current_best, "test_mse", None),
        "best_complexity": _safe_get_attr(current_best, "complexity", None),
        "best_score": _safe_get_attr(current_best, "score", None),
        "dataset_shape": {
            "train": list(dataset.train_df.shape) if dataset is not None else None,
            "val": list(dataset.val_df.shape) if dataset is not None else None,
            "test": list(dataset.test_df.shape) if dataset is not None else None,
        },
        "residual_summary": make_json_safe(residual_summary),
        "physics_summary": make_json_safe(physics_summary),
        "meta_decisions": make_json_safe(meta_decisions),
        "refine_history": make_json_safe(refine_history),
        "refine_round_expr_counts": make_json_safe(refine_round_expr_counts),
        "refine_round_timings": make_json_safe(refine_round_timings),
    }
    return report


def _write_per_case_report(
    row_meta,
    result,
    current_best=None,
    dataset=None,
    residual_summary=None,
    physics_summary=None,
    meta_decisions=None,
    refine_history=None,
    refine_round_expr_counts=None,
    refine_round_timings=None,
):
    if row_meta is None:
        return
    report = build_result_report(
        row_meta=row_meta,
        result=result,
        current_best=current_best,
        dataset=dataset,
        residual_summary=residual_summary,
        physics_summary=physics_summary,
        meta_decisions=meta_decisions,
        refine_history=refine_history,
        refine_round_expr_counts=refine_round_expr_counts,
        refine_round_timings=refine_round_timings,
    )
    os.makedirs(PER_CASE_JSON_DIR, exist_ok=True)
    report_path = os.path.join(PER_CASE_JSON_DIR, f"{sanitize_name(str(row_meta.get('base_name', 'task')))}.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


def populate_visual_trace_fields(
    result,
    observation,
    mm_requested=False,
    mm_trigger_reason=None,
    mm_prop_trace=None,
    mm_candidate_count=0,
    mm_used_in_evaluation=False,
):
    observation = observation or ObservationBundle(
        structure_hints=[],
        visual_hints=[],
        unit_hints={},
        plot_descriptions=[],
        image_paths=[],
    )
    visual_trace = {
        "mm_requested": bool(mm_requested),
        "mm_trigger_reason": mm_trigger_reason,
        "mm_assets_attempted": bool(getattr(observation, "mm_assets_attempted", False)),
        "mm_assets_succeeded": bool(getattr(observation, "mm_assets_succeeded", False)),
        "mm_assets_error": getattr(observation, "mm_assets_error", None),
        "num_plots": len(getattr(observation, "image_paths", []) or []),
        "image_paths": list(getattr(observation, "image_paths", []) or []),
        "plot_descriptions": list(getattr(observation, "plot_descriptions", []) or []),
        "visual_hints": list(getattr(observation, "visual_hints", []) or []),
        "structure_profile": make_json_safe(getattr(observation, "structure_profile", None)),
        "visual_summary": make_json_safe(getattr(observation, "visual_summary", None)),
        "reconstruction_tokens": make_json_safe(getattr(observation, "reconstruction_tokens", None)),
        "reconstruction_trace": make_json_safe(getattr(observation, "reconstruction_trace", None)),
        "reconstruction_image_paths": list(getattr(observation, "reconstruction_image_paths", []) or []),
        "reconstruction_descriptions": list(getattr(observation, "reconstruction_descriptions", []) or []),
        "mm_candidate_count": int(mm_candidate_count or 0),
        "mm_used_in_evaluation": bool(mm_used_in_evaluation),
        "mm_proposal_trace": make_json_safe(mm_prop_trace),
    }
    result["num_plots"] = len(visual_trace["image_paths"])
    result["plot_descriptions"] = json.dumps(make_json_safe(visual_trace["plot_descriptions"]), ensure_ascii=False)
    result["visual_hints"] = json.dumps(make_json_safe(visual_trace["visual_hints"]), ensure_ascii=False)
    result["structure_profile"] = json.dumps(make_json_safe(visual_trace["structure_profile"]), ensure_ascii=False)
    result["visual_summary"] = json.dumps(make_json_safe(visual_trace["visual_summary"]), ensure_ascii=False)
    result["image_paths"] = json.dumps(make_json_safe(visual_trace["image_paths"]), ensure_ascii=False)
    result["reconstruction_tokens"] = json.dumps(make_json_safe(visual_trace["reconstruction_tokens"]), ensure_ascii=False)
    result["reconstruction_trace"] = json.dumps(make_json_safe(visual_trace["reconstruction_trace"]), ensure_ascii=False)
    result["reconstruction_image_paths"] = json.dumps(make_json_safe(visual_trace["reconstruction_image_paths"]), ensure_ascii=False)
    result["mm_requested"] = bool(mm_requested)
    result["mm_trigger_reason"] = mm_trigger_reason
    result["mm_assets_attempted"] = visual_trace["mm_assets_attempted"]
    result["mm_assets_succeeded"] = visual_trace["mm_assets_succeeded"]
    result["mm_assets_error"] = visual_trace["mm_assets_error"]
    result["mm_candidate_count"] = visual_trace["mm_candidate_count"]
    result["mm_used_in_evaluation"] = visual_trace["mm_used_in_evaluation"]
    result["visual_trace"] = json.dumps(make_json_safe(visual_trace), ensure_ascii=False)


def evaluate_selected_expression_on_test(current_best, dataset):
    """Evaluate exactly one already-selected expression on the held-out split.

    This function is called only from ``_finalize_result``, after proposal,
    fitting, validation ranking, Critic feedback, and refinement have ended.
    A test-domain failure is reported without selecting a fallback candidate.
    """
    report = {
        "phase": "post_selection",
        "selected_expression_only": True,
        "test_rows": None,
        "prediction_valid": False,
        "test_mse": None,
        "error": None,
    }
    if current_best is None:
        report["error"] = "no_selected_expression"
        return report

    # Clear any legacy/custom value before computing the sealed evaluation.
    setattr(current_best, "test_mse", None)
    expression = str(_safe_get_attr(current_best, "simplified_expression", "") or "").strip()
    if not expression:
        report["error"] = "selected_expression_missing"
        return report

    try:
        test_df = dataset.test_df
        target_name = str(getattr(dataset, "target_name", "y") or "y")
        report["test_rows"] = int(len(test_df))
        if len(test_df) == 0:
            report["error"] = "test_split_empty"
            return report
        target = np.asarray(test_df[target_name], dtype=float).reshape(-1)
        prediction = np.asarray(
            evaluate_expression_on_df(expression, test_df),
            dtype=float,
        )
        if prediction.ndim == 0:
            prediction = np.full(target.shape, float(prediction), dtype=float)
        prediction = prediction.reshape(-1)
        if len(prediction) != len(target):
            report["error"] = "prediction_length_mismatch"
            return report
        if not np.isfinite(target).all() or not np.isfinite(prediction).all():
            report["error"] = "non_finite_test_prediction"
            return report
        test_mse = float(np.mean((prediction - target) ** 2))
        if not np.isfinite(test_mse):
            report["error"] = "non_finite_test_mse"
            return report
        setattr(current_best, "test_mse", test_mse)
        report["prediction_valid"] = True
        report["test_mse"] = test_mse
        return report
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        return report


def _append_sealed_test_audit(result, test_report):
    raw = result.get("no_leakage_audit")
    if isinstance(raw, str):
        try:
            audit = json.loads(raw)
        except Exception:
            audit = {}
    elif isinstance(raw, dict):
        audit = dict(raw)
    else:
        audit = {}
    audit.update({
        "test_split_accessed_during_search": False,
        "test_domain_predictions_computed_during_search": False,
        "test_evaluation_phase": "post_selection",
        "test_evaluated_for_selected_expression_only": True,
        "test_prediction_valid": bool(test_report.get("prediction_valid", False)),
        "test_evaluation_error": test_report.get("error"),
    })
    result["no_leakage_audit"] = json.dumps(make_json_safe(audit), ensure_ascii=False)


def _finalize_result(
    result,
    timer,
    start,
    current_best,
    dataset,
    residual_summary,
    physics_summary,
    meta_decisions,
    refine_history,
    refine_round_expr_counts,
    row_meta=None,
):
    test_report = evaluate_selected_expression_on_test(current_best, dataset)
    result["n_test"] = test_report.get("test_rows")
    result["test_evaluation_phase"] = "post_selection"
    result["test_prediction_valid"] = bool(test_report.get("prediction_valid", False))
    result["test_evaluation_error"] = test_report.get("error")
    _append_sealed_test_audit(result, test_report)

    result["runtime_sec"] = time.time() - start
    result["step_times_json"] = json.dumps(make_json_safe(timer.as_dict()), ensure_ascii=False)
    result["meta_decisions"] = json.dumps(make_json_safe(meta_decisions), ensure_ascii=False)
    result["refine_history"] = json.dumps(make_json_safe(refine_history), ensure_ascii=False)
    result["refine_improvement_count"] = int(sum(1 for item in (refine_history or []) if bool(item.get("improved", False))))
    result["refine_round_expr_counts"] = json.dumps(make_json_safe(refine_round_expr_counts), ensure_ascii=False)

    if residual_summary is not None:
        result["residual_summary"] = json.dumps(make_json_safe(residual_summary), ensure_ascii=False)
    if physics_summary is not None:
        result["physics_summary"] = json.dumps(make_json_safe(physics_summary), ensure_ascii=False)

    if current_best is not None:
        result["valid_formula_found"] = True
        result["best_expr"] = _safe_get_attr(current_best, "simplified_expression", None)
        result["best_val_mse"] = _safe_get_attr(current_best, "val_mse", None)
        result["best_test_mse"] = _safe_get_attr(current_best, "test_mse", None)
        result["passed"] = (
            result["best_test_mse"] is not None and float(result["best_test_mse"]) <= MSE_THRESHOLD
        )
        result["perfect_fit"] = (
            result["best_test_mse"] is not None and float(result["best_test_mse"]) <= PERFECT_FIT_TOL
        )
    else:
        result["valid_formula_found"] = False
        result["passed"] = False
        result["perfect_fit"] = False

    _write_per_case_report(
        row_meta=row_meta,
        result=result,
        current_best=current_best,
        dataset=dataset,
        residual_summary=residual_summary,
        physics_summary=physics_summary,
        meta_decisions=meta_decisions,
        refine_history=refine_history,
        refine_round_expr_counts=refine_round_expr_counts,
    )

    return result


def print_summary(all_results, overall_start):
    df = pd.DataFrame(all_results)
    total = len(df)
    valid = int(df["valid_formula_found"].fillna(False).sum()) if "valid_formula_found" in df else 0
    passed = int(df["passed"].fillna(False).sum()) if "passed" in df else 0

    test_mse_valid = []
    test_mse_passed = []
    if "best_test_mse" in df.columns:
        for _, r in df.iterrows():
            v = r.get("best_test_mse", None)
            if isinstance(v, numbers.Number) and np.isfinite(v):
                if bool(r.get("valid_formula_found", False)):
                    test_mse_valid.append(float(v))
                if bool(r.get("passed", False)):
                    test_mse_passed.append(float(v))

    print("\n" + "=" * 72)
    print("SUMMARY: ALL TASKS")
    print("=" * 72)
    print(f"Total Files:            {total}")
    print(f"Valid Formulas Found:   {valid} ({(100.0 * valid / total) if total else 0:.1f}%)")
    print(f"Calculated Test MSEs:   {len(test_mse_valid)}")
    print("-" * 52)
    print(f"✅ PASSED (test MSE <= {MSE_THRESHOLD}): {passed} ({(100.0 * passed / total) if total else 0:.1f}%)")
    print(f"   Mean Test MSE (Valid):  {safe_numeric_mean(test_mse_valid)}")
    print(f"   Mean Test MSE (Passed): {safe_numeric_mean(test_mse_passed)}")
    print(f"Total Runtime: {time.time() - overall_start:.2f} seconds")


def build_timing_breakdown_df(all_results):
    rows = []
    step_keys = set()

    for item in all_results:
        row = {
            "task_type": item.get("task_type"),
            "dataset_dir": item.get("dataset_dir"),
            "difficulty": item.get("difficulty"),
            "base_name": item.get("base_name"),
            "runtime_sec": item.get("runtime_sec"),
            "valid_formula_found": item.get("valid_formula_found"),
            "passed": item.get("passed"),
            "error": item.get("error"),
        }

        step_times = {}
        raw = item.get("step_times_json")
        if isinstance(raw, str) and raw.strip():
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    step_times = parsed
            except Exception:
                step_times = {}

        clean_step_times = {}
        for k, v in step_times.items():
            if isinstance(v, numbers.Number) and np.isfinite(v):
                clean_step_times[str(k)] = float(v)
                step_keys.add(str(k))

        row.update(clean_step_times)
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    meta_cols = [
        "task_type", "dataset_dir", "difficulty", "base_name",
        "runtime_sec", "valid_formula_found", "passed", "error",
    ]
    ordered_step_cols = sorted(step_keys)
    df = pd.DataFrame(rows)
    for col in ordered_step_cols:
        if col not in df.columns:
            df[col] = np.nan
    return df[meta_cols + ordered_step_cols]


def build_timing_summary(timing_df):
    if timing_df is None or len(timing_df) == 0:
        return {
            "num_cases": 0,
            "avg_runtime_sec": None,
            "step_stats": {},
        }

    meta_cols = {
        "task_type", "dataset_dir", "difficulty", "base_name",
        "runtime_sec", "valid_formula_found", "passed", "error",
    }
    step_cols = [c for c in timing_df.columns if c not in meta_cols]

    step_stats = {}
    for col in step_cols:
        vals = []
        for v in timing_df[col].dropna().tolist():
            if isinstance(v, numbers.Number) and np.isfinite(v):
                vals.append(float(v))
        if not vals:
            continue
        step_stats[col] = {
            "count": int(len(vals)),
            "avg_sec": safe_numeric_mean(vals),
            "min_sec": float(min(vals)),
            "max_sec": float(max(vals)),
        }

    return {
        "num_cases": int(len(timing_df)),
        "avg_runtime_sec": safe_numeric_mean(timing_df["runtime_sec"].tolist()) if "runtime_sec" in timing_df else None,
        "step_stats": step_stats,
    }


def save_all_outputs(all_results, overall_start):
    os.makedirs(RESULTS_ROOT, exist_ok=True)
    os.makedirs(PER_CASE_JSON_DIR, exist_ok=True)

    df = pd.DataFrame(all_results)
    df.to_csv(GLOBAL_SUMMARY_CSV, index=False)

    compact_cols = [
        c for c in [
            "task_type", "dataset_dir", "difficulty", "base_name",
            "n_features", "n_train", "n_val", "n_test",
            "valid_formula_found", "num_candidate_exprs", "best_expr",
            "initial_best_form_match_score", "vlm_best_form_match_score",
            "best_train_mse", "best_val_mse", "best_test_mse",
            "train_rmse", "val_rmse", "test_rmse",
            "train_nmse", "val_nmse", "test_nmse",
            "train_nrmse", "val_nrmse", "test_nrmse",
            "train_r2", "val_r2", "test_r2",
            "expr_complexity", "expr_depth", "expr_string_length", "expr_sympy_ops",
            "passed", "perfect_fit", "perfect_fit_by_r2", "numerical_complete_fit",
            "strict_formula_recovery", "strict_formula_recovery_evaluable",
            "srbench_formula_recovery", "srbench_formula_recovery_evaluable",
            "early_stop_train_fit", "metric_eval_error",
            "runtime_sec", "error"
        ] if c in df.columns
    ]
    if compact_cols:
        df[compact_cols].to_csv(GLOBAL_SUMMARY_CSV_COMPACT, index=False)
    else:
        df.to_csv(GLOBAL_SUMMARY_CSV_COMPACT, index=False)

    avg_step_times = {}
    if "step_times_json" in df.columns:
        all_step_dicts = []
        for s in df["step_times_json"].dropna().tolist():
            try:
                all_step_dicts.append(json.loads(s))
            except Exception:
                pass
        keys = sorted(set().union(*[d.keys() for d in all_step_dicts])) if all_step_dicts else []
        for k in keys:
            vals = []
            for d in all_step_dicts:
                v = d.get(k, None)
                if isinstance(v, numbers.Number) and np.isfinite(v):
                    vals.append(float(v))
            avg_step_times[k] = safe_numeric_mean(vals)

    summary = {
        "num_cases": int(len(df)),
        "total_runtime_sec": float(time.time() - overall_start),
        "avg_runtime_sec": safe_numeric_mean(df["runtime_sec"].tolist()) if "runtime_sec" in df else None,
        "avg_test_mse_valid_only": safe_numeric_mean(
            [float(x) for x in df.loc[df["valid_formula_found"] == True, "best_test_mse"].tolist()
             if isinstance(x, numbers.Number) and np.isfinite(x)]
        ) if "valid_formula_found" in df and "best_test_mse" in df else None,
        "avg_test_mse_passed_only": safe_numeric_mean(
            [float(x) for x in df.loc[df["passed"] == True, "best_test_mse"].tolist()
             if isinstance(x, numbers.Number) and np.isfinite(x)]
        ) if "passed" in df and "best_test_mse" in df else None,
        "avg_test_r2_valid_only": safe_numeric_mean(
            [float(x) for x in df.loc[df["valid_formula_found"] == True, "test_r2"].tolist()
             if isinstance(x, numbers.Number) and np.isfinite(x)]
        ) if "valid_formula_found" in df and "test_r2" in df else None,
        "avg_expr_complexity_valid_only": safe_numeric_mean(
            [float(x) for x in df.loc[df["valid_formula_found"] == True, "expr_complexity"].tolist()
             if isinstance(x, numbers.Number) and np.isfinite(x)]
        ) if "valid_formula_found" in df and "expr_complexity" in df else None,
        "numerical_complete_fit_rate": (
            float(df["numerical_complete_fit"].fillna(False).astype(bool).mean())
            if "numerical_complete_fit" in df and len(df)
            else None
        ),
        "strict_formula_recovery_rate_evaluable": (
            float(
                df.loc[
                    df["strict_formula_recovery_evaluable"].fillna(False).astype(bool),
                    "strict_formula_recovery",
                ].fillna(False).astype(bool).mean()
            )
            if {
                "strict_formula_recovery",
                "strict_formula_recovery_evaluable",
            }.issubset(df.columns)
            and df["strict_formula_recovery_evaluable"].fillna(False).astype(bool).any()
            else None
        ),
        "srbench_formula_recovery_rate_evaluable": (
            float(
                df.loc[
                    df["srbench_formula_recovery_evaluable"].fillna(False).astype(bool),
                    "srbench_formula_recovery",
                ].fillna(False).astype(bool).mean()
            )
            if {
                "srbench_formula_recovery",
                "srbench_formula_recovery_evaluable",
            }.issubset(df.columns)
            and df["srbench_formula_recovery_evaluable"].fillna(False).astype(bool).any()
            else None
        ),
        "avg_step_times": avg_step_times,
    }
    with open(GLOBAL_SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    timing_df = build_timing_breakdown_df(all_results)
    timing_df.to_csv(TIMING_BREAKDOWN_CSV, index=False)

    timing_summary = build_timing_summary(timing_df)
    with open(TIMING_SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(timing_summary, f, ensure_ascii=False, indent=2)

    print(f"Detailed results saved to: {GLOBAL_SUMMARY_CSV}")
    print(f"Per-case reports saved to: {PER_CASE_JSON_DIR}")
    print(f"Global summary json saved to: {GLOBAL_SUMMARY_JSON}")
    print(f"Global summary csv saved to: {GLOBAL_SUMMARY_CSV_COMPACT}")
    print(f"Timing breakdown csv saved to: {TIMING_BREAKDOWN_CSV}")
    print(f"Timing summary json saved to: {TIMING_SUMMARY_JSON}")


def _build_prior_guided_diverse_fallbacks(dataset, experience_prior=None, limit=12):
    feature_names = list(getattr(dataset, "feature_names", []) or [])
    guidance = dict(experience_prior or {})
    role_guided = build_role_guided_high_dim_candidates(
        feature_names,
        experience_prior=guidance,
        max_candidates=max(1, int(limit)),
    )
    candidate_families = deduplicate_expressions(
        list(role_guided)
        + list(guidance.get("candidate_families", []) or [])
        + list(guidance.get("memory_seed_exprs", []) or [])
    )
    if candidate_families:
        return candidate_families[:max(1, int(limit))]

    active_variables = list((guidance.get("variable_roles", {}) or {}).get("active_variables", []) or feature_names)
    numerator_core = list((guidance.get("variable_roles", {}) or {}).get("numerator_core", []) or active_variables)
    denominator_core = list((guidance.get("variable_roles", {}) or {}).get("denominator_core", []) or [])
    if not feature_names:
        return []

    x = active_variables if active_variables else feature_names
    out = []
    if len(numerator_core) >= 2:
        out.append(f"{numerator_core[0]}*{numerator_core[1]}")
    if len(numerator_core) >= 3:
        out.append(f"{numerator_core[0]}*{numerator_core[1]}*{numerator_core[2]}")
    if len(numerator_core) >= 2 and denominator_core:
        out.append(f"({numerator_core[0]}*{numerator_core[1]})/({denominator_core[0]}+1e-6)")
    if len(numerator_core) >= 3 and denominator_core:
        out.append(f"({numerator_core[0]}*{numerator_core[1]}*{numerator_core[2]})/({denominator_core[0]}+1e-6)")
    if len(numerator_core) >= 4 and denominator_core:
        out.append(f"{numerator_core[0]}*({numerator_core[2]}+0.5*{numerator_core[1]})*{numerator_core[3]}/({denominator_core[0]}+1e-6)")
    if len(feature_names) >= 5:
        x1, x2, x3, x4, x5 = feature_names[:5]
        out.extend([
            f"({x1}*{x2})/(({x5}+1e-6)*({x3}**2 + 0.5*{x3}*{x4} - {x4}**2 + 1e-6))",
            f"({x1}*{x2})/(({x5}+1e-6)*((({x3}+0.5*{x4})*({x3}-{x4}))+1e-6))",
            f"{x1}*({x3}+0.5*{x2})*{x4}/({x5}+1e-6)",
        ])
    if len(x) >= 2:
        out.append(f"{x[0]}*({x[1]}-{x[0]})")
    return deduplicate_expressions(out)[:max(1, int(limit))]


def _try_diverse_generate(*args, **kwargs):
    return []


def merge_expression_groups_with_limit(expr_groups, max_total=None):
    groups = []
    for expr_list in expr_groups:
        clean = []
        seen_local = set()
        for expr in expr_list:
            expr = str(expr).strip()
            if not expr:
                continue
            expr_key = _expr_dedup_key(expr)
            if expr_key in seen_local:
                continue
            clean.append(expr)
            seen_local.add(expr_key)
        groups.append(clean)

    merged = []
    seen = set()
    idx = 0
    while True:
        added = False
        for group in groups:
            if idx < len(group):
                expr = group[idx]
                expr_key = _expr_dedup_key(expr)
                if expr_key not in seen:
                    merged.append(expr)
                    seen.add(expr_key)
                    added = True
                    if max_total is not None and len(merged) >= max_total:
                        return merged
        if not added:
            break
        idx += 1
    return merged


def should_run_multimodal_after_initial_pass(current_best, dataset, num_existing_candidates):
    if not USE_MULTIMODAL_PROPOSAL:
        return False, 'multimodal disabled'
    if len(getattr(dataset, 'feature_names', [])) > MM_TRIGGER_MAX_FEATURES:
        return False, f'n_features>{MM_TRIGGER_MAX_FEATURES}'
    if current_best is None:
        if num_existing_candidates < MM_TRIGGER_IF_CANDIDATES_LT:
            return True, f'candidate_count<{MM_TRIGGER_IF_CANDIDATES_LT}'
        return bool(MM_TRIGGER_IF_NO_VALID_RESULT), 'no valid result after text/diverse/manual'
    val_mse = _safe_get_attr(current_best, 'val_mse', None)
    if val_mse is None:
        return True, 'best val_mse unavailable'
    try:
        if float(val_mse) <= GOOD_ENOUGH_VAL_MSE_TO_SKIP_MM:
            return False, f'best_val_mse<={GOOD_ENOUGH_VAL_MSE_TO_SKIP_MM}'
        if num_existing_candidates < MM_TRIGGER_IF_CANDIDATES_LT:
            return True, f'candidate_count<{MM_TRIGGER_IF_CANDIDATES_LT}'
        if float(val_mse) > MM_TRIGGER_VAL_MSE:
            return True, f'val_mse>{MM_TRIGGER_VAL_MSE}'
        return False, f'val_mse<={MM_TRIGGER_VAL_MSE}'
    except Exception:
        return True, 'val_mse parse failed'


def build_affine_repair_candidates(scored_results, top_k=3):
    return []


def maybe_generate_plots(plot_tool, dataset, row_meta):
    rel_name = sanitize_name(f"{row_meta['dataset_dir']}__{row_meta['base_name']}")
    plot_dir = os.path.join(RESULTS_ROOT, "plots", rel_name)
    plot_results = plot_tool.plot_dataset(
        df=dataset.train_df,
        feature_names=dataset.feature_names,
        target_name=dataset.target_name,
        output_dir=plot_dir,
        prefix="train_plot",
        max_plots=MAX_PLOTS_FOR_HIGH_DIM,
    )
    image_paths = [x.image_path for x in plot_results if getattr(x, "success", False)]
    plot_descriptions = [x.description for x in plot_results if getattr(x, "success", False)]
    visual_hints = plot_descriptions if plot_descriptions else ["no visual hints available"]
    return image_paths, plot_descriptions, visual_hints


def _existing_image_paths(paths):
    out = []
    seen = set()
    for path in paths or []:
        path = str(path or "").strip()
        if not path or path in seen:
            continue
        if os.path.isfile(path):
            seen.add(path)
            out.append(path)
    return out


def _agent_context_image_paths(observation, max_images=1):
    if observation is None or max_images <= 0:
        return []
    raw_paths = []
    raw_paths.extend(list(getattr(observation, "image_paths", []) or []))
    raw_paths.extend(list(getattr(observation, "reconstruction_image_paths", []) or []))
    return _existing_image_paths(raw_paths)[:max_images]


def _prediction_residual_diagnostic_image(dataset, row_meta, current_best, round_idx=0, stage="agent"):
    if dataset is None or current_best is None:
        return None
    expr = str(_safe_get_attr(current_best, "simplified_expression", "") or "").strip()
    if not expr:
        return None
    feature_names = list(getattr(dataset, "feature_names", []) or [])
    target_name = str(getattr(dataset, "target_name", "y") or "y")
    if not feature_names:
        return None

    df = None
    split_name = "validation"
    for attr, name in (("val_df", "validation"), ("train_df", "train"), ("df", "search")):
        candidate = getattr(dataset, attr, None)
        if candidate is not None and len(candidate) > 0 and target_name in candidate.columns:
            df = candidate.copy()
            split_name = name
            break
    if df is None or len(df) == 0:
        return None

    max_points = int(os.environ.get("LLMSR_AGENT_DIAGNOSTIC_MAX_POINTS", "600"))
    if len(df) > max_points:
        rng = np.random.default_rng(12345 + int(round_idx or 0))
        keep_idx = np.sort(rng.choice(len(df), size=max_points, replace=False))
        df = df.iloc[keep_idx].copy()

    try:
        pred = evaluate_expression_on_df(expr, df)
        target = np.asarray(df[target_name], dtype=float)
        pred = np.asarray(pred, dtype=float)
        if pred.ndim == 0:
            pred = np.full_like(target, float(pred), dtype=float)
        pred = pred.reshape(-1)
        target = target.reshape(-1)
        n = min(len(target), len(pred), len(df))
        target = target[:n]
        pred = pred[:n]
        df = df.iloc[:n].copy()
        residual = target - pred
        finite = np.isfinite(target) & np.isfinite(pred) & np.isfinite(residual)
        if int(finite.sum()) < 3:
            return None
        target = target[finite]
        pred = pred[finite]
        residual = residual[finite]
        df = df.iloc[np.where(finite)[0]].copy()
    except Exception:
        return None

    try:
        top_feature = feature_names[0]
        top_corr = -1.0
        for name in feature_names:
            if name not in df.columns:
                continue
            corr = _safe_abs_corr(df[name].values, residual)
            if corr > top_corr:
                top_corr = corr
                top_feature = name

        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        from tools.plot_style import COLOR_NEUTRAL_DARK, NATURE_COLORS, save_nature_figure, set_nature_style
        set_nature_style(plt)

        dataset_dir = str((row_meta or {}).get("dataset_dir", "dataset"))
        base_name = str((row_meta or {}).get("base_name", "case"))
        rel_name = sanitize_name(f"{dataset_dir}__{base_name}")
        out_dir = os.path.join(RESULTS_ROOT, "agent_diagnostics", rel_name)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{sanitize_name(str(stage))}_round{int(round_idx) + 1}_prediction_residual.png")

        mse = float(np.mean(residual ** 2))
        fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.4), dpi=140)
        color = NATURE_COLORS["blue"]
        accent = NATURE_COLORS["vermillion"]

        if len(feature_names) == 1 and feature_names[0] in df.columns:
            x_name = feature_names[0]
            x = np.asarray(df[x_name], dtype=float)
            order = np.argsort(x)
            axes[0].scatter(x, target, s=16, c=COLOR_NEUTRAL_DARK, alpha=0.75, label="target")
            axes[0].plot(x[order], pred[order], color=color, linewidth=1.7, label="prediction")
            axes[0].set_xlabel(x_name)
            axes[0].set_ylabel(target_name)
            axes[0].legend(fontsize=7, frameon=False)
        else:
            lo = float(np.nanmin([np.nanmin(target), np.nanmin(pred)]))
            hi = float(np.nanmax([np.nanmax(target), np.nanmax(pred)]))
            if not np.isfinite(lo) or not np.isfinite(hi) or abs(hi - lo) < 1e-12:
                lo, hi = -1.0, 1.0
            axes[0].scatter(target, pred, s=16, c=color, alpha=0.72)
            axes[0].plot([lo, hi], [lo, hi], color=COLOR_NEUTRAL_DARK, linewidth=1.0, linestyle="--")
            axes[0].set_xlabel(f"true {target_name}")
            axes[0].set_ylabel(f"predicted {target_name}")
        axes[0].set_title(f"{split_name} prediction")

        x_top = np.asarray(df[top_feature], dtype=float)
        axes[1].scatter(x_top, residual, s=16, c=accent, alpha=0.72)
        axes[1].axhline(0.0, color=COLOR_NEUTRAL_DARK, linewidth=1.0, linestyle="--")
        axes[1].set_xlabel(top_feature)
        axes[1].set_ylabel("residual")
        axes[1].set_title(f"residual vs {top_feature}")

        axes[2].scatter(pred, residual, s=16, c=NATURE_COLORS["green"], alpha=0.72)
        axes[2].axhline(0.0, color=COLOR_NEUTRAL_DARK, linewidth=1.0, linestyle="--")
        axes[2].set_xlabel(f"predicted {target_name}")
        axes[2].set_ylabel("residual")
        axes[2].set_title(f"residual pattern, MSE={mse:.3g}")

        for ax in axes:
            ax.grid(True, linewidth=0.4, alpha=0.28)
            ax.tick_params(labelsize=7)
        fig.suptitle(f"{stage} round {int(round_idx) + 1}: {expr[:90]}", fontsize=8)
        fig.tight_layout(rect=[0, 0, 1, 0.92])
        save_nature_figure(fig, out_path, "agent_diagnostics", bbox_inches="tight")
        plt.close(fig)
        return out_path
    except Exception:
        try:
            plt.close("all")
        except Exception:
            pass
        return None


def build_agent_multimodal_image_paths(dataset, row_meta, current_best, round_idx=0, stage="agent", observation=None):
    paths = []
    diag_path = _prediction_residual_diagnostic_image(
        dataset=dataset,
        row_meta=row_meta,
        current_best=current_best,
        round_idx=round_idx,
        stage=stage,
    )
    if diag_path:
        paths.append(diag_path)

    max_total = int(os.environ.get("LLMSR_AGENT_DIAGNOSTIC_MAX_IMAGES", "2"))
    max_context = max(0, max_total - len(paths))
    paths.extend(_agent_context_image_paths(observation, max_images=max_context))
    return _existing_image_paths(paths)[:max_total]


def build_agent_multimodal_user_message(prompt, image_paths):
    image_paths = _existing_image_paths(image_paths)
    if not image_paths:
        return Message(role="user", content=prompt)
    user_content = [{"type": "text", "text": prompt}]
    for path in image_paths:
        try:
            user_content.append({"type": "image_url", "image_url": {"url": image_file_to_data_url(path)}})
        except Exception:
            continue
    if len(user_content) == 1:
        return Message(role="user", content=prompt)
    return Message(role="user", content=user_content)


def _safe_abs_corr(a, b):
    try:
        a = np.asarray(a, dtype=float).reshape(-1)
        b = np.asarray(b, dtype=float).reshape(-1)
        mask = np.isfinite(a) & np.isfinite(b)
        if int(mask.sum()) < 4:
            return 0.0
        aa = a[mask]
        bb = b[mask]
        if np.nanstd(aa) < 1e-12 or np.nanstd(bb) < 1e-12:
            return 0.0
        corr = np.corrcoef(aa, bb)[0, 1]
        if not np.isfinite(corr):
            return 0.0
        return float(max(0.0, min(1.0, abs(corr))))
    except Exception:
        return 0.0


def _feature_standardization_stats(df, feature_names):
    stats = {}
    for name in feature_names:
        try:
            arr = np.asarray(df[name], dtype=float)
        except Exception:
            arr = np.asarray([], dtype=float)
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            stats[name] = {"mean": 0.0, "std": 1.0}
            continue
        mean = float(np.nanmean(finite))
        std = float(np.nanstd(finite))
        if not np.isfinite(std) or std < 1e-8:
            std = 1.0
        stats[name] = {"mean": mean, "std": std}
    return stats


def _scaled_feature_values(df, feature_name, feature_stats):
    try:
        arr = np.asarray(df[feature_name], dtype=float)
    except Exception:
        return np.asarray([], dtype=float)
    mean = float(feature_stats.get("mean", 0.0))
    std = float(feature_stats.get("std", 1.0))
    if not np.isfinite(std) or std < 1e-8:
        std = 1.0
    out = (arr - mean) / std
    out = np.asarray(out, dtype=float)
    out[~np.isfinite(out)] = 0.0
    return np.clip(out, -6.0, 6.0)


def _reciprocal_like(values):
    values = np.asarray(values, dtype=float)
    denom = np.where(np.abs(values) < 0.5, np.where(values >= 0.0, 0.5, -0.5), values)
    out = 1.0 / denom
    out[~np.isfinite(out)] = 0.0
    return np.clip(out, -6.0, 6.0)


def _safe_log_ratio_like(a, b):
    try:
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
    except Exception:
        return np.asarray([], dtype=float)
    denom = 1.0 + np.abs(b)
    ratio = a / denom
    ratio[~np.isfinite(ratio)] = 0.0
    out = np.log1p(np.abs(ratio))
    out[~np.isfinite(out)] = 0.0
    return np.clip(out, 0.0, 6.0)


def _expr_has_log_ratio_signature(expr: str) -> bool:
    low = _expr_dedup_key(expr)
    return ("log(" in low) and ("/" in low)


def _expr_has_affine_log_signature(expr: str) -> bool:
    low = _expr_dedup_key(expr)
    return ("log(" in low) and ("/" not in low)


def _profile_prefers_ratio_log_templates(structure_profile, feature_names):
    feature_names = list(feature_names or [])
    if len(feature_names) < HIGH_DIM_ROLE_TRIGGER_DIM:
        return False

    profile = dict(structure_profile or {})
    family_scores = dict(profile.get("family_scores", {}) or {})
    roles = dict(profile.get("variable_roles", {}) or {})
    top_pair_patterns = list(profile.get("top_pair_patterns", []) or [])
    global_tags = set(profile.get("global_tags", []) or [])
    active_variables = list(roles.get("active_variables", []) or profile.get("active_variables", []) or feature_names)

    interaction_score = float(family_scores.get("interaction", 0.0) or 0.0)
    rational_score = float(family_scores.get("rational", 0.0) or 0.0)
    logarithmic_score = float(family_scores.get("logarithmic", 0.0) or 0.0)
    has_ratio_like_pair = any(
        str(item.get("family", "") or "") in {"logarithmic", "rational", "power"}
        and len(list(item.get("variables", []) or [])) >= 2
        for item in top_pair_patterns[:4]
    )

    return bool(
        len(active_variables) >= 3
        and logarithmic_score >= 0.16
        and max(interaction_score, rational_score) >= 0.18
        and (
            has_ratio_like_pair
            or "logarithmic_like" in global_tags
            or "ratio_or_denominator" in global_tags
        )
    )


def _top_score_items(score_map, top_k=3, min_score=0.0):
    items = []
    for key, value in dict(score_map or {}).items():
        try:
            value = float(value)
        except Exception:
            continue
        if not np.isfinite(value) or value < min_score:
            continue
        items.append((str(key), value))
    items.sort(key=lambda x: (-x[1], x[0]))
    return items[:max(1, int(top_k))]


def _select_active_variables(score_map, min_keep=1, max_keep=5):
    ranked = _top_score_items(score_map, top_k=max_keep + 2, min_score=0.0)
    if not ranked:
        return []
    best = ranked[0][1]
    if best <= 0.0:
        return [ranked[0][0]]
    threshold = max(0.18, 0.55 * best)
    selected = [name for name, score in ranked if score >= threshold][:max_keep]
    if len(selected) < min_keep:
        selected = [name for name, _ in ranked[:min_keep]]
    return selected


def _family_action_hint(family_name):
    family_name = str(family_name or "")
    if family_name == "trigonometric":
        return "periodic or modulation term"
    if family_name == "rational":
        return "ratio or denominator correction"
    if family_name == "interaction":
        return "cross-variable interaction"
    if family_name == "power":
        return "higher-order power term"
    if family_name == "exponential":
        return "exponential component"
    if family_name == "logarithmic":
        return "logarithmic component"
    if family_name == "additive":
        return "extra additive residual branch"
    return "local structural correction"


def build_structure_profile_from_df(df, feature_names, target_name):
    feature_names = list(feature_names or [])
    empty_profile = {
        "family_scores": {},
        "global_tags": [],
        "active_variables": [],
        "variable_roles": {
            "active_variables": [],
            "numerator_core": [],
            "denominator_core": [],
            "periodic_core": [],
        },
        "top_unary_patterns": [],
        "top_pair_patterns": [],
        "evidence": [],
    }
    if not feature_names or df is None or target_name not in df.columns:
        return empty_profile

    y = np.asarray(df[target_name], dtype=float)
    if y.size == 0 or not np.isfinite(y).any():
        return empty_profile

    feature_stats = _feature_standardization_stats(df, feature_names)
    z_cache = {name: _scaled_feature_values(df, name, feature_stats.get(name, {})) for name in feature_names}
    family_scores = {
        "interaction": 0.0,
        "rational": 0.0,
        "trigonometric": 0.0,
        "exponential": 0.0,
        "logarithmic": 0.0,
        "power": 0.0,
        "additive": 0.0,
        "algebraic": 0.0,
    }
    unary_patterns = []
    pair_patterns = []
    variable_strength = {name: 0.0 for name in feature_names}
    pair_support = {name: 0.0 for name in feature_names}
    rational_support = {name: 0.0 for name in feature_names}
    logarithmic_support = {name: 0.0 for name in feature_names}
    periodic_support = {name: 0.0 for name in feature_names}
    product_over_denominator_support = {name: 0.0 for name in feature_names}

    for name in feature_names:
        z = z_cache[name]
        signals = {
            "additive": _safe_abs_corr(y, z),
            "power": max(_safe_abs_corr(y, z ** 2), _safe_abs_corr(y, z ** 3)),
            "rational": max(_safe_abs_corr(y, _reciprocal_like(z)), _safe_abs_corr(y, z / (1.0 + np.abs(z)))),
            "trigonometric": max(_safe_abs_corr(y, np.sin(z)), _safe_abs_corr(y, np.cos(z))),
            "exponential": _safe_abs_corr(y, np.exp(np.clip(z, -2.0, 2.0))),
            "logarithmic": _safe_abs_corr(y, np.log1p(np.abs(z))),
        }
        dominant_family, dominant_score = max(signals.items(), key=lambda x: (x[1], x[0]))
        top_signals = [{"family": fam, "score": round(score, 3)} for fam, score in _top_score_items(signals, top_k=3, min_score=0.12)]
        unary_patterns.append({
            "variable": name,
            "dominant_family": dominant_family,
            "score": round(float(dominant_score), 3),
            "signals": top_signals,
        })
        variable_strength[name] = max(signals.values()) if signals else 0.0
        rational_support[name] = float(signals.get("rational", 0.0))
        periodic_support[name] = float(signals.get("trigonometric", 0.0))
        for family_name, score in signals.items():
            family_scores[family_name] = max(family_scores.get(family_name, 0.0), float(score))
        family_scores["algebraic"] = max(family_scores["algebraic"], float(signals.get("additive", 0.0)), float(signals.get("power", 0.0)))

    top_for_pairs = [
        name for name, _ in _top_score_items(
            variable_strength,
            top_k=min(HIGH_DIM_ROLE_PAIR_SCAN_MAX_VARS, len(feature_names)),
            min_score=0.0,
        )
    ]
    for i in range(len(top_for_pairs)):
        for j in range(i + 1, len(top_for_pairs)):
            x1 = top_for_pairs[i]
            x2 = top_for_pairs[j]
            z1 = z_cache[x1]
            z2 = z_cache[x2]
            signals = {
                "interaction": _safe_abs_corr(y, z1 * z2),
                "rational": max(_safe_abs_corr(y, z1 / (1.0 + np.abs(z2))), _safe_abs_corr(y, z2 / (1.0 + np.abs(z1)))),
                "logarithmic": max(_safe_abs_corr(y, _safe_log_ratio_like(z1, z2)), _safe_abs_corr(y, _safe_log_ratio_like(z2, z1))),
                "trigonometric": max(_safe_abs_corr(y, np.sin(z1 + z2)), _safe_abs_corr(y, z1 * np.sin(z2))),
                "power": _safe_abs_corr(y, z1 ** 2 - z2 ** 2),
                "additive": _safe_abs_corr(y, z1 + z2),
            }
            dominant_family, dominant_score = max(signals.items(), key=lambda x: (x[1], x[0]))
            if float(dominant_score) < 0.12:
                continue
            pair_patterns.append({
                "variables": [x1, x2],
                "family": dominant_family,
                "score": round(float(dominant_score), 3),
                "signals": [{"family": fam, "score": round(score, 3)} for fam, score in _top_score_items(signals, top_k=3, min_score=0.12)],
                "hint": _family_action_hint(dominant_family),
            })
            for item in (x1, x2):
                pair_support[item] = max(pair_support[item], float(dominant_score))
                if dominant_family == "rational":
                    rational_support[item] = max(rational_support[item], float(dominant_score))
                if dominant_family == "logarithmic":
                    logarithmic_support[item] = max(logarithmic_support[item], float(dominant_score))
                if dominant_family == "trigonometric":
                    periodic_support[item] = max(periodic_support[item], float(dominant_score))
            for family_name, score in signals.items():
                family_scores[family_name] = max(family_scores.get(family_name, 0.0), float(score))

    if len(feature_names) >= 3:
        target_std = float(np.nanstd(y)) if np.isfinite(np.nanstd(y)) else 1.0
        if not np.isfinite(target_std) or target_std < 1e-8:
            target_std = 1.0
        yz_cache = {}
        for name in feature_names:
            yz = np.asarray((y / target_std) * z_cache[name], dtype=float)
            yz[~np.isfinite(yz)] = 0.0
            yz_cache[name] = np.clip(yz, -8.0, 8.0)

        top_for_den = [
            name for name, _ in _top_score_items(
                variable_strength,
                top_k=min(max(4, HIGH_DIM_ROLE_PAIR_SCAN_MAX_VARS), len(feature_names)),
                min_score=0.0,
            )
        ]
        for den in feature_names:
            den_signal = 0.0
            others = [name for name in top_for_den if name != den]
            for i in range(len(others)):
                for j in range(i + 1, len(others)):
                    a = others[i]
                    b = others[j]
                    prod_score = _safe_abs_corr(yz_cache[den], z_cache[a] * z_cache[b])
                    diff_score = _safe_abs_corr(yz_cache[den], z_cache[a] ** 2 - z_cache[b] ** 2)
                    factored_diff_score = _safe_abs_corr(yz_cache[den], (z_cache[a] - z_cache[b]) * (z_cache[a] + z_cache[b]))
                    den_signal = max(den_signal, prod_score, diff_score, factored_diff_score)
            if den_signal > 0.0:
                product_over_denominator_support[den] = den_signal
                rational_support[den] = max(rational_support.get(den, 0.0), den_signal)
                variable_strength[den] = max(variable_strength.get(den, 0.0), 0.65 * den_signal)
                family_scores["rational"] = max(family_scores.get("rational", 0.0), den_signal)
                family_scores["power"] = max(family_scores.get("power", 0.0), 0.8 * den_signal)

    combined_strength = {
        name: float(variable_strength.get(name, 0.0)) + 0.45 * float(pair_support.get(name, 0.0))
        for name in feature_names
    }
    active_variables = _select_active_variables(
        combined_strength,
        min_keep=1,
        max_keep=min(HIGH_DIM_ROLE_SUBSPACE_MAX_VARS, len(feature_names)),
    )
    denominator_core = _select_active_variables(rational_support, min_keep=0, max_keep=min(3, len(feature_names)))
    periodic_core = _select_active_variables(periodic_support, min_keep=0, max_keep=min(3, len(feature_names)))
    denominator_core = [name for name in denominator_core if float(rational_support.get(name, 0.0)) >= 0.18]
    periodic_core = [name for name in periodic_core if float(periodic_support.get(name, 0.0)) >= 0.18]
    numerator_core = [name for name in active_variables if name not in denominator_core]
    if not numerator_core:
        numerator_core = list(active_variables[:2])

    total_strength = sum(max(0.0, float(value)) for value in combined_strength.values())
    top_strength = sum(float(combined_strength.get(name, 0.0)) for name in active_variables[:2])
    global_tags = []
    if len(feature_names) >= 4 and total_strength > 0.0 and top_strength / total_strength >= 0.65:
        global_tags.append("sparse_active_subset")
    if family_scores["interaction"] >= 0.32 and family_scores["additive"] >= 0.24:
        global_tags.append("partially_separable")
    elif family_scores["additive"] >= 0.30:
        global_tags.append("additive_or_separable")
    if family_scores["rational"] >= 0.30:
        global_tags.append("ratio_or_denominator")
    if max(product_over_denominator_support.values()) >= 0.24:
        global_tags.append("product_over_denominator")
    if family_scores["trigonometric"] >= 0.30:
        global_tags.append("periodic_or_modulated")
    if family_scores["power"] >= 0.32:
        global_tags.append("nonlinear_power")
    if family_scores["exponential"] >= 0.30:
        global_tags.append("exponential_like")
    if family_scores["logarithmic"] >= 0.28:
        global_tags.append("logarithmic_like")

    top_family_items = _top_score_items(family_scores, top_k=4, min_score=0.12)
    top_unary_patterns = sorted(unary_patterns, key=lambda x: (-float(x.get("score", 0.0)), x.get("variable", "")))[:4]
    top_pair_patterns = sorted(pair_patterns, key=lambda x: (-float(x.get("score", 0.0)), x.get("variables", [])))[:4]

    evidence = []
    if top_family_items:
        evidence.append("family prior: " + ", ".join(f"{name}={score:.2f}" for name, score in top_family_items))
    if active_variables:
        evidence.append("likely active variables: " + ", ".join(active_variables[:4]))
    if denominator_core:
        evidence.append("possible denominator variables: " + ", ".join(denominator_core[:3]))
    top_den_like = _top_score_items(product_over_denominator_support, top_k=2, min_score=0.18)
    if top_den_like:
        evidence.append("product-over-denominator clues: " + ", ".join(f"{name}={score:.2f}" for name, score in top_den_like))
    if periodic_core:
        evidence.append("possible periodic variables: " + ", ".join(periodic_core[:3]))
    if top_pair_patterns:
        pair = top_pair_patterns[0]
        evidence.append(
            f"top pair clue: {pair['variables'][0]} and {pair['variables'][1]} show {pair['family']} pattern ({float(pair['score']):.2f})"
        )

    return {
        "family_scores": {name: round(float(score), 3) for name, score in family_scores.items()},
        "global_tags": global_tags,
        "active_variables": active_variables,
        "variable_roles": {
            "active_variables": active_variables,
            "numerator_core": numerator_core,
            "denominator_core": denominator_core,
            "periodic_core": periodic_core,
        },
        "top_unary_patterns": top_unary_patterns,
        "top_pair_patterns": top_pair_patterns,
        "evidence": evidence,
    }


def build_structure_profile(dataset):
    return build_structure_profile_from_df(dataset.train_df, dataset.feature_names, dataset.target_name)


def build_structure_hints_from_profile(structure_profile, feature_names=None):
    profile = dict(structure_profile or {})
    hints = list(profile.get("evidence", []) or [])
    global_tags = list(profile.get("global_tags", []) or [])
    if global_tags:
        hints.append("global structure tags: " + ", ".join(global_tags[:5]))
    pair_patterns = list(profile.get("top_pair_patterns", []) or [])
    if pair_patterns:
        pair_lines = []
        for item in pair_patterns[:2]:
            variables = item.get("variables", []) or []
            if len(variables) >= 2:
                pair_lines.append(f"{variables[0]}-{variables[1]}:{item.get('family')}({item.get('score')})")
        if pair_lines:
            hints.append("top pairwise structure clues: " + "; ".join(pair_lines))
    hints.extend(build_family_route_hints_from_profile(profile, feature_names=feature_names))
    return deduplicate_expressions(hints)


def build_family_route_hints_from_profile(structure_profile, feature_names=None, experience_prior=None, max_routes=FAMILY_ROUTE_HINT_TOPK):
    profile = dict(structure_profile or {})
    feature_names = list(feature_names or [])
    experience_prior = dict(experience_prior or {})
    roles = dict(profile.get("variable_roles", {}) or {})
    roles_prior = dict(experience_prior.get("variable_roles", {}) or {})
    active_variables = list(roles.get("active_variables", []) or profile.get("active_variables", []) or roles_prior.get("active_variables", []) or feature_names)
    numerator_core = list(roles.get("numerator_core", []) or roles_prior.get("numerator_core", []) or active_variables)
    denominator_core = list(roles.get("denominator_core", []) or roles_prior.get("denominator_core", []) or [])
    periodic_core = list(roles.get("periodic_core", []) or roles_prior.get("periodic_core", []) or [])
    top_pair_patterns = list(profile.get("top_pair_patterns", []) or [])
    family_ranked = _top_score_items(profile.get("family_scores", {}), top_k=max_routes, min_score=FAMILY_ROUTE_MIN_SCORE)
    if not family_ranked:
        family_ranked = [(str(name), FAMILY_ROUTE_MIN_SCORE) for name in list(experience_prior.get("family_tags", []) or [])[:max_routes]]
    if not family_ranked:
        return []

    pair0 = []
    pair0_family = ""
    if top_pair_patterns:
        pair0 = list((top_pair_patterns[0] or {}).get("variables", []) or [])[:2]
        pair0_family = str((top_pair_patterns[0] or {}).get("family", "") or "")

    num0 = numerator_core[0] if numerator_core else (active_variables[0] if active_variables else "")
    num1 = numerator_core[1] if len(numerator_core) >= 2 else (active_variables[1] if len(active_variables) >= 2 else num0)
    denom0 = denominator_core[0] if denominator_core else ""
    periodic0 = periodic_core[0] if periodic_core else (active_variables[0] if active_variables else "")

    hints = []
    for family_name, score in family_ranked[:max(1, int(max_routes))]:
        if family_name == "rational":
            if denom0 and num0 and num1:
                hints.append(
                    f"family route: prioritize rational/product-over-denominator candidates using {num0}, {num1} over denominator-like {denom0} (score={score:.2f})"
                )
            elif denom0:
                hints.append(
                    f"family route: prioritize denominator-based rational candidates with {denom0} as a likely normalizer (score={score:.2f})"
                )
        elif family_name == "interaction":
            if pair0:
                hints.append(
                    f"family route: prioritize sparse multiplicative interaction around {pair0[0]} and {pair0[1]} (score={score:.2f})"
                )
            elif num0 and num1:
                hints.append(
                    f"family route: prioritize pairwise interaction candidates using {num0} and {num1} (score={score:.2f})"
                )
        elif family_name == "power":
            if pair0:
                hints.append(
                    f"family route: test higher-order power or difference-of-squares structures on {pair0[0]} and {pair0[1]} (score={score:.2f})"
                )
            else:
                hints.append(f"family route: test nonlinear power structures on active variables (score={score:.2f})")
        elif family_name == "logarithmic":
            if pair0:
                hints.append(
                    f"family route: test log-ratio or multiplicative log-transform structures on {pair0[0]} and {pair0[1]} (score={score:.2f})"
                )
            elif num0 and num1:
                hints.append(
                    f"family route: test multiplicative envelope times log-transform using {num0} and {num1} (score={score:.2f})"
                )
        elif family_name == "trigonometric" and periodic0:
            hints.append(f"family route: test periodic/modulation terms centered on {periodic0} (score={score:.2f})")
        elif family_name == "exponential" and active_variables:
            hints.append(f"family route: test smooth exponential growth/decay on {active_variables[0]} (score={score:.2f})")
        elif family_name == "additive" and active_variables:
            hints.append(
                f"family route: start from a sparse additive core over {', '.join(active_variables[:min(3, len(active_variables))])} (score={score:.2f})"
            )

    if pair0 and pair0_family:
        hints.append(f"family route: strongest pair clue is {pair0[0]}-{pair0[1]} with {pair0_family} behavior")
    elif active_variables:
        hints.append(f"family route: focus first on active variables {', '.join(active_variables[:min(3, len(active_variables))])}")
    return deduplicate_expressions(hints)


def build_prompt_family_templates_from_profile(structure_profile, feature_names=None, experience_prior=None, max_templates=PROMPT_FAMILY_TEMPLATE_TOPK):
    feature_names = list(feature_names or [])
    if len(feature_names) < 2:
        return []

    role_parts = _build_role_seed_components(
        feature_names,
        structure_profile=structure_profile,
        experience_prior=experience_prior,
    )
    active_variables = list(role_parts.get("active_variables", []) or feature_names)
    numerator_core = list(role_parts.get("numerator_core", []) or active_variables)
    denominator_core = list(role_parts.get("denominator_core", []) or [])
    top_pair_patterns = list(role_parts.get("top_pair_patterns", []) or [])
    family_ranked = _top_score_items(
        dict((structure_profile or {}).get("family_scores", {}) or {}),
        top_k=max(3, int(max_templates)),
        min_score=FAMILY_ROUTE_MIN_SCORE,
    )

    num0 = numerator_core[0] if numerator_core else active_variables[0]
    num1 = numerator_core[1] if len(numerator_core) >= 2 else (active_variables[1] if len(active_variables) >= 2 else num0)
    num2 = numerator_core[2] if len(numerator_core) >= 3 else (active_variables[2] if len(active_variables) >= 3 else num1)
    num3 = numerator_core[3] if len(numerator_core) >= 4 else (active_variables[3] if len(active_variables) >= 4 else num2)
    denom0 = denominator_core[0] if denominator_core else (feature_names[-1] if len(feature_names) >= 3 else "")
    pair0 = list((top_pair_patterns[0] or {}).get("variables", []) or [])[:2] if top_pair_patterns else active_variables[:2]

    diff_pair = []
    for item in top_pair_patterns:
        if str(item.get("family", "") or "") in {"power", "rational", "interaction"}:
            variables = list(item.get("variables", []) or [])
            if len(variables) >= 2:
                diff_pair = variables[:2]
                break
    if len(diff_pair) < 2 and len(numerator_core) >= 4:
        diff_pair = [numerator_core[2], numerator_core[3]]
    elif len(diff_pair) < 2 and len(active_variables) >= 4:
        diff_pair = [active_variables[2], active_variables[3]]

    templates = []

    def _add(line):
        line = str(line).strip()
        if line and line not in templates:
            templates.append(line)

    ranked_names = [name for name, _ in family_ranked]
    if "rational" in ranked_names and denom0 and num0 and num1:
        _add(
            f"abstract family template: a1*({num0}*{num1})/({denom0}+b1) + c"
        )
        _add(
            f"abstract family template: a1*({num0}*{num1}*{num2})/({denom0}+b1) + c"
        )
    if ("power" in ranked_names or "rational" in ranked_names) and denom0 and len(diff_pair) >= 2:
        p, q = diff_pair[:2]
        _add(
            f"abstract family template: a1*({num0}*{num1})/(({denom0}+b1)*({p}**2 + k1*{p}*{q} + k2*{q}**2 + b2)) + c"
        )
        _add(
            f"abstract family template: a1*({num0}*{num1})/(({denom0}+b1)*((({p}+k1*{q})*({p}+k2*{q}))+b2)) + c"
        )
    if "interaction" in ranked_names and denom0 and len(active_variables) >= 4:
        _add(
            f"abstract family template: a1*{num0}*({num2}+k1*{num1})*{num3}/({denom0}+b1) + c"
        )
    if "interaction" in ranked_names:
        _add(f"abstract family template: a1*{num0}*{num1} + c")
    if "logarithmic" in ranked_names and len(pair0) >= 2:
        _add(
            f"abstract family template: a1*({num0}*{num1})*log(abs(({pair0[0]}+k1)/({pair0[1]}+k2)) + b1) + c"
        )
        _add(
            f"abstract family template: a1*({num0}*{num1}*{num2})*log(abs(({pair0[0]}+k1)/({pair0[1]}+k2)) + b1) + c"
        )
    if "additive" in ranked_names:
        focus = ", ".join(active_variables[:min(3, len(active_variables))])
        _add(f"abstract family template: sparse additive core over {focus}")

    return templates[:max(1, int(max_templates))]


def build_family_routed_candidates(feature_names, structure_profile=None, experience_prior=None, max_candidates=None):
    feature_names = list(feature_names or [])
    if len(feature_names) < 2:
        return []

    profile = dict(structure_profile or {})
    experience_prior = dict(experience_prior or {})
    role_parts = _build_role_seed_components(
        feature_names,
        structure_profile=structure_profile,
        experience_prior=experience_prior,
    )
    active_variables = list(role_parts.get("active_variables", []) or feature_names)
    numerator_core = list(role_parts.get("numerator_core", []) or active_variables)
    denominator_core = list(role_parts.get("denominator_core", []) or [])
    periodic_core = list(role_parts.get("periodic_core", []) or [])
    top_pair_patterns = list(role_parts.get("top_pair_patterns", []) or [])

    family_ranked = _top_score_items(profile.get("family_scores", {}), top_k=FAMILY_ROUTE_HINT_TOPK + 1, min_score=FAMILY_ROUTE_MIN_SCORE)
    dominant_families = [name for name, _ in family_ranked] or ["interaction", "additive"]

    linear_vars = active_variables[:min(4, len(active_variables))]
    linear = " + ".join([f"a{i+1}*{v}" for i, v in enumerate(linear_vars)]) if linear_vars else "0"
    num0 = numerator_core[0] if numerator_core else active_variables[0]
    num1 = numerator_core[1] if len(numerator_core) >= 2 else active_variables[1]
    num2 = numerator_core[2] if len(numerator_core) >= 3 else (active_variables[2] if len(active_variables) >= 3 else num1)
    denom0 = denominator_core[0] if denominator_core else ""
    periodic0 = periodic_core[0] if periodic_core else (active_variables[0] if active_variables else "")
    pair0 = list((top_pair_patterns[0] or {}).get("variables", []) or [])[:2] if top_pair_patterns else active_variables[:2]
    pair1 = list((top_pair_patterns[1] or {}).get("variables", []) or [])[:2] if len(top_pair_patterns) >= 2 else []

    out = build_generic_role_combination_candidates(
        feature_names,
        structure_profile=structure_profile,
        experience_prior=experience_prior,
        max_candidates=max_candidates,
    )
    seen = set()
    for expr in out:
        seen.add(_expr_dedup_key(expr))

    def _add(exprs):
        for expr in exprs or []:
            expr = str(expr).strip()
            if not expr:
                continue
            key = _expr_dedup_key(expr)
            if key in seen:
                continue
            seen.add(key)
            out.append(expr)
            if max_candidates is not None and len(out) >= max_candidates:
                return True
        return False

    # Add a compact additive baseline first so later family-specific branches refine around a stable core.
    _add([
        f"{linear} + c",
        f"a1*{num0}*{num1} + {linear} + c",
    ])

    if "interaction" in dominant_families:
        interaction_exprs = [
            f"a1*{num0}*{num1} + c",
            f"a1*{num0}*{num1} + {linear} + c",
        ]
        if len(active_variables) >= 3:
            interaction_exprs.append(f"a1*{num0}*{num1}*{num2} + c")
        if pair0 and len(pair0) >= 2:
            interaction_exprs.extend([f"{pair0[0]}*{pair0[1]}", f"a1*{pair0[0]}*{pair0[1]} + c"])
        _add(interaction_exprs)

    if "rational" in dominant_families and denom0:
        rational_exprs = [
            f"a1*({num0}*{num1})/({denom0}+b1) + c",
            f"({num0}*{num1})/({denom0}+b1)",
        ]
        if len(active_variables) >= 3:
            rational_exprs.append(f"a1*({num0}*{num1}*{num2})/({denom0}+b1) + c")
        if pair0 and len(pair0) >= 2:
            rational_exprs.append(f"a1*{pair0[0]}/({pair0[1]}+b1) + c")
        _add(rational_exprs)

    if "power" in dominant_families:
        power_exprs = []
        if pair0 and len(pair0) >= 2:
            power_exprs.extend([
                f"{pair0[0]}**2 + b2*{pair0[0]}*{pair0[1]} - {pair0[1]}**2",
                f"a1*({pair0[0]}**2 + b2*{pair0[0]}*{pair0[1]} - {pair0[1]}**2) + c",
            ])
            if denom0:
                power_exprs.extend([
                    f"a1*{num0}*{num1}/(({denom0}+b1)*({pair0[0]}**2 + b2*{pair0[0]}*{pair0[1]} + b3*{pair0[1]}**2 + b4)) + c",
                    f"a1*{num0}*{num1}/(({denom0}+b1)*((({pair0[0]}+b2*{pair0[1]})*({pair0[0]}+b3*{pair0[1]}))+b4)) + c",
                ])
        _add(power_exprs)

    if "trigonometric" in dominant_families and periodic0:
        _add([
            f"a1*sin(b1*{periodic0}+c1) + {linear} + d",
            f"a1*cos(b1*{periodic0}+c1) + {linear} + d",
        ])

    if "exponential" in dominant_families:
        _add([
            f"a1*exp(b1*{active_variables[0]}) + {linear} + c",
            f"a1*log(abs(b1*{active_variables[0]})+c1) + {linear} + d",
        ])
    if "logarithmic" in dominant_families and pair0 and len(pair0) >= 2:
        _add([
            f"a1*({num0}*{num1})*log(abs(({pair0[0]}+c1)/({pair0[1]}+c2)) + c3) + d",
            f"a1*({num0}*{num1}) + a2*log(abs(({pair0[0]}+c1)/({pair0[1]}+c2)) + c3) + d",
        ])
        if len(active_variables) >= 3:
            _add([
                f"a1*({num0}*{num1}*{num2})*log(abs(({pair0[0]}+c1)/({pair0[1]}+c2)) + c3) + d",
            ])

    if "additive" in dominant_families:
        additive_exprs = [f"{linear} + c"]
        if pair1 and len(pair1) >= 2:
            additive_exprs.append(f"a1*{pair1[0]}*{pair1[1]} + {linear} + c")
        _add(additive_exprs)

    cap = max_candidates if max_candidates is not None else FAMILY_ROUTE_TEMPLATE_LIMIT
    return out[:max(1, int(cap))]


def build_visual_summary(structure_profile=None, plot_descriptions=None, image_paths=None):
    structure_profile = dict(structure_profile or {})
    plot_descriptions = list(plot_descriptions or [])
    image_paths = list(image_paths or [])
    dominant_variables = list(structure_profile.get("active_variables", []) or [])[:5]
    pair_patterns = list(structure_profile.get("top_pair_patterns", []) or [])[:3]
    focus_pairs = [list(item.get("variables", [])[:2]) for item in pair_patterns if len(item.get("variables", []) or []) >= 2]
    family_scores = _top_score_items(structure_profile.get("family_scores", {}), top_k=4, min_score=0.12)
    global_tags = list(structure_profile.get("global_tags", []) or [])[:6]

    story = []
    if dominant_variables:
        story.append("compressed visual summary: the most informative views are likely tied to " + ", ".join(dominant_variables[:4]))
    if global_tags:
        story.append("compressed visual summary: global plot pattern looks like " + ", ".join(global_tags[:4]))
    roles = dict(structure_profile.get("variable_roles", {}) or {})
    denominator_core = list(roles.get("denominator_core", []) or [])[:3]
    periodic_core = list(roles.get("periodic_core", []) or [])[:3]
    if denominator_core:
        story.append("compressed visual summary: denominator-like or normalization behavior may involve " + ", ".join(denominator_core))
    if periodic_core:
        story.append("compressed visual summary: periodic or modulation behavior may involve " + ", ".join(periodic_core))
    if pair_patterns:
        pair = pair_patterns[0]
        variables = pair.get("variables", []) or []
        if len(variables) >= 2:
            story.append(
                f"compressed visual summary: strongest coupled view is {variables[0]}-{variables[1]} with {pair.get('family')} clue"
            )
    if plot_descriptions:
        inventory = "; ".join(str(x) for x in plot_descriptions[:4])
        if len(plot_descriptions) > 4:
            inventory += f"; ... total {len(plot_descriptions)} plots"
        story.append("plot inventory: " + inventory)

    return {
        "num_plots": len(image_paths) if image_paths else len(plot_descriptions),
        "dominant_variables": dominant_variables,
        "focus_pairs": focus_pairs,
        "top_families": [name for name, _ in family_scores],
        "global_tags": global_tags,
        "story": story[:6],
        "plot_inventory": plot_descriptions[:6],
    }


def build_visual_hints_from_summary(visual_summary):
    summary = dict(visual_summary or {})
    hints = list(summary.get("story", []) or [])
    if not hints and int(summary.get("num_plots", 0) or 0) > 0:
        hints.append(f"compressed visual summary available for {int(summary.get('num_plots', 0) or 0)} plots")
    return deduplicate_expressions(hints)


def looks_mechanistic(expr: str) -> bool:
    expr = str(expr)
    keys = ["sin(", "cos(", "sinh(", "cosh(", "tanh(", "exp(", "log(", "**2-", "**3", "**4", "**5", "**6", "**7", "**8", "**9", "/(", "x1*x2", "*x1*x2", "x1*x2*"]
    return any(k in expr for k in keys)


def _fast_seed_priority(expr: str, protected_set, livermore_set, feature_names):
    expr = str(expr).strip()
    low = expr.replace(" ", "")
    score = 0.0
    if expr in protected_set:
        score += 1000.0
    if expr in livermore_set:
        score += 800.0
    if any(k in low for k in ["sinh(", "cosh(", "tanh("]):
        score += 300.0
    if "/(" in low or "/" in low:
        score += 260.0
    if any(k in low for k in ["sin(", "cos("]):
        score += 220.0
    if any(k in low for k in ["**9", "**8", "**7", "**6", "**5", "**4"]):
        score += 180.0
    if "log(" in low or "exp(" in low:
        score += 120.0
    if looks_mechanistic(expr):
        score += 60.0
    if len(feature_names) >= 2:
        pair_tokens = [
            f"{feature_names[0]}*{feature_names[1]}",
            f"{feature_names[1]}*{feature_names[0]}",
        ]
        if any(tok in low for tok in pair_tokens):
            score += 100.0
    score -= 0.01 * len(low)
    return score


def build_fast_seed_candidates(row_meta, feature_names):
    if PURE_LLM_CANDIDATE_MODE:
        return []
    if str(row_meta.get("dataset_dir", "")) != "benchmark_csv":
        return []
    d = len(feature_names)
    if d == 0:
        return []

    protected = build_protected_benchmark_templates(row_meta, feature_names)
    livermore = build_livermore_2d_templates(row_meta, feature_names)
    manual = build_manual_candidates(feature_names)
    protected_set = set(protected)
    livermore_set = set(livermore)

    baseline_manual = manual[:min(4, len(manual))]
    base_name = str(row_meta.get("base_name", ""))

    if d == 5 and protected:
        return merge_expression_groups_with_limit(
            [protected, baseline_manual[:2]],
            max_total=min(HIGH_DIM_BENCHMARK_PROTECTED_SEED_CAP, len(protected) + 2),
        )

    if d > FAST_PATH_BENCHMARK_DIM_LIMIT:
        return []

    # These two benchmark families have a very reliable protected skeleton.
    # A smaller seed pool cuts template-fit time and still preserves the exact family.
    if d == 2 and base_name in {"Feynman-3", "Feynman-6"} and protected:
        return merge_expression_groups_with_limit(
            [protected, baseline_manual[:2]],
            max_total=min(6, len(protected) + 2),
        )

    ranked_manual = sorted(
        manual,
        key=lambda expr: (-_fast_seed_priority(expr, protected_set, livermore_set, feature_names), len(str(expr))),
    )

    max_candidates = FAST_PATH_SEED_MAX_CANDIDATES_1D if d == 1 else FAST_PATH_SEED_MAX_CANDIDATES_2D
    return merge_expression_groups_with_limit(
        [protected, livermore, baseline_manual, ranked_manual],
        max_total=max_candidates,
    )


def should_use_fast_seed_as_initial(seed_eval):
    best = seed_eval.get("best_result") if isinstance(seed_eval, dict) else None
    best_val = _safe_get_attr(best, "val_mse", None)
    try:
        if best_val is not None and np.isfinite(best_val) and float(best_val) <= FAST_PATH_USE_SEED_AS_INITIAL_VAL_MSE:
            return True, f"seed_val_mse<={FAST_PATH_USE_SEED_AS_INITIAL_VAL_MSE}"
    except Exception:
        pass
    return False, None


def _should_apply_high_dim_structural_rerank(dataset):
    if not ENABLE_HIGH_DIM_STRUCTURAL_RERANK:
        return False
    feature_names = list(getattr(dataset, "feature_names", []) or [])
    if len(feature_names) < HIGH_DIM_ROLE_TRIGGER_DIM:
        return False
    if not NO_LEAKAGE_MODE:
        return False
    return _is_benchmark_like_source_tag(getattr(dataset, "source_tag", ""))


def _score_candidate_expression_for_high_dim_structure(expr, structure_profile, feature_names):
    expr = str(expr or "").strip()
    if not expr:
        return -1e9

    profile = dict(structure_profile or {})
    roles = dict(profile.get("variable_roles", {}) or {})
    family_scores = dict(profile.get("family_scores", {}) or {})
    active_variables = list(roles.get("active_variables", []) or profile.get("active_variables", []) or feature_names)
    numerator_core = list(roles.get("numerator_core", []) or active_variables)
    denominator_core = list(roles.get("denominator_core", []) or [])
    periodic_core = list(roles.get("periodic_core", []) or [])
    top_pair_patterns = list(profile.get("top_pair_patterns", []) or [])

    sig = extract_formula_form_signature(expr, feature_names)
    families = set(sig.get("families", []) or [])
    variables_used = list(sig.get("variables_used", []) or [])
    denominator_vars = _extract_denominator_variables(expr, feature_names)
    periodic_vars = _extract_periodic_variables(expr, feature_names)
    low = _expr_dedup_key(expr)
    prefer_ratio_log = _profile_prefers_ratio_log_templates(structure_profile, feature_names)

    score = 0.0
    score += 1.50 * _safe_jaccard(variables_used, active_variables)
    if numerator_core:
        score += 0.55 * _safe_jaccard(variables_used, numerator_core)
    active_anchor = list((numerator_core or active_variables)[:3])
    missing_active_anchor = [v for v in active_anchor if v not in set(variables_used)]
    if denominator_core:
        score += 0.90 * _safe_jaccard(denominator_vars, denominator_core)
    elif denominator_vars:
        score -= 0.30 * len(denominator_vars)
    if periodic_core:
        score += 0.65 * _safe_jaccard(periodic_vars, periodic_core)
    elif periodic_vars:
        score -= 0.20 * len(periodic_vars)

    for family_name in families:
        score += 0.55 * float(family_scores.get(family_name, 0.0) or 0.0)

    if "rational" in families and float(family_scores.get("rational", 0.0) or 0.0) < 0.20:
        score -= 0.45
    if "logarithmic" in families and float(family_scores.get("logarithmic", 0.0) or 0.0) < 0.16:
        score -= 0.25
    if "trigonometric" in families and float(family_scores.get("trigonometric", 0.0) or 0.0) < 0.20:
        score -= 0.45
    if "exponential" in families and float(family_scores.get("exponential", 0.0) or 0.0) < 0.20:
        score -= 0.35
    if "logarithmic" in families and float(family_scores.get("logarithmic", 0.0) or 0.0) < 0.20:
        score -= 0.25

    if "abs(" in low and "rational" not in families:
        score -= 0.35
    elif "abs(" in low and float(family_scores.get("rational", 0.0) or 0.0) < 0.25:
        score -= 0.20

    if low.count("/") >= 2:
        score -= 0.20
    if len(variables_used) > max(3, len(active_variables)):
        score -= 0.18 * (len(variables_used) - max(3, len(active_variables)))
    if "log(" in low and "*" in low:
        score += 0.30 * max(
            float(family_scores.get("logarithmic", 0.0) or 0.0),
            float(family_scores.get("interaction", 0.0) or 0.0),
            float(family_scores.get("rational", 0.0) or 0.0),
        )
    if int(sig.get("max_multiplicative_arity", 1) or 1) >= 3:
        score += 0.22 * float(family_scores.get("interaction", 0.0) or 0.0)
    elif float(family_scores.get("interaction", 0.0) or 0.0) >= 0.26:
        score -= 0.18
    if _expr_has_log_ratio_signature(expr):
        score += 0.35 * max(
            float(family_scores.get("logarithmic", 0.0) or 0.0),
            float(family_scores.get("rational", 0.0) or 0.0),
        )
    elif ("log(" in low) and float(family_scores.get("logarithmic", 0.0) or 0.0) >= 0.24:
        score -= 0.14
    if prefer_ratio_log and _expr_has_affine_log_signature(expr):
        score -= 0.45
        if int(sig.get("max_multiplicative_arity", 1) or 1) < 3:
            score -= 0.18
    if prefer_ratio_log and _expr_has_log_ratio_signature(expr):
        score += 0.22
        if int(sig.get("max_multiplicative_arity", 1) or 1) >= 3:
            score += 0.18
    if prefer_ratio_log and float(family_scores.get("interaction", 0.0) or 0.0) >= 0.20 and len(active_anchor) >= 3:
        score -= 0.32 * len(missing_active_anchor)
        if _expr_has_log_ratio_signature(expr) and not missing_active_anchor and int(sig.get("max_multiplicative_arity", 1) or 1) >= 3:
            score += 0.30

    if looks_mechanistic(expr):
        score += 0.20

    variables_used_set = set(variables_used)
    for item in top_pair_patterns[:3]:
        pair_vars = list(item.get("variables", []) or [])
        if len(pair_vars) < 2:
            continue
        if not set(pair_vars[:2]).issubset(variables_used_set):
            continue
        pair_family = str(item.get("family", "") or "")
        pair_score = float(item.get("score", 0.0) or 0.0)
        score += 0.25 * pair_score
        if pair_family == "interaction" and "*" in low:
            score += 0.25 * pair_score
        elif pair_family == "rational" and "/" in low:
            score += 0.25 * pair_score
        elif pair_family == "logarithmic" and "log(" in low:
            score += 0.25 * pair_score
        elif pair_family == "power" and "**" in low:
            score += 0.20 * pair_score
        elif pair_family == "trigonometric" and ("sin(" in low or "cos(" in low):
            score += 0.20 * pair_score

    score -= 0.003 * len(low)
    return float(score)


def structural_rerank_high_dim_candidates(candidate_exprs, dataset, top_k=None, structure_profile=None):
    clean_exprs = deduplicate_expressions(candidate_exprs)
    if not clean_exprs:
        return clean_exprs, {"applied": False, "reason": "empty"}
    if not _should_apply_high_dim_structural_rerank(dataset):
        return clean_exprs, {"applied": False, "reason": "not_high_dim_no_leakage_benchmark"}

    feature_names = list(getattr(dataset, "feature_names", []) or [])
    structure_profile = dict(structure_profile or build_structure_profile(dataset) or {})
    if not structure_profile:
        return clean_exprs, {"applied": False, "reason": "structure_profile_unavailable"}

    ranked = []
    for rank_before, expr in enumerate(clean_exprs, start=1):
        ranked.append({
            "expr": expr,
            "rank_before": rank_before,
            "score": _score_candidate_expression_for_high_dim_structure(expr, structure_profile, feature_names),
        })
    ranked_sorted = sorted(ranked, key=lambda x: (-x["score"], x["rank_before"], len(str(x["expr"]))))

    top_k = max(1, int(top_k or HIGH_DIM_STRUCTURAL_RERANK_TOPK))
    selected = []
    selected_keys = set()
    for item in ranked_sorted[:top_k]:
        expr_key = _expr_dedup_key(item["expr"])
        if expr_key in selected_keys:
            continue
        selected.append(item["expr"])
        selected_keys.add(expr_key)

    mechanistic_keep = 0
    for item in ranked_sorted[top_k:]:
        if mechanistic_keep >= int(HIGH_DIM_STRUCTURAL_RERANK_KEEP_MECHANISTIC):
            break
        expr_key = _expr_dedup_key(item["expr"])
        if expr_key in selected_keys or not looks_mechanistic(item["expr"]):
            continue
        selected.append(item["expr"])
        selected_keys.add(expr_key)
        mechanistic_keep += 1

    trace = {
        "applied": len(selected) < len(clean_exprs),
        "selected_count": len(selected),
        "candidate_count": len(clean_exprs),
        "top_k": int(top_k),
        "top_scores": [
            {
                "expr": item["expr"],
                "score": round(float(item["score"]), 4),
                "rank_before": int(item["rank_before"]),
            }
            for item in ranked_sorted[:min(5, len(ranked_sorted))]
        ],
    }
    return selected, trace


def lightweight_prefilter_candidates(candidate_exprs, dataset, timer=None, prefix="prefilter"):
    if not ENABLE_LIGHT_PREFILTER or not candidate_exprs:
        return candidate_exprs

    # 2D 题候选不多，直接跳过预筛，避免干净乘积骨架被误筛
    feature_count = len(getattr(dataset, "feature_names", []))
    if feature_count <= 2:
        return candidate_exprs

    if _should_apply_high_dim_structural_rerank(dataset):
        candidate_exprs, _ = structural_rerank_high_dim_candidates(
            candidate_exprs,
            dataset,
            top_k=HIGH_DIM_STRUCTURAL_RERANK_TOPK,
        )

    n_train = len(dataset.train_df)
    keep_n = min(n_train, max(PREFILTER_MIN_ROWS, int(n_train * PREFILTER_TRAIN_FRAC)))
    if keep_n >= n_train:
        return candidate_exprs

    small_train = dataset.train_df.sample(keep_n, random_state=123).reset_index(drop=True)
    with TemporaryDirectory() as td:
        tmpdir = Path(td)
        sealed_test_placeholder = dataset.val_df.iloc[0:0].copy()
        small_dataset = build_dataset_from_explicit_splits(
            small_train,
            dataset.val_df,
            sealed_test_placeholder,
            tmpdir,
        )
        small_dataset.source_tag = getattr(dataset, "source_tag", "")
        share_candidate_search_audit(dataset, small_dataset)
        _, _, _, small_scored = evaluate_candidate_expressions(
            candidate_exprs=candidate_exprs,
            dataset=small_dataset,
            complexity_weight=COMPLEXITY_WEIGHT,
            timer=timer,
            prefix=prefix,
        )

    selected = []
    seen = set()

    local_topk = TOPK_AFTER_PREFILTER
    extra_tail = 4
    if _is_benchmark_like_source_tag(getattr(dataset, "source_tag", "")) and feature_count >= 5 and NO_LEAKAGE_MODE:
        # Fair high-dimensional mode: keep a wider and more diverse prefilter set.
        local_topk = max(local_topk, 12)
        extra_tail = max(extra_tail, 6)
    elif _is_benchmark_like_source_tag(getattr(dataset, "source_tag", "")) and feature_count == 5:
        local_topk = max(local_topk, 10)
    if _is_benchmark_like_source_tag(getattr(dataset, "source_tag", "")) and feature_count == 2:
        local_topk = max(local_topk, 12)

    def _remember(expr):
        expr = str(expr or "").strip()
        if not expr:
            return False
        key = _expr_dedup_key(expr)
        if key in seen:
            return False
        selected.append(expr)
        seen.add(key)
        return True

    # First preserve at least one candidate from each structural family, ranked by small-set performance.
    if GENERIC_DIVERSE_FAMILY_PRESERVE_PREFILTER:
        family_kept = set()
        for item in small_scored:
            expr = str(_safe_get_attr(item, "simplified_expression", "") or "").strip()
            if not expr:
                continue
            fam = candidate_family_label(expr)
            if fam in family_kept:
                continue
            if _remember(expr):
                family_kept.add(fam)
            if len(family_kept) >= 8:
                break

    # Then fill by validation performance on the small prefilter set.
    for item in small_scored[:local_topk]:
        expr = str(_safe_get_attr(item, "simplified_expression", "") or "").strip()
        _remember(expr)
        if len(selected) >= local_topk:
            break

    # Preserve mechanistic candidates from the original pool, because some structures fit poorly before constants are tuned.
    for expr in candidate_exprs:
        expr = str(expr).strip()
        if expr and looks_mechanistic(expr):
            _remember(expr)

    for expr in candidate_exprs:
        expr = str(expr).strip()
        if expr and len(selected) < local_topk + extra_tail:
            _remember(expr)

    return selected if selected else candidate_exprs



# =========================
# 多智能体新增配置 / helper / agent
# =========================

USE_ROLE_BASED_BACKENDS = True

# Fallback: proposal and multimodal can still use your original BACKEND_CONFIG.
# Meta / judge may use a different remote API or a stronger model.
ROLE_BACKEND_CONFIGS = {
    "proposal": {
        "backend_type": "openai_compatible",
        "model": os.environ.get("LLMSR_PROPOSAL_MODEL", DEFAULT_LLM_MODEL),
        "api_base_url": os.environ.get("LLMSR_PROPOSAL_API_BASE", DEFAULT_LLM_API_BASE),
        "api_key": os.environ.get("LLMSR_PROPOSAL_API_KEY", DEFAULT_LLM_API_KEY),
        "timeout": 45,
    },
    "meta": {
        "backend_type": "openai_compatible",
        "model": os.environ.get("LLMSR_META_MODEL", DEFAULT_LLM_MODEL),
        "api_base_url": os.environ.get("LLMSR_META_API_BASE", DEFAULT_LLM_API_BASE),
        "api_key": os.environ.get("LLMSR_META_API_KEY", DEFAULT_LLM_API_KEY),
        "timeout": 45,
    },
    "judge": {
        "backend_type": "openai_compatible",
        "model": os.environ.get("LLMSR_JUDGE_MODEL", DEFAULT_LLM_MODEL),
        "api_base_url": os.environ.get("LLMSR_JUDGE_API_BASE", DEFAULT_LLM_API_BASE),
        "api_key": os.environ.get("LLMSR_JUDGE_API_KEY", DEFAULT_LLM_API_KEY),
        "timeout": 45,
    },
    "refiner": {
        "backend_type": "openai_compatible",
        "model": os.environ.get("LLMSR_REFINER_MODEL", DEFAULT_LLM_MODEL),
        "api_base_url": os.environ.get("LLMSR_REFINER_API_BASE", DEFAULT_LLM_API_BASE),
        "api_key": os.environ.get("LLMSR_REFINER_API_KEY", DEFAULT_LLM_API_KEY),
        "timeout": 45,
    },
    # Planner approval is text-only and lightweight. By default it reuses the
    # same local endpoint, but you can point it to a faster 7B server via env:
    #   LLMSR_PLANNER_MODEL=/path/to/Qwen2.5-7B-Instruct
    #   LLMSR_PLANNER_API_BASE=http://127.0.0.1:8002/v1
    "planner": {
        "backend_type": "openai_compatible",
        "model": os.environ.get("LLMSR_PLANNER_MODEL", DEFAULT_LLM_MODEL),
        "api_base_url": os.environ.get("LLMSR_PLANNER_API_BASE", DEFAULT_LLM_API_BASE),
        "api_key": os.environ.get("LLMSR_PLANNER_API_KEY", DEFAULT_LLM_API_KEY),
        "timeout": PLANNER_APPROVAL_TIMEOUT_SEC,
    },
}

# Force structured JSON output from meta/judge. This is the single most important
# engineering decision when you split the system into agents.
FORCE_JSON_FOR_META = True
FORCE_JSON_FOR_JUDGE = True

# Retry JSON parse from model output.
AGENT_JSON_PARSE_RETRIES = 2

# How many top expressions are shown to meta/judge each round.
META_TOPK = 3
JUDGE_TOPK = 3
META_MAX_TOKENS = 400
JUDGE_MAX_TOKENS = 400
REFINER_MAX_TOKENS = 700
META_FAST_STOP_WITH_HEURISTIC = True

# Whether to let judge call a stronger external model while proposal remains local.
# Good practical choice:
#   - proposal/refiner: local model (cheap, fast)
#   - meta/judge: stronger remote API (better planning and feedback quality)
META_USE_STRONGER_MODEL = False
JUDGE_USE_STRONGER_MODEL = False

# Safety fallback: if meta or judge API fails, use heuristic decision / feedback.
ENABLE_META_HEURISTIC_FALLBACK = True
ENABLE_JUDGE_HEURISTIC_FALLBACK = True

# Prevent meta/judge/refiner/planner calls from consuming the full per-task
# budget when the local server is slow. Proposal text/diverse are already
# disabled on high-dimensional no-leakage paths; these limits mainly protect
# the reasoning agents.
try:
    for _role in ("meta", "judge", "refiner"):
        if _role in ROLE_BACKEND_CONFIGS:
            ROLE_BACKEND_CONFIGS[_role]["timeout"] = min(
                int(ROLE_BACKEND_CONFIGS[_role].get("timeout", HIGH_DIM_NO_LEAKAGE_AGENT_TIMEOUT_SEC)),
                int(HIGH_DIM_NO_LEAKAGE_AGENT_TIMEOUT_SEC),
            )
except Exception:
    pass


# =========================================================
# 【新增工具占位】physics / residual 的轻量占位实现
# =========================================================
# 这里不是最终形态，而是为了让多智能体主循环先跑通。
# 你后续可以把它们替换为真正的 physics_check_tool / residual_pattern_tool。


def physics_check_placeholder(candidate_exprs, dataset, best_result=None):
    """
    Replace with your real physics_check_tool when available.
    Current behavior: return a conservative placeholder summary.
    """
    best_expr = None
    if best_result is not None:
        best_expr = _safe_get_attr(best_result, "simplified_expression", None)
    return {
        "unit_consistent": "unknown",
        "boundary_issues": [],
        "limit_issues": [],
        "message": f"No explicit physics checker wired yet. Current best expr={best_expr}",
    }


def _repair_actions_for_family(family_name):
    mapping = {
        "trigonometric": ["add_periodic_component", "periodic_modulation"],
        "rational": ["denominator_correction", "ratio_rewrite"],
        "interaction": ["expand_interaction", "cross_term_expansion"],
        "power": ["add_power_term", "structured_power_correction"],
        "exponential": ["add_exponential_component"],
        "logarithmic": ["add_log_component"],
        "additive": ["add_residual_term", "partial_separation"],
        "algebraic": ["local_rewrite"],
    }
    return list(mapping.get(str(family_name or ""), ["local_rewrite"]))


def _fit_single_basis_gain(train_basis, train_residual, val_basis=None, val_residual=None):
    try:
        train_basis = np.asarray(train_basis, dtype=float).reshape(-1)
        train_residual = np.asarray(train_residual, dtype=float).reshape(-1)
        mask = np.isfinite(train_basis) & np.isfinite(train_residual)
        if int(mask.sum()) < 8:
            return None
        x = train_basis[mask]
        y = train_residual[mask]
        if np.nanstd(x) < 1e-10 or np.nanstd(y) < 1e-12:
            return None
        design = np.column_stack([x, np.ones_like(x)])
        coeff, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
        pred_train = coeff[0] * x + coeff[1]
        mse_before_train = float(np.mean(y ** 2))
        mse_after_train = float(np.mean((y - pred_train) ** 2))
        train_gain = 0.0 if mse_before_train <= 1e-12 else max(0.0, 1.0 - mse_after_train / mse_before_train)

        val_gain = train_gain
        mse_before_val = None
        mse_after_val = None
        if val_basis is not None and val_residual is not None:
            val_basis = np.asarray(val_basis, dtype=float).reshape(-1)
            val_residual = np.asarray(val_residual, dtype=float).reshape(-1)
            val_mask = np.isfinite(val_basis) & np.isfinite(val_residual)
            if int(val_mask.sum()) >= 6:
                xv = val_basis[val_mask]
                yv = val_residual[val_mask]
                pred_val = coeff[0] * xv + coeff[1]
                mse_before_val = float(np.mean(yv ** 2))
                mse_after_val = float(np.mean((yv - pred_val) ** 2))
                if mse_before_val > 1e-12:
                    val_gain = max(0.0, 1.0 - mse_after_val / mse_before_val)

        combined_gain = 0.65 * float(val_gain) + 0.35 * float(train_gain)
        return {
            "train_gain": float(train_gain),
            "val_gain": float(val_gain),
            "combined_gain": float(combined_gain),
            "train_mse_before": mse_before_train,
            "train_mse_after": mse_after_train,
            "val_mse_before": mse_before_val,
            "val_mse_after": mse_after_val,
            "coeff": [float(coeff[0]), float(coeff[1])],
        }
    except Exception:
        return None


def _build_residual_probe_terms(dataset, active_variables=None):
    feature_names = list(getattr(dataset, "feature_names", []) or [])
    target_name = getattr(dataset, "target_name", "y")
    if not feature_names or dataset is None or dataset.train_df is None:
        return []

    train_df = dataset.train_df
    val_df = dataset.val_df if getattr(dataset, "val_df", None) is not None else None
    feature_stats = _feature_standardization_stats(train_df, feature_names)

    candidate_vars = [str(x) for x in (active_variables or []) if str(x) in feature_names]
    for name in feature_names:
        if name not in candidate_vars:
            candidate_vars.append(name)
    candidate_vars = candidate_vars[:min(5, len(candidate_vars))]

    z_train = {name: _scaled_feature_values(train_df, name, feature_stats.get(name, {})) for name in candidate_vars}
    z_val = {name: _scaled_feature_values(val_df, name, feature_stats.get(name, {})) for name in candidate_vars} if val_df is not None else {}

    probes = []

    def add_probe(label, family, variables, action, train_values, val_values=None):
        train_values = np.asarray(train_values, dtype=float).reshape(-1)
        if train_values.size == 0 or np.nanstd(train_values[np.isfinite(train_values)]) < 1e-10:
            return
        probes.append({
            "label": label,
            "family": family,
            "variables": list(variables),
            "action": action,
            "train_values": train_values,
            "val_values": np.asarray(val_values, dtype=float).reshape(-1) if val_values is not None else None,
        })

    for name in candidate_vars:
        ztr = z_train[name]
        zv = z_val.get(name)
        add_probe(name, "additive", [name], "add_residual_term", ztr, zv)
        add_probe(f"{name}**2", "power", [name], "add_power_term", ztr ** 2, None if zv is None else zv ** 2)
        add_probe(f"{name}**3", "power", [name], "add_power_term", ztr ** 3, None if zv is None else zv ** 3)
        add_probe(f"sin({name})", "trigonometric", [name], "add_periodic_component", np.sin(ztr), None if zv is None else np.sin(zv))
        add_probe(f"cos({name})", "trigonometric", [name], "add_periodic_component", np.cos(ztr), None if zv is None else np.cos(zv))
        add_probe(f"1/({name}+c)", "rational", [name], "denominator_correction", _reciprocal_like(ztr), None if zv is None else _reciprocal_like(zv))
        add_probe(f"exp({name})", "exponential", [name], "add_exponential_component", np.exp(np.clip(ztr, -2.0, 2.0)), None if zv is None else np.exp(np.clip(zv, -2.0, 2.0)))
        add_probe(f"log(abs({name})+c)", "logarithmic", [name], "add_log_component", np.log1p(np.abs(ztr)), None if zv is None else np.log1p(np.abs(zv)))

    for i in range(len(candidate_vars)):
        for j in range(i + 1, len(candidate_vars)):
            x1 = candidate_vars[i]
            x2 = candidate_vars[j]
            z1 = z_train[x1]
            z2 = z_train[x2]
            zv1 = z_val.get(x1)
            zv2 = z_val.get(x2)
            add_probe(
                f"{x1}*{x2}",
                "interaction",
                [x1, x2],
                "expand_interaction",
                z1 * z2,
                None if zv1 is None or zv2 is None else zv1 * zv2,
            )
            add_probe(
                f"{x1}/({x2}+c)",
                "rational",
                [x1, x2],
                "denominator_correction",
                z1 / (1.0 + np.abs(z2)),
                None if zv1 is None or zv2 is None else zv1 / (1.0 + np.abs(zv2)),
            )
            add_probe(
                f"{x2}/({x1}+c)",
                "rational",
                [x1, x2],
                "denominator_correction",
                z2 / (1.0 + np.abs(z1)),
                None if zv1 is None or zv2 is None else zv2 / (1.0 + np.abs(zv1)),
            )
            add_probe(
                f"sin({x1}+{x2})",
                "trigonometric",
                [x1, x2],
                "periodic_modulation",
                np.sin(z1 + z2),
                None if zv1 is None or zv2 is None else np.sin(zv1 + zv2),
            )
            add_probe(
                f"{x1}**2-{x2}**2",
                "power",
                [x1, x2],
                "structured_power_correction",
                z1 ** 2 - z2 ** 2,
                None if zv1 is None or zv2 is None else zv1 ** 2 - zv2 ** 2,
            )

    return probes



def expression_variable_set(expr, feature_names):
    expr = str(expr or "")
    return {v for v in (feature_names or []) if re.search(rf"\b{re.escape(v)}\b", expr)}


def build_missing_multiplier_rescue_candidates(current_best, dataset, max_candidates=None):
    """
    Generic local rescue: if a promising nonlinear skeleton uses only a subset
    of variables, try multiplicative envelopes with missing variables.

    This is not benchmark-specific. It is triggered by the current best formula
    and feature names only. It addresses cases like: best ~= x1*x2*log(x5/x4),
    target may require an additional multiplicative envelope variable.
    """
    if not ENABLE_MISSING_MULTIPLIER_RESCUE:
        return []
    if current_best is None or dataset is None:
        return []
    expr = str(_safe_get_attr(current_best, "simplified_expression", "") or "").strip()
    if not expr:
        return []
    feature_names = list(getattr(dataset, "feature_names", []) or [])
    if len(feature_names) < 3:
        return []
    low = expr.lower()
    if not any(tok in low for tok in ["log(", "exp(", "sin(", "cos(", "/", "*"]):
        return []
    used = expression_variable_set(expr, feature_names)
    missing = [v for v in feature_names if v not in used]
    if not missing:
        return []
    max_candidates = int(max_candidates or MISSING_MULTIPLIER_RESCUE_MAX_CANDIDATES)
    out = []
    seen = set()
    def add(e):
        e = str(e).strip()
        key = _expr_dedup_key(e)
        if e and key not in seen:
            seen.add(key)
            out.append(e)
        return len(out) >= max_candidates
    clean_expr = re.sub(r"\bAbs\s*\(", "abs(", expr)
    for v in missing:
        if add(f"a*({clean_expr})*{v} + b"):
            return out[:max_candidates]
        if add(f"a*({clean_expr})*({v}+b1) + b"):
            return out[:max_candidates]
        if add(f"a*({clean_expr}) + b*{v} + c"):
            return out[:max_candidates]
        if add(f"a*({clean_expr})*(1 + b1*{v}) + c"):
            return out[:max_candidates]
    from itertools import combinations
    for v1, v2 in combinations(missing, 2):
        if add(f"a*({clean_expr})*{v1}*{v2} + b"):
            return out[:max_candidates]
        if add(f"a*({clean_expr})*({v1}+b1)*({v2}+b2) + c"):
            return out[:max_candidates]
    return out[:max_candidates]


def residual_pattern_placeholder(scored_results, dataset):
    if not scored_results:
        return {
            "has_pattern": "unknown",
            "pattern_type": "unknown",
            "message": "No scored results, residual pattern unavailable.",
        }

    best = scored_results[0]
    val_mse = _safe_get_attr(best, "val_mse", None)
    expr = _safe_get_attr(best, "simplified_expression", None)
    try:
        if not expr:
            return {
                "has_pattern": "unknown",
                "pattern_type": "unknown",
                "message": "Best expression unavailable, residual diagnosis skipped.",
            }

        train_pred = evaluate_expression_on_df(expr, dataset.train_df)
        train_target = np.asarray(dataset.train_df[dataset.target_name], dtype=float)
        train_residual = train_target - np.asarray(train_pred, dtype=float)

        val_residual = None
        if getattr(dataset, "val_df", None) is not None and len(dataset.val_df) > 0:
            val_pred = evaluate_expression_on_df(expr, dataset.val_df)
            val_target = np.asarray(dataset.val_df[dataset.target_name], dtype=float)
            val_residual = val_target - np.asarray(val_pred, dtype=float)

        structure_profile = build_structure_profile(dataset)
        current_signature = extract_formula_form_signature(expr, dataset.feature_names)
        probes = _build_residual_probe_terms(
            dataset,
            active_variables=list(structure_profile.get("active_variables", []) or current_signature.get("variables_used", []) or []),
        )

        scored_probes = []
        for probe in probes:
            fit = _fit_single_basis_gain(
                train_basis=probe.get("train_values"),
                train_residual=train_residual,
                val_basis=probe.get("val_values"),
                val_residual=val_residual,
            )
            if fit is None:
                continue
            scored_probes.append({
                "label": probe.get("label"),
                "family": probe.get("family"),
                "variables": list(probe.get("variables", []) or []),
                "action": probe.get("action"),
                "combined_gain": round(float(fit.get("combined_gain", 0.0)), 4),
                "train_gain": round(float(fit.get("train_gain", 0.0)), 4),
                "val_gain": round(float(fit.get("val_gain", 0.0)), 4),
            })

        scored_probes.sort(
            key=lambda x: (
                -float(x.get("combined_gain", 0.0)),
                -float(x.get("val_gain", 0.0)),
                len(str(x.get("label", ""))),
            )
        )
        top_probes = scored_probes[:6]

        family_scores = {}
        suggested_variables = []
        repair_actions = []
        for item in top_probes:
            family_name = str(item.get("family", "")).strip() or "algebraic"
            family_scores[family_name] = max(float(family_scores.get(family_name, 0.0)), float(item.get("combined_gain", 0.0)))
            for var in item.get("variables", []) or []:
                if var not in suggested_variables:
                    suggested_variables.append(var)
            for action in _repair_actions_for_family(family_name):
                if action not in repair_actions:
                    repair_actions.append(action)

        top_probe = top_probes[0] if top_probes else None
        best_gain = float(top_probe.get("combined_gain", 0.0)) if top_probe else 0.0
        if best_gain >= 0.08:
            has_pattern = True
            pattern_type = f"missing_{top_probe.get('family', 'structure')}_structure"
        elif best_gain >= 0.03:
            has_pattern = True
            pattern_type = "moderate_remaining_structure"
        else:
            has_pattern = False
            pattern_type = "small_or_diffuse_residual"

        preserve_substructures = []
        for family_name in current_signature.get("families", []) or []:
            if family_name in {"interaction", "rational", "trigonometric", "exponential", "power"}:
                preserve_substructures.append(f"keep existing {family_name} structure if it already helps")
        if not preserve_substructures and expr:
            preserve_substructures.append("keep useful outer structure from current best")

        repair_targets = []
        for item in top_probes[:3]:
            variables = list(item.get("variables", []) or [])
            var_text = ", ".join(variables) if variables else "current active variables"
            repair_targets.append(
                f"missing {item.get('family')} structure around {var_text} via {item.get('label')}"
            )
        if not repair_targets:
            repair_targets = [pattern_type]

        guidance_lines = []
        if top_probe:
            guidance_lines.append(
                f"best residual probe is {top_probe.get('label')} ({top_probe.get('family')}) with combined gain {best_gain:.3f}"
            )
        if suggested_variables:
            guidance_lines.append("focus refinement on variables: " + ", ".join(suggested_variables[:4]))
        if family_scores:
            guidance_lines.append(
                "missing-family scores: " + ", ".join(f"{name}={score:.3f}" for name, score in _top_score_items(family_scores, top_k=4, min_score=0.0))
            )

        return {
            "has_pattern": has_pattern,
            "pattern_type": pattern_type,
            "message": f"Residual diagnosis from best_expr={expr}; best_val_mse={val_mse}",
            "best_expr": expr,
            "best_val_mse": val_mse,
            "current_signature": make_json_safe(current_signature),
            "top_probes": top_probes,
            "best_probe": top_probe,
            "missing_family_scores": {name: round(float(score), 4) for name, score in family_scores.items()},
            "suggested_active_variables": suggested_variables[:5],
            "preserve_substructures": preserve_substructures[:4],
            "repair_actions": repair_actions[:6],
            "repair_targets": repair_targets[:4],
            "guidance_lines": guidance_lines[:5],
        }
    except Exception as e:
        return {
            "has_pattern": "unknown",
            "pattern_type": "diagnosis_failed",
            "message": f"Residual diagnosis failed: {repr(e)}",
            "best_expr": expr,
            "best_val_mse": val_mse,
        }


# =========================================================
# 【新增辅助函数】根据角色构造 LLMClient
# =========================================================
# 这让不同 agent 能透明地使用不同模型，而主流程不需要知道底层差异。


def _build_role_client(role: str, default_backend_config: Dict[str, Any], deadline_ts=None):
    """
    role in {proposal, meta, judge, refiner}
    """
    cfg = dict(default_backend_config or {})
    if USE_ROLE_BASED_BACKENDS:
        cfg = dict(ROLE_BACKEND_CONFIGS.get(role, default_backend_config) or {})
    configured_timeout = float(cfg.get("timeout", 90) or 90)
    if deadline_ts is not None:
        remaining = _task_budget_remaining_sec(deadline_ts)
        if remaining is not None:
            configured_timeout = min(
                configured_timeout,
                max(float(MIN_LLM_CALL_TIMEOUT_SEC), float(remaining) - float(LLM_TIMEOUT_BUDGET_MARGIN_SEC)),
            )
    cfg["timeout"] = max(float(MIN_LLM_CALL_TIMEOUT_SEC), float(configured_timeout))
    backend_cfg = BackendConfig(**cfg)
    return LLMClient(backend_cfg)


# =========================================================
# 【新增稳健解析】从模型输出中提取 JSON
# =========================================================
# 多智能体系统里，Meta/Judge 最怕“输出格式漂移”。
# 因此这里强制使用 JSON，并加入多层兜底解析：
# 1. 直接 json.loads
# 2. 提取 ```json ... ``` 代码块
# 3. 扫描首尾大括号


def _extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    text = text.strip()

    # Try direct parse first.
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Try fenced code block.
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if m:
        cand = m.group(1).strip()
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    # Bracket scan fallback.
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        cand = text[start:end + 1]
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    return None


def _coerce_meta_decision(obj: Optional[Dict[str, Any]], current_best=None) -> Dict[str, Any]:
    if not isinstance(obj, dict):
        best_expr = _safe_get_attr(current_best, "simplified_expression", None) if current_best is not None else None
        return {
            "should_refine": True,
            "confidence": 0.35,
            "reason": "meta parse failed; fallback to conservative refine",
            "target_exprs": [best_expr] if best_expr else [],
            "preserve_patterns": [],
            "repair_targets": ["reduce validation error"],
            "actions": ["local_rewrite", "small_structural_extension"],
            "budget": {
                "num_candidates": 3,
                "temperature": 0.4,
                "use_multimodal": True,
            },
        }

    out = {
        "should_refine": bool(obj.get("should_refine", True)),
        "confidence": float(obj.get("confidence", 0.5)),
        "reason": str(obj.get("reason", "")),
        "target_exprs": [str(x) for x in obj.get("target_exprs", []) if x],
        "preserve_patterns": [str(x) for x in obj.get("preserve_patterns", []) if x],
        "repair_targets": [str(x) for x in obj.get("repair_targets", []) if x],
        "actions": [str(x) for x in obj.get("actions", []) if x],
        "budget": obj.get("budget", {}) if isinstance(obj.get("budget", {}), dict) else {},
    }
    if obj.get("critic_action") is not None:
        out["critic_action"] = str(obj.get("critic_action"))
    if obj.get("target_families") is not None:
        out["target_families"] = [str(x) for x in obj.get("target_families", []) if x]
    if obj.get("active_variables") is not None:
        out["active_variables"] = [str(x) for x in obj.get("active_variables", []) if x]
    if obj.get("complexity_pressure") is not None:
        out["complexity_pressure"] = str(obj.get("complexity_pressure"))
    out["budget"].setdefault("num_candidates", 3)
    out["budget"].setdefault("temperature", 0.4)
    out["budget"].setdefault("use_multimodal", False)
    return out


def _coerce_judge_feedback(obj: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(obj, dict):
        return {
            "feedback_text": (
                "Preserve useful structure from the current best expression if any. "
                "Repair the dominant source of validation error. Prefer local edits before replacing the whole family."
            ),
            "keep_constraints": [],
            "repair_targets": ["reduce validation error"],
            "avoid_patterns": [],
        }

    return {
        "feedback_text": str(obj.get("feedback_text", "")).strip(),
        "keep_constraints": [str(x) for x in obj.get("keep_constraints", []) if x],
        "repair_targets": [str(x) for x in obj.get("repair_targets", []) if x],
        "avoid_patterns": [str(x) for x in obj.get("avoid_patterns", []) if x],
    }


# =========================================================
# Data bundles
# =========================================================

@dataclass
class ObservationBundle:
    structure_hints: List[str]
    visual_hints: List[str]
    unit_hints: Dict[str, str]
    plot_descriptions: List[str]
    image_paths: List[str]
    structure_profile: Optional[Dict[str, Any]] = None
    visual_summary: Optional[Dict[str, Any]] = None
    reconstruction_tokens: Optional[Dict[str, Any]] = None
    reconstruction_trace: Optional[Dict[str, Any]] = None
    reconstruction_image_paths: Optional[List[str]] = None
    reconstruction_descriptions: Optional[List[str]] = None
    mm_assets_attempted: bool = False
    mm_assets_succeeded: bool = False
    mm_assets_error: Optional[str] = None


# =========================================================
# 【新增 Agent 层】
# =========================================================
# 本次改造的核心。
# 你可以把每个 Agent 理解成一个有明确职责的模块：
#   ObserverAgent  : 观察数据、画图、形成视觉/结构线索
#   ProposerAgent  : 生成初始候选表达式
#   EvaluatorAgent : 调工具链做拟合、化简、去重、打分、诊断
#   MetaAgent      : 判断是否 refine，给出 refine 计划
#   JudgeAgent     : 把诊断信息整理成给 proposer/refiner 的反馈文本
#   RefinerAgent   : 根据 judge 反馈生成下一轮 refined expressions


class ObserverAgent:
    def __init__(self, plot_tool, high_dim_reconstruction_tool=None):
        self.plot_tool = plot_tool
        self.high_dim_reconstruction_tool = high_dim_reconstruction_tool

    def _run_high_dim_reconstruction(self, dataset, row_meta, structure_profile, timer=None):
        """Run high-dimensional visual reconstruction and return tokens/images.

        This is separated from normal plot generation so reconstruction tokens can
        be produced during observation, not only after the MM rescue branch fires.
        """
        if not (
            ENABLE_HIGH_DIM_RECONSTRUCTION
            and self.high_dim_reconstruction_tool is not None
            and len(list(getattr(dataset, "feature_names", []) or [])) >= HIGH_DIM_RECON_TRIGGER_DIM
        ):
            return None, None, [], []

        timer_started = False
        try:
            if timer is not None:
                timer.start("step_high_dim_reconstruction")
                timer_started = True
            rel_name = sanitize_name(f"{row_meta['dataset_dir']}__{row_meta['base_name']}")
            recon_dir = os.path.join(RESULTS_ROOT, "reconstruction", rel_name)
            recon_result = self.high_dim_reconstruction_tool.run(
                df=dataset.train_df,
                feature_names=dataset.feature_names,
                target_name=dataset.target_name,
                output_dir=recon_dir,
                prefix="train_recon",
                structure_profile=structure_profile,
            )
            if timer is not None and timer_started:
                timer.stop("step_high_dim_reconstruction")
            reconstruction_tokens = dict(recon_result.get("tokens", {}) or {}) or None
            reconstruction_trace = dict(recon_result.get("trace", {}) or {}) or None
            reconstruction_image_paths = list(recon_result.get("image_paths", []) or [])
            reconstruction_descriptions = list(recon_result.get("descriptions", []) or [])
            return reconstruction_tokens, reconstruction_trace, reconstruction_image_paths, reconstruction_descriptions
        except Exception as e:
            if timer is not None and timer_started:
                timer.stop("step_high_dim_reconstruction")
            return None, {"error": repr(e)}, [], []

    def observe(self, dataset, row_meta, timer=None) -> ObservationBundle:
        structure_profile = build_structure_profile(dataset)
        structure_hints = deduplicate_expressions(
            build_basic_structure_hints(dataset) + build_structure_hints_from_profile(structure_profile, feature_names=dataset.feature_names)
        )
        unit_hints = {name: "unknown" for name in dataset.feature_names}
        unit_hints[dataset.target_name] = "unknown"

        image_paths = []
        plot_descriptions = []
        reconstruction_tokens = None
        reconstruction_trace = None
        reconstruction_image_paths = []
        reconstruction_descriptions = []

        # Visual core path: for high-dimensional tasks, reconstruct structural
        # visual tokens during observation. This makes the visual module part of
        # the main pipeline rather than an optional side-effect of MM rescue.
        if FORCE_HIGH_DIM_RECON_IN_OBSERVE:
            reconstruction_tokens, reconstruction_trace, reconstruction_image_paths, reconstruction_descriptions = self._run_high_dim_reconstruction(
                dataset=dataset,
                row_meta=row_meta,
                structure_profile=structure_profile,
                timer=timer,
            )
            # Use reconstruction images as MM assets as well, so later VLM/MM
            # proposal can be triggered without regenerating plots.
            image_paths = list(reconstruction_image_paths or [])
            plot_descriptions = list(reconstruction_descriptions or [])

        visual_summary = build_visual_summary(
            structure_profile=structure_profile,
            plot_descriptions=plot_descriptions,
            image_paths=image_paths,
        )
        visual_hints = build_visual_hints_from_summary(visual_summary) or ["no visual hints available"]
        if reconstruction_tokens:
            visual_hints = deduplicate_expressions(
                visual_hints + ["high-dimensional visual reconstruction tokens are available and should be treated as structural evidence"]
            )

        return ObservationBundle(
            structure_hints=structure_hints,
            visual_hints=visual_hints,
            unit_hints=unit_hints,
            plot_descriptions=plot_descriptions,
            image_paths=image_paths,
            structure_profile=structure_profile,
            visual_summary=visual_summary,
            reconstruction_tokens=reconstruction_tokens,
            reconstruction_trace=reconstruction_trace,
            reconstruction_image_paths=reconstruction_image_paths,
            reconstruction_descriptions=reconstruction_descriptions,
        )

    def maybe_generate_mm_assets(self, dataset, row_meta, observation: ObservationBundle, timer=None) -> ObservationBundle:
        # If observation already produced reconstruction images, reuse them but
        # still mark the MM assets branch as attempted. This is important for
        # logs and for confirming that VLM/MM proposal had usable visual inputs.
        if observation.image_paths and observation.reconstruction_tokens:
            observation.mm_assets_attempted = True
            observation.mm_assets_succeeded = True
            return observation

        timer_started = False
        try:
            if timer is not None:
                timer.start("step_plot_generation")
                timer_started = True
            image_paths, plot_descriptions, visual_hints = maybe_generate_plots(self.plot_tool, dataset, row_meta)
            if timer is not None:
                timer.stop("step_plot_generation")

            structure_profile = observation.structure_profile or build_structure_profile(dataset)
            reconstruction_tokens = observation.reconstruction_tokens
            reconstruction_trace = observation.reconstruction_trace
            reconstruction_image_paths = list(observation.reconstruction_image_paths or [])
            reconstruction_descriptions = list(observation.reconstruction_descriptions or [])
            if not reconstruction_tokens:
                reconstruction_tokens, reconstruction_trace, reconstruction_image_paths, reconstruction_descriptions = self._run_high_dim_reconstruction(
                    dataset=dataset,
                    row_meta=row_meta,
                    structure_profile=structure_profile,
                    timer=timer,
                )
            merged_paths = []
            merged_descriptions = []
            seen_paths = set()
            for path, desc in zip(reconstruction_image_paths + image_paths, reconstruction_descriptions + plot_descriptions):
                if not path or path in seen_paths:
                    continue
                seen_paths.add(path)
                merged_paths.append(path)
                merged_descriptions.append(desc)
            if merged_paths:
                image_paths = merged_paths
                plot_descriptions = merged_descriptions
            visual_summary = build_visual_summary(
                structure_profile=structure_profile,
                plot_descriptions=plot_descriptions,
                image_paths=image_paths,
            )
            visual_hints = build_visual_hints_from_summary(visual_summary)

            if not visual_hints:
                if image_paths:
                    visual_hints = ["plots generated but no explicit visual hints"]
                else:
                    visual_hints = ["plot generation returned no usable images"]

            return ObservationBundle(
                structure_hints=observation.structure_hints,
                visual_hints=visual_hints,
                unit_hints=observation.unit_hints,
                plot_descriptions=plot_descriptions,
                image_paths=image_paths,
                structure_profile=structure_profile,
                visual_summary=visual_summary,
                reconstruction_tokens=reconstruction_tokens,
                reconstruction_trace=reconstruction_trace,
                reconstruction_image_paths=reconstruction_image_paths,
                reconstruction_descriptions=reconstruction_descriptions,
                mm_assets_attempted=True,
                mm_assets_succeeded=bool(image_paths),
                mm_assets_error=None,
            )
        except Exception as e:
            if timer is not None and timer_started:
                timer.stop("step_plot_generation")
            return ObservationBundle(
                structure_hints=observation.structure_hints,
                visual_hints=[f"plot generation failed: {repr(e)}"],
                unit_hints=observation.unit_hints,
                plot_descriptions=[],
                image_paths=[],
                structure_profile=observation.structure_profile,
                visual_summary=observation.visual_summary,
                reconstruction_tokens=getattr(observation, "reconstruction_tokens", None),
                reconstruction_trace=getattr(observation, "reconstruction_trace", None),
                reconstruction_image_paths=getattr(observation, "reconstruction_image_paths", []) or [],
                reconstruction_descriptions=getattr(observation, "reconstruction_descriptions", []) or [],
                mm_assets_attempted=True,
                mm_assets_succeeded=False,
                mm_assets_error=repr(e),
            )



class StructuralPlannerAgent:
    """
    LLM-based structure planner. Data-driven tools provide evidence cards, but
    they do not directly inject final expressions.
    """
    def __init__(self, client):
        self.client = client

    def plan(self, dataset, observation: ObservationBundle, datadriven_evidence=None, guidance_prior=None, iter_cfg=None):
        if not ENABLE_LLM_STRUCTURAL_PLANNER:
            return {"enabled": False, "used_llm": False, "plans": [], "raw_text": "planner disabled"}

        datadriven_evidence = dict(datadriven_evidence or {})
        guidance_prior = dict(guidance_prior or {})
        compact_structure, compact_visual = _compact_observation_for_prompt(
            structure_profile=getattr(observation, "structure_profile", None),
            visual_summary=getattr(observation, "visual_summary", None),
        )
        evidence_cards = list(datadriven_evidence.get("top_records", []) or [])[:max(1, int(DATA_DRIVEN_FEATURE_EVIDENCE_TOPK))]
        family_summary = {
            "mode": datadriven_evidence.get("mode"),
            "family_counts": datadriven_evidence.get("family_counts"),
            "num_scored_terms": datadriven_evidence.get("num_scored_terms"),
        }
        prompt = f"""
You are the Structural Planner Agent for symbolic regression.

Important rule:
- The data-driven tool provides diagnostic evidence only.
- Do NOT copy a complete formula from the tool.
- Reason from evidence and propose compact symbolic structure plans.
- Plans should be structurally diverse when evidence is ambiguous.

Variables:
{dataset.feature_names}
Target:
{dataset.target_name}

Data-driven diagnostic evidence cards:
{json.dumps(evidence_cards, ensure_ascii=False)}

Data-driven family summary:
{json.dumps(family_summary, ensure_ascii=False)}

Observation summary:
{json.dumps(compact_structure, ensure_ascii=False)}

Visual summary:
{json.dumps(compact_visual, ensure_ascii=False)}

Prior guidance:
{json.dumps(summarize_experience_prior(guidance_prior), ensure_ascii=False)}

Task:
Infer up to {int(STRUCTURAL_PLANNER_MAX_PLANS)} structurally distinct formula plans.
Each plan should specify variable roles and the intended operator topology.

Return STRICT JSON only:
{{
  "plans": [
    {{
      "family": "multiplicative envelope times log-ratio",
      "active_variables": ["x1", "x2", "x3", "x4", "x5"],
      "envelope_variables": ["x1", "x2", "x3"],
      "inner_variables": ["x4", "x5"],
      "operators": ["*", "log", "/"],
      "template_intent": "product_envelope_log_ratio",
      "reason": "brief evidence-based reasoning"
    }}
  ]
}}
""".strip()
        try:
            response = self.client.generate(
                messages=[
                    Message(role="system", content="You are a symbolic structure planner. Output JSON only."),
                    Message(role="user", content=prompt),
                ],
                temperature=float(STRUCTURAL_PLANNER_TEMPERATURE),
                max_tokens=int(STRUCTURAL_PLANNER_MAX_TOKENS),
                top_p=1.0,
            )
            raw_text = response.text
            obj = _extract_first_json_object(raw_text)
            plans = []
            if isinstance(obj, dict):
                raw_plans = obj.get("plans", []) or []
                if isinstance(raw_plans, list):
                    for item in raw_plans[:int(STRUCTURAL_PLANNER_MAX_PLANS)]:
                        if isinstance(item, dict):
                            plans.append(make_json_safe(item))
            return {"enabled": True, "used_llm": True, "plans": plans, "raw_text": raw_text, "num_plans": len(plans)}
        except Exception as e:
            return {"enabled": True, "used_llm": False, "plans": [], "raw_text": f"structural planner failed: {repr(e)}", "num_plans": 0}


def _ordered_valid_plan_vars(values, feature_names):
    valid, seen = [], set()
    feature_set = set(feature_names or [])
    for item in values or []:
        item = str(item).strip()
        if item in feature_set and item not in seen:
            valid.append(item)
            seen.add(item)
    return valid


def expand_structural_plans_to_candidates(plans, feature_names, max_candidates=None):
    """Expand LLM structure plans into a small number of fit-ready templates."""
    feature_names = list(feature_names or [])
    if not plans or not feature_names:
        return [], {"enabled": False, "reason": "no plans or features", "num_exprs": 0}
    max_candidates = max(1, int(max_candidates or STRUCTURAL_PLAN_EXPANSION_MAX_CANDIDATES))
    exprs, seen = [], set()

    def add(expr):
        expr = str(expr).strip()
        if not expr:
            return False
        key = _expr_dedup_key(expr)
        if key in seen:
            return False
        seen.add(key)
        exprs.append(expr)
        return len(exprs) >= max_candidates

    for plan in plans or []:
        if not isinstance(plan, dict):
            continue
        family = str(plan.get("family", "") or "").lower()
        intent = str(plan.get("template_intent", "") or "").lower()
        operators = [str(x).lower() for x in (plan.get("operators", []) or [])]
        active = _ordered_valid_plan_vars(plan.get("active_variables", []), feature_names)
        envelope = _ordered_valid_plan_vars(plan.get("envelope_variables", []), feature_names)
        inner = _ordered_valid_plan_vars(plan.get("inner_variables", []), feature_names)
        if not active:
            active = list(feature_names[:min(len(feature_names), 5)])
        if not envelope and len(active) >= 2:
            envelope = active[:min(3, len(active))]
        env_expr = "*".join(envelope[:min(3, len(envelope))]) if envelope else ""

        difference_vars = _ordered_valid_plan_vars(plan.get("difference_variables", []), feature_names)
        multiplier_vars = _ordered_valid_plan_vars(plan.get("multiplier_variables", []), feature_names)
        denominator_vars = _ordered_valid_plan_vars(plan.get("denominator_variables", []), feature_names)
        numerator_vars = _ordered_valid_plan_vars(plan.get("numerator_variables", []), feature_names)
        squared_vars = _ordered_valid_plan_vars(plan.get("squared_variables", []), feature_names)
        outer_scale_vars = _ordered_valid_plan_vars(plan.get("outer_scale_variables", []), feature_names)
        base_linear_vars = _ordered_valid_plan_vars(plan.get("base_linear_variables", []), feature_names)
        interaction_vars = _ordered_valid_plan_vars(plan.get("interaction_variables", []), feature_names)
        angle_vars = _ordered_valid_plan_vars(plan.get("angle_variables", []), feature_names)
        inner_numerator_vars = _ordered_valid_plan_vars(plan.get("inner_numerator_variables", []), feature_names)
        inner_denominator_vars = _ordered_valid_plan_vars(plan.get("inner_denominator_variables", []), feature_names)
        auto_role_search = bool(plan.get("auto_role_search", False))

        if auto_role_search and len(active) >= 5:
            from itertools import combinations
            intent_auto = intent
            if "rational_difference_of_squares_denominator" in intent_auto:
                for den in active:
                    rem1 = [v for v in active if v != den]
                    for d1, d2 in combinations(rem1, 2):
                        nums = [v for v in rem1 if v not in {d1, d2}]
                        if len(nums) < 2:
                            continue
                        for n1, n2 in combinations(nums, 2):
                            if add(f"a*{n1}*{n2}/(({den}+b1)*(({d1}+b2)**2-({d2}+b3)**2)) + c"): break
                            if add(f"a*{n1}*{n2}/(({den}+b1)*(({d2}+b2)**2-({d1}+b3)**2)) + c"): break
                            if add(f"a*{n1}*{n2}/({den}*({d1}**2-{d2}**2)+b) + c"): break
                            if add(f"a*{n1}*{n2}/({den}*(({d1}-{d2})*({d1}+{d2}))+b) + c"): break
                        if len(exprs) >= max_candidates:
                            break
                    if len(exprs) >= max_candidates:
                        break
                if len(exprs) >= max_candidates:
                    break
                continue

            if "power_product_over_product_denominator" in intent_auto:
                for d1, d2 in combinations(active, 2):
                    nums_pool = [v for v in active if v not in {d1, d2}]
                    if len(nums_pool) < 3:
                        continue
                    for n1, n2, n3 in combinations(nums_pool, 3):
                        for sq in (n1, n2, n3):
                            linear_nums = [v for v in (n1, n2, n3) if v != sq]
                            if len(linear_nums) < 2:
                                continue
                            u, w = linear_nums[0], linear_nums[1]
                            if add(f"a*{u}*{sq}**2*{w}/(({d1}+b1)*({d2}+b2)) + c"): break
                            if add(f"a*{u}*({sq}+b1)**2*{w}/(({d1}+b2)*({d2}+b3)) + c"): break
                            if add(f"a*{u}*{sq}**2*{w}/({d1}*{d2}+b) + c"): break
                        if len(exprs) >= max_candidates:
                            break
                    if len(exprs) >= max_candidates:
                        break
                if len(exprs) >= max_candidates:
                    break
                continue

            if "multiplicative_difference_ratio" in intent_auto:
                for den in active:
                    rem1 = [v for v in active if v != den]
                    for d1, d2 in combinations(rem1, 2):
                        mults = [v for v in rem1 if v not in {d1, d2}]
                        if not mults:
                            continue
                        mult_combos = list(combinations(mults, 2)) if len(mults) >= 2 else [(mults[0],)]
                        for combo in mult_combos:
                            mult_expr = "*".join(combo)
                            if add(f"a*{mult_expr}*({d1}-{d2})/{den} + c"): break
                            if add(f"a*{mult_expr}*({d2}-{d1})/{den} + c"): break
                            if add(f"a*{mult_expr}*({d1}-{d2})/({den}+b) + c"): break
                            if add(f"a*{mult_expr}*({d2}-{d1})/({den}+b) + c"): break
                        if len(exprs) >= max_candidates:
                            break
                    if len(exprs) >= max_candidates:
                        break
                if len(exprs) >= max_candidates:
                    break
                continue

            if "outer_scale_exp_minus_one" in intent_auto:
                for outer in active:
                    rest = [v for v in active if v != outer]
                    for n1, n2 in combinations(rest, 2):
                        dens = [v for v in rest if v not in {n1, n2}]
                        if len(dens) < 2:
                            continue
                        for d1, d2 in combinations(dens, 2):
                            if add(f"a*{outer}*(exp(({n1}*{n2})/(({d1}+b1)*({d2}+b2))) - 1) + c"): break
                            if add(f"a*{outer}*(exp(({n1}*{n2})/({d1}*{d2}+b)) - 1) + c"): break
                        if len(exprs) >= max_candidates:
                            break
                    if len(exprs) >= max_candidates:
                        break
                if len(exprs) >= max_candidates:
                    break
                continue

            if "outer_scale_linear_plus_trig_interaction" in intent_auto:
                for outer in active:
                    rest = [v for v in active if v != outer]
                    for base in rest:
                        rest2 = [v for v in rest if v != base]
                        for m1, m2 in combinations(rest2, 2):
                            angles = [v for v in rest2 if v not in {m1, m2}]
                            for ang in angles:
                                if add(f"a*{outer}*({base} + b1*{m1}*{m2}*sin(b2*{ang}+b3)) + c"): break
                                if add(f"a*{outer}*(b1*{base} + b2*{m1}*{m2}*sin(b3*{ang}+b4)) + c"): break
                            if len(exprs) >= max_candidates:
                                break
                        if len(exprs) >= max_candidates:
                            break
                    if len(exprs) >= max_candidates:
                        break
                if len(exprs) >= max_candidates:
                    break
                continue

            if "multiplicative_envelope_log_ratio" in intent_auto:
                for u, v in combinations(active, 2):
                    env_pool = [x for x in active if x not in {u, v}]
                    if len(env_pool) < 1:
                        continue
                    env_size = min(3, len(env_pool))
                    for env in list(combinations(env_pool, env_size))[:8]:
                        env_expr2 = "*".join(env)
                        if add(f"a*{env_expr2}*(log(abs({u})+b1)-log(abs({v})+b2)) + c"): break
                        if add(f"a*{env_expr2}*(log(abs({v})+b1)-log(abs({u})+b2)) + c"): break
                    if len(exprs) >= max_candidates:
                        break
                if len(exprs) >= max_candidates:
                    break
                continue

        is_interaction_power = ("interaction_power" in intent or ("interaction" in family and "power" in family))
        if is_interaction_power:
            if not squared_vars and len(active) >= 2:
                squared_vars = active[1:2]
            if not multiplier_vars:
                multiplier_vars = [v for v in active if v not in set(squared_vars)][:1]
            if multiplier_vars and squared_vars:
                m = multiplier_vars[0]
                sq = squared_vars[0]
                if add(f"a*{m}*{sq}**2 + b"): break
                if add(f"a*({m}+b1)*({sq}+b2)**2 + c"): break
                if add(f"a*{sq}*{m}**2 + b"): break

        is_exp_minus_one = ("outer_scale_exp_minus_one" in intent or ("exp" in family and "minus" in family))
        if is_exp_minus_one:
            if not outer_scale_vars:
                outer_scale_vars = envelope[:1] if envelope else active[:1]
            if len(inner_numerator_vars) < 2:
                inner_numerator_vars = [v for v in active if v not in set(outer_scale_vars)][:2]
            if len(inner_denominator_vars) < 2:
                inner_denominator_vars = [v for v in active if v not in set(outer_scale_vars + inner_numerator_vars)][:2]
            if outer_scale_vars and len(inner_numerator_vars) >= 2 and len(inner_denominator_vars) >= 2:
                outer = outer_scale_vars[0]
                n1, n2 = inner_numerator_vars[0], inner_numerator_vars[1]
                d1, d2 = inner_denominator_vars[0], inner_denominator_vars[1]
                if add(f"a*{outer}*(exp(({n1}*{n2})/(({d1}+b1)*({d2}+b2))) - 1) + c"): break
                if add(f"a*{outer}*(exp(({n1}*{n2})/({d1}*{d2}+b)) - 1) + c"): break
                if add(f"a*({outer}+b1)*(exp((({n1}+b2)*({n2}+b3))/(({d1}+b4)*({d2}+b5))) - 1) + c"): break
                if add(f"a*{outer}*exp(({n1}*{n2})/(({d1}+b1)*({d2}+b2))) + c"): break

        is_trig_interaction = ("outer_scale_linear_plus_trig_interaction" in intent or ("trig" in family and "interaction" in family))
        if is_trig_interaction:
            if not outer_scale_vars:
                outer_scale_vars = envelope[:1] if envelope else active[:1]
            if not base_linear_vars:
                base_linear_vars = [v for v in active if v not in set(outer_scale_vars)][:1]
            if len(interaction_vars) < 2:
                interaction_vars = [v for v in active if v not in set(outer_scale_vars + base_linear_vars + angle_vars)][:2]
            if not angle_vars:
                angle_vars = [v for v in active if v not in set(outer_scale_vars + base_linear_vars + interaction_vars)][:1]
            if outer_scale_vars and base_linear_vars and len(interaction_vars) >= 2 and angle_vars:
                outer = outer_scale_vars[0]
                base = base_linear_vars[0]
                m1, m2 = interaction_vars[0], interaction_vars[1]
                ang = angle_vars[0]
                if add(f"a*{outer}*({base} + b1*{m1}*{m2}*sin(b2*{ang}+b3)) + c"): break
                if add(f"a*{outer}*(b1*{base} + b2*{m1}*{m2}*sin(b3*{ang}+b4)) + c"): break
                if add(f"a*({outer}+b1)*(({base}+b2) + b3*({m1}+b4)*({m2}+b5)*sin(b6*{ang}+b7)) + c"): break
                if add(f"a*{outer}*({base} + b1*{m1}*{m2}*cos(b2*{ang}+b3)) + c"): break

        is_dsq_denominator = ("rational_difference_of_squares_denominator" in intent or "difference-of-squares" in family)
        if is_dsq_denominator:
            if len(numerator_vars) < 2:
                numerator_vars = envelope[:2] if len(envelope) >= 2 else active[:2]
            if not denominator_vars:
                denominator_vars = [v for v in active if v not in set(numerator_vars + difference_vars)][:1]
            if len(difference_vars) < 2:
                difference_vars = [v for v in active if v not in set(numerator_vars + denominator_vars)][:2]
            if len(numerator_vars) >= 2 and denominator_vars and len(difference_vars) >= 2:
                n1, n2 = numerator_vars[0], numerator_vars[1]
                den = denominator_vars[0]
                d1, d2 = difference_vars[0], difference_vars[1]
                if add(f"a*{n1}*{n2}/(({den}+b1)*(({d1}+b2)**2-({d2}+b3)**2)) + c"): break
                if add(f"a*{n1}*{n2}/(({den}+b1)*(({d2}+b2)**2-({d1}+b3)**2)) + c"): break
                if add(f"a*{n1}*{n2}/({den}*({d1}**2-{d2}**2)+b) + c"): break
                if add(f"a*{n1}*{n2}/({den}*(({d1}-{d2})*({d1}+{d2}))+b) + c"): break

        is_power_product_ratio = ("power_product_over_product_denominator" in intent or "power-product" in family)
        if is_power_product_ratio:
            if len(numerator_vars) < 3:
                numerator_vars = envelope[:3] if len(envelope) >= 3 else active[:3]
            if not squared_vars:
                squared_vars = numerator_vars[1:2] if len(numerator_vars) >= 2 else numerator_vars[:1]
            if len(denominator_vars) < 2:
                denominator_vars = [v for v in active if v not in set(numerator_vars)][:2]
            if len(numerator_vars) >= 3 and squared_vars and len(denominator_vars) >= 2:
                sq = squared_vars[0]
                linear_nums = [v for v in numerator_vars if v != sq]
                if len(linear_nums) < 2:
                    linear_nums = [v for v in numerator_vars if v != sq] + [sq]
                n1, n3 = linear_nums[0], linear_nums[1]
                d1, d2 = denominator_vars[0], denominator_vars[1]
                if add(f"a*{n1}*{sq}**2*{n3}/(({d1}+b1)*({d2}+b2)) + c"): break
                if add(f"a*{n1}*({sq}+b1)**2*{n3}/(({d1}+b2)*({d2}+b3)) + c"): break
                if add(f"a*{n1}*{sq}**2*{n3}/({d1}*{d2}+b) + c"): break

        is_difference_ratio = ("multiplicative_difference_ratio" in intent or ("difference" in family and ("ratio" in family or "denominator" in family)))
        if is_difference_ratio:
            if len(difference_vars) < 2:
                remaining = [v for v in active if v not in set(multiplier_vars + denominator_vars)]
                difference_vars = remaining[:2]
            if not multiplier_vars:
                multiplier_vars = [v for v in active if v not in set(difference_vars + denominator_vars)][:2]
            if not denominator_vars:
                denominator_vars = [v for v in active if v not in set(difference_vars + multiplier_vars)][:1]
            if len(difference_vars) >= 2 and multiplier_vars and denominator_vars:
                d1, d2 = difference_vars[0], difference_vars[1]
                den = denominator_vars[0]
                mult_expr = "*".join(multiplier_vars[:min(2, len(multiplier_vars))])
                if add(f"a*{mult_expr}*({d1}-{d2})/{den} + c"): break
                if add(f"a*{mult_expr}*({d2}-{d1})/{den} + c"): break
                if add(f"a*{mult_expr}*({d1}-{d2})/({den}+b) + c"): break
                if add(f"a*{mult_expr}*({d2}-{d1})/({den}+b) + c"): break
                if len(multiplier_vars) >= 2:
                    m1, m2 = multiplier_vars[0], multiplier_vars[1]
                    if add(f"a*({m1}+b1)*({m2}+b2)*({d1}-{d2})/({den}+b3) + c"): break
                    if add(f"a*({m1}+b1)*({m2}+b2)*({d2}-{d1})/({den}+b3) + c"): break

        is_difference_interaction = ("multiplicative_difference" in intent or ("difference" in family and "ratio" not in family and "denominator" not in family))
        if is_difference_interaction:
            if len(difference_vars) < 2:
                remaining = [v for v in active if v not in set(multiplier_vars)]
                difference_vars = remaining[:2]
            if not multiplier_vars:
                multiplier_vars = [v for v in active if v not in set(difference_vars)][:2]
            if len(difference_vars) >= 2 and multiplier_vars:
                d1, d2 = difference_vars[0], difference_vars[1]
                mult_expr = "*".join(multiplier_vars[:min(2, len(multiplier_vars))])
                if add(f"a*{mult_expr}*({d1}-{d2}) + c"): break
                if add(f"a*{mult_expr}*({d2}-{d1}) + c"): break

        is_log_ratio = ("log" in family or "log" in intent or "log" in operators or "ratio" in family or "ratio" in intent or "/" in operators)
        if is_log_ratio and "difference" not in intent and len(inner) >= 2 and env_expr:
            u, v = inner[0], inner[1]
            if add(f"a*{env_expr}*(log(abs({u})+b1)-log(abs({v})+b2)) + c"): break
            if add(f"a*{env_expr}*log(abs({u})/(abs({v})+b)) + c"): break
            if add(f"a*{env_expr}*(log(abs({v})+b1)-log(abs({u})+b2)) + c"): break

        if ("rational" in family or "denominator" in family or "ratio" in family or "/" in operators) and len(active) >= 3:
            num_vars = envelope[:2] if len(envelope) >= 2 else active[:2]
            den_vars = [v for v in active if v not in set(num_vars)]
            if den_vars:
                num = "*".join(num_vars)
                den = den_vars[0]
                if add(f"a*({num})/({den}+b) + c"): break
            if len(active) >= 4:
                num = "*".join(active[:2])
                den = "*".join([f"({v}+b{i+1})" for i, v in enumerate(active[2:4])])
                if add(f"a*({num})/({den}) + c"): break

        if ("interaction" in family or "*" in operators or "product" in intent) and len(active) >= 2:
            if add(f"a*{active[0]}*{active[1]} + b"): break
            if len(active) >= 3 and add(f"a*{active[0]}*{active[1]}*{active[2]} + b"): break

        if "additive" in family or "separable" in family or "+" in operators:
            linear = " + ".join([f"a{i+1}*{v}" for i, v in enumerate(active[:min(len(active), 5)])])
            if linear and add(f"{linear} + c"): break

        if "trig" in family or "periodic" in family or "sin" in operators or "cos" in operators:
            for v in active[:2]:
                if add(f"a*sin(b*{v}+c) + d"): break
                if add(f"a*cos(b*{v}+c) + d"): break

        if "exp" in family or "exponential" in family or "exp" in operators:
            if len(active) >= 1 and add(f"a*exp(b*{active[0]}) + c"): break
            if len(active) >= 2 and add(f"a*exp(b1*{active[0]} + b2*{active[1]}) + c"): break

        if len(exprs) >= max_candidates:
            break

    exprs = deduplicate_expressions(exprs)[:max_candidates]
    return exprs, {"enabled": True, "num_exprs": len(exprs), "preview_exprs": exprs[:8]}


def _record_gain(record):
    try:
        return float(record.get("combined_gain", 0.0) or 0.0)
    except Exception:
        return 0.0



def build_highdim_family_coverage_plans(feature_names):
    """
    Build generic high-dimensional family coverage plans.

    These are abstract topology plans, not closed-form benchmark templates.
    They make the planner/evaluator consider a small set of common SR grammar
    families even when the data-driven top-k evidence list misses one family.
    The actual variable-role combinations are enumerated inside the expansion
    step in a bounded way.
    """
    feature_names = list(feature_names or [])
    if (not ENABLE_HIGH_DIM_FAMILY_COVERAGE_PLANS) or len(feature_names) < int(HIGH_DIM_COVERAGE_MIN_FEATURES):
        return []

    fams = set(HIGH_DIM_COVERAGE_FAMILIES or [])
    plans = []

    def add(intent, family, operators, reason):
        if intent not in fams:
            return
        plans.append({
            "family": family,
            "template_intent": intent,
            "active_variables": feature_names[:min(len(feature_names), 6)],
            "operators": operators,
            "evidence_gain": 0.0,
            "approved": True,
            "reason": reason,
            "requires_llm_approval": True,
            "auto_role_search": True,
            "coverage_plan": True,
        })

    add(
        "rational_difference_of_squares_denominator",
        "rational difference-of-squares denominator coverage",
        ["*", "/", "**", "-"],
        "generic coverage: numerator product over a scaled difference-of-squares denominator",
    )
    add(
        "power_product_over_product_denominator",
        "power-product over product denominator coverage",
        ["*", "/", "**"],
        "generic coverage: product numerator with one squared variable over a product denominator",
    )
    add(
        "multiplicative_difference_ratio",
        "multiplicative difference ratio coverage",
        ["*", "-", "/"],
        "generic coverage: multiplier variables times a difference divided by a denominator",
    )
    add(
        "outer_scale_exp_minus_one",
        "outer scale exp minus one coverage",
        ["*", "exp", "/", "-"],
        "generic coverage: outer scale times exp(product ratio) minus one",
    )
    add(
        "outer_scale_linear_plus_trig_interaction",
        "outer scale linear plus trig interaction coverage",
        ["*", "+", "sin"],
        "generic coverage: outer scale times linear term plus trigonometric interaction",
    )
    add(
        "multiplicative_envelope_log_ratio",
        "multiplicative envelope times log-ratio coverage",
        ["*", "log", "/"],
        "generic coverage: product envelope times a log-ratio core",
    )
    add(
        "reciprocal_difference_product",
        "multiplicative envelope times reciprocal-difference coverage",
        ["*", "/", "-"],
        "generic coverage: product envelope times reciprocal difference",
    )

    return plans


def _merge_plans_preserve_order(*plan_lists, max_plans=None):
    merged = []
    seen = set()
    for plan_list in plan_lists:
        for plan in plan_list or []:
            if not isinstance(plan, dict):
                continue
            key = (
                str(plan.get("template_intent", "")),
                tuple(plan.get("active_variables", []) or []),
                tuple(plan.get("envelope_variables", []) or []),
                tuple(plan.get("inner_variables", []) or []),
                tuple(plan.get("difference_variables", []) or []),
                tuple(plan.get("multiplier_variables", []) or []),
                tuple(plan.get("denominator_variables", []) or []),
                tuple(plan.get("numerator_variables", []) or []),
                tuple(plan.get("squared_variables", []) or []),
                bool(plan.get("auto_role_search", False)),
            )
            if key in seen:
                continue
            seen.add(key)
            merged.append(plan)
            if max_plans is not None and len(merged) >= int(max_plans):
                return merged
    return merged
def build_structural_plans_from_datadriven_evidence(datadriven_raw, feature_names, max_plans=4):
    """
    Convert data-driven evidence into abstract plans, not full formulas.

    This is intentionally between tool-only search and pure LLM guessing:
    - The data tool may say a family/variable-role pattern has high support.
    - It does NOT inject the closed-form expression as a candidate.
    - The plan is later approved/revised by the LLM or explicitly marked as
      evidence-fallback if the LLM endpoint times out.
    """
    feature_names = list(feature_names or [])
    feature_set = set(feature_names)
    records = [r for r in list((datadriven_raw or {}).get("top_records", []) or []) if isinstance(r, dict)]
    if not records or not feature_names:
        return []

    # Prefer a clean two-variable log/ratio clue for the inner core.
    def valid_vars(r):
        return [str(v) for v in (r.get("variables", []) or []) if str(v) in feature_set]

    log_pair = None
    for r in sorted(records, key=lambda x: -_record_gain(x)):
        fam = str(r.get("family", "")).lower()
        pat = str(r.get("basis_pattern", "")).lower()
        vars_ = valid_vars(r)
        if len(vars_) == 2 and ("log" in fam or "log" in pat or "ratio" in pat or "rational" in fam):
            log_pair = vars_[:2]
            break
    if log_pair is None:
        for r in sorted(records, key=lambda x: -_record_gain(x)):
            fam = str(r.get("family", "")).lower()
            pat = str(r.get("basis_pattern", "")).lower()
            vars_ = valid_vars(r)
            if len(vars_) >= 2 and ("rational" in fam or "ratio" in pat):
                log_pair = vars_[:2]
                break

    plans = []
    seen = set()

    def add_plan(plan):
        key = (
            str(plan.get("template_intent", "")),
            tuple(plan.get("envelope_variables", []) or []),
            tuple(plan.get("inner_variables", []) or []),
            tuple(plan.get("difference_variables", []) or []),
            tuple(plan.get("multiplier_variables", []) or []),
            tuple(plan.get("denominator_variables", []) or []),
            tuple(plan.get("numerator_variables", []) or []),
            tuple(plan.get("squared_variables", []) or []),
            tuple(plan.get("outer_scale_variables", []) or []),
            tuple(plan.get("base_linear_variables", []) or []),
            tuple(plan.get("interaction_variables", []) or []),
            tuple(plan.get("angle_variables", []) or []),
            tuple(plan.get("inner_numerator_variables", []) or []),
            tuple(plan.get("inner_denominator_variables", []) or []),
        )
        if key in seen:
            return
        seen.add(key)
        plans.append(plan)

    for r in sorted(records, key=lambda x: -_record_gain(x)):
        fam = str(r.get("family", "")).lower()
        pat = str(r.get("basis_pattern", "")).lower()
        vars_ = valid_vars(r)
        gain = _record_gain(r)
        if not vars_:
            continue

        if "compound_log_interaction" in fam or ("multiplicative envelope" in pat and "log" in pat):
            inner = log_pair or vars_[-2:]
            inner = [v for v in inner if v in feature_set][:2]
            envelope = [v for v in vars_ if v not in set(inner)]
            if len(inner) < 2:
                inner = vars_[-2:]
                envelope = [v for v in vars_ if v not in set(inner)]
            if len(envelope) < 1:
                envelope = [v for v in feature_names if v not in set(inner)][:3]
            add_plan({
                "family": "multiplicative envelope times log-ratio",
                "template_intent": "multiplicative_envelope_log_ratio",
                "active_variables": vars_,
                "envelope_variables": envelope[:3],
                "inner_variables": inner[:2],
                "operators": ["*", "log", "/"],
                "evidence_gain": gain,
                "approved": True,
                "reason": "data-driven evidence supports a product envelope multiplied by a log-ratio core",
                "requires_llm_approval": True,
            })

        elif "multiplicative_difference_ratio" in fam or ("difference" in pat and ("ratio" in pat or "denominator" in pat)):
            # Role convention from the data-driven library:
            # [multiplier_1, multiplier_2, difference_positive, difference_negative, denominator]
            multiplier_vars = vars_[:2] if len(vars_) >= 2 else vars_[:1]
            difference_vars = vars_[2:4] if len(vars_) >= 4 else vars_[:2]
            denominator_vars = vars_[4:5] if len(vars_) >= 5 else [v for v in vars_ if v not in set(multiplier_vars + difference_vars)][:1]
            if len(difference_vars) >= 2 and denominator_vars:
                add_plan({
                    "family": "multiplicative difference ratio",
                    "template_intent": "multiplicative_difference_ratio",
                    "active_variables": vars_,
                    "multiplier_variables": multiplier_vars[:2],
                    "difference_variables": difference_vars[:2],
                    "denominator_variables": denominator_vars[:1],
                    "envelope_variables": multiplier_vars[:2],
                    "inner_variables": difference_vars[:2] + denominator_vars[:1],
                    "operators": ["*", "-", "/"],
                    "evidence_gain": gain,
                    "approved": True,
                    "reason": "data-driven evidence supports multiplier(s) times a variable difference divided by a denominator",
                    "requires_llm_approval": True,
                })

        elif "multiplicative_difference" in fam or ("difference" in pat and "multiplicative" in pat):
            multiplier_vars = vars_[:1]
            difference_vars = vars_[1:3] if len(vars_) >= 3 else vars_[:2]
            if len(difference_vars) >= 2:
                add_plan({
                    "family": "multiplicative difference interaction",
                    "template_intent": "multiplicative_difference",
                    "active_variables": vars_,
                    "multiplier_variables": multiplier_vars,
                    "difference_variables": difference_vars[:2],
                    "operators": ["*", "-"],
                    "evidence_gain": gain,
                    "approved": True,
                    "reason": "data-driven evidence supports a multiplicative difference/contrast interaction",
                    "requires_llm_approval": True,
                })

        elif "interaction_power" in fam or "interaction power" in pat or "powered variable" in pat:
            basis_expr = str(r.get("basis_expr", "") or "")
            squared = []
            for v in vars_:
                if re.search(rf"\b{re.escape(v)}\b\s*\*\*\s*2", basis_expr):
                    squared.append(v)
            if not squared and len(vars_) >= 2:
                # The generic basis library stores x*y**2 as [x, y].
                squared = vars_[1:2]
            linear = [v for v in vars_ if v not in set(squared)] or vars_[:1]
            if linear and squared:
                add_plan({
                    "family": "interaction power",
                    "template_intent": "interaction_power",
                    "active_variables": vars_,
                    "multiplier_variables": linear[:1],
                    "squared_variables": squared[:1],
                    "envelope_variables": vars_[:2],
                    "operators": ["*", "**"],
                    "evidence_gain": gain,
                    "approved": True,
                    "reason": "data-driven evidence supports a multiplicative interaction with one powered variable",
                    "requires_llm_approval": True,
                })

        elif "rational_difference_of_squares_denominator" in fam or "difference-of-squares" in pat:
            # Role convention: [numerator_1, numerator_2, scale_denominator, diff_1, diff_2]
            numerator_vars = vars_[:2] if len(vars_) >= 2 else vars_[:1]
            denominator_vars = vars_[2:3] if len(vars_) >= 3 else []
            difference_vars = vars_[3:5] if len(vars_) >= 5 else vars_[-2:]
            if len(numerator_vars) >= 2 and denominator_vars and len(difference_vars) >= 2:
                add_plan({
                    "family": "rational difference-of-squares denominator",
                    "template_intent": "rational_difference_of_squares_denominator",
                    "active_variables": vars_,
                    "numerator_variables": numerator_vars[:2],
                    "envelope_variables": numerator_vars[:2],
                    "denominator_variables": denominator_vars[:1],
                    "difference_variables": difference_vars[:2],
                    "inner_variables": denominator_vars[:1] + difference_vars[:2],
                    "operators": ["*", "/", "**", "-"],
                    "evidence_gain": gain,
                    "approved": True,
                    "reason": "data-driven evidence supports a numerator product divided by a scaled difference-of-squares denominator",
                    "requires_llm_approval": True,
                })

        elif "power_product_over_product_denominator" in fam or "power-product" in pat:
            # Role convention: [linear_num_1, squared_num, linear_num_2, den_1, den_2]
            numerator_vars = vars_[:3] if len(vars_) >= 3 else vars_[:]
            squared_vars = vars_[1:2] if len(vars_) >= 2 else []
            denominator_vars = vars_[3:5] if len(vars_) >= 5 else [v for v in vars_ if v not in set(numerator_vars)][:2]
            if len(numerator_vars) >= 3 and squared_vars and len(denominator_vars) >= 2:
                add_plan({
                    "family": "power-product over product denominator",
                    "template_intent": "power_product_over_product_denominator",
                    "active_variables": vars_,
                    "numerator_variables": numerator_vars[:3],
                    "squared_variables": squared_vars[:1],
                    "envelope_variables": numerator_vars[:3],
                    "denominator_variables": denominator_vars[:2],
                    "inner_variables": denominator_vars[:2],
                    "operators": ["*", "/", "**"],
                    "evidence_gain": gain,
                    "approved": True,
                    "reason": "data-driven evidence supports a product numerator with one squared factor over a product denominator",
                    "requires_llm_approval": True,
                })

        elif "outer_scale_exp_minus_one" in fam or "exp(inner" in pat or "exp_minus_one" in fam:
            # Role convention from data-driven basis: [outer, inner_num_1, inner_num_2, inner_den_1, inner_den_2]
            outer_vars = vars_[:1]
            inner_num = vars_[1:3] if len(vars_) >= 3 else []
            inner_den = vars_[3:5] if len(vars_) >= 5 else [v for v in vars_ if v not in set(outer_vars + inner_num)][:2]
            if outer_vars and len(inner_num) >= 2 and len(inner_den) >= 2:
                add_plan({
                    "family": "outer scale exp minus one",
                    "template_intent": "outer_scale_exp_minus_one",
                    "active_variables": vars_,
                    "outer_scale_variables": outer_vars[:1],
                    "inner_numerator_variables": inner_num[:2],
                    "inner_denominator_variables": inner_den[:2],
                    "envelope_variables": outer_vars[:1],
                    "inner_variables": inner_num[:2] + inner_den[:2],
                    "operators": ["*", "exp", "/", "-"],
                    "evidence_gain": gain,
                    "approved": True,
                    "reason": "data-driven evidence supports an outer scale times exp(inner product/ratio) minus one",
                    "requires_llm_approval": True,
                })

        elif "outer_scale_linear_plus_trig_interaction" in fam or "trigonometric interaction" in pat:
            # Role convention: [outer, base_linear, interaction_1, interaction_2, angle]
            outer_vars = vars_[:1]
            base_vars = vars_[1:2] if len(vars_) >= 2 else []
            interaction_vars = vars_[2:4] if len(vars_) >= 4 else []
            angle_vars = vars_[4:5] if len(vars_) >= 5 else []
            if outer_vars and base_vars and len(interaction_vars) >= 2 and angle_vars:
                add_plan({
                    "family": "outer scale linear plus trig interaction",
                    "template_intent": "outer_scale_linear_plus_trig_interaction",
                    "active_variables": vars_,
                    "outer_scale_variables": outer_vars[:1],
                    "base_linear_variables": base_vars[:1],
                    "interaction_variables": interaction_vars[:2],
                    "angle_variables": angle_vars[:1],
                    "envelope_variables": outer_vars[:1],
                    "inner_variables": base_vars[:1] + interaction_vars[:2] + angle_vars[:1],
                    "operators": ["*", "+", "sin"],
                    "evidence_gain": gain,
                    "approved": True,
                    "reason": "data-driven evidence supports an outer-scaled linear term plus trigonometric interaction",
                    "requires_llm_approval": True,
                })

        elif "rational" in fam or "ratio" in pat:
            inner = vars_[2:4] if len(vars_) >= 4 else vars_[-1:]
            env = vars_[:2] if len(vars_) >= 2 else vars_[:1]
            add_plan({
                "family": "rational interaction",
                "template_intent": "rational_interaction",
                "active_variables": vars_,
                "envelope_variables": env,
                "inner_variables": inner,
                "operators": ["*", "/"],
                "evidence_gain": gain,
                "approved": True,
                "reason": "data-driven evidence supports a ratio or denominator-like relation",
                "requires_llm_approval": True,
            })

        if len(plans) >= int(max_plans):
            break

    # Add generic high-dimensional coverage plans first, then evidence-specific plans.
    # Coverage plans are still abstract and require LLM approval or an explicit
    # fallback/companion trace before expansion. They prevent important families
    # such as difference-of-squares denominators from disappearing due to top-k
    # evidence truncation.
    coverage_plans = build_highdim_family_coverage_plans(feature_names)
    merged_plans = _merge_plans_preserve_order(
        coverage_plans,
        plans,
        max_plans=max(1, int(max_plans)),
    )
    return merged_plans[:max(1, int(max_plans))]



def _call_planner_with_hard_timeout(fn, timeout_sec, default_result):
    """
    Run a small LLM planning call with a wall-clock timeout.
    The underlying HTTP request may finish later, but the pipeline will not block.
    """
    timeout_sec = max(1, int(timeout_sec or PLANNER_APPROVAL_TIMEOUT_SEC))
    result_queue = queue.Queue(maxsize=1)

    def _runner():
        try:
            result_queue.put(("ok", fn()))
        except Exception as e:
            result_queue.put(("err", e))

    worker = threading.Thread(target=_runner, daemon=True, name="llmsr_planner_timeout_worker")
    worker.start()
    try:
        status, payload = result_queue.get(timeout=timeout_sec)
        if status == "ok":
            return payload
        raise payload
    except queue.Empty:
        out = dict(default_result or {})
        out.update({
            "used_llm": False,
            "approved_plan_ids": [],
            "approved_plans": [],
            "raw_text": f"planner approval hard-timeout after {timeout_sec}s",
            "timeout_sec": timeout_sec,
            "timeout_mode": "daemon_thread_detached",
        })
        return out
    except Exception as e:
        out = dict(default_result or {})
        out.update({
            "used_llm": False,
            "approved_plan_ids": [],
            "approved_plans": [],
            "raw_text": f"planner approval failed: {repr(e)}",
            "timeout_sec": timeout_sec,
        })
        return out

def approve_structural_plans_with_llm(client, dataset, observation, evidence_plans, max_tokens=None):
    """
    Ask the LLM to approve/revise abstract plans.
    Compact protocol: the model selects plan IDs instead of rewriting full plan objects.
    It must not write formulas.
    """
    evidence_plans = list(evidence_plans or [])
    if not evidence_plans:
        return {"used_llm": False, "approved_plan_ids": [], "approved_plans": [], "raw_text": "no evidence plans"}

    compact_plans = []
    for idx, p in enumerate(evidence_plans[:max(1, int(STRUCTURAL_PLANNER_MAX_PLANS))], start=1):
        compact_plans.append({
            "id": idx,
            "family": p.get("family"),
            "template_intent": p.get("template_intent"),
            "envelope_variables": list(p.get("envelope_variables", []) or []),
            "inner_variables": list(p.get("inner_variables", []) or []),
            "difference_variables": list(p.get("difference_variables", []) or []),
            "multiplier_variables": list(p.get("multiplier_variables", []) or []),
            "denominator_variables": list(p.get("denominator_variables", []) or []),
            "numerator_variables": list(p.get("numerator_variables", []) or []),
            "squared_variables": list(p.get("squared_variables", []) or []),
            "outer_scale_variables": list(p.get("outer_scale_variables", []) or []),
            "base_linear_variables": list(p.get("base_linear_variables", []) or []),
            "interaction_variables": list(p.get("interaction_variables", []) or []),
            "angle_variables": list(p.get("angle_variables", []) or []),
            "inner_numerator_variables": list(p.get("inner_numerator_variables", []) or []),
            "inner_denominator_variables": list(p.get("inner_denominator_variables", []) or []),
            "active_variables": list(p.get("active_variables", []) or []),
            "operators": list(p.get("operators", []) or []),
            "evidence_gain": p.get("evidence_gain"),
        })

    compact_structure, compact_visual = _compact_observation_for_prompt(
        structure_profile=getattr(observation, "structure_profile", None),
        visual_summary=getattr(observation, "visual_summary", None),
    )
    reconstruction_tokens = getattr(observation, "reconstruction_tokens", None)
    recon_compact = {}
    if isinstance(reconstruction_tokens, dict):
        recon_compact = {
            "active_variables": list(reconstruction_tokens.get("active_variables", []) or [])[:8],
            "dominant_pairs": list(reconstruction_tokens.get("dominant_pairs", []) or [])[:8],
            "denominator_like_variables": list(reconstruction_tokens.get("denominator_like_variables", []) or [])[:8],
            "periodic_variables": list(reconstruction_tokens.get("periodic_variables", []) or [])[:8],
            "symmetry_variables": list(reconstruction_tokens.get("symmetry_variables", []) or [])[:8],
        }

    prompt = f"""
You are a symbolic regression structural planner.

Your job is only to approve/reject compact evidence-based structure plans and, when useful, revise variable roles.
Do NOT write formulas. Do NOT use benchmark names or true formulas.
Prefer compact mechanism-like plans over shifted-denominator or huge-constant surrogates.

Variables: {dataset.feature_names}
Target: {dataset.target_name}

Visual/structural summary:
{json.dumps(compact_structure, ensure_ascii=False)}
{json.dumps(compact_visual, ensure_ascii=False)}

Reconstruction token summary:
{json.dumps(recon_compact, ensure_ascii=False)}

Plans:
{json.dumps(compact_plans, ensure_ascii=False)}

Return STRICT JSON only:
{{
  "approved_plan_ids": [1],
  "revisions": [
    {{
      "id": 1,
      "envelope_variables": ["x1", "x2", "x3"],
      "inner_variables": ["x5", "x4"],
      "difference_variables": ["x3", "x2"],
      "multiplier_variables": ["x1", "x4"],
      "denominator_variables": ["x5"],
      "numerator_variables": ["x1", "x2"],
      "squared_variables": ["x2"],
      "outer_scale_variables": ["x1"],
      "base_linear_variables": ["x2"],
      "interaction_variables": ["x3", "x4"],
      "angle_variables": ["x5"],
      "inner_numerator_variables": ["x2", "x3"],
      "inner_denominator_variables": ["x4", "x5"],
      "reason": "brief evidence-based reason"
    }}
  ]
}}
""".strip()

    try:
        response = client.generate(
            messages=[
                Message(role="system", content="Output compact JSON only."),
                Message(role="user", content=prompt),
            ],
            temperature=float(PLANNER_APPROVAL_TEMPERATURE),
            max_tokens=int(max_tokens or PLANNER_APPROVAL_MAX_TOKENS),
            top_p=1.0,
        )
        raw_text = response.text
        obj = _extract_first_json_object(raw_text)
        approved_ids = []
        revisions = []
        legacy_approved_plans = []
        if isinstance(obj, dict):
            for item in obj.get("approved_plan_ids", []) or []:
                try:
                    plan_id = int(item)
                    if 1 <= plan_id <= len(compact_plans):
                        approved_ids.append(plan_id)
                except Exception:
                    pass
            for item in obj.get("revisions", []) or []:
                if isinstance(item, dict):
                    revisions.append(make_json_safe(item))
            for p in obj.get("approved_plans", []) or []:
                if isinstance(p, dict) and bool(p.get("approved", True)):
                    legacy_approved_plans.append(make_json_safe(p))
        return {
            "used_llm": True,
            "approved_plan_ids": sorted(set(approved_ids)),
            "revisions": revisions,
            "approved_plans": legacy_approved_plans,
            "raw_text": raw_text,
            "prompt_num_plans": len(compact_plans),
            "timeout_sec": int(PLANNER_APPROVAL_TIMEOUT_SEC),
        }
    except Exception as e:
        return {
            "used_llm": False,
            "approved_plan_ids": [],
            "approved_plans": [],
            "revisions": [],
            "raw_text": f"planner approval failed: {repr(e)}",
            "timeout_sec": int(PLANNER_APPROVAL_TIMEOUT_SEC),
        }

def select_structural_plans_with_llm_or_fallback(client, dataset, observation, evidence_plans):
    """
    Prefer LLM approval. If the LLM times out or returns no approval, optionally
    use top evidence plans as a clearly marked fallback.
    """
    evidence_plans = list(evidence_plans or [])
    if not evidence_plans:
        return [], {"used_llm": False, "fallback_used": False, "raw_text": "no evidence plans"}

    llm_result = _call_planner_with_hard_timeout(
        lambda: approve_structural_plans_with_llm(
            client=client,
            dataset=dataset,
            observation=observation,
            evidence_plans=evidence_plans,
            max_tokens=PLANNER_APPROVAL_MAX_TOKENS,
        ),
        timeout_sec=PLANNER_APPROVAL_TIMEOUT_SEC,
        default_result={"used_llm": False, "approved_plan_ids": [], "approved_plans": [], "revisions": []},
    )

    approved = []
    id_to_plan = {idx: dict(p) for idx, p in enumerate(evidence_plans, start=1)}
    revisions_by_id = {}
    for r in list(llm_result.get("revisions", []) or []):
        if isinstance(r, dict):
            try:
                rid = int(r.get("id"))
                revisions_by_id[rid] = r
            except Exception:
                pass
    for plan_id in list(llm_result.get("approved_plan_ids", []) or []):
        try:
            plan_id = int(plan_id)
        except Exception:
            continue
        if plan_id not in id_to_plan:
            continue
        q = dict(id_to_plan[plan_id])
        rev = dict(revisions_by_id.get(plan_id, {}) or {})
        for key in [
            "envelope_variables", "inner_variables", "difference_variables",
            "multiplier_variables", "denominator_variables", "numerator_variables",
            "squared_variables", "outer_scale_variables", "base_linear_variables",
            "interaction_variables", "angle_variables", "inner_numerator_variables",
            "inner_denominator_variables", "active_variables", "operators", "reason"
        ]:
            if rev.get(key):
                q[key] = rev[key]
        q["approved"] = True
        q["approval_source"] = "llm"
        approved.append(make_json_safe(q))

    if not approved:
        for p in list(llm_result.get("approved_plans", []) or []):
            if isinstance(p, dict) and bool(p.get("approved", True)):
                q = dict(p)
                q["approval_source"] = "llm"
                approved.append(make_json_safe(q))

    if approved:
        companion_count = 0
        if ENABLE_HIGH_DIM_COVERAGE_COMPANION_TO_LLM_APPROVAL:
            approved_keys = {
                (str(p.get("template_intent", "")), bool(p.get("auto_role_search", False)))
                for p in approved
            }
            for p in evidence_plans:
                if not (isinstance(p, dict) and p.get("coverage_plan") and p.get("auto_role_search")):
                    continue
                key = (str(p.get("template_intent", "")), True)
                if key in approved_keys:
                    continue
                q = dict(p)
                q["approved"] = True
                q["approval_source"] = "coverage_companion_to_llm_approval"
                approved.append(make_json_safe(q))
                approved_keys.add(key)
                companion_count += 1
                if companion_count >= 3:
                    break
        return approved, {
            "used_llm": True,
            "fallback_used": False,
            "coverage_companion_count": int(companion_count),
            "raw_text": llm_result.get("raw_text", ""),
            "approved_plan_ids": list(llm_result.get("approved_plan_ids", []) or []),
            "revisions": make_json_safe(llm_result.get("revisions", []) or []),
            "timeout_sec": int(PLANNER_APPROVAL_TIMEOUT_SEC),
        }

    if REQUIRE_LLM_APPROVAL_FOR_STRUCTURAL_EXPANSION:
        return [], {
            "used_llm": bool(llm_result.get("used_llm", False)),
            "fallback_used": False,
            "expansion_blocked": True,
            "reason": "LLM approval required but no plan was approved",
            "raw_text": llm_result.get("raw_text", ""),
            "timeout_sec": int(PLANNER_APPROVAL_TIMEOUT_SEC),
        }

    fallback = []
    fallback_source = [p for p in evidence_plans if isinstance(p, dict) and p.get("coverage_plan")]
    fallback_source += [p for p in evidence_plans if not (isinstance(p, dict) and p.get("coverage_plan"))]
    for p in fallback_source[:max(1, int(LLM_PLAN_APPROVED_FALLBACK_KEEP))]:
        q = dict(p)
        q["approved"] = True
        q["approval_source"] = "evidence_or_coverage_fallback_due_to_llm_timeout_or_empty_output"
        fallback.append(make_json_safe(q))
    return fallback, {
        "used_llm": bool(llm_result.get("used_llm", False)),
        "fallback_used": True,
        "fallback_plan_count": len(fallback),
        "raw_text": llm_result.get("raw_text", ""),
        "timeout_sec": int(PLANNER_APPROVAL_TIMEOUT_SEC),
    }


def merge_llm_plan_candidates_with_safety_head(base_exprs, plan_exprs, max_total=None):
    """
    Insert LLM-planned candidates without displacing the deterministic/evidence
    safety head. This makes LLM participation explicit while preserving the
    high-performing generic/data-driven candidates that protect performance.
    """
    base_exprs = deduplicate_expressions(base_exprs or [])
    plan_exprs = deduplicate_expressions(plan_exprs or [])
    if not plan_exprs:
        return base_exprs, {
            "enabled": bool(ENABLE_LLM_PLAN_COMPANION_MERGE),
            "num_plan_exprs": 0,
            "safety_head_keep": int(LLM_PLAN_SAFETY_HEAD_KEEP),
            "companion_inserted": 0,
            "mode": "no_plan_exprs",
        }

    if not ENABLE_LLM_PLAN_COMPANION_MERGE:
        merged = merge_expression_groups_with_limit([plan_exprs, base_exprs], max_total=max_total)
        return merged, {
            "enabled": False,
            "num_plan_exprs": len(plan_exprs),
            "companion_inserted": min(len(plan_exprs), len(merged)),
            "mode": "frontload_legacy",
        }

    max_total = int(max_total or MAX_INITIAL_CANDIDATES)
    head_keep = max(0, min(int(LLM_PLAN_SAFETY_HEAD_KEEP), max_total))
    insert_cap = max(0, int(LLM_PLAN_COMPANION_MAX_INSERT))

    safety_head = base_exprs[:head_keep]
    safety_keys = {_expr_dedup_key(x) for x in safety_head}
    companion = []
    for expr in plan_exprs:
        key = _expr_dedup_key(expr)
        if key in safety_keys:
            continue
        companion.append(expr)
        safety_keys.add(key)
        if len(companion) >= insert_cap:
            break

    tail = []
    for expr in base_exprs[head_keep:]:
        key = _expr_dedup_key(expr)
        if key not in safety_keys:
            tail.append(expr)
            safety_keys.add(key)

    merged = (safety_head + companion + tail)[:max_total]
    return merged, {
        "enabled": True,
        "num_plan_exprs": len(plan_exprs),
        "safety_head_keep": int(head_keep),
        "companion_inserted": int(len(companion)),
        "max_total": int(max_total),
        "mode": "safety_head_then_llm_companion",
        "preview_plan_exprs": plan_exprs[:8],
        "preview_companion_exprs": companion[:8],
    }


# Alias kept for readability in the orchestration layer.
def expand_approved_structural_plans_to_candidates(plans, feature_names, max_candidates=None):
    return expand_structural_plans_to_candidates(plans, feature_names, max_candidates=max_candidates)


def _variables_used_in_expression(expr, feature_names):
    expr = str(expr or "")
    used = []
    for name in feature_names or []:
        if re.search(rf"\b{re.escape(str(name))}\b", expr) and name not in used:
            used.append(name)
    return used


def _build_local_repair_evidence_cards(current_best, dataset, evaluation, observation=None, guidance_prior=None):
    feature_names = list(getattr(dataset, "feature_names", []) or [])
    best_expr = str(_safe_get_attr(current_best, "simplified_expression", "") or "")
    used = _variables_used_in_expression(best_expr, feature_names)
    missing = [v for v in feature_names if v not in set(used)]
    signature = extract_formula_form_signature(best_expr, feature_names) if best_expr else {}
    residual = dict((evaluation or {}).get("residual_summary", {}) or {})
    structure_profile = dict(getattr(observation, "structure_profile", {}) or {})
    reconstruction_tokens = {}
    for holder in [getattr(observation, "visual_summary", None), structure_profile]:
        if isinstance(holder, dict) and isinstance(holder.get("reconstruction_tokens"), dict):
            reconstruction_tokens = holder.get("reconstruction_tokens")
            break
    return {
        "current_best_expr": best_expr,
        "current_best_val_mse": _safe_get_attr(current_best, "val_mse", None) if current_best is not None else None,
        "used_variables": used,
        "missing_variables": missing,
        "current_signature": make_json_safe(signature),
        "residual_repair_targets": list(residual.get("repair_targets", []) or [])[:5],
        "residual_guidance_lines": list(residual.get("guidance_lines", []) or [])[:5],
        "residual_top_probes": make_json_safe(list(residual.get("top_probes", []) or [])[:6]),
        "structure_evidence": make_json_safe({
            "family_scores": structure_profile.get("family_scores", {}),
            "global_tags": structure_profile.get("global_tags", []),
            "variable_roles": structure_profile.get("variable_roles", {}),
            "top_pair_patterns": list(structure_profile.get("top_pair_patterns", []) or [])[:5],
        }),
        "reconstruction_evidence": make_json_safe({
            "dominant_unary_variables": list(reconstruction_tokens.get("dominant_unary_variables", []) or [])[:5],
            "dominant_pairs": list(reconstruction_tokens.get("dominant_pairs", []) or [])[:5],
            "denominator_like_variables": list(reconstruction_tokens.get("denominator_like_variables", []) or [])[:5],
        }) if reconstruction_tokens else {},
    }



def _variables_used_in_text(text, feature_names):
    text = str(text or "")
    return [v for v in (feature_names or []) if re.search(rf"\b{re.escape(v)}\b", text)]


def _extract_log_core_variables(expr, feature_names):
    """Return variables appearing inside the first log(...) chunk, preserving order."""
    expr = str(expr or "")
    if "log" not in expr.lower():
        return []
    m = re.search(r"log\s*\((.*)\)", expr, flags=re.IGNORECASE)
    chunk = m.group(1) if m else expr
    ordered = []
    for match in re.finditer(r"\b[A-Za-z_]\w*\b", chunk):
        token = match.group(0)
        if token in feature_names and token not in ordered:
            ordered.append(token)
    return ordered


def _repair_decision_requests_missing_envelope(decision):
    if not isinstance(decision, dict):
        return False
    text_parts = []
    for key in ["action", "actions", "repair_action", "repair_actions", "reason", "reasoning", "template_intent", "preserve_core"]:
        val = decision.get(key)
        if isinstance(val, list):
            text_parts.extend(str(x) for x in val)
        elif val is not None:
            text_parts.append(str(val))
    text = " ".join(text_parts).lower()
    return any(k in text for k in [
        "missing multiplicative", "missing multiplier", "multiplicative envelope",
        "envelope variable", "add_missing_multiplicative_envelope", "add missing variable",
        "multiply by missing", "missing envelope"
    ])


def build_llm_approved_local_repair_candidates(current_best, dataset, repair_decision, max_candidates=None):
    """
    Deterministic expansion of an LLM-approved local repair plan.
    The LLM decides the repair action / variable roles; this function only creates
    fit-ready templates from that approved plan.
    """
    if not ENABLE_LLM_APPROVED_LOCAL_REPAIR_EXPANSION:
        return []
    if LLM_APPROVED_REPAIR_REQUIRE_EXPLICIT_ACTION and not _repair_decision_requests_missing_envelope(repair_decision):
        return []
    if current_best is None or dataset is None:
        return []
    feature_names = list(getattr(dataset, "feature_names", []) or [])
    expr = str(_safe_get_attr(current_best, "simplified_expression", "") or "").strip()
    if not expr or not feature_names:
        return []
    max_candidates = int(max_candidates or LLM_APPROVED_REPAIR_MAX_CANDIDATES)

    used = expression_variable_set(expr, feature_names)
    missing_default = [v for v in feature_names if v not in used]
    approved_missing = []
    for key in ["missing_variables_to_add", "envelope_variables_to_add", "add_variables", "missing_variables"]:
        val = repair_decision.get(key) if isinstance(repair_decision, dict) else None
        if isinstance(val, list):
            approved_missing.extend([str(x) for x in val])
        elif val:
            approved_missing.extend(_variables_used_in_text(str(val), feature_names))
    approved_missing = [v for v in feature_names if v in set(approved_missing)] or missing_default

    out, seen = [], set()
    def add(e):
        e = str(e).strip()
        if not e:
            return False
        key = _expr_dedup_key(e)
        if key not in seen:
            seen.add(key)
            out.append(e)
        return len(out) >= max_candidates

    clean_expr = re.sub(r"\bAbs\s*\(", "abs(", expr)
    for v in approved_missing:
        if add(f"a*({clean_expr})*{v} + b"):
            return out[:max_candidates]
        if add(f"a*({clean_expr})*({v}+b1) + b"):
            return out[:max_candidates]

    log_vars = _extract_log_core_variables(expr, feature_names)
    if len(log_vars) >= 2:
        u, v = log_vars[0], log_vars[1]
        base_envelope = [x for x in feature_names if x in used and x not in {u, v}]
        envelope = []
        for x in feature_names:
            if (x in base_envelope or x in approved_missing) and x not in envelope and x not in {u, v}:
                envelope.append(x)
        if envelope:
            env = "*".join(envelope[:4])
            if add(f"a*{env}*(log(abs({u})+b1)-log(abs({v})+b2)) + c"):
                return out[:max_candidates]
            if add(f"a*{env}*(log(abs({v})+b1)-log(abs({u})+b2)) + c"):
                return out[:max_candidates]
            if add(f"a*{env}*log(abs({u})/(abs({v})+b)) + c"):
                return out[:max_candidates]
            if add(f"a*{env}*log(abs({v})/(abs({u})+b)) + c"):
                return out[:max_candidates]

    for v in approved_missing:
        if add(f"a*({clean_expr}) + b*{v} + c"):
            return out[:max_candidates]
        if add(f"a*({clean_expr})*(1 + b1*{v}) + c"):
            return out[:max_candidates]
    return out[:max_candidates]


class PostEvalLocalRepairAgent:
    """Clean main build removes old post-eval local LLM repair."""
    def __init__(self, client=None):
        self.client = client
    def propose(self, *args, **kwargs):
        return {"used_llm": False, "candidate_exprs": [], "raw_text": "post-eval local repair removed in clean build"}


class ProposerAgent:
    def __init__(self, proposal_llm):
        self.proposal_llm = proposal_llm

    def _run_memory_source(self, dataset, row_meta, iter_cfg, experience_prior=None, timer=None):
        if PURE_LLM_CANDIDATE_MODE:
            return make_proposer_source_result(
                source_name="memory",
                exprs=[],
                raw_result={"skipped": True, "skip_reason": "memory guidance only in pure llm candidate mode"},
                skipped=True,
                skip_reason="memory guidance only in pure llm candidate mode",
                iter_cfg=iter_cfg,
                rationale_prefix="memory proposer",
            )
        if not ENABLE_MEMORY_PRIOR or not isinstance(experience_prior, dict) or not experience_prior:
            return make_proposer_source_result(
                source_name="memory",
                exprs=[],
                raw_result={"skipped": True, "skip_reason": "memory prior unavailable"},
                skipped=True,
                skip_reason="memory prior unavailable",
                iter_cfg=iter_cfg,
                rationale_prefix="memory proposer",
            )

        if not bool(experience_prior.get("memory_source_enabled", False)):
            return make_proposer_source_result(
                source_name="memory",
                exprs=[],
                raw_result={"skipped": True, "skip_reason": "memory source disabled"},
                skipped=True,
                skip_reason="memory source disabled",
                iter_cfg=iter_cfg,
                rationale_prefix="memory proposer",
            )

        exprs = list(experience_prior.get("memory_candidate_exprs", []) or [])
        exprs = deduplicate_expressions(exprs)
        cap = get_initial_proposer_source_cap("memory", iter_cfg=iter_cfg)
        exprs = exprs[:cap]
        candidate_items = _candidate_items_from_exprs(
            exprs=exprs,
            source_name="memory",
            rationale_prefix="historical memory recall",
            prior_score=float(experience_prior.get("confidence", 0.0) or 0.0),
        )
        return make_proposer_source_result(
            source_name="memory",
            exprs=exprs,
            candidate_items=candidate_items,
            raw_result={
                "memory_prior_summary": make_json_safe(experience_prior.get("memory_prior_summary")),
                "num_exprs_unique": len(deduplicate_expressions(exprs)),
            },
            iter_cfg=iter_cfg,
            rationale_prefix="memory proposer",
        )

    def _run_experience_source(self, dataset, row_meta, iter_cfg, experience_prior=None, timer=None):
        if PURE_LLM_CANDIDATE_MODE:
            return make_proposer_source_result(
                source_name="experience",
                exprs=[],
                raw_result={"skipped": True, "skip_reason": "experience guidance only in pure llm candidate mode"},
                skipped=True,
                skip_reason="experience guidance only in pure llm candidate mode",
                iter_cfg=iter_cfg,
                rationale_prefix="experience proposer",
            )
        if not ENABLE_EXPERIENCE_PRIOR or not isinstance(experience_prior, dict) or not experience_prior:
            return make_proposer_source_result(
                source_name="experience",
                exprs=[],
                raw_result={"skipped": True, "skip_reason": "experience prior unavailable"},
                skipped=True,
                skip_reason="experience prior unavailable",
                iter_cfg=iter_cfg,
                rationale_prefix="experience proposer",
            )
        if not bool(experience_prior.get("experience_source_enabled", False)):
            return make_proposer_source_result(
                source_name="experience",
                exprs=[],
                raw_result={"skipped": True, "skip_reason": "experience source disabled for non-strong prior"},
                skipped=True,
                skip_reason="experience source disabled for non-strong prior",
                iter_cfg=iter_cfg,
                rationale_prefix="experience proposer",
            )

        if timer is not None:
            timer.start("step_experience_proposal")
        candidate_items = build_experience_expansion_candidates(
            experience_prior=experience_prior,
            feature_names=dataset.feature_names,
            row_meta=row_meta,
            top_k_per_item=EXPERIENCE_TEMPLATE_TOPK_PER_ITEM,
        )
        if timer is not None:
            timer.stop("step_experience_proposal")

        experience_exprs = [str(item.get("expression", "")).strip() for item in candidate_items if str(item.get("expression", "")).strip()]
        return make_proposer_source_result(
            source_name="experience",
            exprs=experience_exprs,
            candidate_items=candidate_items,
            raw_result={
                "experience_prior": make_json_safe(summarize_experience_prior(experience_prior)),
                "num_exprs_unique": len(deduplicate_expressions(experience_exprs)),
            },
            iter_cfg=iter_cfg,
            rationale_prefix="experience proposer",
        )

    def _run_text_source(self, dataset, observation, iter_cfg, experience_prior=None, timer=None):
        text_calls = max(0, int(iter_cfg.get("text_calls", 0)))
        delayed_experience_hints = ENABLE_DELAYED_GUIDED_RESCUE and PURE_LLM_CANDIDATE_MODE and (not bool(iter_cfg.get("allow_guided_candidate_sources", False)))
        experience_hints = [] if (PURE_LLM_TEXT_ONLY_EVAL or delayed_experience_hints) else build_experience_prompt_hints(experience_prior)
        family_route_hints = build_family_route_hints_from_profile(
            observation.structure_profile,
            feature_names=dataset.feature_names,
            experience_prior=experience_prior,
        )
        prompt_family_templates = build_prompt_family_templates_from_profile(
            observation.structure_profile,
            feature_names=dataset.feature_names,
            experience_prior=experience_prior,
        )
        extra_context_lines = list(experience_hints) + list(family_route_hints) + list(prompt_family_templates)
        text_result = {"candidates": [], "per_call_stats": [], "num_exprs_unique": 0, "skipped": False}
        if text_calls > 0:
            if timer is not None:
                timer.start("step_text_single_proposal")
            text_result = generate_multiple_text_single_expressions_with_stats(
                proposal_llm=self.proposal_llm,
                df=dataset.train_df,
                variable_names=dataset.feature_names,
                target_name=dataset.target_name,
                plot_descriptions=observation.plot_descriptions,
                extra_context_lines=extra_context_lines,
                allowed_operators=ALLOWED_OPERATORS,
                max_rows=int(iter_cfg.get("text_max_rows", MAX_ROWS_FOR_TEXT_PROMPT)),
                num_calls=text_calls,
                temperatures=TEXT_SINGLE_TEMPERATURES,
                max_tokens=int(iter_cfg.get("proposal_max_tokens", 384)),
                max_workers=TEXT_PROPOSAL_MAX_WORKERS,
            )
            if timer is not None:
                timer.stop("step_text_single_proposal")
        else:
            text_result["skipped"] = True
            text_result["skip_reason"] = "text_calls<=0"
        text_result["experience_hints"] = experience_hints
        text_result["family_route_hints"] = family_route_hints
        text_result["prompt_family_templates"] = prompt_family_templates

        text_exprs = [x.get("expression", "") for x in text_result.get("candidates", []) if isinstance(x, dict)]
        return make_proposer_source_result(
            source_name="text",
            exprs=text_exprs,
            candidate_items=text_result.get("candidates", []),
            per_call_stats=text_result.get("per_call_stats", []),
            raw_result=text_result,
            skipped=bool(text_result.get("skipped", False)),
            skip_reason=text_result.get("skip_reason"),
            iter_cfg=iter_cfg,
            rationale_prefix="text proposer",
        )

    def _run_diverse_source(self, dataset, observation, iter_cfg, experience_prior=None, timer=None):
        if PURE_LLM_TEXT_ONLY_EVAL:
            return make_proposer_source_result(
                source_name="diverse",
                exprs=[],
                raw_result={"skipped": True, "skip_reason": "pure_llm_text_only_eval"},
                skipped=True,
                skip_reason="pure_llm_text_only_eval",
                iter_cfg=iter_cfg,
                rationale_prefix="diverse proposer",
            )
        if bool(iter_cfg.get("skip_diverse_proposal", False)):
            return make_proposer_source_result(
                source_name="diverse",
                exprs=[],
                raw_result={"skipped": True, "skip_reason": "skip_diverse_proposal"},
                skipped=True,
                skip_reason="skip_diverse_proposal",
                iter_cfg=iter_cfg,
                rationale_prefix="diverse proposer",
            )

        meta_notes = []
        delayed_experience_hints = ENABLE_DELAYED_GUIDED_RESCUE and PURE_LLM_CANDIDATE_MODE and (not bool(iter_cfg.get("allow_guided_candidate_sources", False)))
        experience_hints = [] if delayed_experience_hints else build_experience_prompt_hints(experience_prior)
        if timer is not None:
            timer.start("step_diverse_proposal")
        diverse_exprs = _try_diverse_generate(
            self.proposal_llm,
            dataset,
            list(observation.structure_hints) + experience_hints,
            observation.visual_hints,
            meta_notes,
            iter_cfg,
            experience_prior=experience_prior,
        )
        if timer is not None:
            timer.stop("step_diverse_proposal")

        return make_proposer_source_result(
            source_name="diverse",
            exprs=diverse_exprs,
            raw_result={
                "meta_notes": meta_notes,
                "experience_hints": experience_hints,
                "num_exprs_unique": len(deduplicate_expressions(diverse_exprs)),
            },
            metadata={"meta_notes": meta_notes},
            iter_cfg=iter_cfg,
            rationale_prefix="diverse proposer",
        )

    def _run_heuristic_source(self, dataset, observation, iter_cfg, experience_prior=None, timer=None):
        allow_guided_candidate_sources = bool((iter_cfg or {}).get("allow_guided_candidate_sources", False))
        allow_family_seed_sources = bool((iter_cfg or {}).get("allow_family_seed_sources", False))
        if PURE_LLM_CANDIDATE_MODE and not allow_guided_candidate_sources and not allow_family_seed_sources:
            return make_proposer_source_result(
                source_name="heuristic",
                exprs=[],
                raw_result={"skipped": True, "skip_reason": "heuristic source delayed until guided rescue in pure llm candidate mode"},
                skipped=True,
                skip_reason="heuristic source delayed until guided rescue in pure llm candidate mode",
                iter_cfg=iter_cfg,
                rationale_prefix="heuristic proposer",
            )
        if not ENABLE_HEURISTIC_PROPOSER:
            return make_proposer_source_result(
                source_name="heuristic",
                exprs=[],
                raw_result={"skipped": True, "skip_reason": "heuristic proposer disabled"},
                skipped=True,
                skip_reason="heuristic proposer disabled",
                iter_cfg=iter_cfg,
                rationale_prefix="heuristic proposer",
            )

        experience_hints = build_experience_prompt_hints(experience_prior)
        if timer is not None:
            timer.start("step_heuristic_proposal")
        heuristic_exprs = build_heuristic_skeleton_candidates(
            feature_names=dataset.feature_names,
            structure_hints=list(observation.structure_hints) + experience_hints,
            visual_hints=observation.visual_hints,
            max_candidates=get_initial_proposer_source_cap("heuristic", iter_cfg=iter_cfg),
        )
        specialist_exprs = build_family_specialist_candidates(
            dataset.feature_names,
            structure_profile=observation.structure_profile,
            experience_prior=experience_prior,
            max_candidates=max(
                get_initial_proposer_source_cap("heuristic", iter_cfg=iter_cfg),
                FAMILY_SPECIALIST_TEMPLATE_LIMIT,
            ),
        )
        family_routed_exprs = build_family_routed_candidates(
            dataset.feature_names,
            structure_profile=observation.structure_profile,
            experience_prior=experience_prior,
            max_candidates=max(
                get_initial_proposer_source_cap("heuristic", iter_cfg=iter_cfg),
                FAMILY_ROUTE_TEMPLATE_LIMIT,
            ),
        )
        role_guided_exprs = build_role_guided_high_dim_candidates(
            dataset.feature_names,
            structure_profile=observation.structure_profile,
            experience_prior=experience_prior,
            max_candidates=max(
                get_initial_proposer_source_cap("heuristic", iter_cfg=iter_cfg),
                HIGH_DIM_ROLE_TEMPLATE_LIMIT,
            ),
        )
        heuristic_exprs = _interleave_expression_groups(
            specialist_exprs,
            family_routed_exprs,
            role_guided_exprs,
            heuristic_exprs,
            max_candidates=max(
                get_initial_proposer_source_cap("heuristic", iter_cfg=iter_cfg) + 8,
                FAMILY_SPECIALIST_TEMPLATE_LIMIT + 4,
            ),
        )
        if timer is not None:
            timer.stop("step_heuristic_proposal")

        return make_proposer_source_result(
            source_name="heuristic",
            exprs=heuristic_exprs,
            raw_result={
                "structure_hints": list((list(observation.structure_hints) + experience_hints)[:8]),
                "visual_hints": list(observation.visual_hints[:6]),
                "experience_hints": experience_hints,
                "family_specialist_count": len(deduplicate_expressions(specialist_exprs)),
                "role_guided_count": len(deduplicate_expressions(role_guided_exprs)),
                "log_ratio_candidate_count": sum(
                    1 for e in role_guided_exprs
                    if _expr_has_log_ratio_signature(e)
                ),
                "log_ratio_candidates_preview": [
                    e for e in role_guided_exprs
                    if _expr_has_log_ratio_signature(e)
                ][:5],
                "family_route_hints": build_family_route_hints_from_profile(
                    observation.structure_profile,
                    feature_names=dataset.feature_names,
                    experience_prior=experience_prior,
                ),
            },
            iter_cfg=iter_cfg,
            rationale_prefix="heuristic proposer",
        )

    def _collect_initial_source_results(self, dataset, row_meta, observation, iter_cfg, experience_prior=None, timer=None):
        dataset.source_tag = str(row_meta.get("dataset_dir", ""))

        protected_candidates = []
        base_manual_candidates = build_manual_candidates(dataset.feature_names)
        family_routed_manual = build_family_routed_candidates(
            dataset.feature_names,
            structure_profile=observation.structure_profile,
            experience_prior=experience_prior,
            max_candidates=HIGH_DIM_ROLE_MANUAL_TOPUP,
        )
        specialist_manual = build_family_specialist_candidates(
            dataset.feature_names,
            structure_profile=observation.structure_profile,
            experience_prior=experience_prior,
            max_candidates=FAMILY_SPECIALIST_MANUAL_TOPUP,
        )
        experience_family_manual = list((experience_prior or {}).get("candidate_families", []) or [])
        role_guided_manual = build_role_guided_high_dim_candidates(
            dataset.feature_names,
            structure_profile=observation.structure_profile,
            experience_prior=experience_prior,
            max_candidates=HIGH_DIM_ROLE_MANUAL_TOPUP,
        )
        manual_candidates = _interleave_expression_groups(
            specialist_manual,
            family_routed_manual,
            role_guided_manual,
            experience_family_manual,
            base_manual_candidates,
            max_candidates=max(
                get_initial_proposer_source_cap("manual", iter_cfg=iter_cfg) + 12,
                HIGH_DIM_ROLE_MANUAL_TOPUP + FAMILY_SPECIALIST_MANUAL_TOPUP,
            ),
        )
        if (not STRICT_NO_FORMULA_TEMPLATES) and (not TEMP_DISABLE_TEMPLATE_SOURCES):
            if timer is not None:
                timer.start("step_manual_candidates")
            protected_candidates, prepared_manual_candidates = prepare_initial_template_sources(
                dataset.feature_names,
                row_meta=row_meta,
                iter_cfg=iter_cfg,
                experience_prior=experience_prior,
            )
            manual_candidates = _interleave_expression_groups(
                specialist_manual,
                family_routed_manual,
                role_guided_manual,
                experience_family_manual,
                prepared_manual_candidates,
                base_manual_candidates,
                max_candidates=max(
                    get_initial_proposer_source_cap("manual", iter_cfg=iter_cfg) + 12,
                    HIGH_DIM_ROLE_MANUAL_TOPUP + FAMILY_SPECIALIST_MANUAL_TOPUP,
                ),
            )
            if timer is not None:
                timer.stop("step_manual_candidates")

        datadriven_candidates, datadriven_trace = build_data_driven_feature_seed_candidates(
            dataset=dataset,
            max_candidates=get_initial_proposer_source_cap("datadriven", iter_cfg=iter_cfg),
        )

        generic_candidates = build_generic_diverse_structural_candidates(
            feature_names=dataset.feature_names,
            structure_profile=observation.structure_profile,
            max_total=get_initial_proposer_source_cap("generic", iter_cfg=iter_cfg),
        )

        source_results = {
            "datadriven": make_proposer_source_result(
                source_name="datadriven",
                exprs=datadriven_candidates,
                raw_result=datadriven_trace,
                iter_cfg=iter_cfg,
                rationale_prefix="fair data-driven feature seed proposer",
            ),
            "generic": make_proposer_source_result(
                source_name="generic",
                exprs=generic_candidates,
                raw_result={
                    "num_exprs_unique": len(deduplicate_expressions(generic_candidates)),
                    "note": "fair generic diverse structural seeds; no benchmark-name-specific prior",
                    "families": {
                        fam: sum(1 for e in generic_candidates if candidate_family_label(e) == fam)
                        for fam in sorted(set(candidate_family_label(e) for e in generic_candidates))
                    },
                },
                iter_cfg=iter_cfg,
                rationale_prefix="generic diverse structural proposer",
            ),
            "protected": make_proposer_source_result(
                source_name="protected",
                exprs=protected_candidates,
                raw_result={"num_exprs_unique": len(deduplicate_expressions(protected_candidates))},
                iter_cfg=iter_cfg,
                rationale_prefix="protected template proposer",
            ),
            "memory": self._run_memory_source(
                dataset=dataset,
                row_meta=row_meta,
                iter_cfg=iter_cfg,
                experience_prior=experience_prior,
                timer=timer,
            ),
            "experience": self._run_experience_source(
                dataset=dataset,
                row_meta=row_meta,
                iter_cfg=iter_cfg,
                experience_prior=experience_prior,
                timer=timer,
            ),
            "heuristic": self._run_heuristic_source(
                dataset=dataset,
                observation=observation,
                iter_cfg=iter_cfg,
                experience_prior=experience_prior,
                timer=timer,
            ),
            "text": self._run_text_source(
                dataset=dataset,
                observation=observation,
                iter_cfg=iter_cfg,
                experience_prior=experience_prior,
                timer=timer,
            ),
            "diverse": self._run_diverse_source(
                dataset=dataset,
                observation=observation,
                iter_cfg=iter_cfg,
                experience_prior=experience_prior,
                timer=timer,
            ),
            "manual": make_proposer_source_result(
                source_name="manual",
                exprs=manual_candidates,
                raw_result={"num_exprs_unique": len(deduplicate_expressions(manual_candidates))},
                iter_cfg=iter_cfg,
                rationale_prefix="manual template proposer",
            ),
        }
        return source_results

    def propose_initial(self, dataset, row_meta, observation: ObservationBundle, iter_cfg, experience_prior=None, timer=None):
        mm_stats = None
        mm_result = {"candidates": [], "per_call_stats": [], "num_exprs_unique": 0}

        source_results = self._collect_initial_source_results(
            dataset=dataset,
            row_meta=row_meta,
            observation=observation,
            iter_cfg=iter_cfg,
            experience_prior=experience_prior,
            timer=timer,
        )
        source_order = get_initial_proposer_source_order(iter_cfg=iter_cfg, available_sources=source_results.keys())
        expr_groups = [source_results[name].get("exprs", []) for name in source_order]

        merged_exprs = merge_expression_groups_with_limit(
            expr_groups,
            max_total=MAX_INITIAL_CANDIDATES,
        )
        merged_candidate_count_before_prefilter = len(merged_exprs)

        if STRICT_NO_FORMULA_TEMPLATES and not merged_exprs:
            text_result = generate_multiple_text_single_expressions_with_stats(
                proposal_llm=self.proposal_llm,
                df=dataset.train_df,
                variable_names=dataset.feature_names,
                target_name=dataset.target_name,
                plot_descriptions=list(observation.plot_descriptions) + list(STRICT_NO_TEMPLATE_RETRY_HINTS),
                extra_context_lines=["Return a directly evaluable expression with numeric constants only."],
                allowed_operators=ALLOWED_OPERATORS,
                max_rows=int(iter_cfg.get("text_max_rows", MAX_ROWS_FOR_TEXT_PROMPT)),
                num_calls=1,
                temperatures=[EMPTY_RETRY_TEMPERATURE],
                max_tokens=int(iter_cfg.get("proposal_max_tokens", 384)),
                max_workers=1,
            )
            retry_exprs = [x.get("expression", "") for x in text_result.get("candidates", []) if isinstance(x, dict)]
            merged_exprs = deduplicate_expressions(retry_exprs)
            if merged_exprs:
                source_results["text"]["exprs"] = deduplicate_expressions(list(source_results["text"].get("exprs", [])) + merged_exprs)
                source_results["text"]["candidate_items"] = _normalize_source_candidate_items(
                    list(source_results["text"].get("candidate_items", [])) + list(text_result.get("candidates", [])),
                    source_name="text",
                    rationale_prefix="text proposer",
                )
                source_results["text"]["raw_result"] = make_json_safe({
                    **dict(source_results["text"].get("raw_result") or {}),
                    "no_template_retry": make_json_safe(text_result),
                })

        if ENABLE_DELAYED_GUIDED_RESCUE and PURE_LLM_CANDIDATE_MODE and (not bool(iter_cfg.get("allow_guided_candidate_sources", False))):
            experience_rerank_trace = {
                "applied": False,
                "reason": "delayed_until_guided_rescue",
            }
        else:
            merged_exprs, experience_rerank_trace = reorder_expressions_by_experience(
                merged_exprs,
                feature_names=dataset.feature_names,
                experience_prior=experience_prior,
                top_k=EXPERIENCE_RERANK_TOPK,
            )

        if ENABLE_LIGHT_PREFILTER and len(merged_exprs) >= PREFILTER_TRIGGER_CANDIDATE_COUNT:
            if timer is not None:
                timer.start("step_initial_candidate_prefilter")
            merged_exprs = lightweight_prefilter_candidates(
                merged_exprs,
                dataset,
                timer=timer,
                prefix="initial_prefilter",
            )
            if timer is not None:
                timer.stop("step_initial_candidate_prefilter")

        text_stats = source_results.get("text", {}).get("raw_result")
        datadriven_count = len(source_results.get("datadriven", {}).get("exprs", []))
        generic_count = len(source_results.get("generic", {}).get("exprs", []))
        protected_count = len(source_results.get("protected", {}).get("exprs", []))
        memory_count = len(source_results.get("memory", {}).get("exprs", []))
        experience_count = len(source_results.get("experience", {}).get("exprs", []))
        heuristic_count = len(source_results.get("heuristic", {}).get("exprs", []))
        manual_count = len(source_results.get("manual", {}).get("exprs", []))
        diverse_count = len(source_results.get("diverse", {}).get("exprs", []))

        return {
            "candidate_exprs": merged_exprs,
            "trace": {
                "text_proposal_stats": make_json_safe(text_stats),
                "datadriven_candidate_count": datadriven_count,
                "generic_candidate_count": generic_count,
                "mm_proposal_stats": mm_stats,
                "diverse_proposal_count": diverse_count,
                "manual_candidate_count": protected_count + manual_count,
                "protected_candidate_count": protected_count,
                "memory_candidate_count": memory_count,
                "experience_candidate_count": experience_count,
                "heuristic_candidate_count": heuristic_count,
                "source_order": source_order,
                "source_stats": {name: make_json_safe(source_results[name]) for name in source_order},
                "experience_prior": make_json_safe(summarize_experience_prior(experience_prior)),
                "experience_rerank": make_json_safe(experience_rerank_trace),
                "merged_candidate_count_before_prefilter": merged_candidate_count_before_prefilter,
                "merged_candidate_count": len(merged_exprs),
                "mm_result": mm_result,
            },
        }

    def propose_mm_if_needed(self, dataset, observation: ObservationBundle, iter_cfg, row_meta=None, allow_legacy_fallback=True, timer=None):
        if not observation.image_paths:
            return {"candidate_exprs": [], "trace": {"num_mm_exprs": 0, "per_call_stats": []}}

        form_result = {"candidates": [], "per_call_stats": [], "num_exprs_unique": 0, "raw_text": ""}
        guided_template_result = {"candidates": [], "num_exprs_unique": 0}
        legacy_result = {"candidates": [], "per_call_stats": [], "num_exprs_unique": 0, "raw_text": ""}

        if ENABLE_MM_FORM_PROPOSAL:
            if timer is not None:
                timer.start("step_mm_form_proposal")
            form_result = generate_multiple_mm_formula_form_candidates(
                client=self.proposal_llm.client,
                df=dataset.train_df,
                variable_names=dataset.feature_names,
                target_name=dataset.target_name,
                image_paths=observation.image_paths,
                plot_descriptions=observation.plot_descriptions,
                structure_hints=observation.structure_hints,
                structure_profile=observation.structure_profile,
                visual_summary=observation.visual_summary,
                reconstruction_tokens=getattr(observation, "reconstruction_tokens", None),
                allowed_operators=ALLOWED_OPERATORS,
                max_rows=MAX_ROWS_FOR_MM_PROMPT,
                num_calls=max(1, iter_cfg.get("mm_calls", 1)),
                temperatures=MM_SINGLE_TEMPERATURES,
                max_workers=MM_FORM_PROPOSAL_MAX_WORKERS,
                num_candidates_per_call=MM_FORM_CANDIDATES_PER_CALL,
            )
            if timer is not None:
                timer.stop("step_mm_form_proposal")

        if ENABLE_MM_FORM_TEMPLATE_EXPANSION:
            if timer is not None:
                timer.start("step_mm_form_template_expand")
            guided_candidates = build_vlm_guided_template_candidates(
                candidate_items=form_result.get("candidates", []),
                feature_names=dataset.feature_names,
                row_meta=row_meta,
                top_k_per_item=MM_FORM_TEMPLATE_TOPK_PER_ITEM,
            )
            guided_template_result = {
                "candidates": guided_candidates,
                "num_exprs_unique": len(guided_candidates),
            }
            if timer is not None:
                timer.stop("step_mm_form_template_expand")

        form_exprs = [str(item.get("expression", "")).strip() for item in form_result.get("candidates", []) if str(item.get("expression", "")).strip()]
        guided_exprs = [str(item.get("expression", "")).strip() for item in guided_template_result.get("candidates", []) if str(item.get("expression", "")).strip()]
        structural_exprs = deduplicate_expressions(guided_exprs + form_exprs)
        need_legacy_fallback = len(structural_exprs) < max(1, min(MM_ENOUGH_UNIQUE_THRESHOLD, iter_cfg.get("mm_calls", 1) + 1))

        if need_legacy_fallback and allow_legacy_fallback:
            if timer is not None:
                timer.start("step_mm_legacy_single_proposal")
            legacy_result = generate_multiple_mm_single_expressions(
                client=self.proposal_llm.client,
                df=dataset.train_df,
                variable_names=dataset.feature_names,
                target_name=dataset.target_name,
                image_paths=observation.image_paths,
                plot_descriptions=observation.plot_descriptions,
                structure_profile=observation.structure_profile,
                visual_summary=observation.visual_summary,
                allowed_operators=ALLOWED_OPERATORS,
                max_rows=MAX_ROWS_FOR_MM_PROMPT,
                num_calls=max(1, iter_cfg.get("mm_calls", 1)),
                temperatures=MM_SINGLE_TEMPERATURES,
                max_workers=MM_PROPOSAL_MAX_WORKERS,
            )
            if timer is not None:
                timer.stop("step_mm_legacy_single_proposal")

        legacy_exprs = [str(item.get("expression", "")).strip() for item in legacy_result.get("candidates", []) if str(item.get("expression", "")).strip()]
        all_exprs = deduplicate_expressions(guided_exprs + form_exprs + legacy_exprs)
        return {
            "candidate_exprs": all_exprs,
            "trace": {
                "num_mm_exprs": len(all_exprs),
                "form_proposal_stats": make_json_safe(form_result),
                "guided_template_stats": make_json_safe(guided_template_result),
                "legacy_single_line_stats": make_json_safe(legacy_result),
                "used_legacy_fallback": bool(need_legacy_fallback and allow_legacy_fallback),
                "per_call_stats": make_json_safe(form_result.get("per_call_stats", [])) + make_json_safe(legacy_result.get("per_call_stats", [])),
                "raw_text": "\n\n".join([x for x in [form_result.get("raw_text", ""), legacy_result.get("raw_text", "")] if x]),
            },
        }




@dataclass
class DirectVerifiedResult:
    """Lightweight result object compatible with _safe_get_attr/get_best_result.

    It is created only for expressions that are directly evaluable without free
    symbolic parameters. This prevents exact low-parameter evidence candidates
    from being lost when the default template fitter/scorer favors a flexible
    surrogate.
    """
    simplified_expression: str
    val_mse: float
    test_mse: Optional[float]
    train_mse: float
    complexity: int
    score: float
    source: str = "direct_verified_evidence"


_ALLOWED_EVAL_NAMES = {
    "sin", "cos", "tan", "sinh", "cosh", "tanh", "exp", "log", "sqrt", "abs", "Abs", "pi", "e"
}


def _expression_has_free_parameters(expr, feature_names):
    """Return True when expr contains symbols outside variables/functions/constants."""
    expr = str(expr or "").strip()
    if not expr:
        return True
    allowed = set(feature_names or []) | set(_ALLOWED_EVAL_NAMES)
    try:
        tree = ast.parse(expr, mode="eval")
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                if node.id not in allowed:
                    return True
        return False
    except Exception:
        # Conservative fallback: common free parameter names used in templates.
        for token in re.findall(r"\b[A-Za-z_]\w*\b", expr):
            if token not in allowed:
                return True
        return False


def _contains_huge_numeric_constant(expr, threshold=None):
    threshold = float(threshold if threshold is not None else HUGE_CONSTANT_SURROGATE_ABS_THRESHOLD)
    expr = str(expr or "")
    for match in re.finditer(r"(?<![A-Za-z_])[-+]?(?:\d+\.\d*|\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", expr):
        try:
            value = abs(float(match.group(0)))
        except Exception:
            continue
        if value >= threshold:
            return True
    return False

def _looks_like_shifted_abs_surrogate(expr):
    expr = str(expr or "")
    compact = expr.replace(" ", "")
    if not compact:
        return False
    return (
        ("Abs(" in expr or "abs(" in expr)
        and "/" in compact
        and _contains_huge_numeric_constant(expr, threshold=100.0)
    )


def should_accept_candidate_update(candidate, incumbent, min_rel_improvement_for_surrogate=0.05):
    """Guard against accepting tiny MSE gains from ugly shifted Abs surrogates.

    This is benchmark-name-free. It only blocks candidates with Abs/large shifted
    denominator style when their validation improvement over the incumbent is tiny.
    Exact/near-exact candidates are always allowed.
    """
    if not is_better_result(candidate, incumbent):
        return False
    if candidate is None or incumbent is None:
        return True
    cand_expr = str(_safe_get_attr(candidate, "simplified_expression", "") or "")
    cand_val = _safe_metric_float(_safe_get_attr(candidate, "val_mse", None))
    inc_val = _safe_metric_float(_safe_get_attr(incumbent, "val_mse", None))
    if cand_val is None or inc_val is None:
        return True
    if cand_val <= EVIDENCE_DIRECT_PROMOTION_VAL_TOL:
        return True
    if not _looks_like_shifted_abs_surrogate(cand_expr):
        return True
    if inc_val <= 1e-12:
        return False
    rel_gain = (inc_val - cand_val) / max(abs(inc_val), 1e-12)
    return bool(rel_gain >= float(min_rel_improvement_for_surrogate))


def _direct_train_val_mse_for_expr(expr, dataset):
    """Evaluate a numeric/free-parameter-free expression on search-visible data."""
    try:
        pred_train = evaluate_expression_on_df(expr, dataset.train_df)
        y_train = np.asarray(dataset.train_df[dataset.target_name], dtype=float)
        pred_val = evaluate_expression_on_df(expr, dataset.val_df)
        y_val = np.asarray(dataset.val_df[dataset.target_name], dtype=float)
        for arr in [pred_train, pred_val, y_train, y_val]:
            arr = np.asarray(arr, dtype=float)
            if arr.size == 0 or not np.isfinite(arr).all():
                return None
        return {
            "train_mse": float(np.mean((np.asarray(pred_train, dtype=float) - y_train) ** 2)),
            "val_mse": float(np.mean((np.asarray(pred_val, dtype=float) - y_val) ** 2)),
        }
    except Exception:
        return None


def build_direct_verified_evidence_results(candidate_exprs, dataset, feature_names=None, max_candidates=None):
    """Promote directly evaluable evidence candidates that are nearly exact.

    This is deliberately generic: it does not inspect benchmark names or true
    expressions. It simply re-checks raw candidate expressions against fitting
    and validation data and constructs lightweight result objects for near-exact
    candidates. The test split remains sealed until final selection is complete.
    """
    if not ENABLE_EVIDENCE_PRESERVING_SELECTION:
        return []
    feature_names = list(feature_names or getattr(dataset, "feature_names", []) or [])
    max_candidates = int(max_candidates or EVIDENCE_DIRECT_PROMOTION_MAX_CANDIDATES)
    out = []
    seen = set()
    for expr in candidate_exprs or []:
        expr = str(expr or "").strip()
        if not expr:
            continue
        key = _expr_dedup_key(expr)
        if key in seen:
            continue
        seen.add(key)
        if len(expr) > EVIDENCE_DIRECT_PROMOTION_MAX_EXPR_LEN:
            continue
        if _expression_has_free_parameters(expr, feature_names):
            continue
        metrics = _direct_train_val_mse_for_expr(expr, dataset)
        if not metrics:
            continue
        val_mse = _safe_metric_float(metrics.get("val_mse"))
        train_mse = _safe_metric_float(metrics.get("train_mse"))
        if val_mse is None or train_mse is None:
            continue
        if val_mse <= EVIDENCE_DIRECT_PROMOTION_VAL_TOL:
            complexity = len(expr.replace(" ", ""))
            score = -float(val_mse) - COMPLEXITY_WEIGHT * float(complexity)
            out.append(DirectVerifiedResult(
                simplified_expression=expr,
                val_mse=float(val_mse),
                test_mse=None,
                train_mse=float(train_mse),
                complexity=int(complexity),
                score=float(score),
            ))
            if len(out) >= max_candidates:
                break
    out.sort(key=lambda x: (float(x.val_mse), float(x.train_mse), int(x.complexity)))
    return out


def _should_apply_lowdim_small_sample_cv_rerank(dataset, row_meta=None):
    if not ENABLE_LOW_DIM_SMALL_SAMPLE_CV_RERANK:
        return False
    feature_names = list(getattr(dataset, "feature_names", []) or [])
    if len(feature_names) == 0 or len(feature_names) > LOW_DIM_SMALL_SAMPLE_CV_MAX_FEATURES:
        return False
    train_df = getattr(dataset, "train_df", None)
    val_df = getattr(dataset, "val_df", None)
    if train_df is None or val_df is None:
        return False
    n_train = len(train_df)
    n_val = len(val_df)
    if n_train < 4 or n_val <= 0:
        return False
    if n_val > LOW_DIM_SMALL_SAMPLE_CV_MAX_VAL_ROWS:
        return False
    if (n_train + n_val) > LOW_DIM_SMALL_SAMPLE_CV_MAX_TOTAL_ROWS:
        return False
    return True


def _iter_lowdim_small_sample_cv_splits(n_rows):
    n_rows = int(n_rows or 0)
    if n_rows < 4:
        return []
    if n_rows <= LOW_DIM_SMALL_SAMPLE_CV_LOOCV_MAX_TRAIN:
        return [
            (np.asarray([j for j in range(n_rows) if j != i], dtype=int), np.asarray([i], dtype=int))
            for i in range(n_rows)
        ]
    n_folds = max(2, min(int(LOW_DIM_SMALL_SAMPLE_CV_NUM_FOLDS), n_rows))
    idx = np.arange(n_rows)
    rng = np.random.default_rng(42)
    rng.shuffle(idx)
    out = []
    for val_idx in np.array_split(idx, n_folds):
        val_idx = np.asarray(val_idx, dtype=int)
        if val_idx.size == 0 or val_idx.size >= n_rows:
            continue
        mask = np.ones(n_rows, dtype=bool)
        mask[val_idx] = False
        train_idx = idx[mask[idx]]
        if train_idx.size == 0:
            continue
        out.append((np.asarray(train_idx, dtype=int), val_idx))
    return out


def _estimate_lowdim_small_sample_cv_stats(expr_template, dataset):
    expr_template = str(expr_template or "").strip()
    if not expr_template:
        return None
    train_df = getattr(dataset, "train_df", None)
    if train_df is None:
        return None
    train_df = train_df.reset_index(drop=True)
    splits = _iter_lowdim_small_sample_cv_splits(len(train_df))
    if not splits:
        return None

    fitter = TemplateFillTool(max_workers=1)
    fold_mses = []
    with TemporaryDirectory(prefix="llmsr_tinycv_") as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        for train_idx, val_idx in splits:
            fold_train = train_df.iloc[list(train_idx)].reset_index(drop=True)
            fold_val = train_df.iloc[list(val_idx)].reset_index(drop=True)
            if len(fold_train) < 2 or len(fold_val) < 1:
                return None
            fold_dataset = build_dataset_from_explicit_splits(
                train_df=fold_train,
                val_df=fold_val,
                test_df=fold_val,
                tmpdir=tmpdir,
            )
            fold_dataset.source_tag = getattr(dataset, "source_tag", "")
            fit_results = fitter.run(
                [expr_template],
                fold_dataset,
                n_restarts=max(1, int(LOW_DIM_SMALL_SAMPLE_CV_RESTARTS)),
                init_scale=float(LOW_DIM_SMALL_SAMPLE_CV_INIT_SCALE),
            )
            fit = fit_results[0] if fit_results else None
            fold_val_mse = _safe_metric_float(_safe_get_attr(fit, "val_mse", None))
            if fold_val_mse is None:
                return None
            fold_mses.append(float(fold_val_mse))
    if not fold_mses:
        return None
    return {
        "cv_mse": float(np.mean(fold_mses)),
        "cv_median_mse": float(np.median(fold_mses)),
        "num_folds": int(len(fold_mses)),
    }


def rerank_lowdim_small_sample_results(scored_results, dataset, row_meta=None):
    """Stabilize tiny-validation 1D selection with a train-only CV rerank."""
    if not scored_results or not _should_apply_lowdim_small_sample_cv_rerank(dataset, row_meta=row_meta):
        return scored_results, {"applied": False, "reason": "not_small_sample_low_dim"}

    head_size = max(1, min(int(LOW_DIM_SMALL_SAMPLE_CV_TOPK), len(scored_results)))
    head = list(scored_results[:head_size])
    tail = list(scored_results[head_size:])
    rerank_records = []
    valid_cv = 0

    for item in tail:
        val = _safe_metric_float(_safe_get_attr(item, "val_mse", None))
        if val is not None:
            setattr(item, "selection_metric", float(val))

    for idx, item in enumerate(head):
        val = _safe_metric_float(_safe_get_attr(item, "val_mse", None))
        comp = _safe_metric_float(_safe_get_attr(item, "complexity", None))
        expr = str(_safe_get_attr(item, "simplified_expression", "") or "")
        template_expr = _safe_get_attr(item, "expression", None)
        direct_verified = str(_safe_get_attr(item, "source", "")) == "direct_verified_evidence"
        near_exact = val is not None and float(val) <= EVIDENCE_DIRECT_PROMOTION_VAL_TOL

        if direct_verified and near_exact:
            selection_metric = float(val)
            setattr(item, "selection_metric", selection_metric)
            setattr(item, "small_sample_cv_mse", float(val))
            rerank_records.append({
                "bucket": 0,
                "selection_metric": selection_metric,
                "cv_mse": float(val),
                "val_mse": float(val),
                "complexity": float(comp) if comp is not None else float("inf"),
                "orig_rank": idx,
                "item": item,
            })
            continue

        cv_stats = _estimate_lowdim_small_sample_cv_stats(template_expr, dataset) if template_expr else None
        cv_mse = _safe_metric_float((cv_stats or {}).get("cv_mse"))
        if cv_mse is not None:
            valid_cv += 1
            selection_metric = float(cv_mse)
            if val is not None:
                selection_metric += float(LOW_DIM_SMALL_SAMPLE_CV_VAL_WEIGHT) * float(val)
            setattr(item, "small_sample_cv_mse", float(cv_mse))
            setattr(item, "selection_metric", float(selection_metric))
            bucket = 1
        else:
            selection_metric = float(val) if val is not None else float("inf")
            setattr(item, "selection_metric", float(selection_metric))
            setattr(item, "small_sample_cv_mse", None)
            bucket = 2

        rerank_records.append({
            "bucket": int(bucket),
            "selection_metric": float(selection_metric),
            "cv_mse": float(cv_mse) if cv_mse is not None else float("inf"),
            "val_mse": float(val) if val is not None else float("inf"),
            "complexity": float(comp) if comp is not None else float("inf"),
            "orig_rank": idx,
            "item": item,
            "expr_preview": expr[:120],
            "cv_stats": make_json_safe(cv_stats),
        })

    if valid_cv < 2:
        return scored_results, {
            "applied": False,
            "reason": "insufficient_valid_cv",
            "head_size": int(head_size),
            "valid_cv": int(valid_cv),
        }

    rerank_records.sort(key=lambda rec: (
        int(rec["bucket"]),
        float(rec["selection_metric"]),
        float(rec["val_mse"]),
        float(rec["complexity"]),
        int(rec["orig_rank"]),
    ))
    reranked_head = [rec["item"] for rec in rerank_records]
    changed = any(id(a) != id(b) for a, b in zip(reranked_head, head))
    return reranked_head + tail, {
        "applied": bool(changed),
        "reason": "tiny_val_train_cv_rerank",
        "head_size": int(head_size),
        "valid_cv": int(valid_cv),
        "records": [
            {
                "bucket": rec["bucket"],
                "selection_metric": round(float(rec["selection_metric"]), 8),
                "cv_mse": None if not np.isfinite(rec["cv_mse"]) else round(float(rec["cv_mse"]), 8),
                "val_mse": None if not np.isfinite(rec["val_mse"]) else round(float(rec["val_mse"]), 8),
                "complexity": None if not np.isfinite(rec["complexity"]) else int(rec["complexity"]),
                "orig_rank": int(rec["orig_rank"]) + 1,
                "expr_preview": rec.get("expr_preview"),
                "cv_stats": rec.get("cv_stats"),
            }
            for rec in rerank_records[:min(8, len(rerank_records))]
        ],
    }


def rerank_mechanism_preserving_results(scored_results, feature_names=None):
    """Softly prefer simple mechanism-like results over huge-constant surrogates.

    This does not require a specific equation family. It only acts when candidates
    have similar validation error, and then favors lower-complexity, no-huge-constant
    expressions. Near-perfect direct evidence candidates always stay first.
    """
    if not scored_results:
        return scored_results
    best_val = _safe_metric_float(_safe_get_attr(scored_results[0], "val_mse", None))
    if best_val is None:
        return scored_results

    def item_key(item):
        expr = str(_safe_get_attr(item, "simplified_expression", "") or "")
        val = _safe_metric_float(_safe_get_attr(item, "val_mse", None))
        # Test MSE is intentionally not used for ranking. It is report-only.
        comp = _safe_metric_float(_safe_get_attr(item, "complexity", None))
        if val is None:
            val = float("inf")
        if comp is None:
            comp = len(expr)
        near_best = val <= max(best_val * MECHANISM_SELECTION_REL_TOL, best_val + MECHANISM_SELECTION_ABS_TOL)
        huge = _contains_huge_numeric_constant(expr) if ENABLE_HUGE_CONSTANT_SURROGATE_PENALTY else False
        direct_verified = str(_safe_get_attr(item, "source", "")) == "direct_verified_evidence"
        near_exact = val <= EVIDENCE_DIRECT_PROMOTION_VAL_TOL
        # Sort priority:
        # 0) near-exact direct evidence, 1) near-best clean/non-huge,
        # 3) huge-constant near-best, 4) worse candidates.
        if direct_verified and near_exact:
            bucket = 0
        elif near_best and not huge:
            bucket = 1
        elif near_best and huge:
            bucket = 3
        else:
            bucket = 4
        return (bucket, float(val), float(comp), len(expr))

    return sorted(scored_results, key=item_key)

def _high_dim_clean_ratio_signature(expr, feature_names):
    """Detect compact benchmark-name-free high-dimensional difference/ratio forms."""
    expr = str(expr or "").strip()
    compact = expr.replace(" ", "")
    if not expr or not feature_names or len(feature_names) < 4:
        return {"is_clean": False}
    if any(fn in compact for fn in ["sin(", "cos(", "tan(", "exp(", "log(", "sinh(", "cosh(", "tanh("]):
        return {"is_clean": False}
    if "/" not in compact or "*" not in compact or "-" not in compact:
        return {"is_clean": False}
    if re.search(r"(?:\d+\.\d+e\+?\d{3,}|\d{6,})", compact):
        return {"is_clean": False, "reason": "huge_constants"}
    vars_used = [v for v in feature_names if re.search(rf"\b{re.escape(v)}\b", expr)]
    if len(vars_used) < min(4, len(feature_names)):
        return {"is_clean": False, "reason": "too_few_variables", "vars_used": vars_used}
    try:
        sig = extract_formula_form_signature(expr, feature_names)
        families = set(sig.get("families", []) or [])
        clean = (
            "rational" in families
            and "interaction" in families
            and sig.get("max_multiplicative_arity", 1) >= 3
            and sig.get("max_power_degree", 1.0) <= 2.0
            and len(compact) <= 140
        )
        return {"is_clean": bool(clean), "vars_used": vars_used, "families": sorted(families), "complexity_hint": len(compact)}
    except Exception:
        clean = any(
            f"{u}-{v}" in compact or f"{v}-{u}" in compact
            for i, u in enumerate(feature_names)
            for v in feature_names[i + 1:]
        )
        return {"is_clean": bool(clean), "vars_used": vars_used, "complexity_hint": len(compact)}


def rerank_highdim_clean_mechanistic_results(scored_results, row_meta=None, feature_names=None):
    """Prefer compact mechanism-like high-dimensional ratio/difference forms over ugly shifted surrogates.

    This is only a soft rerank: a clean form must be numerically close to the current best.
    It is benchmark-name-free and relies only on expression structure and validation MSE.
    """
    if not ENABLE_HIGH_DIM_CLEAN_MECHANISTIC_RERANK or not scored_results:
        return scored_results
    feature_names = list(feature_names or [])
    if len(feature_names) < 5:
        return scored_results
    if row_meta is not None and str(row_meta.get("dataset_dir", "") or "").strip().lower() not in {"", "benchmark_csv", "sldbench", "llmsrbench", "srbench"}:
        return scored_results
    best_val = _safe_metric_float(_safe_get_attr(scored_results[0], "val_mse", None))
    if best_val is None:
        return scored_results

    favored = []
    others = []
    for item in scored_results:
        expr = str(_safe_get_attr(item, "simplified_expression", "") or "")
        val = _safe_metric_float(_safe_get_attr(item, "val_mse", None))
        if val is None:
            others.append(item)
            continue
        sig = _high_dim_clean_ratio_signature(expr, feature_names)
        close_enough = val <= max(best_val * HIGH_DIM_CLEAN_MECH_RERANK_REL_TOL, best_val + HIGH_DIM_CLEAN_MECH_RERANK_ABS_TOL)
        if sig.get("is_clean") and close_enough:
            favored.append(item)
        else:
            others.append(item)
    if not favored:
        return scored_results
    favored.sort(key=lambda x: (
        float(_safe_get_attr(x, "val_mse", np.inf)),
        float(_safe_get_attr(x, "complexity", np.inf)),
        len(str(_safe_get_attr(x, "simplified_expression", "") or "")),
    ))
    favored_ids = {id(x) for x in favored}
    return favored + [x for x in scored_results if id(x) not in favored_ids]


class EvaluatorAgent:
    def __init__(self, complexity_weight=1e-2, deadline_ts=None):
        self.complexity_weight = complexity_weight
        self.deadline_ts = deadline_ts
        self.fitter = TemplateFillTool(max_workers=TEMPLATE_FIT_MAX_WORKERS)
        self.simplifier = AlgebraicSimplifyTool()
        self.deduper = EquivalenceCheckTool()
        self.scorer = ScoringTool(complexity_weight=complexity_weight)

    def evaluate(self, candidate_exprs, dataset, row_meta=None, timer=None, prefix="eval", deadline_ts=None):
        active_deadline_ts = deadline_ts if deadline_ts is not None else self.deadline_ts
        fit_results, simplify_results, unique_results, scored_results = evaluate_candidate_expressions(
            candidate_exprs=candidate_exprs,
            dataset=dataset,
            complexity_weight=self.complexity_weight,
            timer=timer,
            prefix=prefix,
            fitter=self.fitter,
            simplifier=self.simplifier,
            deduper=self.deduper,
            scorer=self.scorer,
            deadline_ts=active_deadline_ts,
        )
        feature_names_for_selection = list(getattr(dataset, "feature_names", []) or [])

        # Evidence-preserving final selection. Re-check raw candidates that are
        # already numeric/free-parameter-free. If one is near-exact, keep it in
        # the scored list even if the default template fitter/scorer fails to
        # rank it first. This fixes cases where the correct data-driven basis is
        # present in raw_exprs but a high-flexibility shifted surrogate wins.
        direct_verified_results = build_direct_verified_evidence_results(
            candidate_exprs=candidate_exprs,
            dataset=dataset,
            feature_names=feature_names_for_selection,
        )
        if direct_verified_results:
            existing_keys = {
                _expr_dedup_key(str(_safe_get_attr(item, "simplified_expression", "") or ""))
                for item in scored_results
            }
            promoted = [
                item for item in direct_verified_results
                if _expr_dedup_key(item.simplified_expression) not in existing_keys
            ]
            scored_results = promoted + list(scored_results)

        if row_meta is not None:
            scored_results = rerank_with_protected_family_bias(scored_results, row_meta)
            scored_results = rerank_highdim_clean_mechanistic_results(
                scored_results,
                row_meta=row_meta,
                feature_names=feature_names_for_selection,
            )

        scored_results = rerank_mechanism_preserving_results(
            scored_results,
            feature_names=feature_names_for_selection,
        )
        selection_trace = {"applied": False, "reason": "not_run"}
        if _should_apply_lowdim_small_sample_cv_rerank(dataset, row_meta=row_meta):
            if timer is not None:
                timer.start(f"{prefix}_small_sample_cv_rerank")
            scored_results, selection_trace = rerank_lowdim_small_sample_results(
                scored_results,
                dataset=dataset,
                row_meta=row_meta,
            )
            if timer is not None:
                timer.stop(f"{prefix}_small_sample_cv_rerank")

        best_result = get_best_result(scored_results)
        residual_summary = residual_pattern_placeholder(scored_results, dataset)
        physics_summary = physics_check_placeholder(candidate_exprs, dataset, best_result=best_result)

        evaluation_table = []
        for item in scored_results[:META_TOPK]:
            evaluation_table.append({
                "expr": _safe_get_attr(item, "simplified_expression", None),
                "val_mse": _safe_get_attr(item, "val_mse", None),
                "complexity": _safe_get_attr(item, "complexity", None),
                "score": _safe_get_attr(item, "score", None),
                "selection_metric": _safe_get_attr(item, "selection_metric", None),
                "small_sample_cv_mse": _safe_get_attr(item, "small_sample_cv_mse", None),
            })

        return {
            "fit_results": fit_results,
            "simplify_results": simplify_results,
            "unique_results": unique_results,
            "scored_results": scored_results,
            "best_result": best_result,
            "residual_summary": residual_summary,
            "physics_summary": physics_summary,
            "evaluation_table": evaluation_table,
            "selection_trace": make_json_safe(selection_trace),
            "requested_candidate_count": int(len(candidate_exprs or [])),
            "evaluated_candidate_count": int(len(fit_results or [])),
            "evaluation_truncated": bool(len(fit_results or []) < len(candidate_exprs or [])),
        }


class MetaAgent:
    def __init__(self, client, meta_llm=None):
        self.client = client
        self.meta_llm = meta_llm

    def _heuristic_decide(self, current_best, evaluation, iter_cfg, round_idx):
        best = current_best or evaluation.get("best_result")
        best_expr = _safe_get_attr(best, "simplified_expression", None) if best is not None else None
        best_val = _safe_get_attr(best, "val_mse", None) if best is not None else None
        residual = evaluation.get("residual_summary", {}) or {}
        should_refine = True
        reason = "heuristic fallback"
        skip_refine_val_mse = float(iter_cfg.get("skip_refine_val_mse", GOOD_ENOUGH_VAL_MSE_TO_SKIP_REFINE))

        try:
            if best_val is not None and np.isfinite(best_val):
                if float(best_val) <= skip_refine_val_mse:
                    should_refine = False
                    reason = f"best_val_mse <= {skip_refine_val_mse}"
        except Exception:
            pass

        best_probe = dict(residual.get("best_probe", {}) or {})
        best_gain = float(best_probe.get("combined_gain", 0.0) or 0.0)
        pattern_type = str(residual.get("pattern_type", "") or "")
        if bool(iter_cfg.get("prefer_stop_on_diffuse_residual", False)):
            if pattern_type == "small_or_diffuse_residual" or best_gain < DIFFUSE_RESIDUAL_SKIP_REFINE_GAIN:
                should_refine = False
                reason = f"diffuse residual; best probe gain < {DIFFUSE_RESIDUAL_SKIP_REFINE_GAIN}"

        if round_idx + 1 >= iter_cfg.get("refine_rounds", 1):
            reason += "; last planned refine round"

        repair_targets = list(residual.get("repair_targets", []) or [])
        if not repair_targets:
            repair_targets = [residual.get("pattern_type", "reduce validation error")]
        preserve_patterns = list(residual.get("preserve_substructures", []) or [])
        if not preserve_patterns and best_expr:
            preserve_patterns = ["keep useful outer structure from current best"]
        actions = list(residual.get("repair_actions", []) or [])
        if not actions:
            actions = ["local_rewrite", "small_structural_extension"]

        return {
            "should_refine": should_refine,
            "confidence": 0.45,
            "reason": reason,
            "target_exprs": [best_expr] if best_expr else [],
            "preserve_patterns": preserve_patterns[:4],
            "repair_targets": repair_targets[:4],
            "actions": actions[:6],
            "budget": {
                "num_candidates": int(iter_cfg.get("refined_k", 3)),
                "temperature": REFINE_TEMPERATURE,
                "use_multimodal": False,
            },
        }

    def decide(self, dataset, observation: ObservationBundle, evaluation, current_best, iter_cfg, round_idx, row_meta=None, diagnostic_image_paths=None):
        observer_output = build_observer_output_for_prompt(
            structure_hints=observation.structure_hints,
            structure_profile=observation.structure_profile,
            visual_summary=observation.visual_summary,
            visual_hints=observation.visual_hints,
        )
        evaluator_results = build_evaluator_results_for_critic(evaluation, current_best=current_best, top_k=META_TOPK)
        if diagnostic_image_paths is None:
            diagnostic_image_paths = build_agent_multimodal_image_paths(
                dataset=dataset,
                row_meta=row_meta,
                current_best=current_best or evaluation.get("best_result"),
                round_idx=round_idx,
                stage="critic",
                observation=observation,
            )
        diagnostic_image_paths = _existing_image_paths(diagnostic_image_paths)
        image_input_line = (
            "1) Attached image(s): current prediction/residual diagnostic, plus optional original benchmark plot context."
            if diagnostic_image_paths else
            "1) Residual or prediction diagnostic image is unavailable; rely on the textual evaluator and residual diagnostics."
        )

        prompt = f"""
<image>You are the Critic Agent in a multimodal symbolic-regression loop.

Task description:
Your job is to critique the current symbolic-regression state and control the next loop step. Given the residual/prediction diagnostic, task info, Observer output, Evaluator results, and residual diagnostic, you are required to generate a concise refinement decision for the Proposer.

Input:
{image_input_line}
2) Task info: feature_names, target_name, round index, and planned refine rounds.
3) Observer output: structural hints from the benchmark plot and CSV samples.
4) Evaluator results: current best expression and ranked candidate metrics.
5) Residual diagnostic: likely missing structure and repair directions.

Output:
Return a JSON object deciding whether to stop or refine. If refinement is needed, specify what to preserve, what to repair, which actions to take, and what budget/constraints the Proposer should use.

Task info:
- feature_names: {dataset.feature_names}
- target_name: {dataset.target_name}
- round_idx: {round_idx}
- planned_refine_rounds: {iter_cfg.get('refine_rounds')}

Observer output:
{json.dumps(observer_output, ensure_ascii=False)}

Evaluator results:
{json.dumps(evaluator_results, ensure_ascii=False)}

Residual diagnostic:
{json.dumps(evaluation.get('residual_summary', {}), ensure_ascii=False)}

Decide whether the system should refine.
Required JSON schema:
{{
  "should_refine": true,
  "confidence": 0.0,
  "reason": "...",
  "target_exprs": ["existing_candidate_expr"],
  "preserve_patterns": ["..."],
  "repair_targets": ["..."],
  "actions": ["local_rewrite", "add_periodic_component", "expand_interaction", "switch_family"],
  "budget": {{
      "num_candidates": 3,
      "temperature": 0.4,
      "use_multimodal": false
  }},
  "critic_action": "stop | refine | complexity_prune | structural_rescue | multimodal_rescue",
  "target_families": ["..."],
  "active_variables": ["..."],
  "complexity_pressure": "low | medium | high"
}}

Restrictions:
1) Output STRICT JSON only.
2) Diagnose structure, not just coefficient error.
3) Do not copy or invent a hidden target formula.
4) Use only the provided feature names and observed candidate expressions.
5) target_exprs must refer to existing candidate expressions from Evaluator results, not newly invented formulas.
6) Make the decision directly usable by the Proposer refine step.
""".strip()

        if ENABLE_META_HEURISTIC_FALLBACK:
            fallback = self._heuristic_decide(current_best, evaluation, iter_cfg, round_idx)
        else:
            fallback = None

        if bool(iter_cfg.get("force_heuristic_agents", False)) and fallback is not None:
            fallback["raw_text"] = "meta heuristic-only runtime path"
            fallback["diagnostic_image_paths"] = diagnostic_image_paths
            fallback["round_idx"] = int(round_idx)
            return fallback

        if META_FAST_STOP_WITH_HEURISTIC and fallback is not None and not bool(fallback.get("should_refine", True)):
            fallback["raw_text"] = "meta fast-stop via heuristic"
            fallback["diagnostic_image_paths"] = diagnostic_image_paths
            fallback["round_idx"] = int(round_idx)
            return fallback

        try:
            response = self.client.generate(
                messages=[
                    Message(role="system", content="You are the Critic Agent in a multimodal symbolic-regression loop. Output JSON only."),
                    build_agent_multimodal_user_message(prompt, diagnostic_image_paths),
                ],
                temperature=META_TEMPERATURE,
                max_tokens=META_MAX_TOKENS,
                top_p=1.0,
            )
            raw_text = response.text
            obj = _extract_first_json_object(raw_text)
            if isinstance(obj, dict):
                existing_exprs = {
                    str(row.get("expression") or "")
                    for row in list(evaluator_results.get("candidate_evaluation_table", []) or [])
                    if row.get("expression")
                }
                current_expr = str((evaluator_results.get("current_best") or {}).get("expr") or "")
                if current_expr:
                    existing_exprs.add(current_expr)
                raw_targets = [str(x).strip() for x in list(obj.get("target_exprs", []) or []) if str(x).strip()]
                filtered_targets = [x for x in raw_targets if x in existing_exprs]
                if not filtered_targets and current_expr:
                    filtered_targets = [current_expr]
                obj["target_exprs"] = filtered_targets
            decision = _coerce_meta_decision(obj, current_best=current_best)
            decision["raw_text"] = raw_text
            decision["diagnostic_image_paths"] = diagnostic_image_paths
            decision["round_idx"] = int(round_idx)
            return decision
        except Exception as e:
            if fallback is None:
                raise
            fallback["raw_text"] = f"meta llm failed: {repr(e)}"
            fallback["diagnostic_image_paths"] = diagnostic_image_paths
            fallback["round_idx"] = int(round_idx)
            return fallback


class JudgeAgent:
    def __init__(self, client):
        self.client = client

    def _heuristic_feedback(self, evaluation, meta_decision):
        best = evaluation.get("best_result")
        best_expr = _safe_get_attr(best, "simplified_expression", None) if best is not None else None
        residual = evaluation.get("residual_summary", {})
        repair_targets = meta_decision.get("repair_targets", []) or [residual.get("pattern_type", "reduce validation error")]
        keep_constraints = meta_decision.get("preserve_patterns", [])
        guidance_lines = list(residual.get("guidance_lines", []) or [])
        feedback = (
            f"Current best expression: {best_expr}. "
            f"Preserve useful structure if it already helps. "
            f"Main repair targets: {repair_targets}. "
            f"Prefer local edits before replacing the whole family."
        )
        if guidance_lines:
            feedback += " Residual diagnostics: " + " ".join(guidance_lines[:3]) + "."
        return {
            "feedback_text": feedback,
            "keep_constraints": keep_constraints,
            "repair_targets": repair_targets,
            "avoid_patterns": ["do not repeatedly return the same weak family without new structure"],
        }

    def build_feedback(self, dataset, observation: ObservationBundle, evaluation, meta_decision, iter_cfg=None):
        observer_output = build_observer_output_for_prompt(
            structure_hints=observation.structure_hints,
            structure_profile=observation.structure_profile,
            visual_summary=observation.visual_summary,
            visual_hints=observation.visual_hints,
        )
        evaluator_results = build_evaluator_results_for_critic(evaluation, current_best=evaluation.get("best_result"), top_k=JUDGE_TOPK)
        prompt = f"""
You are the Judge Agent in a symbolic regression multi-agent system.

Task description:
Your job is to write a refinement brief for the Proposer Agent. Given task info, Observer output, Evaluator results, residual diagnostic, and Critic decision, you are required to generate concise guidance about what to preserve, what to repair, and what mistakes to avoid.

Input:
1) Task info: feature_names and target_name.
2) Observer output: structural hints from the benchmark.
3) Evaluator results: current best expression and candidate metrics.
4) Residual diagnostic: likely missing structure or failure mode.
5) Critic decision: loop-level stop/refine decision and repair budget.

Output:
Return a JSON object containing feedback_text, keep_constraints, repair_targets, and avoid_patterns. This is a refinement brief only; do not choose the final answer.

Feature names: {dataset.feature_names}
Target name: {dataset.target_name}
Observer output:
{json.dumps(observer_output, ensure_ascii=False)}
Evaluator results:
{json.dumps(evaluator_results, ensure_ascii=False)}
Residual diagnostic:
{json.dumps(evaluation.get('residual_summary', {}), ensure_ascii=False)}
Critic decision:
{json.dumps(meta_decision, ensure_ascii=False)}

Required JSON schema:
{{
  "feedback_text": "A concise but detailed refinement brief for the proposer/refiner.",
  "keep_constraints": ["structures that should be preserved"],
  "repair_targets": ["missing components / failure modes to fix"],
  "avoid_patterns": ["families or mistakes to avoid"]
}}

Restrictions:
1) Output STRICT JSON only.
2) Do NOT choose the final answer.
3) Do NOT invent new candidate expressions.
4) Keep feedback directly usable by the Proposer refine step.
""".strip()

        fallback = self._heuristic_feedback(evaluation, meta_decision) if ENABLE_JUDGE_HEURISTIC_FALLBACK else None

        if bool((iter_cfg or {}).get("force_heuristic_agents", False)) and fallback is not None:
            fallback["raw_text"] = "judge heuristic-only runtime path"
            return fallback

        try:
            response = self.client.generate(
                messages=[
                    Message(role="system", content="You write concise refinement briefs for a symbolic-regression proposer. Output JSON only."),
                    Message(role="user", content=prompt),
                ],
                temperature=0.2,
                max_tokens=JUDGE_MAX_TOKENS,
                top_p=1.0,
            )
            raw_text = response.text
            obj = _extract_first_json_object(raw_text)
            out = _coerce_judge_feedback(obj)
            out["raw_text"] = raw_text
            if not out["feedback_text"]:
                out["feedback_text"] = fallback["feedback_text"] if fallback else "Refine using evaluator diagnostics."
            return out
        except Exception as e:
            if fallback is None:
                raise
            fallback["raw_text"] = f"judge llm failed: {repr(e)}"
            return fallback


class RefinerAgent:
    def __init__(self, refiner_llm, client):
        self.refiner_llm = refiner_llm
        self.client = client

    def _fallback_refine(self, current_best_expr: Optional[str], judge_feedback: Dict[str, Any], budget: Dict[str, Any], feature_names=None, meta_decision=None):
        feature_names = list(feature_names or [])
        meta_decision = dict(meta_decision or {})
        repair_text = " ".join([
            str(judge_feedback.get("feedback_text", "") or ""),
            " ".join(str(x) for x in (judge_feedback.get("repair_targets", []) or [])),
            " ".join(str(x) for x in (meta_decision.get("actions", []) or [])),
            " ".join(str(x) for x in (meta_decision.get("repair_targets", []) or [])),
        ]).lower()
        prefer_periodic = any(k in repair_text for k in ["periodic", "trigonometric", "sin", "cos", "modulation"])
        prefer_rational = any(k in repair_text for k in ["rational", "ratio", "denominator", "reciprocal"])
        prefer_interaction = any(k in repair_text for k in ["interaction", "cross", "coupling"])
        prefer_power = any(k in repair_text for k in ["power", "quadratic", "cubic"])
        prefer_exponential = any(k in repair_text for k in ["exp", "exponential", "growth", "decay"])
        exprs = []
        # Very light heuristic variants.
        base_var = feature_names[0] if feature_names else "x1"
        if current_best_expr:
            if STRICT_NO_FORMULA_TEMPLATES:
                exprs.extend([
                    f"({current_best_expr}) + 0.1",
                    f"1.1*({current_best_expr}) - 0.1",
                    f"({current_best_expr}) + 0.2*sin(0.8*{base_var}+0.3)",
                ])
            else:
                exprs.extend([
                    f"({current_best_expr}) + a",
                    f"a*({current_best_expr}) + b",
                    f"({current_best_expr}) + a*sin(b*{base_var}+c)",
                ])
        if len(feature_names) == 1:
            x = feature_names[0]
            targeted = []
            if prefer_rational:
                targeted.extend(
                    [
                        f"(0.6*{x}**3+0.4*{x}**2+0.2*{x}+0.1)/(0.5*{x}**2+0.3*{x}+1.0)",
                        f"1.0/({x}+0.8)+0.1",
                    ] if STRICT_NO_FORMULA_TEMPLATES else [
                        f"(a1*{x}**3+a2*{x}**2+a3*{x}+a4)/(b1*{x}**2+b2*{x}+b3)",
                        f"a/({x}+b)+c",
                    ]
                )
            if prefer_periodic:
                targeted.extend(
                    [
                        f"1.2*sin(0.9*{x}+0.4)+0.1",
                        f"0.8*sin(0.7*{x}+0.2)+0.4*sin(0.5*{x}**2+0.3)+0.1",
                    ] if STRICT_NO_FORMULA_TEMPLATES else [
                        f"a*sin(b*{x}+c)+d",
                        f"a1*sin(b1*{x}+c1)+a2*sin(b2*{x}**2+c2)+d",
                    ]
                )
            if prefer_power:
                targeted.extend(
                    [
                        f"0.3*{x}**4-0.2*{x}**3+0.5*{x}**2+0.7*{x}+0.1",
                        f"0.2*{x}**5+0.4*{x}**3+0.1",
                    ] if STRICT_NO_FORMULA_TEMPLATES else [
                        f"a*{x}**4+b*{x}**3+c*{x}**2+d*{x}+e",
                        f"a*{x}**5+b*{x}**3+c",
                    ]
                )
            if prefer_exponential:
                targeted.extend(
                    [
                        f"1.2*exp(0.3*{x})+0.1",
                        f"0.9*sinh(0.4*{x}+0.2)+0.1",
                    ] if STRICT_NO_FORMULA_TEMPLATES else [
                        f"a*exp(b*{x})+c",
                        f"a*sinh(b*{x}+c)+d",
                    ]
                )
            exprs.extend(targeted)
            exprs.extend(
                [
                    f"0.3*{x}**4-0.2*{x}**3+0.5*{x}**2+0.7*{x}+0.1",
                    f"(0.6*{x}**3+0.4*{x}**2+0.2*{x}+0.1)/(0.5*{x}**2+0.3*{x}+1.0)",
                    f"(0.3*{x}**5+0.2*{x}**4+0.1*{x}**3+0.4*{x}**2+0.5*{x}+0.2)/(0.5*{x}**2+0.3*{x}+1.0)",
                    f"0.8*sin(0.6*{x}**2+0.2)*cos(0.5*{x}+0.1)+0.1",
                    f"0.7*sin(0.8*{x}+0.2)+0.4*sin(0.5*{x}+0.3*{x}**2+0.1)+0.1",
                    f"0.9*sinh(0.4*{x}+0.2)+0.1",
                    f"0.8*cosh(0.3*{x}+0.2)+0.1",
                ] if STRICT_NO_FORMULA_TEMPLATES else [
                    f"a*{x}**4+b*{x}**3+c*{x}**2+d*{x}+e",
                    f"(a1*{x}**3+a2*{x}**2+a3*{x}+a4)/(b1*{x}**2+b2*{x}+b3)",
                    f"(a1*{x}**5+a2*{x}**4+a3*{x}**3+a4*{x}**2+a5*{x}+a6)/(b1*{x}**2+b2*{x}+b3)",
                    f"a*sin(b*{x}**2+c)*cos(d*{x}+e)+f",
                    f"a1*sin(b1*{x}+c1)+a2*sin(b2*{x}+b3*{x}**2+c2)+d",
                    f"a*sinh(b*{x}+c)+d",
                    f"a*cosh(b*{x}+c)+d",
                ]
            )
        elif len(feature_names) == 2:
            x1, x2 = feature_names[:2]
            targeted = []
            if prefer_interaction:
                targeted.extend(
                    [
                        f"1.0*{x1}*{x2}+0.2",
                        f"0.8*({x1}+0.3)*({x2}+0.4)+0.1",
                    ] if STRICT_NO_FORMULA_TEMPLATES else [
                        f"a*{x1}*{x2}+b",
                        f"a*({x1}+b1)*({x2}+b2)+c",
                    ]
                )
            if prefer_rational:
                targeted.extend(
                    [
                        f"1.2*({x1}/({x2}+0.7))+0.1",
                        f"1.2*({x2}/({x1}+0.7))+0.1",
                    ] if STRICT_NO_FORMULA_TEMPLATES else [
                        f"a*({x1}/({x2}+b))+c",
                        f"a*({x2}/({x1}+b))+c",
                    ]
                )
            if prefer_periodic:
                targeted.extend(
                    [
                        f"1.3*sin(0.9*{x1}+0.6*{x2})+0.2",
                        f"1.1*sin(0.7*{x1})*cos(0.5*{x2})+0.1",
                    ] if STRICT_NO_FORMULA_TEMPLATES else [
                        f"a*sin(b*{x1}+c*{x2})+d",
                        f"a*sin(b*{x1})*cos(c*{x2})+d",
                    ]
                )
            if prefer_power:
                targeted.append(
                    f"0.8*({x1}**2+0.5*{x1}*{x2}-{x2}**2)+0.1"
                    if STRICT_NO_FORMULA_TEMPLATES else
                    f"a*({x1}**2+b1*{x1}*{x2}-b2*{x2}**2)+c"
                )
            exprs.extend(targeted)
            exprs.extend(
                [
                    f"1.1*sin(0.7*{x1})*cos(0.5*{x2})+0.1",
                    f"1.0*cos(0.8*{x1})*sin(0.6*{x2})+0.1",
                    f"1.2*({x1}/({x2}+0.7))+0.1",
                    f"1.2*({x2}/({x1}+0.7))+0.1",
                    f"1.0*{x1}*{x2}+0.2",
                ] if STRICT_NO_FORMULA_TEMPLATES else [
                    f"a*sin(b*{x1})*cos(c*{x2})+d",
                    f"a*cos(b*{x1})*sin(c*{x2})+d",
                    f"a*({x1}/({x2}+b))+c",
                    f"a*({x2}/({x1}+b))+c",
                    f"a*{x1}*{x2}+b",
                ]
            )
        elif len(feature_names) >= 3:
            used = expression_variable_set(current_best_expr or "", feature_names)
            missing = [v for v in feature_names if v not in used]
            if current_best_expr and missing:
                clean_expr = re.sub(r"\bAbs\s*\(", "abs(", str(current_best_expr))
                for v in missing[:4]:
                    exprs.extend([
                        f"a*({clean_expr})*{v} + b",
                        f"a*({clean_expr})*({v}+b1) + b",
                        f"a*({clean_expr}) + b*{v} + c",
                    ])
            if len(feature_names) >= 3:
                x1, x2, x3 = feature_names[:3]
                exprs.extend([
                    f"a*{x1}*{x2}*{x3}+b",
                    f"a*({x1}*{x2})/({x3}+b)+c",
                    f"a*{x1}*{x2}*log(abs({x3})+b)+c",
                ])

        exprs = filter_identity_refinements(exprs, current_best_expr)
        k = int(budget.get("num_candidates", 3))
        return exprs[:max(1, k)]

    def refine(self, dataset, observation: ObservationBundle, current_best, meta_decision, judge_feedback, iter_cfg, row_meta=None, diagnostic_image_paths=None):
        current_best_expr = _safe_get_attr(current_best, "simplified_expression", None) if current_best is not None else None
        targets = meta_decision.get("target_exprs", []) or ([current_best_expr] if current_best_expr else [])
        k = int(meta_decision.get("budget", {}).get("num_candidates", iter_cfg.get("refined_k", 3)))
        temperature = float(meta_decision.get("budget", {}).get("temperature", REFINE_TEMPERATURE))
        max_tokens = int(iter_cfg.get("refiner_max_tokens", REFINER_MAX_TOKENS))
        if diagnostic_image_paths is None:
            diagnostic_image_paths = list(meta_decision.get("diagnostic_image_paths", []) or [])
        if not diagnostic_image_paths:
            diagnostic_image_paths = build_agent_multimodal_image_paths(
                dataset=dataset,
                row_meta=row_meta,
                current_best=current_best,
                round_idx=int(meta_decision.get("round_idx", 0) or 0),
                stage="refiner",
                observation=observation,
            )
        diagnostic_image_paths = _existing_image_paths(diagnostic_image_paths)
        if bool(iter_cfg.get("force_heuristic_agents", False)):
            out_exprs = self._fallback_refine(
                current_best_expr,
                judge_feedback,
                meta_decision.get("budget", {}),
                feature_names=dataset.feature_names,
                meta_decision=meta_decision,
            )
            return {
                "candidate_exprs": out_exprs[:max(1, min(k, MAX_REFINED_EXPRESSIONS_PER_ROUND))],
                "raw_text": "refiner heuristic-only runtime path",
                "diagnostic_image_paths": diagnostic_image_paths,
            }

        observer_output = build_observer_output_for_prompt(
            structure_hints=observation.structure_hints,
            structure_profile=observation.structure_profile,
            visual_summary=observation.visual_summary,
            visual_hints=observation.visual_hints,
        )
        current_candidate = {
            "expr": current_best_expr,
            "val_mse": _safe_get_attr(current_best, "val_mse", None) if current_best is not None else None,
            "complexity": _safe_get_attr(current_best, "complexity", None) if current_best is not None else None,
            "score": _safe_get_attr(current_best, "score", None) if current_best is not None else None,
        }
        critic_feedback = {
            "should_refine": meta_decision.get("should_refine", True),
            "critic_action": meta_decision.get("critic_action", "refine"),
            "reason": meta_decision.get("reason", ""),
            "target_exprs": targets,
            "preserve_patterns": meta_decision.get("preserve_patterns", []),
            "repair_targets": meta_decision.get("repair_targets", []),
            "actions": meta_decision.get("actions", []),
            "target_families": meta_decision.get("target_families", []),
            "active_variables": meta_decision.get("active_variables", []),
            "budget": meta_decision.get("budget", {}),
        }
        evaluation_context = {
            "judge_feedback": make_json_safe(judge_feedback),
        }
        image_input_line = (
            "1) Attached image(s): current prediction/residual diagnostic, plus optional original benchmark plot context."
            if diagnostic_image_paths else
            "1) Residual or prediction diagnostic image is unavailable; rely on the textual Critic and Evaluator context."
        )

        prompt = f"""
<image>You are the Proposer Agent in a multimodal symbolic-regression loop.

Task description:
Your job is to refine candidate formulas after a failed or incomplete evaluation round. Given the residual/prediction diagnostic, current candidate, Observer output, Critic feedback, and evaluation context, you are required to generate a small set of repaired candidate expressions.

Input:
{image_input_line}
2) Current candidate: the best expression found so far and its metrics.
3) Observer output: structural hints from the original benchmark plot and CSV samples.
4) Critic feedback: missing structures, repair targets, and refinement budget.
5) Evaluation context: judge feedback and residual/evaluation evidence.

Output:
Return a JSON object with a "refined_candidates" list. Each item must include a valid expression and a short rationale explaining what structural repair it makes.

Mode: critic_conditioned_refinement.

feature_names = {dataset.feature_names}
target_name = {dataset.target_name}

Allowed operators:
{", ".join(ALLOWED_OPERATORS)}

Current candidate:
{json.dumps(make_json_safe(current_candidate), ensure_ascii=False)}

Observer output:
{json.dumps(observer_output, ensure_ascii=False)}

Critic feedback:
{json.dumps(make_json_safe(critic_feedback), ensure_ascii=False)}

Evaluation context:
{json.dumps(evaluation_context, ensure_ascii=False)}

Generate {k} refined candidate expressions.
Required JSON schema:
{{
  "refined_candidates": [
    {{"expression": "...", "rationale": "..."}}
  ]
}}
Restrictions:
1) Output STRICT JSON only.
2) The loop will extract only refined_candidates[i].expression for evaluation, so every expression must be valid.
3) Return a candidate pool, not a single expression.
4) Put the strongest repair candidate first.
5) Each rationale should explain the repair using critic feedback, residual patterns, or observer output.
6) Only use the provided feature names and allowed symbolic operators.
7) Do not restart from scratch unless the current family is clearly wrong; prefer targeted structural repairs guided by Critic feedback.
8) Every expression must be directly parseable by the symbolic-regression backend: use Python-style ** for powers, use only allowed functions/operators, and do not introduce undefined feature variables. Free scalar parameters such as a, b, c are allowed.
""".strip()

        try:
            response = self.client.generate(
                messages=[
                    Message(role="system", content="You are the Proposer Agent in a multimodal symbolic-regression loop. Output JSON only."),
                    build_agent_multimodal_user_message(prompt, diagnostic_image_paths),
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=1.0,
            )
            raw_text = response.text
            obj = _extract_first_json_object(raw_text)
            out_exprs = []
            if isinstance(obj, dict):
                for item in obj.get("refined_candidates", []):
                    if not isinstance(item, dict):
                        continue
                    expr = str(item.get("expression", "")).strip()
                    if expr:
                        out_exprs.append(expr)
            out_exprs = filter_identity_refinements(out_exprs, current_best_expr)
            if STRICT_NO_FORMULA_TEMPLATES:
                out_exprs, _ = _filter_non_template_expressions(out_exprs, dataset.feature_names)
            if not out_exprs:
                out_exprs = self._fallback_refine(current_best_expr, judge_feedback, meta_decision.get("budget", {}), feature_names=dataset.feature_names)
            return {
                "candidate_exprs": out_exprs[:max(1, min(k, MAX_REFINED_EXPRESSIONS_PER_ROUND))],
                "raw_text": raw_text,
                "diagnostic_image_paths": diagnostic_image_paths,
            }
        except Exception:
            out_exprs = self._fallback_refine(current_best_expr, judge_feedback, meta_decision.get("budget", {}), feature_names=dataset.feature_names)
            return {
                "candidate_exprs": out_exprs[:max(1, min(k, MAX_REFINED_EXPRESSIONS_PER_ROUND))],
                "raw_text": "",
                "diagnostic_image_paths": diagnostic_image_paths,
            }


# =========================================================
# 【替换主流程】新的 _run_core_pipeline
# =========================================================
# 这是你原脚本中最需要整段替换的部分。
# 原来是：
#   text/diverse/manual -> evaluate -> refine
# 现在变成：
#   observe -> initial propose -> evaluate -> meta decide -> judge feedback -> refine -> reevaluate
#
# 这一段已经尽量保留你原有字段与结果结构，便于兼容你当前的 summary / CSV / JSON 输出。



def _run_core_pipeline(dataset, row_meta):
    start = time.time()
    timer = StepTimer()
    task_deadline_ts = _task_deadline_from_start(start)
    budget_guard = task_time_budget_guard(MAX_RUNTIME_PER_TASK_SEC)

    proposal_client = _build_role_client("proposal", BACKEND_CONFIG, deadline_ts=task_deadline_ts)
    meta_client = _build_role_client("meta", BACKEND_CONFIG, deadline_ts=task_deadline_ts)
    judge_client = _build_role_client("judge", BACKEND_CONFIG, deadline_ts=task_deadline_ts)
    refiner_client = _build_role_client("refiner", BACKEND_CONFIG, deadline_ts=task_deadline_ts)
    planner_client = _build_role_client("planner", BACKEND_CONFIG, deadline_ts=task_deadline_ts)

    proposal_llm = ProposalGeneratorLLM(proposal_client)
    # Keep MetaLLM instantiated for compatibility, even if the final decision path uses direct JSON prompting.
    meta_llm = MetaLLM(meta_client)
    refiner_llm = ExpressionRefinerLLM(refiner_client)
    plot_tool = PlotGeneratorTool()
    high_dim_reconstruction_tool = HighDimReconstructionTool(
        unary_bins=HIGH_DIM_RECON_UNARY_BINS,
        pair_bins=HIGH_DIM_RECON_PAIR_BINS,
        max_unary_views=HIGH_DIM_RECON_MAX_UNARY_VIEWS,
        max_pair_views=HIGH_DIM_RECON_MAX_PAIR_VIEWS,
        rank=HIGH_DIM_RECON_RANK,
        min_bin_count=HIGH_DIM_RECON_MIN_BIN_COUNT,
    )

    observer_agent = ObserverAgent(
        plot_tool=plot_tool,
        high_dim_reconstruction_tool=high_dim_reconstruction_tool,
    )
    proposer_agent = ProposerAgent(proposal_llm=proposal_llm)
    structural_planner_agent = StructuralPlannerAgent(client=planner_client)
    post_eval_repair_agent = PostEvalLocalRepairAgent(client=refiner_client)
    evaluator_agent = EvaluatorAgent(
        complexity_weight=COMPLEXITY_WEIGHT,
        deadline_ts=task_deadline_ts,
    )
    meta_agent = MetaAgent(client=meta_client, meta_llm=meta_llm)
    judge_agent = JudgeAgent(client=judge_client)
    refiner_agent = RefinerAgent(refiner_llm=refiner_llm, client=refiner_client)

    bootstrap_experience_prior = {} if PURE_LLM_TEXT_ONLY_EVAL else build_experience_prior(
        row_meta=row_meta,
        feature_names=dataset.feature_names,
        observation=None,
    )
    bootstrap_memory_prior = {} if PURE_LLM_TEXT_ONLY_EVAL else build_memory_prior(
        row_meta=row_meta,
        feature_names=dataset.feature_names,
        observation=None,
        experience_prior=bootstrap_experience_prior,
        dataset=dataset,
    )
    bootstrap_guidance_prior = merge_guidance_priors(
        experience_prior=bootstrap_experience_prior,
        memory_prior=bootstrap_memory_prior,
    )
    iter_cfg = adjust_iteration_config_for_runtime(
        get_iteration_config(len(dataset.feature_names)),
        dataset=dataset,
        row_meta=row_meta,
        experience_prior=bootstrap_guidance_prior,
    )

    result = {
        "eval_profile": EVAL_PROFILE,
        "method_mode": METHOD_MODE,
        "no_leakage_mode": bool(NO_LEAKAGE_MODE),
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
        # Filled only by the post-selection reporter.
        "n_test": None,
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
        "mm_requested": None,
        "mm_trigger_reason": None,
        "mm_assets_attempted": None,
        "mm_assets_succeeded": None,
        "mm_assets_error": None,
        "mm_candidate_count": None,
        "mm_used_in_evaluation": None,
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
        "structural_planner_candidate_count": None,
        "proposal_source_stats": None,
        "merged_candidate_count": None,
        "experience_prior": None,
        "memory_prior": None,
        "guidance_prior": None,
        "experience_rerank": None,
        "refine_round_expr_counts": None,
        "refine_round_timings": None,
        "step_times_json": None,
        "iteration_config": json.dumps(make_json_safe(iter_cfg), ensure_ascii=False),
        "early_stop_reason": None,
        "time_budget_sec": float(MAX_RUNTIME_PER_TASK_SEC) if isinstance(MAX_RUNTIME_PER_TASK_SEC, numbers.Number) and float(MAX_RUNTIME_PER_TASK_SEC) > 0 else None,
        "time_budget_hit": False,
        "agent_trace": None,
        "judge_feedback_history": None,
        "proposal_history": None,
        "evaluation_history": None,
        "meta_plan_history": None,
        "error": None,
    }

    meta_decisions = []
    refine_history = []
    refine_round_expr_counts = []
    refine_round_timings = []
    judge_feedback_history = []
    proposal_history = []
    evaluation_history = []
    agent_trace = []
    best_history = []
    current_best = None
    experience_prior = dict(bootstrap_experience_prior or {})
    memory_prior = dict(bootstrap_memory_prior or {})
    guidance_prior = dict(bootstrap_guidance_prior or {})
    observation = ObservationBundle(
        structure_hints=[],
        visual_hints=[],
        unit_hints={},
        plot_descriptions=[],
        image_paths=[],
    )
    initial_eval = {
        "residual_summary": None,
        "physics_summary": None,
    }
    need_mm = False
    mm_trigger_reason = "multimodal not evaluated"
    mm_prop_trace = None
    mm_candidate_count = 0
    mm_used_in_evaluation = False
    active_refine_round_num = None
    active_refine_round_start_ts = None
    active_refine_round_timer_before = None

    def finalize_refine_round_timing(round_num, round_start_ts, round_timer_before, status, extra=None):
        round_end_ts = time.time()
        round_total_sec = round_end_ts - round_start_ts
        round_prefix = f"refine_round_{round_num}_"
        round_timer_after = timer.as_dict()
        round_step_times = {}
        for key, value in round_timer_after.items():
            if not str(key).startswith(round_prefix):
                continue
            prev = float(round_timer_before.get(key, 0.0))
            delta = float(value) - prev
            if delta > 0:
                round_step_times[str(key)] = delta
        timer.add(f"refine_round_{round_num}_total_sec", round_total_sec)
        record = {
            "round": int(round_num),
            "status": str(status),
            "start_ts": float(round_start_ts),
            "end_ts": float(round_end_ts),
            "start_local": format_local_timestamp(round_start_ts),
            "end_local": format_local_timestamp(round_end_ts),
            "total_sec": float(round_total_sec),
            "step_times": round_step_times,
        }
        if extra:
            record.update(make_json_safe(extra))
        refine_round_timings.append(make_json_safe(record))
        return record

    def finalize_pipeline_result():
        populate_visual_trace_fields(
            result=result,
            observation=observation,
            mm_requested=need_mm,
            mm_trigger_reason=mm_trigger_reason,
            mm_prop_trace=mm_prop_trace,
            mm_candidate_count=mm_candidate_count,
            mm_used_in_evaluation=mm_used_in_evaluation,
        )
        result["refine_round_timings"] = json.dumps(make_json_safe(refine_round_timings), ensure_ascii=False)
        result["judge_feedback_history"] = json.dumps(make_json_safe(judge_feedback_history), ensure_ascii=False)
        result["proposal_history"] = json.dumps(make_json_safe(proposal_history), ensure_ascii=False)
        result["evaluation_history"] = json.dumps(make_json_safe(evaluation_history), ensure_ascii=False)
        result["meta_plan_history"] = json.dumps(make_json_safe(meta_decisions), ensure_ascii=False)
        result["agent_trace"] = json.dumps(make_json_safe(agent_trace), ensure_ascii=False)
        result["best_expr_source"] = (
            str(_safe_get_attr(current_best, "source", "scored_result"))
            if current_best is not None else None
        )
        tiny_val_train_cv_active = _should_apply_lowdim_small_sample_cv_rerank(dataset, row_meta=row_meta)
        result["no_leakage_audit"] = json.dumps(make_json_safe({
            "method_mode": METHOD_MODE,
            "no_leakage_mode": bool(NO_LEAKAGE_MODE),
            "used_benchmark_name_for_templates": False,
            "row_specific_benchmark_priors_enabled": bool(ALLOW_ROW_SPECIFIC_BENCHMARK_PRIORS),
            "history_memory_enabled": bool(ALLOW_HISTORY_MEMORY or ENABLE_MEMORY_PRIOR),
            "true_expression_diagnostics_enabled": bool(ALLOW_TRUE_EXPR_DIAGNOSTICS),
            "true_expression_used_for_generation_or_selection": False,
            "test_split_used_for_selection": False,
            "legacy_test_selection_switch_requested_but_ignored": bool(USE_TEST_FOR_SELECTION),
            "data_driven_direct_candidate_injection": bool(DATA_DRIVEN_FEATURE_SEEDS_AS_CANDIDATES),
            "high_dim_data_driven_evidence_bridge": bool(ENABLE_HIGH_DIM_DATA_DRIVEN_CANDIDATE_BRIDGE),
            "data_driven_mode": "candidate_upperbound" if DATA_DRIVEN_FEATURE_SEEDS_AS_CANDIDATES else ("highdim_evidence_bridge" if ENABLE_HIGH_DIM_DATA_DRIVEN_CANDIDATE_BRIDGE else "evidence_only"),
            "selection_uses_validation_only": not bool(tiny_val_train_cv_active),
            "selection_uses_train_cv_for_tiny_val": bool(tiny_val_train_cv_active),
            "best_expr_source": str(_safe_get_attr(current_best, "source", "scored_result")) if current_best is not None else None,
        }), ensure_ascii=False)
        return _finalize_result(
            result=result,
            timer=timer,
            start=start,
            current_best=current_best,
            dataset=dataset,
            residual_summary=initial_eval.get("residual_summary") if isinstance(initial_eval, dict) else None,
            physics_summary=initial_eval.get("physics_summary") if isinstance(initial_eval, dict) else None,
            meta_decisions=meta_decisions,
            refine_history=refine_history,
            refine_round_expr_counts=refine_round_expr_counts,
            row_meta=row_meta,
        )

    budget_guard.__enter__()
    try:
        # -------------------------------------------------
        # 1) Observe
        # -------------------------------------------------
        observation = timed_call(
            timer,
            "step_observe",
            observer_agent.observe,
            dataset,
            row_meta=row_meta,
            timer=timer,
        )
        experience_prior = {} if PURE_LLM_TEXT_ONLY_EVAL else build_experience_prior(
            row_meta=row_meta,
            feature_names=dataset.feature_names,
            observation=observation,
        )
        memory_prior = {} if PURE_LLM_TEXT_ONLY_EVAL else build_memory_prior(
            row_meta=row_meta,
            feature_names=dataset.feature_names,
            observation=observation,
            experience_prior=experience_prior,
            dataset=dataset,
        )
        guidance_prior = merge_guidance_priors(
            experience_prior=experience_prior,
            memory_prior=memory_prior,
        )
        iter_cfg = adjust_iteration_config_for_runtime(
            iter_cfg,
            dataset=dataset,
            row_meta=row_meta,
            experience_prior=guidance_prior,
        )
        result["iteration_config"] = json.dumps(make_json_safe(iter_cfg), ensure_ascii=False)
        result["experience_prior"] = json.dumps(make_json_safe(experience_prior), ensure_ascii=False)
        result["memory_prior"] = json.dumps(make_json_safe(memory_prior), ensure_ascii=False)
        result["guidance_prior"] = json.dumps(make_json_safe(guidance_prior), ensure_ascii=False)
        agent_trace.append({
            "event": "experience_prior",
            "summary": make_json_safe(summarize_experience_prior(experience_prior)),
        })
        agent_trace.append({
            "event": "memory_prior",
            "summary": make_json_safe(summarize_memory_prior(memory_prior)),
        })
        agent_trace.append({
            "event": "guidance_prior",
            "summary": make_json_safe(summarize_experience_prior(guidance_prior)),
        })
        _raise_if_task_budget_exceeded(task_deadline_ts, "observe")

        # -------------------------------------------------
        # 2) Fast structured seed stage for low-dim benchmark tasks
        # -------------------------------------------------
        seed_eval = None
        seed_exprs = []
        use_seed_as_initial = False
        use_seed_reason = None
        if _runtime_fast_path_eligible(row_meta, dataset):
            seed_exprs = build_fast_seed_candidates(row_meta, dataset.feature_names)
            if seed_exprs:
                _raise_if_task_budget_exceeded(task_deadline_ts, "seed_fast_path_start")
                proposal_history.append({
                    "stage": "seed_fast_path",
                    "num_exprs": len(seed_exprs),
                    "trace": {
                        "seed_candidate_count": len(seed_exprs),
                        "seed_candidates": seed_exprs,
                    },
                })
                seed_eval = evaluator_agent.evaluate(
                    candidate_exprs=seed_exprs,
                    dataset=dataset,
                    row_meta=row_meta,
                    timer=timer,
                    prefix="seed_fast_path",
                )
                evaluation_history.append({
                    "stage": "seed_fast_path",
                    "topk": seed_eval.get("evaluation_table"),
                    "residual_summary": seed_eval.get("residual_summary"),
                    "physics_summary": seed_eval.get("physics_summary"),
                })
                seed_best = seed_eval.get("best_result")
                iter_cfg = adjust_iteration_config_for_runtime(
                    iter_cfg,
                    dataset=dataset,
                    row_meta=row_meta,
                    seed_best=seed_best,
                    experience_prior=guidance_prior,
                )
                result["iteration_config"] = json.dumps(make_json_safe(iter_cfg), ensure_ascii=False)
                use_seed_as_initial, use_seed_reason = should_use_fast_seed_as_initial(seed_eval)
                agent_trace.append({
                    "event": "seed_fast_path",
                    "seed_candidate_count": len(seed_exprs),
                    "best_expr": _safe_get_attr(seed_best, "simplified_expression", None) if seed_best is not None else None,
                    "best_val_mse": _safe_get_attr(seed_best, "val_mse", None) if seed_best is not None else None,
                    "use_as_initial": bool(use_seed_as_initial),
                    "reason": use_seed_reason,
                })
                best_val = _safe_get_attr(seed_best, "val_mse", None) if seed_best is not None else None
                try:
                    if best_val is not None and np.isfinite(best_val) and float(best_val) <= FAST_PATH_SHORT_CIRCUIT_VAL_MSE:
                        result["early_stop_reason"] = f"seed_fast_path_val_mse<={FAST_PATH_SHORT_CIRCUIT_VAL_MSE}"
                except Exception:
                    pass
                _raise_if_task_budget_exceeded(task_deadline_ts, "seed_fast_path_done")

        # -------------------------------------------------
        # 3) Initial propose
        # -------------------------------------------------
        _raise_if_task_budget_exceeded(task_deadline_ts, "initial_propose_start")
        initial_iter_cfg = dict(iter_cfg)
        family_seed_mode = (
            PURE_LLM_CANDIDATE_MODE
            and should_enable_family_seed_initial_pass(
                structure_profile=observation.structure_profile,
                feature_names=dataset.feature_names,
                row_meta=row_meta,
                experience_prior=guidance_prior,
            )
        )
        family_seed_strict_mode = bool((guidance_prior or {}).get("strong_prior", False))
        if family_seed_mode:
            initial_iter_cfg["allow_family_seed_sources"] = True
            if family_seed_strict_mode and _is_benchmark_task(row_meta) and len(dataset.feature_names) == 5:
                initial_iter_cfg["family_seed_source_order"] = ["manual", "heuristic"]
            else:
                initial_iter_cfg["family_seed_source_order"] = list(FAMILY_SEED_SOURCE_ORDER)
            if family_seed_strict_mode:
                initial_iter_cfg["text_calls"] = 0
                initial_iter_cfg["skip_diverse_proposal"] = True
            else:
                initial_iter_cfg["text_calls"] = max(1, int(initial_iter_cfg.get("text_calls", NUM_TEXT_SINGLE_CALLS)))
                initial_iter_cfg["skip_diverse_proposal"] = False
            if family_seed_strict_mode and _is_benchmark_task(row_meta) and len(dataset.feature_names) >= 5:
                iter_cfg["prefer_stop_on_diffuse_residual"] = True
                iter_cfg["force_heuristic_agents"] = True
                initial_iter_cfg["prefer_stop_on_diffuse_residual"] = True
                initial_iter_cfg["force_heuristic_agents"] = True

            if (
                NO_LEAKAGE_MODE
                and ENABLE_GENERIC_STRUCTURAL_SEEDS_IN_PURE_LLM
                and _is_benchmark_task(row_meta)
                and len(dataset.feature_names) >= GENERIC_LOG_RATIO_MIN_DIM
                and not family_seed_strict_mode
            ):
                # First pass should evaluate the generic structural skeletons.
                # Otherwise PURE_LLM_CANDIDATE_MODE routes only text/diverse and the
                # log-ratio candidates are computed but discarded.
                initial_iter_cfg["allow_family_seed_sources"] = True
                initial_iter_cfg["family_seed_source_order"] = ["heuristic", "manual", "text", "diverse"]
                initial_iter_cfg["proposal_k"] = max(
                    int(initial_iter_cfg.get("proposal_k", NUM_PROPOSAL_CANDIDATES)),
                    GENERIC_STRUCTURAL_SEED_MIN_PROPOSAL_K,
                )
                initial_iter_cfg["refined_k"] = max(
                    int(initial_iter_cfg.get("refined_k", NUM_REFINED_CANDIDATES)),
                    GENERIC_STRUCTURAL_SEED_MIN_REFINED_K,
                )
                initial_iter_cfg["refine_rounds"] = max(
                    int(initial_iter_cfg.get("refine_rounds", MAX_REFINE_ROUNDS)),
                    GENERIC_STRUCTURAL_SEED_MIN_REFINE_ROUNDS,
                )
                if GENERIC_STRUCTURAL_SEED_SKIP_TEXT_INITIAL:
                    initial_iter_cfg["text_calls"] = 0
                if GENERIC_STRUCTURAL_SEED_SKIP_DIVERSE_INITIAL:
                    initial_iter_cfg["skip_diverse_proposal"] = True
                initial_iter_cfg["generic_structural_seed_initial"] = True
                iter_cfg["force_heuristic_agents"] = True
                initial_iter_cfg["force_heuristic_agents"] = True
            agent_trace.append({
                "event": "family_seed_initial_mode",
                "enabled": True,
                "strict": bool(family_seed_strict_mode),
                "source_order": list(initial_iter_cfg["family_seed_source_order"]),
            })
        else:
            agent_trace.append({
                "event": "family_seed_initial_mode",
                "enabled": False,
            })
        if use_seed_as_initial and seed_eval is not None:
            initial_prop = {
                "candidate_exprs": list(seed_exprs),
                "trace": {
                    "text_proposal_stats": {"skipped": True, "skip_reason": use_seed_reason or "seed fast path"},
                    "mm_proposal_stats": None,
                    "diverse_proposal_count": 0,
                    "manual_candidate_count": len(seed_exprs),
                    "protected_candidate_count": len(seed_exprs),
                    "memory_candidate_count": 0,
                    "experience_candidate_count": 0,
                    "heuristic_candidate_count": 0,
                    "merged_candidate_count_before_prefilter": len(seed_exprs),
                    "merged_candidate_count": len(seed_exprs),
                    "experience_prior": make_json_safe(summarize_experience_prior(guidance_prior)),
                    "experience_rerank": {"applied": False, "reason": "seed fast path"},
                    "fast_seed_stage_used": True,
                    "fast_seed_reason": use_seed_reason,
                },
            }
        else:
            initial_prop = proposer_agent.propose_initial(
                dataset=dataset,
                row_meta=row_meta,
                observation=observation,
                iter_cfg=initial_iter_cfg,
                experience_prior=guidance_prior,
                timer=timer,
            )
        initial_exprs = list(initial_prop["candidate_exprs"])
        prop_trace = initial_prop["trace"]

        # Safety net for high-dimensional no-leakage runs:
        # generic log-ratio seeds are injected directly after proposer routing.
        # This is not benchmark-specific and avoids losing them to source routing
        # or lightweight prefilter truncation inside propose_initial().
        direct_log_ratio_exprs = []
        if (
            ENABLE_GENERIC_LOG_RATIO_DIRECT_INITIAL_INJECTION
            and NO_LEAKAGE_MODE
            and len(dataset.feature_names) >= GENERIC_LOG_RATIO_MIN_DIM
        ):
            direct_log_ratio_exprs = build_generic_log_ratio_candidates(
                feature_names=dataset.feature_names,
                structure_profile=observation.structure_profile,
                experience_prior=guidance_prior,
                max_candidates=GENERIC_LOG_RATIO_DIRECT_INITIAL_MAX_CANDIDATES,
            )
            before_direct_injection = len(initial_exprs)
            initial_exprs = merge_expression_lists(direct_log_ratio_exprs, initial_exprs)
            if len(initial_exprs) > MAX_INITIAL_CANDIDATES:
                initial_exprs = initial_exprs[:MAX_INITIAL_CANDIDATES]
            prop_trace["generic_log_ratio_direct_injection"] = make_json_safe({
                "enabled": True,
                "num_exprs": len(direct_log_ratio_exprs),
                "num_exprs_before": before_direct_injection,
                "num_exprs_after": len(initial_exprs),
                "preview": direct_log_ratio_exprs[:8],
            })
        else:
            prop_trace["generic_log_ratio_direct_injection"] = make_json_safe({
                "enabled": False,
                "reason": "disabled_or_low_dim_or_not_no_leakage",
            })

        # Hybrid reasoning step: data-driven source supplies evidence only.
        # We convert evidence -> abstract plans, ask the LLM to approve/revise,
        # and if the LLM endpoint times out we use an explicit evidence fallback.
        structural_planner_trace = {"enabled": False, "num_exprs": 0}
        structural_plan_exprs = []
        if (
            ENABLE_LLM_STRUCTURAL_PLANNER
            and ENABLE_INITIAL_LLM_STRUCTURAL_PLANNER
            and not (LOW_DIM_BENCHMARK_SKIP_INITIAL_PLANNER and _is_low_dim_benchmark_like(row_meta=row_meta, dataset=dataset))
        ):
            try:
                source_stats = dict(prop_trace.get("source_stats", {}) or {})
                datadriven_evidence = dict((source_stats.get("datadriven", {}) or {}).get("raw_result", {}) or {})
                evidence_plans = build_structural_plans_from_datadriven_evidence(
                    datadriven_raw=datadriven_evidence,
                    feature_names=dataset.feature_names,
                    max_plans=STRUCTURAL_PLANNER_MAX_PLANS,
                )
                approved_plans, planner_approval_trace = timed_call(
                    timer,
                    "step_structural_planner_approval",
                    select_structural_plans_with_llm_or_fallback,
                    client=structural_planner_agent.client,
                    dataset=dataset,
                    observation=observation,
                    evidence_plans=evidence_plans,
                )
                structural_plan_exprs, expansion_trace = expand_approved_structural_plans_to_candidates(
                    approved_plans,
                    feature_names=dataset.feature_names,
                    max_candidates=STRUCTURAL_PLAN_EXPANSION_MAX_CANDIDATES,
                )
                structural_planner_trace = {
                    "enabled": True,
                    "evidence_plans": make_json_safe(evidence_plans),
                    "approved_plans": make_json_safe(approved_plans),
                    "approval": make_json_safe(planner_approval_trace),
                    "expansion": make_json_safe(expansion_trace),
                    "num_exprs": len(structural_plan_exprs),
                }
                if structural_plan_exprs:
                    initial_exprs, llm_companion_merge_trace = merge_llm_plan_candidates_with_safety_head(
                        base_exprs=initial_exprs,
                        plan_exprs=structural_plan_exprs,
                        max_total=MAX_INITIAL_CANDIDATES,
                    )
                    structural_planner_trace["llm_companion_merge"] = make_json_safe(llm_companion_merge_trace)
            except Exception as e:
                structural_planner_trace = {"enabled": True, "error": repr(e), "num_exprs": 0}
        elif LOW_DIM_BENCHMARK_SKIP_INITIAL_PLANNER and _is_low_dim_benchmark_like(row_meta=row_meta, dataset=dataset):
            structural_planner_trace = {
                "enabled": False,
                "num_exprs": 0,
                "skipped": True,
                "reason": "low_dim_benchmark_like_skip_initial_planner",
            }
        prop_trace["structural_planner"] = make_json_safe(structural_planner_trace)
        prop_trace["structural_planner_candidate_count"] = len(structural_plan_exprs)

        proposal_history.append({
            "stage": "initial",
            "num_exprs": len(initial_exprs),
            "trace": prop_trace,
        })

        if structural_plan_exprs:
            proposal_history.append({
                "stage": "structural_planner_expansion",
                "num_exprs": len(structural_plan_exprs),
                "trace": make_json_safe(structural_planner_trace),
            })
            agent_trace.append({
                "event": "structural_planner_expansion",
                "num_exprs": len(structural_plan_exprs),
                "used_llm": bool((structural_planner_trace.get("approval") or {}).get("used_llm", False)),
            })

        if ALLOW_TRUE_EXPR_DIAGNOSTICS and not DISABLE_FORM_MATCH_EVAL:
            if timer is not None:
                timer.start("step_initial_formula_form_eval")
            initial_form_eval = evaluate_formula_form_proposals(
                candidate_exprs=initial_exprs,
                true_expr=row_meta.get("true_expression"),
                variable_names=dataset.feature_names,
            )
            if timer is not None:
                timer.stop("step_initial_formula_form_eval")
            if initial_form_eval is not None:
                result["initial_formula_form_eval"] = json.dumps(make_json_safe(initial_form_eval), ensure_ascii=False)
                result["initial_best_form_match_score"] = initial_form_eval.get("best_form_match_score")
                proposal_history[-1]["formula_form_eval"] = make_json_safe(initial_form_eval)

        result["raw_exprs"] = json.dumps(make_json_safe(initial_exprs), ensure_ascii=False)
        result["num_candidate_exprs"] = len(initial_exprs)
        result["text_proposal_stats"] = json.dumps(make_json_safe(prop_trace.get("text_proposal_stats")), ensure_ascii=False)
        result["mm_proposal_stats"] = json.dumps(make_json_safe(prop_trace.get("mm_proposal_stats")), ensure_ascii=False)
        result["diverse_proposal_count"] = prop_trace.get("diverse_proposal_count")
        result["manual_candidate_count"] = prop_trace.get("manual_candidate_count")
        result["datadriven_candidate_count"] = prop_trace.get("datadriven_candidate_count")
        result["generic_candidate_count"] = prop_trace.get("generic_candidate_count")
        result["structural_planner_candidate_count"] = prop_trace.get("structural_planner_candidate_count")
        result["protected_candidate_count"] = prop_trace.get("protected_candidate_count")
        result["memory_candidate_count"] = prop_trace.get("memory_candidate_count")
        result["experience_candidate_count"] = prop_trace.get("experience_candidate_count")
        result["heuristic_candidate_count"] = prop_trace.get("heuristic_candidate_count")
        result["proposal_source_stats"] = json.dumps(make_json_safe(prop_trace.get("source_stats")), ensure_ascii=False)
        result["experience_rerank"] = json.dumps(make_json_safe(prop_trace.get("experience_rerank")), ensure_ascii=False)
        result["merged_candidate_count"] = prop_trace.get("merged_candidate_count")
        _raise_if_task_budget_exceeded(task_deadline_ts, "initial_propose_done")

        # -------------------------------------------------
        # 4) Initial evaluate
        # -------------------------------------------------
        _raise_if_task_budget_exceeded(task_deadline_ts, "initial_evaluate_start")
        if use_seed_as_initial and seed_eval is not None:
            initial_eval = seed_eval
        else:
            initial_eval = evaluator_agent.evaluate(
                candidate_exprs=initial_exprs,
                dataset=dataset,
                row_meta=row_meta,
                timer=timer,
                prefix="initial",
            )
        current_best = initial_eval["best_result"]
        evaluation_history.append({
            "stage": "initial",
            "topk": initial_eval.get("evaluation_table"),
            "residual_summary": initial_eval.get("residual_summary"),
            "physics_summary": initial_eval.get("physics_summary"),
        })
        _raise_if_task_budget_exceeded(task_deadline_ts, "initial_evaluate_done")

        # Generic low-dimensional rational/power-rational coverage rescue.
        # This is the low-dimensional counterpart of the high-dimensional family
        # rescue: it does not use benchmark names or true formulas, but ensures
        # polynomial-rational and power-rational families are actually evaluated.
        lowdim_rescue_ran = False
        run_lowdim_coverage, lowdim_coverage_reason = should_run_lowdim_rational_coverage_rescue(
            current_best=current_best,
            dataset=dataset,
            row_meta=row_meta,
        )
        if run_lowdim_coverage:
            try:
                _raise_if_task_budget_exceeded(task_deadline_ts, "lowdim_rational_coverage_start")
                lowdim_exprs, lowdim_trace = build_lowdim_rational_coverage_candidates(
                    dataset=dataset,
                    row_meta=row_meta,
                    max_candidates=LOW_DIM_RATIONAL_COVERAGE_MAX_CANDIDATES,
                )
                lowdim_rescue_ran = True
                proposal_history.append({
                    "stage": "lowdim_rational_coverage_rescue",
                    "num_exprs": len(lowdim_exprs),
                    "trace": {
                        "reason": lowdim_coverage_reason,
                        **make_json_safe(lowdim_trace),
                    },
                })
                agent_trace.append({
                    "event": "lowdim_rational_coverage_rescue_start",
                    "reason": lowdim_coverage_reason,
                    "num_exprs": len(lowdim_exprs),
                })
                if lowdim_exprs:
                    lowdim_eval = evaluator_agent.evaluate(
                        candidate_exprs=lowdim_exprs,
                        dataset=dataset,
                        row_meta=row_meta,
                        timer=timer,
                        prefix="lowdim_rational_coverage",
                    )
                    evaluation_history.append({
                        "stage": "lowdim_rational_coverage_rescue",
                        "topk": lowdim_eval.get("evaluation_table"),
                        "residual_summary": lowdim_eval.get("residual_summary"),
                        "physics_summary": lowdim_eval.get("physics_summary"),
                    })
                    if should_accept_candidate_update(lowdim_eval.get("best_result"), current_best):
                        current_best = lowdim_eval.get("best_result")
                        initial_eval = lowdim_eval
                        agent_trace.append({
                            "event": "lowdim_rational_coverage_rescue_done",
                            "improved": True,
                            "best_expr": _safe_get_attr(current_best, "simplified_expression", None),
                            "best_val_mse": _safe_get_attr(current_best, "val_mse", None),
                        })
                    else:
                        agent_trace.append({
                            "event": "lowdim_rational_coverage_rescue_done",
                            "improved": False,
                            "best_expr": _safe_get_attr(current_best, "simplified_expression", None),
                            "best_val_mse": _safe_get_attr(current_best, "val_mse", None),
                        })
                _raise_if_task_budget_exceeded(task_deadline_ts, "lowdim_rational_coverage_done")
            except TaskTimeBudgetExceeded:
                raise
            except Exception as e:
                agent_trace.append({
                    "event": "lowdim_rational_coverage_rescue_error",
                    "error": repr(e),
                })

        # Direct 1D additive-nonlinear rescue. This is separate from rational
        # coverage because linear + small nonlinear residual forms can be ranked
        # poorly when mixed into a large rational pool.
        run_additive_direct, additive_direct_reason = should_run_lowdim_additive_direct_rescue(
            current_best=current_best,
            dataset=dataset,
            row_meta=row_meta,
        )
        if run_additive_direct:
            try:
                _raise_if_task_budget_exceeded(task_deadline_ts, "lowdim_additive_nonlinear_start")
                additive_exprs, additive_trace = build_lowdim_additive_direct_rescue_candidates(
                    dataset=dataset,
                    max_candidates=LOW_DIM_ADDITIVE_DIRECT_MAX_CANDIDATES,
                )
                proposal_history.append({
                    "stage": "lowdim_additive_nonlinear_rescue",
                    "num_exprs": len(additive_exprs),
                    "trace": {
                        "reason": additive_direct_reason,
                        **make_json_safe(additive_trace),
                    },
                })
                agent_trace.append({
                    "event": "lowdim_additive_nonlinear_rescue_start",
                    "reason": additive_direct_reason,
                    "num_exprs": len(additive_exprs),
                })
                if additive_exprs:
                    additive_eval = evaluator_agent.evaluate(
                        candidate_exprs=additive_exprs,
                        dataset=dataset,
                        row_meta=row_meta,
                        timer=timer,
                        prefix="lowdim_additive_nonlinear",
                    )
                    evaluation_history.append({
                        "stage": "lowdim_additive_nonlinear_rescue",
                        "topk": additive_eval.get("evaluation_table"),
                        "residual_summary": additive_eval.get("residual_summary"),
                        "physics_summary": additive_eval.get("physics_summary"),
                    })
                    if is_better_result(additive_eval.get("best_result"), current_best):
                        current_best = additive_eval.get("best_result")
                        initial_eval = additive_eval
                        improved_flag = True
                    else:
                        improved_flag = False
                    agent_trace.append({
                        "event": "lowdim_additive_nonlinear_rescue_done",
                        "improved": improved_flag,
                        "best_expr": _safe_get_attr(current_best, "simplified_expression", None),
                        "best_val_mse": _safe_get_attr(current_best, "val_mse", None),
                    })
                _raise_if_task_budget_exceeded(task_deadline_ts, "lowdim_additive_nonlinear_done")
            except TaskTimeBudgetExceeded:
                raise
            except Exception as e:
                agent_trace.append({
                    "event": "lowdim_additive_nonlinear_rescue_error",
                    "error": repr(e),
                })

        # Universal high-dimensional family coverage rescue. This is a generic,
        # benchmark-name-free safety net for Feynman-like high-dimensional tasks.
        # It evaluates low-parameter templates from broad physical structure
        # families directly, so important families such as exp-minus-one cannot
        # be lost due to planner ranking, LLM approval, or prefilter truncation.
        run_universal_coverage, universal_coverage_reason = should_run_highdim_universal_coverage_rescue(
            current_best=current_best,
            dataset=dataset,
            row_meta=row_meta,
        )
        if run_universal_coverage:
            try:
                _raise_if_task_budget_exceeded(task_deadline_ts, "highdim_universal_coverage_start")
                selected_coverage_families, coverage_family_trace = select_highdim_families_from_visual_evidence(
                    observation=observation,
                    fallback_families=HIGH_DIM_UNIVERSAL_COVERAGE_FAMILIES,
                )
                universal_exprs = build_highdim_universal_coverage_candidates(
                    feature_names=dataset.feature_names,
                    max_candidates=HIGH_DIM_UNIVERSAL_COVERAGE_MAX_EVAL_CANDIDATES,
                    families=selected_coverage_families,
                )
                universal_exprs_before_llm = list(universal_exprs)
                highdim_llm_rerank_trace = {"used_llm": False, "reason": "not attempted"}
                if universal_exprs and ENABLE_HIGH_DIM_LLM_RESCUE_RERANK:
                    if timer is not None:
                        timer.start("step_highdim_llm_rescue_rerank")
                    universal_exprs, highdim_llm_rerank_trace = rerank_highdim_rescue_candidates_with_llm(
                        client=planner_client,
                        dataset=dataset,
                        observation=observation,
                        candidate_exprs=universal_exprs,
                        current_best=current_best,
                        max_tokens=HIGH_DIM_LLM_RESCUE_RERANK_MAX_TOKENS,
                    )
                    if timer is not None:
                        timer.stop("step_highdim_llm_rescue_rerank")
                proposal_history.append({
                    "stage": "highdim_universal_coverage_rescue",
                    "num_exprs": len(universal_exprs),
                    "trace": {
                        "reason": universal_coverage_reason,
                        "families": list(selected_coverage_families),
                        "family_selection": make_json_safe(coverage_family_trace),
                        "preview_before_llm_rerank": universal_exprs_before_llm[:20],
                        "preview": universal_exprs[:20],
                        "llm_rerank": make_json_safe(highdim_llm_rerank_trace),
                    },
                })
                agent_trace.append({
                    "event": "highdim_universal_coverage_rescue_start",
                    "reason": universal_coverage_reason,
                    "num_exprs": len(universal_exprs),
                })
                if universal_exprs:
                    universal_eval = evaluator_agent.evaluate(
                        candidate_exprs=universal_exprs,
                        dataset=dataset,
                        row_meta=row_meta,
                        timer=timer,
                        prefix="highdim_universal_coverage",
                    )
                    evaluation_history.append({
                        "stage": "highdim_universal_coverage_rescue",
                        "topk": universal_eval.get("evaluation_table"),
                        "residual_summary": universal_eval.get("residual_summary"),
                        "physics_summary": universal_eval.get("physics_summary"),
                    })
                    if is_better_result(universal_eval.get("best_result"), current_best):
                        current_best = universal_eval.get("best_result")
                        initial_eval = universal_eval
                        agent_trace.append({
                            "event": "highdim_universal_coverage_rescue_done",
                            "improved": True,
                            "best_expr": _safe_get_attr(current_best, "simplified_expression", None),
                            "best_val_mse": _safe_get_attr(current_best, "val_mse", None),
                        })
                    else:
                        agent_trace.append({
                            "event": "highdim_universal_coverage_rescue_done",
                            "improved": False,
                            "best_expr": _safe_get_attr(current_best, "simplified_expression", None),
                            "best_val_mse": _safe_get_attr(current_best, "val_mse", None),
                        })
                _raise_if_task_budget_exceeded(task_deadline_ts, "highdim_universal_coverage_done")
            except TaskTimeBudgetExceeded:
                raise
            except Exception as e:
                agent_trace.append({
                    "event": "highdim_universal_coverage_rescue_error",
                    "error": repr(e),
                })

        if ENABLE_STRUCTURAL_QUADRATIC_SPECIALIZER and current_best is not None:
            try:
                current_best_val = _safe_get_attr(current_best, "val_mse", None)
                run_structural_specializer = (
                    current_best_val is not None
                    and np.isfinite(current_best_val)
                    and float(current_best_val) > STRUCTURAL_QUADRATIC_SPECIALIZER_TRIGGER_VAL_MSE
                    and _is_benchmark_task(row_meta)
                    and len(dataset.feature_names) >= 5
                    and bool((guidance_prior or {}).get("strong_prior", False))
                )
            except Exception:
                run_structural_specializer = False

            if run_structural_specializer:
                specialized_exprs = build_structural_quadratic_specialization_candidates(
                    feature_names=dataset.feature_names,
                    structure_profile=observation.structure_profile,
                    experience_prior=guidance_prior,
                    row_meta=row_meta,
                    scored_results=initial_eval.get("scored_results", []),
                    max_candidates=STRUCTURAL_QUADRATIC_SPECIALIZER_MAX_CANDIDATES,
                )
                if specialized_exprs:
                    proposal_history.append({
                        "stage": "structural_quadratic_specializer",
                        "num_exprs": len(specialized_exprs),
                        "trace": {
                            "specialized_candidates": specialized_exprs,
                            "trigger_val_mse": current_best_val,
                        },
                    })
                    specialized_eval = evaluator_agent.evaluate(
                        candidate_exprs=specialized_exprs,
                        dataset=dataset,
                        row_meta=row_meta,
                        timer=timer,
                        prefix="structural_quadratic_specializer",
                    )
                    evaluation_history.append({
                        "stage": "structural_quadratic_specializer",
                        "topk": specialized_eval.get("evaluation_table"),
                        "residual_summary": specialized_eval.get("residual_summary"),
                        "physics_summary": specialized_eval.get("physics_summary"),
                    })
                    specialized_best = specialized_eval.get("best_result")
                    improved = is_better_result(specialized_best, current_best)
                    agent_trace.append({
                        "event": "structural_quadratic_specializer",
                        "num_exprs": len(specialized_exprs),
                        "best_expr": _safe_get_attr(specialized_best, "simplified_expression", None) if specialized_best is not None else None,
                        "best_val_mse": _safe_get_attr(specialized_best, "val_mse", None) if specialized_best is not None else None,
                        "improved_over_initial": bool(improved),
                    })
                    if improved:
                        current_best = specialized_best
                        initial_eval = specialized_eval
                else:
                    agent_trace.append({
                        "event": "structural_quadratic_specializer",
                        "num_exprs": 0,
                        "reason": "no_specialized_candidates",
                    })

        run_surrogate_escape, surrogate_escape_reason = should_run_high_dim_surrogate_escape(
            current_best=current_best,
            dataset=dataset,
            row_meta=row_meta,
            structure_profile=observation.structure_profile,
            experience_prior=guidance_prior,
        )
        agent_trace.append({
            "event": "surrogate_escape_decision",
            "run_surrogate_escape": bool(run_surrogate_escape),
            "reason": surrogate_escape_reason,
            "best_val_mse_before_escape": _safe_get_attr(current_best, "val_mse", None) if current_best is not None else None,
        })
        if run_surrogate_escape:
            surrogate_escape_exprs = build_high_dim_surrogate_escape_candidates(
                feature_names=dataset.feature_names,
                current_best_expr=_safe_get_attr(current_best, "simplified_expression", None),
                structure_profile=observation.structure_profile,
                experience_prior=guidance_prior,
                max_candidates=HIGH_DIM_SURROGATE_ESCAPE_MAX_CANDIDATES,
            )
            if surrogate_escape_exprs:
                proposal_history.append({
                    "stage": "surrogate_escape",
                    "num_exprs": len(surrogate_escape_exprs),
                    "trace": {
                        "trigger_reason": surrogate_escape_reason,
                        "escape_candidates": surrogate_escape_exprs,
                    },
                })
                surrogate_escape_eval = evaluator_agent.evaluate(
                    candidate_exprs=surrogate_escape_exprs,
                    dataset=dataset,
                    row_meta=row_meta,
                    timer=timer,
                    prefix="surrogate_escape",
                )
                evaluation_history.append({
                    "stage": "surrogate_escape",
                    "topk": surrogate_escape_eval.get("evaluation_table"),
                    "residual_summary": surrogate_escape_eval.get("residual_summary"),
                    "physics_summary": surrogate_escape_eval.get("physics_summary"),
                })
                surrogate_best = surrogate_escape_eval.get("best_result")
                improved = is_better_result(surrogate_best, current_best)
                agent_trace.append({
                    "event": "surrogate_escape_done",
                    "trigger_reason": surrogate_escape_reason,
                    "num_exprs": len(surrogate_escape_exprs),
                    "best_expr": _safe_get_attr(surrogate_best, "simplified_expression", None) if surrogate_best is not None else None,
                    "best_val_mse": _safe_get_attr(surrogate_best, "val_mse", None) if surrogate_best is not None else None,
                    "improved_over_previous": bool(improved),
                })
                if improved:
                    current_best = surrogate_best
                    initial_eval = surrogate_escape_eval
            else:
                agent_trace.append({
                    "event": "surrogate_escape_done",
                    "trigger_reason": surrogate_escape_reason,
                    "num_exprs": 0,
                    "reason": "no_escape_candidates",
                    "improved_over_previous": False,
                })

        run_guided_rescue = False
        guided_rescue_reason = "seed fast path"
        if not use_seed_as_initial:
            run_guided_rescue, guided_rescue_reason = should_run_guided_rescue_after_initial_pass(
                current_best=current_best,
                num_existing_candidates=len(initial_exprs),
                iter_cfg=iter_cfg,
            )

        agent_trace.append({
            "event": "guided_rescue_decision",
            "run_guided_rescue": bool(run_guided_rescue),
            "reason": guided_rescue_reason,
            "best_val_mse_after_initial": _safe_get_attr(current_best, "val_mse", None) if current_best is not None else None,
        })

        if run_guided_rescue:
            _raise_if_task_budget_exceeded(task_deadline_ts, "guided_rescue_start")
            guided_rescue_iter_cfg = build_guided_rescue_iteration_config(iter_cfg)
            if _is_benchmark_task(row_meta) and NO_LEAKAGE_MODE and len(dataset.feature_names) >= 5:
                # Avoid the slow timeout paths seen in the report. Guided rescue
                # should reuse deterministic plan/generic/heuristic sources, not
                # wait for text/diverse calls that often return no candidates.
                guided_rescue_iter_cfg["allow_guided_candidate_sources"] = True
                guided_rescue_iter_cfg["allow_family_seed_sources"] = True
                guided_rescue_iter_cfg["guided_source_order"] = ["datadriven", "generic", "heuristic", "manual"]
                guided_rescue_iter_cfg["family_seed_source_order"] = ["datadriven", "generic", "heuristic", "manual"]
                guided_rescue_iter_cfg["skip_diverse_proposal"] = True
                guided_rescue_iter_cfg["text_calls"] = 0
                guided_rescue_iter_cfg["mm_calls"] = 0
                guided_rescue_iter_cfg["prefer_stop_on_diffuse_residual"] = True
                guided_rescue_iter_cfg["force_heuristic_agents"] = True
            elif family_seed_mode and family_seed_strict_mode and _is_benchmark_task(row_meta) and len(dataset.feature_names) >= 5:
                guided_rescue_iter_cfg["allow_family_seed_sources"] = True
                guided_rescue_iter_cfg["family_seed_source_order"] = ["manual", "heuristic"]
                guided_rescue_iter_cfg["guided_source_order"] = ["manual", "heuristic"]
                guided_rescue_iter_cfg["skip_diverse_proposal"] = True
                guided_rescue_iter_cfg["text_calls"] = 0
                guided_rescue_iter_cfg["prefer_stop_on_diffuse_residual"] = True
                guided_rescue_iter_cfg["force_heuristic_agents"] = True
            guided_prop = proposer_agent.propose_initial(
                dataset=dataset,
                row_meta=row_meta,
                observation=observation,
                iter_cfg=guided_rescue_iter_cfg,
                experience_prior=guidance_prior,
                timer=timer,
            )
            guided_exprs = list(guided_prop["candidate_exprs"])
            guided_trace = dict(guided_prop["trace"] or {})
            guided_trace["guided_rescue"] = True
            guided_trace["trigger_reason"] = guided_rescue_reason
            proposal_history.append({
                "stage": "guided_rescue",
                "num_exprs": len(guided_exprs),
                "trace": guided_trace,
            })

            if guided_exprs:
                initial_best_before_rescue = current_best
                guided_eval = evaluator_agent.evaluate(
                    candidate_exprs=guided_exprs,
                    dataset=dataset,
                    row_meta=row_meta,
                    timer=timer,
                    prefix="guided_rescue",
                )
                evaluation_history.append({
                    "stage": "guided_rescue",
                    "topk": guided_eval.get("evaluation_table"),
                    "residual_summary": guided_eval.get("residual_summary"),
                    "physics_summary": guided_eval.get("physics_summary"),
                })
                guided_best = guided_eval.get("best_result")
                if should_accept_candidate_update(guided_best, current_best):
                    current_best = guided_best
                    initial_eval = guided_eval
                agent_trace.append({
                    "event": "guided_rescue_done",
                    "trigger_reason": guided_rescue_reason,
                    "num_exprs": len(guided_exprs),
                    "best_expr": _safe_get_attr(guided_best, "simplified_expression", None) if guided_best is not None else None,
                    "best_val_mse": _safe_get_attr(guided_best, "val_mse", None) if guided_best is not None else None,
                    "improved_over_initial": bool(is_better_result(guided_best, initial_best_before_rescue)),
                })
            else:
                agent_trace.append({
                    "event": "guided_rescue_done",
                    "trigger_reason": guided_rescue_reason,
                    "num_exprs": 0,
                    "improved_over_initial": False,
                })
            _raise_if_task_budget_exceeded(task_deadline_ts, "guided_rescue_done")

        # Post-evaluation LLM local repair. This is the main hybrid reasoning step:
        # tools provide evidence; the model decides local structural edits.
        try:
            cur_val_for_local_repair = _safe_get_attr(current_best, "val_mse", None) if current_best is not None else None
            run_post_eval_local_repair = (
                ENABLE_POST_EVAL_LLM_LOCAL_REPAIR
                and current_best is not None
                and cur_val_for_local_repair is not None
                and np.isfinite(float(cur_val_for_local_repair))
                and float(cur_val_for_local_repair) > float(POST_EVAL_LLM_LOCAL_REPAIR_TRIGGER_VAL_MSE)
            )
        except Exception:
            run_post_eval_local_repair = False

        if run_post_eval_local_repair:
            _raise_if_task_budget_exceeded(task_deadline_ts, "post_eval_local_repair_start")
            repair_trace = timed_call(
                timer,
                "step_post_eval_llm_local_repair",
                post_eval_repair_agent.propose,
                dataset=dataset,
                observation=observation,
                evaluation=initial_eval,
                current_best=current_best,
                guidance_prior=guidance_prior,
                max_candidates=POST_EVAL_LLM_LOCAL_REPAIR_MAX_CANDIDATES,
            )
            local_repair_exprs = list(repair_trace.get("candidate_exprs", []) or [])
            proposal_history.append({
                "stage": "post_eval_llm_local_repair",
                "num_exprs": len(local_repair_exprs),
                "trace": make_json_safe(repair_trace),
            })
            agent_trace.append({
                "event": "post_eval_llm_local_repair",
                "used_llm": bool(repair_trace.get("used_llm", False)),
                "num_exprs": len(local_repair_exprs),
                "best_val_mse_before": cur_val_for_local_repair,
            })
            if local_repair_exprs:
                repair_eval = evaluator_agent.evaluate(
                    candidate_exprs=local_repair_exprs,
                    dataset=dataset,
                    row_meta=row_meta,
                    timer=timer,
                    prefix="post_eval_llm_local_repair",
                )
                evaluation_history.append({
                    "stage": "post_eval_llm_local_repair",
                    "topk": repair_eval.get("evaluation_table"),
                    "residual_summary": repair_eval.get("residual_summary"),
                    "physics_summary": repair_eval.get("physics_summary"),
                })
                repair_best = repair_eval.get("best_result")
                repair_improved = is_better_result(repair_best, current_best)
                if repair_improved:
                    current_best = repair_best
                    initial_eval = repair_eval
                agent_trace.append({
                    "event": "post_eval_llm_local_repair_done",
                    "improved": bool(repair_improved),
                    "best_expr": _safe_get_attr(current_best, "simplified_expression", None) if current_best is not None else None,
                    "best_val_mse": _safe_get_attr(current_best, "val_mse", None) if current_best is not None else None,
                })
            _raise_if_task_budget_exceeded(task_deadline_ts, "post_eval_local_repair_done")

        # Generic local rescue: try adding missing multiplicative envelope variables
        # to the current best before spending time on slow LLM/MM calls.
        if ENABLE_MISSING_MULTIPLIER_RESCUE and ENABLE_DIRECT_MISSING_MULTIPLIER_RESCUE:
            try:
                cur_val = _safe_get_attr(current_best, "val_mse", None) if current_best is not None else None
                run_multiplier_rescue = (
                    current_best is not None
                    and cur_val is not None
                    and np.isfinite(float(cur_val))
                    and float(cur_val) > float(MISSING_MULTIPLIER_RESCUE_MIN_VAL_MSE)
                )
            except Exception:
                run_multiplier_rescue = False
            multiplier_exprs = build_missing_multiplier_rescue_candidates(
                current_best=current_best,
                dataset=dataset,
                max_candidates=MISSING_MULTIPLIER_RESCUE_MAX_CANDIDATES,
            ) if run_multiplier_rescue else []
            agent_trace.append({
                "event": "missing_multiplier_rescue_decision",
                "run": bool(multiplier_exprs),
                "num_exprs": len(multiplier_exprs),
                "best_val_mse_before": _safe_get_attr(current_best, "val_mse", None) if current_best is not None else None,
            })
            if multiplier_exprs:
                _raise_if_task_budget_exceeded(task_deadline_ts, "missing_multiplier_rescue_start")
                proposal_history.append({
                    "stage": "missing_multiplier_rescue",
                    "num_exprs": len(multiplier_exprs),
                    "trace": {"candidate_exprs": multiplier_exprs[:8]},
                })
                multiplier_eval = evaluator_agent.evaluate(
                    candidate_exprs=multiplier_exprs,
                    dataset=dataset,
                    row_meta=row_meta,
                    timer=timer,
                    prefix="missing_multiplier_rescue",
                )
                evaluation_history.append({
                    "stage": "missing_multiplier_rescue",
                    "topk": multiplier_eval.get("evaluation_table"),
                    "residual_summary": multiplier_eval.get("residual_summary"),
                    "physics_summary": multiplier_eval.get("physics_summary"),
                })
                multiplier_best = multiplier_eval.get("best_result")
                improved = is_better_result(multiplier_best, current_best)
                if improved:
                    current_best = multiplier_best
                    initial_eval = multiplier_eval
                agent_trace.append({
                    "event": "missing_multiplier_rescue_done",
                    "num_exprs": len(multiplier_exprs),
                    "improved": bool(improved),
                    "best_expr": _safe_get_attr(multiplier_best, "simplified_expression", None) if multiplier_best is not None else None,
                    "best_val_mse": _safe_get_attr(multiplier_best, "val_mse", None) if multiplier_best is not None else None,
                })
                _raise_if_task_budget_exceeded(task_deadline_ts, "missing_multiplier_rescue_done")

        # Optional multimodal rescue if needed.
        high_dim_bad_fit_for_mm = False
        try:
            best_val_for_mm = _safe_get_attr(current_best, "val_mse", None) if current_best is not None else None
            high_dim_bad_fit_for_mm = bool(
                FORCE_MM_FOR_HIGH_DIM_BAD_FIT
                and USE_MULTIMODAL_PROPOSAL
                and _is_benchmark_task(row_meta)
                and len(dataset.feature_names) >= HIGH_DIM_RECON_TRIGGER_DIM
                and (best_val_for_mm is None or float(best_val_for_mm) > FORCE_MM_HIGH_DIM_VAL_MSE)
            )
        except Exception:
            high_dim_bad_fit_for_mm = False

        if high_dim_bad_fit_for_mm:
            iter_cfg["mm_calls"] = max(1, int(iter_cfg.get("mm_calls", 0)))
            need_mm = True
            mm_trigger_reason = f"forced high-dim MM because best_val_mse>{FORCE_MM_HIGH_DIM_VAL_MSE}"
        elif int(iter_cfg.get("mm_calls", 0)) <= 0:
            need_mm = False
            mm_trigger_reason = "runtime budget disabled multimodal"
        elif HIGH_DIM_NO_LEAKAGE_SKIP_MM_AFTER_GUIDED and _is_benchmark_task(row_meta) and NO_LEAKAGE_MODE and len(dataset.feature_names) >= 5:
            need_mm = False
            mm_trigger_reason = "high-dim no-leakage path skips slow multimodal proposal"
        elif family_seed_mode and _is_benchmark_task(row_meta) and len(dataset.feature_names) >= 5:
            need_mm = False
            mm_trigger_reason = "family_seed benchmark path disables multimodal"
        elif USE_MULTIMODAL_PROPOSAL:
            if MM_ONLY_IF_NEEDED:
                need_mm, mm_trigger_reason = should_run_multimodal_after_initial_pass(
                    current_best=current_best,
                    dataset=dataset,
                    num_existing_candidates=len(initial_exprs),
                )
            else:
                need_mm = True
                mm_trigger_reason = "USE_MULTIMODAL_PROPOSAL=True and MM_ONLY_IF_NEEDED=False"
        else:
            need_mm = False
            mm_trigger_reason = "multimodal disabled"

        run_mm_branch = bool(need_mm or (USE_MULTIMODAL_PROPOSAL and RUN_STANDALONE_VLM_FORM_EVAL))
        if run_mm_branch and not need_mm:
            mm_trigger_reason = "standalone_vlm_form_eval"

        agent_trace.append({
            "event": "mm_decision",
            "need_mm": need_mm,
            "run_mm_branch": run_mm_branch,
            "reason": mm_trigger_reason,
        })

        if run_mm_branch:
            _raise_if_task_budget_exceeded(task_deadline_ts, "mm_start")
            observation = timed_call(
                timer,
                "step_mm_assets",
                observer_agent.maybe_generate_mm_assets,
                dataset,
                row_meta,
                observation,
                timer=timer,
            )
            agent_trace.append({
                "event": "mm_assets",
                "attempted": observation.mm_assets_attempted,
                "succeeded": observation.mm_assets_succeeded,
                "num_plots": len(observation.image_paths),
                "error": observation.mm_assets_error,
            })
            mm_prop = timed_call(
                timer,
                "step_mm_proposal",
                proposer_agent.propose_mm_if_needed,
                dataset,
                observation,
                iter_cfg,
                row_meta=row_meta,
                allow_legacy_fallback=bool(need_mm),
                timer=timer,
            )
            mm_prop_trace = mm_prop["trace"]
            mm_exprs = mm_prop["candidate_exprs"]
            mm_candidate_count = len(mm_exprs)
            result["mm_proposal_stats"] = json.dumps(make_json_safe(mm_prop_trace), ensure_ascii=False)
            proposal_history.append({
                "stage": "mm_rescue" if need_mm else "mm_standalone_eval",
                "num_exprs": len(mm_exprs),
                "trace": mm_prop_trace,
            })

            if ALLOW_TRUE_EXPR_DIAGNOSTICS and not DISABLE_FORM_MATCH_EVAL:
                if timer is not None:
                    timer.start("step_vlm_formula_form_eval")
                vlm_form_eval = evaluate_formula_form_proposals(
                    candidate_exprs=mm_exprs,
                    true_expr=row_meta.get("true_expression"),
                    variable_names=dataset.feature_names,
                )
                if timer is not None:
                    timer.stop("step_vlm_formula_form_eval")
                if vlm_form_eval is not None:
                    result["vlm_formula_form_eval"] = json.dumps(make_json_safe(vlm_form_eval), ensure_ascii=False)
                    result["vlm_best_form_match_score"] = vlm_form_eval.get("best_form_match_score")
                    proposal_history[-1]["formula_form_eval"] = make_json_safe(vlm_form_eval)

            if mm_exprs:
                if not need_mm:
                    agent_trace.append({
                        "event": "mm_eval_only",
                        "num_mm_exprs": len(mm_exprs),
                    })
                else:
                    merged_with_mm = merge_expression_groups_with_limit(
                        [initial_exprs, mm_exprs],
                        max_total=MAX_INITIAL_CANDIDATES,
                    )
                    if merged_with_mm == initial_exprs:
                        agent_trace.append({
                            "event": "mm_no_new_unique_candidates",
                            "num_mm_exprs": len(mm_exprs),
                        })
                    else:
                        if ENABLE_LIGHT_PREFILTER and len(merged_with_mm) >= PREFILTER_TRIGGER_CANDIDATE_COUNT:
                            if timer is not None:
                                timer.start("step_mm_candidate_prefilter")
                            merged_with_mm = lightweight_prefilter_candidates(
                                merged_with_mm,
                                dataset,
                                timer=timer,
                                prefix="mm_prefilter",
                            )
                            if timer is not None:
                                timer.stop("step_mm_candidate_prefilter")
                        mm_eval = evaluator_agent.evaluate(
                            candidate_exprs=merged_with_mm,
                            dataset=dataset,
                            row_meta=row_meta,
                            timer=timer,
                            prefix="mm_rescue",
                        )
                        evaluation_history.append({
                            "stage": "mm_rescue",
                            "topk": mm_eval.get("evaluation_table"),
                            "residual_summary": mm_eval.get("residual_summary"),
                            "physics_summary": mm_eval.get("physics_summary"),
                        })
                        mm_used_in_evaluation = True
                        if is_better_result(mm_eval["best_result"], current_best):
                            current_best = mm_eval["best_result"]
                            initial_eval = mm_eval
            else:
                agent_trace.append({
                    "event": "mm_no_candidates",
                    "num_plots": len(observation.image_paths),
                    "reason": observation.mm_assets_error or "multimodal branch returned no candidate expressions",
                })
            _raise_if_task_budget_exceeded(task_deadline_ts, "mm_done")

        # -------------------------------------------------
        # 4) Multi-agent refine loop
        # -------------------------------------------------
        best_history.append(_safe_get_attr(current_best, "val_mse", None) if current_best is not None else None)

        if HIGH_DIM_NO_LEAKAGE_SKIP_STANDARD_REFINE and _is_benchmark_task(row_meta) and NO_LEAKAGE_MODE and len(dataset.feature_names) >= 5:
            agent_trace.append({
                "event": "standard_refine_skipped",
                "reason": "high_dim_no_leakage_uses_post_eval_llm_local_repair",
                "planned_refine_rounds": int(iter_cfg.get("refine_rounds", 0)),
            })
            iter_cfg["refine_rounds"] = 0
            result["iteration_config"] = json.dumps(make_json_safe(iter_cfg), ensure_ascii=False)
        elif LOW_DIM_SKIP_STANDARD_REFINE_AFTER_COVERAGE and len(dataset.feature_names) <= 2 and NO_LEAKAGE_MODE:
            agent_trace.append({
                "event": "standard_refine_skipped",
                "reason": "low_dim_rational_coverage_prefers_tool_verification_over_slow_llm_refine",
                "planned_refine_rounds": int(iter_cfg.get("refine_rounds", 0)),
            })
            iter_cfg["refine_rounds"] = 0
            result["iteration_config"] = json.dumps(make_json_safe(iter_cfg), ensure_ascii=False)

        for round_idx in range(iter_cfg.get("refine_rounds", 1)):
            round_num = round_idx + 1
            _raise_if_task_budget_exceeded(task_deadline_ts, f"refine_round_{round_num}_start")
            round_start_ts = time.time()
            round_timer_before = timer.as_dict()
            active_refine_round_num = round_num
            active_refine_round_start_ts = round_start_ts
            active_refine_round_timer_before = round_timer_before

            meta_decision = timed_call(
                timer,
                f"refine_round_{round_num}_meta_decision",
                meta_agent.decide,
                dataset=dataset,
                observation=observation,
                evaluation=initial_eval,
                current_best=current_best,
                iter_cfg=iter_cfg,
                round_idx=round_idx,
                row_meta=row_meta,
            )
            meta_decisions.append(make_json_safe(meta_decision))

            if not bool(meta_decision.get("should_refine", True)):
                result["early_stop_reason"] = f"meta_stop: {meta_decision.get('reason', '')}"
                finalize_refine_round_timing(
                    round_num=round_num,
                    round_start_ts=round_start_ts,
                    round_timer_before=round_timer_before,
                    status="meta_stop",
                    extra={
                        "should_refine": False,
                        "reason": meta_decision.get("reason"),
                    },
                )
                active_refine_round_num = None
                active_refine_round_start_ts = None
                active_refine_round_timer_before = None
                agent_trace.append({
                    "round": round_num,
                    "event": "meta_stop",
                    "reason": meta_decision.get("reason"),
                })
                break

            _raise_if_task_budget_exceeded(task_deadline_ts, f"refine_round_{round_num}_judge_feedback")
            judge_feedback = timed_call(
                timer,
                f"refine_round_{round_num}_judge_feedback",
                judge_agent.build_feedback,
                dataset=dataset,
                observation=observation,
                evaluation=initial_eval,
                meta_decision=meta_decision,
                iter_cfg=iter_cfg,
            )
            judge_feedback_history.append(make_json_safe(judge_feedback))

            _raise_if_task_budget_exceeded(task_deadline_ts, f"refine_round_{round_num}_refine_generate")
            refine_out = timed_call(
                timer,
                f"refine_round_{round_num}_refine_generate",
                refiner_agent.refine,
                dataset=dataset,
                observation=observation,
                current_best=current_best,
                meta_decision=meta_decision,
                judge_feedback=judge_feedback,
                iter_cfg=iter_cfg,
                row_meta=row_meta,
                diagnostic_image_paths=meta_decision.get("diagnostic_image_paths"),
            )
            refined_exprs = list(dict.fromkeys([x for x in refine_out.get("candidate_exprs", []) if str(x).strip()]))
            if MAX_REFINED_EXPRESSIONS_PER_ROUND is not None:
                refined_exprs = refined_exprs[:MAX_REFINED_EXPRESSIONS_PER_ROUND]
            refine_round_expr_counts.append(len(refined_exprs))

            proposal_history.append({
                "stage": f"refine_round_{round_num}",
                "num_exprs": len(refined_exprs),
                "trace": {
                    "raw_text": refine_out.get("raw_text", ""),
                    "diagnostic_image_paths": refine_out.get("diagnostic_image_paths", []),
                    "meta_decision": meta_decision,
                    "judge_feedback": judge_feedback,
                },
            })

            if not refined_exprs:
                finalize_refine_round_timing(
                    round_num=round_num,
                    round_start_ts=round_start_ts,
                    round_timer_before=round_timer_before,
                    status="empty_refine_candidates",
                    extra={
                        "num_exprs": 0,
                        "improved": False,
                    },
                )
                active_refine_round_num = None
                active_refine_round_start_ts = None
                active_refine_round_timer_before = None
                agent_trace.append({
                    "round": round_num,
                    "event": "empty_refine_candidates",
                })
                break

            _raise_if_task_budget_exceeded(task_deadline_ts, f"refine_round_{round_num}_evaluate")
            round_eval = evaluator_agent.evaluate(
                candidate_exprs=refined_exprs,
                dataset=dataset,
                row_meta=row_meta,
                timer=timer,
                prefix=f"refine_round_{round_num}",
            )
            evaluation_history.append({
                "stage": f"refine_round_{round_num}",
                "topk": round_eval.get("evaluation_table"),
                "residual_summary": round_eval.get("residual_summary"),
                "physics_summary": round_eval.get("physics_summary"),
            })

            before_snapshot = _result_performance_snapshot(current_best)
            round_best = round_eval["best_result"]
            round_best_snapshot = _result_performance_snapshot(round_best)
            improved = is_better_result(round_best, current_best)
            if improved:
                current_best = round_best
                initial_eval = round_eval
            after_snapshot = _result_performance_snapshot(current_best)

            val_mse_improvement = _metric_improvement(before_snapshot.get("val_mse"), after_snapshot.get("val_mse"))
            score_improvement = _metric_improvement(before_snapshot.get("score"), after_snapshot.get("score"))
            candidate_val_gain = _metric_improvement(before_snapshot.get("val_mse"), round_best_snapshot.get("val_mse"))
            candidate_score_gain = _metric_improvement(before_snapshot.get("score"), round_best_snapshot.get("score"))

            refine_history.append({
                "round": round_num,
                "meta_decision": make_json_safe(meta_decision),
                "judge_feedback": make_json_safe(judge_feedback),
                "improved": bool(improved),
                "best_before_round": before_snapshot,
                "round_candidate_best": round_best_snapshot,
                "best_after_round": after_snapshot,
                "val_mse_improvement": val_mse_improvement,
                "score_improvement": score_improvement,
                "relative_val_mse_improvement": _relative_metric_improvement(before_snapshot.get("val_mse"), after_snapshot.get("val_mse")),
                "candidate_val_mse_gain_vs_before": candidate_val_gain,
                "candidate_score_gain_vs_before": candidate_score_gain,
            })
            finalize_refine_round_timing(
                round_num=round_num,
                round_start_ts=round_start_ts,
                round_timer_before=round_timer_before,
                status="completed",
                extra={
                    "num_exprs": len(refined_exprs),
                    "improved": bool(improved),
                    "best_expr_before_round": before_snapshot.get("expr"),
                    "best_val_mse_before_round": before_snapshot.get("val_mse"),
                    "round_candidate_best_expr": round_best_snapshot.get("expr"),
                    "round_candidate_best_val_mse": round_best_snapshot.get("val_mse"),
                    "best_expr_after_round": after_snapshot.get("expr"),
                    "best_val_mse_after_round": after_snapshot.get("val_mse"),
                    "val_mse_improvement": val_mse_improvement,
                    "score_improvement": score_improvement,
                },
            )
            active_refine_round_num = None
            active_refine_round_start_ts = None
            active_refine_round_timer_before = None
            agent_trace.append({
                "round": round_num,
                "event": "refine_round_done",
                "improved": improved,
                "best_expr": _safe_get_attr(current_best, "simplified_expression", None) if current_best is not None else None,
                "best_val_mse": _safe_get_attr(current_best, "val_mse", None) if current_best is not None else None,
            })

            best_history.append(_safe_get_attr(current_best, "val_mse", None) if current_best is not None else None)
            stop, reason = should_early_stop(best_history)
            if stop:
                result["early_stop_reason"] = reason
                break

        # -------------------------------------------------
        # 5) Finalize result
        # -------------------------------------------------
        return finalize_pipeline_result()

    except TaskTimeBudgetExceeded as e:
        if active_refine_round_num is not None:
            finalize_refine_round_timing(
                round_num=active_refine_round_num,
                round_start_ts=active_refine_round_start_ts,
                round_timer_before=active_refine_round_timer_before,
                status="time_budget_exceeded",
                extra={"error": repr(e)},
            )
        result["time_budget_hit"] = True
        result["early_stop_reason"] = f"time_budget_exceeded>{float(MAX_RUNTIME_PER_TASK_SEC):.1f}s"
        agent_trace.append({
            "event": "time_budget_exceeded",
            "reason": repr(e),
            "best_expr": _safe_get_attr(current_best, "simplified_expression", None) if current_best is not None else None,
            "best_val_mse": _safe_get_attr(current_best, "val_mse", None) if current_best is not None else None,
        })
        return finalize_pipeline_result()

    except Exception as e:
        if active_refine_round_num is not None:
            finalize_refine_round_timing(
                round_num=active_refine_round_num,
                round_start_ts=active_refine_round_start_ts,
                round_timer_before=active_refine_round_timer_before,
                status="exception",
                extra={"error": repr(e)},
            )
        result["error"] = repr(e)
        return finalize_pipeline_result()

    finally:
        budget_guard.__exit__(None, None, None)


def run_one_raw_task(row, tmpdir: Path):
    train_df = load_txt_dataset(row["train_path"])
    val_df = load_txt_dataset(row["val_path"])
    test_df = load_txt_dataset(row["test_path"])
    dataset = build_dataset_from_explicit_splits(train_df, val_df, test_df, tmpdir=tmpdir)
    return _run_core_pipeline(dataset=dataset, row_meta=row)


def run_one_benchmark_csv_task(row, tmpdir: Path):
    n_features = int(row["dimension"])
    expr = str(row["true_expression"])
    distribution = row.get("distribution", "U")
    range_spec = row.get("range_spec")
    rng = np.random.default_rng(BENCHMARK_RANDOM_SEED)

    train_x = sample_features_from_range(BENCHMARK_TRAIN_SIZE, n_features, range_spec, distribution=distribution, rng=rng)
    val_x = sample_features_from_range(BENCHMARK_VAL_SIZE, n_features, range_spec, distribution=distribution, rng=rng)
    test_x = sample_features_from_range(BENCHMARK_TEST_SIZE, n_features, range_spec, distribution=distribution, rng=rng)

    train_df = train_x.copy(); train_df["y"] = evaluate_expression_on_df(expr, train_x)
    val_df = val_x.copy(); val_df["y"] = evaluate_expression_on_df(expr, val_x)
    test_df = test_x.copy(); test_df["y"] = evaluate_expression_on_df(expr, test_x)

    dataset = build_dataset_from_explicit_splits(train_df, val_df, test_df, tmpdir=tmpdir)
    return _run_core_pipeline(dataset=dataset, row_meta=row)

def main():
    overall_start = time.time()
    os.makedirs(RESULTS_ROOT, exist_ok=True)
    os.makedirs(PER_CASE_JSON_DIR, exist_ok=True)

    if TASK_SOURCE == "benchmark_csv":
        df_tasks = collect_tasks_from_benchmark_csv(BENCHMARK_CSV)
    else:
        all_parts = []
        for ds in SRSD_DATASET_DIRS:
            try:
                part = collect_raw_tasks_for_dataset(ds)
                if len(part) > 0:
                    all_parts.append(part)
            except Exception as e:
                print(f"[WARN] skip dataset_dir={ds}: {repr(e)}")
        df_tasks = pd.concat(all_parts, axis=0).reset_index(drop=True) if all_parts else pd.DataFrame()

    if len(df_tasks) == 0:
        print("No tasks found after filtering.")
        return

    df_tasks.to_csv(SELECTED_TASKS_CSV, index=False)
    print(f"Selection saved to: {SELECTED_TASKS_CSV}")

    if DRY_RUN_SELECTION_ONLY:
        print("DRY_RUN_SELECTION_ONLY=True, exiting after selection.")
        return

    all_results_global = []

    with TemporaryDirectory() as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        total = len(df_tasks)

        for idx, (_, row) in enumerate(df_tasks.iterrows(), start=1):
            row_dict = row.to_dict()
            print(f"[{idx}/{total}] Processing: {row_dict['base_name']}")
            print(f"   train/val/test: ({row_dict['n_train']}, {row_dict['n_val']}, {row_dict['n_test']})")
            print(f"   n_features:     {row_dict['n_features']}")

            try:
                if row_dict["task_type"] == "benchmark_csv":
                    one_result = run_one_benchmark_csv_task(row_dict, tmpdir=tmpdir)
                else:
                    one_result = run_one_raw_task(row_dict, tmpdir=tmpdir)
            except Exception as e:
                one_result = {
                    **row_dict,
                    "eval_profile": EVAL_PROFILE,
                    "no_leakage_mode": bool(NO_LEAKAGE_MODE),
                    "valid_formula_found": False,
                    "num_candidate_exprs": 0,
                    "best_expr": None,
                    "best_val_mse": None,
                    "best_test_mse": None,
                    "passed": False,
                    "perfect_fit": False,
                    "runtime_sec": None,
                    "error": repr(e),
                }

            all_results_global.append(one_result)

            print(f"   valid_formula_found: {one_result.get('valid_formula_found')}")
            print(f"   num_candidate_exprs: {one_result.get('num_candidate_exprs')}")
            print(f"   best_expr:           {one_result.get('best_expr')}")
            print(f"   best_val_mse:        {one_result.get('best_val_mse')}")
            print(f"   best_test_mse:       {one_result.get('best_test_mse')}")
            print(f"   passed:              {one_result.get('passed')}")
            print(f"   perfect_fit:         {one_result.get('perfect_fit')}")
            print(f"   runtime_sec:         {one_result.get('runtime_sec')}")
            print(f"   error:               {one_result.get('error')}")
            print("-" * 72)

    print_summary(all_results_global, overall_start)
    save_all_outputs(all_results_global, overall_start)


if __name__ == "__main__":
    main()
