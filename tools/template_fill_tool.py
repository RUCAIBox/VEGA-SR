# tools/template_fill_tool.py
from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
import os
import threading
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import sympy as sp
from scipy.optimize import least_squares
from sklearn.metrics import mean_squared_error

from .dataset_loader import DatasetBundle


@dataclass
class FitResult:
    """
    单个候选表达式拟合后的结果。
    """
    expression: str
    fitted_expression: str

    parameters: Dict[str, float]

    train_mse: Optional[float]
    val_mse: Optional[float]
    test_mse: Optional[float]

    success: bool
    error_message: Optional[str] = None


class TemplateFillTool:
    """
    对表达式模板进行参数拟合。

    设计思路：
    - 输入是 LLM 生成的表达式字符串，例如 "a * x0 / (b + x1)"
    - 自动识别自由参数，例如 a, b
    - 用 least_squares 拟合这些参数
    - 返回数值化后的拟合表达式和 train/validation 误差

    ``test_df`` is deliberately not read here.  Test evaluation happens once,
    after the search loop has selected its final expression.
    """

    # 你可以按需扩展这一组“默认不是参数”的符号
    DEFAULT_RESERVED_SYMBOLS = {
        "x", "y",
        "x0", "x1", "x2", "x3", "x4", "x5",
        "sin", "cos", "tan", "sinh", "cosh", "tanh",
        "exp", "log", "sqrt", "abs", "Abs",
        "pi", "E"
    }

    SYMPY_LOCALS = {
        "sin": sp.sin,
        "cos": sp.cos,
        "tan": sp.tan,
        "sinh": sp.sinh,
        "cosh": sp.cosh,
        "tanh": sp.tanh,
        "exp": sp.exp,
        "log": sp.log,
        "sqrt": sp.sqrt,
        "abs": sp.Abs,
        "Abs": sp.Abs,
        "pi": sp.pi,
        "E": sp.E,
    }

    NUMPY_LAMBDIFY_MODULE = {
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
    }

    DEFAULT_MAX_NFEV = max(200, int(os.environ.get("LLMSR_TEMPLATE_FIT_MAX_NFEV", "1200")))
    COMPLEX_MAX_NFEV = max(100, int(os.environ.get("LLMSR_TEMPLATE_FIT_COMPLEX_MAX_NFEV", "600")))
    COMPLEX_EXPR_LEN = max(40, int(os.environ.get("LLMSR_TEMPLATE_FIT_COMPLEX_EXPR_LEN", "120")))
    COMPLEX_PARAM_COUNT = max(2, int(os.environ.get("LLMSR_TEMPLATE_FIT_COMPLEX_PARAM_COUNT", "4")))
    DEADLINE_POLL_INTERVAL_SEC = max(0.01, float(os.environ.get("LLMSR_TEMPLATE_FIT_DEADLINE_POLL_SEC", "0.1")))
    STRICT_DEADLINE_SERIAL = os.environ.get("LLMSR_TEMPLATE_FIT_STRICT_DEADLINE_SERIAL", "1").strip().lower() in {"1", "true", "yes", "y"}

    def __init__(self, max_workers: int = 1):
        self.max_workers = max(1, int(max_workers or 1))

    def _choose_max_nfev(self, expr_str: str, param_count: int) -> int:
        expr_len = len(str(expr_str or ""))
        max_nfev = int(self.DEFAULT_MAX_NFEV)
        if expr_len >= self.COMPLEX_EXPR_LEN or int(param_count) >= self.COMPLEX_PARAM_COUNT:
            max_nfev = min(max_nfev, int(self.COMPLEX_MAX_NFEV))
        return max(100, int(max_nfev))

    def _deadline_exceeded(self, deadline_ts: Optional[float], abort_event=None) -> bool:
        if abort_event is not None and abort_event.is_set():
            return True
        if deadline_ts is None:
            return False
        return time.time() >= float(deadline_ts)

    def _raise_if_deadline_exceeded(self, deadline_ts: Optional[float], abort_event=None):
        if self._deadline_exceeded(deadline_ts, abort_event=abort_event):
            if abort_event is not None:
                abort_event.set()
            raise TimeoutError("template fit deadline exceeded")

    def _parse_expr(self, expr_str: str) -> sp.Expr:
        """
        把字符串解析成 sympy 表达式。
        """
        try:
            expr = sp.sympify(expr_str, locals=self.SYMPY_LOCALS)
            return expr
        except Exception as e:
            raise ValueError(f"表达式解析失败: {expr_str}\n{e}")

    def _get_parameter_symbols(self, expr: sp.Expr, feature_names: List[str]) -> List[sp.Symbol]:
        """
        自动识别表达式中的自由参数。

        规则：
        - 表达式中出现的 symbol，如果不在 feature_names 中，就视为待拟合参数
        """
        feature_set = set(feature_names)
        params = []

        for symbol in expr.free_symbols:
            name = str(symbol)
            if name not in feature_set:
                params.append(symbol)

        # 固定排序，保证参数顺序稳定
        params = sorted(params, key=lambda s: str(s))
        return params

    def _build_numeric_function(
        self,
        expr: sp.Expr,
        feature_names: List[str],
        param_symbols: List[sp.Symbol],
    ):
        """
        用 sympy.lambdify 把表达式转成可数值计算的函数。

        输入顺序固定为：
            [x0, x1, ..., params...]
        """
        feature_symbols = [sp.Symbol(name) for name in feature_names]
        ordered_symbols = feature_symbols + param_symbols

        func = sp.lambdify(ordered_symbols, expr, modules=[self.NUMPY_LAMBDIFY_MODULE, "numpy"])
        return func

    def _predict(
        self,
        func,
        df: pd.DataFrame,
        feature_names: List[str],
        param_values: np.ndarray,
    ) -> np.ndarray:
        """
        给定已编译函数和参数值，计算预测结果。
        """
        feature_arrays = [df[name].to_numpy(dtype=float) for name in feature_names]
        args = feature_arrays + list(param_values)

        y_pred = func(*args)
        y_pred = np.asarray(y_pred, dtype=float)

        # 有些表达式返回标量，需要扩成同长度数组
        if y_pred.ndim == 0:
            y_pred = np.full(shape=(len(df),), fill_value=float(y_pred))

        return y_pred

    def _fit_single_expression(
        self,
        expr_str: str,
        dataset: DatasetBundle,
        n_restarts: int = 5,
        init_scale: float = 1.0,
        deadline_ts: Optional[float] = None,
        abort_event=None,
    ) -> FitResult:
        """
        对单个表达式做拟合。
        """
        try:
            self._raise_if_deadline_exceeded(deadline_ts, abort_event=abort_event)
            expr = self._parse_expr(expr_str)
            param_symbols = self._get_parameter_symbols(expr, dataset.feature_names)

            # 没有自由参数时，直接当成固定表达式评估
            func = self._build_numeric_function(expr, dataset.feature_names, param_symbols)

            train_y = dataset.train_df[dataset.target_name].to_numpy(dtype=float)
            val_y = dataset.val_df[dataset.target_name].to_numpy(dtype=float)
            if len(param_symbols) == 0:
                train_pred = self._predict(func, dataset.train_df, dataset.feature_names, np.array([]))
                val_pred = self._predict(func, dataset.val_df, dataset.feature_names, np.array([]))

                return FitResult(
                    expression=expr_str,
                    fitted_expression=str(expr),
                    parameters={},
                    train_mse=float(mean_squared_error(train_y, train_pred)),
                    val_mse=float(mean_squared_error(val_y, val_pred)),
                    test_mse=None,
                    success=True,
                    error_message=None,
                )

            # 定义残差函数，least_squares 会最小化 residuals 的平方和
            def residuals_fn(params: np.ndarray) -> np.ndarray:
                self._raise_if_deadline_exceeded(deadline_ts, abort_event=abort_event)
                pred = self._predict(func, dataset.train_df, dataset.feature_names, params)

                # 防御性处理：过滤 nan/inf
                if not np.all(np.isfinite(pred)):
                    return np.full_like(train_y, 1e6, dtype=float)

                return pred - train_y

            best_params = None
            best_cost = np.inf

            # 多次随机初始化，提高拟合稳定性
            rng = np.random.default_rng(42)
            restart_scales = [max(1e-3, init_scale * factor) for factor in (0.25, 1.0, 2.0, 4.0)]
            param_names = [str(sym).lower() for sym in param_symbols]
            max_nfev = self._choose_max_nfev(expr_str, len(param_symbols))

            def _default_seed():
                seed = np.zeros(len(param_symbols), dtype=float)
                for i, name in enumerate(param_names):
                    if name.startswith(("a", "scale", "gain", "amp", "alpha")):
                        seed[i] = 1.0
                    elif name.startswith(("p", "q", "pow", "exp")):
                        seed[i] = 1.0
                    else:
                        seed[i] = 0.0
                return seed

            deterministic_seeds = []
            if len(param_symbols) > 0:
                base_seed = _default_seed()
                deterministic_seeds.extend([
                    base_seed,
                    np.zeros(len(param_symbols), dtype=float),
                    0.5 * base_seed,
                    2.0 * base_seed,
                    -1.0 * base_seed,
                ])

            init_guesses = []
            seen_seeds = set()

            def _add_seed(seed):
                seed = np.asarray(seed, dtype=float)
                key = tuple(np.round(seed, 8).tolist())
                if key in seen_seeds:
                    return
                seen_seeds.add(key)
                init_guesses.append(seed)

            for seed in deterministic_seeds:
                _add_seed(seed)
                if len(init_guesses) >= max(1, n_restarts):
                    break

            while len(init_guesses) < max(1, n_restarts):
                restart_idx = len(init_guesses)
                scale = restart_scales[restart_idx % len(restart_scales)]
                init_params = rng.normal(loc=0.0, scale=scale, size=len(param_symbols))
                _add_seed(init_params)

            for init_params in init_guesses:
                self._raise_if_deadline_exceeded(deadline_ts, abort_event=abort_event)

                try:
                    res = least_squares(residuals_fn, x0=init_params, max_nfev=max_nfev)
                    if res.cost < best_cost and np.all(np.isfinite(res.x)):
                        best_cost = res.cost
                        best_params = res.x
                except Exception:
                    continue

            if best_params is None:
                return FitResult(
                    expression=expr_str,
                    fitted_expression=expr_str,
                    parameters={},
                    train_mse=None,
                    val_mse=None,
                    test_mse=None,
                    success=False,
                    error_message="参数拟合失败，没有找到可用解。",
                )

            train_pred = self._predict(func, dataset.train_df, dataset.feature_names, best_params)
            val_pred = self._predict(func, dataset.val_df, dataset.feature_names, best_params)

            param_dict = {str(sym): float(val) for sym, val in zip(param_symbols, best_params)}

            # 把参数代回表达式，得到拟合后的数值表达式
            fitted_expr = expr.subs({sym: val for sym, val in zip(param_symbols, best_params)})
            # Search-time validity is defined only on fitting and validation
            # data. A selected expression that is invalid on held-out inputs is
            # reported as such after selection; it never triggers fallback to a
            # different candidate.
            fit_success = bool(
                np.all(np.isfinite(train_pred))
                and np.all(np.isfinite(val_pred))
            )

            return FitResult(
                expression=expr_str,
                fitted_expression=str(fitted_expr),
                parameters=param_dict,
                train_mse=float(mean_squared_error(train_y, train_pred)),
                val_mse=float(mean_squared_error(val_y, val_pred)),
                test_mse=None,
                success=fit_success,
                error_message=None,
            )

        except Exception as e:
            return FitResult(
                expression=expr_str,
                fitted_expression=expr_str,
                parameters={},
                train_mse=None,
                val_mse=None,
                test_mse=None,
                success=False,
                error_message=str(e),
            )

    def run(
        self,
        candidate_expressions: List[str],
        dataset: DatasetBundle,
        n_restarts: int = 5,
        init_scale: float = 1.0,
        deadline_ts: Optional[float] = None,
    ) -> List[FitResult]:
        """
        对一组候选表达式逐个拟合。
        当候选较多时，允许多路并行以缩短 template fit 总时间。
        """
        candidate_expressions = list(candidate_expressions or [])
        if not candidate_expressions:
            return []

        abort_event = threading.Event() if deadline_ts is not None else None
        force_serial = bool(deadline_ts is not None and self.STRICT_DEADLINE_SERIAL)
        if len(candidate_expressions) <= 1 or self.max_workers <= 1 or force_serial:
            results = []
            for expr_str in candidate_expressions:
                if self._deadline_exceeded(deadline_ts, abort_event=abort_event):
                    break
                result = self._fit_single_expression(
                    expr_str=expr_str,
                    dataset=dataset,
                    n_restarts=n_restarts,
                    init_scale=init_scale,
                    deadline_ts=deadline_ts,
                    abort_event=abort_event,
                )
                results.append(result)
                if self._deadline_exceeded(deadline_ts, abort_event=abort_event):
                    break
            return results

        worker_count = min(self.max_workers, len(candidate_expressions))
        indexed_results = {}

        def _fit_one(index: int, expr_str: str):
            return index, self._fit_single_expression(
                expr_str=expr_str,
                dataset=dataset,
                n_restarts=n_restarts,
                init_scale=init_scale,
                deadline_ts=deadline_ts,
                abort_event=abort_event,
            )

        future_to_idx = {}
        executor = ThreadPoolExecutor(max_workers=worker_count)
        try:
            pending = set()
            for idx, expr_str in enumerate(candidate_expressions):
                if self._deadline_exceeded(deadline_ts, abort_event=abort_event):
                    break
                fut = executor.submit(_fit_one, idx, expr_str)
                future_to_idx[fut] = idx
                pending.add(fut)

            while pending:
                timeout = self.DEADLINE_POLL_INTERVAL_SEC
                if deadline_ts is not None:
                    remaining = float(deadline_ts) - time.time()
                    if remaining <= 0:
                        if abort_event is not None:
                            abort_event.set()
                        break
                    timeout = min(timeout, max(0.01, float(remaining)))
                done, pending = wait(pending, timeout=timeout, return_when=FIRST_COMPLETED)
                if not done:
                    continue
                for fut in done:
                    idx = future_to_idx.get(fut)
                    if idx is None or fut.cancelled():
                        continue
                    try:
                        out_idx, result = fut.result()
                    except Exception as e:
                        indexed_results[idx] = FitResult(
                            expression=candidate_expressions[idx],
                            fitted_expression=candidate_expressions[idx],
                            parameters={},
                            train_mse=None,
                            val_mse=None,
                            test_mse=None,
                            success=False,
                            error_message=str(e),
                        )
                        continue
                    indexed_results[out_idx] = result

            if abort_event is not None and abort_event.is_set():
                for fut in list(pending):
                    fut.cancel()
        finally:
            executor.shutdown(wait=True, cancel_futures=True)

        for fut, idx in future_to_idx.items():
            if idx in indexed_results or fut.cancelled() or not fut.done():
                continue
            try:
                out_idx, result = fut.result()
            except Exception as e:
                indexed_results[idx] = FitResult(
                    expression=candidate_expressions[idx],
                    fitted_expression=candidate_expressions[idx],
                    parameters={},
                    train_mse=None,
                    val_mse=None,
                    test_mse=None,
                    success=False,
                    error_message=str(e),
                )
                continue
            indexed_results[out_idx] = result

        results = []
        for idx in range(len(candidate_expressions)):
            if idx in indexed_results:
                results.append(indexed_results[idx])
        return results
