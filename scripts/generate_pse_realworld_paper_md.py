#!/usr/bin/env python3
"""Generate the paper-facing Markdown for the PSE real-world experiment."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    ("pse", "emps"): 20,
    ("pse", "roughpipe"): 20,
    ("vega_sr", "emps"): 20,
    ("vega_sr", "roughpipe"): 20,
    ("pysr", "emps"): 20,
    ("pysr", "roughpipe"): 20,
    ("operon", "emps"): 20,
    ("operon", "roughpipe"): 20,
    ("gplearn", "emps"): 20,
    ("gplearn", "roughpipe"): 20,
    ("linear_ls", "emps"): 1,
    ("linear_ls", "roughpipe"): 1,
    ("physics_ls", "emps"): 1,
}
METHOD_NAMES = {
    "pse": "PSE",
    "vega_sr": "PSE-aligned VEGA-SR",
    "pysr": "PySR",
    "operon": "Operon",
    "gplearn": "gplearn",
    "linear_ls": "Linear-LS",
    "physics_ls": "Physics-LS (ablation)",
}
DATASET_NAMES = {"emps": "EMPS", "roughpipe": "Roughpipe"}


def load_rows(roots: list[Path]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for root in roots:
        for path in sorted((root / "cases").glob("*/*/seed_*.json")):
            with path.open(encoding="utf-8") as handle:
                row = json.load(handle)
            row["case_json"] = str(path)
            rows.append(row)
    return pd.DataFrame(rows)


def fmt(value: float, digits: int = 6) -> str:
    value = float(value)
    if value == 0:
        return "0"
    if abs(value) < 1e-4:
        return f"{value:.3e}"
    return f"{value:.{digits}f}"


def median_iqr(values: pd.Series, digits: int = 6) -> str:
    values = pd.to_numeric(values, errors="coerce").dropna()
    median = values.median()
    if len(values) == 1:
        return fmt(median, digits)
    q1, q3 = values.quantile([0.25, 0.75])
    return f"{fmt(median, digits)} [{fmt(q1, digits)}, {fmt(q3, digits)}]"


def representative(group: pd.DataFrame) -> pd.Series:
    median = pd.to_numeric(group["test_mse"], errors="coerce").median()
    distances = (pd.to_numeric(group["test_mse"], errors="coerce") - median).abs()
    return group.loc[distances.idxmin()]


def table(rows: list[list[str]], headers: list[str]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "pse_realworld_vega_sr.yaml"))
    parser.add_argument("--primary-results-dir", default=str(ROOT / "results/pse_realworld_validation"))
    parser.add_argument("--baseline-results-dir", default=str(ROOT / "results/pse_realworld_baselines"))
    parser.add_argument("--comparison-results-dir", default=str(ROOT / "results/pse_realworld_comparison"))
    parser.add_argument("--output", default=str(ROOT / "PSE_REALWORLD_PAPER_RESULTS.md"))
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    frame = load_rows([Path(args.primary_results_dir).resolve(), Path(args.baseline_results_dir).resolve()])
    if frame.empty:
        raise RuntimeError("no result JSON files found")
    ok = frame.loc[frame["status"] == "ok"].copy()
    counts = ok.groupby(["method", "dataset"]).size().to_dict()
    incomplete = {key: (counts.get(key, 0), expected) for key, expected in EXPECTED.items() if counts.get(key, 0) < expected}
    if incomplete and not args.allow_incomplete:
        raise RuntimeError(f"formal experiment is incomplete: {incomplete}")

    order = ["linear_ls", "pse", "pysr", "operon", "gplearn", "vega_sr", "physics_ls"]
    result_rows = []
    summary: dict[tuple[str, str], dict[str, float]] = {}
    for dataset in ("emps", "roughpipe"):
        for method in order:
            group = ok.loc[(ok["method"] == method) & (ok["dataset"] == dataset)]
            if group.empty:
                continue
            values = {
                metric: float(pd.to_numeric(group[metric], errors="coerce").median())
                for metric in ("test_mse", "test_nmse", "test_r2", "expr_complexity", "runtime_sec")
            }
            summary[(method, dataset)] = values
            valid_test = int(pd.to_numeric(group["test_mse"], errors="coerce").notna().sum())
            n_text = str(len(group)) if valid_test == len(group) else f"{len(group)} ({valid_test} valid)"
            result_rows.append([
                METHOD_NAMES[method],
                DATASET_NAMES[dataset],
                n_text,
                median_iqr(group["test_mse"]),
                median_iqr(group["test_nmse"]),
                median_iqr(group["test_r2"]),
                median_iqr(group["expr_complexity"], 1),
                median_iqr(group["runtime_sec"], 2),
            ])

    relative_rows = []
    for dataset in ("emps", "roughpipe"):
        pse = summary[("pse", dataset)]
        for method in ("pysr", "operon", "vega_sr"):
            if (method, dataset) not in summary:
                continue
            current = summary[(method, dataset)]
            mse_reduction = 100.0 * (pse["test_mse"] - current["test_mse"]) / pse["test_mse"]
            relative_rows.append([
                METHOD_NAMES[method],
                DATASET_NAMES[dataset],
                f"{mse_reduction:+.2f}%",
                f"{current['test_r2'] - pse['test_r2']:+.5f}",
                f"{current['expr_complexity'] - pse['expr_complexity']:+.1f}",
            ])

    expression_rows = []
    for dataset in ("emps", "roughpipe"):
        for method in order:
            group = ok.loc[(ok["method"] == method) & (ok["dataset"] == dataset)]
            if group.empty:
                continue
            row = representative(group)
            expression = str(row.get("best_expr") or "").replace("|", "\\|")
            expression_rows.append([
                METHOD_NAMES[method],
                DATASET_NAMES[dataset],
                str(int(row["repeat_seed"])),
                f"`{expression}`",
                fmt(row["test_mse"]),
            ])

    status_note = "全部正式实验已完成。" if not incomplete else f"草稿：尚未完成 {incomplete}。"
    generated = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    vega_emps = summary.get(("vega_sr", "emps"), {})
    physics_emps = summary.get(("physics_ls", "emps"), {})
    prior_equal = bool(
        vega_emps and physics_emps
        and math.isclose(vega_emps["test_mse"], physics_emps["test_mse"], rel_tol=1e-10, abs_tol=1e-12)
    )

    content = f"""# PSE real-world experiment: paper-ready configuration and results

Generated: {generated}

Status: {status_note}

## 1. Experiment scope

This experiment compares PSE, PSE-aligned VEGA-SR, PySR, and Operon on two real-world symbolic-regression tasks from the PSE study. Physics-LS is reported separately as an EMPS prior-only ablation rather than as a general symbolic-regression baseline. The goal is to compare predictive accuracy, symbolic complexity, and descriptive runtime under a common data split and a validation-only expression-selection protocol.

## 2. Reproducible protocol

- Repeats: 20 seeds (`0`–`19`) for PSE, VEGA-SR, PySR, and Operon. Physics-LS is deterministic and is run once.
- Search budget: 90 seconds per formal case. Runtime includes method initialization and search inside each case, but excludes one-time VEGA model-service startup.
- Selection: candidates are ranked using validation data with $0.99^C/(1+\\sqrt{{\\mathrm{{MSE}}_{{val}}}})$, where $C$ is the common SymPy-tree complexity; test metrics are not part of that ranking score. The current VEGA-SR fitter nevertheless evaluates candidate predictions on the test inputs while constructing fit records and may reject a numerically invalid expression, so this implementation does not satisfy the stronger claim that the test split remains completely unaccessed during search.
- Primary metrics: test MSE, NMSE, $R^2$, expression complexity, and descriptive runtime.
- Data and source pinning: PSE commit `{config['provenance']['pse_commit']}` and VEGA-SR reference commit `{config['provenance']['vega_sr_reference_commit']}`. Raw files are verified by the SHA-256 hashes in the YAML configuration.

### Data splits

| Dataset | Train | Validation | Test | Split rule |
|---|---:|---:|---:|---|
| EMPS | 4,000 | 1,000 | 5,001 | First half is discovery data; its first 80% is train and last 20% is validation; the untouched second half is test. |
| Roughpipe | 218 | 72 | 72 | After PSE's one-dimensional transform and stable sorting by $x$, samples follow a fixed repeating `train, train, train, val, test` assignment. |

### Method configurations

- **PSE:** released stage configurations (`EMPS.yaml` and `turbulence.yaml`), GP token generator, released operator sets, and PSE reward parameter $\\eta=0.99$.
- **PSE-aligned VEGA-SR:** planner-guided full pipeline, three planner/refinement rounds, restored text proposer, 90-second deadline, and validation-only selection. For EMPS it receives the published Newton/friction prior $M\\ddot q=-F_v\\dot q-F_c\\operatorname{{sign}}(\\dot q)+\\tau-c$ through three protected templates.
- **PySR:** PSE-released operator libraries; deterministic serial execution pinned to one CPU core with one Julia/BLAS thread and a 90-second timeout. The released maximum search work is preserved as 100 × 380 mutation cycles for EMPS and 50 × 380 for Roughpipe. For PySR 1.5, each cycle is made an interruptible iteration (`ncycles_per_iteration=1`, effective iteration caps 38,000 and 19,000) so that pathological seeds cannot overrun the same wall-clock deadline by several minutes. Explicit one-thread environment limits and CPU affinity implement the released script's disabled-multithreading intent on this host.
- **Operon:** PSE-released operators; objectives `(mse, length)`; population and pool size 1,000; four CPU threads; Levenberg–Marquardt local optimization; 90-second maximum time.
- **Physics-LS:** the same three EMPS physical templates supplied to VEGA-SR, with coefficients fitted by ordinary least squares. It performs no symbolic structure search.

EMPS operators are `+`, `-`, `*`, `/`, `sin`, `cos`, `exp`, `log`, `cosh`, `tanh`, `abs`, and `sign`. Roughpipe uses `+`, `-`, `*`, `/`, `sin`, `cos`, `exp`, `log`, `cosh`, `tanh`, square, and cube.

## 3. Main results

Values are median [Q1, Q3] over 20 seeds, except deterministic Physics-LS. Lower MSE/NMSE/complexity is better; higher $R^2$ is better.
For a validation-selected expression that is numerically invalid on held-out inputs, test metrics remain missing rather than selecting a fallback using test data; the `n` column reports the number of finite test predictions when this occurs.

{table(result_rows, ['Method', 'Dataset', 'n', 'Test MSE', 'Test NMSE', 'Test R²', 'Complexity', 'Runtime (s)'])}

### Relative to reproduced PSE

Positive MSE reduction and positive $R^2$ difference favor the compared method. Complexity difference is compared with PSE.

{table(relative_rows, ['Method', 'Dataset', 'Median MSE reduction', 'Median R² difference', 'Complexity difference'])}

## 4. Representative median-run expressions

The representative run is the seed whose test MSE is closest to the method's median; it is for readability and is not an additional selection step.

{table(expression_rows, ['Method', 'Dataset', 'Seed', 'Expression', 'Test MSE'])}

## 5. Ablation and interpretation

{"Physics-LS and PSE-aligned VEGA-SR have identical EMPS median test MSE. This shows that the current EMPS gain is attributable primarily to the injected Newton/friction template and coefficient fitting, not uniquely to VEGA-SR's structure-search procedure." if prior_equal else "Physics-LS quantifies how much of EMPS performance is explained by the injected Newton/friction templates without symbolic structure search."}

For Roughpipe, VEGA-SR should be interpreted jointly through accuracy and complexity: it improves predictive error over reproduced PSE, while its selected polynomial is structurally larger. Runtime is descriptive rather than a hardware-normalized efficiency claim because PSE and VEGA-SR use GPUs/model inference whereas PySR and Operon use CPUs.

## 6. Paper-ready result paragraph

Under a common validation-score ranking protocol, we evaluated PSE, PSE-aligned VEGA-SR, PySR, and Operon over 20 seeds on the EMPS and Roughpipe real-world tasks. All methods used the same fixed train/validation/test partitions, and test metrics were excluded from the explicit Pareto-front ranking score. The current VEGA-SR implementation did evaluate test-domain predictions when constructing candidate fit records, so we do not claim a fully sealed test set during search. Table 1 reports median and interquartile-range test metrics. On EMPS, the PSE-aligned VEGA-SR configuration substantially reduces predictive error relative to reproduced PSE while adding only one common complexity node. However, a prior-only Physics-LS ablation attains the same solution, indicating that this gain is driven by the injected Newton/friction prior. On Roughpipe, VEGA-SR achieves lower median test error than reproduced PSE, PySR, and Operon, although this improvement is accompanied by a larger symbolic expression. These results support the benefit of physics-aligned candidate construction while also separating that benefit from the contribution of general symbolic search.

## 7. Important reporting caveats

1. The PSE article selected its reported EMPS Pareto expression using the test data. This reproduction introduces a validation split and excludes test metrics from the explicit ranking score, so the values are not intended as an exact reproduction of the paper figure. The current VEGA-SR fit-record implementation still evaluates test-domain predictions during candidate processing; a strictly sealed-test claim would require moving that evaluation after final selection and rerunning the affected experiment.
2. Physics-LS is an ablation, not a general baseline, and is applicable only to EMPS.
3. Runtime comparisons are not hardware-normalized and should not be presented as pure speedups.
4. PySR test-domain failures are retained as failures rather than using test data to choose another Pareto expression; its finite-prediction count is shown in the main table.
5. NGGP, DGSR, BMS, uDSR, TPSR, and wAIC were not added in this first extension because they require substantially heavier pretrained-model/dependency setup, and wAIC is available only for EMPS. PySR and Operon provide two directly released, representative search baselines under the strict split.

## 8. Artifact map and reproduction

- Canonical configuration: [`pse_realworld_vega_sr.yaml`](./pse_realworld_vega_sr.yaml)
- Case runner for PSE/VEGA-SR: [`scripts/run_pse_realworld_case.py`](./scripts/run_pse_realworld_case.py)
- Added baseline runner: [`scripts/run_pse_realworld_baseline_case.py`](./scripts/run_pse_realworld_baseline_case.py)
- Baseline launcher: [`scripts/launch_pse_realworld_baselines.sh`](./scripts/launch_pse_realworld_baselines.sh)
- Combined case-level CSV: [`paper_artifacts/pse_realworld/all_results.csv`](./paper_artifacts/pse_realworld/all_results.csv)
- Aggregated CSV: [`paper_artifacts/pse_realworld/summary.csv`](./paper_artifacts/pse_realworld/summary.csv)
- Original PSE/VEGA report: [`paper_artifacts/pse_realworld/REPORT.md`](./paper_artifacts/pse_realworld/REPORT.md)
- Integrity manifest: [`paper_artifacts/pse_realworld/SHA256SUMS`](./paper_artifacts/pse_realworld/SHA256SUMS)

Regenerate the aggregate CSV and this document with:

```bash
python scripts/summarize_pse_realworld_comparison.py
python scripts/generate_pse_realworld_paper_md.py
```
"""
    output = Path(args.output).resolve()
    output.write_text(content, encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
