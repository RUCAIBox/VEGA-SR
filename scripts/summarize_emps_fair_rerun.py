#!/usr/bin/env python3
"""Build the portable per-run and summary files required by the EMPS guide."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


METRICS = ("test_nmse", "test_mse", "test_r2", "expr_complexity", "runtime_sec")


def finite(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--method", default="vega_sr")
    parser.add_argument("--dataset", default="emps")
    args = parser.parse_args()
    root = Path(args.results_dir).resolve()
    case_dir = root / "cases" / args.method / args.dataset
    rows: list[dict[str, Any]] = []
    for path in sorted(case_dir.glob("seed_*.json"), key=lambda p: int(p.stem.split("_")[-1])):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            row = {"status": "error", "error": repr(exc)}
        row["result_path"] = str(path.relative_to(root))
        rows.append(row)

    columns = [
        "repeat_seed", "status", "best_expr", "test_nmse", "test_mse", "test_r2",
        "expr_complexity", "runtime_sec", "total_wall_sec", "candidate_count",
        "valid_candidate_count", "prior_mode", "protected_template_count",
        "injected_templates", "test_rows_visible_to_search_pipeline", "result_path", "error",
    ]
    with (root / "per_run_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    valid = [row for row in rows if row.get("status") == "ok" and finite(row.get("test_nmse")) is not None]
    summary: dict[str, Any] = {
        "method": args.method,
        "dataset": args.dataset,
        "n_runs": len(rows),
        "n_valid": len(valid),
        "n_timeouts": sum(row.get("status") == "timed_out" for row in rows),
        "n_errors": sum(row.get("status") == "error" for row in rows),
    }
    for metric in METRICS:
        values = np.asarray([value for row in valid if (value := finite(row.get(metric))) is not None], dtype=float)
        if len(values):
            summary[f"{metric}_median"] = float(np.median(values))
            summary[f"{metric}_q1"] = float(np.quantile(values, 0.25))
            summary[f"{metric}_q3"] = float(np.quantile(values, 0.75))
    with (root / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
