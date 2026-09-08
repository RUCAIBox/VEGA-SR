#!/usr/bin/env python3
"""Combine PSE/VEGA-SR and added baseline case files into one comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary-results-dir", default="results/pse_realworld_validation")
    parser.add_argument("--baseline-results-dir", default="results/pse_realworld_baselines")
    parser.add_argument("--output-dir", default="results/pse_realworld_comparison")
    args = parser.parse_args()

    rows = []
    for source, root in (
        ("primary", Path(args.primary_results_dir).resolve()),
        ("added_baseline", Path(args.baseline_results_dir).resolve()),
    ):
        for path in sorted((root / "cases").glob("*/*/seed_*.json")):
            with path.open(encoding="utf-8") as handle:
                row = json.load(handle)
            row["result_source"] = source
            row["case_json"] = str(path)
            row.pop("pareto_candidates", None)
            row.pop("candidate_evaluation_history", None)
            rows.append(row)
    if not rows:
        print("No case JSON files found.")
        return 0

    frame = pd.DataFrame(rows)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "all_results.csv", index=False)
    statuses = frame.groupby(["method", "dataset", "status"], dropna=False).size().reset_index(name="count")
    statuses.to_csv(output / "status_counts.csv", index=False)

    ok = frame.loc[frame["status"] == "ok"].copy()
    metrics = ["test_mse", "test_nmse", "test_r2", "expr_complexity", "runtime_sec"]
    aggregations = {}
    for metric in metrics:
        if metric in ok.columns:
            aggregations[f"median_{metric}"] = (metric, "median")
            aggregations[f"mean_{metric}"] = (metric, "mean")
    summary = ok.groupby(["method", "dataset"], dropna=False).agg(
        completed=("repeat_seed", "count"),
        valid_test_predictions=("test_mse", "count"),
        **aggregations,
    ).reset_index()
    summary["invalid_test_predictions"] = summary["completed"] - summary["valid_test_predictions"]
    summary["valid_test_prediction_rate"] = summary["valid_test_predictions"] / summary["completed"]
    summary["expected"] = summary["method"].map(
        lambda method: 1 if method in {"physics_ls", "linear_ls"} else 20
    )
    summary["complete"] = summary["completed"] >= summary["expected"]
    summary.to_csv(output / "summary.csv", index=False)
    print(statuses.to_string(index=False))
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
