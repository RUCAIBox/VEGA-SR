#!/usr/bin/env python3
"""Aggregate completed PSE real-world case JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results/pse_realworld_validation")
    args = parser.parse_args()
    root = Path(args.results_dir).resolve()
    rows = []
    for path in sorted((root / "cases").glob("*/*/seed_*.json")):
        with path.open(encoding="utf-8") as handle:
            row = json.load(handle)
        row["case_json"] = str(path)
        row.pop("pareto_candidates", None)
        rows.append(row)
    if not rows:
        print("No completed case JSON files found.")
        return 0
    frame = pd.DataFrame(rows)
    root.mkdir(parents=True, exist_ok=True)
    frame.to_csv(root / "all_results.csv", index=False)
    status_counts = frame.groupby(["method", "dataset", "status"], dropna=False).size().reset_index(name="count")
    status_counts.to_csv(root / "status_counts.csv", index=False)
    ok = frame.loc[frame["status"] == "ok"].copy()
    metrics = ["test_mse", "test_nmse", "test_r2", "expr_complexity", "runtime_sec"]
    aggregations = {}
    for metric in metrics:
        if metric in ok.columns:
            aggregations[f"median_{metric}"] = (metric, "median")
            aggregations[f"mean_{metric}"] = (metric, "mean")
    if aggregations:
        summary = ok.groupby(["method", "dataset"], dropna=False).agg(
            completed=("repeat_seed", "count"),
            **aggregations,
        ).reset_index()
    else:
        summary = ok.groupby(["method", "dataset"], dropna=False).size().reset_index(name="completed")
    summary.to_csv(root / "summary.csv", index=False)
    print(status_counts.to_string(index=False))
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
