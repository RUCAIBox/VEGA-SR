from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from tools.template_fill_tool import TemplateFillTool

import vl_loopsr_core as core


class SearchDataset:
    """Dataset whose held-out split fails loudly if search touches it."""

    def __init__(self):
        self.feature_names = ["x"]
        self.target_name = "y"
        self.train_df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [2.0, 4.0, 6.0]})
        self.val_df = pd.DataFrame({"x": [4.0, 5.0], "y": [8.0, 10.0]})
        self.df = pd.concat([self.train_df, self.val_df], ignore_index=True)

    @property
    def test_df(self):
        raise AssertionError("search accessed the sealed test split")


def test_template_fitter_never_reads_test_split():
    dataset = SearchDataset()
    results = TemplateFillTool(max_workers=1).run(
        ["2*x", "a*x"],
        dataset,
        n_restarts=1,
    )

    assert len(results) == 2
    assert all(item.success for item in results)
    assert all(item.test_mse is None for item in results)
    assert all(item.val_mse is not None for item in results)


def test_direct_evidence_promotion_never_reads_test_split():
    results = core.build_direct_verified_evidence_results(
        ["2*x"],
        SearchDataset(),
        feature_names=["x"],
    )

    assert len(results) == 1
    assert results[0].val_mse == 0.0
    assert results[0].test_mse is None


def test_evaluator_search_never_reads_test_split():
    dataset = SearchDataset()
    evaluator = core.EvaluatorAgent(complexity_weight=0.0)
    evaluator.fitter = TemplateFillTool(max_workers=1)

    evaluation = evaluator.evaluate(["2*x", "a*x"], dataset)
    results = evaluation["scored_results"]

    assert results
    assert all(item.test_mse is None for item in results)


def test_explicit_split_search_frame_excludes_test_rows(tmp_path):
    train = pd.DataFrame({"x": [1.0], "y": [2.0]})
    val = pd.DataFrame({"x": [2.0], "y": [4.0]})
    test = pd.DataFrame({"x": [999.0], "y": [1998.0]})

    dataset = core.build_dataset_from_explicit_splits(
        train,
        val,
        test,
        tmp_path,
    )

    assert dataset.test_df.equals(test)
    assert dataset.df["x"].tolist() == [1.0, 2.0]


def test_test_metric_is_computed_only_for_selected_expression():
    class FinalDataset:
        feature_names = ["x"]
        target_name = "y"
        test_df = pd.DataFrame({"x": [6.0, 7.0], "y": [12.0, 14.0]})

    selected = SimpleNamespace(simplified_expression="2*x", test_mse=None)
    report = core.evaluate_selected_expression_on_test(selected, FinalDataset())

    assert report["phase"] == "post_selection"
    assert report["selected_expression_only"] is True
    assert report["prediction_valid"] is True
    assert report["test_mse"] == 0.0
    assert selected.test_mse == 0.0


def test_invalid_held_out_prediction_is_reported_without_reselection():
    class FinalDataset:
        feature_names = ["x"]
        target_name = "y"
        test_df = pd.DataFrame({"x": [-1.0], "y": [0.0]})

    selected = SimpleNamespace(simplified_expression="log(x)", test_mse=123.0)
    with np.errstate(all="ignore"):
        report = core.evaluate_selected_expression_on_test(selected, FinalDataset())

    assert report["prediction_valid"] is False
    assert report["error"] == "non_finite_test_prediction"
    assert selected.simplified_expression == "log(x)"
    assert selected.test_mse is None
