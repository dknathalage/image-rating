"""Smoke test for parity helpers. Full CoreML gate arrives in Task 14."""
from __future__ import annotations
import pandas as pd
from bench.parity import compare


def test_compare_returns_expected_keys():
    py = pd.DataFrame({"filename": ["a.jpg", "b.jpg"], "pyiqa_score": [5.2, 6.1]})
    sw = pd.DataFrame({"filename": ["a.jpg", "b.jpg"], "coreml_score": [5.25, 6.05]})
    result = compare(py, sw)
    assert set(result.keys()) == {"n", "spearman", "max_abs_delta", "mean_abs_delta"}
    assert result["n"] == 2
    assert result["max_abs_delta"] < 0.1
