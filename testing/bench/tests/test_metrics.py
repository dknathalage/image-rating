import numpy as np
import pytest
from bench.metrics import compute_metrics, MetricsResult

def test_perfect_prediction():
    y = np.array([1, 2, 3, 4, 5])
    m = compute_metrics(y, y)
    assert m.spearman == pytest.approx(1.0)
    assert m.kendall  == pytest.approx(1.0)
    assert m.mae      == pytest.approx(0.0)
    assert m.exact_match == pytest.approx(1.0)
    assert m.off_by_one  == pytest.approx(1.0)

def test_off_by_one_shift():
    y    = np.array([1, 2, 3, 4, 5])
    pred = np.array([2, 3, 4, 5, 5])
    m = compute_metrics(y, pred)
    assert m.mae == pytest.approx(0.8)
    assert m.off_by_one == pytest.approx(1.0)
    assert m.exact_match == pytest.approx(0.2)

def test_worst_residuals():
    y    = np.array([1, 2, 3, 4, 5])
    pred = np.array([1, 2, 3, 4, 1])
    m = compute_metrics(y, pred, worst_k=1)
    assert m.worst_indices.tolist() == [4]
    assert m.worst_residuals.tolist() == [4]

def test_confusion_shape():
    y    = np.array([1, 2, 3, 4, 5, 1, 2])
    pred = np.array([2, 2, 3, 4, 5, 1, 2])
    m = compute_metrics(y, pred)
    assert m.confusion.shape == (5, 5)
    assert m.confusion.sum() == len(y)
