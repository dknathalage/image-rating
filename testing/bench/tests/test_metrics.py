import numpy as np
import pytest
import warnings
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

def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        compute_metrics(np.array([1, 2]), np.array([1, 2, 3]))

def test_out_of_range_raises():
    with pytest.raises(ValueError, match="stars must be in 1..5"):
        compute_metrics(np.array([0, 6]), np.array([1, 2]))

def test_degenerate_flag():
    y = np.array([3, 3, 3, 3])
    with pytest.warns(RuntimeWarning):
        m = compute_metrics(y, y)
    assert m.is_degenerate is True
    assert m.spearman == pytest.approx(0.0)

def test_worst_k_exceeds_length():
    y = np.array([1, 2, 3, 4, 5])
    pred = np.array([1, 2, 3, 4, 1])
    m = compute_metrics(y, pred, worst_k=100)
    assert len(m.worst_indices) == 5
