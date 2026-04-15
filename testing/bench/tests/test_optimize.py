import numpy as np
import pandas as pd
from bench.optimize import optimize_params, SearchSpace


def test_optimize_improves_over_default():
    rng = np.random.default_rng(0)
    n = 300
    tech = rng.random(n)
    aes  = rng.random(n)
    clip = rng.random(n)
    # Oracle: tech dominates
    true_combined = 0.7 * tech + 0.1 * aes + 0.2 * clip
    ranks = pd.Series(true_combined).rank(method="first") - 1
    gt_stars = np.clip((ranks / n // 0.2).astype(int), 0, 4).values + 1

    scores_df = pd.DataFrame({
        "filename": [f"{i}.jpg" for i in range(n)],
        "topiqTechnical": tech,
        "topiqAesthetic": aes,
        "clipEmbedding": [list(rng.random(4)) for _ in range(n)],
    })
    scores_df["clipIQA"] = clip
    labels_df = pd.DataFrame({"filename": scores_df.filename, "gt_stars": gt_stars})

    space = SearchSpace()
    best = optimize_params(scores_df, labels_df, space, n_trials=40, seed=0)
    assert best.params.w_tech > best.params.w_aes
    assert best.metrics.spearman > 0.5


def test_optimize_requires_clipIQA_column():
    import pytest
    df = pd.DataFrame({
        "filename": ["a.jpg"],
        "topiqTechnical": [0.5],
        "topiqAesthetic": [0.5],
        "clipEmbedding": [[0.1, 0.2]],
    })
    labels = pd.DataFrame({"filename": ["a.jpg"], "gt_stars": [3]})
    with pytest.raises(ValueError, match="clipIQA"):
        optimize_params(df, labels, SearchSpace(), n_trials=2, seed=0)


def _make_oracle_data(n=200, seed=0):
    rng = np.random.default_rng(seed)
    tech = rng.random(n)
    aes  = rng.random(n)
    clip = rng.random(n)
    true_combined = 0.7 * tech + 0.1 * aes + 0.2 * clip
    ranks = pd.Series(true_combined).rank(method="first") - 1
    gt_stars = np.clip((ranks / n // 0.2).astype(int), 0, 4).values + 1
    scores_df = pd.DataFrame({
        "filename": [f"{i}.jpg" for i in range(n)],
        "topiqTechnical": tech,
        "topiqAesthetic": aes,
        "clipEmbedding": [list(rng.random(4)) for _ in range(n)],
    })
    scores_df["clipIQA"] = clip
    labels_df = pd.DataFrame({"filename": scores_df.filename, "gt_stars": gt_stars})
    return scores_df, labels_df


def test_optimize_fixed_bucket_edges():
    import pytest
    scores_df, labels_df = _make_oracle_data()
    space = SearchSpace(search_bucket_edges=False)
    best = optimize_params(scores_df, labels_df, space, n_trials=20, seed=1)
    assert best.params.bucket_edges == (0.2, 0.4, 0.6, 0.8)
    assert best.metrics.spearman > 0.3


def test_optimize_reproducible():
    import pytest
    scores_df, labels_df = _make_oracle_data()
    space = SearchSpace()
    r1 = optimize_params(scores_df, labels_df, space, n_trials=10, seed=42)
    r2 = optimize_params(scores_df, labels_df, space, n_trials=10, seed=42)
    assert r1.params.w_tech == pytest.approx(r2.params.w_tech)


def test_optimization_result_serialisable():
    rng = np.random.default_rng(0)
    n = 80
    tech = rng.random(n); aes = rng.random(n); clip = rng.random(n)
    true_combined = 0.7*tech + 0.1*aes + 0.2*clip
    ranks = pd.Series(true_combined).rank(method="first") - 1
    gt_stars = np.clip((ranks / n // 0.2).astype(int), 0, 4).values + 1
    scores = pd.DataFrame({
        "filename":[f"{i}.jpg" for i in range(n)],
        "topiqTechnical":tech, "topiqAesthetic":aes,
        "clipEmbedding":[[0.0]]*n, "clipIQA":clip,
    })
    labels = pd.DataFrame({"filename":scores.filename, "gt_stars":gt_stars})
    from bench.optimize import optimize_params, SearchSpace, result_to_params_dict, result_to_metrics_dict
    r = optimize_params(scores, labels, SearchSpace(), n_trials=10, seed=0)
    params_blob = result_to_params_dict(r, ["tech","aes","clip"], "test", "2026-04-15")
    metrics_blob = result_to_metrics_dict(r)
    import json
    json.dumps(params_blob)  # must not raise
    json.dumps(metrics_blob)
    assert "spearman" in metrics_blob["val"]
    assert params_blob["params"]["w_tech"] >= 0
