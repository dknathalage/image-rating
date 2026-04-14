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
