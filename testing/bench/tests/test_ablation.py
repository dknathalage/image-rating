import numpy as np
import pandas as pd
from bench.ablation import run_ablation, ABLATION_CONFIGS

def test_ablation_runs_all_configs():
    rng = np.random.default_rng(0)
    n = 100
    df = pd.DataFrame({
        "filename": [f"{i}.jpg" for i in range(n)],
        "topiqTechnical": rng.random(n),
        "topiqAesthetic": rng.random(n),
        "clipIQA": rng.random(n),
        "clipEmbedding": [list(rng.random(4)) for _ in range(n)],
        "gt_stars": rng.integers(1, 6, n),
    })
    scores_df = df[["filename","topiqTechnical","topiqAesthetic","clipIQA","clipEmbedding"]]
    labels_df = df[["filename","gt_stars"]]
    results = run_ablation(scores_df, labels_df, n_trials=20, seed=0)
    assert set(results.keys()) == set(ABLATION_CONFIGS.keys())
    for name, r in results.items():
        assert hasattr(r.metrics, "spearman")
