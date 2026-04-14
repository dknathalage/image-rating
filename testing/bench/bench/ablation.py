"""Ensemble membership ablation. Runs the optimizer with each model subset."""
from __future__ import annotations
import pandas as pd
from .optimize import optimize_params, SearchSpace, OptimizationResult


# True = model is IN the ensemble; False = weight forced to zero.
ABLATION_CONFIGS: dict[str, dict[str, bool]] = {
    "baseline": {"tech": True,  "aes": True,  "clip": True},
    "no-clip":  {"tech": True,  "aes": True,  "clip": False},
    "no-aes":   {"tech": True,  "aes": False, "clip": True},
    "no-tech":  {"tech": False, "aes": True,  "clip": True},
    "tech":     {"tech": True,  "aes": False, "clip": False},
    "aes":      {"tech": False, "aes": True,  "clip": False},
    "clip":     {"tech": False, "aes": False, "clip": True},
}


def _space_for(config: dict[str, bool]) -> SearchSpace:
    return SearchSpace(
        w_tech_range=(0.0, 1.0) if config["tech"] else (0.0, 0.0),
        w_aes_range =(0.0, 1.0) if config["aes"]  else (0.0, 0.0),
        w_clip_range=(0.0, 1.0) if config["clip"] else (0.0, 0.0),
    )


def run_ablation(
    scores_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    n_trials: int = 100,
    seed: int = 0,
) -> dict[str, OptimizationResult]:
    results: dict[str, OptimizationResult] = {}
    for name, cfg in ABLATION_CONFIGS.items():
        space = _space_for(cfg)
        results[name] = optimize_params(scores_df, labels_df, space, n_trials=n_trials, seed=seed)
    return results
