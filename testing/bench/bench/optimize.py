"""Optuna TPE parameter search against a labeled scores DataFrame."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import optuna
import pandas as pd
from .ensemble import EnsembleParams, stars_from_subscores
from .metrics import compute_metrics, MetricsResult


@dataclass
class SearchSpace:
    w_tech_range:     tuple[float, float] = (0.0, 1.0)
    w_aes_range:      tuple[float, float] = (0.0, 1.0)
    w_clip_range:     tuple[float, float] = (0.0, 1.0)
    strictness_range: tuple[float, float] = (0.0, 1.0)
    search_bucket_edges: bool = True


@dataclass
class OptimizationResult:
    params: EnsembleParams
    metrics: MetricsResult
    study: optuna.Study


def _objective_factory(
    tech: np.ndarray, aes: np.ndarray, clip: np.ndarray, gt: np.ndarray, space: SearchSpace
):
    def objective(trial: optuna.Trial) -> float:
        wT = trial.suggest_float("w_tech", *space.w_tech_range)
        wA = trial.suggest_float("w_aes",  *space.w_aes_range)
        wC = trial.suggest_float("w_clip", *space.w_clip_range)
        s = trial.suggest_float("strictness", *space.strictness_range)
        if space.search_bucket_edges:
            e1 = trial.suggest_float("e1", 0.05, 0.35)
            e2 = trial.suggest_float("e2", e1 + 0.05, 0.55)
            e3 = trial.suggest_float("e3", e2 + 0.05, 0.75)
            e4 = trial.suggest_float("e4", e3 + 0.05, 0.95)
        else:
            e1, e2, e3, e4 = 0.2, 0.4, 0.6, 0.8
        if wT + wA + wC < 1e-6:
            return 5.0  # degenerate — penalise
        params = EnsembleParams(
            w_tech=wT, w_aes=wA, w_clip=wC, strictness=s,
            bucket_edges=(e1, e2, e3, e4),
        )
        pred = stars_from_subscores(tech, aes, clip, params)
        m = compute_metrics(gt, pred)
        return -m.spearman + 0.2 * m.mae
    return objective


def optimize_params(
    scores_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    space: SearchSpace,
    n_trials: int = 500,
    seed: int = 0,
) -> OptimizationResult:
    df = scores_df.merge(labels_df, on="filename", how="inner")
    if "clipIQA" not in df.columns:
        raise ValueError(
            "optimize_params requires a 'clipIQA' scalar column. "
            "Compute it from clipEmbedding first (see run.py)."
        )
    tech = df["topiqTechnical"].to_numpy()
    aes  = df["topiqAesthetic"].to_numpy()
    clip = df["clipIQA"].to_numpy()
    gt   = df["gt_stars"].to_numpy()

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(
        _objective_factory(tech, aes, clip, gt, space),
        n_trials=n_trials,
        show_progress_bar=False,
    )
    p = study.best_params
    edges = (
        p.get("e1", 0.2),
        p.get("e2", 0.4),
        p.get("e3", 0.6),
        p.get("e4", 0.8),
    )
    best = EnsembleParams(
        w_tech=p["w_tech"], w_aes=p["w_aes"], w_clip=p["w_clip"],
        strictness=p["strictness"],
        bucket_edges=edges,
    )
    pred = stars_from_subscores(tech, aes, clip, best)
    metrics = compute_metrics(gt, pred)
    return OptimizationResult(params=best, metrics=metrics, study=study)
