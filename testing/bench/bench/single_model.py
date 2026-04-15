"""Single-model (MUSIQ-AVA) threshold bucketing + Optuna tuning."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from .metrics import compute_metrics, MetricsResult


def stars_from_thresholds(
    scores: np.ndarray, thresholds: tuple[float, float, float, float]
) -> np.ndarray:
    t1, t2, t3, t4 = thresholds
    stars = np.ones_like(scores, dtype=int)
    stars = stars + (scores > t1).astype(int)
    stars = stars + (scores > t2).astype(int)
    stars = stars + (scores > t3).astype(int)
    stars = stars + (scores > t4).astype(int)
    return np.clip(stars, 1, 5)


def bucket_stars(score: float, thresholds: tuple[float, float, float, float]) -> int:
    t1, t2, t3, t4 = thresholds
    if score <= t1:
        return 1
    if score <= t2:
        return 2
    if score <= t3:
        return 3
    if score <= t4:
        return 4
    return 5


@dataclass
class OptimizeResult:
    thresholds: tuple[float, float, float, float]
    metrics: MetricsResult


def optimize_thresholds(
    scores_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    *,
    n_trials: int = 500,
    seed: int = 0,
) -> OptimizeResult:
    import optuna

    df = scores_df.merge(labels_df, on="filename", how="inner")
    s = df["musiqAesthetic"].to_numpy()
    y = df["gt_stars"].to_numpy().astype(int)
    lo, hi = float(s.min()), float(s.max())

    def objective(trial):
        t1 = trial.suggest_float("t1", lo, hi)
        t2 = trial.suggest_float("t2", t1 + 1e-3, hi)
        t3 = trial.suggest_float("t3", t2 + 1e-3, hi)
        t4 = trial.suggest_float("t4", t3 + 1e-3, hi)
        pred = stars_from_thresholds(s, (t1, t2, t3, t4))
        m = compute_metrics(y, pred)
        return -m.spearman + 0.2 * m.mae

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed)
    )
    study.optimize(objective, n_trials=n_trials)
    t = (
        study.best_params["t1"],
        study.best_params["t2"],
        study.best_params["t3"],
        study.best_params["t4"],
    )
    pred = stars_from_thresholds(s, t)
    m = compute_metrics(y, pred)
    return OptimizeResult(thresholds=t, metrics=m)
