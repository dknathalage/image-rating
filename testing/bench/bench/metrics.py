"""Rating-quality metrics for bench evaluation."""
from dataclasses import dataclass
import numpy as np
from scipy.stats import spearmanr, kendalltau


@dataclass
class MetricsResult:
    spearman: float
    kendall: float
    mae: float
    exact_match: float
    off_by_one: float
    confusion: np.ndarray
    worst_indices: np.ndarray
    worst_residuals: np.ndarray

    def to_dict(self) -> dict:
        return {
            "spearman": float(self.spearman),
            "kendall":  float(self.kendall),
            "mae":      float(self.mae),
            "exact_match": float(self.exact_match),
            "off_by_one":  float(self.off_by_one),
            "confusion": self.confusion.tolist(),
            "worst_indices":   self.worst_indices.tolist(),
            "worst_residuals": self.worst_residuals.tolist(),
        }


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, worst_k: int = 10) -> MetricsResult:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")

    residuals = np.abs(y_pred.astype(int) - y_true.astype(int))
    rho, _ = spearmanr(y_true, y_pred)
    tau, _ = kendalltau(y_true, y_pred)
    rho = 0.0 if np.isnan(rho) else float(rho)
    tau = 0.0 if np.isnan(tau) else float(tau)

    confusion = np.zeros((5, 5), dtype=int)
    for t, p in zip(y_true.astype(int), y_pred.astype(int)):
        if 1 <= t <= 5 and 1 <= p <= 5:
            confusion[t - 1, p - 1] += 1

    k = min(worst_k, len(residuals))
    worst_idx = np.argsort(-residuals)[:k]

    return MetricsResult(
        spearman=rho,
        kendall=tau,
        mae=float(residuals.mean()),
        exact_match=float((residuals == 0).mean()),
        off_by_one=float((residuals <= 1).mean()),
        confusion=confusion,
        worst_indices=worst_idx,
        worst_residuals=residuals[worst_idx],
    )
