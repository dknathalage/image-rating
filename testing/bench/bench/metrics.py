"""Rating-quality metrics for bench evaluation."""
import warnings
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
    is_degenerate: bool = False

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
            "is_degenerate": self.is_degenerate,
        }


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, worst_k: int = 10) -> MetricsResult:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")

    yt_int = y_true.astype(int)
    yp_int = y_pred.astype(int)

    if not (np.all(yt_int >= 1) and np.all(yt_int <= 5)):
        bad = yt_int[(yt_int < 1) | (yt_int > 5)]
        raise ValueError(f"stars must be in 1..5, got {bad.tolist()} in y_true")
    if not (np.all(yp_int >= 1) and np.all(yp_int <= 5)):
        bad = yp_int[(yp_int < 1) | (yp_int > 5)]
        raise ValueError(f"stars must be in 1..5, got {bad.tolist()} in y_pred")

    residuals = np.abs(yp_int - yt_int)
    rho, _ = spearmanr(y_true, y_pred)
    tau, _ = kendalltau(y_true, y_pred)

    is_degenerate = bool(np.isnan(rho) or np.isnan(tau))
    if is_degenerate:
        warnings.warn("rank correlation undefined (constant input)", RuntimeWarning)
    rho = 0.0 if np.isnan(rho) else float(rho)
    tau = 0.0 if np.isnan(tau) else float(tau)

    confusion = np.zeros((5, 5), dtype=int)
    np.add.at(confusion, (yt_int - 1, yp_int - 1), 1)

    k = min(worst_k, len(residuals))
    worst_idx = np.argsort(-residuals, kind='stable')[:k]

    return MetricsResult(
        spearman=rho,
        kendall=tau,
        mae=float(residuals.mean()),
        exact_match=float((residuals == 0).mean()),
        off_by_one=float((residuals <= 1).mean()),
        confusion=confusion,
        worst_indices=worst_idx,
        worst_residuals=residuals[worst_idx],
        is_degenerate=is_degenerate,
    )
