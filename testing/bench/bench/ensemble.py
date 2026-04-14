"""Python mirror of Swift ensemble logic (ProcessingQueue.normalizeAndWriteStars + RatingPipeline.combinedQuality).

Every function here must match Swift behaviour to within 1e-6. Changes to Swift
ensemble logic require parallel changes here — tests catch drift.
"""
from dataclasses import dataclass
import numpy as np


def minmax_normalize(arr: np.ndarray) -> np.ndarray:
    """Mirror Swift `norm(v, lo, hi)`: (v - lo) / (hi - lo), or 0.5 if constant."""
    lo, hi = float(arr.min()), float(arr.max())
    if hi <= lo:
        return np.full_like(arr, 0.5, dtype=float)
    return (arr - lo) / (hi - lo)


def combined_quality(
    tech: np.ndarray,
    aes: np.ndarray,
    clip: np.ndarray,
    weights: tuple[float, float, float],
) -> np.ndarray:
    """Weighted sum of three sub-scores. Weights renormalised to sum to 1.
    Mirrors Swift `combinedQuality` + `ProcessingQueue` weight-sum rescaling.
    """
    w_tech, w_aes, w_clip = weights
    s = w_tech + w_aes + w_clip
    if s <= 0:
        raise ValueError("weights sum to zero")
    if any(w < 0 for w in (w_tech, w_aes, w_clip)):
        raise ValueError("weights must be non-negative")
    w_tech, w_aes, w_clip = w_tech / s, w_aes / s, w_clip / s
    return w_tech * tech + w_aes * aes + w_clip * clip


DEFAULT_BUCKET_EDGES = (0.20, 0.40, 0.60, 0.80)


@dataclass(frozen=True)
class EnsembleParams:
    w_tech: float
    w_aes: float
    w_clip: float
    strictness: float
    bucket_edges: tuple[float, float, float, float] = DEFAULT_BUCKET_EDGES


def percentile_stars(
    scores: np.ndarray,
    strictness: float,
    bucket_edges: tuple[float, float, float, float] = DEFAULT_BUCKET_EDGES,
) -> np.ndarray:
    """Assign 1-5 stars by percentile rank with γ-curve strictness.

    Mirrors Swift `ProcessingQueue.percentileStars`.
    γ = 10^(2s-1): s=0→γ=0.1 (lenient), s=0.5→γ=1 (uniform), s=1→γ=10 (strict).
    """
    n = len(scores)
    if n == 0:
        return np.array([], dtype=int)
    s = max(0.0, min(1.0, float(strictness)))
    gamma = 10.0 ** (2 * s - 1)
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(n, dtype=int)
    ranks[order] = np.arange(n)
    pct = ranks / n
    # γ-curve: clamp pct==0 to 1e-9 to avoid 0^negative = inf
    pct_safe = np.where(pct == 0, 1e-9, pct)
    warped = pct_safe ** gamma
    e1, e2, e3, e4 = bucket_edges
    result = np.ones(n, dtype=int)
    result[warped >= e1] = 2
    result[warped >= e2] = 3
    result[warped >= e3] = 4
    result[warped >= e4] = 5
    return result


def stars_from_subscores(
    tech: np.ndarray,
    aes: np.ndarray,
    clip: np.ndarray,
    params: EnsembleParams,
) -> np.ndarray:
    """Full Swift pipeline: per-dataset min-max → weighted combine → percentile stars."""
    tn = minmax_normalize(tech)
    an = minmax_normalize(aes)
    cn = minmax_normalize(clip)
    combined = combined_quality(tn, an, cn, (params.w_tech, params.w_aes, params.w_clip))
    return percentile_stars(combined, params.strictness, params.bucket_edges)
