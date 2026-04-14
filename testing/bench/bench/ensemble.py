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
    wT, wA, wC = weights
    s = wT + wA + wC
    if s <= 0:
        raise ValueError("weights sum to zero")
    wT, wA, wC = wT / s, wA / s, wC / s
    return wT * tech + wA * aes + wC * clip


DEFAULT_BUCKET_EDGES = (0.20, 0.40, 0.60, 0.80)


@dataclass(frozen=True)
class EnsembleParams:
    wTech: float
    wAes: float
    wClip: float
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
    result = np.empty(n, dtype=int)
    e1, e2, e3, e4 = bucket_edges
    for rank, original_idx in enumerate(order):
        pct = rank / n
        warped = (1e-9 if pct == 0 else pct) ** gamma
        if   warped < e1: result[original_idx] = 1
        elif warped < e2: result[original_idx] = 2
        elif warped < e3: result[original_idx] = 3
        elif warped < e4: result[original_idx] = 4
        else:             result[original_idx] = 5
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
    combined = combined_quality(tn, an, cn, (params.wTech, params.wAes, params.wClip))
    return percentile_stars(combined, params.strictness, params.bucket_edges)
