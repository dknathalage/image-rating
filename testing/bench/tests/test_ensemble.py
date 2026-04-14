import numpy as np
import pytest
from bench.ensemble import minmax_normalize, combined_quality, percentile_stars, stars_from_subscores, EnsembleParams


def test_minmax_normalize_basic():
    arr = np.array([0.0, 0.5, 1.0])
    out = minmax_normalize(arr)
    np.testing.assert_allclose(out, [0.0, 0.5, 1.0])


def test_minmax_normalize_constant_returns_half():
    arr = np.array([0.3, 0.3, 0.3])
    out = minmax_normalize(arr)
    np.testing.assert_allclose(out, [0.5, 0.5, 0.5])


def test_minmax_normalize_shifted():
    arr = np.array([10.0, 20.0, 30.0])
    out = minmax_normalize(arr)
    np.testing.assert_allclose(out, [0.0, 0.5, 1.0])


def test_combined_quality_default_weights():
    tech = np.array([0.5])
    aes  = np.array([0.8])
    clip = np.array([0.3])
    out  = combined_quality(tech, aes, clip, (0.4, 0.4, 0.2))
    np.testing.assert_allclose(out, [0.4*0.5 + 0.4*0.8 + 0.2*0.3])


def test_combined_quality_weights_normalized():
    tech = np.array([1.0])
    aes  = np.array([0.0])
    clip = np.array([0.0])
    out  = combined_quality(tech, aes, clip, (0.6, 0.6, 0.3))
    # renorm: wT = 0.6/1.5 = 0.4
    np.testing.assert_allclose(out, [0.4])


def test_combined_quality_negative_weight_raises():
    tech = np.array([0.5])
    aes  = np.array([0.5])
    clip = np.array([0.5])
    with pytest.raises(ValueError, match="non-negative"):
        combined_quality(tech, aes, clip, (-0.1, 0.5, 0.6))


def test_percentile_stars_uniform_strictness_equal_buckets():
    scores = np.linspace(0, 1, 10)
    stars = percentile_stars(scores, strictness=0.5)
    assert stars.tolist() == [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]


def test_percentile_stars_preserves_order():
    scores = np.array([0.9, 0.1, 0.5, 0.7, 0.3])
    stars = percentile_stars(scores, strictness=0.5)
    assert stars[1] == 1  # 0.1
    assert stars[4] == 2  # 0.3
    assert stars[2] == 3  # 0.5
    assert stars[3] == 4  # 0.7
    assert stars[0] == 5  # 0.9


def test_percentile_stars_strictness_skews():
    scores = np.linspace(0, 1, 100)
    lenient = percentile_stars(scores, strictness=0.0)
    strict  = percentile_stars(scores, strictness=1.0)
    assert (lenient >= 3).sum() > (strict >= 3).sum()


def test_percentile_stars_single_element():
    scores = np.array([0.7])
    stars = percentile_stars(scores, strictness=0.0)
    assert stars.tolist() == [1]
    stars = percentile_stars(scores, strictness=0.5)
    assert stars.tolist() == [1]
    stars = percentile_stars(scores, strictness=1.0)
    assert stars.tolist() == [1]


def test_percentile_stars_empty():
    scores = np.array([])
    stars = percentile_stars(scores, strictness=0.5)
    assert len(stars) == 0
    assert stars.dtype == int


def test_percentile_stars_vectorised_matches_scalar():
    """Vectorised output must match scalar reference on random input."""
    rng = np.random.default_rng(42)
    scores = rng.random(100)
    strictness = 0.7

    # scalar reference (original loop logic)
    n = len(scores)
    s = max(0.0, min(1.0, float(strictness)))
    gamma = 10.0 ** (2 * s - 1)
    order = np.argsort(scores, kind="stable")
    ref = np.empty(n, dtype=int)
    e1, e2, e3, e4 = (0.20, 0.40, 0.60, 0.80)
    for rank, original_idx in enumerate(order):
        pct = rank / n
        warped = (1e-9 if pct == 0 else pct) ** gamma
        if   warped < e1: ref[original_idx] = 1
        elif warped < e2: ref[original_idx] = 2
        elif warped < e3: ref[original_idx] = 3
        elif warped < e4: ref[original_idx] = 4
        else:             ref[original_idx] = 5

    result = percentile_stars(scores, strictness=strictness)
    np.testing.assert_array_equal(result, ref)


def test_stars_from_subscores_pipeline():
    tech = np.array([0.1, 0.5, 0.9, 0.3, 0.7])
    aes  = np.array([0.2, 0.6, 0.8, 0.4, 0.5])
    clip = np.array([0.5, 0.5, 0.5, 0.5, 0.5])  # constant → 0.5 after norm
    params = EnsembleParams(
        w_tech=0.4, w_aes=0.4, w_clip=0.2,
        strictness=0.5,
        bucket_edges=(0.2, 0.4, 0.6, 0.8),
    )
    stars = stars_from_subscores(tech, aes, clip, params)
    assert stars.min() >= 1 and stars.max() <= 5
    assert len(stars) == 5
    assert stars[2] == 5
    assert stars[0] == 1
