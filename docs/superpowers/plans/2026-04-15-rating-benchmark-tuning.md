# Rating Benchmark & Parameter Tuning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Tune Focal's rating ensemble against the AVA ground-truth dataset and stand up a versioned benchmark suite that tracks ensemble quality over time.

**Architecture:** Swift `FocalScorer` CLI runs the deployed `.mlmodelc` models and emits per-image sub-scores as JSON. Python harness in `testing/bench/` loads AVA labels, joins with scores, runs Optuna parameter search, evaluates metrics, and regenerates a committed `LEADERBOARD.md`. Tuned parameters flow back to the app via `testing/bench/params.current.json` + a build-time Swift code generator.

**Tech Stack:** Swift 5.9 (CLI target), Python 3.11+, CoreML (via existing `.mlmodelc`), Optuna (TPE sampler), pandas + scipy + scikit-learn (metrics), pytest (Python tests), XCTest (Swift tests), xcodegen.

**Spec:** `docs/superpowers/specs/2026-04-15-rating-benchmark-tuning-design.md`

---

## File structure

### Created
- `testing/bench/requirements.txt`
- `testing/bench/pytest.ini`
- `testing/bench/dataset_ava.py` — AVA download + parse
- `testing/bench/ensemble.py` — Swift combining logic mirrored in Python
- `testing/bench/metrics.py` — Spearman, Kendall, MAE, confusion
- `testing/bench/score.py` — invokes `FocalScorer` CLI, caches output
- `testing/bench/optimize.py` — Optuna param search
- `testing/bench/ablation.py` — ensemble subset sweep
- `testing/bench/leaderboard.py` — regenerate `LEADERBOARD.md`
- `testing/bench/run.py` — single-command orchestration (`score`/`optimize`/`eval`/`ablate`/`leaderboard`)
- `testing/bench/params.current.json` — shipping params source of truth
- `testing/bench/LEADERBOARD.md` — auto-generated
- `testing/bench/tests/test_ensemble.py`
- `testing/bench/tests/test_metrics.py`
- `testing/bench/tests/test_dataset_ava.py`
- `testing/bench/tests/test_score.py`
- `testing/bench/tests/test_optimize.py`
- `testing/bench/tests/test_leaderboard.py`
- `testing/bench/tests/fixtures/mini_ava.txt` — 50-row AVA subset for tests
- `testing/bench/.gitignore`
- `FocalScorer/main.swift` — CLI entry point
- `FocalScorer/Scorer.swift` — scoring loop
- `ImageRaterTests/FocalScorerSmokeTests.swift` — Swift-side smoke test against `testing/*.JPG`
- `ImageRater/App/FocalSettings+Generated.swift` — build-time generated constants
- `scripts/gen_defaults.py` — reads `params.current.json`, writes generated Swift

### Modified
- `ImageRater/App/FocalSettings.swift` — add `bucketEdges`, `clipLogitScale`
- `ImageRater/Pipeline/RatingPipeline.swift:85` — accept `logitScale` from `FocalSettings`
- `ImageRater/Pipeline/ProcessingQueue.swift:269-291` — use settings-driven `bucketEdges`
- `project.yml` — add `FocalScorer` target, pre-build script phase invoking `gen_defaults.py`
- `.gitignore` — add `testing/bench/data/`, `testing/bench/results/`, `testing/bench/.cache/`, `__pycache__/`, `.venv/`

### Relevant existing code
- `ImageRater/Pipeline/RatingPipeline.swift` — `loadBundledModels`, `rate`, `clipIQAScore`, `combinedQuality`
- `ImageRater/Pipeline/ProcessingQueue.swift:133-291` — `normalizeAndWriteStars`, `percentileStars`
- `ImageRater/LibRaw/LibRawWrapper.swift:15` — `LibRawWrapper.decode(url:)`
- `ImageRater/Pipeline/CLIPTextEmbeddings.swift` — prompt embeddings (keep current; prompt-set swap is optimizer-level)

---

## Notes for implementers

- **Python tooling:** all Python commands run from `testing/bench/`. Use a venv: `python3 -m venv .venv && source .venv/bin/activate`. Install with `pip install -r requirements.txt`.
- **Swift CLI:** xcodegen generates the project from `project.yml`. After editing `project.yml`, run `xcodegen generate` before opening Xcode. CLI build: `xcodebuild -scheme FocalScorer -configuration Release -destination 'platform=macOS,arch=arm64' build`.
- **AVA access:** photos are hosted at dpchallenge.com per-ID URLs. Use `https://images.dpchallenge.com/images_challenge/0-999/<id>.jpg` pattern (implementation details in `dataset_ava.py`). Be polite: rate-limit, retry with backoff, cache. Full download takes hours; dev uses `--sample 500`.
- **Commits:** small and frequent — one feature per commit. Use conventional commits (`feat:`, `test:`, `chore:`, `fix:`, `docs:`). Tests go in the same commit as the code they test.
- **TDD discipline:** write failing test, confirm failure, write minimal code, confirm pass, commit. Per @superpowers:test-driven-development.
- **Version format:** `v<major>.<minor>.<patch>@<sha7>`. Git SHA is captured by `run.py` (short `git rev-parse --short HEAD`). Minor bumps on ensemble membership change, patch bumps on param retune only. Bump is manual (edit `params.current.json`'s `version` field), committed alongside params.

---

## Task 1: Python scaffolding + test harness

**Files:**
- Create: `testing/bench/requirements.txt`
- Create: `testing/bench/pytest.ini`
- Create: `testing/bench/.gitignore`
- Create: `testing/bench/tests/__init__.py`
- Modify: `.gitignore`

- [ ] **Step 1: Write `testing/bench/requirements.txt`**

```
numpy>=1.26
pandas>=2.2
scipy>=1.12
scikit-learn>=1.4
optuna>=3.6
requests>=2.31
tqdm>=4.66
matplotlib>=3.8
pytest>=8.0
pytest-mock>=3.12
```

- [ ] **Step 2: Write `testing/bench/pytest.ini`**

```ini
[pytest]
testpaths = tests
python_files = test_*.py
addopts = -v --tb=short
```

- [ ] **Step 3: Write `testing/bench/.gitignore`**

```
.venv/
__pycache__/
.cache/
data/
results/
*.pyc
```

- [ ] **Step 4: Append to repo `.gitignore`**

Append these lines:
```
testing/bench/.venv/
testing/bench/data/
testing/bench/results/
testing/bench/.cache/
testing/bench/__pycache__/
```

- [ ] **Step 5: Create empty `tests/__init__.py`**

- [ ] **Step 6: Create venv, install, verify pytest runs**

```bash
cd testing/bench
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest --collect-only
```

Expected: `no tests ran in 0.xxs` (no tests yet, but no import errors).

- [ ] **Step 7: Commit**

```bash
git add testing/bench/requirements.txt testing/bench/pytest.ini testing/bench/.gitignore testing/bench/tests/__init__.py .gitignore
git commit -m "chore: scaffold testing/bench Python project"
```

---

## Task 2: `ensemble.py` — port Swift combining logic

The app's star assignment happens in two places:
- `RatingPipeline.combinedQuality` at `ImageRater/Pipeline/RatingPipeline.swift:72` — weighted sum of 3 sub-scores.
- `ProcessingQueue.normalizeAndWriteStars` + `percentileStars` at `ImageRater/Pipeline/ProcessingQueue.swift:133-291` — min-max normalise each sub-score across session, combine, then percentile-rank into 1–5 stars with γ-curve strictness.

Python must reproduce this exactly so tuning reflects shipped behaviour.

**Files:**
- Create: `testing/bench/ensemble.py`
- Create: `testing/bench/tests/test_ensemble.py`

- [ ] **Step 1: Write test for min-max norm**

`testing/bench/tests/test_ensemble.py`:
```python
import numpy as np
import pytest
from bench.ensemble import minmax_normalize

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
```

- [ ] **Step 2: Add `conftest.py` to expose `bench` package**

`testing/bench/conftest.py`:
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
```

Also create `testing/bench/bench/__init__.py` (empty).

- [ ] **Step 3: Run test; expect ImportError**

```bash
cd testing/bench && pytest tests/test_ensemble.py::test_minmax_normalize_basic -v
```

Expected: `ImportError: cannot import name 'minmax_normalize' from 'bench.ensemble'`.

- [ ] **Step 4: Create `bench/ensemble.py` with `minmax_normalize`**

```python
"""Python mirror of Swift ensemble logic (ProcessingQueue.normalizeAndWriteStars + RatingPipeline.combinedQuality).

Every function here must match Swift behaviour to within 1e-6. Changes to Swift
ensemble logic require parallel changes here — tests catch drift.
"""
import numpy as np


def minmax_normalize(arr: np.ndarray) -> np.ndarray:
    """Mirror Swift `norm(v, lo, hi)`: (v - lo) / (hi - lo), or 0.5 if constant."""
    lo, hi = float(arr.min()), float(arr.max())
    if hi <= lo:
        return np.full_like(arr, 0.5, dtype=float)
    return (arr - lo) / (hi - lo)
```

- [ ] **Step 5: Run tests — pass**

```bash
pytest tests/test_ensemble.py -v
```

Expected: 3 passed.

- [ ] **Step 6: Add tests for `combined_quality`**

Append to `tests/test_ensemble.py`:
```python
from bench.ensemble import combined_quality

def test_combined_quality_default_weights():
    # Matches RatingPipeline.combinedQuality defaults (0.4, 0.4, 0.2)
    tech = np.array([0.5])
    aes  = np.array([0.8])
    clip = np.array([0.3])
    out  = combined_quality(tech, aes, clip, (0.4, 0.4, 0.2))
    np.testing.assert_allclose(out, [0.4*0.5 + 0.4*0.8 + 0.2*0.3])

def test_combined_quality_weights_normalized():
    # Weights not summing to 1 → renormalised internally (mirrors ProcessingQueue line 197-199)
    tech = np.array([1.0])
    aes  = np.array([0.0])
    clip = np.array([0.0])
    out  = combined_quality(tech, aes, clip, (0.6, 0.6, 0.3))
    # renorm: wT = 0.6/1.5 = 0.4
    np.testing.assert_allclose(out, [0.4])
```

- [ ] **Step 7: Run; expect failures**

Expected: 2 failures (`combined_quality` not defined).

- [ ] **Step 8: Implement `combined_quality`**

Append to `bench/ensemble.py`:
```python
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
```

- [ ] **Step 9: Run; all pass**

- [ ] **Step 10: Add tests for `percentile_stars`**

Append:
```python
from bench.ensemble import percentile_stars

def test_percentile_stars_uniform_strictness_equal_buckets():
    # strictness=0.5 → γ=1 (identity warp); 10 values get ~2 in each bucket
    scores = np.linspace(0, 1, 10)
    stars = percentile_stars(scores, strictness=0.5)
    # 10 items, edges at 0.2/0.4/0.6/0.8 → buckets of 2 per star
    assert stars.tolist() == [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]

def test_percentile_stars_preserves_order():
    scores = np.array([0.9, 0.1, 0.5, 0.7, 0.3])
    stars = percentile_stars(scores, strictness=0.5)
    # Argsort positions map to ranks 0..4
    # sorted indices: [1, 4, 2, 3, 0] (values 0.1, 0.3, 0.5, 0.7, 0.9)
    # ranks 0..4 → pct 0, 0.2, 0.4, 0.6, 0.8 → buckets 1,2,3,4,5
    assert stars[1] == 1  # 0.1
    assert stars[4] == 2  # 0.3
    assert stars[2] == 3  # 0.5
    assert stars[3] == 4  # 0.7
    assert stars[0] == 5  # 0.9

def test_percentile_stars_strictness_skews():
    scores = np.linspace(0, 1, 100)
    lenient = percentile_stars(scores, strictness=0.0)
    strict  = percentile_stars(scores, strictness=1.0)
    # Lenient (γ=0.1) pushes things up; strict (γ=10) pushes things down
    assert (lenient >= 3).sum() > (strict >= 3).sum()
```

- [ ] **Step 11: Implement `percentile_stars`**

Append to `bench/ensemble.py` (port from `ProcessingQueue.percentileStars`):
```python
DEFAULT_BUCKET_EDGES = (0.20, 0.40, 0.60, 0.80)


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
```

- [ ] **Step 12: Run all ensemble tests — pass**

- [ ] **Step 13: Add the end-to-end `stars_from_subscores` helper**

Test first:
```python
from bench.ensemble import stars_from_subscores, EnsembleParams

def test_stars_from_subscores_pipeline():
    tech = np.array([0.1, 0.5, 0.9, 0.3, 0.7])
    aes  = np.array([0.2, 0.6, 0.8, 0.4, 0.5])
    clip = np.array([0.5, 0.5, 0.5, 0.5, 0.5])  # constant → norms to 0.5
    params = EnsembleParams(
        wTech=0.4, wAes=0.4, wClip=0.2,
        strictness=0.5,
        bucket_edges=(0.2, 0.4, 0.6, 0.8),
    )
    stars = stars_from_subscores(tech, aes, clip, params)
    assert stars.min() >= 1 and stars.max() <= 5
    assert len(stars) == 5
    # Highest combined score gets highest star
    assert stars[2] == 5
    assert stars[0] == 1
```

- [ ] **Step 14: Implement `stars_from_subscores` + `EnsembleParams`**

Append to `bench/ensemble.py`:
```python
from dataclasses import dataclass


@dataclass(frozen=True)
class EnsembleParams:
    wTech: float
    wAes: float
    wClip: float
    strictness: float
    bucket_edges: tuple[float, float, float, float] = DEFAULT_BUCKET_EDGES


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
```

- [ ] **Step 15: Run tests — all pass**

- [ ] **Step 16: Commit**

```bash
git add testing/bench/bench/ testing/bench/tests/test_ensemble.py testing/bench/conftest.py
git commit -m "feat(bench): port Swift ensemble logic to Python with tests"
```

---

## Task 3: `metrics.py` — Spearman, MAE, confusion

**Files:**
- Create: `testing/bench/bench/metrics.py`
- Create: `testing/bench/tests/test_metrics.py`

- [ ] **Step 1: Write tests**

`testing/bench/tests/test_metrics.py`:
```python
import numpy as np
import pytest
from bench.metrics import compute_metrics, MetricsResult

def test_perfect_prediction():
    y = np.array([1, 2, 3, 4, 5])
    m = compute_metrics(y, y)
    assert m.spearman == pytest.approx(1.0)
    assert m.kendall  == pytest.approx(1.0)
    assert m.mae      == pytest.approx(0.0)
    assert m.exact_match == pytest.approx(1.0)
    assert m.off_by_one  == pytest.approx(1.0)

def test_off_by_one_shift():
    y    = np.array([1, 2, 3, 4, 5])
    pred = np.array([2, 3, 4, 5, 5])
    m = compute_metrics(y, pred)
    assert m.mae == pytest.approx(0.8)        # (1+1+1+1+0)/5
    assert m.off_by_one == pytest.approx(1.0) # all within ±1
    assert m.exact_match == pytest.approx(0.2)  # only last matches

def test_worst_residuals():
    y    = np.array([1, 2, 3, 4, 5])
    pred = np.array([1, 2, 3, 4, 1])  # last prediction very wrong
    m = compute_metrics(y, pred, worst_k=1)
    assert m.worst_indices.tolist() == [4]
    assert m.worst_residuals.tolist() == [4]

def test_confusion_shape():
    y    = np.array([1, 2, 3, 4, 5, 1, 2])
    pred = np.array([2, 2, 3, 4, 5, 1, 2])
    m = compute_metrics(y, pred)
    assert m.confusion.shape == (5, 5)
    assert m.confusion.sum() == len(y)
```

- [ ] **Step 2: Run — expect ImportError**

- [ ] **Step 3: Implement `metrics.py`**

```python
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
    confusion: np.ndarray       # shape (5, 5); rows=true, cols=pred
    worst_indices: np.ndarray   # indices of worst predictions
    worst_residuals: np.ndarray # |pred - true| for worst predictions

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
```

- [ ] **Step 4: Run — pass**

- [ ] **Step 5: Commit**

```bash
git add testing/bench/bench/metrics.py testing/bench/tests/test_metrics.py
git commit -m "feat(bench): rating metrics (Spearman, MAE, confusion, worst-k)"
```

---

## Task 4: `dataset_ava.py` — AVA parser + downloader

**Files:**
- Create: `testing/bench/bench/dataset_ava.py`
- Create: `testing/bench/tests/test_dataset_ava.py`
- Create: `testing/bench/tests/fixtures/mini_ava.txt`

AVA.txt format (one photo per line): `index image_id count_1 count_2 ... count_10 semantic_tag_1 semantic_tag_2 challenge_id`. MOS = `Σ(i · count_i) / Σ count_i`, where `i = 1..10`.

Ground-truth 1–5 stars = quintile rank of MOS across chosen dataset (same percentile-ranking approach Focal uses for its own predictions).

- [ ] **Step 1: Write fixture**

`testing/bench/tests/fixtures/mini_ava.txt` (5 lines representing distinct MOS levels):
```
1 953619 0 1 5 7 10 15 20 25 10 7 1 29 100
2 953620 1 2 3 5 10 20 30 15 8 6 2 30 100
3 953621 0 0 1 2 3 5 10 20 30 29 1 31 100
4 953622 5 10 15 20 15 10 8 7 5 5 1 32 100
5 953623 20 25 20 15 10 5 3 1 1 0 1 33 100
```

- [ ] **Step 2: Write tests**

`testing/bench/tests/test_dataset_ava.py`:
```python
from pathlib import Path
import numpy as np
from bench.dataset_ava import parse_ava_txt, compute_mos, stratified_sample, mos_to_stars

FIXTURE = Path(__file__).parent / "fixtures" / "mini_ava.txt"

def test_parse_ava_txt():
    df = parse_ava_txt(FIXTURE)
    assert len(df) == 5
    assert "image_id" in df.columns
    # 10 vote-count columns
    assert all(f"count_{i}" in df.columns for i in range(1, 11))

def test_compute_mos():
    df = parse_ava_txt(FIXTURE)
    df = compute_mos(df)
    # Image 953619: weighted by [0,1,5,7,10,15,20,25,10,7], sum=100 → known value
    expected = (1*0 + 2*1 + 3*5 + 4*7 + 5*10 + 6*15 + 7*20 + 8*25 + 9*10 + 10*7) / 100
    assert abs(df.loc[df.image_id == 953619, "mos"].iloc[0] - expected) < 1e-6

def test_mos_to_stars_quintile():
    df = parse_ava_txt(FIXTURE)
    df = compute_mos(df)
    df = mos_to_stars(df)
    assert df.gt_stars.min() == 1
    assert df.gt_stars.max() == 5
    # Sanity: highest MOS → 5 stars, lowest → 1 star
    max_row = df.loc[df.mos.idxmax()]
    min_row = df.loc[df.mos.idxmin()]
    assert max_row.gt_stars == 5
    assert min_row.gt_stars == 1

def test_stratified_sample_balanced():
    df = parse_ava_txt(FIXTURE)
    df = compute_mos(df)
    df = mos_to_stars(df)
    sampled = stratified_sample(df, n=5, seed=0)
    assert len(sampled) == 5
    # Stratified: every star present
    assert set(sampled.gt_stars.unique()) == {1, 2, 3, 4, 5}
```

- [ ] **Step 3: Run — expect ImportError**

- [ ] **Step 4: Implement `dataset_ava.py`**

```python
"""AVA dataset loader: parse labels, compute per-image MOS, stratified sampling, downloader."""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import requests
from tqdm import tqdm
import time
import hashlib


AVA_COL_NAMES = (
    ["index", "image_id"]
    + [f"count_{i}" for i in range(1, 11)]
    + ["sem_tag_1", "sem_tag_2", "challenge_id"]
)


def parse_ava_txt(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None, names=AVA_COL_NAMES)
    return df


def compute_mos(df: pd.DataFrame) -> pd.DataFrame:
    count_cols = [f"count_{i}" for i in range(1, 11)]
    counts = df[count_cols].values.astype(float)
    total = counts.sum(axis=1)
    weighted = (counts * np.arange(1, 11)).sum(axis=1)
    df = df.copy()
    df["mos"] = np.where(total > 0, weighted / total, np.nan)
    df = df.dropna(subset=["mos"]).reset_index(drop=True)
    return df


def mos_to_stars(df: pd.DataFrame) -> pd.DataFrame:
    """Quintile rank of MOS → 1..5 stars (matches Focal's own percentile-rank star assignment)."""
    df = df.copy()
    ranks = df["mos"].rank(method="first")
    pct = (ranks - 1) / len(df)
    bins = np.array([0.2, 0.4, 0.6, 0.8])
    df["gt_stars"] = np.searchsorted(bins, pct, side="right") + 1
    df["gt_stars"] = df["gt_stars"].clip(1, 5).astype(int)
    return df


def stratified_sample(df: pd.DataFrame, n: int, seed: int = 0) -> pd.DataFrame:
    """Sample n rows with approximately uniform stars. Requires gt_stars column."""
    per_star = max(1, n // 5)
    rng = np.random.default_rng(seed)
    parts = []
    for star in range(1, 6):
        pool = df[df.gt_stars == star]
        k = min(per_star, len(pool))
        if k == 0:
            continue
        parts.append(pool.sample(n=k, random_state=rng.integers(0, 2**32)))
    sampled = pd.concat(parts).reset_index(drop=True)
    # Top up to exactly n with random extras
    if len(sampled) < n:
        extras = df.drop(sampled.index, errors="ignore").sample(
            n=n - len(sampled), random_state=seed
        )
        sampled = pd.concat([sampled, extras]).reset_index(drop=True)
    return sampled.head(n).reset_index(drop=True)


def _dp_challenge_url(image_id: int) -> str:
    """dpchallenge.com image URL for a given AVA image ID."""
    return f"https://images.dpchallenge.com/images_challenge/0-999/{image_id}.jpg"


def download_images(
    df: pd.DataFrame,
    out_dir: Path,
    sleep: float = 0.2,
    timeout: int = 30,
) -> pd.DataFrame:
    """Download images listed in df to out_dir/<image_id>.jpg. Returns df with local_path column."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    headers = {"User-Agent": "focal-bench/1.0"}
    for _, row in tqdm(df.iterrows(), total=len(df), desc="download"):
        image_id = int(row["image_id"])
        dest = out_dir / f"{image_id}.jpg"
        if not dest.exists():
            try:
                resp = requests.get(_dp_challenge_url(image_id), timeout=timeout, headers=headers)
                if resp.status_code == 200 and len(resp.content) > 1000:
                    dest.write_bytes(resp.content)
                    time.sleep(sleep)
                else:
                    continue
            except Exception:
                continue
        if dest.exists():
            rows.append({**row.to_dict(), "local_path": str(dest)})
    return pd.DataFrame(rows)
```

- [ ] **Step 5: Run tests — pass**

- [ ] **Step 6: Commit**

```bash
git add testing/bench/bench/dataset_ava.py testing/bench/tests/test_dataset_ava.py testing/bench/tests/fixtures/mini_ava.txt
git commit -m "feat(bench): AVA dataset parser with MOS, stratified sampling, downloader"
```

---

## Task 5: Extract `bucketEdges` and `clipLogitScale` into `FocalSettings`

The optimizer needs to tune these. Currently `bucketEdges` is hardcoded at `ProcessingQueue.swift:282-288` and `clipLogitScale` at `RatingPipeline.swift:85`.

**Files:**
- Modify: `ImageRater/App/FocalSettings.swift`
- Modify: `ImageRater/Pipeline/RatingPipeline.swift`
- Modify: `ImageRater/Pipeline/ProcessingQueue.swift`
- Modify: `ImageRaterTests/RatingPipelineTests.swift` (if behaviour assertions exist there)

- [ ] **Step 1: Add new constants to `FocalSettings`**

Replace `ImageRater/App/FocalSettings.swift` with:
```swift
// ImageRater/App/FocalSettings.swift
import Foundation

/// Centralised UserDefaults keys for Focal. All app preferences live here.
/// Use these constants with @AppStorage or UserDefaults directly.
enum FocalSettings {

    // MARK: - Cull
    /// Percentile strictness for star assignment (0.0 = lenient, 1.0 = strict). Default: 0.5
    static let cullStrictness    = "focal.cull.strictness"

    // MARK: - Rating model weights
    static let weightTechnical   = "focal.rating.weightTechnical"
    static let weightAesthetic   = "focal.rating.weightAesthetic"
    static let weightClip        = "focal.rating.weightClip"

    // MARK: - Star bucket edges (percentile cut-points in warped space)
    static let bucketEdge1       = "focal.rating.bucketEdge1"
    static let bucketEdge2       = "focal.rating.bucketEdge2"
    static let bucketEdge3       = "focal.rating.bucketEdge3"
    static let bucketEdge4       = "focal.rating.bucketEdge4"

    // MARK: - CLIP-IQA softmax temperature
    static let clipLogitScale    = "focal.rating.clipLogitScale"

    // MARK: - UI
    static let defaultCellSize   = "focal.ui.defaultCellSize"

    // MARK: - Export
    static let autoWriteXMP      = "focal.export.autoWriteXMP"

    // MARK: - Defaults
    static let defaultCullStrictness: Double  = 0.5
    static let defaultWeightTechnical: Double = 0.4
    static let defaultWeightAesthetic: Double = 0.4
    static let defaultWeightClip: Double      = 0.2
    static let defaultBucketEdge1: Double     = 0.20
    static let defaultBucketEdge2: Double     = 0.40
    static let defaultBucketEdge3: Double     = 0.60
    static let defaultBucketEdge4: Double     = 0.80
    static let defaultClipLogitScale: Double  = 100.0
    static let defaultCellSizeValue: Double   = 160
    static let defaultAutoWriteXMP: Bool      = true

    // MARK: - Resolved accessors (UserDefaults override → default)

    static func resolvedBucketEdges() -> (Double, Double, Double, Double) {
        let ud = UserDefaults.standard
        func r(_ key: String, _ d: Double) -> Double {
            ud.object(forKey: key) != nil ? ud.double(forKey: key) : d
        }
        return (
            r(bucketEdge1, defaultBucketEdge1),
            r(bucketEdge2, defaultBucketEdge2),
            r(bucketEdge3, defaultBucketEdge3),
            r(bucketEdge4, defaultBucketEdge4),
        )
    }

    static func resolvedClipLogitScale() -> Double {
        let ud = UserDefaults.standard
        return ud.object(forKey: clipLogitScale) != nil
            ? ud.double(forKey: clipLogitScale)
            : defaultClipLogitScale
    }

    // MARK: - Migration
    static func migrateIfNeeded() {
        let ud = UserDefaults.standard
        if ud.object(forKey: "cullStrictness") != nil,
           ud.object(forKey: cullStrictness) == nil {
            ud.set(ud.double(forKey: "cullStrictness"), forKey: cullStrictness)
            ud.removeObject(forKey: "cullStrictness")
        }
    }
}
```

- [ ] **Step 2: Wire `clipLogitScale` in `RatingPipeline`**

In `ImageRater/Pipeline/RatingPipeline.swift`, change the `rate(...)` body to pass the resolved logit scale into `clipIQAScore`. Default param on `clipIQAScore` stays for call sites that pass explicitly.

Replace this block (around line 51-53):
```swift
let (tech, aes, emb) = try await (techScoreTask, aesScoreTask, clipEmbTask)
let clip     = clipIQAScore(embedding: emb)
let combined = combinedQuality(technical: tech, aesthetic: aes, semantic: clip, weights: weights)
```

with:
```swift
let (tech, aes, emb) = try await (techScoreTask, aesScoreTask, clipEmbTask)
let logitScale = Float(FocalSettings.resolvedClipLogitScale())
let clip     = clipIQAScore(embedding: emb, logitScale: logitScale)
let combined = combinedQuality(technical: tech, aesthetic: aes, semantic: clip, weights: weights)
```

- [ ] **Step 3: Wire `bucketEdges` in `ProcessingQueue.percentileStars`**

Modify `ImageRater/Pipeline/ProcessingQueue.swift:272-291`:

Replace the `percentileStars` function with:
```swift
/// Assigns 1–5 stars using a power-curve skew on percentile rank.
/// γ = 10^(2s−1): s=0→γ=0.1 (lenient), s=0.5→γ=1 (uniform), s=1→γ=10 (strict)
/// Bucket edges read from FocalSettings (defaults 0.2/0.4/0.6/0.8).
private func percentileStars(scores: [Float], strictness: Double) -> [Int] {
    let n = scores.count
    guard n > 0 else { return [] }
    let gamma = pow(10.0, 2 * min(max(strictness, 0), 1) - 1)
    let (e1, e2, e3, e4) = FocalSettings.resolvedBucketEdges()
    let sorted = scores.enumerated().sorted { $0.element < $1.element }
    var result = [Int](repeating: 3, count: n)
    for (rank, (originalIndex, _)) in sorted.enumerated() {
        let pct = Double(rank) / Double(n)
        let warped = pow(pct == 0 ? 1e-9 : pct, gamma)
        result[originalIndex] = switch warped {
        case ..<e1:        1
        case e1..<e2:      2
        case e2..<e3:      3
        case e3..<e4:      4
        default:           5
        }
    }
    return result
}
```

- [ ] **Step 4: Build + run existing Swift tests — should still pass (behaviour unchanged when defaults are active)**

```bash
xcodebuild test -scheme Focal -destination 'platform=macOS,arch=arm64' 2>&1 | tail -30
```

Expected: all existing tests pass.

- [ ] **Step 5: Add a Swift test asserting override propagation**

Append to `ImageRaterTests/RatingPipelineTests.swift` (or add a new file if that doesn't exist):
```swift
func testBucketEdgesReadFromSettings() {
    let ud = UserDefaults.standard
    let keys = [
        FocalSettings.bucketEdge1,
        FocalSettings.bucketEdge2,
        FocalSettings.bucketEdge3,
        FocalSettings.bucketEdge4,
    ]
    defer { keys.forEach { ud.removeObject(forKey: $0) } }
    ud.set(0.1, forKey: FocalSettings.bucketEdge1)
    ud.set(0.3, forKey: FocalSettings.bucketEdge2)
    ud.set(0.5, forKey: FocalSettings.bucketEdge3)
    ud.set(0.7, forKey: FocalSettings.bucketEdge4)
    let edges = FocalSettings.resolvedBucketEdges()
    XCTAssertEqual(edges.0, 0.1, accuracy: 1e-9)
    XCTAssertEqual(edges.3, 0.7, accuracy: 1e-9)
}

func testClipLogitScaleReadFromSettings() {
    let ud = UserDefaults.standard
    defer { ud.removeObject(forKey: FocalSettings.clipLogitScale) }
    ud.set(55.5, forKey: FocalSettings.clipLogitScale)
    XCTAssertEqual(FocalSettings.resolvedClipLogitScale(), 55.5, accuracy: 1e-9)
}
```

- [ ] **Step 6: Run — pass**

- [ ] **Step 7: Commit**

```bash
git add ImageRater/App/FocalSettings.swift ImageRater/Pipeline/RatingPipeline.swift ImageRater/Pipeline/ProcessingQueue.swift ImageRaterTests/RatingPipelineTests.swift
git commit -m "feat: extract bucketEdges + clipLogitScale into FocalSettings"
```

---

## Task 6: `FocalScorer` Swift CLI target

Produces per-image JSON sub-scores against the deployed `.mlmodelc` models. Reuses `RatingPipeline` and `LibRawWrapper` via shared source inclusion — not a separate Swift package.

**Files:**
- Create: `FocalScorer/main.swift`
- Create: `FocalScorer/Scorer.swift`
- Modify: `project.yml` — add `FocalScorer` target

### Output JSON schema

```json
{
  "generatedAt": "2026-04-15T10:00:00Z",
  "modelVersion": "topiq-nr@1, topiq-swin@1, clip-vision@1",
  "images": [
    {
      "filename": "953619.jpg",
      "topiqTechnical": 0.42,
      "topiqAesthetic": 0.67,
      "clipEmbedding": [0.01, -0.03, ...]
    }
  ]
}
```

- [ ] **Step 1: Add `FocalScorer` target to `project.yml`**

Append to `targets:` in `project.yml`:
```yaml
  FocalScorer:
    type: tool
    platform: macOS
    sources:
      - path: FocalScorer
      - path: ImageRater/Pipeline/RatingPipeline.swift
      - path: ImageRater/Pipeline/CLIPTextEmbeddings.swift
      - path: ImageRater/App/FocalSettings.swift
      - path: ImageRater/LibRaw/LibRawWrapper.swift
      - path: ImageRater/Models/RatingResult.swift
    resources:
      - ImageRater/MLModels
    settings:
      base:
        PRODUCT_BUNDLE_IDENTIFIER: com.focal.scorer
        SWIFT_OBJC_BRIDGING_HEADER: "ImageRater/ImageRater-Bridging-Header.h"
        OTHER_LDFLAGS: "-lraw"
        HEADER_SEARCH_PATHS: "/opt/homebrew/include"
        LIBRARY_SEARCH_PATHS: "/opt/homebrew/lib"
```

Add `FocalScorer` scheme below the `Focal` scheme:
```yaml
  FocalScorer:
    build:
      targets:
        FocalScorer: all
    run:
      config: Release
```

- [ ] **Step 2: Regenerate project**

```bash
xcodegen generate
```

Expected: `Loaded project ... Created project at ...`.

- [ ] **Step 3: Write `FocalScorer/main.swift`**

```swift
// FocalScorer/main.swift
import Foundation

@main
struct FocalScorerMain {
    static func main() async {
        let args = CommandLine.arguments
        guard args.count >= 3 else {
            FileHandle.standardError.write(Data("""
            usage: FocalScorer <input-dir> <output-json>
              input-dir:   directory of .jpg/.jpeg/.raf/.nef/.arw/.cr3 files
              output-json: path to write the scores JSON
            """.utf8))
            exit(2)
        }
        let inputDir  = URL(fileURLWithPath: args[1])
        let outputURL = URL(fileURLWithPath: args[2])
        do {
            try await Scorer.scoreDirectory(inputDir: inputDir, outputURL: outputURL)
        } catch {
            FileHandle.standardError.write(Data("error: \(error)\n".utf8))
            exit(1)
        }
    }
}
```

- [ ] **Step 4: Write `FocalScorer/Scorer.swift`**

```swift
// FocalScorer/Scorer.swift
import Foundation
import CoreImage

enum Scorer {

    struct OutputImage: Codable {
        let filename: String
        let topiqTechnical: Float
        let topiqAesthetic: Float
        let clipEmbedding: [Float]
    }

    struct Output: Codable {
        let generatedAt: String
        let modelVersion: String
        let images: [OutputImage]
    }

    static let supportedExts: Set<String> = [
        "jpg", "jpeg", "png", "raf", "nef", "arw", "cr3", "dng"
    ]

    static func scoreDirectory(inputDir: URL, outputURL: URL) async throws {
        let fm = FileManager.default
        let all = try fm.contentsOfDirectory(at: inputDir, includingPropertiesForKeys: nil)
        let files = all.filter { supportedExts.contains($0.pathExtension.lowercased()) }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        let models = try RatingPipeline.loadBundledModels()
        var results: [OutputImage] = []
        results.reserveCapacity(files.count)
        for (i, url) in files.enumerated() {
            FileHandle.standardError.write(Data("[\(i+1)/\(files.count)] \(url.lastPathComponent)\n".utf8))
            guard let cg = LibRawWrapper.decode(url: url) else {
                FileHandle.standardError.write(Data("  skip: decode failed\n".utf8))
                continue
            }
            let r = await RatingPipeline.rate(image: cg, models: models)
            if case .rated(let s) = r {
                results.append(OutputImage(
                    filename: url.lastPathComponent,
                    topiqTechnical: s.topiqTechnicalScore,
                    topiqAesthetic: s.topiqAestheticScore,
                    clipEmbedding:  s.clipEmbedding
                ))
            }
        }
        let out = Output(
            generatedAt: ISO8601DateFormatter().string(from: Date()),
            modelVersion: "topiq-nr, topiq-swin, clip-vision",
            images: results
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(out)
        try data.write(to: outputURL)
        FileHandle.standardError.write(Data("wrote \(results.count) scores → \(outputURL.path)\n".utf8))
    }
}
```

- [ ] **Step 5: Ensure `RatedScores` includes `clipEmbedding`**

Check `ImageRater/Models/RatingResult.swift`. If the existing `RatedScores` struct does not expose `clipEmbedding: [Float]`, add it. (From `RatingPipeline.rate` line 63 it already does; verify.)

```bash
grep -n clipEmbedding ImageRater/Models/RatingResult.swift
```

Expected: line containing `let clipEmbedding: [Float]`. If missing, add it as a stored property and update `RatingPipeline.rate` construction site.

- [ ] **Step 6: Build FocalScorer**

```bash
xcodebuild -scheme FocalScorer -configuration Release -destination 'platform=macOS,arch=arm64' build 2>&1 | tail -10
```

Expected: `** BUILD SUCCEEDED **`.

- [ ] **Step 7: Smoke run against `testing/` directory**

```bash
BIN=$(find ~/Library/Developer/Xcode/DerivedData -name FocalScorer -type f -perm +111 | head -1)
"$BIN" testing/ /tmp/scores-smoke.json
head -20 /tmp/scores-smoke.json
```

Expected: JSON output with >100 `images` entries, each with `topiqTechnical`, `topiqAesthetic`, `clipEmbedding` length 512 (or whatever the CLIP-vision output dim is).

- [ ] **Step 8: Add a Swift smoke test**

Create `ImageRaterTests/FocalScorerSmokeTests.swift`:
```swift
import XCTest
@testable import Focal

final class FocalScorerSmokeTests: XCTestCase {
    func testRatingPipelineProducesScoresForFixture() async throws {
        let repoRoot = URL(fileURLWithPath: #file)
            .deletingLastPathComponent().deletingLastPathComponent()
        let fixture = repoRoot.appendingPathComponent("testing/DSCF0013.JPG")
        try XCTSkipUnless(FileManager.default.fileExists(atPath: fixture.path),
                          "testing fixture missing")
        guard let cg = LibRawWrapper.decode(url: fixture) else {
            return XCTFail("decode failed")
        }
        let models = try RatingPipeline.loadBundledModels()
        let r = await RatingPipeline.rate(image: cg, models: models)
        if case .rated(let s) = r {
            XCTAssertGreaterThan(s.topiqTechnicalScore, 0)
            XCTAssertGreaterThan(s.topiqAestheticScore, 0)
            XCTAssertEqual(s.clipEmbedding.count, 512)
        } else {
            XCTFail("expected .rated")
        }
    }
}
```

- [ ] **Step 9: Run Swift tests — pass**

- [ ] **Step 10: Commit**

```bash
git add FocalScorer/ project.yml Focal.xcodeproj ImageRaterTests/FocalScorerSmokeTests.swift
git commit -m "feat: add FocalScorer CLI target for benchmark scoring"
```

---

## Task 7: `score.py` — invoke FocalScorer + cache

**Files:**
- Create: `testing/bench/bench/score.py`
- Create: `testing/bench/tests/test_score.py`

- [ ] **Step 1: Write tests**

`testing/bench/tests/test_score.py`:
```python
import json
import subprocess
from pathlib import Path
import pandas as pd
import pytest
from bench.score import content_hash_dir, load_scores_json, run_scorer

def test_content_hash_dir_stable(tmp_path):
    (tmp_path / "a.jpg").write_bytes(b"A")
    (tmp_path / "b.jpg").write_bytes(b"B")
    h1 = content_hash_dir(tmp_path)
    h2 = content_hash_dir(tmp_path)
    assert h1 == h2
    (tmp_path / "c.jpg").write_bytes(b"C")
    h3 = content_hash_dir(tmp_path)
    assert h1 != h3

def test_load_scores_json(tmp_path):
    blob = {
        "generatedAt": "2026-01-01T00:00:00Z",
        "modelVersion": "v1",
        "images": [
            {"filename": "a.jpg", "topiqTechnical": 0.5, "topiqAesthetic": 0.7, "clipEmbedding": [0.1, 0.2]},
            {"filename": "b.jpg", "topiqTechnical": 0.3, "topiqAesthetic": 0.6, "clipEmbedding": [0.3, 0.4]},
        ],
    }
    p = tmp_path / "scores.json"
    p.write_text(json.dumps(blob))
    df = load_scores_json(p)
    assert list(df.columns) >= ["filename", "topiqTechnical", "topiqAesthetic", "clipEmbedding"]
    assert len(df) == 2

def test_run_scorer_invokes_binary(tmp_path, monkeypatch):
    calls = []
    def fake_run(cmd, **kw):
        calls.append(cmd)
        out_path = Path(cmd[2])
        out_path.write_text(json.dumps({"generatedAt":"","modelVersion":"","images":[]}))
        class R: returncode = 0; stdout = b""; stderr = b""
        return R()
    monkeypatch.setattr(subprocess, "run", fake_run)
    run_scorer(Path("/fake/bin"), tmp_path, tmp_path / "out.json")
    assert len(calls) == 1
    assert calls[0][0] == "/fake/bin"
```

- [ ] **Step 2: Run — ImportError**

- [ ] **Step 3: Implement `score.py`**

```python
"""Run Swift FocalScorer CLI against an image directory and cache JSON."""
from __future__ import annotations
from pathlib import Path
import hashlib
import json
import subprocess
import pandas as pd


def content_hash_dir(path: Path) -> str:
    """Deterministic hash of all file contents in path (sorted by name)."""
    h = hashlib.sha256()
    for p in sorted(path.rglob("*")):
        if p.is_file():
            h.update(p.name.encode())
            h.update(p.read_bytes())
    return h.hexdigest()[:16]


def run_scorer(scorer_bin: Path, image_dir: Path, output_json: Path) -> None:
    """Invoke FocalScorer CLI. Raises CalledProcessError on failure."""
    output_json.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [str(scorer_bin), str(image_dir), str(output_json)],
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"FocalScorer failed: {result.stderr.decode(errors='ignore')}")


def load_scores_json(path: Path) -> pd.DataFrame:
    blob = json.loads(path.read_text())
    return pd.DataFrame(blob["images"])


def score_with_cache(
    scorer_bin: Path,
    image_dir: Path,
    cache_dir: Path,
) -> pd.DataFrame:
    """Score image_dir via FocalScorer, caching JSON by content hash."""
    digest = content_hash_dir(image_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / f"scores_{digest}.json"
    if not cached.exists():
        run_scorer(scorer_bin, image_dir, cached)
    return load_scores_json(cached)
```

- [ ] **Step 4: Run — pass**

- [ ] **Step 5: Commit**

```bash
git add testing/bench/bench/score.py testing/bench/tests/test_score.py
git commit -m "feat(bench): scorer invocation with content-hash caching"
```

---

## Task 8: `optimize.py` — Optuna TPE search

**Files:**
- Create: `testing/bench/bench/optimize.py`
- Create: `testing/bench/tests/test_optimize.py`

- [ ] **Step 1: Write tests (synthetic data)**

`testing/bench/tests/test_optimize.py`:
```python
import numpy as np
import pandas as pd
from bench.optimize import optimize_params, SearchSpace

def test_optimize_improves_over_default(monkeypatch):
    rng = np.random.default_rng(0)
    n = 300
    tech = rng.random(n)
    aes  = rng.random(n)
    clip = rng.random(n)
    # Construct ground truth from a skewed oracle so a specific weight combo is optimal
    true_combined = 0.7 * tech + 0.1 * aes + 0.2 * clip
    # GT stars = quintile rank
    ranks = pd.Series(true_combined).rank(method="first") - 1
    gt_stars = (ranks / n // 0.2).clip(0, 4).astype(int).values + 1

    scores_df = pd.DataFrame({
        "filename": [f"{i}.jpg" for i in range(n)],
        "topiqTechnical": tech,
        "topiqAesthetic": aes,
        "clipEmbedding": [list(rng.random(4)) for _ in range(n)],
    })
    scores_df["clipIQA"] = clip   # helper: treat supplied clip scalar as clipIQA
    labels_df = pd.DataFrame({"filename": scores_df.filename, "gt_stars": gt_stars})

    space = SearchSpace()
    best = optimize_params(scores_df, labels_df, space, n_trials=40, seed=0)
    assert best.params.wTech > best.params.wAes  # oracle weighted tech highest
    assert best.metrics.spearman > 0.5
```

- [ ] **Step 2: Run — ImportError**

- [ ] **Step 3: Implement `optimize.py`**

```python
"""Optuna TPE parameter search against a labeled scores DataFrame."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
import optuna
import pandas as pd
from .ensemble import EnsembleParams, stars_from_subscores
from .metrics import compute_metrics, MetricsResult


@dataclass
class SearchSpace:
    wTech_range:      tuple[float, float] = (0.0, 1.0)
    wAes_range:       tuple[float, float] = (0.0, 1.0)
    wClip_range:      tuple[float, float] = (0.0, 1.0)
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
        wT = trial.suggest_float("wTech", *space.wTech_range)
        wA = trial.suggest_float("wAes",  *space.wAes_range)
        wC = trial.suggest_float("wClip", *space.wClip_range)
        if wT + wA + wC < 1e-6:
            return 5.0  # degenerate
        s  = trial.suggest_float("strictness", *space.strictness_range)
        if space.search_bucket_edges:
            e1 = trial.suggest_float("e1", 0.05, 0.35)
            e2 = trial.suggest_float("e2", e1 + 0.05, 0.55)
            e3 = trial.suggest_float("e3", e2 + 0.05, 0.75)
            e4 = trial.suggest_float("e4", e3 + 0.05, 0.95)
        else:
            e1, e2, e3, e4 = 0.2, 0.4, 0.6, 0.8
        params = EnsembleParams(
            wTech=wT, wAes=wA, wClip=wC, strictness=s,
            bucket_edges=(e1, e2, e3, e4),
        )
        pred = stars_from_subscores(tech, aes, clip, params)
        m = compute_metrics(gt, pred)
        # Minimize loss: -Spearman + 0.2 * MAE
        return -m.spearman + 0.2 * m.mae
    return objective


def _clip_scalar_from_embedding(embeddings: list) -> np.ndarray:
    """Placeholder: treat first dimension. Real pipeline uses prompt-based score,
    but at benchmark time that transformation is constant across the dataset
    (only weights vary), so collapse to the pre-computed CLIP-IQA scalar if
    supplied in a 'clipIQA' column.
    """
    raise NotImplementedError  # never called; resolved in optimize_params


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
    study.optimize(_objective_factory(tech, aes, clip, gt, space), n_trials=n_trials, show_progress_bar=False)
    p = study.best_params
    edges = (
        p.get("e1", 0.2),
        p.get("e2", 0.4),
        p.get("e3", 0.6),
        p.get("e4", 0.8),
    )
    best = EnsembleParams(
        wTech=p["wTech"], wAes=p["wAes"], wClip=p["wClip"],
        strictness=p["strictness"],
        bucket_edges=edges,
    )
    pred = stars_from_subscores(tech, aes, clip, best)
    metrics = compute_metrics(gt, pred)
    return OptimizationResult(params=best, metrics=metrics, study=study)
```

- [ ] **Step 4: Run tests — pass**

- [ ] **Step 5: Commit**

```bash
git add testing/bench/bench/optimize.py testing/bench/tests/test_optimize.py
git commit -m "feat(bench): optuna TPE parameter optimizer"
```

---

## Task 9: `ablation.py` — ensemble subset sweep

**Files:**
- Create: `testing/bench/bench/ablation.py`
- Create: `testing/bench/tests/test_ablation.py`

- [ ] **Step 1: Write tests**

```python
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
```

- [ ] **Step 2: Implement `ablation.py`**

```python
"""Ensemble membership ablation. Runs the optimizer with each model subset."""
from __future__ import annotations
from dataclasses import dataclass
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
        wTech_range=(0.0, 1.0) if config["tech"] else (0.0, 0.0),
        wAes_range =(0.0, 1.0) if config["aes"]  else (0.0, 0.0),
        wClip_range=(0.0, 1.0) if config["clip"] else (0.0, 0.0),
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
```

- [ ] **Step 3: Run — pass**

- [ ] **Step 4: Commit**

```bash
git add testing/bench/bench/ablation.py testing/bench/tests/test_ablation.py
git commit -m "feat(bench): ensemble membership ablation sweep"
```

---

## Task 10: CLIP-IQA scalar helper in Python

The optimizer needs a scalar `clipIQA` column but `FocalScorer` emits only the embedding. We replicate `RatingPipeline.clipIQAScore` in Python (reading the same prompt embeddings) so tuning `clipLogitScale` varies that scalar per trial.

**Files:**
- Create: `testing/bench/bench/clip_iqa.py`
- Create: `testing/bench/tests/test_clip_iqa.py`
- Create: `testing/bench/bench/prompt_embeddings.json` (export from Swift once)

- [ ] **Step 1: Export prompts**

Add an XCTest that writes `CLIPTextEmbeddings.positivePrompts` + `negativePrompts` to JSON. Example: `ImageRaterTests/ExportPromptEmbeddings.swift` writes to `/tmp/prompt_embeddings.json`, then copy to `testing/bench/bench/prompt_embeddings.json`. This is a one-time bootstrap.

Schema:
```json
{"positive": [[...512 floats...], ...], "negative": [[...], ...]}
```

- [ ] **Step 2: Write tests**

```python
import json
import numpy as np
from pathlib import Path
from bench.clip_iqa import clip_iqa_score

def test_clip_iqa_symmetric_prompts_returns_half():
    """With identical pos/neg prompts the softmax is 0.5 for every pair."""
    emb = np.ones(4) / 2   # unit-norm placeholder
    prompts = {"positive": [[0.1, 0.2, 0.3, 0.4]], "negative": [[0.1, 0.2, 0.3, 0.4]]}
    s = clip_iqa_score(emb, prompts, logit_scale=100.0)
    assert 0.49 < s < 0.51

def test_clip_iqa_positive_favored():
    emb = np.array([1.0, 0.0, 0.0, 0.0])
    prompts = {
        "positive": [[1.0, 0.0, 0.0, 0.0]],
        "negative": [[0.0, 1.0, 0.0, 0.0]],
    }
    s = clip_iqa_score(emb, prompts, logit_scale=100.0)
    assert s > 0.9
```

- [ ] **Step 3: Implement `clip_iqa.py`**

```python
"""Python port of RatingPipeline.clipIQAScore."""
import numpy as np


def clip_iqa_score(embedding: np.ndarray, prompts: dict, logit_scale: float = 100.0) -> float:
    emb = np.asarray(embedding, dtype=np.float32)
    norm = np.linalg.norm(emb)
    if norm < 1e-6:
        return 0.5
    if not (0.999 < norm < 1.001):
        emb = emb / norm
    pos = np.asarray(prompts["positive"], dtype=np.float32)
    neg = np.asarray(prompts["negative"], dtype=np.float32)
    assert pos.shape == neg.shape
    dot_pos = logit_scale * (emb[None, :] * pos).sum(axis=1)
    dot_neg = logit_scale * (emb[None, :] * neg).sum(axis=1)
    max_v = np.maximum(dot_pos, dot_neg)
    e_pos = np.exp(dot_pos - max_v)
    e_neg = np.exp(dot_neg - max_v)
    probs = e_pos / (e_pos + e_neg)
    return float(probs.mean())
```

- [ ] **Step 4: Run — pass**

- [ ] **Step 5: Commit**

```bash
git add testing/bench/bench/clip_iqa.py testing/bench/bench/prompt_embeddings.json testing/bench/tests/test_clip_iqa.py ImageRaterTests/ExportPromptEmbeddings.swift
git commit -m "feat(bench): CLIP-IQA scalar helper with exported prompt embeddings"
```

---

## Task 11: `leaderboard.py` + `LEADERBOARD.md`

**Files:**
- Create: `testing/bench/bench/leaderboard.py`
- Create: `testing/bench/tests/test_leaderboard.py`

- [ ] **Step 1: Write tests**

```python
from pathlib import Path
import json
from bench.leaderboard import regenerate_leaderboard

def test_leaderboard_sorted(tmp_path):
    results_dir = tmp_path / "results"
    for ver, spearman in [("v0.1.0@aaa1111", 0.55), ("v0.2.0@bbb2222", 0.67), ("v0.3.0@ccc3333", 0.60)]:
        d = results_dir / ver
        d.mkdir(parents=True)
        (d / "metrics.json").write_text(json.dumps({"val": {"spearman": spearman, "mae": 0.8, "off_by_one": 0.9, "exact_match": 0.5}}))
        (d / "params.json").write_text(json.dumps({"ensemble": ["tech","aes","clip"], "notes": f"ver {ver}", "date": "2026-04-15"}))
    md_path = tmp_path / "LEADERBOARD.md"
    regenerate_leaderboard(results_dir, md_path)
    text = md_path.read_text()
    # Best first
    v2_pos = text.index("v0.2.0")
    v3_pos = text.index("v0.3.0")
    v1_pos = text.index("v0.1.0")
    assert v2_pos < v3_pos < v1_pos
```

- [ ] **Step 2: Implement `leaderboard.py`**

```python
"""Regenerate LEADERBOARD.md from testing/bench/results/*/metrics.json."""
from __future__ import annotations
from pathlib import Path
import json


def regenerate_leaderboard(results_dir: Path, md_path: Path) -> None:
    rows = []
    for version_dir in sorted(results_dir.iterdir()):
        if not version_dir.is_dir():
            continue
        m = version_dir / "metrics.json"
        p = version_dir / "params.json"
        if not m.exists() or not p.exists():
            continue
        metrics = json.loads(m.read_text()).get("val", {})
        params = json.loads(p.read_text())
        rows.append({
            "version": version_dir.name,
            "date": params.get("date", ""),
            "spearman": metrics.get("spearman", float("nan")),
            "mae": metrics.get("mae", float("nan")),
            "off_by_one": metrics.get("off_by_one", float("nan")),
            "ensemble": "+".join(params.get("ensemble", [])),
            "notes": params.get("notes", ""),
        })

    rows.sort(key=lambda r: -r["spearman"])

    lines = [
        "# Rating Ensemble Leaderboard",
        "",
        "Sorted by val Spearman (desc). Regenerated by `run.py leaderboard`.",
        "",
        "| Version | Date | Spearman | MAE | ±1 acc | Ensemble | Notes |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['version']} | {r['date']} | {r['spearman']:.3f} | {r['mae']:.2f} | "
            f"{r['off_by_one']*100:.0f}% | {r['ensemble']} | {r['notes']} |"
        )
    md_path.write_text("\n".join(lines) + "\n")
```

- [ ] **Step 3: Run — pass**

- [ ] **Step 4: Commit**

```bash
git add testing/bench/bench/leaderboard.py testing/bench/tests/test_leaderboard.py
git commit -m "feat(bench): leaderboard markdown generation"
```

---

## Task 12: `run.py` orchestration

**Files:**
- Create: `testing/bench/run.py`
- Create: `testing/bench/params.current.json`

- [ ] **Step 1: Create starter `params.current.json`**

```json
{
  "version": "v0.1.0",
  "date": "2026-04-15",
  "ensemble": ["tech", "aes", "clip"],
  "notes": "initial defaults (pre-tuning)",
  "params": {
    "wTech": 0.4,
    "wAes": 0.4,
    "wClip": 0.2,
    "strictness": 0.5,
    "bucket_edges": [0.2, 0.4, 0.6, 0.8],
    "clipLogitScale": 100.0
  }
}
```

- [ ] **Step 2: Write `run.py`**

```python
#!/usr/bin/env python3
"""Focal bench orchestration: score / eval / optimize / ablate / leaderboard."""
from __future__ import annotations
import argparse
import json
import subprocess
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from bench.dataset_ava import parse_ava_txt, compute_mos, mos_to_stars, stratified_sample, download_images
from bench.score import score_with_cache
from bench.clip_iqa import clip_iqa_score
from bench.ensemble import EnsembleParams, stars_from_subscores
from bench.metrics import compute_metrics
from bench.optimize import optimize_params, SearchSpace
from bench.ablation import run_ablation
from bench.leaderboard import regenerate_leaderboard


BENCH_DIR     = Path(__file__).parent
DATA_DIR      = BENCH_DIR / "data"
RESULTS_DIR   = BENCH_DIR / "results"
CACHE_DIR     = BENCH_DIR / ".cache"
PROMPTS_PATH  = BENCH_DIR / "bench" / "prompt_embeddings.json"


def git_sha_short() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "nogit"


def locate_scorer_bin() -> Path:
    candidates = list(Path.home().rglob("FocalScorer"))
    bins = [c for c in candidates if c.is_file() and c.stat().st_mode & 0o111]
    if not bins:
        raise FileNotFoundError(
            "FocalScorer binary not found. Build with: "
            "xcodebuild -scheme FocalScorer -configuration Release -destination 'platform=macOS,arch=arm64' build"
        )
    # Most recent
    bins.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return bins[0]


def load_params(path: Path) -> dict:
    return json.loads(path.read_text())


def save_params(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2))


def version_with_sha(version: str) -> str:
    return f"{version}@{git_sha_short()}"


def cmd_download(args):
    ava_txt = DATA_DIR / "ava" / "AVA.txt"
    if not ava_txt.exists():
        raise SystemExit(
            f"AVA.txt missing at {ava_txt}. Download AVA labels first: "
            "https://github.com/mtobeiyf/ava_downloader"
        )
    df = parse_ava_txt(ava_txt)
    df = compute_mos(df)
    df = mos_to_stars(df)
    sampled = stratified_sample(df, n=args.sample, seed=0)
    out = DATA_DIR / "ava" / "images"
    downloaded = download_images(sampled, out)
    labels = downloaded[["image_id", "mos", "gt_stars", "local_path"]].copy()
    labels["filename"] = labels["local_path"].apply(lambda p: Path(p).name)
    labels[["filename", "image_id", "mos", "gt_stars"]].to_csv(DATA_DIR / "ava" / "labels.csv", index=False)
    print(f"downloaded {len(downloaded)} / {len(sampled)}; labels written to data/ava/labels.csv")


def _scores_with_clip_scalar(scores_df: pd.DataFrame, logit_scale: float) -> pd.DataFrame:
    prompts = json.loads(PROMPTS_PATH.read_text())
    scores_df = scores_df.copy()
    scores_df["clipIQA"] = scores_df["clipEmbedding"].apply(
        lambda e: clip_iqa_score(np.asarray(e, dtype=np.float32), prompts, logit_scale)
    )
    return scores_df


def _load_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = pd.read_csv(DATA_DIR / "ava" / "labels.csv")
    bin_ = locate_scorer_bin()
    scores = score_with_cache(bin_, DATA_DIR / "ava" / "images", CACHE_DIR)
    return scores, labels


def cmd_score(args):
    scores, _ = _load_dataset()
    print(scores.head())
    print(f"scored {len(scores)} images (cached under {CACHE_DIR})")


def cmd_eval(args):
    scores, labels = _load_dataset()
    payload = load_params(Path(args.params))
    p = payload["params"]
    scores = _scores_with_clip_scalar(scores, p["clipLogitScale"])
    df = scores.merge(labels, on="filename", how="inner")
    params = EnsembleParams(
        wTech=p["wTech"], wAes=p["wAes"], wClip=p["wClip"],
        strictness=p["strictness"],
        bucket_edges=tuple(p["bucket_edges"]),
    )
    pred = stars_from_subscores(
        df["topiqTechnical"].to_numpy(),
        df["topiqAesthetic"].to_numpy(),
        df["clipIQA"].to_numpy(),
        params,
    )
    m = compute_metrics(df["gt_stars"].to_numpy(), pred)

    version = version_with_sha(payload["version"])
    out_dir = RESULTS_DIR / version
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps({"val": m.to_dict()}, indent=2))
    (out_dir / "params.json").write_text(json.dumps({
        **payload, "date": datetime.utcnow().strftime("%Y-%m-%d")
    }, indent=2))

    df_out = df.copy()
    df_out["pred_stars"] = pred
    df_out[["filename", "gt_stars", "pred_stars", "topiqTechnical", "topiqAesthetic", "clipIQA"]].to_parquet(out_dir / "scores.parquet")

    print(f"Spearman={m.spearman:.3f}  MAE={m.mae:.2f}  ±1={m.off_by_one*100:.0f}%")
    print(f"results → {out_dir}")


def cmd_optimize(args):
    scores, labels = _load_dataset()
    # For optimizer, fix logit_scale at current params (then retune in a second pass).
    current = load_params(BENCH_DIR / "params.current.json")
    scores = _scores_with_clip_scalar(scores, current["params"]["clipLogitScale"])
    res = optimize_params(scores, labels, SearchSpace(), n_trials=args.trials, seed=0)
    candidate = {
        "version": args.version,
        "date": datetime.utcnow().strftime("%Y-%m-%d"),
        "ensemble": ["tech", "aes", "clip"],
        "notes": f"optuna TPE {args.trials} trials",
        "params": {
            "wTech": res.params.wTech,
            "wAes": res.params.wAes,
            "wClip": res.params.wClip,
            "strictness": res.params.strictness,
            "bucket_edges": list(res.params.bucket_edges),
            "clipLogitScale": current["params"]["clipLogitScale"],
        },
    }
    out = Path(args.out) if args.out else BENCH_DIR / "params.candidate.json"
    save_params(out, candidate)
    print(f"best Spearman={res.metrics.spearman:.3f}  MAE={res.metrics.mae:.2f}")
    print(f"candidate → {out}")


def cmd_ablate(args):
    scores, labels = _load_dataset()
    current = load_params(BENCH_DIR / "params.current.json")
    scores = _scores_with_clip_scalar(scores, current["params"]["clipLogitScale"])
    results = run_ablation(scores, labels, n_trials=args.trials, seed=0)
    for name, r in sorted(results.items(), key=lambda kv: -kv[1].metrics.spearman):
        print(f"{name:10s}  Spearman={r.metrics.spearman:.3f}  MAE={r.metrics.mae:.2f}")


def cmd_leaderboard(args):
    regenerate_leaderboard(RESULTS_DIR, BENCH_DIR / "LEADERBOARD.md")
    print(f"leaderboard → {BENCH_DIR / 'LEADERBOARD.md'}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("download")
    d.add_argument("--sample", type=int, default=500)
    d.set_defaults(func=cmd_download)

    s = sub.add_parser("score")
    s.set_defaults(func=cmd_score)

    e = sub.add_parser("eval")
    e.add_argument("--params", default=str(BENCH_DIR / "params.current.json"))
    e.set_defaults(func=cmd_eval)

    o = sub.add_parser("optimize")
    o.add_argument("--trials", type=int, default=500)
    o.add_argument("--version", default="v0.2.0")
    o.add_argument("--out")
    o.set_defaults(func=cmd_optimize)

    a = sub.add_parser("ablate")
    a.add_argument("--trials", type=int, default=100)
    a.set_defaults(func=cmd_ablate)

    l = sub.add_parser("leaderboard")
    l.set_defaults(func=cmd_leaderboard)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Make executable**

```bash
chmod +x testing/bench/run.py
```

- [ ] **Step 4: Commit**

```bash
git add testing/bench/run.py testing/bench/params.current.json
git commit -m "feat(bench): run.py orchestration (download/score/eval/optimize/ablate/leaderboard)"
```

---

## Task 13: `gen_defaults.py` — param bridge to Swift

**Files:**
- Create: `scripts/gen_defaults.py`
- Create: `ImageRater/App/FocalSettings+Generated.swift` (initial skeleton; regenerated by script)
- Modify: `project.yml` (pre-build phase)
- Modify: `ImageRater/App/FocalSettings.swift` — point `default...` to generated constants

- [ ] **Step 1: Write `scripts/gen_defaults.py`**

```python
#!/usr/bin/env python3
"""Generate Swift constants from testing/bench/params.current.json.

Run manually or via xcodegen pre-build phase. Overwrites
ImageRater/App/FocalSettings+Generated.swift.
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
PARAMS_PATH = ROOT / "testing" / "bench" / "params.current.json"
OUT_PATH    = ROOT / "ImageRater" / "App" / "FocalSettings+Generated.swift"


TEMPLATE = """// Auto-generated from testing/bench/params.current.json. DO NOT EDIT.
// Regenerate via `python3 scripts/gen_defaults.py`.
import Foundation

extension FocalSettings {{
    static let generatedVersion: String               = "{version}"
    static let generatedWeightTechnical: Double       = {wTech}
    static let generatedWeightAesthetic: Double       = {wAes}
    static let generatedWeightClip: Double            = {wClip}
    static let generatedCullStrictness: Double        = {strictness}
    static let generatedBucketEdge1: Double           = {e1}
    static let generatedBucketEdge2: Double           = {e2}
    static let generatedBucketEdge3: Double           = {e3}
    static let generatedBucketEdge4: Double           = {e4}
    static let generatedClipLogitScale: Double        = {clipLogit}
}}
"""


def main() -> None:
    payload = json.loads(PARAMS_PATH.read_text())
    p = payload["params"]
    e = p["bucket_edges"]
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(TEMPLATE.format(
        version=payload["version"],
        wTech=p["wTech"], wAes=p["wAes"], wClip=p["wClip"],
        strictness=p["strictness"],
        e1=e[0], e2=e[1], e3=e[2], e4=e[3],
        clipLogit=p["clipLogitScale"],
    ))
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it once**

```bash
python3 scripts/gen_defaults.py
```

Expected: `wrote ImageRater/App/FocalSettings+Generated.swift`. Open file; verify contents.

- [ ] **Step 3: Update `FocalSettings.swift` defaults to point at generated**

In `ImageRater/App/FocalSettings.swift`, replace the `Defaults` section:
```swift
    // MARK: - Defaults (backed by FocalSettings+Generated.swift)
    static var defaultCullStrictness: Double   { generatedCullStrictness }
    static var defaultWeightTechnical: Double  { generatedWeightTechnical }
    static var defaultWeightAesthetic: Double  { generatedWeightAesthetic }
    static var defaultWeightClip: Double       { generatedWeightClip }
    static var defaultBucketEdge1: Double      { generatedBucketEdge1 }
    static var defaultBucketEdge2: Double      { generatedBucketEdge2 }
    static var defaultBucketEdge3: Double      { generatedBucketEdge3 }
    static var defaultBucketEdge4: Double      { generatedBucketEdge4 }
    static var defaultClipLogitScale: Double   { generatedClipLogitScale }
    static let defaultCellSizeValue: Double    = 160
    static let defaultAutoWriteXMP: Bool       = true
```

- [ ] **Step 4: Add build phase to `project.yml`**

Under the `Focal` target, add `preBuildScripts`:
```yaml
    preBuildScripts:
      - name: Regenerate FocalSettings+Generated.swift
        script: |
          cd "$PROJECT_DIR"
          /usr/bin/env python3 scripts/gen_defaults.py
        outputFiles:
          - $(SRCROOT)/ImageRater/App/FocalSettings+Generated.swift
```

- [ ] **Step 5: Regenerate + build**

```bash
xcodegen generate
xcodebuild -scheme Focal build 2>&1 | tail -10
```

Expected: `** BUILD SUCCEEDED **`.

- [ ] **Step 6: Swift test — generated default matches params.current.json**

Add `ImageRaterTests/GeneratedDefaultsTests.swift`:
```swift
import XCTest
@testable import Focal

final class GeneratedDefaultsTests: XCTestCase {
    func testGeneratedWeightsMatchParamsJSON() throws {
        let url = URL(fileURLWithPath: #file)
            .deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("testing/bench/params.current.json")
        let data = try Data(contentsOf: url)
        let json = try JSONSerialization.jsonObject(with: data) as! [String: Any]
        let p = json["params"] as! [String: Any]

        XCTAssertEqual(FocalSettings.generatedWeightTechnical, p["wTech"] as! Double, accuracy: 1e-9)
        XCTAssertEqual(FocalSettings.generatedWeightAesthetic, p["wAes"]  as! Double, accuracy: 1e-9)
        XCTAssertEqual(FocalSettings.generatedWeightClip,      p["wClip"] as! Double, accuracy: 1e-9)
    }
}
```

- [ ] **Step 7: Run — pass**

- [ ] **Step 8: Commit**

```bash
git add scripts/gen_defaults.py ImageRater/App/FocalSettings+Generated.swift ImageRater/App/FocalSettings.swift project.yml Focal.xcodeproj ImageRaterTests/GeneratedDefaultsTests.swift
git commit -m "feat: generate FocalSettings defaults from params.current.json at build time"
```

---

## Task 14: End-to-end dry run + dataset bootstrap

Do a real run against a small subset to prove the pipeline end-to-end.

- [ ] **Step 1: Obtain AVA.txt**

```bash
mkdir -p testing/bench/data/ava
# AVA.txt distributed via https://github.com/mtobeiyf/ava_downloader
# manual: download AVA.txt to testing/bench/data/ava/AVA.txt
ls -l testing/bench/data/ava/AVA.txt
```

Expected: ~255,000 lines.

- [ ] **Step 2: Download 500-image subset**

```bash
cd testing/bench
source .venv/bin/activate
./run.py download --sample 500
```

Expected: `downloaded N / 500; labels written to data/ava/labels.csv` where N ≥ 400 (some AVA images 404 on dpchallenge).

- [ ] **Step 3: Build FocalScorer release binary**

```bash
xcodebuild -scheme FocalScorer -configuration Release -destination 'platform=macOS,arch=arm64' build
```

- [ ] **Step 4: Score**

```bash
./run.py score
```

Expected: `scored N images (cached under ...)`. Takes ~5-10 min for 500 images on M-series Mac.

- [ ] **Step 5: Baseline eval**

```bash
./run.py eval
```

Expected output like:
```
Spearman=0.4xx  MAE=0.xx  ±1=xx%
results → .../v0.1.0@<sha>
```

- [ ] **Step 6: Short optimize run**

```bash
./run.py optimize --trials 50 --version v0.2.0
```

Expected: `best Spearman=0.5xx  MAE=0.xx  candidate → params.candidate.json`.

- [ ] **Step 7: Eval candidate**

```bash
./run.py eval --params params.candidate.json
```

Expected: metrics printed + results written.

- [ ] **Step 8: Regenerate leaderboard**

```bash
./run.py leaderboard
cat LEADERBOARD.md
```

Expected: table with v0.1.0 + v0.2.0 rows sorted by Spearman desc.

- [ ] **Step 9: Commit seed results**

```bash
git add testing/bench/LEADERBOARD.md
git commit -m "bench: seed leaderboard with v0.1.0 baseline + v0.2.0 short run"
```

Results JSONs under `testing/bench/results/` are gitignored (see Task 1) except for `LEADERBOARD.md`.

- [ ] **Step 10: Ablation sweep**

```bash
./run.py ablate --trials 80
```

Expected: 7-row table ranking ensemble subsets. If `no-clip` ≥ `baseline`, flag in a follow-up commit / note; model-swap decisions are downstream.

- [ ] **Step 11: Final commit**

```bash
git commit --allow-empty -m "bench: initial ablation sweep complete — see run output"
```

---

## Done criteria

- `pytest testing/bench/tests -v` — all green.
- `xcodebuild test -scheme Focal` — all green.
- `xcodebuild -scheme FocalScorer build` — succeeds.
- `./run.py eval` — prints Spearman/MAE/±1 without errors on seeded 500-image AVA subset.
- `./run.py optimize --trials 50` — produces `params.candidate.json` with Spearman ≥ baseline.
- `./run.py leaderboard` — regenerates `LEADERBOARD.md` with ≥ 2 versions.
- `params.current.json` change + `gen_defaults.py` → `FocalSettings+Generated.swift` reflects change → `xcodebuild build` picks it up.
