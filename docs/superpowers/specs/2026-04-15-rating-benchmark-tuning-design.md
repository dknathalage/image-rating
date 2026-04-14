# Rating Benchmark & Parameter Tuning — Design

**Date:** 2026-04-15
**Status:** Approved for planning

## Goal

Tune the Focal rating ensemble (models + hand-tunable parameters) so predicted 1–5 star ratings match average photographer consensus, measured against a public ground-truth dataset. Stand up a versioned benchmark suite that tracks ensemble quality over time and prevents regressions.

## Non-goals

- Retraining the ML models from scratch.
- Modeling an individual user's taste (that comes later; infrastructure here should support it).
- On-device tuning.

## Ground truth dataset

**AVA (Aesthetic Visual Analysis) — eval subset (~20k photos).**

- Ratings are 1–10 MOS from photo.net photography community (matches "average photographer consensus" target better than generic crowdworker datasets like KonIQ or SPAQ).
- Per-image MOS computed from 10-bin vote histogram in `AVA.txt`: `MOS = Σ(i · votes_i) / Σ votes_i`.
- Ground truth 1–5 stars = quintile rank of MOS across the dataset (same percentile logic the app uses to assign stars).
- Matches how Focal assigns stars: rank-based, not absolute-value-based.
- Download: `testing/bench/data/ava/images/` via `dataset_ava.py` download script. Gitignored. License: research use.
- Fallback: `--sample 2000` stratified by MOS when full 20k is too slow.

## Architecture

```
┌─────────────────────────┐     ┌──────────────────────────┐
│ FocalScorer (Swift CLI) │───→ │ per-image scores JSON    │
│ - loads .mlmodelc       │     │ {tech, aes, clipEmb[512]}│
│ - decodes img (LibRaw)  │     └──────────────────────────┘
│ - runs 3 models (prod)  │                │
└─────────────────────────┘                ▼
                                 ┌────────────────────────┐
                                 │ focal-bench (Python)   │
                                 │ - load AVA labels      │
                                 │ - join scores + labels │
                                 │ - optuna optimizer     │
                                 │ - metric suite         │
                                 │ - leaderboard.md       │
                                 └────────────────────────┘
```

**Key principle:** the *deployed* model artifacts (`.mlmodelc`) are used for scoring, not raw PyTorch weights. Swift owns inference; Python owns stats, optimization, and tracking.

**Boundary:** JSON schema for per-image scores decouples the two sides. Either side can be replaced independently.

### Components

**`FocalScorer` (new Swift CLI target)**
- Reuses `RatingPipeline.loadBundledModels()` and `LibRawWrapper`.
- Input: directory of images + output JSON path.
- Output: JSON array of `{ filename, topiqTechnical, topiqAesthetic, clipEmbedding }`.
- Decouples scoring (expensive) from tuning (cheap) — score once, tune many times against cached results.

**`testing/bench/` (Python harness)**
```
testing/bench/
  run.py                   # end-to-end orchestration
  score.py                 # invokes FocalScorer CLI, caches JSON
  optimize.py              # optuna TPE param search
  ensemble.py              # combines sub-scores → stars (mirrors Swift logic)
  metrics.py               # Spearman, Kendall, MAE, off-by-one, exact-match, confusion
  leaderboard.py           # regenerates LEADERBOARD.md from results/
  dataset_ava.py           # download + parse AVA
  data/ava/                # gitignored
  results/<version>/
    params.json
    metrics.json
    trials.jsonl           # optuna history when applicable
    scores.parquet         # per-image preds vs ground truth
    confusion.png
  LEADERBOARD.md           # auto-generated, committed
  params.current.json      # params the shipping app uses
```

## Tuning surface

### Continuous / discrete parameters (optuna search space)

| Parameter | Type | Range | Source |
|---|---|---|---|
| `wTech` | continuous | 0.0–1.0 | `FocalSettings.weightTechnical` |
| `wAes` | continuous | 0.0–1.0 | `FocalSettings.weightAesthetic` |
| `wClip` | continuous | 0.0–1.0 | `FocalSettings.weightClip` |
| `strictness` | continuous | 0.0–1.0 | `FocalSettings.cullStrictness` (drives γ curve) |
| `bucketEdges` | 4-tuple | sorted in (0,1) | currently hardcoded `[0.2, 0.4, 0.6, 0.8]` — extract to settings |
| `clipLogitScale` | continuous | 10–200 | `RatingPipeline.clipIQAScore` logitScale |
| `clipPromptSet` | categorical | {current, expanded-8pair, aesthetic-focused} | `CLIPTextEmbeddings` |
| `normMode` | categorical | {per-session, fixed-dataset} | `ProcessingQueue.normalizeAndWriteStars` |

Weights normalized to sum=1 inside the objective function (not constrained in search).

### Ensemble membership (separate ablation sweep)

Evaluate each subset of `{TOPIQ-NR, TOPIQ-Swin, CLIP-IQA}`:
- baseline (all 3)
- no-clip, no-aes, no-tech (drop-one)
- tech-only, aes-only, clip-only (keep-one)

Drop any model that doesn't improve Spearman meaningfully over the best subset without it.

### Candidate model swaps (if current models underperform)

| Swap | Model | Rationale |
|---|---|---|
| aesthetic | NIMA-VGG16 | Already in `scripts/nima_weights/`, classic AVA baseline |
| aesthetic | MUSIQ | Transformer, SOTA on AVA, Core ML convertible |
| aesthetic/technical | MANIQA | Multi-dim attention, top KonIQ |
| technical | HyperIQA | Semantic-aware technical quality |

Swap triggers: val Spearman < 0.60 after tuning current ensemble. Order: NIMA first (local), then MUSIQ (requires conversion). Stop when Spearman ≥ 0.65 OR marginal gain per swap < 0.02.

## Optimization

- **Optimizer:** Optuna TPE sampler, 500 trials.
- **Objective (minimize):** `loss = -Spearman(predStars, gtStars) + 0.2 · MAE(predStars, gtStars)`.
  - Rank correlation dominates because star rating is ordinal.
  - MAE penalty prevents pathological solutions that rank well but collapse to a single bucket.
- **Split:** 80/20 train/val, fixed seed. Final reported metric on val. Held-out test set for leaderboard reporting only.
- **Output:** `results/<version>/best_params.json` + full trial history.

## Metrics

- **Spearman ρ** (primary) — rank correlation between predicted and ground-truth stars.
- **Kendall τ** — secondary rank metric.
- **MAE** in star units.
- **Off-by-one accuracy** — % predictions within ±1 star.
- **Exact-match accuracy** — % predictions matching ground truth exactly.
- **Per-star confusion matrix** (5×5).
- **Worst-10 diagnostic** — images with largest residual, for error analysis.

## Versioning

**Algo version = any change to the ensemble's inputs or outputs.**

Triggers on any of:
- Ensemble membership change (add/remove/swap model).
- Model weights file change (`.mlmodelc` replaced).
- Any hand-tunable parameter change (weights, buckets, strictness, logitScale, prompts, normMode).
- Architecture change (Python or Swift code path that affects scoring).

**Version format:** `v<major>.<minor>.<patch>@<git-sha-short>`, e.g. `v0.3.1@abc1234`.
- **Minor** bumps: ensemble membership change.
- **Patch** bumps: param retune, same models.
- **SHA**: git SHA at eval time, auto-captured.

Params commit to repo → param change = git commit → SHA change. `run.py` reads current params + models + SHA and computes the version automatically. No manual tag drift.

## Leaderboard

`testing/bench/LEADERBOARD.md`, auto-generated, committed. Sorted by val Spearman desc.

```
| Version       | Date       | Spearman | MAE  | ±1 acc | Models         | Notes           |
|---------------|------------|----------|------|--------|----------------|-----------------|
| v0.3.0@ab1234 | 2026-04-15 | 0.67     | 0.82 | 89%    | tech+aes       | dropped CLIP    |
| v0.2.1@de5678 | 2026-04-14 | 0.63     | 0.91 | 86%    | tech+aes+clip  | retune weights  |
```

Regressions (Spearman worse than predecessor) marked with ⚠️.

## Integration back to the app

`params.current.json` is the single source of truth for shipping parameters.

**Build-time bridge:**
- `scripts/gen_defaults.py` reads `params.current.json` and writes `ImageRater/App/FocalSettings+Generated.swift`:
  ```swift
  extension FocalSettings {
      static let generatedWeightTechnical: Double = 0.38
      static let generatedWeightAesthetic:  Double = 0.47
      // ...
  }
  ```
- `FocalSettings.defaultWeightTechnical` (etc.) point at the generated constants. User preferences still override at runtime.
- `project.yml` gets a build phase that runs `gen_defaults.py` before compile.
- Currently hardcoded values (bucket edges, logitScale) are extracted into `FocalSettings` so the generator can drive them.

## Workflow

1. Collect/extend dataset (AVA first; later, optional user-labeled set for taste calibration).
2. `./testing/bench/run.py score` — Swift scorer emits sub-scores, cached by content hash.
3. `./testing/bench/run.py optimize` — Optuna tunes params, writes `best_params.json`.
4. `./testing/bench/run.py eval --params candidate.json` — runs metric suite.
5. If improved: promote candidate → `params.current.json`, bump version, commit.
6. `./testing/bench/run.py leaderboard` — regenerates markdown.
Regression tracking is manual via the leaderboard + git log. No CI gate.

## Phased delivery

Build simplest end-to-end slice first. Validate locally before expanding.

### Phase 0 — MVP smoke test (this plan)

Goal: run defaults against a small AVA subset end-to-end, print metrics. Prove the pipeline works before investing in tuning infrastructure.

1. `testing/bench/dataset_ava.py` — download + parse AVA; **500-image stratified subset** (not full 20k).
2. `FocalScorer` Swift CLI target — decodes images (LibRaw) + runs the 3 `.mlmodelc` models, emits per-image sub-scores JSON. Reuses `RatingPipeline.loadBundledModels()`.
3. `testing/bench/score.py` — invokes `FocalScorer`, caches JSON output.
4. `testing/bench/ensemble.py` — mirrors Swift combining + percentile bucket logic (`combinedQuality` + `percentileStars`).
5. `testing/bench/metrics.py` — Spearman, Kendall, MAE, off-by-one, exact-match, 5×5 confusion.
6. `testing/bench/run.py` — one-command eval with default params; prints metrics to stdout.

**Success criteria:** `run.py` runs locally without errors, reports Spearman for defaults `(wTech=0.4, wAes=0.4, wClip=0.2, strictness=0.5)`. Sanity baseline for whether the current ensemble correlates with AVA at all.

### Phase 1 — optimizer + leaderboard

- `testing/bench/optimize.py` — Optuna TPE param search over the tuning surface.
- `testing/bench/leaderboard.py` + `LEADERBOARD.md` generation, committed + auto-regenerated.
- Scale up AVA subset to full 20k eval split.

### Phase 2 — ablation + model swaps

- Ensemble membership sweep across the 7 subset configurations.
- Candidate swaps (NIMA first, then MUSIQ / MANIQA / HyperIQA as needed).

### Phase 3 — param bridge to shipping app

- `params.current.json` as single source of truth.
- `scripts/gen_defaults.py` writes `ImageRater/App/FocalSettings+Generated.swift`.
- xcodegen build phase wires it in.
- Extract hardcoded `bucketEdges` and `clipLogitScale` into `FocalSettings`.

## Open questions / deferred

- User-taste calibration: how to collect a labeled set of the user's own photos for personal-preference tuning. Architecture supports it (swap dataset loader); UX/collection mechanism deferred.
- MUSIQ / MANIQA Core ML conversion: effort unknown; only undertaken if NIMA swap doesn't hit the target.
- Model file hashing for version detection: currently uses git SHA of `.mlmodelc` directory; may need dedicated hash if models are downloaded at runtime rather than bundled.
