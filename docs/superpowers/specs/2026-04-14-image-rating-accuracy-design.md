# Image Rating Accuracy & Consistency — Design Spec
Date: 2026-04-14

## Problem

Current NIMA-based pipeline has three gaps:
1. **Low accuracy** — NIMA gets SRCC ~0.612 on AVA aesthetic benchmark; better models exist
2. **No variety control** — high-quality near-duplicate shots all score equally; user ends up with 10 of the same frame at 5★
3. **Scores opaque** — UI shows only final star rating; no visibility into technical vs aesthetic vs similarity breakdown

User shoots mixed content (portraits, landscapes, events, travel). Needs one-pass rating with consistent scale across shoot types, best-within-shoot awareness, and variety in the selected set.

---

## Goals

- Replace NIMA with SOTA ensemble (3 models, all CoreML-convertible)
- Expose technical, aesthetic, semantic scores separately in UI
- Automatically penalize near-duplicate shots via diversity scoring
- Consistent star scale across sessions via percentile normalization
- Quick-filter sidebar for slicing by any characteristic

---

## Model Ensemble

Three ML models run `async let` concurrently per image alongside a fourth concurrent Vision/CI task for cull scores (inference only — models pre-loaded before the loop; see Pipeline Flow), replacing the current sequential NIMA pair.

| Model | Role | Benchmark | Input Resolution | CoreML Feasibility |
|-------|------|-----------|------------------|--------------------|
| **TOPIQ-NR** (ResNet50, IEEE TIP 2024) | Technical quality (blur, noise, exposure, artifacts) | SRCC 0.928 KonIQ-10k | 224×224 | Excellent — ResNet50 traces cleanly |
| **TOPIQ-Swin** (Swin Transformer) | Aesthetic quality (composition, color, feel) | SRCC 0.791 AVA | 384×384 (confirm against pyiqa checkpoint) | Excellent — Swin traces with static input |
| **CLIP-IQA+** (antonym prompt pairs) | Semantic/content quality prior | SRCC 0.909 KonIQ | 224×224 (CLIP standard) | Zero-cost — reuses existing `clip.mlpackage` |

**vs current:** NIMA-aesthetic gets 0.612 SRCC on AVA. TOPIQ-Swin gets 0.791. ~29% relative improvement.

**CLIP-IQA+ implementation:** Encode antonym text pair ("Good photo" / "Bad photo"), compute softmax of cosine similarities with image embedding. No new weights — same `clip.mlpackage`, different text inputs in `RatingPipeline.swift`.

**Score normalization for display:** All three models output raw scores in [0, 1] after CoreML conversion (pyiqa convention). For display in the UI, multiply by 10 to produce a [0, 10] scale. Store raw [0, 1] values in CoreData; apply ×10 display transform in `DetailView`.

**Combined quality score (on [0, 1] raw scale):**
```
combined_quality = 0.4 × topiq_technical + 0.4 × topiq_aesthetic + 0.2 × clip_iqa
```
Weights user-configurable in app settings (save to UserDefaults). Default: 0.4 / 0.4 / 0.2.

**Model delivery:** All three models bundled directly in the app binary as Xcode resources. No download, no manifest, no network dependency. `topiq-nr.mlpackage`, `topiq-swin.mlpackage`, and the existing `clip.mlpackage` are added to the Xcode target's Copy Bundle Resources phase. `ModelStore` is not involved for these models. App updates deliver model updates.

---

## Diversity Scoring

### Similarity Clustering

CLIP embeddings (512-dim float32) computed per image during CLIP-IQA+ scoring — no extra inference cost. Cosine similarity threshold: **0.92** (tunable). Images with similarity > threshold grouped into same cluster.

### Diversity Strategy

**MMR (Maximal Marginal Relevance)** used for all session sizes — greedy O(nK), trivially implementable in Swift with no external dependencies, and sufficient for the variety goal stated.

```
score_i = λ × combined_quality(i) - (1-λ) × max_j∈selected sim(i, j)
λ = 0.6  (quality-vs-diversity balance, configurable)
```

After MMR ordering, assign cluster rank and diversity factor:

| Cluster rank | Diversity factor |
|---|---|
| 1–2 | 1.0 (full score) |
| 3 | 0.85 |
| 4 | 0.70 |
| 5+ | 0.55 |

**Final score (stored as `finalScore`):**
```
final_score = combined_quality × diversity_factor
```

### Session Normalization

After diversity scoring, percentile-normalize `finalScore` within the session. Map to consistent star thresholds, written to `ratingStars` (existing CoreData field):

| Percentile | Stars |
|---|---|
| Top 5% | ★★★★★ |
| 5–20% | ★★★★ |
| 20–50% | ★★★ |
| 50–80% | ★★ |
| Bottom 20% | ★ |

`ratingStars` = AI-assigned star from percentile normalization. `combinedQualityScore` and `finalScore` stored as absolute floats alongside, enabling cross-session comparison without renormalization.

---

## Data Model Changes

New fields on `ImageRecord` (Core Data lightweight migration):

```swift
// Ensemble scores (new)
var topiqTechnicalScore: Float    // raw [0,1]; replaces NIMA technical
var topiqAestheticScore: Float    // raw [0,1]; replaces NIMA aesthetic
var clipIQAScore: Float           // raw [0,1]; new
var combinedQualityScore: Float   // weighted ensemble, pre-diversity, raw [0,1]
var finalScore: Float             // combinedQualityScore × diversityFactor, raw [0,1]
var diversityFactor: Float        // 0.55–1.0

// Diversity (new)
var clipEmbedding: Data           // 512-dim float32 blob for MMR similarity
var clusterID: Int32              // similarity cluster index
var clusterRank: Int32            // rank within cluster (1 = best)
// Note: clusterSize derived on-demand via CoreData count predicate on clusterID
// (not stored to avoid denormalization)

// Cull characteristics (store instead of discard)
var blurScore: Float              // Sobel edge variance (was computed but not persisted)
var exposureScore: Float          // EV bias float (was binary decision, now scalar; see CullPipeline changes)

// Legacy (keep nullable for backward compat)
var aestheticScore: Float?        // old NIMA aesthetic
var technicalScore: Float?        // old NIMA technical

// Existing field (unchanged)
// var ratingStars: Int16?        // AI star rating, written by percentile normalization
// var userOverride: NSNumber?    // manual override; unchanged
```

**`processState` during two-pass execution:**
- After pass 1 (per-image inference): `processState = "rated"` (new intermediate state)
- After pass 2 (session-level clustering + diversity): `processState = "done"` (existing state)
- UI shows "Scoring…" during pass 1, "Ranking variety…" during pass 2

---

## Pipeline Flow

Two-pass execution. All three models pre-loaded sequentially before the per-image loop (same pattern as current NIMA loading in `ProcessingQueue`), preventing serial model compilation inside the concurrent inference blocks.

```
Pre-load models (sequential, once per session):
  topiq-nr.mlpackage → compiled MLModel
  topiq-swin.mlpackage → compiled MLModel
  clip.mlpackage → compiled MLModel (already loaded if CLIP-IQA+ was prior)
     ↓
Pass 1 — Per-image (progress: "Scoring X of N")
  [async let — 3 ML inferences + Vision/CI, all concurrent per image]
    TOPIQ-NR      → topiqTechnicalScore [0,1]
    TOPIQ-Swin    → topiqAestheticScore [0,1]
    CLIP          → clipEmbedding (512-dim) + clipIQAScore [0,1]
    Vision/CI     → blurScore (Sobel variance) + exposureScore (EV bias float)
  combined_quality = 0.4×tech + 0.4×aes + 0.2×clip
  Write: topiqTechnicalScore, topiqAestheticScore, clipIQAScore,
         combinedQualityScore, clipEmbedding, blurScore, exposureScore
  processState = "rated"
     ↓
Pass 2 — Session-level (progress: "Ranking variety…")
  Load all clipEmbeddings from CoreData
  Step A — threshold clustering (cosine sim > 0.92):
    → clusterID per image
  Step B — MMR (λ=0.6) on combined_quality + cosine similarity within clusters:
    → selection order → clusterRank per image (position in MMR order within cluster)
    → diversityFactor per image (from rank table)
  finalScore = combinedQualityScore × diversityFactor
  Percentile-normalize finalScore within session
    → ratingStars (Int16, written to existing field)
  Write: clusterID, clusterRank, diversityFactor, finalScore, ratingStars
  processState = "done"
     ↓
Update UI
```

---

## CullPipeline Changes

`CullPipeline` currently returns only `CullResult` (accept/reject enum). To store numeric scores, the return type changes:

```swift
struct CullScores {
    let result: CullResult      // existing accept/reject decision
    let blurScore: Float        // Sobel edge variance (higher = sharper); maps to ImageRecord.blurScore
    let exposureScore: Float    // EV bias float: 0.0 = neutral, +1.0 = 1EV over, -1.0 = 1EV under; maps to ImageRecord.exposureScore
}
```

`exposureScore` is derived from the existing histogram fractions: map `overFraction` and `underFraction` to a signed float. Re-enable the exposure check with a configurable threshold (e.g. `|exposureScore| > 1.5` = reject). The `blurScore` is the existing Sobel variance already computed in `checkBlur()`.

---

## UI Changes

### Detail Sidebar — Scores Panel

Scores displayed scaled to [0, 10] (raw × 10):

```
Technical    [████████░░] 7.8
Aesthetic    [███████░░░] 6.9
Semantic     [██████░░░░] 6.1
─────────────────────────────
Combined AI  8.3  ★★★★★
             (manual override: ★★★★)
```

### Detail Sidebar — Characteristics Panel

Cluster size derived via CoreData count predicate on `clusterID` at view render time:

```
Blur         Sharp
Exposure     +0.3 EV (normal)
Cluster      #4 · rank 2 of 7
Diversity    0.85× (rank 2 penalty)
```

### Grid Badges

- Star rating (existing, fixed thresholds)
- "C" badge on cluster representative (`clusterRank == 1`)
- Dim overlay (30% opacity) on `diversityFactor < 0.60`

### Filter Sidebar (extends existing `RatingFilterView`)

Extends the filter sidebar from `2026-04-14-rating-filter-sidebar-design.md`. New controls added to `RatingFilterView` below the existing star-toggle rows. Slider state stored in `ContentView` alongside `ratingFilter: Set<Int>`, persisted to `UserDefaults` per session.

```
Stars        [★][★★][★★★][★★★★][★★★★★]   ← existing

Technical    [====|----] 6.0+              ← new sliders (display scale 0–10)
Aesthetic    [====|----] 5.0+
Blur         [toggle] Hide blurry
Exposure     [toggle] Hide bad exposure

Diversity
  [toggle] Cluster reps only  (clusterRank == 1)
  [toggle] Variety set        (highest ratingStars per clusterID)
```

"Variety set" computes best per cluster from CoreData, not a stored field.

---

## Files to Create / Modify

| File | Change |
|------|--------|
| `RatingPipeline.swift` | Replace NIMA with TOPIQ-NR + TOPIQ-Swin + CLIP-IQA+; pre-load models before loop; `async let` inference per image; return `RatingResult` with 3 scores + embedding |
| `DiversityScorer.swift` | New: MMR implementation; cluster assignment; diversity factor computation |
| `ProcessingQueue.swift` | Two-pass architecture; new `"rated"` processState; pass 2 loads embeddings from CoreData, runs DiversityScorer, writes final fields |
| `CullPipeline.swift` | Return `CullScores` struct (blurScore + exposureScore + CullResult); re-enable exposure check with configurable threshold |
| `ImageRecord+CoreData.swift` | Add new fields per Data Model section; lightweight migration |
| `DetailView.swift` | Add scores panel (×10 display transform) + characteristics panel; cluster size via count predicate |
| `GridView.swift` | Add C badge, dim overlay for `diversityFactor < 0.60` |
| `RatingFilterView.swift` | Add score sliders, blur/exposure toggles, diversity toggles (extends existing spec) |
| `ContentView.swift` | Add new filter state vars alongside `ratingFilter`; persist to UserDefaults |
| `project.yml` | Add `topiq-nr.mlpackage` and `topiq-swin.mlpackage` to Copy Bundle Resources |
| `convert_topiq.py` | New: convert TOPIQ-NR + TOPIQ-Swin from pyiqa to CoreML; verify input resolution (224×224 NR, confirm Swin); output [0,1] score range |

---

## Out of Scope

- Real-time rating during import (batch-only)
- Scene classification (portrait vs landscape weights) — YAGNI; TOPIQ handles mixed content well
- k-DPP for theoretically optimal diversity — MMR sufficient for stated goals, avoids O(n³) eigendecomposition
- UNIAA / AesMamba — CoreML conversion not feasible
- User-trainable models — out of scope v1
