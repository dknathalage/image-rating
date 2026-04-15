# MUSIQ-AVA CoreML Conversion Design

**Status:** Draft
**Date:** 2026-04-15
**Depends on:** `2026-04-15-rating-benchmark-tuning-design.md` (v0.3.0 bench proof that MUSIQ-alone ≥ ensemble)

## Goal

Replace Focal's three-model rating ensemble (TOPIQ-NR + TOPIQ-Swin + CLIP-IQA) with a single MUSIQ-AVA CoreML model, matching the Python bench's Spearman 0.76 / ±1 acc 91% on AVA. Hard cut-over — no A/B flag, old models and ensemble code deleted.

## Decisions (from brainstorming)

| # | Decision | Rationale |
|---|---|---|
| Q1 | **Hard replace** old scorer | v0.3.0 bench proves single-model parity; dual-path dead weight. Rollback = revert commit. |
| Q2 | **Near-exact parity** (Spearman ≥ 0.97 vs pyiqa on 100-image sample, max \|Δ\| ≤ 0.1) | Bucket thresholds sit ~0.45 apart on raw MUSIQ scale; drift > 0.2 flips borderline ratings. fp32 avoids the risk for ~30% latency cost. |
| Q3 | **A1 — Swift preprocess, transformer-only CoreML** | Static patch-tensor input `[1, 193, 3075]` plays well with Apple Neural Engine; preserves MUSIQ's multi-scale accuracy (both 224 + 384); preprocessing stays inspectable Swift. A2 drops a scale (loses ρ ≥ 0.03); A3 forces dynamic shapes (NE fallback to CPU). |

## Architecture

```
CGImage
 └─> MUSIQPreprocessor.patchTensor(image)   [Swift, pure fn]
       └─> CoreML MUSIQ (musiq-ava.mlmodelc) [inference, fp32]
             └─> Float MOS (raw, ~[3.4, 7.0])
                   └─> RatingPipeline.bucketStars(mos)
                         └─> Int star ∈ 1...5
```

One-way pipeline; no cycles; no cross-module mutable state. Ensemble scaffolding (`combinedQuality`, `clipIQAScore`, CLIP prompt embeddings, per-subscore weights, bucket edges in normalized space) is deleted.

Bench is the single source of truth for bucket thresholds. `scripts/gen_defaults.py` reads `testing/bench/params.current.json` at build time and generates Swift constants into `ImageRater/App/FocalSettings+Generated.swift`.

## Components

### New

- **`scripts/convert_musiq.py`** — offline PyTorch → CoreML converter.
  - Loads pyiqa MUSIQ-AVA checkpoint.
  - Wraps model body (embedding + 14-layer transformer encoder + head + `dist_to_mos`) in a `nn.Module` that accepts static-shape patch tensor `[1, 193, 3075]`, strips the built-in `get_multiscale_patches` call.
  - `torch.jit.trace` with example input.
  - `coremltools.convert(..., inputs=[TensorType(shape=(1,193,3075))], compute_precision=FP32, minimum_deployment_target=macOS14)`.
  - Writes `ImageRater/MLModels/musiq-ava.mlpackage` (Xcode compiles to `.mlmodelc` at build time).
  - Deterministic: `torch.manual_seed(0)`, model in `eval()`, no dropout.

- **`ImageRater/Pipeline/MUSIQPreprocessor.swift`** — pure preprocessing functions. Mirrors `pyiqa/data/multiscale_trans_util.py` line-by-line.
  - `normalize(_ cg: CGImage) -> [Float]` — sRGB uint8 → planar float `[H·W·3]` in `[-1, 1]`.
  - `aspectResize(_ pixels: [Float], h: Int, w: Int, longerSide: Int) -> (pixels: [Float], rh: Int, rw: Int)` — bicubic, aspect-preserved.
  - `unfoldPatches(_ pixels: [Float], h: Int, w: Int, patch: Int = 32) -> (patches: [Float], countH: Int, countW: Int)` — TF-SAME padding + 32×32 unfold, row-major.
  - `hashSpatialPositions(countH: Int, countW: Int, gridSize: Int = 10) -> [Float]` — nearest-interp over `0..<gridSize`, hashes `h·gridSize + w`.
  - `padToMaxSeqLen(_ rows: [Float], cols: Int, activeN: Int, maxSeqLen: Int) -> [Float]` — append zero rows.
  - `patchTensor(_ cg: CGImage) -> MLMultiArray` — orchestrator. Runs two scales (224, 384), concats, returns shape `[1, 193, 3075]` float32.
  - Throws `RatingError.imageTooSmall` if longer side < 32.

- **`testing/bench/bench/parity.py`** (helper) + `testing/bench/tests/test_musiq_parity.py` — Python side of parity validation.
  - `pyiqa_vs_coreml(image_dir, coreml_scores_csv) -> (spearman, max_abs_delta, mean_abs_delta)`.
  - Gate: Spearman ≥ 0.97, max \|Δ\| ≤ 0.1, mean \|Δ\| ≤ 0.03 on AVA-100 sample (first 100 images from `testing/bench/data/ava/images/`).

### Rewritten

- **`ImageRater/Pipeline/RatingPipeline.swift`**
  - `BundledModels` collapses to `{musiq: MLModel}`. `loadBundledModels()` loads single model.
  - `rate(image:models:)` → preprocessor → CoreML → `RatedScores`. No weights arg.
  - `bucketStars(mos: Float, thresholds: (Float, Float, Float, Float)) -> Int` — pure fn; reads thresholds from `FocalSettings`.
  - Drop: `combinedQuality`, `clipIQAScore`, `inferEmbedding`, `extractFloatArray`, CLIP-IQA logic.

- **`ImageRater/Models/RatingResult.swift`**
  - `RatedScores = { musiqAesthetic: Float, stars: Int }`. Previous fields (`topiqTechnicalScore`, `topiqAestheticScore`, `clipIQAScore`, `combinedQualityScore`, `clipEmbedding`) deleted.

- **`FocalScorer/Scorer.swift`**
  - `OutputImage = { filename: String, musiqAesthetic: Float }`.
  - `Output.modelVersion = "musiq-ava"`.
  - Bench's `score_with_cache` in `testing/bench/bench/score.py` already tolerates schema change because it just `pd.DataFrame(blob["images"])`; downstream consumers are rewritten (below).

- **`ImageRater/App/FocalSettings.swift`** / `FocalSettings+Generated.swift`
  - Delete: `defaultWeightTechnical`, `defaultWeightAesthetic`, `defaultWeightClip`, `defaultCullStrictness`, `defaultBucketEdge1..4`, `defaultClipLogitScale`, and matching `generated…` fields.
  - Add: `defaultMUSIQThreshold1..4 = generatedMUSIQThreshold1..4`.
  - Keep UI-only: `defaultCellSizeValue`, `defaultAutoWriteXMP`.

- **`scripts/gen_defaults.py`**
  - Read new schema: `{version, model, thresholds: [t1..t4]}`.
  - Emit `generatedVersion`, `generatedMUSIQThreshold1..4` (4 constants).
  - Delete old generated constants.

- **`testing/bench/params.current.json`** (schema v0.4.0)
  ```json
  {
    "version": "v0.4.0",
    "date": "2026-04-15",
    "model": "musiq-ava",
    "notes": "single-model rating; CoreML on-device",
    "thresholds": [4.465, 5.181, 5.634, 6.068]
  }
  ```

- **`testing/bench/run.py`**
  - `cmd_eval`: read `thresholds`, apply `bucketStars`, compute metrics.
  - `cmd_optimize`: tune 4 thresholds only (objective `-spearman + 0.2·mae`), writes v0.4.0 schema.
  - `cmd_ablate`: delete (single-model has nothing to ablate).
  - `_scores_with_clip_scalar`: delete.
  - `_load_dataset`: keeps MUSIQ path (already integrated via `musiq_scorer.py`).

### Deleted

- `ImageRater/MLModels/topiq-nr.mlmodelc`
- `ImageRater/MLModels/topiq-swin.mlmodelc`
- `ImageRater/MLModels/clip-vision.mlmodelc`
- `ImageRater/Pipeline/CLIPTextEmbeddings.swift`
- `testing/bench/bench/clip_iqa.py` + `testing/bench/tests/test_clip_iqa.py`
- `testing/bench/bench/ensemble.py` + `testing/bench/tests/test_ensemble.py`
- `testing/bench/bench/ablation.py` + `testing/bench/tests/test_ablation.py`
- `testing/bench/bench/prompt_embeddings.json`
- `FocalScorer` CLIP code in `Scorer.swift` OutputImage
- Any `RatingPipeline.combinedQuality` / `clipIQAScore` unit tests in `FocalTests/`

### Touched (minor)

- `ImageRater/Pipeline/RatingQueue.swift` — drop weights parameter from `rate()` call.
- `ImageRater/UI/DetailView.swift` — remove any `topiqTechnical/Aesthetic/clip` field reads, replace with `musiqAesthetic` + `stars`.
- `ImageRater/Export/MetadataWriter.swift` — if it embeds subscores in XMP, switch to single score + stars.
- `project.yml` — update `FocalScorer` target sources (drop `CLIPTextEmbeddings.swift`), update bundled model list.

## Data flow (preprocessing math)

**Step 1 — Pixel → float, normalize to `[-1, 1]`:**
```
CGImage (sRGB uint8) → CVPixelBuffer BGRA → planar RGB [H, W, 3]
pixel = (pixel / 255.0 - 0.5) * 2    # matches pyiqa MUSIQ.forward line 412
```

**Step 2 — For each scale in `[224, 384]` (scale_id ∈ {0, 1}):**

a. **Aspect-preserving bicubic resize:**
```
ratio = scale / max(h, w)
rh = round(h · ratio);  rw = round(w · ratio)
# interpolation = bicubic, align_corners = False (matches F.interpolate)
```

b. **Unfold 32×32 patches, stride 32, TF-SAME padding:**
```
count_h = ceil(rh / 32);  count_w = ceil(rw / 32)
pad_h = (count_h - 1)·32 + 32 - rh
pad_w = (count_w - 1)·32 + 32 - rw
top    = pad_h // 2;  bottom = pad_h - top
left   = pad_w // 2;  right  = pad_w - left
unfold(padded, kernel=32, stride=32) → [count_h·count_w, 3072]   # C=3
```
Row-major over patches, column-major within each patch (matches PyTorch `F.unfold`).

c. **Hash spatial position indices (grid_size=10):**
```
pos_h = nearest_interp([0..9], count_h) → [count_h]
pos_w = nearest_interp([0..9], count_w) → [count_w]
spatial_p[i*count_w + j] = pos_h[i] * 10 + pos_w[j]      # 0..99
spatial_p ∈ ℝ^(count_h·count_w)
```

d. **Scale and mask vectors:**
```
scale_p = [scale_id] · (count_h·count_w)
mask_p  = [1.0]      · (count_h·count_w)
```

e. **Concat row-wise:** `[patches | spatial_p | scale_p | mask_p]` → `[N, 3075]`, N = count_h·count_w.

f. **Pad to `max_seq_len = ceil(scale / 32)² `:**
```
224 → max_seq_len = 49
384 → max_seq_len = 144
```
Pad with zero rows (patches zero, pos zero, scale zero, mask zero → attention ignores).

**Step 3 — Concat scales:** `[49, 3075] ‖ [144, 3075]` → `[193, 3075]`, prepend batch dim → `[1, 193, 3075]`.

**Step 4 — CoreML inference:**
```
MLMultiArray float32 shape [1, 193, 3075] → model.prediction → mos: Float
```

**Step 5 — Bucket to stars:**
```
if      mos ≤ t1: 1
else if mos ≤ t2: 2
else if mos ≤ t3: 3
else if mos ≤ t4: 4
else:             5
```

**Parity-critical details (Swift must match Python exactly):**

- **Bicubic resize.** PyTorch `F.interpolate(mode='bicubic', align_corners=False)` uses cubic convolution with a=-0.5. `vImageScale_ARGB8888` + `kvImageHighQualityResampling` approximates Lanczos, not cubic; `CIFilter.bicubicScaleTransform` uses CoreImage's kernel with its own colorspace pipeline (unpredictable gamma). **Decision: ship a custom Swift bicubic kernel** (cubic convolution with a=-0.5, matching PyTorch's `F.interpolate` exactly). Deterministic, no colorspace surprises, validated against Python reference in preprocessor parity test. Avoid CIFilter / vImage.
- **Rounding.** PyTorch `round(x)` = banker's (nearest-even). Swift `Double.rounded()` default = `.toNearestOrEven`. Confirm both match for ratios like 0.333·1200 = 399.6 → 400 in both.
- **TF-SAME pad asymmetry.** Bottom/right get the extra pixel when pad count is odd. Confirmed from pyiqa source.
- **Patch ordering.** `F.unfold` emits `[C, kH, kW, N]` → reshape `[N, C·kH·kW]`. Each patch's values = `[C0 row0 col0..31, C0 row1 col0..31, …, C2 row31 col31]`. Swift must match.
- **Color space.** pyiqa reads via `torchvision.io.read_image` (no gamma decode) → `tensor / 255`. Swift via `CGImage` is already sRGB with no implicit linearization (CGContext draw straight copy to BGRA uint8). Direct `/255 * 2 - 1` matches.

## Error handling

| Class | Condition | Behavior |
|---|---|---|
| **Bundle** | `musiq-ava.mlmodelc` missing | `RatingError.modelNotFound("musiq-ava")` at app launch. Build-config bug; fail loud. |
| **Build** | `params.current.json` missing during `gen_defaults.py` | Exit non-zero; Xcode preBuildScript halts. |
| **Conversion** | `scripts/convert_musiq.py` fails | Script exits non-zero. Developer workflow only. |
| **Input** | `CGImage` decode fails (RAW) | `LibRawWrapper.decode` returns nil → `Scorer.scoreDirectory` skips with log. Unchanged. |
| **Input** | Longer side < 32px | `RatingError.imageTooSmall`; `rate()` returns `.unrated`. |
| **Input** | `CVPixelBufferCreate` fails | `RatingError.pixelBufferCreationFailed`; returns `.unrated`. Existing. |
| **Inference** | CoreML `prediction` throws | Catch in `rate()`, log, return `.unrated`. Existing pattern. |
| **Inference** | Output not scalar or empty | `RatingError.inferenceOutputMismatch`; returns `.unrated`. |
| **Inference** | MOS is NaN or Inf | Warning log + `.unrated`. Guards fp16 overflow (future-proof; fp32 current build). |
| **Post** | MOS outside `[2.0, 8.0]` | Clamp + warn; continue bucketing. Monotonic bucket → lands in 1★ or 5★. |

**Out of scope:**
- Graceful degradation to old ensemble: no fallback path (hard-replace, Q1).
- Runtime model-version check: `generatedVersion` baked into bundle.
- Per-channel preprocessor validation at runtime: parity tests cover once during development.

**Parity development gate (not a runtime error class):**
If `testing/bench/tests/test_musiq_parity.py` reports Spearman < 0.97 or max \|Δ\| > 0.1 on AVA-100 sample, CI blocks merge.

## Testing

### Swift unit tests (`FocalTests/`)

| Test | Assertion |
|---|---|
| `MUSIQPreprocessor.resize_224_portrait` | 1200×800 + longerSide=224 → (224, 149); corners + center pixel values match Python reference ±1/255. |
| `MUSIQPreprocessor.resize_384_landscape` | 800×1200 + longerSide=224 → (149, 224); longerSide=384 → (256, 384). |
| `MUSIQPreprocessor.unfoldPatches_exact` | Synthetic 3×64×64 with known pixels → verify 4 patches (count=2×2), ordering + contents match `F.unfold(kernel=32, stride=32)`. |
| `MUSIQPreprocessor.hashSpatialPositions_7x5` | count_h=7, count_w=5, grid=10 → exact integer sequence matches pyiqa `get_hashed_spatial_pos_emb_index`. |
| `MUSIQPreprocessor.patchTensor_shape` | Any input → `MLMultiArray` shape `[1, 193, 3075]`, dtype float32. |
| `MUSIQPreprocessor.padToMaxSeqLen` | Post-resize tensor with count_h=count_w=7 (scale=224, 7×7=49 patches) → 49 active, 0 pads. Post-resize count_h=count_w=4 (e.g. synthetic test input pre-sized to 128×128) → 16 active, 33 zero pads (pad rows sum to 0). Test takes pre-resized tensor as input to isolate padding logic from resize. |
| `MUSIQPreprocessor.imageTooSmall_throws` | 20×30 image → throws `RatingError.imageTooSmall`. |
| `RatingPipeline.bucketStars_boundaries` | score=t1-ε → 1★; t1+ε → 2★; … t4+ε → 5★. |
| `RatingPipeline.bucketStars_extremes` | score=1.0 → 1★; score=10.0 → 5★. |
| `RatingPipeline.rate_unrated_on_nan` | Mocked model returning NaN → `.unrated`. |
| `RatingPipeline.rate_unrated_on_inf` | Mocked model returning Inf → `.unrated`. |

### Python parity tests (`testing/bench/tests/test_musiq_parity.py`)

| Test | Assertion |
|---|---|
| `test_convert_musiq_deterministic` | Running `convert_musiq.py` twice produces identical model spec. Hash the MIL protobuf (`mlmodel.get_spec().SerializeToString()` SHA-256), not the full `.mlpackage` directory — coremltools embeds timestamps/UUIDs in manifests. Catches dropout-at-trace bugs without false positives. |
| `test_pyiqa_vs_coreml_100image_sample` | AVA-100 sample: Spearman(pyiqa, coreml) ≥ 0.97; max \|Δ\| ≤ 0.1; mean \|Δ\| ≤ 0.03. **Merge gate.** |
| `test_pyiqa_vs_swift_preprocessor_patch_tensor` | On 5 images: \|patch_tensor_python - patch_tensor_swift\|_∞ ≤ 0.005. Catches bicubic / padding divergence before it hits inference. |

Parity driver: Python script invokes `FocalScorer` CLI on `testing/bench/data/ava/images/` sample, loads JSON, compares to cached pyiqa scores. Non-blocking on non-Apple-Silicon CI runners.

### Bench regression tests (`testing/bench/tests/test_run.py`)

| Test | Assertion |
|---|---|
| `run.py eval` using CoreML-scored JSON | Spearman ≥ 0.70 vs gt_stars on shipped 500-AVA. |
| `run.py optimize` with thresholds-only objective | Reproduces Spearman ≈ 0.764 (matches pre-CoreML single-model number within noise). |
| `test_run_single_model_schema` | `params.current.json` v0.4.0 loads; no `w_tech`/`w_aes`/`w_clip`/`clip_logit_scale`/`bucket_edges`/`strictness` keys. |

### Deleted tests

- `testing/bench/tests/test_ensemble.py`
- `testing/bench/tests/test_ablation.py`
- `testing/bench/tests/test_clip_iqa.py`
- Any `test_combined_quality` / `test_clip_iqa_score` in `FocalTests/`

### Manual checkpoint (before PR merge)

1. Build + run Focal app on a real import batch (≥ 50 photos) on Apple Silicon Mac.
2. Bulk import time < 2× current ensemble baseline.
3. Star distribution visually sane (not all clustered at 3★).
4. Spot-check: pick 10 photos; app's stars == `run.py eval` stars on same files.

## Success criteria

- CoreML MUSIQ on-device Spearman ≥ 0.97 vs pyiqa reference on AVA-100.
- Max \|Δ\| per image ≤ 0.1 raw MUSIQ score.
- Focal app rating pipeline: ≤ 2× current ensemble per-image latency.
- All deleted code deleted (no dead files, no unreferenced symbols).
- Bench `run.py` round-trips v0.4.0 schema without touching ensemble code.
- Zero dev-mode fallbacks: no flag toggles, no `if oldPipeline`, no vestigial ensemble types in `RatingResult`.
