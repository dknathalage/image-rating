# MUSIQ-AVA CoreML Conversion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Focal's three-model ensemble (TOPIQ-NR + TOPIQ-Swin + CLIP-IQA) with a single MUSIQ-AVA CoreML model that matches the Python bench's Spearman 0.76 on AVA, hard cut-over, no A/B flag.

**Architecture:** Offline PyTorch → CoreML conversion (`scripts/convert_musiq.py`) produces `musiq-ava.mlpackage` with static patch-tensor input `[1, 193, 3075]`. Swift `MUSIQPreprocessor` mirrors pyiqa preprocessing exactly (custom bicubic, TF-SAME padding, hash positions). `RatingPipeline` calls preprocessor → CoreML → bucketStars. Old ensemble code deleted.

**Tech Stack:** PyTorch 2.11, pyiqa 0.1.15, coremltools 9.0, Swift 5.9, CoreML, XCTest, pytest, Optuna.

**Spec:** `docs/superpowers/specs/2026-04-15-musiq-coreml-conversion-design.md`

---

## Reference Paths

- pyiqa MUSIQ arch: `/opt/homebrew/lib/python3.14/site-packages/pyiqa/archs/musiq_arch.py`
- pyiqa preprocessing: `/opt/homebrew/lib/python3.14/site-packages/pyiqa/data/multiscale_trans_util.py`
- Current ensemble: `ImageRater/Pipeline/RatingPipeline.swift`
- Bench MUSIQ scorer: `testing/bench/bench/musiq_scorer.py`
- AVA 500-image cache: `testing/bench/data/ava/images/`, `testing/bench/data/ava/score_musiq_ava.csv`

## File Inventory

### New

| Path | Purpose |
|------|---------|
| `scripts/convert_musiq.py` | Offline PyTorch → CoreML converter |
| `ImageRater/Pipeline/MUSIQPreprocessor.swift` | Swift preprocessing (resize, unfold, positions) |
| `ImageRaterTests/MUSIQPreprocessorTests.swift` | Preprocessor unit tests |
| `ImageRaterTests/Fixtures/musiq_reference/` | Python-generated reference tensors for parity |
| `testing/bench/bench/parity.py` | Python parity driver helpers |
| `testing/bench/tests/test_musiq_parity.py` | Parity tests + merge gate |
| `testing/bench/tests/fixtures/make_musiq_fixtures.py` | Script that produces Swift test fixtures |

### Modified

| Path | Change |
|------|--------|
| `ImageRater/Pipeline/RatingPipeline.swift` | Rewrite: single model, bucketStars, drop CLIP/ensemble |
| `ImageRater/Models/RatingResult.swift` | `RatedScores = { musiqAesthetic, stars }` |
| `ImageRater/Pipeline/RatingQueue.swift` | Drop weights argument |
| `ImageRater/UI/DetailView.swift` | Read `musiqAesthetic` + `stars` only |
| `ImageRater/Export/MetadataWriter.swift` | XMP: single score + stars |
| `ImageRater/App/FocalSettings.swift` | Drop ensemble defaults, add thresholds |
| `ImageRater/App/FocalSettings+Generated.swift` | Regenerated — 4 threshold constants |
| `FocalScorer/Scorer.swift` | JSON schema: `{filename, musiqAesthetic}` |
| `scripts/gen_defaults.py` | Emit `generatedMUSIQThreshold1..4` only |
| `testing/bench/params.current.json` | Schema v0.4.0 |
| `testing/bench/run.py` | `cmd_eval`/`cmd_optimize` single-model; drop `cmd_ablate` |
| `testing/bench/tests/test_run.py` | Update assertions for v0.4.0 schema |
| `ImageRaterTests/RatingPipelineTests.swift` | Rewrite for single-model path |
| `ImageRaterTests/GeneratedDefaultsTests.swift` | Assert thresholds constants |
| `project.yml` | Drop old .mlmodelc from sources, add musiq-ava.mlpackage |

### Deleted

| Path | Why |
|------|-----|
| `ImageRater/MLModels/topiq-nr.mlmodelc` | Not used |
| `ImageRater/MLModels/topiq-swin.mlmodelc` | Not used |
| `ImageRater/MLModels/clip-vision.mlmodelc` | Not used |
| `ImageRater/Pipeline/CLIPTextEmbeddings.swift` | CLIP-IQA removed |
| `testing/bench/bench/clip_iqa.py` + `tests/test_clip_iqa.py` | CLIP-IQA removed |
| `testing/bench/bench/ensemble.py` + `tests/test_ensemble.py` | Ensemble removed |
| `testing/bench/bench/ablation.py` + `tests/test_ablation.py` | Single model — nothing to ablate |
| `testing/bench/bench/prompt_embeddings.json` | CLIP prompts unused |

---

## Task Order and Dependencies

```
Task 1 (converter) ──┐
                     ├──> Task 7 (preprocessor parity) ──┐
Task 2 (test fixtures)┘                                  │
                                                         │
Task 3 (bicubic) ──> Task 4 (unfold) ──> Task 5 (positions) ──> Task 6 (orchestrator)
                                                                    │
Task 8 (schema v0.4.0) ──> Task 9 (RatingPipeline rewrite) <────────┘
                                                │
                                                v
                              Task 10 (bench run.py rewrite)
                                                │
                                                v
                              Task 11 (caller migration)
                                                │
                                                v
                              Task 12 (delete old code + models)
                                                │
                                                v
                              Task 13 (Xcode bundle integration)
                                                │
                                                v
                              Task 14 (CoreML parity gate)
                                                │
                                                v
                              Task 15 (end-to-end validation)
```

Tasks 1–2 can run in parallel. Tasks 3–6 are sequential (each depends on prior primitives). Task 7 needs Tasks 1, 2, 6.

---

## Task 1: Offline PyTorch → CoreML Converter

**Files:**
- Create: `scripts/convert_musiq.py`
- Create: `testing/bench/tests/test_convert_musiq.py`

Offline developer-only script. Produces `ImageRater/MLModels/musiq-ava.mlpackage` from the pyiqa checkpoint. Trace wraps only the model body (embedding → transformer → head → `dist_to_mos`), skipping `get_multiscale_patches` so the traced graph accepts the Swift-produced patch tensor directly.

- [ ] **Step 1: Write failing determinism test**

```python
# testing/bench/tests/test_convert_musiq.py
"""Converter determinism test — same seed, same MIL proto hash."""
from __future__ import annotations
import hashlib
import subprocess
from pathlib import Path
import coremltools as ct


ROOT = Path(__file__).resolve().parents[3]
CONVERT = ROOT / "scripts" / "convert_musiq.py"
OUT = ROOT / "ImageRater" / "MLModels" / "musiq-ava.mlpackage"


def _spec_hash(pkg: Path) -> str:
    m = ct.models.MLModel(str(pkg), compute_units=ct.ComputeUnit.CPU_ONLY)
    return hashlib.sha256(m.get_spec().SerializeToString()).hexdigest()


def test_convert_musiq_deterministic(tmp_path):
    subprocess.check_call(["python3", str(CONVERT)])
    h1 = _spec_hash(OUT)
    subprocess.check_call(["python3", str(CONVERT)])
    h2 = _spec_hash(OUT)
    assert h1 == h2, f"Converter non-deterministic: {h1[:8]} != {h2[:8]}"
```

- [ ] **Step 2: Run test to verify it fails**

```
cd testing/bench
python3 -m pytest tests/test_convert_musiq.py -v
```

Expected: FAIL with `FileNotFoundError: scripts/convert_musiq.py`.

- [ ] **Step 3: Implement `scripts/convert_musiq.py`**

```python
# scripts/convert_musiq.py
#!/usr/bin/env python3
"""Convert pyiqa MUSIQ-AVA → CoreML .mlpackage with static patch-tensor input.

Output shape: [1, 193, 3075] (49 patches for scale 224 + 144 patches for
scale 384, each row = 3072 pixel values + [spatial_pos, scale_id, mask]).

Writes: ImageRater/MLModels/musiq-ava.mlpackage (committed into Xcode project).
"""
from __future__ import annotations
import sys
from pathlib import Path

import torch
import torch.nn as nn
import coremltools as ct
import numpy as np
import pyiqa


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = REPO_ROOT / "ImageRater" / "MLModels" / "musiq-ava.mlpackage"

# Static shape constants matching the Swift preprocessor.
SEQ_LEN = 49 + 144               # scales [224, 384] → max_seq_len (7²) + (12²)
PATCH_DIM = 32 * 32 * 3          # 3072
ROW_DIM = PATCH_DIM + 3          # 3075 (+ spatial_pos, scale_id, mask)


class MUSIQBody(nn.Module):
    """MUSIQ forward skipping `get_multiscale_patches` (Swift provides patches)."""

    def __init__(self, musiq: nn.Module):
        super().__init__()
        self.patch_size = musiq.patch_size
        self.conv_root = musiq.conv_root
        self.gn_root = musiq.gn_root
        self.root_pool = musiq.root_pool
        self.block1 = musiq.block1
        self.embedding = musiq.embedding
        self.transformer_encoder = musiq.transformer_encoder
        self.head = musiq.head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, seq_len, 3075] — already preprocessed patches.
        b, seq_len, _ = x.shape
        num_crops = 1

        inputs_spatial_positions = x[:, :, -3]
        inputs_scale_positions = x[:, :, -2]
        inputs_masks = x[:, :, -1].bool()
        patches = x[:, :, :-3]

        patches = patches.reshape(-1, 3, self.patch_size, self.patch_size)
        f = self.conv_root(patches)
        f = self.gn_root(f)
        f = self.root_pool(f)
        f = self.block1(f)
        f = f.permute(0, 2, 3, 1).reshape(b, seq_len, -1)
        f = self.embedding(f)
        f = self.transformer_encoder(
            f, inputs_spatial_positions, inputs_scale_positions, inputs_masks
        )
        q = self.head(f[:, 0])
        q = q.reshape(b, num_crops, -1).mean(dim=1)

        # dist_to_mos: E[score] over bins 1..10
        bins = torch.arange(1, q.shape[-1] + 1, dtype=q.dtype, device=q.device)
        mos = (q * bins).sum(dim=-1)
        return mos


def main():
    torch.manual_seed(0)
    np.random.seed(0)

    print("Loading pyiqa MUSIQ-AVA...", flush=True)
    metric = pyiqa.create_metric("musiq-ava", device="cpu", as_loss=False)
    metric.eval()

    body = MUSIQBody(metric.net).eval()

    # Dummy input matching Swift tensor shape.
    dummy = torch.randn(1, SEQ_LEN, ROW_DIM, dtype=torch.float32)
    # Ensure scale_id plausible (0/1) and mask=1 to exercise full graph.
    dummy[:, :49, -2] = 0.0
    dummy[:, 49:, -2] = 1.0
    dummy[:, :, -1] = 1.0

    print("Tracing...", flush=True)
    with torch.no_grad():
        traced = torch.jit.trace(body, dummy, strict=False)

    print("Converting to CoreML (fp32)...", flush=True)
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="patch_tensor", shape=(1, SEQ_LEN, ROW_DIM), dtype=np.float32)],
        outputs=[ct.TensorType(name="mos", dtype=np.float32)],
        compute_precision=ct.precision.FLOAT32,
        minimum_deployment_target=ct.target.macOS14,
        convert_to="mlprogram",
    )
    mlmodel.short_description = "MUSIQ-AVA aesthetic quality predictor (on-device)"
    mlmodel.author = "Focal"

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if OUT_PATH.exists():
        import shutil
        shutil.rmtree(OUT_PATH)
    mlmodel.save(str(OUT_PATH))
    print(f"Wrote {OUT_PATH}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run determinism test**

```
cd testing/bench
python3 -m pytest tests/test_convert_musiq.py -v
```

Expected: PASS. (First run produces the .mlpackage; second run reproduces identical spec proto.)

- [ ] **Step 5: Sanity check output spec**

```
python3 -c "
import coremltools as ct
m = ct.models.MLModel('ImageRater/MLModels/musiq-ava.mlpackage', compute_units=ct.ComputeUnit.CPU_ONLY)
spec = m.get_spec()
print('in:', [(i.name, i.type) for i in spec.description.input])
print('out:', [(o.name, o.type) for o in spec.description.output])
block = next(iter(spec.mlProgram.functions['main'].block_specializations.values()))
print('ops:', sum(1 for _ in block.operations))
"
```

Expected output: input `patch_tensor` shape `[1, 193, 3075]`, output `mos` scalar, >1000 ops (14 transformer blocks × ~80 ops each).

- [ ] **Step 6: Commit**

```
git add scripts/convert_musiq.py testing/bench/tests/test_convert_musiq.py ImageRater/MLModels/musiq-ava.mlpackage
git commit -m "feat: PyTorch → CoreML converter for MUSIQ-AVA body"
```

---

## Task 2: Test Fixtures — Python-generated reference tensors

**Files:**
- Create: `testing/bench/tests/fixtures/make_musiq_fixtures.py`
- Create: `ImageRaterTests/Fixtures/musiq_reference/` (generated binary fixtures)

Produces reference tensors that Swift tests compare against. Runs once on demand via `make_musiq_fixtures.py` — outputs must be committed so Swift CI doesn't need PyTorch.

- [ ] **Step 1: Implement fixture generator**

```python
# testing/bench/tests/fixtures/make_musiq_fixtures.py
"""Generate deterministic reference tensors for Swift preprocessor parity tests.

Emits binary float32 little-endian files (and .txt shape/label sidecars) into
ImageRaterTests/Fixtures/musiq_reference/. Committed to repo so Swift tests run
without torch."""
from __future__ import annotations
import sys, struct
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from pyiqa.data.multiscale_trans_util import (
    extract_image_patches,
    get_hashed_spatial_pos_emb_index,
    resize_preserve_aspect_ratio,
    _extract_patches_and_positions_from_image,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
OUT_DIR = REPO_ROOT / "ImageRaterTests" / "Fixtures" / "musiq_reference"


def write_tensor(path: Path, t: torch.Tensor) -> None:
    """Write float32 raw + shape sidecar."""
    arr = t.contiguous().detach().cpu().float().numpy()
    path.write_bytes(arr.tobytes())
    (path.with_suffix(".shape")).write_text(",".join(str(d) for d in arr.shape))


def make_image(h: int, w: int, seed: int = 0) -> torch.Tensor:
    """Deterministic [1, 3, h, w] float tensor in [0, 1]."""
    g = torch.Generator().manual_seed(seed)
    return torch.rand(1, 3, h, w, generator=g)


def main():
    torch.manual_seed(0)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Fixture 1: resize 1200x800 longer=224 -> (149, 224)
    img_1200x800 = make_image(1200, 800, seed=1)
    write_tensor(OUT_DIR / "img_1200x800.f32", img_1200x800)
    r224, rh, rw = resize_preserve_aspect_ratio(img_1200x800, 1200, 800, 224)
    write_tensor(OUT_DIR / "resize_1200x800_224.f32", r224)
    (OUT_DIR / "resize_1200x800_224.dims").write_text(f"{rh},{rw}")

    # Fixture 2: resize 800x1200 longer=384 -> (256, 384)
    img_800x1200 = make_image(800, 1200, seed=2)
    write_tensor(OUT_DIR / "img_800x1200.f32", img_800x1200)
    r384, rh, rw = resize_preserve_aspect_ratio(img_800x1200, 800, 1200, 384)
    write_tensor(OUT_DIR / "resize_800x1200_384.f32", r384)
    (OUT_DIR / "resize_800x1200_384.dims").write_text(f"{rh},{rw}")

    # Fixture 3: unfold 3x64x64 ranked-values input -> 4 patches of 3072.
    g = torch.Generator().manual_seed(42)
    img_64 = torch.rand(1, 3, 64, 64, generator=g)
    write_tensor(OUT_DIR / "img_64x64.f32", img_64)
    patches = extract_image_patches(img_64, 32, 32).transpose(1, 2)  # [1, 4, 3072]
    write_tensor(OUT_DIR / "unfold_64x64.f32", patches)

    # Fixture 4: hash spatial positions for (count_h=7, count_w=5, grid=10)
    hsp_7x5 = get_hashed_spatial_pos_emb_index(10, 7, 5)  # [1, 35]
    write_tensor(OUT_DIR / "hsp_7x5.f32", hsp_7x5)

    # Fixture 5: full multiscale patch tensor for 500x400 input, scales [224, 384]
    img_multi = make_image(500, 400, seed=3)
    write_tensor(OUT_DIR / "img_500x400.f32", img_multi)
    outs = []
    for scale_id, longer in enumerate([224, 384]):
        resized, rh, rw = resize_preserve_aspect_ratio(img_multi, 500, 400, longer)
        max_seq_len = int(np.ceil(longer / 32) ** 2)
        out = _extract_patches_and_positions_from_image(
            resized, 32, 32, 10, 1, rh, rw, 3, scale_id, max_seq_len,
        )
        outs.append(out)
    full = torch.cat(outs, dim=-1).transpose(1, 2)  # [1, 193, 3075]
    write_tensor(OUT_DIR / "patch_tensor_500x400.f32", full)

    # MUSIQ normalization convention: (pix - 0.5) * 2. Emit both variants
    # so Swift can test either; preprocessor uses normalized input.
    write_tensor(OUT_DIR / "img_500x400_normalized.f32", (img_multi - 0.5) * 2)

    print(f"Wrote fixtures to {OUT_DIR}")


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Generate fixtures**

```
python3 testing/bench/tests/fixtures/make_musiq_fixtures.py
ls -la ImageRaterTests/Fixtures/musiq_reference/
```

Expected: 10+ `.f32` + `.shape`/`.dims` sidecar files.

- [ ] **Step 3: Commit**

```
git add testing/bench/tests/fixtures/make_musiq_fixtures.py ImageRaterTests/Fixtures/musiq_reference/
git commit -m "test: Python-generated reference fixtures for Swift preprocessor parity"
```

---

## Task 3: Swift MUSIQPreprocessor — Bicubic Resize

**Files:**
- Create: `ImageRater/Pipeline/MUSIQPreprocessor.swift` (new; subsequent tasks append)
- Create: `ImageRaterTests/MUSIQPreprocessorTests.swift`

Custom bicubic kernel matching PyTorch `F.interpolate(mode='bicubic', align_corners=False)`. Uses cubic convolution with a=-0.5, which is PyTorch's default bicubic kernel. Do not use CoreImage / vImage — colorspace drift breaks parity.

- [ ] **Step 1: Write failing resize test**

```swift
// ImageRaterTests/MUSIQPreprocessorTests.swift
import XCTest
@testable import ImageRater

final class MUSIQPreprocessorTests: XCTestCase {

    private func fixturesURL() -> URL {
        Bundle(for: MUSIQPreprocessorTests.self).bundleURL
            .appendingPathComponent("Fixtures/musiq_reference")
    }

    private func loadTensor(_ name: String) -> [Float] {
        let url = fixturesURL().appendingPathComponent("\(name).f32")
        let data = try! Data(contentsOf: url)
        return data.withUnsafeBytes { ptr in
            Array(ptr.bindMemory(to: Float.self))
        }
    }

    private func loadShape(_ name: String, ext: String = "shape") -> [Int] {
        let url = fixturesURL().appendingPathComponent("\(name).\(ext)")
        let s = try! String(contentsOf: url, encoding: .utf8)
        return s.trimmingCharacters(in: .whitespacesAndNewlines)
                .split(separator: ",").map { Int($0)! }
    }

    func test_resize_1200x800_longerSide_224() {
        let src = loadTensor("img_1200x800")                    // [1, 3, 1200, 800]
        let expected = loadTensor("resize_1200x800_224")
        let dims = loadShape("resize_1200x800_224", ext: "dims") // [rh, rw]
        XCTAssertEqual(dims, [149, 224])

        let (resized, rh, rw) = MUSIQPreprocessor.aspectResize(
            pixels: src, h: 1200, w: 800, channels: 3, longerSide: 224
        )
        XCTAssertEqual(rh, 149)
        XCTAssertEqual(rw, 224)
        XCTAssertEqual(resized.count, expected.count)
        // Per-pixel tolerance: bicubic in fp32 drifts ≤ 1e-3 between backends.
        var maxDelta: Float = 0
        for (a, b) in zip(resized, expected) {
            maxDelta = max(maxDelta, abs(a - b))
        }
        XCTAssertLessThan(maxDelta, 5e-3, "Max |Δ| = \(maxDelta)")
    }

    func test_resize_800x1200_longerSide_384() {
        let src = loadTensor("img_800x1200")
        let expected = loadTensor("resize_800x1200_384")
        let dims = loadShape("resize_800x1200_384", ext: "dims")
        XCTAssertEqual(dims, [256, 384])

        let (resized, rh, rw) = MUSIQPreprocessor.aspectResize(
            pixels: src, h: 800, w: 1200, channels: 3, longerSide: 384
        )
        XCTAssertEqual(rh, 256)
        XCTAssertEqual(rw, 384)
        var maxDelta: Float = 0
        for (a, b) in zip(resized, expected) { maxDelta = max(maxDelta, abs(a - b)) }
        XCTAssertLessThan(maxDelta, 5e-3)
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

```
xcodebuild -scheme ImageRater -destination 'platform=macOS,arch=arm64' test \
  -only-testing:ImageRaterTests/MUSIQPreprocessorTests 2>&1 | tail -20
```

Expected: build failure — `MUSIQPreprocessor.aspectResize` not defined.

- [ ] **Step 3: Implement custom bicubic kernel + aspectResize**

```swift
// ImageRater/Pipeline/MUSIQPreprocessor.swift
import Foundation
import CoreML

enum MUSIQPreprocessor {

    // MARK: - Bicubic resize (matches PyTorch F.interpolate mode='bicubic', align_corners=False)

    /// Cubic convolution kernel (Keys 1981) with a=-0.5, PyTorch's default bicubic.
    private static func cubicKernel(_ x: Float) -> Float {
        let a: Float = -0.5
        let ax = abs(x)
        if ax <= 1 {
            return ((a + 2) * ax - (a + 3)) * ax * ax + 1
        } else if ax < 2 {
            return (((ax - 5) * ax + 8) * ax - 4) * a
        }
        return 0
    }

    /// Resample one channel. Pixels stored row-major [h, w].
    /// Uses `align_corners=False`: sample at (out + 0.5) * (in / out) - 0.5.
    private static func resampleChannel(
        src: UnsafePointer<Float>, srcH: Int, srcW: Int,
        dst: UnsafeMutablePointer<Float>, dstH: Int, dstW: Int
    ) {
        let scaleY = Float(srcH) / Float(dstH)
        let scaleX = Float(srcW) / Float(dstW)

        for y in 0..<dstH {
            let srcY = (Float(y) + 0.5) * scaleY - 0.5
            let y0 = Int(floor(srcY)) - 1  // 4-tap kernel needs y0..y0+3

            var wy = [Float](repeating: 0, count: 4)
            for k in 0..<4 {
                wy[k] = cubicKernel(srcY - Float(y0 + k))
            }

            for x in 0..<dstW {
                let srcX = (Float(x) + 0.5) * scaleX - 0.5
                let x0 = Int(floor(srcX)) - 1

                var wx = [Float](repeating: 0, count: 4)
                for k in 0..<4 {
                    wx[k] = cubicKernel(srcX - Float(x0 + k))
                }

                var acc: Float = 0
                for ky in 0..<4 {
                    let yi = max(0, min(srcH - 1, y0 + ky))
                    let row = src + yi * srcW
                    var rowAcc: Float = 0
                    for kx in 0..<4 {
                        let xi = max(0, min(srcW - 1, x0 + kx))
                        rowAcc += row[xi] * wx[kx]
                    }
                    acc += rowAcc * wy[ky]
                }
                dst[y * dstW + x] = acc
            }
        }
    }

    /// Aspect-preserving resize. Longer side becomes `longerSide`. Per-channel.
    /// Input layout: planar [C, H, W] flattened in C-major, H-major, W-minor order.
    static func aspectResize(
        pixels: [Float], h: Int, w: Int, channels: Int, longerSide: Int
    ) -> (pixels: [Float], rh: Int, rw: Int) {
        let ratio = Double(longerSide) / Double(max(h, w))
        let rh = Int((Double(h) * ratio).rounded(.toNearestOrEven))
        let rw = Int((Double(w) * ratio).rounded(.toNearestOrEven))
        var out = [Float](repeating: 0, count: channels * rh * rw)
        pixels.withUnsafeBufferPointer { srcBuf in
            out.withUnsafeMutableBufferPointer { dstBuf in
                for c in 0..<channels {
                    let srcBase = srcBuf.baseAddress! + c * h * w
                    let dstBase = dstBuf.baseAddress! + c * rh * rw
                    resampleChannel(src: srcBase, srcH: h, srcW: w,
                                    dst: dstBase, dstH: rh, dstW: rw)
                }
            }
        }
        return (out, rh, rw)
    }
}
```

- [ ] **Step 4: Run test to verify it passes**

```
xcodebuild -scheme ImageRater -destination 'platform=macOS,arch=arm64' test \
  -only-testing:ImageRaterTests/MUSIQPreprocessorTests/test_resize_1200x800_longerSide_224 \
  -only-testing:ImageRaterTests/MUSIQPreprocessorTests/test_resize_800x1200_longerSide_384 \
  2>&1 | tail -10
```

Expected: both tests pass with max |Δ| ≤ 5e-3.

If tests fail with max |Δ| > 5e-3, the sampling convention (`align_corners`) is wrong. Compare against `F.interpolate(align_corners=False)` formula documentation before tightening kernel.

- [ ] **Step 5: Commit**

```
git add ImageRater/Pipeline/MUSIQPreprocessor.swift ImageRaterTests/MUSIQPreprocessorTests.swift
git commit -m "feat(preprocessor): custom bicubic resize matching PyTorch F.interpolate"
```

---

## Task 4: Swift MUSIQPreprocessor — Patch Unfold

**Files:**
- Modify: `ImageRater/Pipeline/MUSIQPreprocessor.swift` (append `unfoldPatches`)
- Modify: `ImageRaterTests/MUSIQPreprocessorTests.swift` (append unfold test)

Implements TF-SAME padding + 32×32 stride-32 unfold. Row-major over patches, column-major within patch (matches PyTorch `F.unfold`).

- [ ] **Step 1: Write failing unfold test**

```swift
// Append to ImageRaterTests/MUSIQPreprocessorTests.swift

func test_unfoldPatches_64x64_produces_4_patches() {
    let src = loadTensor("img_64x64")            // [1, 3, 64, 64]
    let expected = loadTensor("unfold_64x64")    // [1, 4, 3072]

    let (patches, countH, countW) = MUSIQPreprocessor.unfoldPatches(
        pixels: src, h: 64, w: 64, channels: 3, patch: 32
    )
    XCTAssertEqual(countH, 2)
    XCTAssertEqual(countW, 2)
    XCTAssertEqual(patches.count, 4 * 3072)

    var maxDelta: Float = 0
    for (a, b) in zip(patches, expected) { maxDelta = max(maxDelta, abs(a - b)) }
    XCTAssertLessThan(maxDelta, 1e-5)
}
```

- [ ] **Step 2: Run test to verify it fails**

```
xcodebuild ... -only-testing:ImageRaterTests/MUSIQPreprocessorTests/test_unfoldPatches_64x64_produces_4_patches
```

Expected: build failure — `unfoldPatches` not defined.

- [ ] **Step 3: Implement `unfoldPatches`**

Append to `MUSIQPreprocessor.swift`:

```swift
    // MARK: - Patch unfold (32×32 stride-32 with TF-SAME padding)

    /// TF-SAME padding: output count = ceil(input / stride). Extra pixel on bottom/right.
    static func unfoldPatches(
        pixels: [Float], h: Int, w: Int, channels: Int, patch: Int
    ) -> (patches: [Float], countH: Int, countW: Int) {
        let stride = patch
        let countH = (h + stride - 1) / stride
        let countW = (w + stride - 1) / stride
        let padH = (countH - 1) * stride + patch - h     // ≥ 0
        let padW = (countW - 1) * stride + patch - w
        let top = padH / 2, left = padW / 2              // matches pyiqa F.pad ordering

        let numPatches = countH * countW
        let rowDim = channels * patch * patch
        var out = [Float](repeating: 0, count: numPatches * rowDim)

        pixels.withUnsafeBufferPointer { srcBuf in
            out.withUnsafeMutableBufferPointer { dstBuf in
                for py in 0..<countH {
                    for px in 0..<countW {
                        let patchIdx = py * countW + px
                        let dstBase = dstBuf.baseAddress! + patchIdx * rowDim
                        // For each channel, copy 32 rows of 32 pixels.
                        for c in 0..<channels {
                            let srcChannel = srcBuf.baseAddress! + c * h * w
                            // dst layout per patch: [C, patch, patch] flattened to rowDim.
                            let dstChannel = dstBase + c * patch * patch
                            for dy in 0..<patch {
                                let srcY = py * stride + dy - top
                                for dx in 0..<patch {
                                    let srcX = px * stride + dx - left
                                    let dstIdx = dy * patch + dx
                                    if srcY >= 0 && srcY < h && srcX >= 0 && srcX < w {
                                        dstChannel[dstIdx] = srcChannel[srcY * w + srcX]
                                    } else {
                                        dstChannel[dstIdx] = 0
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        return (out, countH, countW)
    }
```

- [ ] **Step 4: Run test to verify it passes**

```
xcodebuild ... -only-testing:ImageRaterTests/MUSIQPreprocessorTests/test_unfoldPatches_64x64_produces_4_patches
```

Expected: PASS, max |Δ| ≤ 1e-5.

- [ ] **Step 5: Commit**

```
git add ImageRater/Pipeline/MUSIQPreprocessor.swift ImageRaterTests/MUSIQPreprocessorTests.swift
git commit -m "feat(preprocessor): 32x32 patch unfold with TF-SAME padding"
```

---

## Task 5: Swift MUSIQPreprocessor — Hash Spatial Positions

**Files:**
- Modify: `ImageRater/Pipeline/MUSIQPreprocessor.swift`
- Modify: `ImageRaterTests/MUSIQPreprocessorTests.swift`

- [ ] **Step 1: Write failing hash positions test**

```swift
func test_hashSpatialPositions_7x5_matches_pyiqa() {
    let expected = loadTensor("hsp_7x5")  // [1, 35]
    let out = MUSIQPreprocessor.hashSpatialPositions(countH: 7, countW: 5, gridSize: 10)
    XCTAssertEqual(out.count, 35)
    for (a, b) in zip(out, expected) {
        XCTAssertEqual(a, b, accuracy: 0.0)  // integer indices — exact
    }
}
```

- [ ] **Step 2: Run to verify fail**

Expected: `hashSpatialPositions` not found.

- [ ] **Step 3: Implement hashSpatialPositions**

```swift
    // MARK: - Hash spatial positions (grid_size=10)

    /// Nearest-interp [0..gridSize-1] to `count`, then flatten count_h × count_w
    /// into grid_size-based hash `h * gridSize + w`. Matches pyiqa's
    /// F.interpolate(mode='nearest').
    ///
    /// Nearest-interp formula PyTorch uses (align_corners irrelevant for nearest):
    /// index[i] = floor(i * in_size / out_size) for out_size ≥ in_size.
    /// For grid=10, count=7: indices = [0, 1, 2, 4, 5, 7, 8] (from floor(i * 10 / 7)).
    static func hashSpatialPositions(countH: Int, countW: Int, gridSize: Int) -> [Float] {
        let posH = nearestInterp(count: countH, gridSize: gridSize)
        let posW = nearestInterp(count: countW, gridSize: gridSize)
        var out = [Float](repeating: 0, count: countH * countW)
        for i in 0..<countH {
            for j in 0..<countW {
                out[i * countW + j] = Float(posH[i] * gridSize + posW[j])
            }
        }
        return out
    }

    private static func nearestInterp(count: Int, gridSize: Int) -> [Int] {
        // PyTorch F.interpolate(mode='nearest'): index = floor(i * in / out)
        var out = [Int](repeating: 0, count: count)
        for i in 0..<count {
            out[i] = (i * gridSize) / count
        }
        return out
    }
```

- [ ] **Step 4: Run to verify pass**

Expected: PASS. If fails, verify `nearestInterp` formula — PyTorch uses `floor(i * in / out)` not `round`.

- [ ] **Step 5: Commit**

```
git add ImageRater/Pipeline/MUSIQPreprocessor.swift ImageRaterTests/MUSIQPreprocessorTests.swift
git commit -m "feat(preprocessor): hash spatial position indices"
```

---

## Task 6: Swift MUSIQPreprocessor — Orchestrator (`patchTensor`)

**Files:**
- Modify: `ImageRater/Pipeline/MUSIQPreprocessor.swift`
- Modify: `ImageRater/Pipeline/RatingPipeline.swift` (add `RatingError.imageTooSmall`)
- Modify: `ImageRaterTests/MUSIQPreprocessorTests.swift`

Combines resize + unfold + positions + scale + mask + pad-to-max-seq-len + concat into final `[1, 193, 3075]` MLMultiArray.

- [ ] **Step 1: Write failing orchestrator + shape + too-small tests**

```swift
func test_patchTensor_500x400_matches_pyiqa_reference() throws {
    let src = loadTensor("img_500x400_normalized")  // normalized float tensor
    let expected = loadTensor("patch_tensor_500x400") // [1, 193, 3075]

    let tensor = try MUSIQPreprocessor.patchTensorFromNormalizedPixels(
        pixels: src, h: 500, w: 400, channels: 3
    )
    XCTAssertEqual(tensor.shape.map { $0.intValue }, [1, 193, 3075])
    XCTAssertEqual(tensor.dataType, .float32)

    let ptr = tensor.dataPointer.bindMemory(to: Float.self, capacity: tensor.count)
    var maxDelta: Float = 0
    for i in 0..<expected.count {
        maxDelta = max(maxDelta, abs(ptr[i] - expected[i]))
    }
    // Combined resize + patch + pos tolerance (parity driver enforces tighter 5e-3).
    XCTAssertLessThan(maxDelta, 1e-2, "Max |Δ| = \(maxDelta)")
}

func test_patchTensor_image_too_small_throws() {
    let src = [Float](repeating: 0.5, count: 3 * 20 * 30)
    XCTAssertThrowsError(
        try MUSIQPreprocessor.patchTensorFromNormalizedPixels(
            pixels: src, h: 20, w: 30, channels: 3
        )
    ) { err in
        guard case RatingError.imageTooSmall = err else {
            return XCTFail("Expected .imageTooSmall, got \(err)")
        }
    }
}
```

- [ ] **Step 2: Run to verify fail**

Expected: `patchTensorFromNormalizedPixels` not defined; `RatingError.imageTooSmall` may also not exist.

- [ ] **Step 3: Add `RatingError.imageTooSmall`**

Edit `ImageRater/Pipeline/RatingPipeline.swift`:

```swift
enum RatingError: Error {
    case pixelBufferCreationFailed
    case modelNotFound(String)
    case inferenceOutputMismatch
    case imageTooSmall
}
```

- [ ] **Step 4: Implement orchestrator**

Append to `MUSIQPreprocessor.swift`:

```swift
    // MARK: - Orchestrator

    static let scales: [Int] = [224, 384]      // sorted ascending
    static let patchSize: Int = 32
    static let gridSize: Int = 10
    static let seqLen: Int = 193               // 49 + 144
    static let rowDim: Int = 32 * 32 * 3 + 3   // 3075

    /// Build patch tensor from normalized planar RGB pixels in `[-1, 1]`.
    /// Input layout: channel-major `[C, H, W]`.
    static func patchTensorFromNormalizedPixels(
        pixels: [Float], h: Int, w: Int, channels: Int
    ) throws -> MLMultiArray {
        guard max(h, w) >= patchSize else { throw RatingError.imageTooSmall }

        let tensor = try MLMultiArray(shape: [1, NSNumber(value: seqLen), NSNumber(value: rowDim)],
                                      dataType: .float32)
        // Zero-fill so padded rows stay zero.
        let tPtr = tensor.dataPointer.bindMemory(to: Float.self, capacity: tensor.count)
        tPtr.initialize(repeating: 0, count: tensor.count)

        var rowOffset = 0
        for (scaleId, scale) in scales.enumerated() {
            let (resized, rh, rw) = aspectResize(
                pixels: pixels, h: h, w: w, channels: channels, longerSide: scale
            )
            let (patches, countH, countW) = unfoldPatches(
                pixels: resized, h: rh, w: rw, channels: channels, patch: patchSize
            )
            let positions = hashSpatialPositions(countH: countH, countW: countW, gridSize: gridSize)
            let activeN = countH * countW
            let side = (scale + patchSize - 1) / patchSize
            let maxSeqLen = side * side

            // Write active rows: [patch (3072), spatial_p, scale_id, mask].
            let patchDim = channels * patchSize * patchSize
            for i in 0..<activeN {
                let dst = tPtr + (rowOffset + i) * rowDim
                let patchSrc = patches.withUnsafeBufferPointer { $0.baseAddress! + i * patchDim }
                dst.update(from: patchSrc, count: patchDim)
                dst[patchDim] = positions[i]
                dst[patchDim + 1] = Float(scaleId)
                dst[patchDim + 2] = 1.0
            }
            // Pad rows already zeroed — attention mask stays 0 → ignored.
            rowOffset += maxSeqLen
        }
        assert(rowOffset == seqLen, "rowOffset \(rowOffset) != seqLen \(seqLen)")
        return tensor
    }
```

- [ ] **Step 5: Run to verify pass**

```
xcodebuild ... -only-testing:ImageRaterTests/MUSIQPreprocessorTests
```

Expected: all 5 preprocessor tests pass.

- [ ] **Step 6: Commit**

```
git add ImageRater/Pipeline/MUSIQPreprocessor.swift ImageRater/Pipeline/RatingPipeline.swift ImageRaterTests/MUSIQPreprocessorTests.swift
git commit -m "feat(preprocessor): patchTensor orchestrator + RatingError.imageTooSmall"
```

---

## Task 7: CGImage → Normalized Pixels + End-to-End Preprocessor Parity

**Files:**
- Modify: `ImageRater/Pipeline/MUSIQPreprocessor.swift` (add `patchTensor(cgImage:)`)
- Modify: `ImageRaterTests/MUSIQPreprocessorTests.swift` (add CGImage wrapper test)
- Create: `testing/bench/bench/parity.py`
- Create: `testing/bench/tests/test_musiq_parity.py` (preprocessor slice only — CoreML gate arrives in Task 14)

- [ ] **Step 1: Write failing CGImage wrapper test**

```swift
func test_patchTensor_cgImage_wraps_normalization() throws {
    // 64×64 synthetic CGImage; just verify pipeline runs and emits correct shape.
    let ctx = CGContext(
        data: nil, width: 64, height: 64,
        bitsPerComponent: 8, bytesPerRow: 64 * 4,
        space: CGColorSpace(name: CGColorSpace.sRGB)!,
        bitmapInfo: CGBitmapInfo.byteOrder32Little.rawValue | CGImageAlphaInfo.noneSkipFirst.rawValue
    )!
    ctx.setFillColor(CGColor(red: 0.5, green: 0.3, blue: 0.7, alpha: 1))
    ctx.fill(CGRect(x: 0, y: 0, width: 64, height: 64))
    let cg = ctx.makeImage()!
    let tensor = try MUSIQPreprocessor.patchTensor(cgImage: cg)
    XCTAssertEqual(tensor.shape.map { $0.intValue }, [1, 193, 3075])
}
```

- [ ] **Step 2: Run to verify fail**

Expected: `patchTensor(cgImage:)` not defined.

- [ ] **Step 3: Implement CGImage reader**

Append to `MUSIQPreprocessor.swift`:

```swift
    // MARK: - CGImage entry point

    /// Read CGImage → planar RGB `[-1, 1]` → patch tensor. Uses CVPixelBuffer
    /// BGRA straight-copy; no implicit gamma, matches pyiqa torchvision.io.read_image.
    static func patchTensor(cgImage: CGImage) throws -> MLMultiArray {
        let h = cgImage.height, w = cgImage.width
        let pixels = try readPlanarRGB(cgImage: cgImage)       // [C, H, W] in [-1, 1]
        return try patchTensorFromNormalizedPixels(
            pixels: pixels, h: h, w: w, channels: 3
        )
    }

    private static func readPlanarRGB(cgImage: CGImage) throws -> [Float] {
        let h = cgImage.height, w = cgImage.width
        var pb: CVPixelBuffer?
        let attrs = [kCVPixelBufferCGImageCompatibilityKey: true,
                     kCVPixelBufferCGBitmapContextCompatibilityKey: true] as CFDictionary
        guard CVPixelBufferCreate(kCFAllocatorDefault, w, h, kCVPixelFormatType_32BGRA,
                                  attrs, &pb) == kCVReturnSuccess,
              let buf = pb else {
            throw RatingError.pixelBufferCreationFailed
        }
        CVPixelBufferLockBaseAddress(buf, [])
        defer { CVPixelBufferUnlockBaseAddress(buf, []) }
        guard let sRGB = CGColorSpace(name: CGColorSpace.sRGB),
              let ctx = CGContext(
                data: CVPixelBufferGetBaseAddress(buf),
                width: w, height: h, bitsPerComponent: 8,
                bytesPerRow: CVPixelBufferGetBytesPerRow(buf),
                space: sRGB,
                bitmapInfo: CGBitmapInfo.byteOrder32Little.rawValue |
                            CGImageAlphaInfo.noneSkipFirst.rawValue
              ) else {
            throw RatingError.pixelBufferCreationFailed
        }
        ctx.draw(cgImage, in: CGRect(x: 0, y: 0, width: w, height: h))

        let rowBytes = CVPixelBufferGetBytesPerRow(buf)
        let base = CVPixelBufferGetBaseAddress(buf)!.assumingMemoryBound(to: UInt8.self)
        var out = [Float](repeating: 0, count: 3 * h * w)
        let rSlice = h * w * 0, gSlice = h * w * 1, bSlice = h * w * 2
        for y in 0..<h {
            let row = base + y * rowBytes
            for x in 0..<w {
                let px = row + x * 4
                // BGRA little-endian layout → px[0]=B, px[1]=G, px[2]=R.
                let b = Float(px[0]) / 255, g = Float(px[1]) / 255, r = Float(px[2]) / 255
                out[rSlice + y * w + x] = (r - 0.5) * 2
                out[gSlice + y * w + x] = (g - 0.5) * 2
                out[bSlice + y * w + x] = (b - 0.5) * 2
            }
        }
        return out
    }
```

- [ ] **Step 4: Run to verify pass**

```
xcodebuild ... -only-testing:ImageRaterTests/MUSIQPreprocessorTests
```

Expected: all preprocessor tests pass.

- [ ] **Step 5: Implement parity driver helpers**

```python
# testing/bench/bench/parity.py
"""Python reference vs Swift/CoreML parity helpers."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import pyiqa
from scipy.stats import spearmanr


def pyiqa_scores(image_dir: Path, limit: int = 100) -> pd.DataFrame:
    """Run pyiqa MUSIQ-AVA on first `limit` jpgs, return [filename, score]."""
    metric = pyiqa.create_metric("musiq-ava", device="cpu", as_loss=False)
    metric.eval()
    files = sorted(image_dir.glob("*.jpg"))[:limit]
    rows = []
    with torch.no_grad():
        for f in files:
            s = float(metric(str(f)).detach().cpu().item())
            rows.append({"filename": f.name, "pyiqa_score": s})
    return pd.DataFrame(rows)


def compare(py_df: pd.DataFrame, sw_df: pd.DataFrame) -> dict:
    """Returns spearman, max/mean abs delta on joined frame."""
    m = py_df.merge(sw_df, on="filename", how="inner")
    rho, _ = spearmanr(m["pyiqa_score"], m["coreml_score"])
    delta = (m["pyiqa_score"] - m["coreml_score"]).abs()
    return {
        "n": len(m),
        "spearman": float(rho),
        "max_abs_delta": float(delta.max()),
        "mean_abs_delta": float(delta.mean()),
    }
```

- [ ] **Step 6: Commit**

```
git add ImageRater/Pipeline/MUSIQPreprocessor.swift ImageRaterTests/MUSIQPreprocessorTests.swift testing/bench/bench/parity.py
git commit -m "feat(preprocessor): CGImage entry point + parity helpers"
```

---

## Task 8: Schema Migration — params.current.json v0.4.0 + Generated Defaults

**Files:**
- Modify: `testing/bench/params.current.json`
- Modify: `scripts/gen_defaults.py`
- Modify: `ImageRater/App/FocalSettings.swift`
- Modify: `ImageRaterTests/GeneratedDefaultsTests.swift`

Schema change first so downstream code compiles against new shape before RatingPipeline is rewritten.

- [ ] **Step 1: Write failing GeneratedDefaultsTests**

Edit `ImageRaterTests/GeneratedDefaultsTests.swift` (likely currently tests ensemble weights — rewrite):

```swift
import XCTest
@testable import ImageRater

final class GeneratedDefaultsTests: XCTestCase {
    func test_thresholds_monotonic_and_in_expected_range() {
        let t1 = FocalSettings.defaultMUSIQThreshold1
        let t2 = FocalSettings.defaultMUSIQThreshold2
        let t3 = FocalSettings.defaultMUSIQThreshold3
        let t4 = FocalSettings.defaultMUSIQThreshold4
        XCTAssertLessThan(t1, t2)
        XCTAssertLessThan(t2, t3)
        XCTAssertLessThan(t3, t4)
        XCTAssertTrue((1.0...10.0).contains(Double(t1)), "t1 out of MUSIQ range")
        XCTAssertTrue((1.0...10.0).contains(Double(t4)), "t4 out of MUSIQ range")
    }

    func test_version_matches_params_current_json() {
        XCTAssertEqual(FocalSettings.generatedVersion, "v0.4.0")
    }
}
```

- [ ] **Step 2: Run to verify fail**

```
xcodebuild ... -only-testing:ImageRaterTests/GeneratedDefaultsTests
```

Expected: build fails because `defaultMUSIQThreshold1..4` and `generatedVersion = v0.4.0` not yet defined.

- [ ] **Step 3: Update params.current.json to v0.4.0 schema**

Replace `testing/bench/params.current.json`:

```json
{
  "version": "v0.4.0",
  "date": "2026-04-15",
  "model": "musiq-ava",
  "notes": "single-model rating; CoreML on-device",
  "thresholds": [4.465, 5.181, 5.634, 6.068]
}
```

- [ ] **Step 4: Rewrite `scripts/gen_defaults.py`**

```python
#!/usr/bin/env python3
"""Generate Swift FocalSettings constants from params.current.json (v0.4.0)."""
from __future__ import annotations
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARAMS = ROOT / "testing" / "bench" / "params.current.json"
OUT = ROOT / "ImageRater" / "App" / "FocalSettings+Generated.swift"


def main() -> int:
    if not PARAMS.exists():
        print(f"error: {PARAMS} missing", file=sys.stderr)
        return 1
    data = json.loads(PARAMS.read_text())
    if data.get("version", "").split("@")[0] != "v0.4.0":
        print(f"error: expected schema v0.4.0, got {data.get('version')}", file=sys.stderr)
        return 1
    if data.get("model") != "musiq-ava":
        print(f"error: expected model 'musiq-ava', got {data.get('model')}", file=sys.stderr)
        return 1
    t = data["thresholds"]
    if len(t) != 4 or not all(isinstance(x, (int, float)) for x in t):
        print(f"error: thresholds must be 4 floats, got {t}", file=sys.stderr)
        return 1

    src = f"""// Auto-generated by scripts/gen_defaults.py. Do not edit.
// Regenerated at build time from testing/bench/params.current.json.
import Foundation

extension FocalSettings {{
    static let generatedVersion: String = {json.dumps(data['version'])}
    static let generatedModel: String = {json.dumps(data['model'])}
    static let generatedMUSIQThreshold1: Float = {float(t[0])}
    static let generatedMUSIQThreshold2: Float = {float(t[1])}
    static let generatedMUSIQThreshold3: Float = {float(t[2])}
    static let generatedMUSIQThreshold4: Float = {float(t[3])}
}}
"""
    OUT.write_text(src)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: Update `FocalSettings.swift`**

Edit `ImageRater/App/FocalSettings.swift` — delete ensemble defaults, add threshold defaults. Delete these lines (exact match will vary; grep `defaultWeightTechnical` / `defaultBucketEdge` / `defaultClipLogitScale` / `defaultCullStrictness`):

```swift
static var defaultWeightTechnical: Float { generatedWeightTechnical }
static var defaultWeightAesthetic: Float { generatedWeightAesthetic }
static var defaultWeightClip: Float { generatedWeightClip }
static var defaultCullStrictness: Float { generatedCullStrictness }
static var defaultBucketEdge1: Float { generatedBucketEdge1 }
static var defaultBucketEdge2: Float { generatedBucketEdge2 }
static var defaultBucketEdge3: Float { generatedBucketEdge3 }
static var defaultBucketEdge4: Float { generatedBucketEdge4 }
static var defaultClipLogitScale: Float { generatedClipLogitScale }
```

Add:

```swift
static var defaultMUSIQThreshold1: Float { generatedMUSIQThreshold1 }
static var defaultMUSIQThreshold2: Float { generatedMUSIQThreshold2 }
static var defaultMUSIQThreshold3: Float { generatedMUSIQThreshold3 }
static var defaultMUSIQThreshold4: Float { generatedMUSIQThreshold4 }
```

Grep `FocalSettings.resolvedClipLogitScale` across the codebase and remove callers (it now returns a constant that doesn't exist). Same for `FocalSettings.resolvedCullStrictness` and bucket-edge resolvers if present.

- [ ] **Step 6: Regenerate + run tests**

```
python3 scripts/gen_defaults.py
cat ImageRater/App/FocalSettings+Generated.swift   # sanity-check contents
xcodebuild ... -only-testing:ImageRaterTests/GeneratedDefaultsTests
```

Expected: gen script prints "wrote …"; generated Swift file contains `defaultMUSIQThreshold` constants + `generatedVersion = "v0.4.0"`; tests pass.

If full `xcodebuild test` fails with "cannot find `generatedWeightTechnical`" etc., grep for each missing identifier and remove the call sites — they're dead paths tied to the removed ensemble. Task 9 + 11 will finish the callers; it's expected this commit leaves the build broken until then. Mark commit message `wip`.

- [ ] **Step 7: Commit (WIP)**

```
git add testing/bench/params.current.json scripts/gen_defaults.py ImageRater/App/FocalSettings.swift ImageRater/App/FocalSettings+Generated.swift ImageRaterTests/GeneratedDefaultsTests.swift
git commit -m "wip: params schema v0.4.0 + FocalSettings threshold defaults"
```

---

## Task 9: RatingPipeline Rewrite — Single Model + bucketStars

**Files:**
- Modify: `ImageRater/Pipeline/RatingPipeline.swift` (rewrite)
- Modify: `ImageRater/Models/RatingResult.swift`
- Modify: `ImageRaterTests/RatingPipelineTests.swift`

- [ ] **Step 1: Write failing bucketStars + rate tests**

Rewrite `ImageRaterTests/RatingPipelineTests.swift`:

```swift
import XCTest
@testable import ImageRater

final class RatingPipelineTests: XCTestCase {

    // MARK: - bucketStars

    private let t: (Float, Float, Float, Float) = (4.465, 5.181, 5.634, 6.068)

    func test_bucketStars_belowFirstThreshold() {
        XCTAssertEqual(RatingPipeline.bucketStars(mos: 3.0, thresholds: t), 1)
        XCTAssertEqual(RatingPipeline.bucketStars(mos: 4.464, thresholds: t), 1)
    }
    func test_bucketStars_boundaries() {
        XCTAssertEqual(RatingPipeline.bucketStars(mos: 4.465, thresholds: t), 1) // ≤ t1
        XCTAssertEqual(RatingPipeline.bucketStars(mos: 4.466, thresholds: t), 2)
        XCTAssertEqual(RatingPipeline.bucketStars(mos: 5.181, thresholds: t), 2) // ≤ t2
        XCTAssertEqual(RatingPipeline.bucketStars(mos: 5.182, thresholds: t), 3)
        XCTAssertEqual(RatingPipeline.bucketStars(mos: 5.634, thresholds: t), 3)
        XCTAssertEqual(RatingPipeline.bucketStars(mos: 5.635, thresholds: t), 4)
        XCTAssertEqual(RatingPipeline.bucketStars(mos: 6.068, thresholds: t), 4)
        XCTAssertEqual(RatingPipeline.bucketStars(mos: 6.069, thresholds: t), 5)
    }
    func test_bucketStars_extremes() {
        XCTAssertEqual(RatingPipeline.bucketStars(mos: 1.0, thresholds: t), 1)
        XCTAssertEqual(RatingPipeline.bucketStars(mos: 10.0, thresholds: t), 5)
    }

    // MARK: - rate error paths

    func test_rate_returns_unrated_on_nan() async throws {
        let models = MockModel.make(scalar: Float.nan)
        let cg = TestImage.solid(64, 64)
        let r = await RatingPipeline.rate(image: cg, models: models)
        guard case .unrated = r else { XCTFail("expected .unrated"); return }
    }
    func test_rate_returns_unrated_on_inf() async throws {
        let models = MockModel.make(scalar: Float.infinity)
        let cg = TestImage.solid(64, 64)
        let r = await RatingPipeline.rate(image: cg, models: models)
        guard case .unrated = r else { XCTFail("expected .unrated"); return }
    }
}
```

Add helpers next to the test (or in a `TestSupport` file) — `MockModel.make(scalar:)` returns a `BundledModels` whose `musiq` is a stub conforming to a protocol that `RatingPipeline.rate` goes through. The simplest path is to keep the stub inside the test file and gate on a protocol.

Introduce `MUSIQInferring` protocol in `RatingPipeline.swift`:

```swift
protocol MUSIQInferring {
    func predict(patchTensor: MLMultiArray) async throws -> Float
}
```

Wrap `MLModel` with a `CoreMLMUSIQ: MUSIQInferring` struct, and store `MUSIQInferring` inside `BundledModels`. This makes `rate()` unit-testable without a real `.mlmodelc`.

- [ ] **Step 2: Run to verify fail**

Build will fail. Many call sites still reference `combinedQuality`, `clipIQAScore`, etc. Move to implementation.

- [ ] **Step 3: Rewrite `RatingResult`**

```swift
// ImageRater/Models/RatingResult.swift
import Foundation

struct RatedScores: Codable, Equatable {
    let musiqAesthetic: Float
    let stars: Int
}

enum RatingResult {
    case rated(RatedScores)
    case unrated
}
```

- [ ] **Step 4: Rewrite `RatingPipeline.swift`**

```swift
// ImageRater/Pipeline/RatingPipeline.swift
import CoreML
import CoreImage
import Foundation
import OSLog

private let log = Logger(subsystem: "com.focal.app", category: "RatingPipeline")

enum RatingError: Error {
    case pixelBufferCreationFailed
    case modelNotFound(String)
    case inferenceOutputMismatch
    case imageTooSmall
}

protocol MUSIQInferring {
    func predict(patchTensor: MLMultiArray) async throws -> Float
}

struct CoreMLMUSIQ: MUSIQInferring {
    let model: MLModel
    func predict(patchTensor: MLMultiArray) async throws -> Float {
        let input = try MLDictionaryFeatureProvider(dictionary: ["patch_tensor": patchTensor])
        let out = try await model.prediction(from: input)
        for name in out.featureNames {
            if let arr = out.featureValue(for: name)?.multiArrayValue, arr.count > 0 {
                return arr[0].floatValue
            }
            if let d = out.featureValue(for: name)?.doubleValue { return Float(d) }
        }
        throw RatingError.inferenceOutputMismatch
    }
}

enum RatingPipeline {

    struct BundledModels {
        let musiq: MUSIQInferring
    }

    static func loadBundledModels() throws -> BundledModels {
        let config = MLModelConfiguration()
        config.computeUnits = isAppleSilicon ? .all : .cpuOnly
        let model = try loadBundledModel(named: "musiq-ava", configuration: config)
        return BundledModels(musiq: CoreMLMUSIQ(model: model))
    }

    static func rate(image: CGImage, models: BundledModels) async -> RatingResult {
        do {
            let tensor = try MUSIQPreprocessor.patchTensor(cgImage: image)
            let raw = try await models.musiq.predict(patchTensor: tensor)
            guard raw.isFinite else {
                log.warning("MUSIQ returned non-finite \(raw); unrated")
                return .unrated
            }
            let clamped = min(max(raw, 1.0), 10.0)
            if clamped != raw {
                log.warning("MUSIQ out-of-range \(raw); clamped to \(clamped)")
            }
            let stars = bucketStars(mos: clamped, thresholds: defaultThresholds())
            return .rated(RatedScores(musiqAesthetic: clamped, stars: stars))
        } catch {
            log.error("rate failed: \(String(describing: error))")
            return .unrated
        }
    }

    static func bucketStars(mos: Float, thresholds: (Float, Float, Float, Float)) -> Int {
        if mos <= thresholds.0 { return 1 }
        if mos <= thresholds.1 { return 2 }
        if mos <= thresholds.2 { return 3 }
        if mos <= thresholds.3 { return 4 }
        return 5
    }

    static func defaultThresholds() -> (Float, Float, Float, Float) {
        (FocalSettings.defaultMUSIQThreshold1,
         FocalSettings.defaultMUSIQThreshold2,
         FocalSettings.defaultMUSIQThreshold3,
         FocalSettings.defaultMUSIQThreshold4)
    }

    // MARK: - Bundle loader (unchanged signature)

    private static func loadBundledModel(named name: String, configuration: MLModelConfiguration) throws -> MLModel {
        if let url = Bundle.main.url(forResource: name, withExtension: "mlmodelc") {
            return try MLModel(contentsOf: url, configuration: configuration)
        }
        if let envDir = ProcessInfo.processInfo.environment["FOCAL_MLMODELS_DIR"] {
            let url = URL(fileURLWithPath: envDir).appendingPathComponent("\(name).mlmodelc")
            if FileManager.default.fileExists(atPath: url.path) {
                return try MLModel(contentsOf: url, configuration: configuration)
            }
        }
        let exeDir = Bundle.main.bundleURL.deletingLastPathComponent()
        let siblingURL = exeDir.appendingPathComponent("\(name).mlmodelc")
        if FileManager.default.fileExists(atPath: siblingURL.path) {
            return try MLModel(contentsOf: siblingURL, configuration: configuration)
        }
        guard let pkgURL = Bundle.main.url(forResource: name, withExtension: "mlpackage") else {
            throw RatingError.modelNotFound(name)
        }
        let compiled = try MLModel.compileModel(at: pkgURL)
        return try MLModel(contentsOf: compiled, configuration: configuration)
    }

    private static var isAppleSilicon: Bool {
        #if arch(arm64)
        return true
        #else
        return false
        #endif
    }
}
```

- [ ] **Step 5: Add test helpers**

```swift
// ImageRaterTests/TestSupport.swift (new)
import CoreML
import CoreGraphics
@testable import ImageRater

enum TestImage {
    static func solid(_ w: Int, _ h: Int) -> CGImage {
        let ctx = CGContext(
            data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
            space: CGColorSpace(name: CGColorSpace.sRGB)!,
            bitmapInfo: CGBitmapInfo.byteOrder32Little.rawValue | CGImageAlphaInfo.noneSkipFirst.rawValue
        )!
        ctx.setFillColor(CGColor(red: 0.4, green: 0.6, blue: 0.2, alpha: 1))
        ctx.fill(CGRect(x: 0, y: 0, width: w, height: h))
        return ctx.makeImage()!
    }
}

struct MockMUSIQ: MUSIQInferring {
    let scalar: Float
    func predict(patchTensor: MLMultiArray) async throws -> Float { scalar }
}

enum MockModel {
    static func make(scalar: Float) -> RatingPipeline.BundledModels {
        RatingPipeline.BundledModels(musiq: MockMUSIQ(scalar: scalar))
    }
}
```

- [ ] **Step 6: Run tests**

```
xcodebuild ... -only-testing:ImageRaterTests/RatingPipelineTests
```

Expected: all bucketStars + nan/inf tests pass.

- [ ] **Step 7: Commit (still WIP — callers unchanged)**

```
git add ImageRater/Pipeline/RatingPipeline.swift ImageRater/Models/RatingResult.swift ImageRaterTests/RatingPipelineTests.swift ImageRaterTests/TestSupport.swift
git commit -m "wip: single-model RatingPipeline + bucketStars + MUSIQInferring protocol"
```

---

## Task 10: Bench `run.py` — Single-model eval/optimize + drop ablate

**Files:**
- Modify: `testing/bench/run.py`
- Modify: `testing/bench/tests/test_run.py`
- Create/rewrite: `testing/bench/bench/single_model.py` (bucket tuning helpers)

- [ ] **Step 1: Write failing run.py tests**

Edit `testing/bench/tests/test_run.py`:

```python
def test_params_current_json_is_v040(tmp_path):
    """params.current.json must match v0.4.0 schema."""
    import json
    payload = json.loads((Path(__file__).parents[1] / "params.current.json").read_text())
    assert payload["version"] == "v0.4.0"
    assert payload["model"] == "musiq-ava"
    assert len(payload["thresholds"]) == 4
    assert all(isinstance(x, (int, float)) for x in payload["thresholds"])
    # Old ensemble fields must be absent.
    assert "params" not in payload
    assert "bucket_edges" not in payload

def test_argparse_drops_ablate_subcommand():
    """cmd_ablate was removed; argparse should reject 'ablate'."""
    from run import _build_parser  # noqa
    import argparse, pytest
    p = _build_parser()
    with pytest.raises(SystemExit):
        p.parse_args(["ablate"])
```

Also delete existing assertions that reference `params.params.w_tech` etc.

- [ ] **Step 2: Run to verify fail**

```
cd testing/bench
python3 -m pytest tests/test_run.py -v
```

Expected: old tests fail (ensemble fields gone); new tests fail (argparse still has ablate).

- [ ] **Step 3: Rewrite `run.py` eval/optimize/ablate**

Replace `cmd_eval`, `cmd_optimize`, `cmd_ablate`, and `_scores_with_clip_scalar` usage:

```python
# testing/bench/run.py (relevant diff; keep download/score/leaderboard identical)

# Delete: from bench.clip_iqa import clip_iqa_score, load_prompt_embeddings
# Delete: from bench.ensemble import EnsembleParams, stars_from_subscores
# Delete: from bench.optimize import optimize_params, SearchSpace, result_to_params_dict, result_to_metrics_dict
# Delete: from bench.ablation import run_ablation

from bench.single_model import (
    bucket_stars, optimize_thresholds, stars_from_thresholds,
)

# Delete PROMPTS_PATH, _scores_with_clip_scalar

def _load_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = pd.read_csv(DATA_DIR / "ava" / "labels.csv")
    bin_ = locate_scorer_bin()
    scores = score_with_cache(bin_, DATA_DIR / "ava" / "images", CACHE_DIR)
    # Task 10 runs before Task 11: FocalScorer still emits legacy columns here.
    # Override with Python pyiqa MUSIQ as authoritative source for benches.
    # Task 11 rewrites FocalScorer to emit musiqAesthetic; simplify then.
    musiq = score_musiq_with_cache(DATA_DIR / "ava" / "images", CACHE_DIR)
    scores = scores[["filename"]].merge(musiq, on="filename", how="inner")
    return scores, labels


def cmd_eval(args):
    scores, labels = _load_dataset()
    payload = load_params(Path(args.params))
    thresholds = tuple(payload["thresholds"])
    df = scores.merge(labels, on="filename", how="inner")
    pred = stars_from_thresholds(df["musiqAesthetic"].to_numpy(), thresholds)
    m = compute_metrics(df["gt_stars"].to_numpy(), pred)

    version = version_with_sha(payload["version"])
    out_dir = RESULTS_DIR / version
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps({"val": m.to_dict()}, indent=2))
    (out_dir / "params.json").write_text(json.dumps({**payload, "date": _today()}, indent=2))
    df_out = df.copy()
    df_out["pred_stars"] = pred
    df_out[["filename", "gt_stars", "pred_stars", "musiqAesthetic"]].to_parquet(
        out_dir / "scores.parquet"
    )
    print(f"Spearman={m.spearman:.3f}  MAE={m.mae:.2f}  ±1={m.off_by_one*100:.0f}%")
    print(f"results → {out_dir}")


def cmd_optimize(args):
    scores, labels = _load_dataset()
    res = optimize_thresholds(scores, labels, n_trials=args.trials, seed=0)

    payload = {
        "version": args.version,
        "date": _today(),
        "model": "musiq-ava",
        "notes": f"optuna TPE {args.trials} trials (thresholds only)",
        "thresholds": list(res.thresholds),
    }
    out = Path(args.out) if args.out else BENCH_DIR / "params.candidate.json"
    save_params(out, payload)

    version = version_with_sha(args.version)
    out_dir = RESULTS_DIR / version
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps({"val": res.metrics.to_dict()}, indent=2))
    (out_dir / "params.json").write_text(json.dumps(payload, indent=2))

    print(f"best Spearman={res.metrics.spearman:.3f}  MAE={res.metrics.mae:.2f}")
    print(f"candidate → {out}")


# Delete cmd_ablate entirely.

def _build_parser():
    ap = argparse.ArgumentParser(description="Focal bench orchestration")
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("download", help="fetch AVA subset + write labels.csv")
    d.add_argument("--sample", type=int, default=500)
    d.set_defaults(func=lambda a: cmd_download(a))

    s = sub.add_parser("score", help="run FocalScorer on images (cached)")
    s.set_defaults(func=lambda a: cmd_score(a))

    e = sub.add_parser("eval", help="evaluate thresholds params.json against labels")
    e.add_argument("--params", default=str(BENCH_DIR / "params.current.json"))
    e.set_defaults(func=lambda a: cmd_eval(a))

    o = sub.add_parser("optimize", help="Optuna TPE search for 4 bucket thresholds")
    o.add_argument("--trials", type=int, default=500)
    o.add_argument("--version", default="v0.4.0")
    o.add_argument("--out")
    o.set_defaults(func=lambda a: cmd_optimize(a))

    l = sub.add_parser("leaderboard", help="regenerate LEADERBOARD.md")
    l.set_defaults(func=lambda a: cmd_leaderboard(a))

    return ap
```

- [ ] **Step 4: Implement `single_model.py`**

```python
# testing/bench/bench/single_model.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import optuna
import pandas as pd
from .metrics import compute_metrics, Metrics


def stars_from_thresholds(scores: np.ndarray, thresholds: tuple[float, float, float, float]) -> np.ndarray:
    t1, t2, t3, t4 = thresholds
    stars = np.ones_like(scores, dtype=int)
    stars = stars + (scores > t1).astype(int)
    stars = stars + (scores > t2).astype(int)
    stars = stars + (scores > t3).astype(int)
    stars = stars + (scores > t4).astype(int)
    return np.clip(stars, 1, 5)


def bucket_stars(score: float, thresholds: tuple[float, float, float, float]) -> int:
    t1, t2, t3, t4 = thresholds
    if score <= t1: return 1
    if score <= t2: return 2
    if score <= t3: return 3
    if score <= t4: return 4
    return 5


@dataclass
class OptimizeResult:
    thresholds: tuple[float, float, float, float]
    metrics: Metrics


def optimize_thresholds(
    scores_df: pd.DataFrame, labels_df: pd.DataFrame, *, n_trials: int = 500, seed: int = 0
) -> OptimizeResult:
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
    t = (study.best_params["t1"], study.best_params["t2"],
         study.best_params["t3"], study.best_params["t4"])
    pred = stars_from_thresholds(s, t)
    m = compute_metrics(y, pred)
    return OptimizeResult(thresholds=t, metrics=m)
```

- [ ] **Step 5: Run tests**

```
cd testing/bench
python3 -m pytest tests/test_run.py -v
```

Expected: new tests pass. Delete old fixtures that reference ensemble.

- [ ] **Step 6: Sanity regression — verify pipeline still reproduces Spearman**

```
# Temporarily update Scorer JSON output schema locally to single-column first
# (see Task 11), or patch _load_dataset to rename legacy columns.
# If FocalScorer not yet rewritten, skip and resume after Task 11.
python3 run.py eval 2>&1 | tail -5
```

Expected (once Task 11 lands): `Spearman=0.76..`. If Task 11 not done, skip — this is the merge gate for Task 11.

- [ ] **Step 7: Commit**

```
git add testing/bench/run.py testing/bench/bench/single_model.py testing/bench/tests/test_run.py
git commit -m "feat(bench): single-model run.py (thresholds-only optimize, no ablate)"
```

---

## Task 11: Caller Migration — FocalScorer, RatingQueue, DetailView, MetadataWriter

**Files:**
- Modify: `FocalScorer/Scorer.swift`
- Modify: `ImageRater/Pipeline/RatingQueue.swift`
- Modify: `ImageRater/UI/DetailView.swift`
- Modify: `ImageRater/Export/MetadataWriter.swift`
- Modify: `ImageRaterTests/FocalScorerSmokeTests.swift`

- [ ] **Step 1: Grep ensemble field usage to inventory call sites**

```
grep -rn "topiqTechnicalScore\|topiqAestheticScore\|clipIQAScore\|combinedQualityScore\|clipEmbedding\|resolvedClipLogitScale\|resolvedCullStrictness\|defaultBucketEdge" ImageRater/ FocalScorer/ ImageRaterTests/
```

Expected: 10–30 hits. Each one needs migration to `musiqAesthetic` + `stars`.

- [ ] **Step 2: Rewrite `Scorer.swift`**

```swift
// FocalScorer/Scorer.swift
import Foundation
import CoreImage

enum Scorer {
    struct OutputImage: Codable {
        let filename: String
        let musiqAesthetic: Float
        let stars: Int
    }
    struct Output: Codable {
        let generatedAt: String
        let modelVersion: String
        let images: [OutputImage]
    }

    static let supportedExts: Set<String> = ["jpg", "jpeg", "png", "raf", "nef", "arw", "cr3", "dng"]

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
                    musiqAesthetic: s.musiqAesthetic,
                    stars: s.stars
                ))
            }
        }
        let out = Output(
            generatedAt: ISO8601DateFormatter().string(from: Date()),
            modelVersion: "musiq-ava",
            images: results
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(out).write(to: outputURL)
        FileHandle.standardError.write(Data("wrote \(results.count) scores -> \(outputURL.path)\n".utf8))
    }
}
```

- [ ] **Step 3: Update bench `score.py` JSON parser (if needed)**

`testing/bench/bench/score.py` uses `pd.DataFrame(blob["images"])` which adapts automatically. Verify:

```
python3 -c "import pandas as pd, json; df = pd.DataFrame([{'filename':'a.jpg','musiqAesthetic':5.5,'stars':3}]); print(df.columns.tolist())"
```

Expected: `['filename', 'musiqAesthetic', 'stars']`.

- [ ] **Step 4: CoreData migration — add v4 model with MUSIQ fields**

Existing model `ImageRater/CoreData/ImageRater.xcdatamodeld/ImageRater 3.xcdatamodel` stores `topiqTechnicalScore`, `topiqAestheticScore`, `clipEmbedding`. Add a new version:

1. Duplicate `ImageRater 3.xcdatamodel` → `ImageRater 4.xcdatamodel` (Xcode: Editor → Add Model Version). If scripted, copy the directory and increment `currentVersion` in `.xcdatamodeld/.xccurrentversion`.
2. In `ImageRater 4.xcdatamodel/contents`:
   - Remove `topiqTechnicalScore`, `topiqAestheticScore`, `clipEmbedding` attributes.
   - Add `<attribute name="musiqAesthetic" optional="YES" attributeType="Float" usesScalarValueType="YES"/>`.
   - Add `<attribute name="stars" optional="YES" attributeType="Integer 16" usesScalarValueType="YES"/>` (if not present on `ImageRecord`).
3. Mark as current version.
4. CoreData lightweight migration handles column drops + adds automatically (enabled by default in `NSPersistentContainer` with `shouldInferMappingModelAutomatically = true`; verify in `PersistenceController.swift` / equivalent). Existing rows: dropped columns discarded, new `musiqAesthetic`/`stars` populated as `nil` until re-rated.

Grep `topiqTechnicalScore`/`topiqAestheticScore`/`clipEmbedding`/`combinedQualityScore` across `ImageRater/` — remove readers/writers. Replace with `musiqAesthetic`/`stars`.

- [ ] **Step 5: Migrate `RatingQueue.swift`**

- Grep for `rate(image:` calls; remove `weights:` parameter.
- Replace stored subscore writes with `ratedScores.musiqAesthetic` + `ratedScores.stars`.

- [ ] **Step 6: Migrate `DetailView.swift`, `MetadataWriter.swift`**

- `DetailView`: collapse subscore rows to single "Aesthetic: X.XX" line + star count.
- `MetadataWriter` XMP: replace `Focal:TOPIQTechnical`/`Focal:TOPIQAesthetic`/`Focal:CLIPScore` with `Focal:MUSIQAesthetic` and `Focal:Stars`.

- [ ] **Step 7: Update `FocalScorerSmokeTests`**

Ensure it asserts on `musiqAesthetic` and `stars` fields.

- [ ] **Step 8: Build + run full test suite**

```
xcodebuild -scheme ImageRater -destination 'platform=macOS,arch=arm64' test 2>&1 | tail -20
```

Expected: clean build, all Swift tests pass.

If build fails at any site with `topiqTechnicalScore not found`, the grep step missed a caller — update and re-run.

- [ ] **Step 9: Commit**

```
git add FocalScorer/Scorer.swift \
        ImageRater/Pipeline/RatingQueue.swift \
        ImageRater/UI/DetailView.swift \
        ImageRater/Export/MetadataWriter.swift \
        ImageRaterTests/FocalScorerSmokeTests.swift \
        "ImageRater/CoreData/ImageRater.xcdatamodeld/ImageRater 4.xcdatamodel" \
        ImageRater/CoreData/ImageRater.xcdatamodeld/.xccurrentversion
git commit -m "refactor: migrate callers to single-model RatedScores {musiqAesthetic, stars}"
```

---

## Task 12: Delete Dead Ensemble Code + Models

**Files:**
- Delete: `ImageRater/MLModels/topiq-nr.mlmodelc`
- Delete: `ImageRater/MLModels/topiq-swin.mlmodelc`
- Delete: `ImageRater/MLModels/clip-vision.mlmodelc`
- Delete: `ImageRater/Pipeline/CLIPTextEmbeddings.swift`
- Delete: `testing/bench/bench/clip_iqa.py` + `testing/bench/tests/test_clip_iqa.py`
- Delete: `testing/bench/bench/ensemble.py` + `testing/bench/tests/test_ensemble.py`
- Delete: `testing/bench/bench/ablation.py` + `testing/bench/tests/test_ablation.py`
- Delete: `testing/bench/bench/prompt_embeddings.json`
- Delete: `testing/bench/bench/optimize.py` (ensemble optimizer; replaced by `single_model.optimize_thresholds`) — **confirm not referenced elsewhere first**
- Modify: `project.yml` (drop old .mlmodelc from FocalScorer sources + app resources)

- [ ] **Step 1: Verify each delete candidate has zero references**

```
for sym in combinedQuality clipIQAScore EnsembleParams stars_from_subscores run_ablation clip_iqa_score load_prompt_embeddings; do
  echo "=== $sym ==="
  grep -rn "$sym" ImageRater/ FocalScorer/ ImageRaterTests/ testing/bench/ 2>/dev/null | grep -v ".git" | head
done
```

Expected: only hits in files also being deleted. Any unexpected hit = go back and remove the caller.

- [ ] **Step 2: Delete files**

```
rm -rf ImageRater/MLModels/topiq-nr.mlmodelc ImageRater/MLModels/topiq-swin.mlmodelc ImageRater/MLModels/clip-vision.mlmodelc
rm ImageRater/Pipeline/CLIPTextEmbeddings.swift
rm testing/bench/bench/clip_iqa.py testing/bench/tests/test_clip_iqa.py
rm testing/bench/bench/ensemble.py testing/bench/tests/test_ensemble.py
rm testing/bench/bench/ablation.py testing/bench/tests/test_ablation.py
rm testing/bench/bench/prompt_embeddings.json
# optimize.py: verify no importers outside bench itself
grep -rn "from bench.optimize\|from .optimize" testing/bench/ | grep -v "bench/optimize.py"
# if output is empty, it's safe:
rm testing/bench/bench/optimize.py testing/bench/tests/test_optimize.py 2>/dev/null || true
```

- [ ] **Step 3: Update `project.yml`**

Edit the `Focal` / `ImageRater` target's resource/source list and the `FocalScorer` target's source list: drop references to deleted files and `.mlmodelc` bundles. Add `ImageRater/MLModels/musiq-ava.mlpackage` to the resource list (Xcode auto-compiles to `.mlmodelc`).

**Pre-regen sanity (pbxproj is generated):** worktree may have unstaged `Focal.xcodeproj/project.pbxproj` edits from earlier sessions. xcodegen overwrites pbxproj wholesale — any manual edits not mirrored in `project.yml` will be lost. Verify:

```
git diff --stat Focal.xcodeproj/project.pbxproj
```

If non-empty: inspect the diff, port any missing target/file additions into `project.yml`, then regenerate.

```
xcodegen generate
```

- [ ] **Step 4: Build + test**

```
xcodebuild -scheme ImageRater -destination 'platform=macOS,arch=arm64' test 2>&1 | tail -20
cd testing/bench && python3 -m pytest 2>&1 | tail -10
```

Expected: clean build, all tests pass. Swift build will resolve `musiq-ava` resource bundled correctly.

- [ ] **Step 5: Commit**

```
git add -A
git commit -m "refactor: delete TOPIQ/CLIP ensemble code + models (replaced by MUSIQ)"
```

---

## Task 13: Xcode Bundle Integration for `musiq-ava.mlpackage`

**Files:**
- Modify: `project.yml` (may already be done in Task 12; verify)
- Modify: `Focal.xcodeproj/project.pbxproj` (regenerated via xcodegen)

- [ ] **Step 1: Verify app bundle contains compiled model**

```
xcodebuild -scheme ImageRater -configuration Release -destination 'platform=macOS,arch=arm64' build
APP=$(xcodebuild -scheme ImageRater -configuration Release -destination 'platform=macOS,arch=arm64' -showBuildSettings | awk -F= '/CODESIGNING_FOLDER_PATH/ {print $2}' | xargs)
ls "$APP/Contents/Resources/" | grep -i musiq
```

Expected: `musiq-ava.mlmodelc` present. Old `topiq-nr.mlmodelc` etc. absent.

- [ ] **Step 2: Launch-smoke test**

Add a new test in `FocalScorerSmokeTests.swift` or `IntegrationTests.swift`:

```swift
func test_musiq_model_loads() throws {
    let models = try RatingPipeline.loadBundledModels()
    XCTAssertNotNil(models.musiq)
}

func test_musiq_end_to_end_on_solid_image() async throws {
    let models = try RatingPipeline.loadBundledModels()
    let cg = TestImage.solid(256, 256)
    let r = await RatingPipeline.rate(image: cg, models: models)
    guard case .rated(let s) = r else { XCTFail(); return }
    XCTAssertTrue(s.musiqAesthetic.isFinite)
    XCTAssertTrue((1...5).contains(s.stars))
}
```

- [ ] **Step 3: Run**

```
xcodebuild -scheme ImageRater -destination 'platform=macOS,arch=arm64' test \
  -only-testing:ImageRaterTests/IntegrationTests
```

Expected: passes. If `modelNotFound("musiq-ava")`, recheck `project.yml` resource entries.

- [ ] **Step 4: Commit**

```
git add project.yml Focal.xcodeproj/project.pbxproj ImageRaterTests/IntegrationTests.swift
git commit -m "feat: bundle musiq-ava.mlpackage into Focal app + end-to-end smoke test"
```

---

## Task 14: Parity Gate — CoreML vs pyiqa on AVA-100

**Files:**
- Create: `testing/bench/tests/test_musiq_parity.py`
- Create: `testing/bench/scripts/run_coreml_on_sample.py`

- [ ] **Step 1: Build Release FocalScorer binary**

```
xcodebuild -scheme FocalScorer -configuration Release -destination 'platform=macOS,arch=arm64' build
```

- [ ] **Step 2: Write failing parity test**

```python
# testing/bench/tests/test_musiq_parity.py
"""CoreML MUSIQ vs pyiqa reference — merge gate."""
from __future__ import annotations
import json, subprocess
from pathlib import Path

import pandas as pd
import pytest

from bench.parity import pyiqa_scores, compare

ROOT = Path(__file__).resolve().parents[2]
IMAGES = ROOT / "testing" / "bench" / "data" / "ava" / "images"
TMP_OUT = ROOT / "testing" / "bench" / ".cache" / "parity_coreml.json"
SAMPLE = 100


def _locate_scorer_bin():
    import glob, os
    home = os.path.expanduser("~")
    cand = glob.glob(f"{home}/Library/Developer/Xcode/DerivedData/**/Build/Products/Release/FocalScorer", recursive=True)
    cand = [c for c in cand if os.access(c, os.X_OK)]
    cand.sort(key=os.path.getmtime, reverse=True)
    if not cand:
        pytest.skip("FocalScorer Release binary not built")
    return cand[0]


@pytest.mark.skipif(not IMAGES.exists(), reason="AVA sample missing")
def test_coreml_parity_ava_100():
    bin_ = _locate_scorer_bin()
    # Run FocalScorer on first SAMPLE images via a temp subfolder.
    tmp_dir = ROOT / "testing" / "bench" / ".cache" / "parity_sample"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(IMAGES.glob("*.jpg"))[:SAMPLE]
    for f in files:
        dst = tmp_dir / f.name
        if not dst.exists():
            dst.symlink_to(f)
    TMP_OUT.parent.mkdir(parents=True, exist_ok=True)
    subprocess.check_call([bin_, str(tmp_dir), str(TMP_OUT)])

    blob = json.loads(TMP_OUT.read_text())
    sw_df = pd.DataFrame(blob["images"]).rename(columns={"musiqAesthetic": "coreml_score"})

    py_df = pyiqa_scores(tmp_dir, limit=SAMPLE)
    report = compare(py_df, sw_df)

    print(f"\nParity report: {report}")
    assert report["n"] >= SAMPLE - 5, f"joined rows too few: {report}"
    assert report["spearman"] >= 0.97, f"Spearman {report['spearman']} < 0.97"
    assert report["max_abs_delta"] <= 0.10, f"max |Δ| {report['max_abs_delta']} > 0.10"
    assert report["mean_abs_delta"] <= 0.03, f"mean |Δ| {report['mean_abs_delta']} > 0.03"
```

- [ ] **Step 3: Run**

```
cd testing/bench
python3 -m pytest tests/test_musiq_parity.py -v -s
```

Expected: PASS. Typical report Spearman ≈ 0.995, max |Δ| ≤ 0.05, mean |Δ| ≤ 0.02.

**If it fails:**
- `Spearman < 0.97`: preprocessor divergence. Re-run Task 7 preprocessor tests with tighter tolerance. Common culprits: bicubic kernel sign error, nearest-interp formula, BGRA/RGB swap.
- `max |Δ| 0.10 < x < 0.3`: probable fp32 vs fp64 drift in bicubic; kernel OK, just numerical. Acceptable to relax to `max_abs_delta ≤ 0.15` with comment if investigation rules out semantic bug.
- `max |Δ| > 0.3`: semantic bug (wrong scale, wrong mask, wrong channel order). Fix before shipping.

- [ ] **Step 4: Commit**

```
git add testing/bench/tests/test_musiq_parity.py
git commit -m "test: CoreML vs pyiqa parity gate (Spearman >= 0.97 on AVA-100)"
```

---

## Task 15: End-to-End Validation + Leaderboard Update

**Files:**
- Run: `python3 run.py eval`
- Run: `python3 run.py leaderboard`
- Modify: `testing/bench/LEADERBOARD.md` (regenerated)

- [ ] **Step 1: Fresh bench eval with CoreML scorer + v0.4.0 params**

```
rm -rf testing/bench/.cache/scores_*.json   # invalidate old FocalScorer cache
cd testing/bench && python3 run.py score   # re-scores via new FocalScorer
python3 run.py eval
```

Expected: `Spearman=0.76..  MAE=0.66  ±1=88..91%`. Delta from the pre-CoreML 0.764 ≤ 0.01.

- [ ] **Step 2: Regenerate leaderboard**

```
python3 run.py leaderboard
cat LEADERBOARD.md
```

Expected: v0.4.0 row on top.

- [ ] **Step 3: Manual UX checkpoint**

Open Focal.app, import 50+ real photos, verify:

- Star distribution not uniformly 3★.
- Bulk import time ≤ 2× pre-CoreML baseline (should be ~equal or faster).
- Spot-check 10 files — star in UI == star from `python3 run.py eval`'s `scores.parquet`.

- [ ] **Step 4: Final commit**

```
git add testing/bench/LEADERBOARD.md
git commit -m "feat: ship v0.4.0 single-model MUSIQ rating pipeline"
```

- [ ] **Step 5: Merge-ready check**

```
xcodebuild -scheme ImageRater -destination 'platform=macOS,arch=arm64' test 2>&1 | tail -5
cd testing/bench && python3 -m pytest 2>&1 | tail -5
git status
git log --oneline main..HEAD | head
```

Expected:
- All Swift + Python tests green.
- `git status` clean.
- Commit log shows ~14 commits since main; the last is `feat: ship v0.4.0 …`.

---

## Ship Criteria (copied from spec; re-verify)

- [ ] Parity: Spearman ≥ 0.97 CoreML vs pyiqa on AVA-100.
- [ ] Per-image max |Δ| ≤ 0.10 raw MUSIQ score.
- [ ] Bench `run.py eval` Spearman ≥ 0.70 on 500-AVA.
- [ ] No references to `topiqTechnicalScore` / `clipEmbedding` / `combinedQuality` / `EnsembleParams` anywhere in repo (`grep -r` returns empty).
- [ ] `params.current.json` is v0.4.0 schema.
- [ ] App builds, runs, and rates images end-to-end.
- [ ] All tests green (`xcodebuild test` + `pytest`).

## Rollback Plan

```
git revert <range-of-task-commits>
xcodegen generate
```

Ensemble models in `ImageRater/MLModels/` are restored from git history. Old `RatingPipeline` signature restored. No data migration needed (user-facing persistence unaffected — RatedScores is a computed value, not stored persistently except as stars int which is forward-compatible).
