# Image Rating Accuracy & Consistency — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace NIMA models with a 3-model ensemble (TOPIQ-NR + TOPIQ-Swin + CLIP-IQA+), add MMR diversity scoring, and expose all scores in the detail sidebar and filter panel.

**Architecture:** Two-pass pipeline — pass 1 runs 3 ML inferences + Vision/CI concurrently per image and writes scores to CoreData with `processState = "rated"`; pass 2 runs session-level MMR clustering and percentile normalization, writes `clusterID`, `clusterRank`, `diversityFactor`, `finalScore`, `ratingStars`, and sets `processState = "done"`. All models bundled in app binary (no download). New `DiversityScorer.swift` contains the pure-algorithm pass 2 logic.

**Tech Stack:** Swift 5.9 · CoreML · CoreData lightweight migration · SwiftUI · Accelerate (vDSP_dotpr) · Python 3 + pyiqa + coremltools 8 + open_clip (dev-only conversion)

**Spec:** `docs/superpowers/specs/2026-04-14-image-rating-accuracy-design.md`

---

## File Map

| File | Action | Purpose |
|------|--------|---------|
| `scripts/convert_topiq.py` | Create | Convert TOPIQ-NR, TOPIQ-Swin, CLIP vision encoder to CoreML; emit CLIPTextEmbeddings.swift |
| `models/topiq-nr.mlpackage` | Create (generated) | TOPIQ-NR ResNet50 technical quality scorer |
| `models/topiq-swin.mlpackage` | Create (generated) | TOPIQ-Swin aesthetic quality scorer |
| `models/clip-vision.mlpackage` | Create (generated) | CLIP ViT-B/32 vision encoder → 512-dim embedding |
| `ImageRater/Pipeline/CLIPTextEmbeddings.swift` | Create (generated) | Pre-computed "Good photo"/"Bad photo" CLIP text embeddings |
| `ImageRater/CoreData/ImageRater.xcdatamodeld/` | Modify | Add v2 model with 11 new fields; set as current version |
| `ImageRater/Models/RatingResult.swift` | Modify | New `RatedScores` struct with 3 scores + embedding |
| `ImageRater/Pipeline/CullPipeline.swift` | Modify | Add `CullScores` struct; return numeric blurScore + exposureScore; re-enable exposure check |
| `ImageRater/Pipeline/RatingPipeline.swift` | Rewrite | Ensemble inference (3 models async concurrent); bundled model loading |
| `ImageRater/Pipeline/DiversityScorer.swift` | Create | Cosine similarity, threshold clustering, MMR ordering, percentile normalization |
| `ImageRater/Pipeline/ProcessingQueue.swift` | Modify | Two-pass architecture; replace ModelStore with bundled model loading |
| `ImageRater/UI/DetailView.swift` | Modify | Scores bars panel + characteristics panel in sidebar |
| `ImageRater/UI/Components/ThumbnailCell.swift` | Modify | "C" cluster-rep badge; dim overlay for diversityFactor < 0.60 |
| `ImageRater/UI/Components/RatingFilterView.swift` | Modify | Score sliders + blur/exposure toggles + diversity toggles |
| `ImageRater/App/ContentView.swift` | Modify | New filter state vars; filteredImages logic; UserDefaults persistence |
| `project.yml` | Modify | Add three .mlpackage files to Copy Bundle Resources |
| `ImageRaterTests/ImageRecordMigrationTests.swift` | Create | CoreData migration: new fields read/write |
| `ImageRaterTests/CullPipelineTests.swift` | Modify | CullScores return type tests |
| `ImageRaterTests/RatingPipelineTests.swift` | Modify | Ensemble formula + CLIP-IQA+ unit tests |
| `ImageRaterTests/DiversityScorerTests.swift` | Create | Cosine sim, clustering, MMR, percentile normalization tests |
| `ImageRaterTests/ProcessingQueueTests.swift` | Modify | Two-pass state transition tests |

---

## Task 0: Convert TOPIQ and CLIP models to CoreML (Python, dev-only)

Run once. Commit generated artifacts. No Xcode changes yet.

**Files:**
- Create: `scripts/convert_topiq.py`
- Output: `models/topiq-nr.mlpackage`, `models/topiq-swin.mlpackage`, `models/clip-vision.mlpackage`
- Output: `ImageRater/Pipeline/CLIPTextEmbeddings.swift`

- [ ] **Step 1: Install Python dependencies**

```bash
pip install pyiqa coremltools torch torchvision open_clip_torch
```

- [ ] **Step 2: Write convert_topiq.py**

```python
#!/usr/bin/env python3
"""
Convert TOPIQ-NR, TOPIQ-Swin, and CLIP ViT-B/32 vision encoder to CoreML.

Outputs (relative to repo root):
  models/topiq-nr.mlpackage         (224×224 RGB, Float score [0,1])
  models/topiq-swin.mlpackage       (384×384 RGB, Float score [0,1])
  models/clip-vision.mlpackage      (224×224 RGB, 512-dim Float embedding, L2-normalised)
  ImageRater/Pipeline/CLIPTextEmbeddings.swift  (pre-computed text embeddings)

Usage:
  cd /path/to/image-rating && python scripts/convert_topiq.py
"""
import os, json
import torch
import numpy as np
import coremltools as ct
import pyiqa
import open_clip

REPO_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(REPO_ROOT, 'models')
SWIFT_DIR  = os.path.join(REPO_ROOT, 'ImageRater', 'Pipeline')


class TOPIQWrapper(torch.nn.Module):
    """Strip pyiqa metric wrapper; expose raw net for tracing."""
    def __init__(self, metric):
        super().__init__()
        self.net = metric.net

    def forward(self, x):
        score = self.net(x)
        return torch.clamp(score.squeeze(-1), 0.0, 1.0).reshape(1)


def _convert(wrapper, input_size, name):
    dummy = torch.randn(1, 3, input_size, input_size)
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, dummy)
    mlmodel = ct.convert(
        traced,
        inputs=[ct.ImageType(
            name="image",
            shape=(1, 3, input_size, input_size),
            scale=1.0 / 255.0,
            bias=[0, 0, 0],
            color_layout=ct.colorlayout.RGB,
        )],
        outputs=[ct.TensorType(name="score", dtype=np.float32)],
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.macOS14,
    )
    out = os.path.join(MODELS_DIR, f'{name}.mlpackage')
    mlmodel.save(out)
    print(f'  → {out}')
    return out


def convert_topiq_nr():
    print('Converting TOPIQ-NR (ResNet50, 224×224)...')
    metric = pyiqa.create_metric('topiq_nr', as_loss=False, device='cpu')
    metric.eval()
    _convert(TOPIQWrapper(metric), 224, 'topiq-nr')


def convert_topiq_swin():
    # topiq_nr-flive is the Swin-B variant; verify input size from pyiqa config
    print('Converting TOPIQ-Swin (384×384)...')
    metric = pyiqa.create_metric('topiq_nr-flive', as_loss=False, device='cpu')
    metric.eval()
    # Confirm actual input size used by pyiqa transform
    input_size = 384  # Swin-B default; override if pyiqa logs a different crop size
    print(f'  Using input_size={input_size} — verify against pyiqa transform output')
    _convert(TOPIQWrapper(metric), input_size, 'topiq-swin')


def convert_clip_vision():
    print('Converting CLIP ViT-B/32 vision encoder (224×224)...')
    model, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
    model.eval()

    class VisionWrapper(torch.nn.Module):
        def __init__(self, enc):
            super().__init__()
            self.enc = enc
        def forward(self, x):
            f = self.enc(x)
            return torch.nn.functional.normalize(f, dim=-1)

    wrapper = VisionWrapper(model.visual)
    wrapper.eval()
    dummy = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, dummy)

    mlmodel = ct.convert(
        traced,
        inputs=[ct.ImageType(
            name="image",
            shape=(1, 3, 224, 224),
            # OpenAI CLIP normalisation constants
            scale=1.0 / 255.0,
            bias=[0.48145466, 0.4578275, 0.40821073],
            color_layout=ct.colorlayout.RGB,
        )],
        outputs=[ct.TensorType(name="embedding", dtype=np.float32)],
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.macOS14,
    )
    out = os.path.join(MODELS_DIR, 'clip-vision.mlpackage')
    mlmodel.save(out)
    print(f'  → {out}')
    return model  # return full model for text embedding step


def emit_swift_embeddings(clip_model):
    print('Computing CLIP-IQA+ text embeddings ("Good photo" / "Bad photo")...')
    tok = open_clip.get_tokenizer('ViT-B-32')
    tokens = tok(["Good photo", "Bad photo"])
    with torch.no_grad():
        feats = clip_model.encode_text(tokens)
        feats = torch.nn.functional.normalize(feats, dim=-1)
    good = feats[0].numpy().tolist()
    bad  = feats[1].numpy().tolist()

    swift = f'''// Auto-generated by scripts/convert_topiq.py — do not edit manually.
// CLIP ViT-B/32 text embeddings for CLIP-IQA+ antonym quality prompts.

import Foundation

enum CLIPTextEmbeddings {{
    /// L2-normalised 512-dim embedding for "Good photo" (CLIP ViT-B/32, OpenAI weights).
    static let goodPhoto: [Float] = {good}

    /// L2-normalised 512-dim embedding for "Bad photo" (CLIP ViT-B/32, OpenAI weights).
    static let badPhoto: [Float] = {bad}
}}
'''
    out = os.path.join(SWIFT_DIR, 'CLIPTextEmbeddings.swift')
    with open(out, 'w') as f:
        f.write(swift)
    print(f'  → {out}')


if __name__ == '__main__':
    os.makedirs(MODELS_DIR, exist_ok=True)
    convert_topiq_nr()
    convert_topiq_swin()
    clip = convert_clip_vision()
    emit_swift_embeddings(clip)
    print('\nDone. Next: add .mlpackage files to project.yml Copy Bundle Resources.')
```

- [ ] **Step 3: Run the script**

```bash
cd /Users/dknathalage/repos/image-rating
python scripts/convert_topiq.py
```

Expected: four files created — three `.mlpackage` dirs and `CLIPTextEmbeddings.swift`.

- [ ] **Step 4: Verify score ranges are [0, 1]**

```python
import pyiqa, torch
m = pyiqa.create_metric('topiq_nr', device='cpu')
m.eval()
img = torch.rand(1, 3, 224, 224)
with torch.no_grad():
    s = m(img).item()   # use __call__, not m.net — more stable across pyiqa versions
assert 0.0 <= s <= 1.0, f"Out of range: {s}"
print(f"topiq_nr score: {s:.4f}  ✓")
```

- [ ] **Step 5: Commit**

```bash
git add scripts/convert_topiq.py models/topiq-nr.mlpackage models/topiq-swin.mlpackage models/clip-vision.mlpackage ImageRater/Pipeline/CLIPTextEmbeddings.swift
git commit -m "feat: add TOPIQ + CLIP CoreML models and CLIP-IQA+ text embeddings"
```

---

## Task 1: CoreData — Add new fields (lightweight migration)

**Files:**
- Modify: `ImageRater/CoreData/ImageRater.xcdatamodeld/`
- Create: `ImageRaterTests/ImageRecordMigrationTests.swift`

- [ ] **Step 1: Write the failing test**

```swift
// ImageRaterTests/ImageRecordMigrationTests.swift
import XCTest
import CoreData
@testable import ImageRater

final class ImageRecordMigrationTests: XCTestCase {

    func testNewFieldsReadWrite() throws {
        let ctx = makeInMemoryContext()

        let session = Session(context: ctx)
        session.id = UUID(); session.createdAt = Date(); session.folderPath = "/tmp"

        let r = ImageRecord(context: ctx)
        r.id = UUID(); r.filePath = "/tmp/a.jpg"; r.processState = "pending"; r.session = session

        r.topiqTechnicalScore = 0.75
        r.topiqAestheticScore = 0.82
        r.clipIQAScore        = 0.68
        r.combinedQualityScore = 0.77
        r.finalScore          = 0.65
        r.diversityFactor     = 0.85
        r.clipEmbedding       = Data(repeating: 1, count: 512 * 4)
        r.clusterID           = 3
        r.clusterRank         = 2
        r.blurScore           = 420.5
        r.exposureScore       = 0.3

        try ctx.save()

        ctx.refresh(r, mergeChanges: false)
        XCTAssertEqual(r.topiqTechnicalScore,  0.75,  accuracy: 0.001)
        XCTAssertEqual(r.topiqAestheticScore,  0.82,  accuracy: 0.001)
        XCTAssertEqual(r.clipIQAScore,         0.68,  accuracy: 0.001)
        XCTAssertEqual(r.combinedQualityScore, 0.77,  accuracy: 0.001)
        XCTAssertEqual(r.finalScore,           0.65,  accuracy: 0.001)
        XCTAssertEqual(r.diversityFactor,      0.85,  accuracy: 0.001)
        XCTAssertEqual(r.clipEmbedding?.count, 512 * 4)
        XCTAssertEqual(r.clusterID,   3)
        XCTAssertEqual(r.clusterRank, 2)
        XCTAssertEqual(r.blurScore,     420.5, accuracy: 0.1)
        XCTAssertEqual(r.exposureScore, 0.3,   accuracy: 0.001)
    }

    private func makeInMemoryContext() -> NSManagedObjectContext {
        let c = NSPersistentContainer(name: "ImageRater")
        let d = NSPersistentStoreDescription()
        d.type = NSInMemoryStoreType
        c.persistentStoreDescriptions = [d]
        c.loadPersistentStores { _, e in if let e { XCTFail("\(e)") } }
        return c.viewContext
    }
}
```

- [ ] **Step 2: Run — verify compile error**

```bash
xcodebuild test -project ImageRater.xcodeproj -scheme ImageRater \
  -only-testing ImageRaterTests/ImageRecordMigrationTests 2>&1 | grep "error:"
```

Expected: `value of type 'ImageRecord' has no member 'topiqTechnicalScore'`

- [ ] **Step 3: Add CoreData model version 2**

In Xcode: select `ImageRater.xcdatamodeld` → Editor → Add Model Version → name `"ImageRater 2"` → Finish.

In the `ImageRater 2` entity editor for `ImageRecord`, add these attributes:

| Name | Type | Optional | Default |
|------|------|----------|---------|
| `topiqTechnicalScore` | Float | NO | 0.0 |
| `topiqAestheticScore` | Float | NO | 0.0 |
| `clipIQAScore` | Float | NO | 0.0 |
| `combinedQualityScore` | Float | NO | 0.0 |
| `finalScore` | Float | NO | 0.0 |
| `diversityFactor` | Float | NO | 1.0 |
| `clipEmbedding` | Binary Data | YES | — |
| `clusterID` | Integer 32 | NO | -1 |
| `clusterRank` | Integer 32 | NO | 0 |
| `blurScore` | Float | NO | 0.0 |
| `exposureScore` | Float | NO | 0.0 |

Set `ImageRater 2` as the current version: select `.xcdatamodeld` → File Inspector → "Model Version" → choose `ImageRater 2`.

**Note:** Xcode auto-generates the `ImageRecord+CoreData.swift` extension from the model. Do NOT create this file manually — it will be regenerated on every build and will conflict.

- [ ] **Step 4: Verify lightweight migration already enabled — no change needed**

`ImageRater/CoreData/PersistenceController.swift` lines 16–18 already set both migration options:

```swift
let description = container.persistentStoreDescriptions.first!
description.setOption(true as NSNumber, forKey: NSMigratePersistentStoresAutomaticallyOption)
description.setOption(true as NSNumber, forKey: NSInferMappingModelAutomaticallyOption)
```

No edit required. Proceed to Step 5.

- [ ] **Step 5: Run test — verify passes**

```bash
xcodebuild test -project ImageRater.xcodeproj -scheme ImageRater \
  -only-testing ImageRaterTests/ImageRecordMigrationTests 2>&1 | grep -E "passed|failed|error:"
```

Expected: `Test Suite 'ImageRecordMigrationTests' passed`

- [ ] **Step 6: Commit**

```bash
git add ImageRater/CoreData/ ImageRater/PersistenceController.swift ImageRaterTests/ImageRecordMigrationTests.swift
git commit -m "feat: CoreData v2 migration — add ensemble scores, diversity, and cull characteristic fields"
```

---

## Task 2: CullPipeline — Return CullScores struct

**Files:**
- Modify: `ImageRater/Pipeline/CullPipeline.swift`
- Modify: `ImageRaterTests/CullPipelineTests.swift`

- [ ] **Step 1: Add failing tests**

Add `import AppKit` at the top of `CullPipelineTests.swift` if not already present (required for `NSColor` in the helpers below).

Add these methods to `CullPipelineTests`:

```swift
func testCullReturnsBlurScoreForSharpImage() async {
    // A synthetic sharp image (high-frequency checkerboard) → high blurScore
    let sharp = makeCheckerboardImage(size: 256)
    let result = await CullPipeline.cull(
        image: sharp, blurThreshold: 100, earThreshold: 0.15, exposureLeniency: 0.95)
    XCTAssertGreaterThan(result.blurScore, 0, "Sharp image must have non-zero blurScore")
}

func testCullReturnsMeasurableExposureScoreForWhiteImage() async {
    let white = makeSolidColorImage(size: 256, color: .white)
    let result = await CullPipeline.cull(
        image: white, blurThreshold: 0, earThreshold: 0.15, exposureLeniency: 0.95)
    // All-white = overexposed → positive exposureScore
    XCTAssertGreaterThan(result.exposureScore, 0,
        "All-white image should have positive exposureScore")
}

func testCullReturnsMeasurableExposureScoreForBlackImage() async {
    let black = makeSolidColorImage(size: 256, color: .black)
    let result = await CullPipeline.cull(
        image: black, blurThreshold: 0, earThreshold: 0.15, exposureLeniency: 0.95)
    // All-black = underexposed → negative exposureScore
    XCTAssertLessThan(result.exposureScore, 0,
        "All-black image should have negative exposureScore")
}

// Helper — add to test file if not already present
private func makeSolidColorImage(size: Int, color: NSColor) -> CGImage {
    let bmi = CGBitmapInfo.byteOrder32Little.rawValue | CGImageAlphaInfo.noneSkipFirst.rawValue
    let ctx = CGContext(data: nil, width: size, height: size, bitsPerComponent: 8,
                        bytesPerRow: 4 * size, space: CGColorSpaceCreateDeviceRGB(),
                        bitmapInfo: bmi)!
    ctx.setFillColor(color.cgColor)
    ctx.fill(CGRect(x: 0, y: 0, width: size, height: size))
    return ctx.makeImage()!
}
```

- [ ] **Step 2: Run — verify fails**

```bash
xcodebuild test -project ImageRater.xcodeproj -scheme ImageRater \
  -only-testing ImageRaterTests/CullPipelineTests 2>&1 | grep -E "error:|failed"
```

Expected: compile error — `cull()` returns `CullResult` not `CullScores`

- [ ] **Step 3: Update existing tests that call `checkBlur` and `checkExposure` directly**

`checkBlur` and `checkExposure` return types are changing. Existing tests in `CullPipelineTests.swift` that call these functions and access `.rejected` / `.reason` directly will break. Update them to use `.result.rejected` / `.result.reason`:

```swift
// Before (will no longer compile):
let result = CullPipeline.checkBlur(image: img, threshold: 500)
XCTAssertTrue(result.rejected)

// After:
let (result, _) = CullPipeline.checkBlur(image: img, threshold: 500)
XCTAssertTrue(result.rejected)
```

Apply same pattern to any `checkExposure` call sites in the test file.

- [ ] **Step 3b: Add CullScores and update CullPipeline.swift**

At the top of `CullPipeline.swift`, add before the `CullPipeline` enum:

```swift
/// Combined output of the cull pipeline: accept/reject decision + numeric quality scores.
struct CullScores {
    let result: CullResult       // accept/reject with optional reason
    let blurScore: Float         // Sobel edge variance; higher = sharper; maps to ImageRecord.blurScore
    let exposureScore: Float     // EV bias float: 0.0=neutral, positive=over, negative=under;
                                 // maps to ImageRecord.exposureScore
}
```

Change `checkBlur` return type to `(result: CullResult, variance: Float)`:

```swift
static func checkBlur(image: CGImage, threshold: Float) -> (result: CullResult, variance: Float) {
    // ... existing downsampling + CIEdges + computeVariance logic unchanged ...
    // Replace the final return with:
    let variance: Float = /* existing computed variance variable */
    let result: CullResult = variance < threshold ? .reject(.blurry) : .keep
    return (result: result, variance: variance)
}
```

Change `checkExposure` return type to `(result: CullResult, exposureScore: Float)`:

```swift
static func checkExposure(image: CGImage, exposureLeniency: Float) -> (result: CullResult, exposureScore: Float) {
    // ... existing histogram logic unchanged ...
    // After computing topFraction and bottomFraction, add:
    let exposureScore: Float = topFraction - bottomFraction  // positive = over, negative = under

    let result: CullResult
    if topFraction > (1.0 - exposureLeniency) {
        result = .reject(.overexposed)
    } else if bottomFraction > (1.0 - exposureLeniency) {
        result = .reject(.underexposed)
    } else {
        result = .keep
    }
    return (result: result, exposureScore: exposureScore)
}
```

Replace the `cull()` function signature and body:

```swift
static func cull(image: CGImage, blurThreshold: Float, earThreshold: Float,
                 exposureLeniency: Float) async -> CullScores {
    let (blurResult, blurVariance) = checkBlur(image: image, threshold: blurThreshold)

    // Always compute exposure score for UI display, even if already rejected
    let (exposureResult, exposureScore) = checkExposure(image: image,
                                                         exposureLeniency: exposureLeniency)

    if blurResult.rejected {
        return CullScores(result: blurResult, blurScore: blurVariance, exposureScore: exposureScore)
    }

    let eyeResult = await checkEyesClosed(cgImage: image, earThreshold: earThreshold)
    if eyeResult.rejected {
        return CullScores(result: eyeResult, blurScore: blurVariance, exposureScore: exposureScore)
    }

    return CullScores(result: exposureResult, blurScore: blurVariance, exposureScore: exposureScore)
}
```

- [ ] **Step 4: Run tests — verify pass**

```bash
xcodebuild test -project ImageRater.xcodeproj -scheme ImageRater \
  -only-testing ImageRaterTests/CullPipelineTests 2>&1 | grep -E "passed|failed|error:"
```

Expected: `Test Suite 'CullPipelineTests' passed`

- [ ] **Step 5: Commit**

```bash
git add ImageRater/Pipeline/CullPipeline.swift ImageRaterTests/CullPipelineTests.swift
git commit -m "feat: CullPipeline returns CullScores with numeric blurScore and exposureScore"
```

---

## Task 3: RatingResult — New struct + RatingPipeline rewrite

**Files:**
- Modify: `ImageRater/Models/RatingResult.swift`
- Rewrite: `ImageRater/Pipeline/RatingPipeline.swift`
- Modify: `ImageRaterTests/RatingPipelineTests.swift`

- [ ] **Step 1: Add failing unit tests**

Add to `ImageRaterTests/RatingPipelineTests.swift`:

```swift
func testCombinedQualityWeighting() {
    let score = RatingPipeline.combinedQuality(
        technical: 0.8, aesthetic: 0.6, semantic: 0.5,
        weights: (technical: 0.4, aesthetic: 0.4, semantic: 0.2))
    XCTAssertEqual(score, 0.4*0.8 + 0.4*0.6 + 0.2*0.5, accuracy: 0.001)
}

func testClipIQAPlusScoreIsBetweenZeroAndOne() {
    // Any L2-normalised embedding must produce a score in [0,1]
    var emb = [Float](repeating: 0, count: 512)
    emb[0] = 1.0   // unit vector
    let score = RatingPipeline.clipIQAScore(embedding: emb)
    XCTAssertGreaterThanOrEqual(score, 0.0)
    XCTAssertLessThanOrEqual(score, 1.0)
}

func testPixelBufferCreation384x384() throws {
    let img = makeSolidColorCGImage(size: 512)
    let buffer = try RatingPipeline.cgImageToPixelBuffer(img, width: 384, height: 384)
    XCTAssertEqual(CVPixelBufferGetWidth(buffer), 384)
    XCTAssertEqual(CVPixelBufferGetHeight(buffer), 384)
}

// Helper
private func makeSolidColorCGImage(size: Int) -> CGImage {
    let bmi = CGBitmapInfo.byteOrder32Little.rawValue | CGImageAlphaInfo.noneSkipFirst.rawValue
    let ctx = CGContext(data: nil, width: size, height: size, bitsPerComponent: 8,
                        bytesPerRow: 4 * size, space: CGColorSpaceCreateDeviceRGB(),
                        bitmapInfo: bmi)!
    ctx.setFillColor(CGColor(gray: 0.5, alpha: 1))
    ctx.fill(CGRect(x: 0, y: 0, width: size, height: size))
    return ctx.makeImage()!
}
```

- [ ] **Step 2: Run — verify fails**

Expected: `RatingPipeline.combinedQuality` not found

- [ ] **Step 3: Remove `absoluteStars`-dependent tests from RatingPipelineTests.swift**

`absoluteStars(combined:)` is removed in the new pipeline. Delete the tests that call it (typically 6 tests checking star threshold boundaries). They will be replaced by `testCombinedQualityWeighting` and `testClipIQAPlusScoreIsBetweenZeroAndOne` added in Step 1.

Also add a temporary stub to `ProcessingQueue.swift` to keep the project compiling between this commit and Task 5. Replace the body of `process()` with:

```swift
func process(sessionID: NSManagedObjectID, onProgress: (@Sendable (Int, Int, String) -> Void)? = nil) async throws {
    fatalError("Replaced in Task 5 — do not call until ProcessingQueue is updated")
}
```

This ensures the project builds after this commit. Task 5 will replace the stub with the real implementation.

- [ ] **Step 3b: Replace RatingResult.swift**

```swift
// ImageRater/Models/RatingResult.swift
import Foundation

/// Scores from a successful rating inference run.
struct RatedScores: Equatable {
    let topiqTechnicalScore: Float      // TOPIQ-NR output [0,1]
    let topiqAestheticScore: Float      // TOPIQ-Swin output [0,1]
    let clipIQAScore: Float             // CLIP-IQA+ antonym softmax [0,1]
    let combinedQualityScore: Float     // weighted ensemble [0,1]
    let clipEmbedding: [Float]          // 512-dim L2-normalised; used for MMR diversity
}

/// Result of rating a single image.
enum RatingResult: Equatable {
    case unrated
    case rated(RatedScores)
}
```

- [ ] **Step 4: Rewrite RatingPipeline.swift**

```swift
// ImageRater/Pipeline/RatingPipeline.swift
import CoreML
import CoreImage
import CoreVideo
import Foundation
import OSLog

private let log = Logger(subsystem: "com.imagerating", category: "RatingPipeline")

enum RatingError: Error {
    case pixelBufferCreationFailed
    case modelNotFound(String)
}

enum RatingPipeline {

    // MARK: - Model loading (call once before processing loop)

    struct BundledModels {
        let technical: MLModel   // TOPIQ-NR
        let aesthetic: MLModel   // TOPIQ-Swin
        let clip: MLModel        // CLIP vision encoder
    }

    /// Load all three bundled models. Call once; pass result into rate() for every image.
    /// Throws if any .mlpackage/.mlmodelc is missing from the app bundle.
    static func loadBundledModels() throws -> BundledModels {
        let config = MLModelConfiguration()
        config.computeUnits = isAppleSilicon ? .all : .cpuOnly
        return BundledModels(
            technical: try loadBundledModel(named: "topiq-nr",    configuration: config),
            aesthetic: try loadBundledModel(named: "topiq-swin",  configuration: config),
            clip:      try loadBundledModel(named: "clip-vision", configuration: config)
        )
    }

    // MARK: - Inference

    /// Rate a single image with all three models concurrently. Never throws — returns .unrated on failure.
    static func rate(
        image: CGImage,
        models: BundledModels,
        weights: (technical: Float, aesthetic: Float, semantic: Float) = (0.4, 0.4, 0.2)
    ) async -> RatingResult {
        do {
            async let techScoreTask  = inferScore(image: image, model: models.technical, inputSize: 224)
            async let aesScoreTask   = inferScore(image: image, model: models.aesthetic, inputSize: 384)
            async let clipEmbTask    = inferEmbedding(image: image, model: models.clip,  inputSize: 224)

            let (tech, aes, emb) = try await (techScoreTask, aesScoreTask, clipEmbTask)
            let clip     = clipIQAScore(embedding: emb)
            let combined = combinedQuality(technical: tech, aesthetic: aes, semantic: clip, weights: weights)

            log.info("TOPIQ-NR \(tech, format: .fixed(precision: 3))  TOPIQ-Swin \(aes, format: .fixed(precision: 3))  CLIP-IQA+ \(clip, format: .fixed(precision: 3))  combined \(combined, format: .fixed(precision: 3))")

            return .rated(RatedScores(
                topiqTechnicalScore:  tech,
                topiqAestheticScore:  aes,
                clipIQAScore:         clip,
                combinedQualityScore: combined,
                clipEmbedding:        emb
            ))
        } catch {
            log.error("Rating failed: \(error)")
            return .unrated
        }
    }

    // MARK: - Helpers (internal, exposed for testing)

    static func combinedQuality(
        technical: Float, aesthetic: Float, semantic: Float,
        weights: (technical: Float, aesthetic: Float, semantic: Float)
    ) -> Float {
        weights.technical * technical + weights.aesthetic * aesthetic + weights.semantic * semantic
    }

    /// CLIP-IQA+: softmax([dot(img, good), dot(img, bad)])[0] = P("Good photo")
    static func clipIQAScore(embedding: [Float]) -> Float {
        let good = CLIPTextEmbeddings.goodPhoto
        let bad  = CLIPTextEmbeddings.badPhoto
        let dotGood = zip(embedding, good).reduce(0) { $0 + $1.0 * $1.1 }
        let dotBad  = zip(embedding, bad).reduce(0)  { $0 + $1.0 * $1.1 }
        let maxVal  = max(dotGood, dotBad)
        let eG = exp(dotGood - maxVal)
        let eB = exp(dotBad  - maxVal)
        return eG / (eG + eB)
    }

    // MARK: - Pixel buffer

    static func cgImageToPixelBuffer(_ image: CGImage, width: Int, height: Int) throws -> CVPixelBuffer {
        var buffer: CVPixelBuffer?
        let attrs: CFDictionary = [
            kCVPixelBufferCGImageCompatibilityKey:       true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true
        ] as CFDictionary
        guard CVPixelBufferCreate(kCFAllocatorDefault, width, height,
                                  kCVPixelFormatType_32BGRA, attrs, &buffer) == kCVReturnSuccess,
              let pb = buffer else { throw RatingError.pixelBufferCreationFailed }
        CVPixelBufferLockBaseAddress(pb, [])
        defer { CVPixelBufferUnlockBaseAddress(pb, []) }
        guard let ctx = CGContext(
            data: CVPixelBufferGetBaseAddress(pb),
            width: width, height: height,
            bitsPerComponent: 8,
            bytesPerRow: CVPixelBufferGetBytesPerRow(pb),
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGBitmapInfo.byteOrder32Little.rawValue | CGImageAlphaInfo.noneSkipFirst.rawValue
        ) else { throw RatingError.pixelBufferCreationFailed }
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        return pb
    }

    // MARK: - Private

    private static func inferScore(image: CGImage, model: MLModel, inputSize: Int) async throws -> Float {
        let buf   = try cgImageToPixelBuffer(image, width: inputSize, height: inputSize)
        let input = try MLDictionaryFeatureProvider(dictionary: ["image": buf])
        let out   = try await model.prediction(from: input)
        return extractScalar(from: out)
    }

    private static func inferEmbedding(image: CGImage, model: MLModel, inputSize: Int) async throws -> [Float] {
        let buf   = try cgImageToPixelBuffer(image, width: inputSize, height: inputSize)
        let input = try MLDictionaryFeatureProvider(dictionary: ["image": buf])
        let out   = try await model.prediction(from: input)
        return extractFloatArray(from: out)
    }

    private static func extractScalar(from output: MLFeatureProvider) -> Float {
        for name in output.featureNames {
            if let arr = output.featureValue(for: name)?.multiArrayValue { return Float(arr[0]) }
            if let d   = output.featureValue(for: name)?.doubleValue     { return Float(d) }
        }
        return 0.5
    }

    private static func extractFloatArray(from output: MLFeatureProvider) -> [Float] {
        for name in output.featureNames {
            if let arr = output.featureValue(for: name)?.multiArrayValue {
                return (0..<arr.count).map { Float(arr[$0]) }
            }
        }
        return Array(repeating: 0, count: 512)
    }

    private static func loadBundledModel(named name: String, configuration: MLModelConfiguration) throws -> MLModel {
        // Xcode compiles .mlpackage → .mlmodelc at build time
        if let url = Bundle.main.url(forResource: name, withExtension: "mlmodelc") {
            return try MLModel(contentsOf: url, configuration: configuration)
        }
        // Fallback: compile at runtime on first launch after an update
        guard let pkgURL = Bundle.main.url(forResource: name, withExtension: "mlpackage") else {
            throw RatingError.modelNotFound(name)
        }
        let compiled = try MLModel.compileModel(at: pkgURL)
        return try MLModel(contentsOf: compiled, configuration: configuration)
    }

    private static var isAppleSilicon: Bool {
        var info = utsname()
        uname(&info)
        return withUnsafePointer(to: &info.machine) {
            $0.withMemoryRebound(to: CChar.self, capacity: 1) { String(cString: $0) }
        }.hasPrefix("arm")
    }
}
```

- [ ] **Step 5: Run tests — verify pass**

```bash
xcodebuild test -project ImageRater.xcodeproj -scheme ImageRater \
  -only-testing ImageRaterTests/RatingPipelineTests 2>&1 | grep -E "passed|failed|error:"
```

Expected: unit tests pass; bundle-model integration tests skip (no models in test bundle yet — expected)

- [ ] **Step 6: Commit**

```bash
git add ImageRater/Models/RatingResult.swift ImageRater/Pipeline/RatingPipeline.swift ImageRaterTests/RatingPipelineTests.swift
git commit -m "feat: replace NIMA with TOPIQ-NR + TOPIQ-Swin + CLIP-IQA+ ensemble"
```

---

## Task 4: DiversityScorer — MMR + percentile normalization

**Files:**
- Create: `ImageRater/Pipeline/DiversityScorer.swift`
- Create: `ImageRaterTests/DiversityScorerTests.swift`

- [ ] **Step 1: Write the failing tests**

```swift
// ImageRaterTests/DiversityScorerTests.swift
import XCTest
import Accelerate
@testable import ImageRater

final class DiversityScorerTests: XCTestCase {

    // MARK: — Cosine similarity

    func testCosineSimilarityIdenticalVectors() {
        let v: [Float] = [1, 0, 0, 0]
        XCTAssertEqual(DiversityScorer.cosineSimilarity(v, v), 1.0, accuracy: 0.001)
    }

    func testCosineSimilarityOrthogonalVectors() {
        XCTAssertEqual(DiversityScorer.cosineSimilarity([1, 0, 0, 0], [0, 1, 0, 0]), 0.0, accuracy: 0.001)
    }

    func testCosineSimilarityOppositeVectors() {
        XCTAssertEqual(DiversityScorer.cosineSimilarity([1, 0, 0, 0], [-1, 0, 0, 0]), -1.0, accuracy: 0.001)
    }

    // MARK: — Threshold clustering

    func testClusteringGroupsIdenticalEmbeddings() {
        let e: [Float] = [1, 0, 0, 0]
        let ids = DiversityScorer.clusterByThreshold(embeddings: [e, e], threshold: 0.9)
        XCTAssertEqual(ids[0], ids[1], "Identical embeddings must be in same cluster")
    }

    func testClusteringKeepsOrthogonalEmbeddingsApart() {
        let ids = DiversityScorer.clusterByThreshold(
            embeddings: [[1, 0, 0, 0], [0, 1, 0, 0]], threshold: 0.9)
        XCTAssertNotEqual(ids[0], ids[1])
    }

    func testClusteringDoesNotAssignMinusOne() {
        // All images must end up in a valid cluster (>= 0)
        let embs: [[Float]] = (0..<10).map { i in
            var v = [Float](repeating: 0, count: 4)
            v[i % 4] = 1; return v
        }
        let ids = DiversityScorer.clusterByThreshold(embeddings: embs, threshold: 0.9)
        XCTAssertTrue(ids.allSatisfy { $0 >= 0 })
    }

    // MARK: — MMR ordering

    func testMMRSingleImageIsRank1WithFullFactor() {
        let items = DiversityScorer.mmrOrder(
            embeddings: [[1, 0, 0, 0]], qualityScores: [0.9], lambda: 0.6)
        XCTAssertEqual(items.count, 1)
        XCTAssertEqual(items[0].clusterRank, 1)
        XCTAssertEqual(items[0].diversityFactor, 1.0, accuracy: 0.001)
    }

    func testMMRPenalizesRank3WithFactor085() {
        // 5 near-identical vectors: rank 3 should get 0.85
        let same: [Float] = [1, 0, 0, 0]
        let embs  = Array(repeating: same, count: 5)
        let scores = (0..<5).map { Float(5 - $0) / 5.0 }   // descending quality
        let items = DiversityScorer.mmrOrder(embeddings: embs, qualityScores: scores, lambda: 0.6)
        let rank3 = items.first { $0.clusterRank == 3 }
        XCTAssertNotNil(rank3)
        XCTAssertEqual(rank3!.diversityFactor, 0.85, accuracy: 0.001)
    }

    func testMMRPenalizesRank5PlusWithFactor055() {
        let same: [Float] = [1, 0, 0, 0]
        let embs  = Array(repeating: same, count: 6)
        let scores = (0..<6).map { Float(6 - $0) / 6.0 }
        let items = DiversityScorer.mmrOrder(embeddings: embs, qualityScores: scores, lambda: 0.6)
        let rank6 = items.first { $0.clusterRank == 6 }
        XCTAssertNotNil(rank6)
        XCTAssertEqual(rank6!.diversityFactor, 0.55, accuracy: 0.001)
    }

    func testMMRPrefersDiverseOverNearDuplicate() {
        // diverse (c) should rank above near-dup (b) despite lower raw quality
        let a: [Float] = [1, 0, 0, 0]
        let b: [Float] = [1, 0, 0, 0]   // near-dup of a
        let c: [Float] = [0, 1, 0, 0]   // diverse

        let items = DiversityScorer.mmrOrder(
            embeddings: [a, b, c],
            qualityScores: [0.9, 0.88, 0.7],
            lambda: 0.6)

        let rankC = items.first { $0.originalIndex == 2 }!.clusterRank
        let rankB = items.first { $0.originalIndex == 1 }!.clusterRank
        XCTAssertLessThan(rankC, rankB, "Diverse image should rank above near-duplicate")
    }

    // MARK: — Percentile normalization

    func testPercentileTop5PercentGets5Stars() {
        let scores = (1...100).map { Float($0) / 100.0 }  // 0.01...1.00
        let stars = DiversityScorer.percentileToStars(finalScores: scores)
        // Index 99 = highest score → rank 0 → percentile 0.0 → 5★
        XCTAssertEqual(stars[99], 5)
    }

    func testPercentileBottom20PercentGets1Star() {
        let scores = (1...100).map { Float($0) / 100.0 }
        let stars = DiversityScorer.percentileToStars(finalScores: scores)
        // Index 0 = lowest score → rank 99 → percentile 0.99 → 1★
        XCTAssertEqual(stars[0], 1)
    }

    func testPercentileSingleImageGets5Stars() {
        let stars = DiversityScorer.percentileToStars(finalScores: [0.7])
        XCTAssertEqual(stars[0], 5)
    }

    func testPercentileAllValuesAre1Through5() {
        let scores = (0..<20).map { Float($0) / 19.0 }
        let stars = DiversityScorer.percentileToStars(finalScores: scores)
        XCTAssertTrue(stars.allSatisfy { $0 >= 1 && $0 <= 5 })
    }
}
```

- [ ] **Step 2: Run — verify compile error**

```bash
xcodebuild test -project ImageRater.xcodeproj -scheme ImageRater \
  -only-testing ImageRaterTests/DiversityScorerTests 2>&1 | grep "error:"
```

Expected: `cannot find type 'DiversityScorer' in scope`

- [ ] **Step 3: Implement DiversityScorer.swift**

```swift
// ImageRater/Pipeline/DiversityScorer.swift
import Foundation
import Accelerate

/// Pure-algorithm diversity scoring. No CoreML, CoreData, or side effects.
/// All functions are static and safe to call from any context.
enum DiversityScorer {

    // MARK: - Public output type

    struct MMRItem {
        let originalIndex: Int
        let clusterRank: Int       // 1-based; 1 = selected first (best quality + diversity)
        let diversityFactor: Float // multiplied against combinedQualityScore
    }

    // MARK: - Cosine similarity

    /// Dot product of two equal-length Float vectors (L2-normalised embeddings ≡ cosine similarity).
    static func cosineSimilarity(_ a: [Float], _ b: [Float]) -> Float {
        var result: Float = 0
        vDSP_dotpr(a, 1, b, 1, &result, vDSP_Length(a.count))
        return result
    }

    // MARK: - Threshold clustering

    /// Greedy single-link threshold clustering.
    /// Each image joins the first existing cluster whose seed has sim ≥ threshold,
    /// or seeds a new cluster. O(n²) worst-case; fast in practice for n ≤ 2000.
    /// Returns Int32 cluster ID (≥ 0) for each input index.
    static func clusterByThreshold(embeddings: [[Float]], threshold: Float) -> [Int32] {
        var ids = [Int32](repeating: -1, count: embeddings.count)
        var seeds: [[Float]] = []

        for (i, emb) in embeddings.enumerated() {
            var assigned = false
            for (ci, seed) in seeds.enumerated() {
                if cosineSimilarity(emb, seed) >= threshold {
                    ids[i] = Int32(ci)
                    assigned = true
                    break
                }
            }
            if !assigned {
                ids[i] = Int32(seeds.count)
                seeds.append(emb)
            }
        }
        return ids
    }

    // MARK: - MMR ordering

    /// Maximal Marginal Relevance: greedily selects the next image that maximises
    /// λ × quality - (1-λ) × max_similarity_to_already_selected.
    ///
    /// λ = 0.6: quality counts 60%, diversity 40%.
    /// Returns MMRItems in selection order. clusterRank = position in that order (1-based).
    static func mmrOrder(
        embeddings: [[Float]],
        qualityScores: [Float],
        lambda: Float = 0.6
    ) -> [MMRItem] {
        let n = embeddings.count
        guard n > 0 else { return [] }

        var selected: [Int] = []
        var remaining = IndexSet(0..<n)
        var result = [MMRItem]()
        result.reserveCapacity(n)

        while !remaining.isEmpty {
            var bestIdx = remaining.first!
            var bestScore = -Float.infinity

            for idx in remaining {
                let quality = lambda * qualityScores[idx]
                let penalty: Float
                if selected.isEmpty {
                    penalty = 0
                } else {
                    penalty = (1 - lambda) * selected
                        .map { cosineSimilarity(embeddings[idx], embeddings[$0]) }
                        .max()!
                }
                let s = quality - penalty
                if s > bestScore { bestScore = s; bestIdx = idx }
            }

            selected.append(bestIdx)
            remaining.remove(bestIdx)
            let rank = selected.count
            result.append(MMRItem(
                originalIndex:  bestIdx,
                clusterRank:    rank,
                diversityFactor: diversityFactor(rank: rank)
            ))
        }
        return result
    }

    // MARK: - Percentile normalisation

    /// Map finalScores array to 1–5 star ratings using percentile thresholds:
    ///   top 5%     → 5★
    ///   5–20%      → 4★
    ///   20–50%     → 3★
    ///   50–80%     → 2★
    ///   bottom 20% → 1★
    ///
    /// Returns Int16 array aligned with input indices.
    static func percentileToStars(finalScores: [Float]) -> [Int16] {
        let n = finalScores.count
        guard n > 0 else { return [] }

        let sortedDesc = finalScores.indices.sorted { finalScores[$0] > finalScores[$1] }
        var stars = [Int16](repeating: 1, count: n)
        for (rank, originalIdx) in sortedDesc.enumerated() {
            let p = Float(rank) / Float(n)   // 0.0 = best
            stars[originalIdx] = switch p {
            case ..<0.05:      5
            case 0.05..<0.20:  4
            case 0.20..<0.50:  3
            case 0.50..<0.80:  2
            default:           1
            }
        }
        return stars
    }

    // MARK: - Private

    private static func diversityFactor(rank: Int) -> Float {
        switch rank {
        case 1, 2: return 1.0
        case 3:    return 0.85
        case 4:    return 0.70
        default:   return 0.55
        }
    }
}
```

- [ ] **Step 4: Run tests — verify pass**

```bash
xcodebuild test -project ImageRater.xcodeproj -scheme ImageRater \
  -only-testing ImageRaterTests/DiversityScorerTests 2>&1 | grep -E "passed|failed|error:"
```

Expected: `Test Suite 'DiversityScorerTests' passed`

- [ ] **Step 5: Commit**

```bash
git add ImageRater/Pipeline/DiversityScorer.swift ImageRaterTests/DiversityScorerTests.swift
git commit -m "feat: add DiversityScorer with MMR ordering and percentile normalization"
```

---

## Task 5: ProcessingQueue — Two-pass architecture

**Files:**
- Modify: `ImageRater/Pipeline/ProcessingQueue.swift`
- Modify: `ImageRaterTests/ProcessingQueueTests.swift`

- [ ] **Step 1: Add `rated` to ProcessState**

In `ProcessingQueue.swift`, add to the `ProcessState` enum:

```swift
static let rated = "rated"   // pass 1 complete; diversity scoring pending
```

Update `markInterrupted` to include `rated` (and keep `culling` for backward compat with stores that may have that state from a prior interrupted run):

```swift
where record.processState == ProcessState.culling
   || record.processState == ProcessState.rating
   || record.processState == ProcessState.rated {
```

Note: the new `process()` body in Step 2 never writes `processState = "culling"` — cull and rate are concurrent in pass 1. The `culling` branch in `markInterrupted` is dead code for new sessions but harmless to leave for any existing interrupted sessions.

- [ ] **Step 2: Replace process() with two-pass implementation**

Remove the `ModelStore.shared.prepareModels()`, `ModelStore.shared.model(named: "nima-aesthetic")`, and `ModelStore.shared.model(named: "nima-technical")` calls — these are replaced by `RatingPipeline.loadBundledModels()` at the top of the new function. Replace the entire `process()` function body:

```swift
func process(
    sessionID: NSManagedObjectID,
    onProgress: (@Sendable (Int, Int, String) -> Void)? = nil
) async throws {

    // Load bundled models once before the per-image loop
    onProgress?(0, 0, "Compiling models…")
    let models = try RatingPipeline.loadBundledModels()

    let (imageIDs, configSnapshot) = try await context.perform { [self] in
        guard let session = try? self.context.existingObject(with: sessionID) as? Session,
              let images = session.images?.allObjects as? [ImageRecord] else {
            throw CocoaError(.coreData)
        }
        let snapshot = self.fetchOrCreateConfigSync()
        let sorted = images.sorted { ($0.filePath ?? "") < ($1.filePath ?? "") }
        return (sorted.map(\.objectID), snapshot)
    }

    let total = imageIDs.count
    var decodeErrorCount = 0
    log.info("Session processing started: \(total) images")

    // ─── PASS 1: per-image inference ─────────────────────────────────────

    do {
        for (i, imageID) in imageIDs.enumerated() {
            try Task.checkCancellation()

            let (filePath, skip): (String?, Bool) = await context.perform { [self] in
                guard let r = try? self.context.existingObject(with: imageID) as? ImageRecord else {
                    return (nil, true)
                }
                let skip = r.processState == ProcessState.done
                        || r.processState == ProcessState.rated
                return (r.filePath, skip)
            }
            if skip { continue }

            onProgress?(i, total, "Scoring \(i + 1) of \(total)")

            guard let path = filePath,
                  let cgImage = LibRawWrapper.decode(url: URL(filePath: path)) else {
                decodeErrorCount += 1
                await context.perform { [self] in
                    guard let r = try? self.context.existingObject(with: imageID) as? ImageRecord else { return }
                    r.decodeError = true
                    r.processState = ProcessState.done
                    try? self.context.save()
                }
                continue
            }

            // Check user override — if set, skip AI rating but still write sidecar later
            let hasOverride = await context.perform { [self] in
                (try? self.context.existingObject(with: imageID) as? ImageRecord)?.userOverride != nil
            }

            async let cullTask   = CullPipeline.cull(
                image: cgImage,
                blurThreshold:    configSnapshot.blurThreshold,
                earThreshold:     configSnapshot.earThreshold,
                exposureLeniency: configSnapshot.exposureLeniency)
            let ratingResult = hasOverride ? RatingResult.unrated
                             : await RatingPipeline.rate(image: cgImage, models: models)
            let cullScores = await cullTask

            await context.perform { [self] in
                guard let r = try? self.context.existingObject(with: imageID) as? ImageRecord else { return }
                // Cull fields
                r.cullRejected  = cullScores.result.rejected
                r.cullReason    = cullScores.result.reason?.rawValue
                r.blurScore     = cullScores.blurScore
                r.exposureScore = cullScores.exposureScore

                // Rating fields (only if no user override)
                if !hasOverride, case .rated(let scores) = ratingResult {
                    r.topiqTechnicalScore  = scores.topiqTechnicalScore
                    r.topiqAestheticScore  = scores.topiqAestheticScore
                    r.clipIQAScore         = scores.clipIQAScore
                    r.combinedQualityScore = scores.combinedQualityScore
                    // Store 512-dim embedding as raw bytes
                    r.clipEmbedding = scores.clipEmbedding.withUnsafeBufferPointer {
                        Data(buffer: $0)
                    }
                }
                r.processState = ProcessState.rated
                try? self.context.save()
            }
        }
    } catch is CancellationError {
        await markInterrupted(sessionID: sessionID)
        throw CancellationError()
    }

    try Task.checkCancellation()

    // ─── PASS 2: session-level diversity + normalization ──────────────────

    onProgress?(total, total, "Ranking variety…")
    try await runDiversityPass(sessionID: sessionID)

    // Write XMP sidecars now that final star ratings are assigned
    await writeSidecars(imageIDs: imageIDs, sessionID: sessionID)

    onProgress?(total, total, "Done")
    log.info("Session complete — \(total) images, \(decodeErrorCount) decode errors")
}
```

- [ ] **Step 3: Add runDiversityPass() and writeSidecars()**

Add as private methods inside the `ProcessingQueue` actor:

```swift
private func runDiversityPass(sessionID: NSManagedObjectID) async throws {
    // Load embeddings and quality scores for all rated (or done) images
    let (imageIDs, embeddings, qualityScores): ([NSManagedObjectID], [[Float]], [Float]) =
        try await context.perform { [self] in
            guard let session = try? self.context.existingObject(with: sessionID) as? Session,
                  let images = session.images?.allObjects as? [ImageRecord] else {
                throw CocoaError(.coreData)
            }
            let sorted = images
                .filter { $0.processState == ProcessState.rated || $0.processState == ProcessState.done }
                .sorted { ($0.filePath ?? "") < ($1.filePath ?? "") }

            let ids = sorted.map { $0.objectID }
            let embs: [[Float]] = sorted.map { r in
                guard let data = r.clipEmbedding else { return [Float](repeating: 0, count: 512) }
                let count = data.count / MemoryLayout<Float>.size
                return data.withUnsafeBytes { Array($0.bindMemory(to: Float.self).prefix(count)) }
            }
            let scores = sorted.map { $0.combinedQualityScore }
            return (ids, embs, scores)
        }

    guard !imageIDs.isEmpty else { return }

    // Step A: threshold clustering → clusterID
    let clusterIDs = DiversityScorer.clusterByThreshold(embeddings: embeddings, threshold: 0.92)

    // Step B: MMR ordering → clusterRank + diversityFactor
    let mmrItems = DiversityScorer.mmrOrder(embeddings: embeddings, qualityScores: qualityScores, lambda: 0.6)
    let mmrByIdx = Dictionary(uniqueKeysWithValues: mmrItems.map { ($0.originalIndex, $0) })

    // Compute finalScore per image (aligned to imageIDs order)
    var finalScores = [Float](repeating: 0, count: imageIDs.count)
    for item in mmrItems {
        finalScores[item.originalIndex] = qualityScores[item.originalIndex] * item.diversityFactor
    }

    // Percentile normalisation → star ratings
    let starRatings = DiversityScorer.percentileToStars(finalScores: finalScores)

    // Write all diversity fields
    try await context.perform { [self] in
        for (i, imageID) in imageIDs.enumerated() {
            guard let r = try? self.context.existingObject(with: imageID) as? ImageRecord else { continue }
            let mmr = mmrByIdx[i]
            r.clusterID      = clusterIDs[i]
            r.clusterRank    = Int32(mmr?.clusterRank ?? 1)
            r.diversityFactor = mmr?.diversityFactor ?? 1.0
            r.finalScore     = finalScores[i]
            r.ratingStars    = NSNumber(value: starRatings[i])
            r.processState   = ProcessState.done
        }
        try self.context.save()
    }
}

private func writeSidecars(imageIDs: [NSManagedObjectID], sessionID: NSManagedObjectID) async {
    await context.perform { [self] in
        for imageID in imageIDs {
            guard let r = try? self.context.existingObject(with: imageID) as? ImageRecord,
                  let path = r.filePath else { continue }
            let stars: Int
            if let override = r.userOverride, override.int16Value > 0 {
                stars = Int(override.int16Value)
            } else if let s = r.ratingStars {
                stars = Int(s.int16Value)
            } else {
                continue
            }
            try? MetadataWriter.writeSidecar(stars: stars, for: URL(filePath: path))
        }
    }
}
```

- [ ] **Step 4: Run ProcessingQueueTests**

```bash
xcodebuild test -project ImageRater.xcodeproj -scheme ImageRater \
  -only-testing ImageRaterTests/ProcessingQueueTests 2>&1 | grep -E "passed|failed|error:"
```

Expected: all ProcessingQueueTests pass (existing cancellation + config tests still work)

- [ ] **Step 5: Commit**

```bash
git add ImageRater/Pipeline/ProcessingQueue.swift ImageRaterTests/ProcessingQueueTests.swift
git commit -m "feat: two-pass pipeline with diversity scoring and percentile star normalization"
```

---

## Task 6: project.yml — Bundle models in app binary

**Files:**
- Modify: `project.yml`

- [ ] **Step 1: Add models to resources**

Confirm `project.yml` is at the repo root (same level as the `models/` directory created by Task 0). In `project.yml`, find the `ImageRater` target resources section and add the three `.mlpackage` entries:

```yaml
# Before (existing):
    resources:
      - ImageRater/CoreData/ImageRater.xcdatamodeld

# After:
    resources:
      - ImageRater/CoreData/ImageRater.xcdatamodeld
      - models/topiq-nr.mlpackage
      - models/topiq-swin.mlpackage
      - models/clip-vision.mlpackage
```

- [ ] **Step 2: Regenerate Xcode project**

```bash
cd /Users/dknathalage/repos/image-rating && xcodegen generate
```

Expected: `ImageRater.xcodeproj` regenerated with new resources

- [ ] **Step 3: Verify build succeeds**

```bash
xcodebuild build -project ImageRater.xcodeproj -scheme ImageRater \
  -destination 'platform=macOS' 2>&1 | tail -5
```

Expected: `** BUILD SUCCEEDED **`

- [ ] **Step 4: Commit**

```bash
git add project.yml ImageRater.xcodeproj
git commit -m "build: bundle topiq-nr, topiq-swin, clip-vision models in app binary"
```

---

## Task 7: DetailView — Scores panel + characteristics panel

**Files:**
- Modify: `ImageRater/UI/DetailView.swift`

No unit tests — verify by running app and inspecting sidebar.

- [ ] **Step 1: Add ScoreBarView helper**

Add this view at the bottom of `DetailView.swift` (before the closing brace of the file):

```swift
/// Progress bar showing a [0,1] raw score as a labelled [0,10] bar.
private struct ScoreBarView: View {
    let label: String
    let rawScore: Float   // [0, 1] as stored in CoreData

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack {
                Text(label).font(.caption2).foregroundStyle(.secondary)
                Spacer()
                Text(String(format: "%.1f", rawScore * 10))
                    .font(.caption2.monospacedDigit())
            }
            GeometryReader { geo in
                ZStack(alignment: .leading) {
                    RoundedRectangle(cornerRadius: 2).fill(Color.secondary.opacity(0.2))
                    RoundedRectangle(cornerRadius: 2).fill(Color.accentColor)
                        .frame(width: geo.size.width * CGFloat(min(rawScore, 1)))
                }
            }
            .frame(height: 4)
        }
    }
}
```

- [ ] **Step 2: Replace the existing AI Scores block in metadataPane**

`DetailView.swift` uses `metaSectionDivider` + `metaRow` helpers in a flat `VStack`, not SwiftUI `Section`. Find the block at lines 135–139:

```swift
if (record.ratingStars?.int16Value ?? 0) > 0 {
    metaSectionDivider("AI Scores")
    metaRow("Technical", String(format: "%.2f", record.clipScore?.floatValue ?? 0))
    metaRow("Aesthetic", String(format: "%.2f", record.aestheticScore?.floatValue ?? 0))
}
```

Replace it with:

```swift
// AI SCORES — shown if ensemble has run
if record.combinedQualityScore > 0 {
    metaSectionDivider("AI Scores")
    ScoreBarView(label: "Technical", rawScore: record.topiqTechnicalScore)
    ScoreBarView(label: "Aesthetic", rawScore: record.topiqAestheticScore)
    ScoreBarView(label: "Semantic",  rawScore: record.clipIQAScore)
    Divider().padding(.vertical, 2)
    metaRow("Combined", String(format: "%.1f", record.combinedQualityScore * 10))
    if let s = record.ratingStars, s.int16Value > 0 {
        metaRow("AI stars", String(repeating: "★", count: Int(s.int16Value)))
    }
    if let o = record.userOverride, o.int16Value > 0 {
        metaRow("Manual", String(repeating: "★", count: Int(o.int16Value)))
    }
}
```

- [ ] **Step 3: Add CHARACTERISTICS block after AI Scores (using metaSectionDivider / metaRow pattern)**

Insert immediately after the AI Scores block, before `metaSectionDivider("Rating")`:

```swift
// CHARACTERISTICS — shown once cull + diversity pass have run
if record.blurScore > 0 || record.clusterRank > 0 {
    metaSectionDivider("Characteristics")
    if record.blurScore > 0 {
        let blurLabel = record.blurScore > 300 ? "Sharp"
                      : record.blurScore > 100 ? "Soft" : "Blurry"
        metaRow("Blur", blurLabel)
    }
    if record.exposureScore != 0 {
        metaRow("Exposure", exposureLabel(record.exposureScore))
    }
    if record.clusterRank > 0, let ctx = record.managedObjectContext {
        let size = clusterSize(id: record.clusterID, in: ctx)
        metaRow("Cluster", "#\(record.clusterID) · rank \(record.clusterRank) of \(size)")
        metaRow("Diversity", String(format: "%.2f×", record.diversityFactor))
    }
}
```

- [ ] **Step 4: Add helper functions to DetailView**

Inside `DetailView` body (or as private funcs):

```swift
private func exposureLabel(_ score: Float) -> String {
    switch score {
    case let s where s >  1.5: return String(format: "+%.1f EV (over)", s)
    case let s where s < -1.5: return String(format: "%.1f EV (under)", s)
    case let s where s >  0.3: return String(format: "+%.1f EV", s)
    case let s where s < -0.3: return String(format: "%.1f EV", s)
    default: return "Normal"
    }
}

private func clusterSize(id: Int32, in ctx: NSManagedObjectContext) -> Int {
    let req = ImageRecord.fetchRequest()
    req.predicate = NSPredicate(format: "clusterID == %d", id)
    return (try? ctx.count(for: req)) ?? 0
}
```

- [ ] **Step 5: Build and verify**

```bash
xcodebuild build -project ImageRater.xcodeproj -scheme ImageRater \
  -destination 'platform=macOS' 2>&1 | grep "error:"
```

Run app → process a session → click an image → verify sidebar shows score bars and characteristics.

- [ ] **Step 6: Commit**

```bash
git add ImageRater/UI/DetailView.swift
git commit -m "feat: add AI score bars and characteristics panel to detail sidebar"
```

---

## Task 8: ThumbnailCell — Cluster badge + dim overlay

**Files:**
- Modify: `ImageRater/UI/Components/ThumbnailCell.swift`

- [ ] **Step 1: Add cluster representative badge**

`ScoreBadge` in `ThumbnailCell.swift` is a direct child of the root `ZStack` at line 76 (not inside an overlay). Add the "C" badge as a new `.overlay(alignment: .bottomLeading)` on the `ZStack`:

```swift
// "C" badge — best image in its similarity cluster (clusterRank == 1)
.overlay(alignment: .bottomLeading) {
    if record.clusterRank == 1 && record.clusterID >= 0 {
        Text("C")
            .font(.system(size: 8, weight: .bold))
            .foregroundStyle(.white)
            .padding(.horizontal, 3).padding(.vertical, 1)
            .background(Color.purple.opacity(0.8))
            .clipShape(RoundedRectangle(cornerRadius: 2))
            .padding(4)
    }
}
```

- [ ] **Step 2: Add dim overlay for near-duplicates**

Add after the closing brace of the `isProcessing` `.overlay { ... }` block (after line 74):

```swift
// Dim near-duplicate images (diversityFactor computed and < 0.60)
.overlay {
    if record.diversityFactor > 0 && record.diversityFactor < 0.60 {
        Color.black.opacity(0.30)
            .cornerRadius(6)
            .allowsHitTesting(false)
    }
}
```

- [ ] **Step 3: Build and verify**

```bash
xcodebuild build -project ImageRater.xcodeproj -scheme ImageRater \
  -destination 'platform=macOS' 2>&1 | grep "error:"
```

Run app → process a burst folder → verify:
- Cluster representative shows purple "C" badge at bottom-left
- Rank 5+ near-duplicates (diversityFactor < 0.6) show 30% dark overlay

- [ ] **Step 4: Commit**

```bash
git add ImageRater/UI/Components/ThumbnailCell.swift
git commit -m "feat: add cluster-rep badge and near-duplicate dim overlay to grid thumbnails"
```

---

## Task 9: Filter sidebar — Score sliders + diversity toggles

**Files:**
- Modify: `ImageRater/UI/Components/RatingFilterView.swift`
- Modify: `ImageRater/App/ContentView.swift`

- [ ] **Step 1: Add filter state to ContentView**

Add alongside the existing `@State private var ratingFilter: Set<Int> = []`:

```swift
@State private var minTechnicalScore: Double = 0.0   // [0, 10] display scale
@State private var minAestheticScore: Double = 0.0
@State private var hideBlurry: Bool = false
@State private var hideExposure: Bool = false
@State private var clusterRepsOnly: Bool = false
@State private var varietySetOnly: Bool = false
```

- [ ] **Step 2: Persist filter state in ContentView**

Add `.onAppear` and `.onChange` modifiers to the root view in ContentView:

```swift
.onAppear {
    let ud = UserDefaults.standard
    minTechnicalScore = ud.double(forKey: "filterMinTech")
    minAestheticScore = ud.double(forKey: "filterMinAes")
    hideBlurry        = ud.bool(forKey: "filterHideBlurry")
    hideExposure      = ud.bool(forKey: "filterHideExposure")
    clusterRepsOnly   = ud.bool(forKey: "filterClusterReps")
    varietySetOnly    = ud.bool(forKey: "filterVarietySet")
}
.onChange(of: minTechnicalScore) { UserDefaults.standard.set($1, forKey: "filterMinTech") }
.onChange(of: minAestheticScore) { UserDefaults.standard.set($1, forKey: "filterMinAes") }
.onChange(of: hideBlurry)        { UserDefaults.standard.set($1, forKey: "filterHideBlurry") }
.onChange(of: hideExposure)      { UserDefaults.standard.set($1, forKey: "filterHideExposure") }
.onChange(of: clusterRepsOnly)   { UserDefaults.standard.set($1, forKey: "filterClusterReps") }
.onChange(of: varietySetOnly)    { UserDefaults.standard.set($1, forKey: "filterVarietySet") }
```

- [ ] **Step 3: Update filteredImages in ContentView**

Replace the existing `filteredImages` computed property:

```swift
private var filteredImages: [ImageRecord] {
    var images = Array(sessionImages)

    if !ratingFilter.isEmpty {
        images = images.filter { ratingFilter.contains(effectiveRating($0)) }
    }
    if minTechnicalScore > 0 {
        images = images.filter { Double($0.topiqTechnicalScore * 10) >= minTechnicalScore }
    }
    if minAestheticScore > 0 {
        images = images.filter { Double($0.topiqAestheticScore * 10) >= minAestheticScore }
    }
    if hideBlurry {
        images = images.filter { $0.cullReason != CullReason.blurry.rawValue }
    }
    if hideExposure {
        images = images.filter { abs($0.exposureScore) <= 1.5 }
    }
    if clusterRepsOnly {
        images = images.filter { $0.clusterRank == 1 }
    }
    if varietySetOnly {
        var bestPerCluster: [Int32: ImageRecord] = [:]
        for img in images {
            let cid = img.clusterID
            guard cid >= 0 else { continue }
            if let existing = bestPerCluster[cid] {
                let existingStars = existing.ratingStars?.int16Value ?? 0
                let imgStars = img.ratingStars?.int16Value ?? 0
                if imgStars > existingStars { bestPerCluster[cid] = img }
            } else {
                bestPerCluster[cid] = img
            }
        }
        images = Array(bestPerCluster.values)
    }
    return images
}
```

- [ ] **Step 4: Pass new bindings to RatingFilterView**

Find where `RatingFilterView` is instantiated and update:

```swift
RatingFilterView(
    images: Array(sessionImages),
    ratingFilter: $ratingFilter,
    minTechnicalScore: $minTechnicalScore,
    minAestheticScore: $minAestheticScore,
    hideBlurry: $hideBlurry,
    hideExposure: $hideExposure,
    clusterRepsOnly: $clusterRepsOnly,
    varietySetOnly: $varietySetOnly
)
```

- [ ] **Step 5: Update RatingFilterView**

Replace `RatingFilterView` entirely:

```swift
// ImageRater/UI/Components/RatingFilterView.swift
import SwiftUI
import CoreData

struct RatingFilterView: View {
    let images: [ImageRecord]
    @Binding var ratingFilter: Set<Int>
    @Binding var minTechnicalScore: Double
    @Binding var minAestheticScore: Double
    @Binding var hideBlurry: Bool
    @Binding var hideExposure: Bool
    @Binding var clusterRepsOnly: Bool
    @Binding var varietySetOnly: Bool

    private var ratingCounts: [Int: Int] {
        var counts = [Int: Int]()
        for img in images {
            let r = effectiveRating(img)
            counts[r, default: 0] += 1
        }
        return counts
    }

    private func effectiveRating(_ r: ImageRecord) -> Int {
        if let o = r.userOverride, o.int16Value > 0 { return Int(o.int16Value) }
        if let s = r.ratingStars { return Int(s.int16Value) }
        return 0
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {

            // ── Star filter (existing behaviour) ──────────────────────────
            VStack(alignment: .leading, spacing: 2) {
                ForEach([5, 4, 3, 2, 1, 0], id: \.self) { rating in
                    let count    = ratingCounts[rating] ?? 0
                    let selected = ratingFilter.contains(rating)
                    Button {
                        if selected { ratingFilter.remove(rating) }
                        else        { ratingFilter.insert(rating) }
                    } label: {
                        HStack {
                            Text(rating == 0 ? "Unrated" : String(repeating: "★", count: rating))
                                .foregroundStyle(selected ? Color.accentColor
                                               : (count == 0 ? Color.secondary : Color.primary))
                            Spacer()
                            Text("\(count)").foregroundStyle(.secondary)
                        }
                        .contentShape(Rectangle())
                    }
                    .buttonStyle(.plain)
                }
            }

            Divider()

            // ── Score sliders ─────────────────────────────────────────────
            VStack(alignment: .leading, spacing: 4) {
                Text("SCORES").font(.caption2).foregroundStyle(.secondary)
                Text("Technical ≥ \(Int(minTechnicalScore))").font(.caption2)
                Slider(value: $minTechnicalScore, in: 0...10, step: 1)
                Text("Aesthetic ≥ \(Int(minAestheticScore))").font(.caption2)
                Slider(value: $minAestheticScore, in: 0...10, step: 1)
            }

            Divider()

            // ── Quality toggles ───────────────────────────────────────────
            VStack(alignment: .leading, spacing: 4) {
                Text("QUALITY").font(.caption2).foregroundStyle(.secondary)
                Toggle("Hide blurry",        isOn: $hideBlurry).font(.caption)
                Toggle("Hide bad exposure",  isOn: $hideExposure).font(.caption)
            }

            Divider()

            // ── Diversity toggles ─────────────────────────────────────────
            VStack(alignment: .leading, spacing: 4) {
                Text("DIVERSITY").font(.caption2).foregroundStyle(.secondary)
                Toggle("Cluster reps only", isOn: $clusterRepsOnly)
                    .font(.caption)
                    .onChange(of: clusterRepsOnly) { _, v in if v { varietySetOnly = false } }
                Toggle("Variety set", isOn: $varietySetOnly)
                    .font(.caption)
                    .onChange(of: varietySetOnly) { _, v in if v { clusterRepsOnly = false } }
            }
        }
    }
}
```

- [ ] **Step 6: Build and verify**

```bash
xcodebuild build -project ImageRater.xcodeproj -scheme ImageRater \
  -destination 'platform=macOS' 2>&1 | grep "error:"
```

Run app. Verify:
- Score sliders appear below star toggles
- Moving Technical slider to 7 hides low-scoring images
- "Cluster reps only" shows one per burst group
- "Variety set" shows best per cluster
- Settings survive app restart

- [ ] **Step 7: Commit**

```bash
git add ImageRater/UI/Components/RatingFilterView.swift ImageRater/App/ContentView.swift
git commit -m "feat: add score sliders, exposure/blur toggles, diversity filters to sidebar"
```

---

## Task 10: Full test suite + integration smoke test

- [ ] **Step 1: Run all tests**

```bash
xcodebuild test -project ImageRater.xcodeproj -scheme ImageRater 2>&1 \
  | grep -E "Test Suite|error:|failed"
```

Expected: all suites pass; no compilation errors

- [ ] **Step 2: Manual integration test**

1. Launch app
2. Import folder with 20+ images (mix of portraits, landscapes, burst shots)
3. Click "Process" — progress should show "Scoring X of N" then "Ranking variety…"
4. Open detail view of a rated image — verify score bars show [0, 10] values for Technical / Aesthetic / Semantic
5. Check a burst sequence — first two images should show full brightness; rank 5+ should be 30% dimmed
6. Click the "C" badge images — confirm they are the sharpest / highest quality in each burst group
7. Toggle "Variety set" — grid should collapse to one best image per burst group
8. Move Technical slider to 7.0 — low-tech images disappear
9. Close and reopen app — filter settings should persist

- [ ] **Step 3: Final commit**

All files were committed task-by-task. If anything remains unstaged:

```bash
git status   # verify only expected files are modified
git add ImageRater/Pipeline/ ImageRater/UI/ ImageRater/App/ ImageRaterTests/
git commit -m "feat: image rating accuracy overhaul complete — TOPIQ ensemble + MMR diversity"
```
