# NIMA Ensemble Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the CLIP + LAION aesthetic predictor with a two-model NIMA ensemble (aesthetic + technical quality) that produces calibrated absolute scores, eliminating per-session z-score recalculation.

**Architecture:** Two MobileNet-V2-based NIMA models (one trained on AVA aesthetic labels, one on AVA technical labels) run in parallel per image. Their scores are averaged and mapped to 1–5 stars via fixed thresholds derived from NIMA's calibrated output range. Because the thresholds are absolute, every image gets a stable score regardless of what else is in the session.

**Tech Stack:** Python 3.12, TensorFlow 2.x, coremltools 8+, Swift 5.9, CoreML, CoreData

---

## File Map

| Action   | Path                                          | Responsibility                                      |
|----------|-----------------------------------------------|-----------------------------------------------------|
| Create   | `scripts/convert_nima.py`                     | Download NIMA Keras weights, convert both variants to CoreML |
| Create   | `scripts/requirements-nima.txt`               | TF + coremltools deps for NIMA conversion           |
| Modify   | `ImageRater/Pipeline/RatingPipeline.swift`    | New `rate()` using 2 NIMA models; absolute star mapping; remove z-score |
| Modify   | `ImageRater/Pipeline/ProcessingQueue.swift`   | Load `nima-aesthetic` + `nima-technical`; remove rawScores accumulation; write stars immediately |
| Modify   | `ImageRater/Models/RatingResult.swift`        | Rename `clipScore` → `technicalScore` to reflect new semantics |
| Modify   | `ImageRaterTests/RatingPipelineTests.swift`   | Rewrite tests for new API (old tests reference removed methods) |

---

## Background: NIMA

NIMA (Neural Image Assessment, Google 2017) fine-tunes MobileNet on the AVA dataset (255k photos rated by humans in photography competitions). It outputs a probability distribution over 10 quality bins; the mean of that distribution is the score [1–10].

Two variants share the same architecture but different weights:
- **Aesthetic**: trained on aesthetic quality ratings — composition, lighting, colour
- **Technical**: trained on technical quality ratings — sharpness, noise, exposure, distortion

Combining them with equal weight covers both what *looks* good and what is *technically* correct.

**Calibrated thresholds** (derived from AVA dataset statistics, validated in the NIMA paper):
```
combined < 4.0  →  1★  (technically or aesthetically poor)
combined < 4.8  →  2★  (below average)
combined < 5.6  →  3★  (average — most photos land here)
combined < 6.4  →  4★  (good)
combined ≥ 6.4  →  5★  (exceptional)
```

---

## Task 1: Create NIMA requirements file

**Files:**
- Create: `scripts/requirements-nima.txt`

- [ ] **Step 1: Write requirements file**

```
tensorflow>=2.13,<3.0
tf_keras
coremltools>=8.0
numpy
```

- [ ] **Step 2: Verify install works**

```bash
cd /Users/dknathalage/repos/image-rating/scripts
python -m venv .venv-nima
source .venv-nima/bin/activate
pip install -r requirements-nima.txt
python -c "import tensorflow as tf; import coremltools as ct; print('OK')"
```

Expected: `OK`

---

## Task 2: Write NIMA conversion script

**Files:**
- Create: `scripts/convert_nima.py`

- [ ] **Step 1: Write the conversion script**

```python
#!/usr/bin/env python3
"""
Convert NIMA aesthetic and technical quality models to CoreML .mlpackage.

Outputs:
  models/nima-aesthetic.mlpackage  -- image -> score (1-10, AVA aesthetic quality)
  models/nima-technical.mlpackage  -- image -> score (1-10, AVA technical quality)

Weights sourced from:
  https://github.com/idealo/image-quality-assessment
  (MobileNet aesthetic: weights_mobilenet_aesthetic_0.07.hdf5)
  (MobileNet technical: weights_mobilenet_technical_0.11.hdf5)

Both models accept 224x224 RGB images and output a scalar mean opinion score [1–10].
"""

import urllib.request
import sys
from pathlib import Path

import numpy as np
import coremltools as ct

MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

WEIGHTS_DIR = Path(__file__).parent / "nima_weights"
WEIGHTS_DIR.mkdir(exist_ok=True)

AESTHETIC_WEIGHTS_URL = (
    "https://github.com/idealo/image-quality-assessment/raw/master/"
    "weights/MobileNet/aesthetic/weights_mobilenet_aesthetic_0.07.hdf5"
)
TECHNICAL_WEIGHTS_URL = (
    "https://github.com/idealo/image-quality-assessment/raw/master/"
    "weights/MobileNet/technical/weights_mobilenet_technical_0.11.hdf5"
)

AESTHETIC_WEIGHTS_PATH = WEIGHTS_DIR / "weights_mobilenet_aesthetic_0.07.hdf5"
TECHNICAL_WEIGHTS_PATH = WEIGHTS_DIR / "weights_mobilenet_technical_0.11.hdf5"


def download_weights():
    for url, path, label in [
        (AESTHETIC_WEIGHTS_URL, AESTHETIC_WEIGHTS_PATH, "aesthetic"),
        (TECHNICAL_WEIGHTS_URL, TECHNICAL_WEIGHTS_PATH, "technical"),
    ]:
        if path.exists():
            print(f"  {label} weights already present: {path.name}")
            continue
        print(f"  Downloading NIMA {label} weights…")
        try:
            urllib.request.urlretrieve(url, path)
            print(f"  Downloaded: {path.name}")
        except Exception as e:
            print(f"  ERROR: could not download {label} weights: {e}")
            print(f"  Please manually download from:")
            print(f"    {url}")
            print(f"  and place at: {path}")
            sys.exit(1)


def build_nima_model(weights_path: Path):
    """
    Build NIMA MobileNet model from Keras weights.

    Architecture mirrors idealo/image-quality-assessment:
      MobileNet (no top, avg pooling) -> Dropout(0.75) -> Dense(10, softmax)
    Input: (224, 224, 3) float32 [0–255] RGB
    Output: scalar mean opinion score [1–10]
    """
    # Import TF inside function so the module loads without TF if only checking
    import tensorflow as tf
    import tf_keras as keras

    base = keras.applications.MobileNet(
        input_shape=(224, 224, 3),
        include_top=False,
        pooling="avg",
        weights=None,
    )
    x = keras.layers.Dropout(0.75)(base.output)
    x = keras.layers.Dense(10, activation="softmax", name="predictions")(x)
    model = keras.Model(inputs=base.input, outputs=x)
    model.load_weights(str(weights_path))
    model.trainable = False

    # Wrap: accept [0-255] RGB, output mean opinion score [1-10]
    inp = keras.Input(shape=(224, 224, 3), name="image")
    # Normalise to MobileNet's expected range: [0, 1] then (x - 0.5) / 0.5
    x_norm = keras.layers.Rescaling(scale=1.0 / 127.5, offset=-1.0)(inp)
    dist = model(x_norm)  # (batch, 10) softmax distribution
    bins = tf.constant(
        [[float(i) for i in range(1, 11)]], dtype=tf.float32
    )  # (1, 10)
    score = tf.reduce_sum(dist * bins, axis=1, keepdims=True)  # (batch, 1)
    wrapped = keras.Model(inputs=inp, outputs=score, name="nima_scorer")
    return wrapped


def convert_nima(keras_model, output_name: str, save_path: Path):
    print(f"  Converting to CoreML: {save_path.name}")
    mlmodel = ct.convert(
        keras_model,
        inputs=[
            ct.ImageType(
                name="image",
                shape=(1, 224, 224, 3),
                color_layout=ct.colorlayout.RGB,
                bias=[-1.0, -1.0, -1.0],       # undo Rescaling wrapper — let CoreML handle
                scale=1.0 / 127.5,
            )
        ],
        outputs=[ct.TensorType(name="score")],
        minimum_deployment_target=ct.target.macOS15,
        compute_units=ct.ComputeUnit.ALL,
        convert_to="mlprogram",
    )
    mlmodel.save(str(save_path))
    print(f"  Saved: {save_path}")


def verify_model(save_path: Path):
    """Quick sanity check: score a black 224×224 image, expect ~4-6 range."""
    import coremltools as ct
    import numpy as np
    from PIL import Image

    model = ct.models.MLModel(str(save_path))
    black = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
    result = model.predict({"image": black})
    score = float(result["score"])
    assert 1.0 <= score <= 10.0, f"Score out of expected range: {score}"
    print(f"  Sanity check passed: black image → {score:.3f}")


def main():
    print("=== Downloading NIMA weights ===")
    download_weights()

    for label, weights_path, out_name in [
        ("aesthetic", AESTHETIC_WEIGHTS_PATH, "nima-aesthetic"),
        ("technical", TECHNICAL_WEIGHTS_PATH, "nima-technical"),
    ]:
        out_path = MODELS_DIR / f"{out_name}.mlpackage"
        if out_path.exists():
            print(f"\n=== Skipping {out_name} (already exists) ===")
            continue

        print(f"\n=== Building NIMA {label} model ===")
        keras_model = build_nima_model(weights_path)

        print(f"\n=== Converting NIMA {label} to CoreML ===")
        convert_nima(keras_model, "score", out_path)

        print(f"\n=== Verifying {out_name} ===")
        verify_model(out_path)

    print("\n=== Done ===")
    print(f"Models written to: {MODELS_DIR.resolve()}")
    print("Copy .mlpackage files to Application Support/ImageRater/models/")
    print("renaming each to <name>-local.mlpackage for local development.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the conversion**

```bash
cd /Users/dknathalage/repos/image-rating/scripts
source .venv-nima/bin/activate
python convert_nima.py
```

Expected output (summary):
```
=== Downloading NIMA weights ===
  Downloading NIMA aesthetic weights…
  Downloading NIMA technical weights…
=== Building NIMA aesthetic model ===
=== Converting NIMA aesthetic to CoreML ===
  Saved: .../models/nima-aesthetic.mlpackage
  Sanity check passed: black image → X.XXX
=== Building NIMA technical model ===
...
=== Done ===
```

- [ ] **Step 3: Install models for local development**

```bash
# Find Application Support models dir (run app once first if this dir doesn't exist)
MODELS_DIR="$HOME/Library/Application Support/ImageRater/models"
mkdir -p "$MODELS_DIR"

cp -r models/nima-aesthetic.mlpackage "$MODELS_DIR/nima-aesthetic-local.mlpackage"
cp -r models/nima-technical.mlpackage "$MODELS_DIR/nima-technical-local.mlpackage"

# Write sidecar files so ModelStore treats them as local imports
echo "local" > "$MODELS_DIR/nima-aesthetic-local.mlpackage.sha256"
echo "local" > "$MODELS_DIR/nima-technical-local.mlpackage.sha256"
```

- [ ] **Step 4: Commit script**

```bash
cd /Users/dknathalage/repos/image-rating
git add scripts/convert_nima.py scripts/requirements-nima.txt
git commit -m "feat: add NIMA aesthetic+technical CoreML conversion script"
```

---

## Task 3: Update RatingResult model

**Files:**
- Modify: `ImageRater/Models/RatingResult.swift`

The `clipScore` field is repurposed to hold the NIMA technical score. Rename for clarity.

- [ ] **Step 1: Rewrite RatingResult**

```swift
import Foundation

struct RatingResult: Equatable {
    /// 1–5 stars. 0 = unrated/failed.
    let stars: Int
    /// NIMA aesthetic quality score [1–10]. Higher = more aesthetically pleasing.
    let aestheticScore: Float
    /// NIMA technical quality score [1–10]. Higher = sharper, less noise, better exposure.
    let technicalScore: Float

    static let unrated = RatingResult(stars: 0, aestheticScore: 0, technicalScore: 0)

    /// Combined score used for star mapping: simple average of both dimensions.
    var combinedScore: Float { (aestheticScore + technicalScore) / 2.0 }
}
```

- [ ] **Step 2: Fix any compile errors from renamed `clipScore` field**

Search across the project and replace:
```
ratingResult.clipScore  →  ratingResult.technicalScore
result.clipScore        →  result.technicalScore
```

CoreData still uses `clipScore` as the field name (schema change is expensive). The ProcessingQueue update in Task 5 will store `technicalScore` into `record.clipScore`.

- [ ] **Step 3: Commit**

```bash
git add ImageRater/Models/RatingResult.swift
git commit -m "refactor: rename clipScore → technicalScore in RatingResult"
```

---

## Task 4: Rewrite RatingPipeline

**Files:**
- Modify: `ImageRater/Pipeline/RatingPipeline.swift`

Remove: `rawCombinedScore`, `assignStarsRelative`, old `rate()` signature.  
Add: `absoluteStars(combined:)`, new `rate()` taking two NIMA models.

- [ ] **Step 1: Write the tests first**

Replace `ImageRaterTests/RatingPipelineTests.swift`:

```swift
import XCTest
@testable import ImageRater

final class RatingPipelineTests: XCTestCase {

    // MARK: - absoluteStars thresholds

    func testAbsoluteStars_below4_is1Star() {
        XCTAssertEqual(RatingPipeline.absoluteStars(combined: 3.9), 1)
        XCTAssertEqual(RatingPipeline.absoluteStars(combined: 1.0), 1)
    }

    func testAbsoluteStars_4to4_8_is2Stars() {
        XCTAssertEqual(RatingPipeline.absoluteStars(combined: 4.0), 2)
        XCTAssertEqual(RatingPipeline.absoluteStars(combined: 4.7), 2)
    }

    func testAbsoluteStars_4_8to5_6_is3Stars() {
        XCTAssertEqual(RatingPipeline.absoluteStars(combined: 4.8), 3)
        XCTAssertEqual(RatingPipeline.absoluteStars(combined: 5.5), 3)
    }

    func testAbsoluteStars_5_6to6_4_is4Stars() {
        XCTAssertEqual(RatingPipeline.absoluteStars(combined: 5.6), 4)
        XCTAssertEqual(RatingPipeline.absoluteStars(combined: 6.3), 4)
    }

    func testAbsoluteStars_6_4plus_is5Stars() {
        XCTAssertEqual(RatingPipeline.absoluteStars(combined: 6.4), 5)
        XCTAssertEqual(RatingPipeline.absoluteStars(combined: 10.0), 5)
    }

    func testAbsoluteStars_clampsBelowRange() {
        XCTAssertEqual(RatingPipeline.absoluteStars(combined: 0.0), 1)
    }

    // MARK: - Pixel buffer creation

    func testPixelBufferCreationSucceeds() throws {
        let ctx = CGContext(data: nil, width: 10, height: 10,
                           bitsPerComponent: 8, bytesPerRow: 0,
                           space: CGColorSpaceCreateDeviceRGB(),
                           bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        let cgImage = ctx.makeImage()!
        let pb = try RatingPipeline.cgImageToPixelBuffer(cgImage, width: 224, height: 224)
        XCTAssertEqual(CVPixelBufferGetWidth(pb), 224)
        XCTAssertEqual(CVPixelBufferGetHeight(pb), 224)
    }
}
```

- [ ] **Step 2: Run tests to confirm they fail (method missing)**

```bash
xcodebuild test -scheme ImageRater -destination 'platform=macOS' \
  -only-testing:ImageRaterTests/RatingPipelineTests 2>&1 | tail -20
```

Expected: compile error — `absoluteStars` not found.

- [ ] **Step 3: Rewrite RatingPipeline.swift**

```swift
import CoreML
import CoreImage
import Foundation
import OSLog

private let log = Logger(subsystem: "com.imagerating", category: "RatingPipeline")

enum RatingError: Error {
    case pixelBufferCreationFailed
}

enum RatingPipeline {

    // MARK: - Star mapping

    /// Map a combined NIMA score [1–10] to 1–5 stars using calibrated absolute thresholds.
    ///
    /// Thresholds derived from AVA dataset statistics (NIMA paper, 2017):
    ///   < 4.0  →  1★  (poor — technically or aesthetically deficient)
    ///   < 4.8  →  2★  (below average)
    ///   < 5.6  →  3★  (average — centre of the AVA distribution)
    ///   < 6.4  →  4★  (good)
    ///   ≥ 6.4  →  5★  (exceptional)
    static func absoluteStars(combined score: Float) -> Int {
        switch score {
        case ..<4.0: return 1
        case ..<4.8: return 2
        case ..<5.6: return 3
        case ..<6.4: return 4
        default:     return 5
        }
    }

    // MARK: - Inference

    /// Run NIMA aesthetic + technical inference on a CGImage.
    /// Returns .unrated on any failure — never throws.
    static func rate(image: CGImage,
                     nimaAestheticModel: MLModel,
                     nimaTechnicalModel: MLModel) async -> RatingResult {
        do {
            log.debug("Creating pixel buffer \(image.width)×\(image.height) → 224×224")
            let pixelBuffer = try cgImageToPixelBuffer(image, width: 224, height: 224)

            log.debug("Running NIMA aesthetic inference")
            let aestheticInput = try MLDictionaryFeatureProvider(dictionary: ["image": pixelBuffer])
            let aestheticOutput = try await nimaAestheticModel.prediction(from: aestheticInput)
            let aestheticScore = aestheticOutput.featureValue(for: "score").flatMap { fv -> Float? in
                if let arr = fv.multiArrayValue { return arr[0].floatValue }
                return Float(fv.doubleValue)
            } ?? 5.0
            log.info("NIMA aesthetic: \(aestheticScore, format: .fixed(precision: 4))")

            log.debug("Running NIMA technical inference")
            let technicalInput = try MLDictionaryFeatureProvider(dictionary: ["image": pixelBuffer])
            let technicalOutput = try await nimaTechnicalModel.prediction(from: technicalInput)
            let technicalScore = technicalOutput.featureValue(for: "score").flatMap { fv -> Float? in
                if let arr = fv.multiArrayValue { return arr[0].floatValue }
                return Float(fv.doubleValue)
            } ?? 5.0
            log.info("NIMA technical: \(technicalScore, format: .fixed(precision: 4))")

            let combined = (aestheticScore + technicalScore) / 2.0
            let stars = absoluteStars(combined: combined)
            log.info("Combined \(combined, format: .fixed(precision: 3)) → \(stars)★ (aes \(aestheticScore, format: .fixed(precision: 3)), tech \(technicalScore, format: .fixed(precision: 3)))")

            return RatingResult(stars: stars, aestheticScore: aestheticScore, technicalScore: technicalScore)
        } catch {
            log.error("Rating failed: \(error)")
            return .unrated
        }
    }

    // MARK: - Internal

    static func cgImageToPixelBuffer(_ image: CGImage, width: Int, height: Int) throws -> CVPixelBuffer {
        var buffer: CVPixelBuffer?
        let attrs: CFDictionary = [
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true
        ] as CFDictionary
        let status = CVPixelBufferCreate(kCFAllocatorDefault, width, height,
                                         kCVPixelFormatType_32BGRA, attrs, &buffer)
        guard status == kCVReturnSuccess, let pb = buffer else {
            throw RatingError.pixelBufferCreationFailed
        }
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
}
```

- [ ] **Step 4: Run tests — expect pass**

```bash
xcodebuild test -scheme ImageRater -destination 'platform=macOS' \
  -only-testing:ImageRaterTests/RatingPipelineTests 2>&1 | tail -20
```

Expected: `TEST SUCCEEDED`

- [ ] **Step 5: Commit**

```bash
git add ImageRater/Pipeline/RatingPipeline.swift ImageRaterTests/RatingPipelineTests.swift
git commit -m "feat: replace CLIP/LAION with NIMA ensemble, absolute star thresholds"
```

---

## Task 5: Update ProcessingQueue

**Files:**
- Modify: `ImageRater/Pipeline/ProcessingQueue.swift`

Remove: rawScores accumulation, z-score rescaling, sidecar-deferred write.  
Change: load `nima-aesthetic` + `nima-technical`; write stars immediately after each image.

- [ ] **Step 1: Replace model loading (lines ~54–58)**

```swift
// Replace:
log.info("Loading CLIP model")
let clipModel = try await ModelStore.shared.model(named: "clip")
log.info("Loading aesthetic model")
let aestheticModel = try await ModelStore.shared.model(named: "aesthetic")

// With:
log.info("Loading NIMA aesthetic model")
let nimaAestheticModel = try await ModelStore.shared.model(named: "nima-aesthetic")
log.info("Loading NIMA technical model")
let nimaTechnicalModel = try await ModelStore.shared.model(named: "nima-technical")
```

- [ ] **Step 2: Remove the RawScore struct and rawScores array**

Delete these lines entirely:
```swift
struct RawScore {
    let id: NSManagedObjectID
    let clipScore: Float
    let aestheticScore: Float
    let filePath: String?
}
var rawScores: [RawScore] = []
```

- [ ] **Step 3: Replace the rating branch with immediate star write**

Replace everything from `if !hasOverride {` through `log.info("[\(completed+1)/\(total)] Rescaled ...` with:

```swift
if !hasOverride {
    ratedCount += 1
    log.info("[\(completed + 1)/\(total)] Running models on \(filename)")
    onProgress?(completed, total, "Running models: \(completed + 1) of \(total) (rated \(ratedCount))")
    await context.perform { [self] in
        guard let record = try? self.context.existingObject(with: imageID) as? ImageRecord else { return }
        record.processState = ProcessState.rating
        try? self.context.save()
    }

    let ratingResult = await RatingPipeline.rate(
        image: image,
        nimaAestheticModel: nimaAestheticModel,
        nimaTechnicalModel: nimaTechnicalModel
    )

    await context.perform { [self] in
        guard let record = try? self.context.existingObject(with: imageID) as? ImageRecord else { return }
        record.ratingStars     = NSNumber(value: ratingResult.stars)
        record.aestheticScore  = NSNumber(value: ratingResult.aestheticScore)
        record.clipScore       = NSNumber(value: ratingResult.technicalScore)  // reuse existing field
        record.processState    = ProcessState.done
        try? self.context.save()
    }
    if let path = filePath {
        try? MetadataWriter.writeSidecar(stars: ratingResult.stars, for: URL(filePath: path))
    }
```

- [ ] **Step 4: Remove the deferred sidecar block after the do/catch**

Delete:
```swift
// Write sidecars with final star values after all rescaling is complete.
let finalScores = rawScores.map { ... }
let finalStars = RatingPipeline.assignStarsRelative(scores: finalScores)
for (entry, stars) in zip(rawScores, finalStars) { ... }
```

- [ ] **Step 5: Build and verify no compile errors**

```bash
xcodebuild build -scheme ImageRater -destination 'platform=macOS' 2>&1 | tail -20
```

Expected: `BUILD SUCCEEDED`

- [ ] **Step 6: Commit**

```bash
git add ImageRater/Pipeline/ProcessingQueue.swift
git commit -m "refactor: load NIMA models, write stars immediately, remove z-score recalculation"
```

---

## Task 6: Run full test suite

- [ ] **Step 1: Run all tests**

```bash
xcodebuild test -scheme ImageRater -destination 'platform=macOS' 2>&1 | grep -E "PASS|FAIL|error:"
```

Expected: all tests pass. If any test references the old `combineScores`, `starsFromAestheticScore`, or `assignStarsRelative` APIs, update them to use `absoluteStars(combined:)`.

- [ ] **Step 2: Commit if any test fixes were needed**

```bash
git add ImageRaterTests/
git commit -m "test: update remaining tests for NIMA pipeline API"
```

---

## Task 7: Manual smoke test

After installing the converted models (Task 2, Step 3):

- [ ] **Step 1: Launch the app and process a small session (5–10 images)**

Watch the console for:
```
Loading NIMA aesthetic model
Loading NIMA technical model
Models ready
[1/N] Running models on DSCF0001.JPG
NIMA aesthetic: X.XXXX
NIMA technical: X.XXXX
Combined X.XXX → N★
```

- [ ] **Step 2: Verify star distribution is non-trivial**

With NIMA, a typical 100-photo session should produce stars spread across 1–4 (5★ requires genuinely exceptional work). If everything scores 3★, the combined score is clustering around 4.8–5.6 — note the actual values and consider shifting thresholds ±0.3.

- [ ] **Step 3: Verify no per-session rescaling in logs**

There should be NO lines like `Rescaled N rated image(s)`. Each image is scored independently.

---

## Threshold Tuning Reference

If the initial thresholds produce a poor distribution (too many 3★), adjust `absoluteStars` using this guide:

| Session results          | Adjustment                             |
|--------------------------|----------------------------------------|
| Everything 3★            | Lower all thresholds by 0.3–0.5        |
| Too many 1★ (good shots) | Raise lower threshold (4.0 → 3.5)     |
| No 5★ ever               | Lower upper threshold (6.4 → 5.8)     |
| Flat distribution        | Compress middle range (4.8–5.6 → 4.6–5.4) |

Recommended iteration: run 3 sessions, note median combined score, set 3★ center = median ± 0.4.
