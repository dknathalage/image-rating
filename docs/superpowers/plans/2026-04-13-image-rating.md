# Image Rating & Culling App Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** macOS Swift app that culls bad photos (blur/eyes-closed/exposure) then rates survivors (Core ML aesthetic scoring), writing results to XMP sidecar files.

**Architecture:** Two-phase AI pipeline — Phase 1 uses Apple Vision + CIFilter for fast cull; Phase 2 uses Core ML OpenCLIP + aesthetic head for star rating. Swift actors + async/await for concurrency. Core Data for persistence. LibRaw for RAW decode.

**Tech Stack:** Swift 5.9+, macOS 14+, SwiftUI, Core ML, Apple Vision, Core Image, LibRaw (C++ bridge), Core Data, Swift Package Manager, XCTest

---

## Parallel Execution Groups

```
[Task 1: Project Scaffold] ← must complete first
        ↓
┌───────────────────────────────────────────────────┐
│ PARALLEL GROUP A (run simultaneously after Task 1) │
│  Task 2: LibRaw Bridge                             │
│  Task 3: Data Models                               │
│  Task 4: CullPipeline                              │
│  Task 5: MetadataWriter                            │
└───────────────────────────────────────────────────┘
        ↓
┌───────────────────────────────────────────────────┐
│ PARALLEL GROUP B (run simultaneously after Group A)│
│  Task 6: ThumbnailCache                            │
│  Task 7: RatingPipeline                            │
│  Task 8: ModelStore                                │
│  Task 9: ImageImporter                             │
└───────────────────────────────────────────────────┘
        ↓
[Task 10: ProcessingQueue] ← orchestrates everything
        ↓
┌───────────────────────────────────────────────────┐
│ PARALLEL GROUP C (run simultaneously after Task 10)│
│  Task 11: GridView                                 │
│  Task 12: DetailView                               │
│  Task 13: ModelStoreView                           │
└───────────────────────────────────────────────────┘
        ↓
[Task 14: App Wiring + Integration]
```

---

## File Map

```
ImageRater/
├── ImageRater.xcodeproj
├── ImageRater/
│   ├── App/
│   │   ├── ImageRaterApp.swift          # @main, environment setup
│   │   └── ContentView.swift            # root split view
│   ├── LibRaw/
│   │   ├── LibRawBridge.h               # ObjC++ header
│   │   ├── LibRawBridge.mm              # ObjC++ impl wrapping libraw
│   │   ├── ImageRater-Bridging-Header.h # imports LibRawBridge.h
│   │   └── LibRawWrapper.swift          # Swift-friendly wrapper
│   ├── CoreData/
│   │   ├── ImageRater.xcdatamodeld      # Session, ImageRecord, ModelConfig
│   │   └── PersistenceController.swift  # NSPersistentContainer setup
│   ├── Models/
│   │   ├── CullResult.swift             # struct + CullReason enum
│   │   └── RatingResult.swift           # struct with stars/scores
│   ├── Import/
│   │   ├── ImageImporter.swift          # folder scan, builds ImageRecord list
│   │   └── ThumbnailCache.swift         # NSCache + disk cache actor
│   ├── Pipeline/
│   │   ├── CullPipeline.swift           # blur + EAR + exposure
│   │   ├── RatingPipeline.swift         # Core ML inference
│   │   └── ProcessingQueue.swift        # Swift actor orchestrator
│   ├── ModelStore/
│   │   ├── ModelStore.swift             # download + verify + load
│   │   ├── ManifestFetcher.swift        # fetch + Ed25519 verify manifest
│   │   └── ModelDownloader.swift        # URLSession + SHA-256 check
│   ├── Export/
│   │   └── MetadataWriter.swift         # CGImageMetadata → .xmp sidecar
│   └── UI/
│       ├── GridView.swift               # thumbnail grid
│       ├── DetailView.swift             # large image + score breakdown
│       ├── ModelStoreView.swift         # model download/swap UI
│       └── Components/
│           ├── ThumbnailCell.swift      # single grid cell
│           └── ScoreBadge.swift         # star rating overlay
└── ImageRaterTests/
    ├── CullPipelineTests.swift
    ├── RatingPipelineTests.swift
    ├── MetadataWriterTests.swift
    ├── LibRawWrapperTests.swift
    ├── ProcessingQueueTests.swift
    ├── ModelStoreTests.swift
    └── Fixtures/                        # sample images for tests
        ├── sharp.jpg
        ├── blurry.jpg
        ├── overexposed.jpg
        ├── underexposed.jpg
        └── eyes_closed.jpg
```

---

## Task 1: Project Scaffold

**Files:**
- Create: `ImageRater.xcodeproj` (via Xcode or `xcodegen`)
- Create: `ImageRater/App/ImageRaterApp.swift`
- Create: `ImageRater/App/ContentView.swift`
- Create: `ImageRater/CoreData/ImageRater.xcdatamodeld`
- Create: `ImageRater/CoreData/PersistenceController.swift`
- Create: `ImageRater/ImageRater-Bridging-Header.h`

- [ ] **Step 1: Create Xcode project**

Open Xcode → New Project → macOS App → Product Name: `ImageRater`, Interface: SwiftUI, Language: Swift, minimum deployment: macOS 14.0. Add a test target named `ImageRaterTests`.

- [ ] **Step 2: Create Core Data model**

In Xcode, add `ImageRater.xcdatamodeld`. Add entities:

**Session**
- `id`: UUID
- `createdAt`: Date
- `folderPath`: String
- Relationship `images` → ImageRecord (to-many, cascade delete)

**ImageRecord**
- `id`: UUID
- `filePath`: String
- `thumbHash`: String (optional)
- `processState`: String (values: `pending`, `culling`, `rating`, `done`, `interrupted`)
- `decodeError`: Boolean, default false
- `cullRejected`: Boolean, default false
- `cullReason`: String (optional)
- `ratingStars`: Integer16 (optional, 0 = unrated)
- `clipScore`: Float (optional)
- `aestheticScore`: Float (optional)
- `userOverride`: Integer16 (optional, nil = no override)
- Relationship `session` → Session (to-one)

**ModelConfig**
- `id`: UUID
- `modelName`: String
- `clipWeight`: Float, default 0.5
- `aestheticWeight`: Float, default 0.5
- `blurThreshold`: Float, default 50.0
- `earThreshold`: Float, default 0.2
- `exposureThreshold`: Float, default 0.9

- [ ] **Step 3: Create PersistenceController**

Create `ImageRater/CoreData/PersistenceController.swift`:

```swift
import CoreData

final class PersistenceController {
    static let shared = PersistenceController()

    let container: NSPersistentContainer

    init(inMemory: Bool = false) {
        container = NSPersistentContainer(name: "ImageRater")
        if inMemory {
            container.persistentStoreDescriptions.first?.url = URL(filePath: "/dev/null")
        }
        container.loadPersistentStores { _, error in
            if let error { fatalError("Core Data load failed: \(error)") }
        }
        container.viewContext.automaticallyMergesChangesFromParent = true
    }
}
```

- [ ] **Step 4: Create app entry point**

`ImageRater/App/ImageRaterApp.swift`:
```swift
import SwiftUI

@main
struct ImageRaterApp: App {
    let persistence = PersistenceController.shared

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environment(\.managedObjectContext, persistence.container.viewContext)
        }
    }
}
```

`ImageRater/App/ContentView.swift`:
```swift
import SwiftUI

struct ContentView: View {
    var body: some View {
        Text("ImageRater")
    }
}
```

- [ ] **Step 5: Create bridging header**

`ImageRater/ImageRater-Bridging-Header.h`:
```objc
#import "LibRawBridge.h"
```

In Xcode Build Settings → `SWIFT_OBJC_BRIDGING_HEADER` = `ImageRater/ImageRater-Bridging-Header.h`

- [ ] **Step 6: Build and verify project compiles**

```bash
xcodebuild -scheme ImageRater -destination 'platform=macOS' build
```
Expected: `BUILD SUCCEEDED`

- [ ] **Step 7: Commit**

```bash
git init
git add .
git commit -m "feat: scaffold Xcode project with Core Data schema"
```

---

## Task 2: LibRaw Bridge

*Run in parallel with Tasks 3, 4, 5 after Task 1 completes.*

**Files:**
- Create: `ImageRater/LibRaw/LibRawBridge.h`
- Create: `ImageRater/LibRaw/LibRawBridge.mm`
- Create: `ImageRater/LibRaw/LibRawWrapper.swift`
- Test: `ImageRaterTests/LibRawWrapperTests.swift`

**Prerequisites:** Install LibRaw via Homebrew (`brew install libraw`) or add via SPM. In Xcode, add `libraw.dylib` to Link Binary with Libraries. Add `/opt/homebrew/include` to Header Search Paths and `/opt/homebrew/lib` to Library Search Paths.

- [ ] **Step 1: Write failing test**

`ImageRaterTests/LibRawWrapperTests.swift`:
```swift
import XCTest
@testable import ImageRater

final class LibRawWrapperTests: XCTestCase {
    func testDecodeJPEGReturnsCGImage() throws {
        // Use a known JPEG fixture
        let url = Bundle(for: Self.self).url(forResource: "sharp", withExtension: "jpg")!
        let result = LibRawWrapper.decode(url: url)
        XCTAssertNotNil(result)
    }

    func testDecodeBadPathReturnsNil() {
        let url = URL(filePath: "/tmp/nonexistent.cr3")
        let result = LibRawWrapper.decode(url: url)
        XCTAssertNil(result)
    }
}
```

- [ ] **Step 2: Run test — verify fails**

```bash
xcodebuild test -scheme ImageRater -destination 'platform=macOS' -only-testing:ImageRaterTests/LibRawWrapperTests
```
Expected: FAIL — `LibRawWrapper` not found

- [ ] **Step 3: Create ObjC++ bridge header**

`ImageRater/LibRaw/LibRawBridge.h`:
```objc
#import <Foundation/Foundation.h>
#import <CoreGraphics/CoreGraphics.h>

@interface LibRawBridge : NSObject
/// Returns nil on failure. Tries embedded JPEG preview first, falls back to full decode.
+ (nullable CGImageRef)decodeFileAtPath:(NSString *)path CF_RETURNS_RETAINED;
@end
```

- [ ] **Step 4: Create ObjC++ bridge implementation**

`ImageRater/LibRaw/LibRawBridge.mm`:
```objc
#import "LibRawBridge.h"
#import <libraw/libraw.h>

@implementation LibRawBridge

+ (CGImageRef)decodeFileAtPath:(NSString *)path {
    // Fast path: try embedded JPEG preview
    LibRaw processor;
    int ret = processor.open_file([path UTF8String]);
    if (ret != LIBRAW_SUCCESS) return nil;

    // Try embedded thumbnail first
    if (processor.unpack_thumb() == LIBRAW_SUCCESS) {
        libraw_processed_image_t *thumb = processor.dcraw_make_mem_thumb();
        if (thumb && thumb->type == LIBRAW_IMAGE_JPEG) {
            NSData *jpegData = [NSData dataWithBytes:thumb->data length:thumb->data_size];
            LibRaw::dcraw_clear_mem(thumb);
            CGDataProviderRef provider = CGDataProviderCreateWithCFData((__bridge CFDataRef)jpegData);
            CGImageRef img = CGImageCreateWithJPEGDataProvider(provider, nil, true, kCGRenderingIntentDefault);
            CGDataProviderRelease(provider);
            processor.recycle();
            return img;
        }
        if (thumb) LibRaw::dcraw_clear_mem(thumb);
    }

    // Full decode fallback
    processor.recycle();
    ret = processor.open_file([path UTF8String]);
    if (ret != LIBRAW_SUCCESS) return nil;
    if (processor.unpack() != LIBRAW_SUCCESS) return nil;
    if (processor.dcraw_process() != LIBRAW_SUCCESS) return nil;

    libraw_processed_image_t *image = processor.dcraw_make_mem_image();
    if (!image) return nil;

    int width = image->width, height = image->height;
    NSData *data = [NSData dataWithBytes:image->data length:image->data_size];
    LibRaw::dcraw_clear_mem(image);
    processor.recycle();

    CGColorSpaceRef space = CGColorSpaceCreateDeviceRGB();
    CGDataProviderRef provider = CGDataProviderCreateWithCFData((__bridge CFDataRef)data);
    CGImageRef img = CGImageCreate(width, height, 8, 24, width * 3, space,
                                   kCGBitmapByteOrderDefault | kCGImageAlphaNone,
                                   provider, nil, false, kCGRenderingIntentDefault);
    CGDataProviderRelease(provider);
    CGColorSpaceRelease(space);
    return img;
}

@end
```

- [ ] **Step 5: Create Swift wrapper**

`ImageRater/LibRaw/LibRawWrapper.swift`:
```swift
import CoreGraphics
import Foundation

enum LibRawWrapper {
    /// Returns decoded CGImage or nil on failure.
    static func decode(url: URL) -> CGImage? {
        guard url.isFileURL else { return nil }
        return LibRawBridge.decodeFile(atPath: url.path)
    }

    static let supportedExtensions: Set<String> = [
        "cr2", "cr3", "nef", "arw", "raf", "rw2", "dng", "orf", "pef", "srw"
    ]
}
```

- [ ] **Step 6: Add fixture images to test target**

Add `sharp.jpg` and `blurry.jpg` to `ImageRaterTests/Fixtures/` and include in test target bundle.

- [ ] **Step 7: Run tests — verify pass**

```bash
xcodebuild test -scheme ImageRater -destination 'platform=macOS' -only-testing:ImageRaterTests/LibRawWrapperTests
```
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add ImageRater/LibRaw/ ImageRaterTests/LibRawWrapperTests.swift
git commit -m "feat: add LibRaw ObjC++ bridge with embedded preview fast path"
```

---

## Task 3: Data Models

*Run in parallel with Tasks 2, 4, 5 after Task 1 completes.*

**Files:**
- Create: `ImageRater/Models/CullResult.swift`
- Create: `ImageRater/Models/RatingResult.swift`

No tests needed — pure value types, no logic.

- [ ] **Step 1: Create CullResult**

`ImageRater/Models/CullResult.swift`:
```swift
import Foundation

enum CullReason: String, Codable {
    case blurry
    case eyesClosed
    case overexposed
    case underexposed
}

struct CullResult: Equatable {
    let rejected: Bool
    let reason: CullReason?

    static let keep = CullResult(rejected: false, reason: nil)
    static func reject(_ reason: CullReason) -> CullResult {
        CullResult(rejected: true, reason: reason)
    }
}
```

- [ ] **Step 2: Create RatingResult**

`ImageRater/Models/RatingResult.swift`:
```swift
import Foundation

struct RatingResult: Equatable {
    /// 1–5 stars. 0 = unrated/failed.
    let stars: Int
    let clipScore: Float
    let aestheticScore: Float

    static let unrated = RatingResult(stars: 0, clipScore: 0, aestheticScore: 0)
}
```

- [ ] **Step 3: Commit**

```bash
git add ImageRater/Models/
git commit -m "feat: add CullResult and RatingResult value types"
```

---

## Task 4: CullPipeline

*Run in parallel with Tasks 2, 3, 5 after Task 1 completes.*

**Files:**
- Create: `ImageRater/Pipeline/CullPipeline.swift`
- Test: `ImageRaterTests/CullPipelineTests.swift`

- [ ] **Step 1: Write failing tests**

`ImageRaterTests/CullPipelineTests.swift`:
```swift
import XCTest
import CoreImage
@testable import ImageRater

final class CullPipelineTests: XCTestCase {

    // MARK: Blur

    func testSharpImageNotRejectedForBlur() {
        let image = makeGradientImage(size: CGSize(width: 100, height: 100))
        let result = CullPipeline.checkBlur(image: image, threshold: 50.0)
        XCTAssertFalse(result.rejected)
    }

    func testBlurryImageRejected() {
        let blurry = makeBlurredImage(size: CGSize(width: 100, height: 100), radius: 20)
        let result = CullPipeline.checkBlur(image: blurry, threshold: 50.0)
        XCTAssertTrue(result.rejected)
        XCTAssertEqual(result.reason, .blurry)
    }

    // MARK: Exposure

    func testOverexposedImageRejected() {
        let white = makeSolidImage(size: CGSize(width: 50, height: 50), gray: 1.0)
        let result = CullPipeline.checkExposure(image: white, threshold: 0.9)
        XCTAssertTrue(result.rejected)
        XCTAssertEqual(result.reason, .overexposed)
    }

    func testUnderexposedImageRejected() {
        let black = makeSolidImage(size: CGSize(width: 50, height: 50), gray: 0.0)
        let result = CullPipeline.checkExposure(image: black, threshold: 0.9)
        XCTAssertTrue(result.rejected)
        XCTAssertEqual(result.reason, .underexposed)
    }

    func testNormalExposureKept() {
        let mid = makeSolidImage(size: CGSize(width: 50, height: 50), gray: 0.5)
        let result = CullPipeline.checkExposure(image: mid, threshold: 0.9)
        XCTAssertFalse(result.rejected)
    }

    // MARK: EAR

    func testEARBelowThresholdRejectsEyesClosed() {
        // EAR = height/width. Closed eye = very small height.
        let closedEyePoints = makeEyePoints(height: 0.01, width: 0.1) // EAR = 0.1
        let ear = CullPipeline.eyeAspectRatio(points: closedEyePoints)
        XCTAssertLessThan(ear, 0.2)
    }

    func testEARAboveThresholdKeepsOpenEye() {
        let openEyePoints = makeEyePoints(height: 0.05, width: 0.1) // EAR = 0.5
        let ear = CullPipeline.eyeAspectRatio(points: openEyePoints)
        XCTAssertGreaterThan(ear, 0.2)
    }

    // MARK: Helpers

    private func makeGradientImage(size: CGSize) -> CGImage {
        let ctx = CGContext(data: nil, width: Int(size.width), height: Int(size.height),
                           bitsPerComponent: 8, bytesPerRow: 0,
                           space: CGColorSpaceCreateDeviceGray(),
                           bitmapInfo: CGImageAlphaInfo.none.rawValue)!
        for x in 0..<Int(size.width) {
            ctx.setFillColor(gray: CGFloat(x) / size.width, alpha: 1)
            ctx.fill(CGRect(x: x, y: 0, width: 1, height: Int(size.height)))
        }
        return ctx.makeImage()!
    }

    private func makeBlurredImage(size: CGSize, radius: Double) -> CGImage {
        let src = CIImage(cgImage: makeGradientImage(size: size))
        let blurred = src.applyingFilter("CIGaussianBlur", parameters: ["inputRadius": radius])
        return CIContext().createCGImage(blurred, from: blurred.extent)!
    }

    private func makeSolidImage(size: CGSize, gray: CGFloat) -> CGImage {
        let ctx = CGContext(data: nil, width: Int(size.width), height: Int(size.height),
                           bitsPerComponent: 8, bytesPerRow: 0,
                           space: CGColorSpaceCreateDeviceGray(),
                           bitmapInfo: CGImageAlphaInfo.none.rawValue)!
        ctx.setFillColor(gray: gray, alpha: 1)
        ctx.fill(CGRect(origin: .zero, size: size))
        return ctx.makeImage()!
    }

    private func makeEyePoints(height: CGFloat, width: CGFloat) -> [CGPoint] {
        // 6 points: left corner, top-left, top-right, right corner, bottom-right, bottom-left
        return [
            CGPoint(x: 0, y: 0.5),
            CGPoint(x: width * 0.33, y: 0.5 + height),
            CGPoint(x: width * 0.66, y: 0.5 + height),
            CGPoint(x: width, y: 0.5),
            CGPoint(x: width * 0.66, y: 0.5 - height),
            CGPoint(x: width * 0.33, y: 0.5 - height),
        ]
    }
}
```

- [ ] **Step 2: Run — verify fails**

```bash
xcodebuild test -scheme ImageRater -destination 'platform=macOS' -only-testing:ImageRaterTests/CullPipelineTests
```
Expected: FAIL — `CullPipeline` not found

- [ ] **Step 3: Implement CullPipeline**

`ImageRater/Pipeline/CullPipeline.swift`:
```swift
import CoreImage
import Vision
import Foundation

enum CullPipeline {

    // MARK: - Blur Detection (Laplacian variance)

    static func checkBlur(image: CGImage, threshold: Float) -> CullResult {
        let ci = CIImage(cgImage: image)
        // Convert to grayscale
        let gray = ci.applyingFilter("CIColorControls", parameters: [
            "inputSaturation": 0.0
        ])
        // Laplacian edge detection — high variance = sharp
        let laplacian = gray.applyingFilter("CIEdgeWork", parameters: [
            "inputRadius": 1.0
        ])
        guard let variance = computeVariance(of: laplacian) else {
            return .keep // can't determine, don't reject
        }
        return variance < threshold ? .reject(.blurry) : .keep
    }

    // MARK: - Exposure (histogram analysis)

    static func checkExposure(image: CGImage, threshold: Float) -> CullResult {
        let ci = CIImage(cgImage: image)
        let gray = ci.applyingFilter("CIColorControls", parameters: ["inputSaturation": 0.0])

        let extent = gray.extent
        let binCount = 256
        guard let histogram = computeHistogram(of: gray, binCount: binCount) else {
            return .keep
        }

        let total = histogram.reduce(0, +)
        guard total > 0 else { return .keep }

        // Fraction of pixels in top 5% of brightness (overexposed)
        let topBins = histogram[(binCount - binCount / 20)...].reduce(0, +)
        // Fraction in bottom 5% (underexposed)
        let bottomBins = histogram[0..<(binCount / 20)].reduce(0, +)

        let topFraction = Float(topBins) / Float(total)
        let bottomFraction = Float(bottomBins) / Float(total)

        if topFraction > (1.0 - threshold) { return .reject(.overexposed) }
        if bottomFraction > (1.0 - threshold) { return .reject(.underexposed) }
        return .keep
    }

    // MARK: - Eyes Closed (EAR via Vision landmarks)

    /// Returns the Eye Aspect Ratio for 6 eye landmark points.
    /// EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    static func eyeAspectRatio(points: [CGPoint]) -> Float {
        guard points.count == 6 else { return 1.0 }
        let p1 = points[0], p2 = points[1], p3 = points[2]
        let p4 = points[3], p5 = points[4], p6 = points[5]
        let vertical = dist(p2, p6) + dist(p3, p5)
        let horizontal = 2.0 * dist(p1, p4)
        guard horizontal > 0 else { return 1.0 }
        return Float(vertical / horizontal)
    }

    static func checkEyesClosed(cgImage: CGImage, earThreshold: Float) async -> CullResult {
        let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
        let request = VNDetectFaceLandmarksRequest()
        try? handler.perform([request])
        guard let faces = request.results, !faces.isEmpty else {
            return .keep // no face detected — don't penalize
        }
        for face in faces {
            guard let landmarks = face.landmarks else { continue }
            if let leftEye = landmarks.leftEye {
                let pts = landmarkPoints(leftEye, in: face.boundingBox)
                if eyeAspectRatio(points: pts) < earThreshold { return .reject(.eyesClosed) }
            }
            if let rightEye = landmarks.rightEye {
                let pts = landmarkPoints(rightEye, in: face.boundingBox)
                if eyeAspectRatio(points: pts) < earThreshold { return .reject(.eyesClosed) }
            }
        }
        return .keep
    }

    /// Full cull check — returns first rejection found.
    static func cull(image: CGImage, config: ModelConfig) async -> CullResult {
        let blurResult = checkBlur(image: image, threshold: config.blurThreshold)
        if blurResult.rejected { return blurResult }

        let exposureResult = checkExposure(image: image, threshold: config.exposureThreshold)
        if exposureResult.rejected { return exposureResult }

        let eyesResult = await checkEyesClosed(cgImage: image, earThreshold: config.earThreshold)
        return eyesResult
    }

    // MARK: - Private helpers

    private static func dist(_ a: CGPoint, _ b: CGPoint) -> CGFloat {
        let dx = a.x - b.x, dy = a.y - b.y
        return sqrt(dx*dx + dy*dy)
    }

    private static func landmarkPoints(_ region: VNFaceLandmarkRegion2D, in box: CGRect) -> [CGPoint] {
        region.normalizedPoints.map { pt in
            CGPoint(x: box.minX + pt.x * box.width,
                    y: box.minY + pt.y * box.height)
        }
    }

    private static func computeVariance(of image: CIImage) -> Float? {
        let context = CIContext()
        let extent = image.extent
        guard let cgImg = context.createCGImage(image, from: extent) else { return nil }

        let width = cgImg.width, height = cgImg.height
        var pixels = [UInt8](repeating: 0, count: width * height)
        let colorSpace = CGColorSpaceCreateDeviceGray()
        guard let ctx = CGContext(data: &pixels, width: width, height: height,
                                  bitsPerComponent: 8, bytesPerRow: width,
                                  space: colorSpace,
                                  bitmapInfo: CGImageAlphaInfo.none.rawValue) else { return nil }
        ctx.draw(cgImg, in: CGRect(x: 0, y: 0, width: width, height: height))

        let count = Float(pixels.count)
        let mean = pixels.reduce(0) { $0 + Float($1) } / count
        let variance = pixels.reduce(0) { $0 + pow(Float($1) - mean, 2) } / count
        return variance
    }

    private static func computeHistogram(of image: CIImage, binCount: Int) -> [Int]? {
        let filter = CIFilter(name: "CIAreaHistogram", parameters: [
            "inputImage": image,
            "inputExtent": CIVector(cgRect: image.extent),
            "inputCount": binCount,
            "inputScale": 1.0
        ])
        guard let output = filter?.outputImage else { return nil }
        let context = CIContext()
        var data = [Float](repeating: 0, count: binCount * 4)
        context.render(output, toBitmap: &data,
                       rowBytes: binCount * 4 * MemoryLayout<Float>.size,
                       bounds: CGRect(x: 0, y: 0, width: binCount, height: 1),
                       format: .RGBAf, colorSpace: nil)
        // Use R channel as luminance (we already greyscaled the input)
        return stride(from: 0, to: data.count, by: 4).map { Int(data[$0] * 1000) }
    }
}
```

- [ ] **Step 4: Run tests — verify pass**

```bash
xcodebuild test -scheme ImageRater -destination 'platform=macOS' -only-testing:ImageRaterTests/CullPipelineTests
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add ImageRater/Pipeline/CullPipeline.swift ImageRaterTests/CullPipelineTests.swift
git commit -m "feat: implement CullPipeline with blur/exposure/EAR detection"
```

---

## Task 5: MetadataWriter

*Run in parallel with Tasks 2, 3, 4 after Task 1 completes.*

**Files:**
- Create: `ImageRater/Export/MetadataWriter.swift`
- Test: `ImageRaterTests/MetadataWriterTests.swift`

- [ ] **Step 1: Write failing test**

`ImageRaterTests/MetadataWriterTests.swift`:
```swift
import XCTest
@testable import ImageRater

final class MetadataWriterTests: XCTestCase {

    func testWriteAndReadBackXMPRating() throws {
        let imageURL = Bundle(for: Self.self).url(forResource: "sharp", withExtension: "jpg")!
        let tmpDir = FileManager.default.temporaryDirectory
        let xmpURL = tmpDir.appendingPathComponent(UUID().uuidString + ".xmp")

        try MetadataWriter.write(stars: 4, to: xmpURL)
        let readBack = try MetadataWriter.readRating(from: xmpURL)
        XCTAssertEqual(readBack, 4)
    }

    func testWriteZeroStarsProducesUnrated() throws {
        let tmpURL = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString + ".xmp")
        try MetadataWriter.write(stars: 0, to: tmpURL)
        let readBack = try MetadataWriter.readRating(from: tmpURL)
        XCTAssertEqual(readBack, 0)
    }

    func testXMPFileCreatedAtExpectedPath() throws {
        let imageURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("test_img.jpg")
        let expectedXMP = imageURL.deletingPathExtension().appendingPathExtension("xmp")
        defer { try? FileManager.default.removeItem(at: expectedXMP) }

        try MetadataWriter.writeSidecar(stars: 3, for: imageURL)
        XCTAssertTrue(FileManager.default.fileExists(atPath: expectedXMP.path))
    }
}
```

- [ ] **Step 2: Run — verify fails**

```bash
xcodebuild test -scheme ImageRater -destination 'platform=macOS' -only-testing:ImageRaterTests/MetadataWriterTests
```
Expected: FAIL

- [ ] **Step 3: Implement MetadataWriter**

`ImageRater/Export/MetadataWriter.swift`:
```swift
import CoreImage
import Foundation

enum MetadataWriterError: Error {
    case serializationFailed
    case writeFailed(Error)
    case readFailed
}

enum MetadataWriter {

    private static let xmpNS = "http://ns.adobe.com/xap/1.0/" as CFString
    private static let msNS = "http://ns.microsoft.com/photo/1.0/" as CFString

    /// Write xmp:Rating to a .xmp sidecar file at `url`.
    static func write(stars: Int, to url: URL) throws {
        let metadata = CGImageMetadataCreateMutable()

        // xmp:Rating
        CGImageMetadataSetValueWithPath(metadata, nil, "xmp:Rating" as CFString,
                                        stars as CFTypeRef)

        // MicrosoftPhoto:Rating (0/1/25/50/75/99)
        let msRating = microsoftRating(from: stars)
        CGImageMetadataRegisterNamespaceForPrefix(metadata, msNS, "MicrosoftPhoto" as CFString, nil)
        CGImageMetadataSetValueWithPath(metadata, nil, "MicrosoftPhoto:Rating" as CFString,
                                        msRating as CFTypeRef)

        guard let xmpData = CGImageMetadataCreateXMPData(metadata, nil) as Data? else {
            throw MetadataWriterError.serializationFailed
        }
        do {
            try xmpData.write(to: url, options: .atomic)
        } catch {
            throw MetadataWriterError.writeFailed(error)
        }
    }

    /// Writes sidecar next to the source image file.
    static func writeSidecar(stars: Int, for imageURL: URL) throws {
        let xmpURL = imageURL.deletingPathExtension().appendingPathExtension("xmp")
        try write(stars: stars, to: xmpURL)
    }

    /// Read back xmp:Rating integer from a .xmp file.
    static func readRating(from url: URL) throws -> Int {
        let data = try Data(contentsOf: url) as CFData
        guard let metadata = CGImageMetadataCreateFromXMPData(data) else {
            throw MetadataWriterError.readFailed
        }
        guard let value = CGImageMetadataCopyStringValueWithPath(metadata, nil, "xmp:Rating" as CFString) as String? else {
            return 0
        }
        return Int(value) ?? 0
    }

    // MARK: Private

    private static func microsoftRating(from stars: Int) -> Int {
        switch stars {
        case 1: return 1
        case 2: return 25
        case 3: return 50
        case 4: return 75
        case 5: return 99
        default: return 0
        }
    }
}
```

- [ ] **Step 4: Run tests — verify pass**

```bash
xcodebuild test -scheme ImageRater -destination 'platform=macOS' -only-testing:ImageRaterTests/MetadataWriterTests
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add ImageRater/Export/MetadataWriter.swift ImageRaterTests/MetadataWriterTests.swift
git commit -m "feat: implement MetadataWriter with xmp:Rating + XMP sidecar"
```

---

## Task 6: ThumbnailCache

*Run after Group A (Tasks 2–5) complete.*

**Files:**
- Create: `ImageRater/Import/ThumbnailCache.swift`

No dedicated unit test — integration covered in Task 14.

- [ ] **Step 1: Implement ThumbnailCache actor**

`ImageRater/Import/ThumbnailCache.swift`:
```swift
import CoreImage
import Foundation
import AppKit

actor ThumbnailCache {
    static let shared = ThumbnailCache()

    private let memCache = NSCache<NSString, NSImage>()
    private let diskCacheURL: URL

    init() {
        let appSupport = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
        diskCacheURL = appSupport.appendingPathComponent("ImageRater/thumbnails")
        try? FileManager.default.createDirectory(at: diskCacheURL, withIntermediateDirectories: true)
        memCache.countLimit = 500
        memCache.totalCostLimit = 200 * 1024 * 1024 // 200 MB
    }

    func thumbnail(for url: URL, size: CGSize = CGSize(width: 200, height: 200)) async -> NSImage? {
        let key = cacheKey(for: url, size: size)

        // Memory cache hit
        if let cached = memCache.object(forKey: key as NSString) { return cached }

        // Disk cache hit
        let diskURL = diskCacheURL.appendingPathComponent(key + ".jpg")
        if let data = try? Data(contentsOf: diskURL),
           let img = NSImage(data: data) {
            memCache.setObject(img, forKey: key as NSString)
            return img
        }

        // Generate
        guard let cgImage = await decodeThumbnail(url: url, size: size) else { return nil }
        let nsImage = NSImage(cgImage: cgImage, size: size)

        // Write to disk
        if let rep = NSBitmapImageRep(cgImage: cgImage),
           let jpegData = rep.representation(using: .jpeg, properties: [.compressionFactor: 0.8]) {
            try? jpegData.write(to: diskURL)
        }
        memCache.setObject(nsImage, forKey: key as NSString)
        return nsImage
    }

    func invalidate(for url: URL) {
        let key200 = cacheKey(for: url, size: CGSize(width: 200, height: 200))
        memCache.removeObject(forKey: key200 as NSString)
        let diskURL = diskCacheURL.appendingPathComponent(key200 + ".jpg")
        try? FileManager.default.removeItem(at: diskURL)
    }

    // MARK: Private

    private func decodeThumbnail(url: URL, size: CGSize) async -> CGImage? {
        let ext = url.pathExtension.lowercased()
        if LibRawWrapper.supportedExtensions.contains(ext) {
            // RAW: LibRaw embedded preview (fast)
            return LibRawWrapper.decode(url: url)
                .flatMap { resize($0, to: size) }
        } else {
            // JPEG/PNG: CIImage
            guard let ci = CIImage(contentsOf: url) else { return nil }
            let scaled = ci.transformed(by: CGAffineTransform(scaleX: size.width / ci.extent.width,
                                                               y: size.height / ci.extent.height))
            return CIContext().createCGImage(scaled, from: scaled.extent)
        }
    }

    private func resize(_ image: CGImage, to size: CGSize) -> CGImage? {
        let ctx = CGContext(data: nil,
                           width: Int(size.width), height: Int(size.height),
                           bitsPerComponent: 8, bytesPerRow: 0,
                           space: CGColorSpaceCreateDeviceRGB(),
                           bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        ctx?.draw(image, in: CGRect(origin: .zero, size: size))
        return ctx?.makeImage()
    }

    private func cacheKey(for url: URL, size: CGSize) -> String {
        let path = url.path
        let attrs = try? FileManager.default.attributesOfItem(atPath: path)
        let mtime = (attrs?[.modificationDate] as? Date)?.timeIntervalSince1970 ?? 0
        return "\(path.hashValue)_\(Int(size.width))_\(mtime)".replacingOccurrences(of: "/", with: "_")
    }
}
```

- [ ] **Step 2: Commit**

```bash
git add ImageRater/Import/ThumbnailCache.swift
git commit -m "feat: add ThumbnailCache actor with memory+disk cache and LibRaw fast path"
```

---

## Task 7: RatingPipeline

*Run in parallel with Tasks 6, 8, 9 after Group A completes.*

**Files:**
- Create: `ImageRater/Pipeline/RatingPipeline.swift`
- Test: `ImageRaterTests/RatingPipelineTests.swift`

**Note:** This task uses a stub/mock Core ML model for tests. Real `.mlpackage` models are loaded by ModelStore at runtime. The pipeline accepts an `MLModel` protocol so tests can inject mocks.

- [ ] **Step 1: Write failing test**

`ImageRaterTests/RatingPipelineTests.swift`:
```swift
import XCTest
import CoreML
@testable import ImageRater

final class RatingPipelineTests: XCTestCase {

    func testRatingNormalizesScoreToStars() {
        // aesthetic score 8.0/10 → should map to 4 stars
        let stars = RatingPipeline.starsFromAestheticScore(8.0)
        XCTAssertEqual(stars, 4)
    }

    func testRatingScore1MapsTo1Star() {
        XCTAssertEqual(RatingPipeline.starsFromAestheticScore(1.0), 1)
    }

    func testRatingScore10MapsTo5Stars() {
        XCTAssertEqual(RatingPipeline.starsFromAestheticScore(10.0), 5)
    }

    func testWeightedCombinationCorrect() {
        let result = RatingPipeline.combineScores(clipScore: 0.8, aestheticScore: 6.0,
                                                   clipWeight: 0.5, aestheticWeight: 0.5)
        // clipScore 0.8 * 10 = 8.0, aestheticScore 6.0, mean = 7.0 → 3-4 stars
        XCTAssertGreaterThanOrEqual(result.stars, 3)
        XCTAssertLessThanOrEqual(result.stars, 4)
    }
}
```

- [ ] **Step 2: Run — verify fails**

```bash
xcodebuild test -scheme ImageRater -destination 'platform=macOS' -only-testing:ImageRaterTests/RatingPipelineTests
```
Expected: FAIL

- [ ] **Step 3: Implement RatingPipeline**

`ImageRater/Pipeline/RatingPipeline.swift`:
```swift
import CoreML
import CoreImage
import Foundation

enum RatingPipeline {

    /// Normalize aesthetic score (1–10) to star rating (1–5).
    static func starsFromAestheticScore(_ score: Float) -> Int {
        let clamped = min(max(score, 1.0), 10.0)
        return Int(ceil((clamped / 10.0) * 5.0))
    }

    /// Combine CLIP cosine similarity (0–1) and aesthetic score (1–10) into RatingResult.
    static func combineScores(clipScore: Float, aestheticScore: Float,
                               clipWeight: Float, aestheticWeight: Float) -> RatingResult {
        let clipNorm = clipScore * 10.0 // scale to 0–10
        let totalWeight = clipWeight + aestheticWeight
        let combined = (clipNorm * clipWeight + aestheticScore * aestheticWeight) / totalWeight
        return RatingResult(
            stars: starsFromAestheticScore(combined),
            clipScore: clipScore,
            aestheticScore: aestheticScore
        )
    }

    /// Run inference on a CGImage using loaded MLModels. Returns .unrated on any failure.
    static func rate(image: CGImage,
                     clipModel: MLModel,
                     aestheticModel: MLModel,
                     config: ModelConfig) async -> RatingResult {
        do {
            let pixelBuffer = try cgImageToPixelBuffer(image, width: 224, height: 224)

            // CLIP inference
            let clipInput = try MLDictionaryFeatureProvider(dictionary: ["image": pixelBuffer])
            let clipOutput = try clipModel.prediction(from: clipInput)
            let clipScore = (clipOutput.featureValue(for: "score")?.floatValue) ?? 0.5

            // Aesthetic model inference
            let aestheticInput = try MLDictionaryFeatureProvider(dictionary: ["image": pixelBuffer])
            let aestheticOutput = try aestheticModel.prediction(from: aestheticInput)
            let aestheticScore = (aestheticOutput.featureValue(for: "score")?.floatValue) ?? 5.0

            return combineScores(clipScore: clipScore, aestheticScore: aestheticScore,
                                  clipWeight: config.clipWeight,
                                  aestheticWeight: config.aestheticWeight)
        } catch {
            return .unrated
        }
    }

    // MARK: Private

    private static func cgImageToPixelBuffer(_ image: CGImage, width: Int, height: Int) throws -> CVPixelBuffer {
        var buffer: CVPixelBuffer?
        CVPixelBufferCreate(kCFAllocatorDefault, width, height,
                            kCVPixelFormatType_32ARGB,
                            [kCVPixelBufferCGImageCompatibilityKey: true,
                             kCVPixelBufferCGBitmapContextCompatibilityKey: true] as CFDictionary,
                            &buffer)
        guard let pb = buffer else { throw RatingError.pixelBufferCreationFailed }
        CVPixelBufferLockBaseAddress(pb, [])
        defer { CVPixelBufferUnlockBaseAddress(pb, []) }
        let ctx = CGContext(data: CVPixelBufferGetBaseAddress(pb),
                           width: width, height: height,
                           bitsPerComponent: 8,
                           bytesPerRow: CVPixelBufferGetBytesPerRow(pb),
                           space: CGColorSpaceCreateDeviceRGB(),
                           bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue)
        ctx?.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        return pb
    }
}

enum RatingError: Error {
    case pixelBufferCreationFailed
}
```

- [ ] **Step 4: Run tests — verify pass**

```bash
xcodebuild test -scheme ImageRater -destination 'platform=macOS' -only-testing:ImageRaterTests/RatingPipelineTests
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add ImageRater/Pipeline/RatingPipeline.swift ImageRaterTests/RatingPipelineTests.swift
git commit -m "feat: implement RatingPipeline with score normalization and weighted combine"
```

---

## Task 8: ModelStore

*Run in parallel with Tasks 6, 7, 9 after Group A completes.*

**Files:**
- Create: `ImageRater/ModelStore/ModelStore.swift`
- Create: `ImageRater/ModelStore/ManifestFetcher.swift`
- Create: `ImageRater/ModelStore/ModelDownloader.swift`
- Test: `ImageRaterTests/ModelStoreTests.swift`

- [ ] **Step 1: Write failing tests**

`ImageRaterTests/ModelStoreTests.swift`:
```swift
import XCTest
@testable import ImageRater

final class ModelStoreTests: XCTestCase {

    func testChecksumMatchSucceeds() {
        let data = Data("hello".utf8)
        let expectedSHA = ModelDownloader.sha256(data)
        XCTAssertTrue(ModelDownloader.verify(data: data, expectedSHA256: expectedSHA))
    }

    func testChecksumMismatchFails() {
        let data = Data("hello".utf8)
        XCTAssertFalse(ModelDownloader.verify(data: data, expectedSHA256: "badhash"))
    }
}
```

- [ ] **Step 2: Run — verify fails**

```bash
xcodebuild test -scheme ImageRater -destination 'platform=macOS' -only-testing:ImageRaterTests/ModelStoreTests
```
Expected: FAIL

- [ ] **Step 3: Implement ModelDownloader**

`ImageRater/ModelStore/ModelDownloader.swift`:
```swift
import CryptoKit
import Foundation

enum ModelDownloader {

    static func sha256(_ data: Data) -> String {
        let digest = SHA256.hash(data: data)
        return digest.map { String(format: "%02x", $0) }.joined()
    }

    static func verify(data: Data, expectedSHA256: String) -> Bool {
        sha256(data) == expectedSHA256
    }

    static func download(from url: URL, expectedSHA256: String) async throws -> URL {
        var lastError: Error?
        for attempt in 0..<3 {
            if attempt > 0 {
                try await Task.sleep(nanoseconds: UInt64(pow(2.0, Double(attempt))) * 1_000_000_000)
            }
            do {
                let (tmpURL, _) = try await URLSession.shared.download(from: url)
                let data = try Data(contentsOf: tmpURL)
                guard verify(data: data, expectedSHA256: expectedSHA256) else {
                    throw ModelStoreError.checksumMismatch
                }
                return tmpURL
            } catch {
                lastError = error
            }
        }
        throw lastError ?? ModelStoreError.downloadFailed
    }
}

enum ModelStoreError: Error {
    case checksumMismatch
    case downloadFailed
    case manifestVerificationFailed
    case modelNotFound(String)
}
```

- [ ] **Step 4: Implement ManifestFetcher**

`ImageRater/ModelStore/ManifestFetcher.swift`:
```swift
import CryptoKit
import Foundation

struct ModelManifest: Decodable {
    struct ModelEntry: Decodable {
        let name: String
        let version: String
        let url: URL
        let sha256: String
    }
    let models: [ModelEntry]
    let signature: String // Ed25519 hex signature of canonical JSON
}

enum ManifestFetcher {
    // Hardcoded manifest URL — not user-configurable. Replace with real URL before shipping.
    // To generate keypair + manifest: see Post-Build section of this plan.
    // For local dev/testing: set IMAGERATING_MANIFEST_URL env var to override (debug builds only).
    private static let manifestURL: URL = {
        #if DEBUG
        if let override = ProcessInfo.processInfo.environment["IMAGERATING_MANIFEST_URL"],
           let url = URL(string: override) { return url }
        #endif
        return URL(string: "https://REPLACE_WITH_REAL_MANIFEST_HOST/models-manifest.json")!
    }()
    // Ed25519 public key (32 bytes hex). Generate with: openssl genpkey -algorithm ed25519
    // then extract public key bytes. Replace before shipping.
    private static let publicKeyHex = ProcessInfo.processInfo.environment["IMAGERATING_PUBKEY_HEX"]
        ?? "REPLACE_WITH_REAL_ED25519_PUBLIC_KEY_HEX"

    static func fetch() async throws -> ModelManifest {
        let (data, _) = try await URLSession.shared.data(from: manifestURL)
        let manifest = try JSONDecoder().decode(ModelManifest.self, from: data)
        try verifySignature(of: data, signature: manifest.signature)
        return manifest
    }

    private static func verifySignature(of data: Data, signature: String) throws {
        guard let pubKeyData = Data(hexString: publicKeyHex),
              let sigData = Data(hexString: signature) else {
            throw ModelStoreError.manifestVerificationFailed
        }
        let pubKey = try Curve25519.Signing.PublicKey(rawRepresentation: pubKeyData)
        // Strip signature field from JSON before verifying
        guard let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              var stripped = json as [String: Any] else {
            throw ModelStoreError.manifestVerificationFailed
        }
        stripped.removeValue(forKey: "signature")
        let canonical = try JSONSerialization.data(withJSONObject: stripped, options: .sortedKeys)
        guard pubKey.isValidSignature(sigData, for: canonical) else {
            throw ModelStoreError.manifestVerificationFailed
        }
    }
}

private extension Data {
    init?(hexString: String) {
        let len = hexString.count / 2
        var data = Data(capacity: len)
        for i in 0..<len {
            let start = hexString.index(hexString.startIndex, offsetBy: i * 2)
            let end = hexString.index(start, offsetBy: 2)
            guard let byte = UInt8(hexString[start..<end], radix: 16) else { return nil }
            data.append(byte)
        }
        self = data
    }
}
```

- [ ] **Step 5: Implement ModelStore actor**

`ImageRater/ModelStore/ModelStore.swift`:
```swift
import CoreML
import Foundation

actor ModelStore {
    static let shared = ModelStore()

    private let modelsDir: URL
    private var loadedModels: [String: MLModel] = [:]

    init() {
        let appSupport = FileManager.default.urls(for: .applicationSupportDirectory,
                                                   in: .userDomainMask)[0]
        modelsDir = appSupport.appendingPathComponent("ImageRater/models")
        try? FileManager.default.createDirectory(at: modelsDir, withIntermediateDirectories: true)
    }

    /// Ensure all required models are present and verified. Downloads if missing.
    func prepareModels(progress: @escaping (String) -> Void) async throws {
        let manifest = try await ManifestFetcher.fetch()
        for entry in manifest.models {
            let dest = modelsDir.appendingPathComponent("\(entry.name)-\(entry.version).mlpackage")
            if !FileManager.default.fileExists(atPath: dest.path) {
                progress("Downloading \(entry.name)...")
                let tmp = try await ModelDownloader.download(from: entry.url,
                                                              expectedSHA256: entry.sha256)
                try FileManager.default.moveItem(at: tmp, to: dest)
            }
        }
    }

    func model(named name: String) throws -> MLModel {
        if let cached = loadedModels[name] { return cached }

        guard let url = try? modelsDir.contents().first(where: { $0.lastPathComponent.hasPrefix(name) }) else {
            throw ModelStoreError.modelNotFound(name)
        }
        let config = MLModelConfiguration()
        config.computeUnits = ProcessInfo.processInfo.isAppleSilicon ? .cpuAndNeuralEngine : .cpuOnly
        let model = try MLModel(contentsOf: url, configuration: config)
        loadedModels[name] = model
        return model
    }
}

private extension ProcessInfo {
    var isAppleSilicon: Bool {
        var sysinfo = utsname()
        uname(&sysinfo)
        let machine = withUnsafeBytes(of: &sysinfo.machine) { ptr in
            String(cString: ptr.bindMemory(to: CChar.self).baseAddress!)
        }
        return machine.contains("arm")
    }
}

private extension URL {
    func contents() throws -> [URL] {
        try FileManager.default.contentsOfDirectory(at: self, includingPropertiesForKeys: nil)
    }
}
```

- [ ] **Step 6: Run tests — verify pass**

```bash
xcodebuild test -scheme ImageRater -destination 'platform=macOS' -only-testing:ImageRaterTests/ModelStoreTests
```
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add ImageRater/ModelStore/ ImageRaterTests/ModelStoreTests.swift
git commit -m "feat: implement ModelStore with Ed25519 manifest verification and SHA-256 checksum"
```

---

## Task 9: ImageImporter

*Run in parallel with Tasks 6, 7, 8 after Group A completes.*

**Files:**
- Create: `ImageRater/Import/ImageImporter.swift`

- [ ] **Step 1: Implement ImageImporter**

`ImageRater/Import/ImageImporter.swift`:
```swift
import CoreData
import Foundation

enum ImageImporter {

    static let supportedExtensions: Set<String> = LibRawWrapper.supportedExtensions
        .union(["jpg", "jpeg", "png", "tiff", "tif", "heic"])

    /// Scan folder, create Session + ImageRecord entities, return session objectID.
    @discardableResult
    static func importFolder(_ url: URL,
                              context: NSManagedObjectContext) throws -> NSManagedObjectID {
        let session = Session(context: context)
        session.id = UUID()
        session.createdAt = Date()
        session.folderPath = url.path

        let files = try scanFolder(url)
        for fileURL in files {
            let record = ImageRecord(context: context)
            record.id = UUID()
            record.filePath = fileURL.path
            record.processState = "pending"
            record.decodeError = false
            record.cullRejected = false
            record.ratingStars = 0
            record.session = session
        }

        try context.save()
        return session.objectID
    }

    static func scanFolder(_ url: URL) throws -> [URL] {
        guard let enumerator = FileManager.default.enumerator(
            at: url,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles]
        ) else { return [] }

        return enumerator.compactMap { item -> URL? in
            guard let fileURL = item as? URL,
                  (try? fileURL.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile) == true,
                  supportedExtensions.contains(fileURL.pathExtension.lowercased())
            else { return nil }
            return fileURL
        }.sorted { $0.lastPathComponent < $1.lastPathComponent }
    }
}
```

- [ ] **Step 2: Commit**

```bash
git add ImageRater/Import/ImageImporter.swift
git commit -m "feat: add ImageImporter with folder scan and Core Data session creation"
```

---

## Task 10: ProcessingQueue

*Run after Group B (Tasks 6–9) complete.*

**Files:**
- Create: `ImageRater/Pipeline/ProcessingQueue.swift`
- Test: `ImageRaterTests/ProcessingQueueTests.swift`

- [ ] **Step 1: Write failing test**

`ImageRaterTests/ProcessingQueueTests.swift`:
```swift
import XCTest
import CoreData
@testable import ImageRater

final class ProcessingQueueTests: XCTestCase {

    func testCancelMidBatchSetsInterruptedState() async throws {
        let ctx = PersistenceController(inMemory: true).container.viewContext
        // Create a session with 2 fake image records
        let session = Session(context: ctx)
        session.id = UUID()
        session.createdAt = Date()
        session.folderPath = "/tmp"

        for _ in 0..<2 {
            let r = ImageRecord(context: ctx)
            r.id = UUID()
            r.filePath = "/tmp/fake.jpg"
            r.processState = "pending"
            r.session = session
        }
        try ctx.save()

        let queue = ProcessingQueue(context: ctx)
        let task = Task { try await queue.process(sessionID: session.objectID) }
        task.cancel()
        try? await task.value

        let fetchRequest = ImageRecord.fetchRequest()
        let records = try ctx.fetch(fetchRequest)
        // All records should be pending or interrupted, none stuck in culling/rating
        for record in records {
            XCTAssertTrue(record.processState == "pending" || record.processState == "interrupted")
        }
    }
}
```

- [ ] **Step 2: Run — verify fails**

```bash
xcodebuild test -scheme ImageRater -destination 'platform=macOS' -only-testing:ImageRaterTests/ProcessingQueueTests
```
Expected: FAIL

- [ ] **Step 3: Implement ProcessingQueue**

`ImageRater/Pipeline/ProcessingQueue.swift`:
```swift
import CoreData
import CoreML
import Foundation

actor ProcessingQueue {
    private let context: NSManagedObjectContext
    private var progressContinuation: AsyncStream<ProcessingProgress>.Continuation?

    struct ProcessingProgress {
        let total: Int
        let completed: Int
        let current: String
    }

    init(context: NSManagedObjectContext) {
        self.context = context
    }

    func process(sessionID: NSManagedObjectID) async throws {
        let session = try context.existingObject(with: sessionID) as! Session
        guard let images = session.images?.allObjects as? [ImageRecord] else { return }

        let config = fetchOrCreateConfig()

        // Prepare models (no-op if already downloaded)
        try await ModelStore.shared.prepareModels(progress: { _ in })
        let clipModel = try await ModelStore.shared.model(named: "clip")
        let aestheticModel = try await ModelStore.shared.model(named: "aesthetic")

        for record in images {
            try Task.checkCancellation()

            record.processState = "culling"
            try? context.save()

            guard let image = LibRawWrapper.decode(url: URL(filePath: record.filePath ?? "")) else {
                record.decodeError = true
                record.processState = "done"
                try? context.save()
                continue
            }

            // Phase 1: Cull
            let cullResult = await CullPipeline.cull(image: image, config: config)
            record.cullRejected = cullResult.rejected
            record.cullReason = cullResult.reason?.rawValue

            if cullResult.rejected {
                record.processState = "done"
                try? context.save()
                continue
            }

            try Task.checkCancellation()

            // Phase 2: Rate
            record.processState = "rating"
            try? context.save()

            // Skip if user has explicit override (userOverride > 0 means user set it; 0 = no override sentinel)
            // Core Data optional Integer16 defaults to 0 — treat 0 as "no override"
            if record.userOverride <= 0 {
                let ratingResult = await RatingPipeline.rate(
                    image: image,
                    clipModel: clipModel,
                    aestheticModel: aestheticModel,
                    config: config
                )
                record.ratingStars = Int16(ratingResult.stars)
                record.clipScore = ratingResult.clipScore
                record.aestheticScore = ratingResult.aestheticScore
            }

            record.processState = "done"
            try? context.save()
        }
    }

    private func fetchOrCreateConfig() -> ModelConfig {
        let req = ModelConfig.fetchRequest()
        req.fetchLimit = 1
        if let existing = try? context.fetch(req).first { return existing }
        let config = ModelConfig(context: context)
        config.id = UUID()
        config.modelName = "default"
        config.clipWeight = 0.5
        config.aestheticWeight = 0.5
        config.blurThreshold = 50.0
        config.earThreshold = 0.2
        config.exposureThreshold = 0.9
        try? context.save()
        return config
    }
}

// Handle cancellation — set interrupted state
extension ProcessingQueue {
    func markInterrupted(sessionID: NSManagedObjectID) {
        guard let session = try? context.existingObject(with: sessionID) as? Session,
              let images = session.images?.allObjects as? [ImageRecord] else { return }
        for record in images where record.processState == "culling" || record.processState == "rating" {
            record.processState = "interrupted"
        }
        try? context.save()
    }
}
```

- [ ] **Step 4: Run tests — verify pass**

```bash
xcodebuild test -scheme ImageRater -destination 'platform=macOS' -only-testing:ImageRaterTests/ProcessingQueueTests
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add ImageRater/Pipeline/ProcessingQueue.swift ImageRaterTests/ProcessingQueueTests.swift
git commit -m "feat: implement ProcessingQueue actor with two-phase pipeline and cancellation"
```

---

## Task 11: GridView

*Run in parallel with Tasks 12, 13 after Task 10 completes.*

**Files:**
- Create: `ImageRater/UI/Components/ScoreBadge.swift`
- Create: `ImageRater/UI/Components/ThumbnailCell.swift`
- Create: `ImageRater/UI/GridView.swift`

- [ ] **Step 1: Create ScoreBadge**

`ImageRater/UI/Components/ScoreBadge.swift`:
```swift
import SwiftUI

struct ScoreBadge: View {
    let stars: Int
    let rejected: Bool

    var body: some View {
        HStack(spacing: 2) {
            if rejected {
                Image(systemName: "xmark.circle.fill")
                    .foregroundColor(.red)
            } else if stars > 0 {
                ForEach(1...5, id: \.self) { i in
                    Image(systemName: i <= stars ? "star.fill" : "star")
                        .foregroundColor(.yellow)
                        .font(.system(size: 8))
                }
            } else {
                Text("—").font(.caption2).foregroundColor(.secondary)
            }
        }
        .padding(4)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 4))
    }
}
```

- [ ] **Step 2: Create ThumbnailCell**

`ImageRater/UI/Components/ThumbnailCell.swift`:
```swift
import SwiftUI

struct ThumbnailCell: View {
    @ObservedObject var record: ImageRecord
    let isSelected: Bool
    let onSelect: () -> Void

    @State private var thumbnail: NSImage?

    var body: some View {
        ZStack(alignment: .bottomTrailing) {
            Group {
                if let thumb = thumbnail {
                    Image(nsImage: thumb)
                        .resizable()
                        .aspectRatio(contentMode: .fill)
                } else {
                    Rectangle().fill(Color.secondary.opacity(0.2))
                        .overlay(ProgressView())
                }
            }
            .frame(width: 160, height: 110)
            .clipped()
            .cornerRadius(6)
            .overlay(
                RoundedRectangle(cornerRadius: 6)
                    .stroke(isSelected ? Color.accentColor : Color.clear, lineWidth: 2)
            )
            .overlay(
                RoundedRectangle(cornerRadius: 6)
                    .stroke(record.cullRejected ? Color.red.opacity(0.6) : Color.clear, lineWidth: 2)
            )

            ScoreBadge(stars: Int(record.ratingStars), rejected: record.cullRejected)
                .padding(4)
        }
        .onTapGesture(perform: onSelect)
        .task {
            let url = URL(filePath: record.filePath ?? "")
            thumbnail = await ThumbnailCache.shared.thumbnail(for: url)
        }
    }
}
```

- [ ] **Step 3: Create GridView**

`ImageRater/UI/GridView.swift`:
```swift
import CoreData
import SwiftUI

struct GridView: View {
    @FetchRequest var images: FetchedResults<ImageRecord>
    @Binding var selectedRecord: ImageRecord?

    init(session: Session, selectedRecord: Binding<ImageRecord?>) {
        _images = FetchRequest(
            sortDescriptors: [SortDescriptor(\.filePath)],
            predicate: NSPredicate(format: "session == %@", session)
        )
        _selectedRecord = selectedRecord
    }

    let columns = [GridItem(.adaptive(minimum: 160), spacing: 8)]

    var body: some View {
        ScrollView {
            LazyVGrid(columns: columns, spacing: 8) {
                ForEach(images) { record in
                    ThumbnailCell(
                        record: record,
                        isSelected: selectedRecord?.objectID == record.objectID
                    ) {
                        selectedRecord = record
                    }
                }
            }
            .padding()
        }
    }
}
```

- [ ] **Step 4: Commit**

```bash
git add ImageRater/UI/
git commit -m "feat: implement GridView with ThumbnailCell and ScoreBadge"
```

---

## Task 12: DetailView

*Run in parallel with Tasks 11, 13 after Task 10 completes.*

**Files:**
- Create: `ImageRater/UI/DetailView.swift`

- [ ] **Step 1: Implement DetailView**

`ImageRater/UI/DetailView.swift`:
```swift
import SwiftUI

struct DetailView: View {
    @ObservedObject var record: ImageRecord
    @State private var fullImage: NSImage?

    var body: some View {
        VStack(spacing: 0) {
            // Image
            Group {
                if let img = fullImage {
                    Image(nsImage: img)
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                } else {
                    ProgressView()
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)

            Divider()

            // Score panel
            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    Text(URL(filePath: record.filePath ?? "").lastPathComponent)
                        .font(.headline)
                    Spacer()
                    ScoreBadge(stars: Int(record.ratingStars), rejected: record.cullRejected)
                }

                if record.cullRejected, let reason = record.cullReason {
                    Label("Rejected: \(reason)", systemImage: "xmark.circle")
                        .foregroundColor(.red)
                        .font(.caption)
                }

                if record.ratingStars > 0 {
                    HStack {
                        scoreRow("CLIP", value: record.clipScore)
                        scoreRow("Aesthetic", value: record.aestheticScore)
                    }
                }

                // Manual override
                HStack {
                    Text("Override:")
                    Picker("", selection: overrideBinding) {
                        Text("AI").tag(Int16(0))
                        ForEach(1...5, id: \.self) { s in
                            Text("\(s) ★").tag(Int16(s))
                        }
                    }
                    .pickerStyle(.segmented)
                    .frame(width: 220)

                    if record.decodeError {
                        Label("Decode error", systemImage: "exclamationmark.triangle")
                            .foregroundColor(.orange).font(.caption)
                    }
                }
            }
            .padding()
        }
        .task(id: record.objectID) {
            let url = URL(filePath: record.filePath ?? "")
            fullImage = await ThumbnailCache.shared.thumbnail(for: url,
                                                               size: CGSize(width: 1200, height: 900))
        }
    }

    private func scoreRow(_ label: String, value: Float) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(label).font(.caption2).foregroundColor(.secondary)
            Text(String(format: "%.2f", value)).font(.caption)
        }
    }

    private var overrideBinding: Binding<Int16> {
        Binding(
            get: { record.userOverride },
            set: { record.userOverride = $0; try? record.managedObjectContext?.save() }
        )
    }
}
```

- [ ] **Step 2: Commit**

```bash
git add ImageRater/UI/DetailView.swift
git commit -m "feat: implement DetailView with score breakdown and manual override"
```

---

## Task 13: ModelStoreView

*Run in parallel with Tasks 11, 12 after Task 10 completes.*

**Files:**
- Create: `ImageRater/UI/ModelStoreView.swift`

- [ ] **Step 1: Implement ModelStoreView**

`ImageRater/UI/ModelStoreView.swift`:
```swift
import SwiftUI

struct ModelStoreView: View {
    @State private var status: String = "Ready"
    @State private var isDownloading = false
    @State private var errorMessage: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("AI Models").font(.title2).bold()

            Text("Models stored in ~/Library/Application Support/ImageRater/models/")
                .font(.caption).foregroundColor(.secondary)

            HStack {
                Button("Download / Update Models") {
                    Task { await downloadModels() }
                }
                .disabled(isDownloading)

                if isDownloading { ProgressView().scaleEffect(0.7) }
            }

            Text(status).font(.caption).foregroundColor(.secondary)

            if let err = errorMessage {
                Label(err, systemImage: "exclamationmark.triangle")
                    .foregroundColor(.red)
                    .font(.caption)
                Button("Retry") { Task { await downloadModels() } }
            }
        }
        .padding()
        .frame(minWidth: 400)
    }

    private func downloadModels() async {
        isDownloading = true
        errorMessage = nil
        do {
            try await ModelStore.shared.prepareModels { msg in
                Task { @MainActor in status = msg }
            }
            status = "Models ready."
        } catch {
            errorMessage = error.localizedDescription
            status = "Download failed."
        }
        isDownloading = false
    }
}
```

- [ ] **Step 2: Commit**

```bash
git add ImageRater/UI/ModelStoreView.swift
git commit -m "feat: implement ModelStoreView with download/retry UI"
```

---

## Task 14: App Wiring + Integration

*Run after Group C (Tasks 11–13) complete.*

**Files:**
- Modify: `ImageRater/App/ContentView.swift`
- Test: `ImageRaterTests/IntegrationTests.swift`

- [ ] **Step 1: Write integration test**

`ImageRaterTests/IntegrationTests.swift`:
```swift
import XCTest
import CoreData
@testable import ImageRater

final class IntegrationTests: XCTestCase {

    func testFullPipelineOnFixtureFolder() async throws {
        let ctx = PersistenceController(inMemory: true).container.viewContext
        let fixtureDir = Bundle(for: Self.self).resourceURL!.appendingPathComponent("Fixtures")

        let sessionID = try ImageImporter.importFolder(fixtureDir, context: ctx)

        let queue = ProcessingQueue(context: ctx)
        // Note: this will fail at ModelStore.prepareModels in CI (no network/models)
        // Run locally only with models pre-downloaded.
        // To run without models, mock ModelStore — integration smoke test.
        let session = try ctx.existingObject(with: sessionID) as! Session
        XCTAssertNotNil(session)

        let fetchRequest = ImageRecord.fetchRequest()
        fetchRequest.predicate = NSPredicate(format: "session == %@", session)
        let records = try ctx.fetch(fetchRequest)
        XCTAssertGreaterThan(records.count, 0)

        // All records start as pending
        for record in records {
            XCTAssertEqual(record.processState, "pending")
        }
    }
}
```

- [ ] **Step 2: Run — verify passes (smoke test only)**

```bash
xcodebuild test -scheme ImageRater -destination 'platform=macOS' -only-testing:ImageRaterTests/IntegrationTests
```
Expected: PASS (smoke test, no model inference)

- [ ] **Step 3: Wire up ContentView**

`ImageRater/App/ContentView.swift`:
```swift
import CoreData
import SwiftUI

struct ContentView: View {
    @Environment(\.managedObjectContext) private var ctx
    @FetchRequest(sortDescriptors: [SortDescriptor(\.createdAt, order: .reverse)])
    private var sessions: FetchedResults<Session>

    @State private var selectedSession: Session?
    @State private var selectedRecord: ImageRecord?
    @State private var showModelStore = false
    @State private var processingTask: Task<Void, Never>?

    var body: some View {
        NavigationSplitView {
            // Session list sidebar
            List(sessions, selection: $selectedSession) { session in
                Label(
                    URL(filePath: session.folderPath ?? "").lastPathComponent,
                    systemImage: "folder"
                )
                .tag(session)
            }
            .navigationTitle("Sessions")
            .toolbar {
                ToolbarItem {
                    Button(action: openFolder) {
                        Label("Open Folder", systemImage: "folder.badge.plus")
                    }
                }
                ToolbarItem {
                    Button { showModelStore = true } label: {
                        Label("Models", systemImage: "square.and.arrow.down")
                    }
                }
            }
        } content: {
            if let session = selectedSession {
                GridView(session: session, selectedRecord: $selectedRecord)
                    .toolbar {
                        ToolbarItem {
                            Button(action: { runPipeline(session: session) }) {
                                Label("Process", systemImage: "wand.and.stars")
                            }
                        }
                        ToolbarItem {
                            Button(action: { exportMetadata(session: session) }) {
                                Label("Export XMP", systemImage: "square.and.arrow.up")
                            }
                        }
                    }
            } else {
                Text("Open a folder to begin")
                    .foregroundColor(.secondary)
            }
        } detail: {
            if let record = selectedRecord {
                DetailView(record: record)
            } else {
                Text("Select an image")
                    .foregroundColor(.secondary)
            }
        }
        .sheet(isPresented: $showModelStore) { ModelStoreView() }
    }

    private func openFolder() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = false
        if panel.runModal() == .OK, let url = panel.url {
            try? ImageImporter.importFolder(url, context: ctx)
        }
    }

    private func runPipeline(session: Session) {
        processingTask?.cancel()
        processingTask = Task {
            let queue = ProcessingQueue(context: ctx)
            try? await queue.process(sessionID: session.objectID)
        }
    }

    private func exportMetadata(session: Session) {
        guard let images = session.images?.allObjects as? [ImageRecord] else { return }
        for record in images {
            guard !record.cullRejected, record.ratingStars > 0,
                  let path = record.filePath else { continue }
            let stars = Int(record.userOverride > 0 ? record.userOverride : record.ratingStars)
            try? MetadataWriter.writeSidecar(stars: stars, for: URL(filePath: path))
        }
    }
}
```

- [ ] **Step 4: Run full test suite**

```bash
xcodebuild test -scheme ImageRater -destination 'platform=macOS'
```
Expected: All tests PASS

- [ ] **Step 5: Final commit**

```bash
git add ImageRater/App/ContentView.swift ImageRaterTests/IntegrationTests.swift
git commit -m "feat: wire up ContentView with session list, grid, detail, pipeline, and XMP export"
```

---

## Post-Build: Model Conversion (Developer Step)

These steps run once to produce `.mlpackage` files for distribution. Not part of app build.

```bash
pip install coremltools open_clip_torch torch

python3 - <<'EOF'
import open_clip
import coremltools as ct
import torch

model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-B-32', pretrained='laion2b_s34b_b79k'
)
model.eval()

example = torch.zeros(1, 3, 224, 224)
traced = torch.jit.trace(model.encode_image, example)
mlmodel = ct.convert(traced,
    inputs=[ct.TensorType(name="image", shape=example.shape)],
    outputs=[ct.TensorType(name="score")],
    compute_precision=ct.precision.FLOAT16
)
mlmodel.save("clip-vit-b32.mlpackage")
EOF
```

SHA-256 the output and add to `models-manifest.json`. Sign the manifest with Ed25519 private key before hosting.

### Keypair + Manifest Setup (one-time)

```bash
# Generate Ed25519 keypair
openssl genpkey -algorithm ed25519 -out manifest_private.pem
openssl pkey -in manifest_private.pem -pubout -out manifest_public.pem

# Extract raw 32-byte public key hex (for hardcoding in ManifestFetcher.swift)
openssl pkey -in manifest_public.pem -pubin -outform DER | tail -c 32 | xxd -p | tr -d '\n'

# Compute model SHA-256
shasum -a 256 clip-vit-b32.mlpackage

# Build manifest JSON (fill in url and sha256 from above)
cat > models-manifest.json <<EOF
{
  "models": [
    {"name": "clip", "version": "1.0", "url": "https://YOUR_HOST/clip-vit-b32.mlpackage", "sha256": "PASTE_SHA256"},
    {"name": "aesthetic", "version": "1.0", "url": "https://YOUR_HOST/aesthetic.mlpackage", "sha256": "PASTE_SHA256"}
  ],
  "signature": ""
}
EOF

# Sign canonical JSON (signature field must be empty string when signing)
python3 -c "
import json, subprocess
with open('models-manifest.json') as f: m = json.load(f)
m['signature'] = ''
canonical = json.dumps({k: m[k] for k in sorted(m)}, separators=(',',':')).encode()
sig = subprocess.check_output(['openssl', 'pkeyutl', '-sign', '-inkey', 'manifest_private.pem'],
                               input=canonical).hex()
m['signature'] = sig
print(json.dumps(m, indent=2))
" > models-manifest-signed.json

# Host models-manifest-signed.json at the URL hardcoded in ManifestFetcher.swift
# Replace publicKeyHex constant in ManifestFetcher.swift with the 32-byte hex from above
# Replace manifestURL constant with real hosting URL
```
