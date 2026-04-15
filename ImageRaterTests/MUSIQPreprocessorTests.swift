// ImageRaterTests/MUSIQPreprocessorTests.swift
import XCTest
@testable import Focal

final class MUSIQPreprocessorTests: XCTestCase {

    private func fixturesURL() -> URL {
        // Resources from ImageRaterTests/Fixtures/ are copied flat into
        // the xctest bundle's Resources directory by Xcode.
        Bundle(for: MUSIQPreprocessorTests.self).resourceURL!
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
        // Fixture writes [224, 149] (rh first): H=1200 is longer side → rh=224
        XCTAssertEqual(dims, [224, 149])

        let (resized, rh, rw) = MUSIQPreprocessor.aspectResize(
            pixels: src, h: 1200, w: 800, channels: 3, longerSide: 224
        )
        XCTAssertEqual(rh, 224)
        XCTAssertEqual(rw, 149)
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
        // Fixture writes [256, 384] (rh first): W=1200 is longer side → rw=384
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

    func test_hashSpatialPositions_7x5_matches_pyiqa() {
        let expected = loadTensor("hsp_7x5")  // [1, 35]
        let out = MUSIQPreprocessor.hashSpatialPositions(countH: 7, countW: 5, gridSize: 10)
        XCTAssertEqual(out.count, 35)
        for (a, b) in zip(out, expected) {
            XCTAssertEqual(a, b, accuracy: 0.0)  // integer indices — exact
        }
    }

    func test_patchTensor_500x400_matches_pyiqa_reference() throws {
        let src = loadTensor("img_500x400")
        let expected = loadTensor("patch_tensor_500x400")

        let tensor = try MUSIQPreprocessor.patchTensor(
            pixels: src, h: 500, w: 400, channels: 3
        )
        XCTAssertEqual(tensor.shape.map { $0.intValue }, [1, 193, 3075])
        XCTAssertEqual(tensor.dataType, .float32)

        let ptr = tensor.dataPointer.bindMemory(to: Float.self, capacity: tensor.count)
        var maxDelta: Float = 0
        for i in 0..<expected.count {
            maxDelta = max(maxDelta, abs(ptr[i] - expected[i]))
        }
        XCTAssertLessThan(maxDelta, 1e-2, "Max |Δ| = \(maxDelta)")
    }

    func test_patchTensor_image_too_small_throws() {
        let src = [Float](repeating: 0.5, count: 3 * 20 * 30)
        XCTAssertThrowsError(
            try MUSIQPreprocessor.patchTensor(
                pixels: src, h: 20, w: 30, channels: 3
            )
        ) { err in
            guard case RatingError.imageTooSmall = err else {
                return XCTFail("Expected .imageTooSmall, got \(err)")
            }
        }
    }
}
