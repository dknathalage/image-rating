import XCTest
@testable import ImageRater

final class RatingPipelineTests: XCTestCase {

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

    func testPixelBufferCreation384x384() throws {
        let img = makeSolidColorCGImage(size: 512)
        let buffer = try RatingPipeline.cgImageToPixelBuffer(img, width: 384, height: 384)
        XCTAssertEqual(CVPixelBufferGetWidth(buffer), 384)
        XCTAssertEqual(CVPixelBufferGetHeight(buffer), 384)
    }

    // MARK: - Combined quality weighting

    func testCombinedQualityWeighting() {
        let score = RatingPipeline.combinedQuality(
            technical: 0.8, aesthetic: 0.6, semantic: 0.5,
            weights: (technical: 0.4, aesthetic: 0.4, semantic: 0.2))
        XCTAssertEqual(score, 0.4*0.8 + 0.4*0.6 + 0.2*0.5, accuracy: 0.001)
    }

    // MARK: - CLIP-IQA+ score

    func testClipIQAPlusScoreIsBetweenZeroAndOne() {
        // Any L2-normalised embedding must produce a score in [0,1]
        var emb = [Float](repeating: 0, count: 512)
        emb[0] = 1.0   // unit vector
        let score = RatingPipeline.clipIQAScore(embedding: emb)
        XCTAssertGreaterThanOrEqual(score, 0.0)
        XCTAssertLessThanOrEqual(score, 1.0)
    }

    // MARK: - Helpers

    private func makeSolidColorCGImage(size: Int) -> CGImage {
        let bmi = CGBitmapInfo.byteOrder32Little.rawValue | CGImageAlphaInfo.noneSkipFirst.rawValue
        let ctx = CGContext(data: nil, width: size, height: size, bitsPerComponent: 8,
                            bytesPerRow: 4 * size, space: CGColorSpaceCreateDeviceRGB(),
                            bitmapInfo: bmi)!
        ctx.setFillColor(CGColor(gray: 0.5, alpha: 1))
        ctx.fill(CGRect(x: 0, y: 0, width: size, height: size))
        return ctx.makeImage()!
    }
}
