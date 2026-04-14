import XCTest
@testable import Focal

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

    func testClipIQAGoodPhotoEmbeddingScoresAboveHalf() {
        // Feeding the "Good photo" text embedding to clipIQAScore should strongly prefer "Good photo"
        let score = RatingPipeline.clipIQAScore(embedding: CLIPTextEmbeddings.goodPhoto, logitScale: Float(FocalSettings.resolvedClipLogitScale()))
        XCTAssertGreaterThan(score, 0.5, "Good photo embedding should score above 0.5")
    }

    func testClipIQABadPhotoEmbeddingScoresBelowHalf() {
        // Feeding the "Bad photo" text embedding should prefer "Bad photo" → low score
        let score = RatingPipeline.clipIQAScore(embedding: CLIPTextEmbeddings.badPhoto, logitScale: Float(FocalSettings.resolvedClipLogitScale()))
        XCTAssertLessThan(score, 0.5, "Bad photo embedding should score below 0.5")
    }

    func testClipIQAScoreIsBoundedZeroToOne() {
        // Softmax output must always be in [0,1]
        let score = RatingPipeline.clipIQAScore(embedding: CLIPTextEmbeddings.goodPhoto, logitScale: Float(FocalSettings.resolvedClipLogitScale()))
        XCTAssertGreaterThanOrEqual(score, 0.0)
        XCTAssertLessThanOrEqual(score, 1.0)
    }

    // MARK: - FocalSettings bucket edges + logit scale

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
