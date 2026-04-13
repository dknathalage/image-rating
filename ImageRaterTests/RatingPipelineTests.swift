import XCTest
@testable import ImageRater

final class RatingPipelineTests: XCTestCase {

    func testStarsFromScore1Returns1() {
        XCTAssertEqual(RatingPipeline.starsFromAestheticScore(1.0), 1)
    }

    func testStarsFromScore10Returns5() {
        XCTAssertEqual(RatingPipeline.starsFromAestheticScore(10.0), 5)
    }

    func testStarsFromScore8Returns4() {
        XCTAssertEqual(RatingPipeline.starsFromAestheticScore(8.0), 4)
    }

    func testStarsFromScoreClampsBelow1() {
        XCTAssertEqual(RatingPipeline.starsFromAestheticScore(0.0), 1)
    }

    func testStarsFromScoreClampsAbove10() {
        XCTAssertEqual(RatingPipeline.starsFromAestheticScore(11.0), 5)
    }

    func testCombineScoresEqualWeights() {
        // clipScore 0.8 → clipNorm 8.0; aestheticScore 6.0; weighted avg 7.0 → 4 stars
        let result = RatingPipeline.combineScores(clipScore: 0.8, aestheticScore: 6.0,
                                                   clipWeight: 0.5, aestheticWeight: 0.5)
        XCTAssertEqual(result.clipScore, 0.8)
        XCTAssertEqual(result.aestheticScore, 6.0)
        XCTAssertEqual(result.stars, 4)
    }

    func testCombineScoresZeroWeightReturnsUnrated() {
        let result = RatingPipeline.combineScores(clipScore: 0.8, aestheticScore: 6.0,
                                                   clipWeight: 0.0, aestheticWeight: 0.0)
        XCTAssertEqual(result, .unrated)
    }

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
