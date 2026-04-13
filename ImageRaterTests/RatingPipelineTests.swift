import XCTest
@testable import ImageRater

final class RatingPipelineTests: XCTestCase {

    // MARK: - absoluteStars thresholds

    func testAbsoluteStars_below3_8_is1Star() {
        XCTAssertEqual(RatingPipeline.absoluteStars(combined: 3.7), 1)
        XCTAssertEqual(RatingPipeline.absoluteStars(combined: 1.0), 1)
    }

    func testAbsoluteStars_3_8to4_2_is2Stars() {
        XCTAssertEqual(RatingPipeline.absoluteStars(combined: 3.8), 2)
        XCTAssertEqual(RatingPipeline.absoluteStars(combined: 4.1), 2)
    }

    func testAbsoluteStars_4_2to4_6_is3Stars() {
        XCTAssertEqual(RatingPipeline.absoluteStars(combined: 4.2), 3)
        XCTAssertEqual(RatingPipeline.absoluteStars(combined: 4.5), 3)
    }

    func testAbsoluteStars_4_6to5_0_is4Stars() {
        XCTAssertEqual(RatingPipeline.absoluteStars(combined: 4.6), 4)
        XCTAssertEqual(RatingPipeline.absoluteStars(combined: 4.9), 4)
    }

    func testAbsoluteStars_5_0plus_is5Stars() {
        XCTAssertEqual(RatingPipeline.absoluteStars(combined: 5.0), 5)
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
