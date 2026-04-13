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
