import XCTest
@testable import Focal

final class RatingPipelineTests: XCTestCase {

    // MARK: - bucketStars

    private let t: (Float, Float, Float, Float) = (4.465, 5.181, 5.634, 6.068)

    func test_bucketStars_belowFirstThreshold() {
        XCTAssertEqual(RatingPipeline.bucketStars(mos: 3.0, thresholds: t), 1)
        XCTAssertEqual(RatingPipeline.bucketStars(mos: 4.464, thresholds: t), 1)
    }
    func test_bucketStars_boundaries() {
        XCTAssertEqual(RatingPipeline.bucketStars(mos: 4.465, thresholds: t), 1)
        XCTAssertEqual(RatingPipeline.bucketStars(mos: 4.466, thresholds: t), 2)
        XCTAssertEqual(RatingPipeline.bucketStars(mos: 5.181, thresholds: t), 2)
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
