import XCTest
@testable import Focal

final class FocalScorerSmokeTests: XCTestCase {
    func testMUSIQModelLoads() throws {
        _ = try RatingPipeline.loadBundledModels()
    }

    func testRatingPipelineProducesScoresForSolidImage() async throws {
        let models = try RatingPipeline.loadBundledModels()
        let cg = TestImage.solid(256, 256)
        let r = await RatingPipeline.rate(image: cg, models: models)
        guard case .rated(let s) = r else {
            return XCTFail("expected .rated, got \(r)")
        }
        XCTAssertTrue(s.musiqAesthetic.isFinite)
        XCTAssertTrue((1.0...10.0).contains(s.musiqAesthetic))
        XCTAssertTrue((1...5).contains(s.stars))
    }
}
