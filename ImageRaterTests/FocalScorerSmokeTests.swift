import XCTest
@testable import Focal

final class FocalScorerSmokeTests: XCTestCase {
    func testRatingPipelineProducesScoresForFixture() async throws {
        let bundle = Bundle(for: type(of: self))
        guard let fixtureURL = bundle.url(forResource: "sharp", withExtension: "jpg") else {
            throw XCTSkip("fixture sharp.jpg missing from test bundle")
        }
        guard let cg = LibRawWrapper.decode(url: fixtureURL) else {
            return XCTFail("decode failed")
        }
        let models = try RatingPipeline.loadBundledModels()
        let r = await RatingPipeline.rate(image: cg, models: models)
        if case .rated(let s) = r {
            XCTAssertGreaterThan(s.musiqAesthetic, 0)
            XCTAssertTrue((1...5).contains(s.stars))
        } else {
            XCTFail("expected .rated")
        }
    }
}
