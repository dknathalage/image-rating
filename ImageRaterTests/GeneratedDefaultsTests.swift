import XCTest
@testable import Focal

final class GeneratedDefaultsTests: XCTestCase {
    func test_thresholds_monotonic_and_in_expected_range() {
        let t1 = FocalSettings.defaultMUSIQThreshold1
        let t2 = FocalSettings.defaultMUSIQThreshold2
        let t3 = FocalSettings.defaultMUSIQThreshold3
        let t4 = FocalSettings.defaultMUSIQThreshold4
        XCTAssertLessThan(t1, t2)
        XCTAssertLessThan(t2, t3)
        XCTAssertLessThan(t3, t4)
        XCTAssertTrue((1.0...10.0).contains(Double(t1)), "t1 out of MUSIQ range")
        XCTAssertTrue((1.0...10.0).contains(Double(t4)), "t4 out of MUSIQ range")
    }
    func test_version_matches_params_current_json() {
        XCTAssertEqual(FocalSettings.generatedVersion, "v0.4.0")
    }
}
