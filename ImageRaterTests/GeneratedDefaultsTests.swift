import XCTest
@testable import Focal

final class GeneratedDefaultsTests: XCTestCase {
    func testGeneratedWeightsMatchParamsJSON() throws {
        let url = URL(fileURLWithPath: #file)
            .deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("testing/bench/params.current.json")
        let data = try Data(contentsOf: url)
        let json = try JSONSerialization.jsonObject(with: data) as! [String: Any]
        let p = json["params"] as! [String: Any]

        XCTAssertEqual(FocalSettings.generatedWeightTechnical, p["w_tech"] as! Double, accuracy: 1e-9)
        XCTAssertEqual(FocalSettings.generatedWeightAesthetic, p["w_aes"]  as! Double, accuracy: 1e-9)
        XCTAssertEqual(FocalSettings.generatedWeightClip,      p["w_clip"] as! Double, accuracy: 1e-9)
    }
}
