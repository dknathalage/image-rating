import XCTest
@testable import Focal

final class LibRawWrapperTests: XCTestCase {

    func testDecodeJPEGReturnsCGImage() throws {
        // Use a known JPEG fixture from the test bundle
        guard let url = Bundle(for: Self.self).url(forResource: "sharp", withExtension: "jpg") else {
            XCTFail("sharp.jpg fixture not found — add it to ImageRaterTests/Fixtures/")
            return
        }
        let result = LibRawWrapper.decode(url: url)
        XCTAssertNotNil(result)
    }

    func testDecodeBadPathReturnsNil() {
        let url = URL(filePath: "/tmp/nonexistent_\(UUID().uuidString).cr3")
        let result = LibRawWrapper.decode(url: url)
        XCTAssertNil(result)
    }

    func testDecodeNonFileURLReturnsNil() {
        let url = URL(string: "https://example.com/photo.jpg")!
        let result = LibRawWrapper.decode(url: url)
        XCTAssertNil(result)
    }

    func testSupportedExtensionsContainsCommonRAWFormats() {
        XCTAssertTrue(LibRawWrapper.supportedExtensions.contains("cr3"))
        XCTAssertTrue(LibRawWrapper.supportedExtensions.contains("nef"))
        XCTAssertTrue(LibRawWrapper.supportedExtensions.contains("arw"))
        XCTAssertTrue(LibRawWrapper.supportedExtensions.contains("raf"))
    }
}
