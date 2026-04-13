import XCTest
@testable import ImageRater

final class ModelStoreTests: XCTestCase {

    func testSHA256ConsistentForSameData() {
        let data = Data("hello world".utf8)
        let hash1 = ModelDownloader.sha256(data)
        let hash2 = ModelDownloader.sha256(data)
        XCTAssertEqual(hash1, hash2)
    }

    func testSHA256DifferentForDifferentData() {
        let a = ModelDownloader.sha256(Data("hello".utf8))
        let b = ModelDownloader.sha256(Data("world".utf8))
        XCTAssertNotEqual(a, b)
    }

    func testVerifyMatchingChecksumSucceeds() {
        let data = Data("test payload".utf8)
        let hash = ModelDownloader.sha256(data)
        XCTAssertTrue(ModelDownloader.verify(data: data, expectedSHA256: hash))
    }

    func testVerifyMismatchedChecksumFails() {
        let data = Data("test payload".utf8)
        XCTAssertFalse(ModelDownloader.verify(data: data, expectedSHA256: "badhash"))
    }

    func testVerifyEmptyDataHasKnownHash() {
        // SHA-256 of empty data is well-known
        let hash = ModelDownloader.sha256(Data())
        XCTAssertEqual(hash, "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")
    }
}
