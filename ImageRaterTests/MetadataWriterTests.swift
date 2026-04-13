import XCTest
@testable import ImageRater

final class MetadataWriterTests: XCTestCase {

    func testWriteAndReadBackXMPRating() throws {
        let tmpURL = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString + ".xmp")
        defer { try? FileManager.default.removeItem(at: tmpURL) }

        try MetadataWriter.write(stars: 4, to: tmpURL)
        let readBack = try MetadataWriter.readRating(from: tmpURL)
        XCTAssertEqual(readBack, 4)
    }

    func testWriteZeroStarsProducesUnrated() throws {
        let tmpURL = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString + ".xmp")
        defer { try? FileManager.default.removeItem(at: tmpURL) }

        try MetadataWriter.write(stars: 0, to: tmpURL)
        let readBack = try MetadataWriter.readRating(from: tmpURL)
        XCTAssertEqual(readBack, 0)
    }

    func testWriteAllStarValues() throws {
        for stars in 1...5 {
            let tmpURL = FileManager.default.temporaryDirectory
                .appendingPathComponent(UUID().uuidString + ".xmp")
            defer { try? FileManager.default.removeItem(at: tmpURL) }
            try MetadataWriter.write(stars: stars, to: tmpURL)
            let readBack = try MetadataWriter.readRating(from: tmpURL)
            XCTAssertEqual(readBack, stars, "Failed for \(stars) stars")
        }
    }

    func testXMPFileCreatedAtExpectedSidecarPath() throws {
        let tmpDir = FileManager.default.temporaryDirectory
        let imageURL = tmpDir.appendingPathComponent("test_img_\(UUID().uuidString).jpg")
        let expectedXMP = imageURL.deletingPathExtension().appendingPathExtension("xmp")
        defer { try? FileManager.default.removeItem(at: expectedXMP) }

        try MetadataWriter.writeSidecar(stars: 3, for: imageURL)
        XCTAssertTrue(FileManager.default.fileExists(atPath: expectedXMP.path))
    }

    func testWriteOutOfRangeStarsThrows() {
        let tmpURL = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString + ".xmp")
        XCTAssertThrowsError(try MetadataWriter.write(stars: 6, to: tmpURL))
        XCTAssertThrowsError(try MetadataWriter.write(stars: -1, to: tmpURL))
    }
}
