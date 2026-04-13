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

    // MARK: - Unzip tests

    func testUnzipExtractsMLPackage() throws {
        // Create a minimal fake .mlpackage directory
        let tmp = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let fakePackage = tmp.appendingPathComponent("test.mlpackage")
        try FileManager.default.createDirectory(at: fakePackage, withIntermediateDirectories: true)
        // Add a sentinel file inside the package
        try "hello".write(to: fakePackage.appendingPathComponent("Manifest.json"),
                          atomically: true, encoding: .utf8)

        // Zip it (path-stripped: zip from inside tmp so archive root is test.mlpackage)
        let zipURL = tmp.appendingPathComponent("test.zip")
        let zipper = Process()
        zipper.executableURL = URL(filePath: "/usr/bin/zip")
        zipper.currentDirectoryURL = tmp
        zipper.arguments = ["-rq", zipURL.path, "test.mlpackage"]
        try zipper.run(); zipper.waitUntilExit()
        XCTAssertEqual(zipper.terminationStatus, 0)

        // Unzip and verify
        let outputDir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let pkgURL = try ModelDownloader.unzip(zipURL, to: outputDir)
        XCTAssertEqual(pkgURL.pathExtension, "mlpackage")
        XCTAssertTrue(FileManager.default.fileExists(atPath: pkgURL.path))

        // Cleanup
        try? FileManager.default.removeItem(at: tmp)
        try? FileManager.default.removeItem(at: outputDir)
    }

    func testUnzipThrowsOnNonZip() throws {
        let badFile = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".zip")
        try "not a zip".write(to: badFile, atomically: true, encoding: .utf8)
        let outputDir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {
            try? FileManager.default.removeItem(at: badFile)
            try? FileManager.default.removeItem(at: outputDir)
        }
        XCTAssertThrowsError(try ModelDownloader.unzip(badFile, to: outputDir))
    }

    func testUnzipThrowsWhenNoMLPackageInZip() throws {
        // Create zip with a non-.mlpackage file
        let tmp = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: tmp, withIntermediateDirectories: true)
        let textFile = tmp.appendingPathComponent("readme.txt")
        try "hello".write(to: textFile, atomically: true, encoding: .utf8)
        let zipURL = tmp.appendingPathComponent("test.zip")
        let zipper = Process()
        zipper.executableURL = URL(filePath: "/usr/bin/zip")
        zipper.currentDirectoryURL = tmp
        zipper.arguments = ["-q", zipURL.path, "readme.txt"]
        try zipper.run(); zipper.waitUntilExit()

        let outputDir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {
            try? FileManager.default.removeItem(at: tmp)
            try? FileManager.default.removeItem(at: outputDir)
        }
        XCTAssertThrowsError(try ModelDownloader.unzip(zipURL, to: outputDir))
    }
}
