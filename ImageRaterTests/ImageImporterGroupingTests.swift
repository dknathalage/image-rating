import XCTest
import CoreData
@testable import ImageRater

final class ImageImporterGroupingTests: XCTestCase {

    // MARK: groupByBaseName

    func testSingleJpegAndRawPaired() {
        let urls = makeURLs(["IMG_001.jpg", "IMG_001.cr2"])
        let groups = ImageImporter.groupByBaseName(urls)
        XCTAssertEqual(groups.count, 1)
        XCTAssertEqual(groups[0].count, 2)
    }

    func testJpegWithMultipleRawsPaired() {
        let urls = makeURLs(["IMG_001.jpg", "IMG_001.cr2", "IMG_001.dng"])
        let groups = ImageImporter.groupByBaseName(urls)
        XCTAssertEqual(groups.count, 1)
        XCTAssertEqual(groups[0].count, 3)
    }

    func testOrphanRawIsOwnGroup() {
        let urls = makeURLs(["IMG_001.cr2"])
        let groups = ImageImporter.groupByBaseName(urls)
        XCTAssertEqual(groups.count, 1)
        XCTAssertEqual(groups[0].count, 1)
    }

    func testJpegOnlyIsOwnGroup() {
        let urls = makeURLs(["IMG_001.jpg"])
        let groups = ImageImporter.groupByBaseName(urls)
        XCTAssertEqual(groups.count, 1)
        XCTAssertEqual(groups[0].count, 1)
    }

    func testMultipleRawsNoJpegAllSingles() {
        let urls = makeURLs(["IMG_001.cr2", "IMG_001.dng"])
        let groups = ImageImporter.groupByBaseName(urls)
        XCTAssertEqual(groups.count, 2)
        XCTAssertTrue(groups.allSatisfy { $0.count == 1 })
    }

    func testMixedPairedAndSingles() {
        let urls = makeURLs(["A.jpg", "A.cr2", "B.jpg", "C.nef"])
        let groups = ImageImporter.groupByBaseName(urls)
        // A: paired (2), B: single (1), C: single (1)
        XCTAssertEqual(groups.count, 3)
        let groupSizes = groups.map(\.count).sorted()
        XCTAssertEqual(groupSizes, [1, 1, 2])
    }

    // MARK: importFolder pairing fields

    func testPrimaryRecordFieldsForPair() throws {
        let ctx = makeInMemoryContext()
        let tmpDir = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmpDir) }

        let jpegURL = tmpDir.appendingPathComponent("IMG_001.jpg")
        let rawURL  = tmpDir.appendingPathComponent("IMG_001.cr2")
        FileManager.default.createFile(atPath: jpegURL.path, contents: Data())
        FileManager.default.createFile(atPath: rawURL.path,  contents: Data())

        try ImageImporter.importFolder(tmpDir, context: ctx)

        let records = try ctx.fetch(ImageRecord.fetchRequest())
        XCTAssertEqual(records.count, 2)

        let primary   = records.first { $0.filePath?.hasSuffix(".jpg") == true }
        let companion = records.first { $0.filePath?.hasSuffix(".cr2") == true }

        XCTAssertNotNil(primary)
        XCTAssertNotNil(companion)
        XCTAssertTrue(primary!.isGroupPrimary)
        XCTAssertFalse(companion!.isGroupPrimary)
        XCTAssertNotNil(primary!.groupID)
        XCTAssertEqual(primary!.groupID, companion!.groupID)
        let expectedRawPath = try rawURL.resourceValues(forKeys: [.canonicalPathKey]).canonicalPath
        XCTAssertEqual(primary!.scoringFilePath, expectedRawPath)
        XCTAssertNil(companion!.scoringFilePath)
    }

    func testOrphanRawRecordIsUnpaired() throws {
        let ctx = makeInMemoryContext()
        let tmpDir = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmpDir) }

        let rawURL = tmpDir.appendingPathComponent("IMG_001.cr2")
        FileManager.default.createFile(atPath: rawURL.path, contents: Data())

        try ImageImporter.importFolder(tmpDir, context: ctx)

        let records = try ctx.fetch(ImageRecord.fetchRequest())
        XCTAssertEqual(records.count, 1)
        XCTAssertTrue(records[0].isGroupPrimary)
        XCTAssertNil(records[0].groupID)
        XCTAssertNil(records[0].scoringFilePath)
    }

    // MARK: Helpers

    private func makeURLs(_ names: [String]) -> [URL] {
        let base = URL(filePath: "/fake")
        return names.map { base.appendingPathComponent($0) }
            .sorted { $0.lastPathComponent.localizedStandardCompare($1.lastPathComponent) == .orderedAscending }
    }

    private func makeInMemoryContext() -> NSManagedObjectContext {
        let c = NSPersistentContainer(name: "ImageRater")
        let d = NSPersistentStoreDescription()
        d.type = NSInMemoryStoreType
        c.persistentStoreDescriptions = [d]
        c.loadPersistentStores { _, error in
            if let error { XCTFail("Store load failed: \(error)") }
        }
        return c.viewContext
    }
}
