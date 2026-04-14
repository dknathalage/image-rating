// ImageRaterTests/ImageRecordMigrationTests.swift
import XCTest
import CoreData
@testable import ImageRater

final class ImageRecordMigrationTests: XCTestCase {

    func testNewFieldsReadWrite() throws {
        let ctx = makeInMemoryContext()

        let session = Session(context: ctx)
        session.id = UUID(); session.createdAt = Date(); session.folderPath = "/tmp"

        let r = ImageRecord(context: ctx)
        r.id = UUID(); r.filePath = "/tmp/a.jpg"; r.processState = "pending"; r.session = session

        r.topiqTechnicalScore = 0.75
        r.topiqAestheticScore = 0.82
        r.clipIQAScore        = 0.68
        r.combinedQualityScore = 0.77
        r.finalScore          = 0.65
        r.diversityFactor     = 0.85
        r.clipEmbedding       = Data(repeating: 1, count: 512 * 4)
        r.clusterID           = 3
        r.clusterRank         = 2
        r.blurScore           = 420.5
        r.exposureScore       = 0.3

        try ctx.save()

        ctx.refresh(r, mergeChanges: false)
        XCTAssertEqual(r.topiqTechnicalScore,  0.75,  accuracy: 0.001)
        XCTAssertEqual(r.topiqAestheticScore,  0.82,  accuracy: 0.001)
        XCTAssertEqual(r.clipIQAScore,         0.68,  accuracy: 0.001)
        XCTAssertEqual(r.combinedQualityScore, 0.77,  accuracy: 0.001)
        XCTAssertEqual(r.finalScore,           0.65,  accuracy: 0.001)
        XCTAssertEqual(r.diversityFactor,      0.85,  accuracy: 0.001)
        XCTAssertEqual(r.clipEmbedding?.count, 512 * 4)
        XCTAssertEqual(r.clusterID,   3)
        XCTAssertEqual(r.clusterRank, 2)
        XCTAssertEqual(r.blurScore,     420.5, accuracy: 0.1)
        XCTAssertEqual(r.exposureScore, 0.3,   accuracy: 0.001)
    }

    func testV3GroupingFieldsReadWrite() throws {
        let ctx = makeInMemoryContext()

        let session = Session(context: ctx)
        session.id = UUID(); session.createdAt = Date(); session.folderPath = "/tmp"

        let jpeg = ImageRecord(context: ctx)
        jpeg.id = UUID(); jpeg.filePath = "/tmp/a.jpg"; jpeg.processState = "pending"; jpeg.session = session
        jpeg.groupID = "test-group-1"
        jpeg.isGroupPrimary = true
        jpeg.scoringFilePath = "/tmp/a.cr2"

        let raw = ImageRecord(context: ctx)
        raw.id = UUID(); raw.filePath = "/tmp/a.cr2"; raw.processState = "pending"; raw.session = session
        raw.groupID = "test-group-1"
        raw.isGroupPrimary = false

        let solo = ImageRecord(context: ctx)
        solo.id = UUID(); solo.filePath = "/tmp/b.jpg"; solo.processState = "pending"; solo.session = session
        // groupID nil, isGroupPrimary defaults to true

        try ctx.save()

        ctx.refresh(jpeg, mergeChanges: false)
        ctx.refresh(raw, mergeChanges: false)
        ctx.refresh(solo, mergeChanges: false)

        XCTAssertEqual(jpeg.groupID, "test-group-1")
        XCTAssertTrue(jpeg.isGroupPrimary)
        XCTAssertEqual(jpeg.scoringFilePath, "/tmp/a.cr2")

        XCTAssertEqual(raw.groupID, "test-group-1")
        XCTAssertFalse(raw.isGroupPrimary)
        XCTAssertNil(raw.scoringFilePath)

        XCTAssertNil(solo.groupID)
        XCTAssertTrue(solo.isGroupPrimary)
        XCTAssertNil(solo.scoringFilePath)
    }

    private func makeInMemoryContext() -> NSManagedObjectContext {
        let c = NSPersistentContainer(name: "ImageRater")
        let d = NSPersistentStoreDescription()
        d.type = NSInMemoryStoreType
        c.persistentStoreDescriptions = [d]
        c.loadPersistentStores { _, error in
            if let error { XCTFail("Failed to load in-memory store: \(error)"); return }
        }
        return c.viewContext
    }
}
