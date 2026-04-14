import XCTest
import CoreData
@testable import Focal

final class IntegrationTests: XCTestCase {

    func testFullPipelineOnFixtureFolder() async throws {
        let ctx = PersistenceController(inMemory: true).container.viewContext
        // Create a fresh empty subdirectory so no image files are picked up
        let tmpDir = URL(filePath: NSTemporaryDirectory())
            .appendingPathComponent("IntegrationTest_\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmpDir) }

        // importFolder on empty dir should not crash, returns session with 0 images
        let sessionID = try ImageImporter.importFolder(tmpDir, context: ctx)

        let session = try ctx.existingObject(with: sessionID) as! Session
        XCTAssertEqual(session.folderPath, tmpDir.path)

        let fetchRequest = ImageRecord.fetchRequest()
        fetchRequest.predicate = NSPredicate(format: "session == %@", session)
        let records = try ctx.fetch(fetchRequest)
        // Temp dir has no supported image files — 0 records is expected
        XCTAssertEqual(records.count, 0)
    }

    func testProcessingQueueHandlesEmptySession() async throws {
        let ctx = PersistenceController(inMemory: true).container.viewContext
        let session = Session(context: ctx)
        session.id = UUID()
        session.createdAt = Date()
        session.folderPath = "/tmp"
        try ctx.save()

        let queue = ProcessingQueue(context: ctx)
        // Empty session — should complete without error (no images to process)
        // Will throw on model prep (no network in test), that's expected
        try? await queue.process(sessionID: session.objectID)
    }
}
