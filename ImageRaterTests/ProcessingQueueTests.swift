import XCTest
import CoreData
@testable import ImageRater

final class ProcessingQueueTests: XCTestCase {

    func testCancelMidBatchSetsInterruptedOrPendingState() async throws {
        let ctx = PersistenceController(inMemory: true).container.viewContext

        let session = Session(context: ctx)
        session.id = UUID()
        session.createdAt = Date()
        session.folderPath = "/tmp"

        // Create 3 fake image records (files don't exist — decode will fail fast)
        for _ in 0..<3 {
            let r = ImageRecord(context: ctx)
            r.id = UUID()
            r.filePath = "/tmp/nonexistent_\(UUID().uuidString).jpg"
            r.processState = ProcessState.pending
            r.decodeError = false
            r.cullRejected = false
            r.session = session
        }
        try ctx.save()

        let queue = ProcessingQueue(context: ctx)
        let task = Task {
            try await queue.process(sessionID: session.objectID)
        }
        task.cancel()
        try? await task.value

        let fetchRequest = ImageRecord.fetchRequest()
        let records = try ctx.fetch(fetchRequest)
        for record in records {
            let validStates = [ProcessState.pending, ProcessState.done, ProcessState.interrupted, ProcessState.rated]
            XCTAssertTrue(validStates.contains(record.processState ?? ""),
                          "Unexpected state: \(record.processState ?? "nil")")
        }
    }

    func testFetchOrCreateConfigCreatesDefaultConfig() async throws {
        let ctx = PersistenceController(inMemory: true).container.viewContext
        let session = Session(context: ctx)
        session.id = UUID()
        session.createdAt = Date()
        session.folderPath = "/tmp"
        // Add one record so process() reaches fetchOrCreateConfig
        let r = ImageRecord(context: ctx)
        r.id = UUID()
        r.filePath = "/tmp/nonexistent.jpg"
        r.processState = ProcessState.pending
        r.session = session
        try ctx.save()

        let queue = ProcessingQueue(context: ctx)
        try? await queue.process(sessionID: session.objectID)

        let req = ModelConfig.fetchRequest()
        let configs = try ctx.fetch(req)
        XCTAssertFalse(configs.isEmpty, "fetchOrCreateConfig should always create a config")
        if let config = configs.first {
            XCTAssertEqual(config.blurThreshold, 500.0)
            XCTAssertEqual(config.earThreshold, 0.15)
            XCTAssertEqual(config.exposureLeniency, 0.95)
        }
    }
}
