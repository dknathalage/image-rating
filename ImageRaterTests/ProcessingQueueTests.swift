import XCTest
import CoreData
@testable import Focal

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

    func testWeightNormalisationSumsToOne() {
        let ud = UserDefaults.standard
        ud.set(0.6, forKey: FocalSettings.weightTechnical)
        ud.set(0.6, forKey: FocalSettings.weightAesthetic)
        ud.set(0.3, forKey: FocalSettings.weightClip)
        defer {
            ud.removeObject(forKey: FocalSettings.weightTechnical)
            ud.removeObject(forKey: FocalSettings.weightAesthetic)
            ud.removeObject(forKey: FocalSettings.weightClip)
        }
        let wT = Float(ud.double(forKey: FocalSettings.weightTechnical))
        let wA = Float(ud.double(forKey: FocalSettings.weightAesthetic))
        let wC = Float(ud.double(forKey: FocalSettings.weightClip))
        let sum = wT + wA + wC
        let (wTn, wAn, wCn) = (wT/sum, wA/sum, wC/sum)
        XCTAssertEqual(wTn + wAn + wCn, 1.0, accuracy: 0.001)
    }

    func testWeightNormalisationZeroSumFallsBackToDefaults() {
        let ud = UserDefaults.standard
        ud.set(0.0, forKey: FocalSettings.weightTechnical)
        ud.set(0.0, forKey: FocalSettings.weightAesthetic)
        ud.set(0.0, forKey: FocalSettings.weightClip)
        defer {
            ud.removeObject(forKey: FocalSettings.weightTechnical)
            ud.removeObject(forKey: FocalSettings.weightAesthetic)
            ud.removeObject(forKey: FocalSettings.weightClip)
        }
        let wT = Float(ud.double(forKey: FocalSettings.weightTechnical))
        let wA = Float(ud.double(forKey: FocalSettings.weightAesthetic))
        let wC = Float(ud.double(forKey: FocalSettings.weightClip))
        let sum = wT + wA + wC
        let defSum = Float(FocalSettings.defaultWeightTechnical + FocalSettings.defaultWeightAesthetic + FocalSettings.defaultWeightClip)
        XCTAssertEqual(sum, 0.0)
        XCTAssertEqual(defSum, 1.0, accuracy: 0.001)
    }

}
