import CoreData
import Foundation
import OSLog

private let log = Logger(subsystem: "com.imagerating", category: "ProcessingQueue")

/// Process states — typed constants to avoid raw string bugs.
enum ProcessState {
    static let pending = "pending"
    static let culling = "culling"
    static let rating = "rating"
    static let done = "done"
    static let interrupted = "interrupted"
}

actor ProcessingQueue {

    private let context: NSManagedObjectContext

    /// Pass a private-queue context (container.newBackgroundContext()) for strict thread safety.
    /// Using viewContext is acceptable when process() is called from the main actor.
    init(context: NSManagedObjectContext) {
        self.context = context
    }

    /// Run both pipeline phases on all pending images in a session.
    /// Respects Swift Task cancellation — sets interrupted state on cancel.
    /// `onProgress` receives (completed, total, statusMessage) on every step.
    func process(
        sessionID: NSManagedObjectID,
        onProgress: (@Sendable (Int, Int, String) -> Void)? = nil
    ) async throws {
        fatalError("Replaced in Task 5 — do not call until ProcessingQueue is updated")
    }

    // MARK: Private

    private struct ConfigSnapshot {
        let blurThreshold: Float
        let earThreshold: Float
        let exposureLeniency: Float
    }

    /// Must be called within context.perform. Creates or migrates config to current defaults.
    private func fetchOrCreateConfigSync() -> ConfigSnapshot {
        let req = ModelConfig.fetchRequest()
        req.fetchLimit = 1
        let config: ModelConfig
        if let existing = (try? context.fetch(req))?.first {
            config = existing
            // Migrate stale defaults — old CIEdgeWork threshold was 5000, new CIEdges scale is ~500
            if config.blurThreshold >= 1000 {
                log.info("Migrating stale blurThreshold \(config.blurThreshold) → 500")
                config.blurThreshold = 500
            }
            if config.earThreshold > 0.2 {
                log.info("Migrating stale earThreshold \(config.earThreshold) → 0.15")
                config.earThreshold = 0.15
            }
            try? context.save()
        } else {
            config = ModelConfig(context: context)
            config.id = UUID()
            config.modelName = "default"
            config.blurThreshold = 500       // CIEdges variance on 512px; ~870 real photos, <200 truly blurry
            config.earThreshold = 0.15
            config.exposureLeniency = 0.95
            try? context.save()
        }
        return ConfigSnapshot(
            blurThreshold: config.blurThreshold,
            earThreshold: config.earThreshold,
            exposureLeniency: config.exposureLeniency
        )
    }
}

// MARK: - External API

extension ProcessingQueue {
    /// Set any in-progress records to interrupted. Call on app background or task cancel.
    func markInterrupted(sessionID: NSManagedObjectID) async {
        await context.perform { [self] in
            guard let session = try? self.context.existingObject(with: sessionID) as? Session,
                  let images = session.images?.allObjects as? [ImageRecord] else { return }
            for record in images
            where record.processState == ProcessState.culling || record.processState == ProcessState.rating {
                record.processState = ProcessState.interrupted
            }
            try? self.context.save()
        }
    }
}
