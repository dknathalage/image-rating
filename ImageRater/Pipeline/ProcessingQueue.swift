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
        // Fetch session data and config on the context's queue
        let (imageIDs, configSnapshot) = try await context.perform { [self] in
            guard let session = try? self.context.existingObject(with: sessionID) as? Session,
                  let images = session.images?.allObjects as? [ImageRecord] else {
                throw CocoaError(.coreData)
            }
            let snapshot = self.fetchOrCreateConfigSync()
            let sorted = images.sorted { ($0.filePath ?? "") < ($1.filePath ?? "") }
            return (sorted.map(\.objectID), snapshot)
        }

        let total = imageIDs.count
        var completed = 0
        log.info("Session processing started: \(total) images")
        onProgress?(0, total, "Loading models…")

        // Attempt manifest sync; ignore network failures if models are already present locally.
        // model(named:) will throw clearly if a required model is missing.
        try? await ModelStore.shared.prepareModels(progress: { _ in })
        try Task.checkCancellation()
        onProgress?(0, total, "Compiling models…")
        log.info("Loading NIMA aesthetic model")
        let nimaAestheticModel = try await ModelStore.shared.model(named: "nima-aesthetic")
        log.info("Loading NIMA technical model")
        let nimaTechnicalModel = try await ModelStore.shared.model(named: "nima-technical")
        log.info("Models ready")
        try Task.checkCancellation()

        var ratedCount = 0
        var decodeErrorCount = 0

        do {
            for imageID in imageIDs {
                try Task.checkCancellation()

                // Skip already-done records (re-run safety)
                let alreadyDone = await context.perform { [self] in
                    (try? self.context.existingObject(with: imageID) as? ImageRecord)?.processState == ProcessState.done
                }
                if alreadyDone {
                    log.debug("Skipping already-done image \(completed + 1)")
                    completed += 1
                    continue
                }

                // Read filePath on context queue
                let filePath = await context.perform { [self] in
                    (try? self.context.existingObject(with: imageID) as? ImageRecord)?.filePath
                }

                // Decode image (heavy I/O — outside context perform)
                let filename = filePath.map { URL(filePath: $0).lastPathComponent } ?? "unknown"
                log.info("[\(completed + 1)/\(total)] Decoding \(filename)")
                guard let path = filePath,
                      let image = LibRawWrapper.decode(url: URL(filePath: path)) else {
                    decodeErrorCount += 1
                    log.error("[\(completed + 1)/\(total)] Decode failed: \(filename)")
                    await context.perform { [self] in
                        guard let record = try? self.context.existingObject(with: imageID) as? ImageRecord else { return }
                        record.decodeError = true
                        record.processState = ProcessState.done
                        try? self.context.save()
                    }
                    completed += 1
                    onProgress?(completed, total, "Decode error (\(decodeErrorCount) so far) — skipping")
                    continue
                }
                log.info("[\(completed + 1)/\(total)] Decoded \(filename): \(image.width)×\(image.height)")

                // Rate (skip if user has override)
                let hasOverride = await context.perform { [self] in
                    (try? self.context.existingObject(with: imageID) as? ImageRecord)?.userOverride != nil
                }

                if !hasOverride {
                    ratedCount += 1
                    log.info("[\(completed + 1)/\(total)] Running models on \(filename)")
                    onProgress?(completed, total, "Running models: \(completed + 1) of \(total) (rated \(ratedCount))")
                    await context.perform { [self] in
                        guard let record = try? self.context.existingObject(with: imageID) as? ImageRecord else { return }
                        record.processState = ProcessState.rating
                        try? self.context.save()
                    }

                    let ratingResult = await RatingPipeline.rate(
                        image: image,
                        nimaAestheticModel: nimaAestheticModel,
                        nimaTechnicalModel: nimaTechnicalModel
                    )

                    await context.perform { [self] in
                        guard let record = try? self.context.existingObject(with: imageID) as? ImageRecord else { return }
                        record.ratingStars     = NSNumber(value: ratingResult.stars)
                        record.aestheticScore  = NSNumber(value: ratingResult.aestheticScore)
                        record.clipScore       = NSNumber(value: ratingResult.technicalScore)  // reuse existing CoreData field
                        record.processState    = ProcessState.done
                        try? self.context.save()
                    }
                    if let path = filePath {
                        try? MetadataWriter.writeSidecar(stars: ratingResult.stars, for: URL(filePath: path))
                    }
                } else {
                    let overrideStars = await context.perform { [self] in
                        (try? self.context.existingObject(with: imageID) as? ImageRecord)?.userOverride?.intValue ?? 0
                    }
                    await context.perform { [self] in
                        guard let record = try? self.context.existingObject(with: imageID) as? ImageRecord else { return }
                        record.processState = ProcessState.done
                        try? self.context.save()
                    }
                    if let path = filePath, overrideStars > 0 {
                        try? MetadataWriter.writeSidecar(stars: overrideStars, for: URL(filePath: path))
                    }
                }

                completed += 1
                let summary = "rated \(ratedCount)" + (decodeErrorCount > 0 ? ", \(decodeErrorCount) errors" : "")
                if completed == total {
                    log.info("Session complete — \(summary)")
                }
                onProgress?(completed, total, completed == total ? "Done — \(summary)" : "Processing \(completed + 1) of \(total) (\(summary))")
            }
        } catch is CancellationError {
            await markInterrupted(sessionID: sessionID)
            throw CancellationError()
        }

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
