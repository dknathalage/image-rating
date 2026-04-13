import CoreData
import Foundation

actor ProcessingQueue {

    private let context: NSManagedObjectContext

    init(context: NSManagedObjectContext) {
        self.context = context
    }

    /// Run both pipeline phases on all pending images in a session.
    /// Respects Task cancellation — sets interrupted state on cancel.
    func process(sessionID: NSManagedObjectID) async throws {
        guard let session = try? context.existingObject(with: sessionID) as? Session else { return }
        guard let images = session.images?.allObjects as? [ImageRecord] else { return }

        let config = fetchOrCreateConfig()

        // Ensure models ready before processing starts (never mid-batch)
        try await ModelStore.shared.prepareModels(progress: { _ in })

        let clipModel = try await ModelStore.shared.model(named: "clip")
        let aestheticModel = try await ModelStore.shared.model(named: "aesthetic")

        for record in images {
            // Check cancellation at each image boundary
            if Task.isCancelled {
                markInterrupted(images: images)
                return
            }

            // Skip if already done (re-run safety)
            if record.processState == "done" { continue }

            record.processState = "culling"
            try? context.save()

            // Decode image
            guard let filePath = record.filePath,
                  let image = LibRawWrapper.decode(url: URL(filePath: filePath)) else {
                record.decodeError = true
                record.processState = "done"
                try? context.save()
                continue
            }

            // Phase 1: Cull
            let cullResult = await CullPipeline.cull(
                image: image,
                blurThreshold: config.blurThreshold,
                earThreshold: config.earThreshold,
                exposureLeniency: config.exposureThreshold
            )
            record.cullRejected = cullResult.rejected
            record.cullReason = cullResult.reason?.rawValue

            if cullResult.rejected {
                record.processState = "done"
                try? context.save()
                continue
            }

            if Task.isCancelled {
                markInterrupted(images: images)
                return
            }

            // Phase 2: Rate (skip if user has override)
            record.processState = "rating"
            try? context.save()

            let hasOverride = (record.userOverride?.intValue ?? 0) > 0
            if !hasOverride {
                let ratingResult = await RatingPipeline.rate(
                    image: image,
                    clipModel: clipModel,
                    aestheticModel: aestheticModel,
                    clipWeight: config.clipWeight,
                    aestheticWeight: config.aestheticWeight
                )
                record.ratingStars = NSNumber(value: ratingResult.stars)
                record.clipScore = NSNumber(value: ratingResult.clipScore)
                record.aestheticScore = NSNumber(value: ratingResult.aestheticScore)
            }

            record.processState = "done"
            try? context.save()
        }
    }

    // MARK: Private

    private func fetchOrCreateConfig() -> ModelConfig {
        let req = ModelConfig.fetchRequest()
        req.fetchLimit = 1
        if let existing = (try? context.fetch(req))?.first { return existing }
        let config = ModelConfig(context: context)
        config.id = UUID()
        config.modelName = "default"
        config.blurThreshold = 5000.0   // calibrated: CIEdgeWork variance ~6800 sharp, ~2900 blurry
        config.earThreshold = 0.2
        config.exposureThreshold = 0.9  // leniency: rejects if >10% pixels are extreme
        config.clipWeight = 0.5
        config.aestheticWeight = 0.5
        try? context.save()
        return config
    }

    private func markInterrupted(images: [ImageRecord]) {
        for record in images where record.processState == "culling" || record.processState == "rating" {
            record.processState = "interrupted"
        }
        try? context.save()
    }
}
