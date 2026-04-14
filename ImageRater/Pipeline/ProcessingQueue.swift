import CoreData
import Foundation
import OSLog

private let log = Logger(subsystem: "com.imagerating", category: "ProcessingQueue")

/// Process states — typed constants to avoid raw string bugs.
enum ProcessState {
    static let pending = "pending"
    static let culling = "culling"
    static let rating = "rating"
    static let rated = "rated"   // pass 1 complete; diversity scoring pending
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

        let (imageIDs, configSnapshot) = try await context.perform { [self] in
            guard let session = try? self.context.existingObject(with: sessionID) as? Session,
                  let images = session.images?.allObjects as? [ImageRecord] else {
                throw CocoaError(.coreData)
            }
            let snapshot = self.fetchOrCreateConfigSync()
            let sorted = images.sorted { ($0.filePath ?? "") < ($1.filePath ?? "") }
            return (sorted.map(\.objectID), snapshot)
        }

        // Load bundled models once before the per-image loop
        onProgress?(0, 0, "Compiling models…")
        let models = try RatingPipeline.loadBundledModels()

        let total = imageIDs.count
        var decodeErrorCount = 0
        log.info("Session processing started: \(total) images")

        // ─── PASS 1: per-image inference ─────────────────────────────────────

        do {
            for (i, imageID) in imageIDs.enumerated() {
                try Task.checkCancellation()

                let (filePath, skip): (String?, Bool) = await context.perform { [self] in
                    guard let r = try? self.context.existingObject(with: imageID) as? ImageRecord else {
                        return (nil, true)
                    }
                    let skip = r.processState == ProcessState.done
                            || r.processState == ProcessState.rated
                    return (r.filePath, skip)
                }
                if skip { continue }

                onProgress?(i, total, "Scoring \(i + 1) of \(total)")

                guard let path = filePath,
                      let cgImage = LibRawWrapper.decode(url: URL(filePath: path)) else {
                    decodeErrorCount += 1
                    await context.perform { [self] in
                        guard let r = try? self.context.existingObject(with: imageID) as? ImageRecord else { return }
                        r.decodeError = true
                        r.processState = ProcessState.done
                        try? self.context.save()
                    }
                    continue
                }

                // Check user override — if set, skip AI rating but still run cull
                let hasOverride = await context.perform { [self] in
                    (try? self.context.existingObject(with: imageID) as? ImageRecord)?.userOverride != nil
                }

                async let cullTask   = CullPipeline.cull(
                    image: cgImage,
                    blurThreshold:    configSnapshot.blurThreshold,
                    earThreshold:     configSnapshot.earThreshold,
                    exposureLeniency: configSnapshot.exposureLeniency)
                async let ratingTask = rateIfNeeded(image: cgImage, models: models, hasOverride: hasOverride)
                let (cullScores, ratingResult) = await (cullTask, ratingTask)

                await context.perform { [self] in
                    guard let r = try? self.context.existingObject(with: imageID) as? ImageRecord else { return }
                    // Cull fields
                    r.cullRejected  = cullScores.result.rejected
                    r.cullReason    = cullScores.result.reason?.rawValue
                    r.blurScore     = cullScores.blurScore
                    r.exposureScore = cullScores.exposureScore

                    // Rating fields (only if no user override)
                    if !hasOverride, case .rated(let scores) = ratingResult {
                        r.topiqTechnicalScore  = scores.topiqTechnicalScore
                        r.topiqAestheticScore  = scores.topiqAestheticScore
                        r.clipIQAScore         = scores.clipIQAScore
                        r.combinedQualityScore = scores.combinedQualityScore
                        // Store 512-dim embedding as raw bytes
                        r.clipEmbedding = scores.clipEmbedding.withUnsafeBufferPointer {
                            Data(buffer: $0)
                        }
                    }
                    r.processState = ProcessState.rated
                    try? self.context.save()
                }
            }
        } catch is CancellationError {
            await markInterrupted(sessionID: sessionID)
            throw CancellationError()
        }

        try Task.checkCancellation()

        // ─── PASS 2: session-level diversity + normalization ──────────────────

        onProgress?(total, total, "Ranking variety…")
        try await runDiversityPass(sessionID: sessionID)

        // Write XMP sidecars now that final star ratings are assigned
        await writeSidecars(imageIDs: imageIDs, sessionID: sessionID)

        onProgress?(total, total, "Done")
        log.info("Session complete — \(total) images, \(decodeErrorCount) decode errors")
    }

    // MARK: Private

    private func runDiversityPass(sessionID: NSManagedObjectID) async throws {
        // Load embeddings and quality scores for all rated (or done) images.
        // hasEmbedding[i] is true only when clipEmbedding was written (i.e., rating succeeded).
        // Overridden images and decode-error images have no embedding — exclude them from
        // clustering/MMR to avoid zero-vector cosine similarity corruption.
        let (imageIDs, embeddings, qualityScores, hasEmbedding):
            ([NSManagedObjectID], [[Float]], [Float], [Bool]) =
            try await context.perform { [self] in
                guard let session = try? self.context.existingObject(with: sessionID) as? Session,
                      let images = session.images?.allObjects as? [ImageRecord] else {
                    throw CocoaError(.coreData)
                }
                let sorted = images
                    .filter { $0.processState == ProcessState.rated || $0.processState == ProcessState.done }
                    .sorted { ($0.filePath ?? "") < ($1.filePath ?? "") }

                let ids = sorted.map { $0.objectID }
                let hasEmb = sorted.map { $0.clipEmbedding != nil }
                let embs: [[Float]] = sorted.map { r in
                    guard let data = r.clipEmbedding else { return [] }
                    let count = data.count / MemoryLayout<Float>.size
                    return data.withUnsafeBytes { Array($0.bindMemory(to: Float.self).prefix(count)) }
                }
                let scores = sorted.map { $0.combinedQualityScore }
                return (ids, embs, scores, hasEmb)
            }

        guard !imageIDs.isEmpty else { return }

        // Only images with valid 512-dim embeddings participate in clustering and MMR.
        // Others receive diversityFactor=1.0 (no penalty) and clusterID=-1.
        let validIndices = hasEmbedding.indices.filter { hasEmbedding[$0] }
        let validEmbeddings  = validIndices.map { embeddings[$0] }
        let validQuality     = validIndices.map { qualityScores[$0] }

        // Per-image diversity results (defaults for images without embeddings)
        var clusterIDs      = [Int32](repeating: -1,  count: imageIDs.count)
        var clusterRanks    = [Int](repeating: 1,     count: imageIDs.count)
        var diversityFactors = [Float](repeating: 1.0, count: imageIDs.count)

        if !validEmbeddings.isEmpty {
            // Step A: threshold clustering → clusterID
            let validClusterIDs = DiversityScorer.clusterByThreshold(
                embeddings: validEmbeddings, threshold: 0.92)
            for (vIdx, iIdx) in validIndices.enumerated() {
                clusterIDs[iIdx] = validClusterIDs[vIdx]
            }

            // Step B: MMR ordering → clusterRank + diversityFactor
            let mmrItems = DiversityScorer.mmrOrder(
                embeddings: validEmbeddings, qualityScores: validQuality, lambda: 0.6)
            for item in mmrItems {
                let iIdx = validIndices[item.originalIndex]
                clusterRanks[iIdx]    = item.clusterRank
                diversityFactors[iIdx] = item.diversityFactor
            }
        }

        // Compute finalScore for ALL images (including those without embeddings)
        let finalScores = qualityScores.indices.map { qualityScores[$0] * diversityFactors[$0] }

        // Percentile normalisation → star ratings
        let starRatings = DiversityScorer.percentileToStars(finalScores: finalScores)

        // Write all diversity fields
        try await context.perform { [self] in
            for (i, imageID) in imageIDs.enumerated() {
                guard let r = try? self.context.existingObject(with: imageID) as? ImageRecord else { continue }
                r.clusterID       = clusterIDs[i]
                r.clusterRank     = Int32(clusterRanks[i])
                r.diversityFactor = diversityFactors[i]
                r.finalScore      = finalScores[i]
                r.ratingStars     = NSNumber(value: starRatings[i])
                r.processState    = ProcessState.done
            }
            try self.context.save()
        }
    }

    private func rateIfNeeded(
        image: CGImage,
        models: RatingPipeline.BundledModels,
        hasOverride: Bool
    ) async -> RatingResult {
        guard !hasOverride else { return .unrated }
        return await RatingPipeline.rate(image: image, models: models)
    }

    private func writeSidecars(imageIDs: [NSManagedObjectID], sessionID: NSManagedObjectID) async {
        await context.perform { [self] in
            for imageID in imageIDs {
                guard let r = try? self.context.existingObject(with: imageID) as? ImageRecord,
                      let path = r.filePath else { continue }
                let stars: Int
                if let override = r.userOverride, override.int16Value > 0 {
                    stars = Int(override.int16Value)
                } else if let s = r.ratingStars {
                    stars = Int(s.int16Value)
                } else {
                    continue
                }
                try? MetadataWriter.writeSidecar(stars: stars, for: URL(filePath: path))
            }
        }
    }

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
            where record.processState == ProcessState.culling
               || record.processState == ProcessState.rating
               || record.processState == ProcessState.rated {
                record.processState = ProcessState.interrupted
            }
            try? self.context.save()
        }
    }
}
