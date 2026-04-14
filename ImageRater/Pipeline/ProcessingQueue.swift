import CoreData
import Foundation
import OSLog

private let log = Logger(subsystem: "com.focal.app", category: "ProcessingQueue")

enum ProcessState {
    static let pending = "pending"
    static let culling = "culling"
    static let rating = "rating"
    static let rated = "rated"
    static let done = "done"
    static let interrupted = "interrupted"
}

private struct WriteTask: Sendable {
    let url: URL
    let stars: Int
}

actor ProcessingQueue {

    private let context: NSManagedObjectContext

    init(context: NSManagedObjectContext) {
        self.context = context
    }

    func process(
        sessionID: NSManagedObjectID,
        onProgress: (@Sendable (Int, Int, String) -> Void)? = nil
    ) async throws {

        let imageIDs = try await context.perform { [self] in
            guard let session = try? self.context.existingObject(with: sessionID) as? Session,
                  let images = session.images?.allObjects as? [ImageRecord] else {
                throw CocoaError(.coreData)
            }
            let sorted = images.sorted { ($0.filePath ?? "") < ($1.filePath ?? "") }
            return sorted.map(\.objectID)
        }

        onProgress?(0, 0, "Compiling models…")

        let total = imageIDs.count
        var decodeErrorCount = 0
        log.info("Session processing started: \(total) images")

        // ─── PASS 1: per-image inference — models scoped here, released before Pass 2 ──

        do {
            let models = try RatingPipeline.loadBundledModels()
            do {
                for (i, imageID) in imageIDs.enumerated() {
                    try Task.checkCancellation()

                    let (filePath, scoringPath, skip): (String?, String?, Bool) = await context.perform { [self] in
                        guard let r = try? self.context.existingObject(with: imageID) as? ImageRecord else {
                            return (nil, nil, true)
                        }
                        if !r.isGroupPrimary { return (nil, nil, true) }
                        let skip = r.processState == ProcessState.done
                                || r.processState == ProcessState.rated
                        return (r.filePath, r.scoringFilePath, skip)
                    }
                    if skip { continue }

                    onProgress?(i, total, "Scoring \(i + 1) of \(total)")

                    let decodePath = scoringPath ?? filePath
                    guard let path = decodePath else {
                        decodeErrorCount += 1
                        await context.perform { [self] in
                            guard let r = try? self.context.existingObject(with: imageID) as? ImageRecord else { return }
                            r.decodeError = true
                            r.processState = ProcessState.done
                            try? self.context.save()
                        }
                        continue
                    }

                    let hasOverride = await context.perform { [self] in
                        (try? self.context.existingObject(with: imageID) as? ImageRecord)?.userOverride != nil
                    }

                    // scoreImage is a separate function: cgImage decoded + inference run inside,
                    // then released before this await point resumes — avoids holding ~160MB
                    // across the subsequent context.perform await.
                    let (ratingResult, decodeError) = await scoreImage(
                        at: path, models: models, hasOverride: hasOverride)

                    if decodeError {
                        decodeErrorCount += 1
                    }
                    await context.perform { [self] in
                        guard let r = try? self.context.existingObject(with: imageID) as? ImageRecord else { return }

                        if decodeError {
                            r.decodeError = true
                        } else if !hasOverride, case .rated(let scores) = ratingResult {
                            r.topiqTechnicalScore  = scores.topiqTechnicalScore
                            r.topiqAestheticScore  = scores.topiqAestheticScore
                            r.clipIQAScore         = scores.clipIQAScore
                            r.combinedQualityScore = scores.combinedQualityScore
                        }
                        r.processState = ProcessState.done
                        try? self.context.save()
                    }
                }
            } catch is CancellationError {
                await markInterrupted(sessionID: sessionID)
                throw CancellationError()
            }
        } // models released here — frees ML model memory before file I/O phase

        try Task.checkCancellation()

        // ─── PASS 2: percentile star assignment + file writes ─────────────────

        await normalizeAndWriteStars(sessionID: sessionID, onProgress: onProgress)

        onProgress?(total, total, "Done")
        log.info("Session complete — \(total) images, \(decodeErrorCount) decode errors")
    }

    // MARK: Private

    /// Rank all processed primaries by session-normalised combined score, assign percentile-based
    /// 1–5 star ratings scaled by strictness, propagate to companions, and write metadata.
    /// Each sub-score (technical, aesthetic, CLIP) is min-max normalised across the session
    /// before combining so that a narrow-range model (e.g. aesthetic ±0.10) contributes
    /// as much as a wide-range model (e.g. technical ±0.60).
    private func normalizeAndWriteStars(
        sessionID: NSManagedObjectID,
        onProgress: (@Sendable (Int, Int, String) -> Void)?
    ) async {
        struct Entry {
            let id: NSManagedObjectID
            let tech: Float
            let aes: Float
            let clip: Float
            let filePath: String?
            let groupID: String?
            let overrideStars: Int?
            let decodeError: Bool
        }

        let entries: [Entry] = await context.perform { [self] in
            guard let session = try? self.context.existingObject(with: sessionID) as? Session,
                  let images = session.images?.allObjects as? [ImageRecord] else { return [] }
            return images
                .filter { $0.isGroupPrimary && ($0.processState == ProcessState.done || $0.processState == ProcessState.rated) }
                .map { r in
                    Entry(
                        id: r.objectID,
                        tech: r.topiqTechnicalScore,
                        aes:  r.topiqAestheticScore,
                        clip: r.clipIQAScore,
                        filePath: r.filePath,
                        groupID: r.groupID,
                        overrideStars: r.userOverride.map { Int($0.int16Value) },
                        decodeError: r.decodeError
                    )
                }
        }

        guard !entries.isEmpty else { return }

        let storedStrictness = UserDefaults.standard.double(forKey: FocalSettings.cullStrictness)
        let strictness = storedStrictness == 0 ? FocalSettings.defaultCullStrictness : storedStrictness

        // Percentile rank only valid (non-override, non-decode-error) images
        let validIndices = entries.indices.filter { !entries[$0].decodeError && entries[$0].overrideStars == nil }
        let valid = validIndices.map { entries[$0] }

        // Min-max normalise each sub-score across the valid pool
        func minMax(_ vals: [Float]) -> (Float, Float) {
            (vals.min() ?? 0, vals.max() ?? 1)
        }
        func norm(_ v: Float, _ lo: Float, _ hi: Float) -> Float {
            hi > lo ? (v - lo) / (hi - lo) : 0.5
        }
        let (tLo, tHi) = minMax(valid.map(\.tech))
        let (aLo, aHi) = minMax(valid.map(\.aes))
        let (cLo, cHi) = minMax(valid.map(\.clip))

        let ud = UserDefaults.standard
        let wTech = ud.object(forKey: FocalSettings.weightTechnical) != nil
            ? Float(ud.double(forKey: FocalSettings.weightTechnical))
            : Float(FocalSettings.defaultWeightTechnical)
        let wAes = ud.object(forKey: FocalSettings.weightAesthetic) != nil
            ? Float(ud.double(forKey: FocalSettings.weightAesthetic))
            : Float(FocalSettings.defaultWeightAesthetic)
        let wClip = ud.object(forKey: FocalSettings.weightClip) != nil
            ? Float(ud.double(forKey: FocalSettings.weightClip))
            : Float(FocalSettings.defaultWeightClip)
        let wSum = wTech + wAes + wClip
        let (wTn, wAn, wCn) = wSum > 0
            ? (wTech/wSum, wAes/wSum, wClip/wSum)
            : (Float(FocalSettings.defaultWeightTechnical),
               Float(FocalSettings.defaultWeightAesthetic),
               Float(FocalSettings.defaultWeightClip))

        let validScores: [Float] = valid.map { e in
            wTn * norm(e.tech, tLo, tHi) +
            wAn * norm(e.aes,  aLo, aHi) +
            wCn * norm(e.clip, cLo, cHi)
        }

        let percentileRatings = percentileStars(scores: validScores, strictness: strictness)

        // Map star assignments back to full entry array
        var starsForIndex = [Int](repeating: 0, count: entries.count)
        for (pos, entryIdx) in validIndices.enumerated() {
            starsForIndex[entryIdx] = percentileRatings[pos]
        }

        // Write stars to CoreData and collect file tasks
        var writeTasks: [WriteTask] = []

        await context.perform { [self] in
            guard let session = try? self.context.existingObject(with: sessionID) as? Session,
                  let allImages = session.images?.allObjects as? [ImageRecord] else { return }

            var companionsByGroup: [String: [ImageRecord]] = [:]
            for img in allImages where !img.isGroupPrimary {
                guard let gid = img.groupID else { continue }
                companionsByGroup[gid, default: []].append(img)
            }

            for (i, entry) in entries.enumerated() {
                guard let r = try? self.context.existingObject(with: entry.id) as? ImageRecord else { continue }

                let stars: Int
                if let o = entry.overrideStars {
                    stars = o
                } else if entry.decodeError {
                    stars = 0
                } else {
                    stars = starsForIndex[i]
                    r.ratingStars = NSNumber(value: Int16(stars))
                }

                if let fp = entry.filePath {
                    writeTasks.append(WriteTask(url: URL(filePath: fp), stars: stars))
                }

                // Copy stars + write tasks for companions
                if let gid = entry.groupID, let companions = companionsByGroup[gid] {
                    for c in companions {
                        c.ratingStars = NSNumber(value: Int16(stars))
                        if let cp = c.filePath {
                            writeTasks.append(WriteTask(url: URL(filePath: cp), stars: stars))
                        }
                    }
                }
            }
            try? self.context.save()
        }

        let writeTotal = writeTasks.count
        onProgress?(0, writeTotal, "Saving ratings…")
        for (i, task) in writeTasks.enumerated() where task.stars > 0 {
            try? MetadataWriter.writeSidecar(stars: task.stars, for: task.url)
            onProgress?(i + 1, writeTotal, "Saving ratings…")
        }
    }

    /// Assigns 1–5 stars using a power-curve skew on percentile rank.
    /// γ = 10^(2s−1): s=0→γ=0.1 (lenient), s=0.5→γ=1 (uniform), s=1→γ=10 (strict)
    /// Buckets are evenly spaced [0.2, 0.4, 0.6, 0.8] in warped space.
    private func percentileStars(scores: [Float], strictness: Double) -> [Int] {
        let n = scores.count
        guard n > 0 else { return [] }
        let gamma = pow(10.0, 2 * min(max(strictness, 0), 1) - 1)

        let sorted = scores.enumerated().sorted { $0.element < $1.element }
        var result = [Int](repeating: 3, count: n)
        for (rank, (originalIndex, _)) in sorted.enumerated() {
            let pct = Double(rank) / Double(n)
            let warped = pow(pct == 0 ? 1e-9 : pct, gamma)
            result[originalIndex] = switch warped {
            case ..<0.20:       1
            case 0.20..<0.40:   2
            case 0.40..<0.60:   3
            case 0.60..<0.80:   4
            default:            5
            }
        }
        return result
    }

    /// Decode image + run inference in one function so cgImage is released
    /// when this returns — before the caller's next await point.
    private func scoreImage(
        at path: String,
        models: RatingPipeline.BundledModels,
        hasOverride: Bool
    ) async -> (RatingResult, Bool) {
        guard let cgImage = LibRawWrapper.decode(url: URL(filePath: path)) else {
            return (.unrated, true) // decodeError = true
        }
        let result = await rateIfNeeded(image: cgImage, models: models, hasOverride: hasOverride)
        return (result, false)
        // cgImage ARC-released here, before caller continues
    }

    private func rateIfNeeded(
        image: CGImage,
        models: RatingPipeline.BundledModels,
        hasOverride: Bool
    ) async -> RatingResult {
        guard !hasOverride else { return .unrated }
        return await RatingPipeline.rate(image: image, models: models)
    }

    /// Run the full pipeline on a specific subset of images.
    /// Scores only the supplied imageIDs, then normalises star ratings across the whole session.
    func process(
        imageIDs: [NSManagedObjectID],
        onProgress: (@Sendable (Int, Int, String) -> Void)? = nil
    ) async throws {
        guard !imageIDs.isEmpty else { return }

        // Resolve sessionID from first image
        let sessionID: NSManagedObjectID = try await context.perform { [self] in
            guard let record = try? self.context.existingObject(with: imageIDs[0]) as? ImageRecord,
                  let session = record.session else { throw CocoaError(.coreData) }
            return session.objectID
        }

        onProgress?(0, 0, "Compiling models…")
        let total = imageIDs.count

        do {
            let models = try RatingPipeline.loadBundledModels()
            do {
                for (i, imageID) in imageIDs.enumerated() {
                    try Task.checkCancellation()

                    let (filePath, scoringPath, skip): (String?, String?, Bool) = await context.perform { [self] in
                        guard let r = try? self.context.existingObject(with: imageID) as? ImageRecord else {
                            return (nil, nil, true)
                        }
                        if !r.isGroupPrimary { return (nil, nil, true) }
                        let skip = r.processState == ProcessState.done || r.processState == ProcessState.rated
                        return (r.filePath, r.scoringFilePath, skip)
                    }
                    if skip { continue }

                    onProgress?(i, total, "Scoring \(i + 1) of \(total)")

                    let decodePath = scoringPath ?? filePath
                    guard let path = decodePath else {
                        await context.perform { [self] in
                            guard let r = try? self.context.existingObject(with: imageID) as? ImageRecord else { return }
                            r.decodeError = true
                            r.processState = ProcessState.done
                            try? self.context.save()
                        }
                        continue
                    }

                    let hasOverride = await context.perform { [self] in
                        (try? self.context.existingObject(with: imageID) as? ImageRecord)?.userOverride != nil
                    }

                    let (ratingResult, decodeError) = await scoreImage(at: path, models: models, hasOverride: hasOverride)

                    await context.perform { [self] in
                        guard let r = try? self.context.existingObject(with: imageID) as? ImageRecord else { return }
                        if decodeError {
                            r.decodeError = true
                        } else if !hasOverride, case .rated(let scores) = ratingResult {
                            r.topiqTechnicalScore  = scores.topiqTechnicalScore
                            r.topiqAestheticScore  = scores.topiqAestheticScore
                            r.clipIQAScore         = scores.clipIQAScore
                            r.combinedQualityScore = scores.combinedQualityScore
                        }
                        r.processState = ProcessState.done
                        try? self.context.save()
                    }
                }
            } catch is CancellationError {
                await markInterrupted(sessionID: sessionID)
                throw CancellationError()
            }
        }

        try Task.checkCancellation()
        await normalizeAndWriteStars(sessionID: sessionID, onProgress: onProgress)
        onProgress?(total, total, "Done")
    }
}

// MARK: - External API

extension ProcessingQueue {
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
