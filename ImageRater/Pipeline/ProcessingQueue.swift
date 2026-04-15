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
                            r.musiqAesthetic = scores.musiqAesthetic
                            r.ratingStars = NSNumber(value: Int16(scores.stars))
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

        // ─── PASS 2: propagate stars to companions + sidecar writes ───────────

        await writeStarsAndSidecars(sessionID: sessionID, onProgress: onProgress)

        onProgress?(total, total, "Done")
        log.info("Session complete — \(total) images, \(decodeErrorCount) decode errors")
    }

    // MARK: Private

    /// Propagates chosen star value (override or AI) to companions by groupID and writes
    /// XMP sidecars for all rated images. Per-image star bucketing already happened in
    /// `RatingPipeline.rate()`, so no session-level normalization needed.
    private func writeStarsAndSidecars(
        sessionID: NSManagedObjectID,
        onProgress: (@Sendable (Int, Int, String) -> Void)?
    ) async {
        struct Entry {
            let id: NSManagedObjectID
            let filePath: String?
            let groupID: String?
            let overrideStars: Int?
            let ratingStars: Int
        }

        let entries: [Entry] = await context.perform { [self] in
            guard let session = try? self.context.existingObject(with: sessionID) as? Session,
                  let images = session.images?.allObjects as? [ImageRecord] else { return [] }
            return images
                .filter { $0.isGroupPrimary && ($0.processState == ProcessState.done || $0.processState == ProcessState.rated) }
                .map { r in
                    Entry(
                        id: r.objectID,
                        filePath: r.filePath,
                        groupID: r.groupID,
                        overrideStars: r.userOverride.map { Int($0.int16Value) },
                        ratingStars: Int(r.ratingStars?.int16Value ?? 0)
                    )
                }
        }

        guard !entries.isEmpty else { return }

        var writeTasks: [WriteTask] = []

        await context.perform { [self] in
            guard let session = try? self.context.existingObject(with: sessionID) as? Session,
                  let allImages = session.images?.allObjects as? [ImageRecord] else { return }

            var companionsByGroup: [String: [ImageRecord]] = [:]
            for img in allImages where !img.isGroupPrimary {
                guard let gid = img.groupID else { continue }
                companionsByGroup[gid, default: []].append(img)
            }

            for entry in entries {
                let stars = entry.overrideStars ?? entry.ratingStars
                if stars <= 0 { continue }

                if let fp = entry.filePath {
                    writeTasks.append(WriteTask(url: URL(filePath: fp), stars: stars))
                }

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
    /// Scores only the supplied imageIDs, then propagates stars + writes sidecars.
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
                            r.musiqAesthetic = scores.musiqAesthetic
                            r.ratingStars = NSNumber(value: Int16(scores.stars))
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
        await writeStarsAndSidecars(sessionID: sessionID, onProgress: onProgress)
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
