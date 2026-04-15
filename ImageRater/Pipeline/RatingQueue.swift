import AppKit
import CoreData
import Foundation

/// Coalesces manual rating changes and persists them with zero main-thread disk work.
///
/// Hot path (keypress):
///   1. Caller sets record.userOverride in-memory → SwiftUI updates instantly.
///   2. Caller calls enqueue(id:stars:) — just stores in a dict, no I/O.
///   3. Advance to next image.
///
/// Drain (300 ms after last enqueue):
///   1. ctx.save() scheduled via DispatchQueue.main.async — deferred to the next idle
///      run-loop slot, never preempts a keypress.
///   2. Companion lookup + XMP writes run inside performBackgroundTask — fully off
///      main thread. Uses a targeted predicate fetch instead of allObjects.
///
/// flush() — explicit synchronous save on session-switch / view-disappear.
@MainActor
final class RatingQueue {

    private var context: NSManagedObjectContext?
    private var container: NSPersistentContainer?
    private var progress: RatingProgress?
    private var drainTask: Task<Void, Never>?
    /// objectID → effective stars (0 = clear override, still needs save)
    private var pending: [NSManagedObjectID: Int] = [:]
    private var hasPending = false
    private var terminateObserver: NSObjectProtocol?

    func configure(context: NSManagedObjectContext, container: NSPersistentContainer, progress: RatingProgress) {
        self.context = context
        self.container = container
        self.progress = progress
        // Synchronous flush on app quit — prevents loss of pending ratings.
        // willTerminate fires after user confirms quit; blocking here keeps the
        // process alive until CoreData save + XMP writes complete.
        if terminateObserver == nil {
            terminateObserver = NotificationCenter.default.addObserver(
                forName: NSApplication.willTerminateNotification,
                object: nil,
                queue: .main
            ) { [weak self] _ in
                MainActor.assumeIsolated { self?.flushSynchronous() }
            }
        }
    }

    /// Synchronous flush — used by willTerminate. Blocks until save + XMP writes done.
    private func flushSynchronous() {
        drainTask?.cancel()
        drainTask = nil
        guard hasPending, let context, let container else { return }
        hasPending = false
        let snapshot = pending
        pending.removeAll()
        if context.hasChanges { try? context.save() }

        guard !snapshot.isEmpty else { return }
        guard UserDefaults.standard.object(forKey: FocalSettings.autoWriteXMP) == nil
                ? FocalSettings.defaultAutoWriteXMP
                : UserDefaults.standard.bool(forKey: FocalSettings.autoWriteXMP)
        else { return }

        // Private-queue context + performAndWait — blocks main until companion lookup
        // + sidecar writes done. container.performBackgroundTask is async fire-and-forget
        // and would let the process exit before writes complete.
        let bgCtx = NSManagedObjectContext(concurrencyType: .privateQueueConcurrencyType)
        bgCtx.persistentStoreCoordinator = container.persistentStoreCoordinator
        // willTerminate path — main thread is blocked, no UI updates possible. Skip progress.
        bgCtx.performAndWait {
            let xmpTasks = Self.expandToCompanions(snapshot: snapshot, ctx: bgCtx)
            Self.applyXMPTasks(xmpTasks, progress: nil)
        }
    }

    /// Enqueue a rating change. Call after setting record.userOverride in-memory.
    /// stars == 0 means the override was cleared (still triggers a save, no XMP write).
    func enqueue(id: NSManagedObjectID, stars: Int) {
        hasPending = true
        pending[id] = stars
        scheduleDrain()
    }

    /// Flush synchronously — call on session switch or view disappear so the store
    /// is up-to-date before the caller continues.
    func flush() {
        drainTask?.cancel()
        drainTask = nil
        guard hasPending, let context, let container else { return }
        hasPending = false
        let snapshot = pending
        pending.removeAll()
        if context.hasChanges { try? context.save() }
        writeXMPInBackground(snapshot: snapshot, container: container)
    }

    // MARK: - Private

    private func scheduleDrain() {
        drainTask?.cancel()
        drainTask = Task { [weak self] in
            try? await Task.sleep(for: .milliseconds(300))
            guard !Task.isCancelled else { return }
            self?.drainAsync()
        }
    }

    private func drainAsync() {
        guard hasPending, let context, let container else { return }
        hasPending = false
        let snapshot = pending
        pending.removeAll()

        // Defer save to next idle run-loop slot — never blocks a keypress.
        DispatchQueue.main.async {
            if context.hasChanges { try? context.save() }
        }

        writeXMPInBackground(snapshot: snapshot, container: container)
    }

    /// Looks up file paths (including companions) in a private-queue CoreData context
    /// and writes XMP sidecars — entirely off the main thread.
    private func writeXMPInBackground(
        snapshot: [NSManagedObjectID: Int],
        container: NSPersistentContainer
    ) {
        guard !snapshot.isEmpty else { return }
        guard UserDefaults.standard.object(forKey: FocalSettings.autoWriteXMP) == nil
                ? FocalSettings.defaultAutoWriteXMP
                : UserDefaults.standard.bool(forKey: FocalSettings.autoWriteXMP)
        else { return }

        let progress = self.progress
        Task.detached(priority: .background) {
            container.performBackgroundTask { bgCtx in
                let xmpTasks = Self.expandToCompanions(snapshot: snapshot, ctx: bgCtx)
                Self.applyXMPTasks(xmpTasks, progress: progress)
            }
        }
    }

    /// Expand each (primary objectID, stars) pair into (URL, stars) pairs for the
    /// primary file plus all RAW companions in its group. Must run on `ctx`'s queue.
    private static func expandToCompanions(
        snapshot: [NSManagedObjectID: Int],
        ctx: NSManagedObjectContext
    ) -> [(URL, Int)] {
        var tasks: [(URL, Int)] = []
        for (objectID, stars) in snapshot {
            guard let record = try? ctx.existingObject(with: objectID) as? ImageRecord,
                  let path = record.filePath else { continue }
            tasks.append((URL(filePath: path), stars))
            if let gid = record.groupID {
                let req = ImageRecord.fetchRequest()
                req.predicate = NSPredicate(
                    format: "groupID == %@ AND isGroupPrimary == NO", gid)
                req.propertiesToFetch = ["filePath"]
                if let companions = try? ctx.fetch(req) {
                    for c in companions {
                        if let cp = c.filePath {
                            tasks.append((URL(filePath: cp), stars))
                        }
                    }
                }
            }
        }
        return tasks
    }

    /// Apply each rating to disk: stars > 0 writes/embeds the rating; stars == 0
    /// strips it (deletes RAW sidecar / removes embedded xmp:Rating).
    /// Reports progress through `progress` if supplied; safe to pass nil from blocking
    /// paths (e.g. willTerminate where the main thread can't render UI updates).
    private static func applyXMPTasks(_ tasks: [(URL, Int)], progress: RatingProgress?) {
        let total = tasks.count
        if let progress {
            DispatchQueue.main.async { progress.start(phase: "Saving ratings…", total: total) }
        }
        for (url, stars) in tasks {
            if stars > 0 {
                try? MetadataWriter.writeSidecar(stars: stars, for: url)
            } else {
                try? MetadataWriter.clearRating(for: url)
            }
            if let progress {
                DispatchQueue.main.async { progress.tick() }
            }
        }
        if let progress {
            DispatchQueue.main.async { progress.finish() }
        }
    }
}
