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
    private var drainTask: Task<Void, Never>?
    /// objectID → effective stars (0 = clear override, still needs save)
    private var pending: [NSManagedObjectID: Int] = [:]
    private var hasPending = false

    func configure(context: NSManagedObjectContext, container: NSPersistentContainer) {
        self.context = context
        self.container = container
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
        let toWrite = snapshot.filter { $0.value > 0 }
        guard !toWrite.isEmpty else { return }
        guard UserDefaults.standard.object(forKey: FocalSettings.autoWriteXMP) == nil
                ? FocalSettings.defaultAutoWriteXMP
                : UserDefaults.standard.bool(forKey: FocalSettings.autoWriteXMP)
        else { return }

        Task.detached(priority: .background) {
            container.performBackgroundTask { bgCtx in
                var xmpTasks: [(URL, Int)] = []
                for (objectID, stars) in toWrite {
                    guard let record = try? bgCtx.existingObject(with: objectID) as? ImageRecord,
                          let path = record.filePath else { continue }
                    xmpTasks.append((URL(filePath: path), stars))

                    // Fetch companions by groupID — targeted predicate, not allObjects.
                    if let gid = record.groupID {
                        let req = ImageRecord.fetchRequest()
                        req.predicate = NSPredicate(
                            format: "groupID == %@ AND isGroupPrimary == NO", gid)
                        req.propertiesToFetch = ["filePath"]
                        if let companions = try? bgCtx.fetch(req) {
                            for c in companions {
                                if let cp = c.filePath {
                                    xmpTasks.append((URL(filePath: cp), stars))
                                }
                            }
                        }
                    }
                }
                for (url, stars) in xmpTasks {
                    try? MetadataWriter.writeSidecar(stars: stars, for: url)
                }
            }
        }
    }
}
