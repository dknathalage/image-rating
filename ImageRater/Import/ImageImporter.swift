import CoreData
import Foundation

enum ImageImporter {

    static let supportedExtensions: Set<String> = LibRawWrapper.supportedExtensions
        .union(["jpg", "jpeg", "png", "tiff", "tif", "heic", "heif"])

    /// Scan folder, upsert Session + ImageRecord entities in context, return Session objectID.
    /// If a session with the same folderPath already exists, only new files are added and
    /// existing records (with their ratings) are preserved. Creates a new session otherwise.
    /// Must be called on `context`'s queue (wrap in `context.performAndWait {}` from other threads).
    @discardableResult
    static func importFolder(_ url: URL,
                              context: NSManagedObjectContext) throws -> NSManagedObjectID {
        // Upsert: reuse existing session for the same folder path
        let req = Session.fetchRequest()
        req.predicate = NSPredicate(format: "folderPath == %@", url.path)
        req.fetchLimit = 1

        let session: Session
        let existingPaths: Set<String>

        if let existing = (try? context.fetch(req))?.first {
            session = existing
            existingPaths = Set(
                (session.images?.allObjects as? [ImageRecord] ?? []).compactMap { $0.filePath }
            )
        } else {
            session = Session(context: context)
            session.id = UUID()
            session.createdAt = Date()
            session.folderPath = url.path
            existingPaths = []
        }

        let files = try scanFolder(url)
        let newFiles = files.filter { !existingPaths.contains($0.path) }

        guard !newFiles.isEmpty else {
            return session.objectID
        }

        let groups = groupByBaseName(newFiles)

        for group in groups {
            if group.count == 1 {
                // Unpaired single file
                createRecord(for: group[0], session: session, context: context,
                             groupID: nil, isPrimary: true, scoringPath: nil)
            } else {
                // Paired: exactly one non-RAW (JPEG) + ≥1 RAW
                let jpegURL = group.first { !MetadataWriter.rawExtensions.contains($0.pathExtension.lowercased()) }!
                let rawURLs = group.filter { MetadataWriter.rawExtensions.contains($0.pathExtension.lowercased()) }
                let gid = UUID().uuidString
                // JPEG is primary; scoringFilePath = first RAW (group is sorted alphabetically)
                createRecord(for: jpegURL, session: session, context: context,
                             groupID: gid, isPrimary: true, scoringPath: rawURLs[0].path)
                for rawURL in rawURLs {
                    createRecord(for: rawURL, session: session, context: context,
                                 groupID: gid, isPrimary: false, scoringPath: nil)
                }
            }
        }

        try context.save()
        return session.objectID
    }

    /// Recursively scan folder for supported image files. Returns sorted list.
    static func scanFolder(_ url: URL) throws -> [URL] {
        guard let enumerator = FileManager.default.enumerator(
            at: url,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles, .skipsPackageDescendants]
        ) else { return [] }

        return enumerator.compactMap { item -> URL? in
            guard let fileURL = item as? URL,
                  (try? fileURL.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile) == true,
                  supportedExtensions.contains(fileURL.pathExtension.lowercased())
            else { return nil }
            return fileURL
        }.sorted { $0.lastPathComponent.localizedStandardCompare($1.lastPathComponent) == .orderedAscending }
    }

    /// Group URLs by lowercased base name. Returns array of groups sorted by base name.
    /// A group is a pair only when it has exactly one non-RAW file and ≥1 RAW.
    /// Ambiguous groups (multiple non-RAWs, or multiple RAWs with no non-RAW) split into singles.
    static func groupByBaseName(_ urls: [URL]) -> [[URL]] {
        var byBase: [String: [URL]] = [:]
        for url in urls {
            let base = url.deletingPathExtension().lastPathComponent.lowercased()
            byBase[base, default: []].append(url)
        }

        var result: [[URL]] = []
        for (_, group) in byBase.sorted(by: { $0.key < $1.key }) {
            let nonRaw = group.filter { !MetadataWriter.rawExtensions.contains($0.pathExtension.lowercased()) }
            let raw    = group.filter {  MetadataWriter.rawExtensions.contains($0.pathExtension.lowercased()) }

            if nonRaw.count == 1 && raw.count >= 1 {
                // Valid pair: one JPEG + ≥1 RAW
                let sorted = (nonRaw + raw).sorted {
                    $0.lastPathComponent.localizedStandardCompare($1.lastPathComponent) == .orderedAscending
                }
                result.append(sorted)
            } else {
                // Ambiguous or single-type — each file is its own entry
                for url in group {
                    result.append([url])
                }
            }
        }
        return result
    }

    // MARK: Private

    private static func createRecord(for url: URL,
                                      session: Session,
                                      context: NSManagedObjectContext,
                                      groupID: String?,
                                      isPrimary: Bool,
                                      scoringPath: String?) {
        let record = ImageRecord(context: context)
        record.id = UUID()
        record.filePath = url.path
        record.processState = "pending"
        record.decodeError = false
        record.cullRejected = false
        record.session = session
        record.groupID = groupID
        record.isGroupPrimary = isPrimary
        record.scoringFilePath = scoringPath
    }
}
