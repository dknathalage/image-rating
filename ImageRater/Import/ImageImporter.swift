import CoreData
import Foundation

enum ImageImporter {

    static let supportedExtensions: Set<String> = LibRawWrapper.supportedExtensions
        .union(["jpg", "jpeg", "png", "tiff", "tif", "heic", "heif"])

    /// Scan folder, create Session + ImageRecord entities in context, return Session objectID.
    /// Must be called on `context`'s queue (wrap in `context.performAndWait {}` from other threads).
    @discardableResult
    static func importFolder(_ url: URL,
                              context: NSManagedObjectContext) throws -> NSManagedObjectID {
        let session = Session(context: context)
        session.id = UUID()
        session.createdAt = Date()
        session.folderPath = url.path

        let files = try scanFolder(url)
        for fileURL in files {
            let record = ImageRecord(context: context)
            record.id = UUID()
            record.filePath = fileURL.path
            record.processState = "pending"
            record.decodeError = false
            record.cullRejected = false
            record.session = session
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
}
