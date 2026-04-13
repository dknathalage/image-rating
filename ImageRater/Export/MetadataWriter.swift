import CoreImage
import Foundation

enum MetadataWriterError: Error {
    case serializationFailed
    case writeFailed(Error)
    case readFailed
}

enum MetadataWriter {

    // XMP Basic namespace
    private static let xmpNS = "http://ns.adobe.com/xap/1.0/" as CFString
    // MicrosoftPhoto namespace
    private static let msNS = "http://ns.microsoft.com/photo/1.0/" as CFString

    /// Write xmp:Rating (0–5) to a .xmp sidecar file at `url`.
    static func write(stars: Int, to url: URL) throws {
        let metadata = CGImageMetadataCreateMutable()

        // Register MicrosoftPhoto namespace
        CGImageMetadataRegisterNamespaceForPrefix(metadata, msNS, "MicrosoftPhoto" as CFString, nil)

        // xmp:Rating (XMP Basic — already registered by default)
        CGImageMetadataSetValueWithPath(metadata, nil, "xmp:Rating" as CFString,
                                        stars as CFTypeRef)

        // MicrosoftPhoto:Rating (0/1/25/50/75/99)
        let msRating = microsoftRating(from: stars)
        CGImageMetadataSetValueWithPath(metadata, nil, "MicrosoftPhoto:Rating" as CFString,
                                        msRating as CFTypeRef)

        guard let xmpData = CGImageMetadataCreateXMPData(metadata, nil) as Data? else {
            throw MetadataWriterError.serializationFailed
        }
        do {
            try xmpData.write(to: url, options: .atomic)
        } catch {
            throw MetadataWriterError.writeFailed(error)
        }
    }

    /// Writes sidecar next to the source image file (replaces extension with .xmp).
    static func writeSidecar(stars: Int, for imageURL: URL) throws {
        let xmpURL = imageURL.deletingPathExtension().appendingPathExtension("xmp")
        try write(stars: stars, to: xmpURL)
    }

    /// Read back xmp:Rating integer from a .xmp file.
    static func readRating(from url: URL) throws -> Int {
        let data = try Data(contentsOf: url) as CFData
        guard let metadata = CGImageMetadataCreateFromXMPData(data) else {
            throw MetadataWriterError.readFailed
        }
        guard let value = CGImageMetadataCopyStringValueWithPath(
            metadata, nil, "xmp:Rating" as CFString) as String? else {
            return 0
        }
        return Int(value) ?? 0
    }

    // MARK: Private

    private static func microsoftRating(from stars: Int) -> Int {
        switch stars {
        case 1: return 1
        case 2: return 25
        case 3: return 50
        case 4: return 75
        case 5: return 99
        default: return 0
        }
    }
}
