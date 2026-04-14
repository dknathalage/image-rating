import CoreImage
import Foundation
import ImageIO

enum MetadataWriterError: Error {
    case serializationFailed
    case writeFailed(Error)
    case readFailed
    case embeddingFailed
}

enum MetadataWriter {

    /// RAW formats that require XMP sidecar files (cannot be re-encoded).
    static let rawExtensions: Set<String> = [
        "cr2", "cr3", "nef", "nrw", "arw", "srf", "sr2",
        "raf", "orf", "rw2", "pef", "dng", "3fr", "fff",
        "iiq", "cap", "raw", "rwl", "mrw", "x3f",
        "srw", "rw1", "kdc", "dcr", "erf", "mef", "mos",
        "nex", "ptx", "r3d"
    ]

    /// Write xmp:Rating (0–5) to a .xmp sidecar file at `url`.
    static func write(stars: Int, to url: URL) throws {
        guard (0...5).contains(stars) else { throw MetadataWriterError.serializationFailed }
        try xmpPacket(rating: stars, label: nil).write(to: url)
    }

    /// Writes metadata to the image file: embeds directly for non-RAW (JPEG/PNG/HEIC/TIFF),
    /// writes XMP sidecar for RAW formats. Photomator reads embedded metadata for non-RAW.
    static func writeSidecar(stars: Int, for imageURL: URL) throws {
        if isRAW(imageURL) {
            try write(stars: stars, to: sidecarURL(for: imageURL))
        } else {
            try embedInFile(stars: stars, rejected: false, for: imageURL)
        }
    }

    /// Write xmp:Rating = -1 and xmp:Label = "Rejected" to explicit sidecar URL.
    static func writeRejected(to url: URL) throws {
        try xmpPacket(rating: -1, label: "Rejected").write(to: url)
    }

    /// Writes rejection metadata: embeds in file for non-RAW, writes sidecar for RAW.
    static func writeSidecarRejected(for imageURL: URL) throws {
        if isRAW(imageURL) {
            try writeRejected(to: sidecarURL(for: imageURL))
        } else {
            try embedInFile(stars: -1, rejected: true, for: imageURL)
        }
    }

    /// Read back xmp:Rating integer from a .xmp sidecar file.
    static func readRating(from url: URL) throws -> Int {
        let data = try Data(contentsOf: url) as CFData
        guard let metadata = CGImageMetadataCreateFromXMPData(data) else {
            throw MetadataWriterError.readFailed
        }
        guard let value = CGImageMetadataCopyStringValueWithPath(
            metadata, nil, "xmp:Rating" as CFString) as String? else { return 0 }
        return Int(value) ?? 0
    }

    // MARK: Private

    private static func isRAW(_ url: URL) -> Bool {
        Self.rawExtensions.contains(url.pathExtension.lowercased())
    }

    private static func sidecarURL(for imageURL: URL) -> URL {
        imageURL.deletingPathExtension().appendingPathExtension("xmp")
    }

    /// Embeds xmp:Rating (and optionally xmp:Label) directly into the image file using
    /// CGImageDestination with kCGImageDestinationMergeMetadata — preserves all existing
    /// metadata and original image compression (lossless metadata edit for JPEG).
    private static func embedInFile(stars: Int, rejected: Bool, for imageURL: URL) throws {
        guard let source = CGImageSourceCreateWithURL(imageURL as CFURL, nil),
              let uti = CGImageSourceGetType(source) else {
            throw MetadataWriterError.readFailed
        }

        let metadata = CGImageMetadataCreateMutable()
        // Register XMP namespace — safe to ignore failure (already known by ImageIO)
        CGImageMetadataRegisterNamespaceForPrefix(
            metadata,
            "http://ns.adobe.com/xap/1.0/" as CFString,
            "xmp" as CFString,
            nil
        )

        guard let ratingTag = CGImageMetadataTagCreate(
            "http://ns.adobe.com/xap/1.0/" as CFString,
            "xmp" as CFString,
            "Rating" as CFString,
            .string,
            String(stars) as CFTypeRef
        ) else {
            throw MetadataWriterError.embeddingFailed
        }
        CGImageMetadataSetTagWithPath(metadata, nil, "xmp:Rating" as CFString, ratingTag)

        if rejected, let labelTag = CGImageMetadataTagCreate(
            "http://ns.adobe.com/xap/1.0/" as CFString,
            "xmp" as CFString,
            "Label" as CFString,
            .string,
            "Rejected" as CFTypeRef
        ) {
            CGImageMetadataSetTagWithPath(metadata, nil, "xmp:Label" as CFString, labelTag)
        }

        let tempURL = imageURL.deletingLastPathComponent()
            .appendingPathComponent("._imgrater_\(UUID().uuidString)_\(imageURL.lastPathComponent)")

        guard let destination = CGImageDestinationCreateWithURL(
            tempURL as CFURL, uti, 1, nil
        ) else {
            throw MetadataWriterError.embeddingFailed
        }

        let options: [CFString: Any] = [
            kCGImageDestinationMetadata: metadata,
            kCGImageDestinationMergeMetadata: true
        ]
        var copyError: Unmanaged<CFError>?
        guard CGImageDestinationCopyImageSource(destination, source, options as CFDictionary, &copyError) else {
            try? FileManager.default.removeItem(at: tempURL)
            throw MetadataWriterError.embeddingFailed
        }

        do {
            _ = try FileManager.default.replaceItemAt(imageURL, withItemAt: tempURL)
        } catch {
            try? FileManager.default.removeItem(at: tempURL)
            throw MetadataWriterError.writeFailed(error)
        }
    }

    /// Builds a valid XMP document for sidecar use (RAW files).
    private static func xmpPacket(rating: Int, label: String?) -> XMPPacket {
        var children = "         <xmp:Rating>\(rating)</xmp:Rating>\n"
        if let label { children += "         <xmp:Label>\(label)</xmp:Label>\n" }
        let body = """
        <x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="ImageRater">
           <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
              <rdf:Description rdf:about=""
                    xmlns:xmp="http://ns.adobe.com/xap/1.0/">
        \(children.dropLast())
              </rdf:Description>
           </rdf:RDF>
        </x:xmpmeta>
        """
        return XMPPacket(body)
    }
}

// MARK: - XMPPacket

private struct XMPPacket {
    let body: String
    init(_ body: String) { self.body = body }

    func write(to url: URL) throws {
        guard let data = body.data(using: .utf8) else {
            throw MetadataWriterError.serializationFailed
        }
        do {
            try data.write(to: url, options: .atomic)
        } catch {
            throw MetadataWriterError.writeFailed(error)
        }
    }
}
