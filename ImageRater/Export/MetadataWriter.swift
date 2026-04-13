import CoreImage
import Foundation

enum MetadataWriterError: Error {
    case serializationFailed
    case writeFailed(Error)
    case readFailed
}

enum MetadataWriter {

    /// Write xmp:Rating (0–5) to a .xmp sidecar file at `url`.
    static func write(stars: Int, to url: URL) throws {
        guard (0...5).contains(stars) else { throw MetadataWriterError.serializationFailed }
        try xmpPacket(rating: stars, label: nil).write(to: url)
    }

    /// Writes sidecar next to the source image file (replaces extension with .xmp).
    static func writeSidecar(stars: Int, for imageURL: URL) throws {
        try write(stars: stars, to: sidecarURL(for: imageURL))
    }

    /// Write xmp:Rating = -1 and xmp:Label = "Rejected".
    /// Photomator, Lightroom, and Capture One all honour xmp:Rating = -1 as "rejected".
    static func writeRejected(to url: URL) throws {
        try xmpPacket(rating: -1, label: "Rejected").write(to: url)
    }

    /// Writes a rejection sidecar next to the source image file.
    static func writeSidecarRejected(for imageURL: URL) throws {
        try writeRejected(to: sidecarURL(for: imageURL))
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

    private static func sidecarURL(for imageURL: URL) -> URL {
        imageURL.deletingPathExtension().appendingPathExtension("xmp")
    }

    /// Builds a valid XMP document parseable by CGImageMetadataCreateFromXMPData.
    /// Note: xpacket processing instructions (<?xpacket ...?>) break CGImageMetadata parsing
    /// on macOS — omit the wrapper so tests and in-process reads work correctly.
    /// External tools (Lightroom, Photomator) accept both forms.
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
