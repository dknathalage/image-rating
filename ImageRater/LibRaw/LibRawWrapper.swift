import CoreGraphics
import Foundation
import ImageIO

enum LibRawWrapper {
    /// Embedded JPEG preview only — fast, no full RAW decode. Use for thumbnails.
    /// Returns nil if no embedded preview (rare on modern cameras).
    static func preview(url: URL) -> CGImage? {
        guard url.isFileURL, supportedExtensions.contains(url.pathExtension.lowercased()) else { return nil }
        return LibRawBridge.preview(atPath: url.path)
    }

    /// Full decode: embedded preview first, falls back to full LibRaw decode.
    /// Use for processing pipeline only — expensive for RAW files.
    static func decode(url: URL) -> CGImage? {
        guard url.isFileURL else { return nil }
        let ext = url.pathExtension.lowercased()
        if supportedExtensions.contains(ext) {
            return LibRawBridge.decodeFile(atPath: url.path)
        }
        // Fall back to ImageIO for standard formats (JPEG, PNG, HEIC, etc.)
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
              let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            return nil
        }
        return image
    }

    static let supportedExtensions: Set<String> = [
        "cr2", "cr3", "nef", "arw", "raf", "rw2", "dng", "orf", "pef", "srw",
        "3fr", "dcr", "erf", "mef", "mrw", "nrw", "ptx", "r3d", "rwl", "x3f"
    ]
}
