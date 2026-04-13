import CoreGraphics
import Foundation
import ImageIO

enum LibRawWrapper {
    /// Decode image at URL to CGImage. Returns nil on failure.
    /// For RAW files: tries embedded JPEG preview first, falls back to full LibRaw decode.
    /// For standard image files (JPEG, PNG, etc.): uses ImageIO directly.
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
