import CoreGraphics
import Foundation
import ImageIO

struct ImageMetadata {
    var pixelWidth: Int?
    var pixelHeight: Int?
    var fileSize: Int64?
    var cameraMake: String?
    var cameraModel: String?
    var lens: String?
    var focalLength: Double?
    var focalLength35mm: Double?
    var aperture: Double?
    var shutterSpeed: Double?
    var iso: Int?
    var exposureBias: Double?
    var dateTaken: Date?
    var colorSpace: String?
    var whiteBalance: String?
    var flash: String?
    var meteringMode: String?
    var exposureProgram: String?

    // MARK: Formatted helpers

    var dimensionsString: String? {
        guard let w = pixelWidth, let h = pixelHeight else { return nil }
        return "\(w) × \(h)"
    }

    var fileSizeString: String? {
        guard let bytes = fileSize else { return nil }
        let mb = Double(bytes) / 1_048_576
        return mb >= 1 ? String(format: "%.1f MB", mb) : String(format: "%d KB", bytes / 1024)
    }

    var apertureString: String? {
        guard let f = aperture else { return nil }
        return String(format: "f/%.1f", f)
    }

    var shutterString: String? {
        guard let s = shutterSpeed, s > 0 else { return nil }
        if s >= 1 {
            return s == s.rounded() ? String(format: "%.0fs", s) : String(format: "%.1fs", s)
        }
        let denom = Int((1.0 / s).rounded())
        return "1/\(denom)s"
    }

    var isoString: String? {
        guard let iso else { return nil }
        return "ISO \(iso)"
    }

    var focalLengthString: String? {
        guard let fl = focalLength else { return nil }
        let base = fl == fl.rounded() ? String(format: "%.0fmm", fl) : String(format: "%.1fmm", fl)
        if let eq = focalLength35mm, eq > 0, Int(eq) != Int(fl) {
            return "\(base) (\(Int(eq))mm eq)"
        }
        return base
    }

    var exposureBiasString: String? {
        guard let ev = exposureBias else { return nil }
        if ev == 0 { return "0 EV" }
        return ev > 0 ? String(format: "+%.1f EV", ev) : String(format: "%.1f EV", ev)
    }

    var dateTakenString: String? {
        guard let d = dateTaken else { return nil }
        let fmt = DateFormatter()
        fmt.dateStyle = .medium
        fmt.timeStyle = .short
        return fmt.string(from: d)
    }

    // MARK: Read from URL

    static func read(from url: URL) -> ImageMetadata {
        var meta = ImageMetadata()

        let attrs = try? FileManager.default.attributesOfItem(atPath: url.path)
        if let sz = attrs?[.size] {
            meta.fileSize = (sz as? Int64) ?? Int64((sz as? Int) ?? 0)
        }

        guard let source = CGImageSourceCreateWithURL(url as CFURL, [
            kCGImageSourceShouldCache: false
        ] as CFDictionary),
              let props = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [String: Any] else {
            return meta
        }

        meta.pixelWidth = props[kCGImagePropertyPixelWidth as String] as? Int
        meta.pixelHeight = props[kCGImagePropertyPixelHeight as String] as? Int

        if let tiff = props[kCGImagePropertyTIFFDictionary as String] as? [String: Any] {
            meta.cameraMake = (tiff[kCGImagePropertyTIFFMake as String] as? String)?.trimmingCharacters(in: .whitespaces)
            meta.cameraModel = (tiff[kCGImagePropertyTIFFModel as String] as? String)?.trimmingCharacters(in: .whitespaces)
        }

        if let exif = props[kCGImagePropertyExifDictionary as String] as? [String: Any] {
            meta.aperture = exif[kCGImagePropertyExifFNumber as String] as? Double
            meta.shutterSpeed = exif[kCGImagePropertyExifExposureTime as String] as? Double
            meta.focalLength = exif[kCGImagePropertyExifFocalLength as String] as? Double
            meta.focalLength35mm = exif[kCGImagePropertyExifFocalLenIn35mmFilm as String] as? Double
            meta.lens = (exif[kCGImagePropertyExifLensModel as String] as? String)?.trimmingCharacters(in: .whitespaces)
            meta.exposureBias = exif[kCGImagePropertyExifExposureBiasValue as String] as? Double

            if let isos = exif[kCGImagePropertyExifISOSpeedRatings as String] as? [Int] {
                meta.iso = isos.first
            } else {
                meta.iso = exif[kCGImagePropertyExifISOSpeedRatings as String] as? Int
            }

            if let dateStr = exif[kCGImagePropertyExifDateTimeOriginal as String] as? String {
                let fmt = DateFormatter()
                fmt.dateFormat = "yyyy:MM:dd HH:mm:ss"
                meta.dateTaken = fmt.date(from: dateStr)
            }

            if let wb = exif[kCGImagePropertyExifWhiteBalance as String] as? Int {
                meta.whiteBalance = wb == 0 ? "Auto" : "Manual"
            }

            if let flash = exif[kCGImagePropertyExifFlash as String] as? Int {
                meta.flash = (flash & 0x1) != 0 ? "Fired" : "No flash"
            }

            if let mm = exif[kCGImagePropertyExifMeteringMode as String] as? Int {
                meta.meteringMode = meteringModeString(mm)
            }

            if let ep = exif[kCGImagePropertyExifExposureProgram as String] as? Int {
                meta.exposureProgram = exposureProgramString(ep)
            }
        }

        return meta
    }

    private static func meteringModeString(_ v: Int) -> String {
        switch v {
        case 1: return "Average"
        case 2: return "Center-weighted"
        case 3: return "Spot"
        case 4: return "Multi-spot"
        case 5: return "Multi-segment"
        case 6: return "Partial"
        default: return "Unknown"
        }
    }

    private static func exposureProgramString(_ v: Int) -> String {
        switch v {
        case 1: return "Manual"
        case 2: return "Program"
        case 3: return "Aperture priority"
        case 4: return "Shutter priority"
        case 5: return "Creative"
        case 6: return "Action"
        case 7: return "Portrait"
        case 8: return "Landscape"
        default: return "Unknown"
        }
    }
}
