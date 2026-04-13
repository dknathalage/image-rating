import CoreImage
import AppKit
import Foundation

actor ThumbnailCache {
    static let shared = ThumbnailCache()

    private let memCache = NSCache<NSString, NSImage>()
    private let diskCacheURL: URL

    init() {
        let caches = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
        diskCacheURL = caches.appendingPathComponent("ImageRater/thumbnails")
        try? FileManager.default.createDirectory(at: diskCacheURL, withIntermediateDirectories: true)
        memCache.countLimit = 500
        memCache.totalCostLimit = 200 * 1024 * 1024 // 200 MB
    }

    /// Returns thumbnail for the given URL at requested size. Checks memory, then disk, then generates.
    func thumbnail(for url: URL, size: CGSize = CGSize(width: 200, height: 200)) async -> NSImage? {
        let key = cacheKey(for: url, size: size)

        // Memory hit
        if let cached = memCache.object(forKey: key as NSString) { return cached }

        // Disk hit
        let diskURL = diskCacheURL.appendingPathComponent(key + ".jpg")
        if let data = try? Data(contentsOf: diskURL), let img = NSImage(data: data) {
            memCache.setObject(img, forKey: key as NSString)
            return img
        }

        // Generate
        guard let cgImage = decodeThumbnail(url: url, size: size) else { return nil }
        let nsImage = NSImage(cgImage: cgImage, size: size)

        // Write to disk cache
        let rep = NSBitmapImageRep(cgImage: cgImage)
        if let jpegData = rep.representation(using: .jpeg, properties: [.compressionFactor: 0.8]) {
            try? jpegData.write(to: diskURL)
        }
        memCache.setObject(nsImage, forKey: key as NSString)
        return nsImage
    }

    /// Remove cached thumbnail for a URL (call after user changes rating to force refresh).
    func invalidate(for url: URL) {
        let key = cacheKey(for: url, size: CGSize(width: 200, height: 200))
        memCache.removeObject(forKey: key as NSString)
        let diskURL = diskCacheURL.appendingPathComponent(key + ".jpg")
        try? FileManager.default.removeItem(at: diskURL)
    }

    // MARK: Private

    private func decodeThumbnail(url: URL, size: CGSize) -> CGImage? {
        let ext = url.pathExtension.lowercased()
        if LibRawWrapper.supportedExtensions.contains(ext) {
            return LibRawWrapper.decode(url: url).flatMap { resize($0, to: size) }
        }
        // ImageIO path for JPEG/PNG/HEIC etc.
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
              let cgImage = CGImageSourceCreateImageAtIndex(source, 0, [
                  kCGImageSourceThumbnailMaxPixelSize: max(size.width, size.height),
                  kCGImageSourceCreateThumbnailFromImageAlways: true,
                  kCGImageSourceCreateThumbnailWithTransform: true
              ] as CFDictionary) else { return nil }
        return cgImage
    }

    private func resize(_ image: CGImage, to size: CGSize) -> CGImage? {
        guard let ctx = CGContext(data: nil,
                                  width: Int(size.width), height: Int(size.height),
                                  bitsPerComponent: 8, bytesPerRow: 0,
                                  space: CGColorSpaceCreateDeviceRGB(),
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return nil }
        ctx.draw(image, in: CGRect(origin: .zero, size: size))
        return ctx.makeImage()
    }

    private func cacheKey(for url: URL, size: CGSize) -> String {
        let path = url.path
        let attrs = try? FileManager.default.attributesOfItem(atPath: path)
        let mtime = (attrs?[.modificationDate] as? Date)?.timeIntervalSince1970 ?? 0
        // Use hash of path + mtime + size as key; sanitize slashes for use as filename
        let raw = "\(path.hashValue)_\(Int(size.width))x\(Int(size.height))_\(Int(mtime))"
        return raw.replacingOccurrences(of: "/", with: "_")
            .replacingOccurrences(of: ":", with: "_")
    }
}
