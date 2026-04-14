import CoreImage
import AppKit
import CryptoKit
import Foundation

actor ThumbnailCache {
    static let shared = ThumbnailCache()

    private let memCache = NSCache<NSString, NSImage>()
    private let diskCacheURL: URL
    // Track all sizes ever requested so invalidate() clears every variant.
    private var knownSizes: [CGSize] = []

    init() {
        let caches = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
        diskCacheURL = caches.appendingPathComponent("ImageRater/thumbnails")
        try? FileManager.default.createDirectory(at: diskCacheURL, withIntermediateDirectories: true)
        memCache.countLimit = 300
        memCache.totalCostLimit = 150 * 1024 * 1024 // 150 MB — NSCache evicts under pressure
    }

    /// Returns thumbnail for the given URL at requested size.
    func thumbnail(for url: URL, size: CGSize = CGSize(width: 200, height: 200)) async -> NSImage? {
        let key = Self.cacheKey(for: url, size: size)
        let diskURL = diskCacheURL.appendingPathComponent(key + ".jpg")
        if !knownSizes.contains(size) { knownSizes.append(size) }

        // Memory hit — fast, no I/O
        if let cached = memCache.object(forKey: key as NSString) { return cached }

        // Offload all blocking I/O (disk read, RAW decode, disk write) off actor executor.
        // Throttled via decodeQueue to prevent concurrent dyld lock contention during LibRaw / ImageIO init.
        let nsImage = await withCheckedContinuation { (continuation: CheckedContinuation<NSImage?, Never>) in
            Self.decodeQueue.addOperation {
                // Disk hit
                if let data = try? Data(contentsOf: diskURL), let img = NSImage(data: data) {
                    continuation.resume(returning: img)
                    return
                }
                // Generate: decode + write disk cache
                guard let cgImage = ThumbnailCache.decodeThumbnail(url: url, size: size) else {
                    continuation.resume(returning: nil)
                    return
                }
                let actualSize = CGSize(width: cgImage.width, height: cgImage.height)
                let nsImage = NSImage(cgImage: cgImage, size: actualSize)
                let rep = NSBitmapImageRep(cgImage: cgImage)
                if let jpegData = rep.representation(using: .jpeg, properties: [.compressionFactor: 0.8]) {
                    try? jpegData.write(to: diskURL)
                }
                continuation.resume(returning: nsImage)
            }
        }

        guard let nsImage else { return nil }
        // Don't mem-cache large detail-view thumbnails — they bloat memory fast
        if size.width <= 400 {
            let cost = Int(size.width * size.height) * 4
            memCache.setObject(nsImage, forKey: key as NSString, cost: cost)
        }
        return nsImage
    }

    /// Prefetch up to `limit` uncached thumbnails in the background.
    /// Intentionally capped — prefetching everything in a large session causes huge memory spikes.
    func prefetch(urls: [URL], size: CGSize, limit: Int = 60) {
        var queued = 0
        for url in urls {
            guard queued < limit else { break }
            let key = Self.cacheKey(for: url, size: size)
            if memCache.object(forKey: key as NSString) != nil { continue }
            let diskURL = diskCacheURL.appendingPathComponent(key + ".jpg")
            if FileManager.default.fileExists(atPath: diskURL.path) { continue }
            let op = BlockOperation {
                guard let cgImage = ThumbnailCache.decodeThumbnail(url: url, size: size) else { return }
                let rep = NSBitmapImageRep(cgImage: cgImage)
                if let jpegData = rep.representation(using: .jpeg, properties: [.compressionFactor: 0.8]) {
                    try? jpegData.write(to: diskURL)
                }
            }
            op.queuePriority = .low
            Self.decodeQueue.addOperation(op)
            queued += 1
        }
    }

    /// Remove all cached variants of a URL (call after user changes rating to force refresh).
    func invalidate(for url: URL) {
        for size in knownSizes {
            let key = Self.cacheKey(for: url, size: size)
            memCache.removeObject(forKey: key as NSString)
            let diskURL = diskCacheURL.appendingPathComponent(key + ".jpg")
            try? FileManager.default.removeItem(at: diskURL)
        }
    }

    // Throttle concurrent RAW/ImageIO decodes — LibRaw holds ~50–300 MB per decode.
    // 4 concurrent = up to ~1.2 GB peak during prefetch; safe headroom on 16 GB machines.
    private static let decodeQueue: OperationQueue = {
        let q = OperationQueue()
        q.maxConcurrentOperationCount = 4
        q.qualityOfService = .userInitiated
        return q
    }()

    // MARK: Private — static so they can be called from detached Tasks without actor hop

    private static func decodeThumbnail(url: URL, size: CGSize) -> CGImage? {
        let ext = url.pathExtension.lowercased()
        if LibRawWrapper.supportedExtensions.contains(ext),
           let img = LibRawWrapper.preview(url: url) {
            // Fast path: LibRaw embedded JPEG extraction succeeded.
            return resize(img, to: size)
        }
        // ImageIO fallback: handles standard formats AND embedded previews in RAW files
        // that LibRaw can't decode (e.g. Fujifilm X-H2 RA21 compressed variant).
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
              let cgImage = CGImageSourceCreateImageAtIndex(source, 0, [
                  kCGImageSourceThumbnailMaxPixelSize: max(size.width, size.height),
                  kCGImageSourceCreateThumbnailFromImageAlways: true,
                  kCGImageSourceCreateThumbnailWithTransform: true
              ] as CFDictionary) else { return nil }
        return cgImage
    }

    private static func resize(_ image: CGImage, to maxSize: CGSize) -> CGImage? {
        let imgW = CGFloat(image.width), imgH = CGFloat(image.height)
        guard imgW > 0, imgH > 0 else { return nil }
        let ratio = min(maxSize.width / imgW, maxSize.height / imgH)
        let targetW = Int(imgW * ratio), targetH = Int(imgH * ratio)
        guard let ctx = CGContext(data: nil,
                                  width: targetW, height: targetH,
                                  bitsPerComponent: 8, bytesPerRow: 0,
                                  space: CGColorSpaceCreateDeviceRGB(),
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return nil }
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: targetW, height: targetH))
        return ctx.makeImage()
    }

    private static func cacheKey(for url: URL, size: CGSize) -> String {
        let path = url.path
        let attrs = try? FileManager.default.attributesOfItem(atPath: path)
        let mtime = (attrs?[.modificationDate] as? Date)?.timeIntervalSince1970 ?? 0
        let fileSize = (attrs?[.size] as? Int) ?? 0
        let raw = "\(path)_\(Int(size.width))x\(Int(size.height))_\(Int(mtime))_\(fileSize)"
        let digest = SHA256.hash(data: Data(raw.utf8))
        return digest.map { String(format: "%02x", $0) }.joined()
    }
}
