import CoreImage
import AppKit
import CryptoKit
import Foundation

/// Thread-safe box used to hand a BlockOperation reference across async/actor boundaries
/// so withTaskCancellationHandler can cancel it if the Swift task is cancelled.
private final class OperationHolder: @unchecked Sendable {
    var operation: BlockOperation?
}

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
        memCache.totalCostLimit = 300 * 1024 * 1024 // 300 MB — fits ~18 detail images + grid thumbs
    }

    /// Returns thumbnail for the given URL at requested size.
    /// Three-tier load: memory → disk (fast queue) → full decode (throttled queue).
    /// `isPrefetch: true` routes the Tier-3 decode to prefetchQueue (isolated from decodeQueue)
    /// so background pre-fetches never block user-visible detail-view loads.
    func thumbnail(
        for url: URL,
        size: CGSize = CGSize(width: 200, height: 200),
        isPrefetch: Bool = false
    ) async -> NSImage? {
        let key = Self.cacheKey(for: url, size: size)
        let diskURL = diskCacheURL.appendingPathComponent(key + ".jpg")
        if !knownSizes.contains(size) { knownSizes.append(size) }

        // Tier 1: memory — no I/O
        if let cached = memCache.object(forKey: key as NSString) { return cached }

        // Tier 2: disk hit — lightweight read, uses high-concurrency diskReadQueue.
        // This is separate from decodeQueue so cached thumbnails don't compete with
        // heavy RAW decodes for queue slots.
        if let img = await Self.readFromDisk(diskURL) {
            memCache.setObject(img, forKey: key as NSString, cost: Int(size.width * size.height) * 4)
            return img
        }

        // Tier 3: full decode — throttled via dedicated queues.
        // isPrefetch uses prefetchQueue (isolated from decodeQueue) so background pre-fetches
        // never occupy the slots that user-visible detail-view loads need.
        let queue = isPrefetch ? Self.prefetchQueue : Self.decodeQueue
        let holder = OperationHolder()
        let decoded: NSImage? = await withTaskCancellationHandler {
            await withCheckedContinuation { (c: CheckedContinuation<NSImage?, Never>) in
                let op = BlockOperation()
                holder.operation = op
                op.addExecutionBlock {
                    var resumed = false
                    defer { if !resumed { c.resume(returning: nil) } }
                    if op.isCancelled { c.resume(returning: nil); resumed = true; return }
                    guard let cgImage = ThumbnailCache.decodeThumbnail(url: url, size: size) else {
                        c.resume(returning: nil); resumed = true; return
                    }
                    let actualSize = CGSize(width: cgImage.width, height: cgImage.height)
                    let img = NSImage(cgImage: cgImage, size: actualSize)
                    let rep = NSBitmapImageRep(cgImage: cgImage)
                    if let data = rep.representation(using: .jpeg, properties: [.compressionFactor: 0.8]) {
                        try? data.write(to: diskURL)
                    }
                    c.resume(returning: img)
                    resumed = true
                }
                queue.addOperation(op)
            }
        } onCancel: {
            holder.operation?.cancel()
        }

        guard let decoded else { return nil }
        memCache.setObject(decoded, forKey: key as NSString, cost: Int(size.width * size.height) * 4)
        return decoded
    }

    /// Returns the largest in-memory cached image for `url` with no disk I/O.
    /// Use as an instant placeholder before the async decode completes.
    func bestAvailableCached(for url: URL) -> NSImage? {
        for size in knownSizes.sorted(by: { $0.width > $1.width }) {
            let key = Self.cacheKey(for: url, size: size)
            if let img = memCache.object(forKey: key as NSString) { return img }
        }
        return nil
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

    // MARK: - Private

    /// Read a JPEG from the disk cache. Runs on diskReadQueue (cancellable).
    private static func readFromDisk(_ url: URL) async -> NSImage? {
        let holder = OperationHolder()
        return await withTaskCancellationHandler {
            await withCheckedContinuation { (c: CheckedContinuation<NSImage?, Never>) in
                let op = BlockOperation()
                holder.operation = op
                op.addExecutionBlock {
                    var resumed = false
                    defer { if !resumed { c.resume(returning: nil) } }
                    if op.isCancelled { c.resume(returning: nil); resumed = true; return }
                    guard let data = try? Data(contentsOf: url),
                          let img = NSImage(data: data) else {
                        c.resume(returning: nil); resumed = true; return
                    }
                    c.resume(returning: img)
                    resumed = true
                }
                Self.diskReadQueue.addOperation(op)
            }
        } onCancel: {
            holder.operation?.cancel()
        }
    }

    // Disk reads: many concurrent OK — just file I/O, no large memory allocations.
    private static let diskReadQueue: OperationQueue = {
        let q = OperationQueue()
        q.maxConcurrentOperationCount = 8
        q.qualityOfService = .userInitiated
        return q
    }()

    // User-visible decodes (detail view): throttled to 2 concurrent.
    // LibRaw extracts a full-res JPEG preview (~96 MB) before resize; 2 concurrent
    // keeps peak IOSurface usage below the system limit on 16 GB machines.
    private static let decodeQueue: OperationQueue = {
        let q = OperationQueue()
        q.maxConcurrentOperationCount = 2
        q.qualityOfService = .userInitiated
        return q
    }()

    // Background pre-fetch decodes: fully isolated from decodeQueue so pre-fetch
    // ops never occupy slots that the detail view needs.
    private static let prefetchQueue: OperationQueue = {
        let q = OperationQueue()
        q.maxConcurrentOperationCount = 1
        q.qualityOfService = .background
        return q
    }()

    // MARK: - Static helpers (no actor hop needed)

    private static func decodeThumbnail(url: URL, size: CGSize) -> CGImage? {
        let ext = url.pathExtension.lowercased()
        if LibRawWrapper.supportedExtensions.contains(ext),
           let img = LibRawWrapper.preview(url: url) {
            return resize(img, to: size)
        }
        // IMPORTANT: use CGImageSourceCreateThumbnailAtIndex, NOT CGImageSourceCreateImageAtIndex.
        // The kCGImageSourceThumbnail* options are silently ignored by CreateImageAtIndex —
        // it returns a full-resolution image, causing IOSurface OOM when displayed in the grid.
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
              let cgImage = CGImageSourceCreateThumbnailAtIndex(source, 0, [
                  kCGImageSourceThumbnailMaxPixelSize: max(size.width, size.height),
                  kCGImageSourceCreateThumbnailFromImageAlways: true,
                  kCGImageSourceCreateThumbnailWithTransform: true
              ] as CFDictionary) else { return nil }
        // Resize as a safety net — ImageIO may return slightly different dimensions.
        return resize(cgImage, to: size) ?? cgImage
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
