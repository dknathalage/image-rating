import CryptoKit
import Foundation

enum ModelStoreError: Error {
    case checksumMismatch
    case downloadFailed
    case manifestVerificationFailed
    case modelNotFound(String)
    case unzipFailed
}

enum ModelDownloader {

    /// SHA-256 of in-memory data. Used for small payloads (manifests, test data).
    static func sha256(_ data: Data) -> String {
        let digest = SHA256.hash(data: data)
        return digest.map { String(format: "%02x", $0) }.joined()
    }

    /// SHA-256 of in-memory data vs expected hash.
    static func verify(data: Data, expectedSHA256: String) -> Bool {
        sha256(data) == expectedSHA256
    }

    /// Streaming SHA-256 of a file. Reads in 4 MB chunks — safe for multi-GB model files.
    static func sha256OfFile(at url: URL) throws -> String {
        let fileHandle = try FileHandle(forReadingFrom: url)
        defer { try? fileHandle.close() }
        var hasher = SHA256()
        let chunkSize = 4 * 1024 * 1024
        while true {
            let chunk = fileHandle.readData(ofLength: chunkSize)
            guard !chunk.isEmpty else { break }
            hasher.update(data: chunk)
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }

    /// Verify a file on disk by streaming SHA-256.
    static func verify(fileAt url: URL, expectedSHA256: String) throws -> Bool {
        try sha256OfFile(at: url) == expectedSHA256
    }

    /// Download from URL, verify SHA-256, return temp file URL. Retries 3× with exponential backoff.
    static func download(from url: URL, expectedSHA256: String) async throws -> URL {
        var lastError: Error = ModelStoreError.downloadFailed
        for attempt in 0..<4 {
            if attempt > 0 {
                let delay = UInt64(pow(2.0, Double(attempt))) * 1_000_000_000
                try await Task.sleep(nanoseconds: delay)
            }
            do {
                let (zipURL, _) = try await URLSession.shared.download(from: url)
                guard (try? verify(fileAt: zipURL, expectedSHA256: expectedSHA256)) == true else {
                    throw ModelStoreError.checksumMismatch
                }
                let unzipDir = FileManager.default.temporaryDirectory
                    .appendingPathComponent(UUID().uuidString)
                do {
                    let pkgURL = try unzip(zipURL, to: unzipDir)
                    try? FileManager.default.removeItem(at: zipURL)
                    return pkgURL
                } catch {
                    try? FileManager.default.removeItem(at: unzipDir)
                    throw error
                }
            } catch {
                lastError = error
                if case ModelStoreError.checksumMismatch = error { throw error } // don't retry bad checksum
                if case ModelStoreError.unzipFailed = error { throw error }
            }
        }
        throw lastError
    }

    /// Extract a zip archive to `destDir`. Returns URL of the extracted `.mlpackage` directory.
    /// The zip must contain exactly one root-level `.mlpackage` directory (no path prefix).
    static func unzip(_ zipURL: URL, to destDir: URL) throws -> URL {
        try FileManager.default.createDirectory(at: destDir, withIntermediateDirectories: true)
        let process = Process()
        process.executableURL = URL(filePath: "/usr/bin/unzip")
        process.arguments = ["-q", zipURL.path, "-d", destDir.path]
        try process.run()
        process.waitUntilExit()
        guard process.terminationStatus == 0 else {
            throw ModelStoreError.unzipFailed
        }
        let contents = try FileManager.default.contentsOfDirectory(
            at: destDir, includingPropertiesForKeys: nil
        )
        guard let pkg = contents.first(where: { $0.pathExtension == "mlpackage" }) else {
            try? FileManager.default.removeItem(at: destDir)
            throw ModelStoreError.unzipFailed
        }
        return pkg
    }
}
