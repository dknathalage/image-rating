import CryptoKit
import Foundation

enum ModelStoreError: Error {
    case checksumMismatch
    case downloadFailed
    case manifestVerificationFailed
    case modelNotFound(String)
}

enum ModelDownloader {

    static func sha256(_ data: Data) -> String {
        let digest = SHA256.hash(data: data)
        return digest.map { String(format: "%02x", $0) }.joined()
    }

    static func verify(data: Data, expectedSHA256: String) -> Bool {
        sha256(data) == expectedSHA256
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
                let (tmpURL, _) = try await URLSession.shared.download(from: url)
                let data = try Data(contentsOf: tmpURL)
                guard verify(data: data, expectedSHA256: expectedSHA256) else {
                    throw ModelStoreError.checksumMismatch
                }
                return tmpURL
            } catch {
                lastError = error
                if case ModelStoreError.checksumMismatch = error { throw error } // don't retry bad checksum
            }
        }
        throw lastError
    }
}
