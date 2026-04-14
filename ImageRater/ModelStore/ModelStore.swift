import CoreML
import Foundation
import OSLog

private let log = Logger(subsystem: "com.focal.app", category: "ModelStore")

actor ModelStore {
    static let shared = ModelStore()

    private let modelsDir: URL
    private var loadedModels: [String: MLModel] = [:]

    init() {
        let appSupport = FileManager.default.urls(for: .applicationSupportDirectory,
                                                   in: .userDomainMask)[0]
        modelsDir = appSupport.appendingPathComponent("ImageRater/models")
        try? FileManager.default.createDirectory(at: modelsDir, withIntermediateDirectories: true)
    }

    /// Ensure all required models are present and checksum-verified. Re-downloads if missing or corrupt.
    func prepareModels(
        progress: @escaping @Sendable (String) -> Void,
        downloadProgress: (@Sendable (Double) -> Void)? = nil
    ) async throws {
        let manifest = try await ManifestFetcher.fetch()
        for entry in manifest.models {
            let dest = modelsDir.appendingPathComponent("\(entry.name)-\(entry.version).mlpackage")
            let needsDownload: Bool
            if FileManager.default.fileExists(atPath: dest.path) {
                // Verify via sidecar (.sha256 file stores the zip's SHA-256 on install).
                // "local" sentinel = manually imported model, never re-download.
                let sidecar = dest.appendingPathExtension("sha256")
                let raw = try? String(contentsOf: sidecar, encoding: .utf8)
                let stored = raw?.trimmingCharacters(in: .whitespacesAndNewlines)
                let valid = stored == entry.sha256 || stored == "local"
                if valid {
                    needsDownload = false
                } else {
                    progress("\(entry.name) sidecar missing or mismatched — re-downloading…")
                    try? FileManager.default.removeItem(at: dest)
                    try? FileManager.default.removeItem(at: sidecar)
                    needsDownload = true
                }
            } else {
                // Also check for a locally-imported model (name-local.mlpackage).
                // If the user imported a model manually, don't auto-download the versioned one.
                let localDest = modelsDir.appendingPathComponent("\(entry.name)-local.mlpackage")
                let localSidecar = localDest.appendingPathExtension("sha256")
                let localRaw = try? String(contentsOf: localSidecar, encoding: .utf8)
                let localStored = localRaw?.trimmingCharacters(in: .whitespacesAndNewlines)
                if localStored == "local" {
                    needsDownload = false   // local import serves this model slot
                } else {
                    needsDownload = true
                }
            }
            if needsDownload {
                progress("Downloading \(entry.name)…")
                let tmp = try await ModelDownloader.download(from: entry.url, expectedSHA256: entry.sha256, onProgress: downloadProgress)
                do {
                    try FileManager.default.moveItem(at: tmp, to: dest)
                } catch {
                    // Cross-volume fallback (Application Support on a different volume)
                    try FileManager.default.copyItem(at: tmp, to: dest)
                    try? FileManager.default.removeItem(at: tmp)
                }
                // Write sidecar so next launch skips re-download
                let sidecar = dest.appendingPathExtension("sha256")
                try entry.sha256.write(to: sidecar, atomically: true, encoding: .utf8)
                progress("\(entry.name) ready.")
            }
        }
    }

    /// Load and cache a model by name. Throws if model file not found.
    func model(named name: String) throws -> MLModel {
        if let cached = loadedModels[name] { return cached }
        guard let url = try? FileManager.default.contentsOfDirectory(
            at: modelsDir, includingPropertiesForKeys: nil
        ).first(where: { $0.pathExtension == "mlpackage" && $0.lastPathComponent.hasPrefix(name) }) else {
            throw ModelStoreError.modelNotFound(name)
        }
        // .mlpackage must be compiled before loading; cache the .mlmodelc next to the package
        let compiledURL = url.deletingPathExtension().appendingPathExtension("mlmodelc")
        if !FileManager.default.fileExists(atPath: compiledURL.path) {
            log.info("Compiling \(name) model — first run only")
            let tmp = try MLModel.compileModel(at: url)
            try FileManager.default.moveItem(at: tmp, to: compiledURL)
            log.info("Compiled \(name) → \(compiledURL.lastPathComponent)")
        } else {
            log.debug("Using cached compiled model: \(compiledURL.lastPathComponent)")
        }
        log.info("Loading \(name) model")
        let mlModel = try loadModel(at: compiledURL)
        log.info("Loaded \(name) model successfully")
        loadedModels[name] = mlModel
        return mlModel
    }

    /// Copy a local .mlpackage into the models directory under the given name.
    /// Replaces any existing model with the same name. Invalidates cache.
    func importModel(from url: URL, name: String) async throws {
        let dest = modelsDir.appendingPathComponent("\(name)-local.mlpackage")
        // Run blocking file operations off the actor executor to avoid freezing model access
        try await Task.detached(priority: .userInitiated) { [dest] in
            if FileManager.default.fileExists(atPath: dest.path) {
                try FileManager.default.removeItem(at: dest)
            }
            try FileManager.default.copyItem(at: url, to: dest)
            let sidecar = dest.appendingPathExtension("sha256")
            try "local".write(to: sidecar, atomically: true, encoding: .utf8)
        }.value
        loadedModels.removeValue(forKey: name)
        // Invalidate compiled cache so it's recompiled from the new package
        let compiledURL = dest.deletingPathExtension().appendingPathExtension("mlmodelc")
        try? FileManager.default.removeItem(at: compiledURL)
    }

    /// Returns display names of all .mlpackage files in the models directory.
    func installedModelNames() -> [String] {
        let contents = (try? FileManager.default.contentsOfDirectory(
            at: modelsDir, includingPropertiesForKeys: nil
        )) ?? []
        return contents
            .filter { $0.pathExtension == "mlpackage" }
            .map { $0.deletingPathExtension().lastPathComponent }
            .sorted()
    }

    // MARK: Private

    /// Try preferred compute units; fall back to cpuOnly if exec-plan build fails (error -14).
    private func loadModel(at url: URL) throws -> MLModel {
        let preferred = MLModelConfiguration()
        preferred.computeUnits = isAppleSilicon ? .all : .cpuOnly
        log.debug("Attempting load with computeUnits=\(self.isAppleSilicon ? "all" : "cpuOnly")")
        do {
            return try MLModel(contentsOf: url, configuration: preferred)
        } catch let err as NSError where isAppleSilicon && err.code == -14 {
            log.warning("Exec-plan build failed (code -14) — falling back to cpuOnly")
            let fallback = MLModelConfiguration()
            fallback.computeUnits = .cpuOnly
            return try MLModel(contentsOf: url, configuration: fallback)
        }
    }

    private var isAppleSilicon: Bool {
        var sysinfo = utsname()
        uname(&sysinfo)
        return withUnsafeBytes(of: &sysinfo.machine) { ptr in
            String(cString: ptr.bindMemory(to: CChar.self).baseAddress!)
        }.hasPrefix("arm")
    }
}
