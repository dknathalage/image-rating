import CoreML
import Foundation

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
    func prepareModels(progress: @escaping (String) -> Void) async throws {
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
                let tmp = try await ModelDownloader.download(from: entry.url, expectedSHA256: entry.sha256)
                do {
                    try FileManager.default.moveItem(at: tmp, to: dest)
                } catch {
                    // Cross-volume fallback (Application Support on a different volume)
                    try FileManager.default.copyItem(at: tmp, to: dest)
                    try? FileManager.default.removeItem(at: tmp)
                }
                // Write sidecar so next launch skips re-download
                let sidecar = dest.appendingPathExtension("sha256")
                try? entry.sha256.write(to: sidecar, atomically: true, encoding: .utf8)
                progress("\(entry.name) ready.")
            }
        }
    }

    /// Load and cache a model by name. Throws if model file not found.
    func model(named name: String) throws -> MLModel {
        if let cached = loadedModels[name] { return cached }
        guard let url = try? FileManager.default.contentsOfDirectory(
            at: modelsDir, includingPropertiesForKeys: nil
        ).first(where: { $0.lastPathComponent.hasPrefix(name) }) else {
            throw ModelStoreError.modelNotFound(name)
        }
        let config = MLModelConfiguration()
        config.computeUnits = isAppleSilicon ? .cpuAndNeuralEngine : .cpuOnly
        let mlModel = try MLModel(contentsOf: url, configuration: config)
        loadedModels[name] = mlModel
        return mlModel
    }

    /// Copy a local .mlpackage into the models directory under the given name.
    /// Replaces any existing model with the same name. Invalidates cache.
    func importModel(from url: URL, name: String) throws {
        let dest = modelsDir.appendingPathComponent("\(name)-local.mlpackage")
        if FileManager.default.fileExists(atPath: dest.path) {
            try FileManager.default.removeItem(at: dest)
        }
        try FileManager.default.copyItem(at: url, to: dest)
        // Write "local" sentinel so prepareModels never overwrites a manually imported model
        let sidecar = dest.appendingPathExtension("sha256")
        try? "local".write(to: sidecar, atomically: true, encoding: .utf8)
        loadedModels.removeValue(forKey: name)
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

    private var isAppleSilicon: Bool {
        var sysinfo = utsname()
        uname(&sysinfo)
        return withUnsafeBytes(of: &sysinfo.machine) { ptr in
            String(cString: ptr.bindMemory(to: CChar.self).baseAddress!)
        }.hasPrefix("arm")
    }
}
