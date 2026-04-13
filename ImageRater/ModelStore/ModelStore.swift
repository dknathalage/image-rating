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

    /// Ensure all required models are present and verified. Downloads if missing.
    func prepareModels(progress: @escaping (String) -> Void) async throws {
        let manifest = try await ManifestFetcher.fetch()
        for entry in manifest.models {
            let dest = modelsDir.appendingPathComponent("\(entry.name)-\(entry.version).mlpackage")
            if !FileManager.default.fileExists(atPath: dest.path) {
                progress("Downloading \(entry.name)…")
                let tmp = try await ModelDownloader.download(from: entry.url,
                                                              expectedSHA256: entry.sha256)
                try FileManager.default.moveItem(at: tmp, to: dest)
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

    // MARK: Private

    private var isAppleSilicon: Bool {
        var sysinfo = utsname()
        uname(&sysinfo)
        return withUnsafeBytes(of: &sysinfo.machine) { ptr in
            String(cString: ptr.bindMemory(to: CChar.self).baseAddress!)
        }.hasPrefix("arm")
    }
}
