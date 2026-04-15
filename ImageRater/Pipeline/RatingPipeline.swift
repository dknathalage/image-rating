// ImageRater/Pipeline/RatingPipeline.swift
import CoreML
import CoreGraphics
import Foundation
import OSLog

private let log = Logger(subsystem: "com.focal.app", category: "RatingPipeline")

enum RatingError: Error {
    case pixelBufferCreationFailed
    case modelNotFound(String)
    case inferenceOutputMismatch
    case imageTooSmall
}

/// Abstract MUSIQ inference so tests can inject mock scalars.
protocol MUSIQInferring {
    func predict(patchTensor: MLMultiArray) async throws -> Float
}

/// Concrete CoreML-backed MUSIQ inferrer.
struct CoreMLMUSIQ: MUSIQInferring {
    let model: MLModel

    func predict(patchTensor: MLMultiArray) async throws -> Float {
        let input = try MLDictionaryFeatureProvider(dictionary: ["patch_tensor": patchTensor])
        let out = try await model.prediction(from: input)
        for name in out.featureNames {
            if let arr = out.featureValue(for: name)?.multiArrayValue, arr.count > 0 {
                return arr[0].floatValue
            }
            if let d = out.featureValue(for: name)?.doubleValue {
                return Float(d)
            }
        }
        throw RatingError.inferenceOutputMismatch
    }
}

enum RatingPipeline {

    // MARK: - Model loading (call once before processing loop)

    struct BundledModels {
        let musiq: MUSIQInferring
    }

    /// Load MUSIQ model. Call once; pass result into rate() for every image.
    static func loadBundledModels() throws -> BundledModels {
        let config = MLModelConfiguration()
        config.computeUnits = isAppleSilicon ? .all : .cpuOnly
        let musiq = try loadBundledModel(named: "musiq-ava", configuration: config)
        return BundledModels(musiq: CoreMLMUSIQ(model: musiq))
    }

    // MARK: - Inference

    /// Rate a single image. Never throws — returns .unrated on failure.
    static func rate(
        image: CGImage,
        models: BundledModels
    ) async -> RatingResult {
        do {
            let tensor = try MUSIQPreprocessor.patchTensor(cgImage: image)
            let raw = try await models.musiq.predict(patchTensor: tensor)
            guard raw.isFinite else {
                log.warning("MUSIQ produced non-finite score \(raw); returning .unrated")
                return .unrated
            }
            let clamped = min(max(raw, 1.0), 10.0)
            if clamped != raw {
                log.warning("MUSIQ score \(raw) clamped to \(clamped)")
            }
            let stars = bucketStars(mos: clamped, thresholds: defaultThresholds())
            log.info("MUSIQ \(raw, format: .fixed(precision: 3)) → \(clamped, format: .fixed(precision: 3)) → \(stars)★")
            return .rated(RatedScores(musiqAesthetic: clamped, stars: stars))
        } catch {
            log.error("Rating failed: \(error)")
            return .unrated
        }
    }

    // MARK: - bucketStars

    /// Map MUSIQ MOS into 1..5 stars using 4 thresholds.
    /// mos <= t.0 → 1; <= t.1 → 2; <= t.2 → 3; <= t.3 → 4; else 5.
    static func bucketStars(mos: Float, thresholds t: (Float, Float, Float, Float)) -> Int {
        if mos <= t.0 { return 1 }
        if mos <= t.1 { return 2 }
        if mos <= t.2 { return 3 }
        if mos <= t.3 { return 4 }
        return 5
    }

    /// Read the current MUSIQ thresholds from FocalSettings.
    static func defaultThresholds() -> (Float, Float, Float, Float) {
        (
            FocalSettings.defaultMUSIQThreshold1,
            FocalSettings.defaultMUSIQThreshold2,
            FocalSettings.defaultMUSIQThreshold3,
            FocalSettings.defaultMUSIQThreshold4
        )
    }

    // MARK: - Private

    private static func loadBundledModel(named name: String, configuration: MLModelConfiguration) throws -> MLModel {
        // Xcode compiles .mlpackage → .mlmodelc at build time
        if let url = Bundle.main.url(forResource: name, withExtension: "mlmodelc") {
            return try MLModel(contentsOf: url, configuration: configuration)
        }
        // CLI tool fallback: env var override for ad-hoc model dir
        if let envDir = ProcessInfo.processInfo.environment["FOCAL_MLMODELS_DIR"] {
            let url = URL(fileURLWithPath: envDir).appendingPathComponent("\(name).mlmodelc")
            if FileManager.default.fileExists(atPath: url.path) {
                return try MLModel(contentsOf: url, configuration: configuration)
            }
        }
        // CLI tool fallback: sibling to binary (for `tool` target without resource bundle)
        let exeDir = Bundle.main.bundleURL.deletingLastPathComponent()
        let siblingURL = exeDir.appendingPathComponent("\(name).mlmodelc")
        if FileManager.default.fileExists(atPath: siblingURL.path) {
            return try MLModel(contentsOf: siblingURL, configuration: configuration)
        }
        // Fallback: compile at runtime on first launch after an update
        guard let pkgURL = Bundle.main.url(forResource: name, withExtension: "mlpackage") else {
            throw RatingError.modelNotFound(name)
        }
        let compiled = try MLModel.compileModel(at: pkgURL)
        return try MLModel(contentsOf: compiled, configuration: configuration)
    }

    private static var isAppleSilicon: Bool {
        #if arch(arm64)
        return true
        #else
        return false
        #endif
    }
}
