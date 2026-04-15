// ImageRater/Pipeline/RatingPipeline.swift
import CoreML
import CoreImage
import CoreVideo
import Foundation
import OSLog

private let log = Logger(subsystem: "com.focal.app", category: "RatingPipeline")

enum RatingError: Error {
    case pixelBufferCreationFailed
    case modelNotFound(String)
    case inferenceOutputMismatch
    case imageTooSmall
}

enum RatingPipeline {

    // MARK: - Model loading (call once before processing loop)

    struct BundledModels {
        let technical: MLModel   // TOPIQ-NR
        let aesthetic: MLModel   // TOPIQ-Swin
        let clip: MLModel        // CLIP vision encoder
    }

    /// Load all three bundled models. Call once; pass result into rate() for every image.
    /// Throws if any .mlpackage/.mlmodelc is missing from the app bundle.
    static func loadBundledModels() throws -> BundledModels {
        let config = MLModelConfiguration()
        config.computeUnits = isAppleSilicon ? .all : .cpuOnly
        return BundledModels(
            technical: try loadBundledModel(named: "topiq-nr",    configuration: config),
            aesthetic: try loadBundledModel(named: "topiq-swin",  configuration: config),
            clip:      try loadBundledModel(named: "clip-vision", configuration: config)
        )
    }

    // MARK: - Inference

    /// Rate a single image with all three models concurrently. Never throws — returns .unrated on failure.
    static func rate(
        image: CGImage,
        models: BundledModels,
        weights: (technical: Float, aesthetic: Float, semantic: Float) = (0.4, 0.4, 0.2)
    ) async -> RatingResult {
        do {
            async let techScoreTask  = inferScore(image: image, model: models.technical, inputSize: 224)
            async let aesScoreTask   = inferScore(image: image, model: models.aesthetic, inputSize: 384)
            async let clipEmbTask    = inferEmbedding(image: image, model: models.clip,  inputSize: 224)

            let (tech, aes, emb) = try await (techScoreTask, aesScoreTask, clipEmbTask)
            let logitScale = Float(FocalSettings.resolvedClipLogitScale())
            let clip     = clipIQAScore(embedding: emb, logitScale: logitScale)
            let combined = combinedQuality(technical: tech, aesthetic: aes, semantic: clip, weights: weights)

            log.info("TOPIQ-NR \(tech, format: .fixed(precision: 3))  TOPIQ-Swin \(aes, format: .fixed(precision: 3))  CLIP-IQA+ \(clip, format: .fixed(precision: 3))  combined \(combined, format: .fixed(precision: 3))")

            return .rated(RatedScores(
                topiqTechnicalScore:  tech,
                topiqAestheticScore:  aes,
                clipIQAScore:         clip,
                combinedQualityScore: combined,
                clipEmbedding:        emb
            ))
        } catch {
            log.error("Rating failed: \(error)")
            return .unrated
        }
    }

    // MARK: - Helpers (internal, exposed for testing)

    static func combinedQuality(
        technical: Float, aesthetic: Float, semantic: Float,
        weights: (technical: Float, aesthetic: Float, semantic: Float)
    ) -> Float {
        precondition(abs(weights.technical + weights.aesthetic + weights.semantic - 1.0) < 0.01,
                     "combinedQuality: weights must sum to 1.0, got \(weights.technical + weights.aesthetic + weights.semantic)")
        return weights.technical * technical + weights.aesthetic * aesthetic + weights.semantic * semantic
    }

    /// CLIP-IQA+: average softmax P("positive prompt") across all antonym pairs.
    /// logitScale matches CLIP ViT-B/32's learned temperature (exp(logit_scale) ≈ 100).
    /// Without this, raw cosine similarities (~0.2–0.3) are too small for softmax to
    /// discriminate; all scores collapse to ~0.5.
    static func clipIQAScore(embedding: [Float], logitScale: Float) -> Float {
        let posPrompts = CLIPTextEmbeddings.positivePrompts
        let negPrompts = CLIPTextEmbeddings.negativePrompts
        precondition(!posPrompts.isEmpty && posPrompts.count == negPrompts.count,
                     "clipIQAScore: mismatched or empty antonym pairs")
        precondition(embedding.count == posPrompts[0].count,
                     "clipIQAScore: embedding size \(embedding.count) != text embedding size \(posPrompts[0].count)")
        // L2-normalise defensively: coremltools may drop F.normalize from the traced VisionWrapper.
        let norm = sqrtf(embedding.reduce(0) { $0 + $1 * $1 })
        guard norm > 1e-6 else {
            log.warning("CLIP embedding near-zero (norm=\(norm)) — vision model may be broken; returning 0.5")
            return 0.5
        }
        let emb = norm < 0.999 || norm > 1.001 ? embedding.map { $0 / norm } : embedding
        var total: Float = 0
        for (pos, neg) in zip(posPrompts, negPrompts) {
            let dotPos: Float = logitScale * zip(emb, pos).reduce(0) { $0 + $1.0 * $1.1 }
            let dotNeg: Float = logitScale * zip(emb, neg).reduce(0) { $0 + $1.0 * $1.1 }
            let maxVal  = max(dotPos, dotNeg)
            let eP = expf(dotPos - maxVal)
            let eN = expf(dotNeg - maxVal)
            total += eP / (eP + eN)
        }
        return total / Float(posPrompts.count)
    }

    // MARK: - Pixel buffer

    static func cgImageToPixelBuffer(_ image: CGImage, width: Int, height: Int) throws -> CVPixelBuffer {
        var buffer: CVPixelBuffer?
        let attrs: CFDictionary = [
            kCVPixelBufferCGImageCompatibilityKey:       true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true
        ] as CFDictionary
        guard CVPixelBufferCreate(kCFAllocatorDefault, width, height,
                                  kCVPixelFormatType_32BGRA, attrs, &buffer) == kCVReturnSuccess,
              let pb = buffer else { throw RatingError.pixelBufferCreationFailed }
        CVPixelBufferLockBaseAddress(pb, [])
        defer { CVPixelBufferUnlockBaseAddress(pb, []) }
        guard let sRGB = CGColorSpace(name: CGColorSpace.sRGB),
              let ctx = CGContext(
            data: CVPixelBufferGetBaseAddress(pb),
            width: width, height: height,
            bitsPerComponent: 8,
            bytesPerRow: CVPixelBufferGetBytesPerRow(pb),
            space: sRGB,
            bitmapInfo: CGBitmapInfo.byteOrder32Little.rawValue | CGImageAlphaInfo.noneSkipFirst.rawValue
        ) else { throw RatingError.pixelBufferCreationFailed }
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        return pb
    }

    // MARK: - Private

    private static func inferScore(image: CGImage, model: MLModel, inputSize: Int) async throws -> Float {
        let buf   = try cgImageToPixelBuffer(image, width: inputSize, height: inputSize)
        let input = try MLDictionaryFeatureProvider(dictionary: ["image": buf])
        let out   = try await model.prediction(from: input)
        return try extractScalar(from: out)
    }

    private static func inferEmbedding(image: CGImage, model: MLModel, inputSize: Int) async throws -> [Float] {
        let buf   = try cgImageToPixelBuffer(image, width: inputSize, height: inputSize)
        let input = try MLDictionaryFeatureProvider(dictionary: ["image": buf])
        let out   = try await model.prediction(from: input)
        return try extractFloatArray(from: out)
    }

    private static func extractScalar(from output: MLFeatureProvider) throws -> Float {
        for name in output.featureNames {
            if let arr = output.featureValue(for: name)?.multiArrayValue { return arr[0].floatValue }
            if let d   = output.featureValue(for: name)?.doubleValue     { return Float(d) }
        }
        throw RatingError.inferenceOutputMismatch
    }

    private static func extractFloatArray(from output: MLFeatureProvider) throws -> [Float] {
        for name in output.featureNames {
            if let arr = output.featureValue(for: name)?.multiArrayValue {
                return (0..<arr.count).map { arr[$0].floatValue }
            }
        }
        throw RatingError.inferenceOutputMismatch
    }

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
