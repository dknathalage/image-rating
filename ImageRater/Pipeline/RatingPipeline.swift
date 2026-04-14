// ImageRater/Pipeline/RatingPipeline.swift
import CoreML
import CoreImage
import CoreVideo
import Foundation
import OSLog

private let log = Logger(subsystem: "com.imagerating", category: "RatingPipeline")

enum RatingError: Error {
    case pixelBufferCreationFailed
    case modelNotFound(String)
    case inferenceOutputMismatch
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
            let clip     = clipIQAScore(embedding: emb)
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

    /// CLIP-IQA+: softmax([dot(img, good), dot(img, bad)])[0] = P("Good photo")
    static func clipIQAScore(embedding: [Float]) -> Float {
        let good = CLIPTextEmbeddings.goodPhoto
        let bad  = CLIPTextEmbeddings.badPhoto
        precondition(embedding.count == good.count,
                     "clipIQAScore: embedding size \(embedding.count) does not match text embedding size \(good.count)")
        let dotGood: Float = zip(embedding, good).reduce(0) { $0 + $1.0 * $1.1 }
        let dotBad:  Float = zip(embedding, bad).reduce(0)  { $0 + $1.0 * $1.1 }
        let maxVal  = max(dotGood, dotBad)
        let eG = expf(dotGood - maxVal)
        let eB = expf(dotBad  - maxVal)
        return eG / (eG + eB)
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
        guard let ctx = CGContext(
            data: CVPixelBufferGetBaseAddress(pb),
            width: width, height: height,
            bitsPerComponent: 8,
            bytesPerRow: CVPixelBufferGetBytesPerRow(pb),
            space: CGColorSpace(name: CGColorSpace.sRGB)!,
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
