import CoreML
import CoreImage
import Foundation

enum RatingError: Error {
    case pixelBufferCreationFailed
}

enum RatingPipeline {

    /// Normalize aesthetic score (1–10) to 1–5 star rating.
    static func starsFromAestheticScore(_ score: Float) -> Int {
        let clamped = min(max(score, 1.0), 10.0)
        return Int(ceil((clamped / 10.0) * 5.0))
    }

    /// Combine CLIP cosine similarity (0–1) and aesthetic score (1–10) into RatingResult.
    /// clipWeight and aestheticWeight are relative (don't need to sum to 1).
    static func combineScores(clipScore: Float, aestheticScore: Float,
                               clipWeight: Float, aestheticWeight: Float) -> RatingResult {
        let clipNorm = clipScore * 10.0 // scale CLIP 0–1 to 0–10
        let totalWeight = clipWeight + aestheticWeight
        guard totalWeight > 0 else { return .unrated }
        let combined = (clipNorm * clipWeight + aestheticScore * aestheticWeight) / totalWeight
        return RatingResult(
            stars: starsFromAestheticScore(combined),
            clipScore: clipScore,
            aestheticScore: aestheticScore
        )
    }

    /// Run Core ML inference on a CGImage. Returns .unrated on any failure (never throws).
    /// NOTE: output feature name "score" must match the .mlpackage output layer name from coremltools conversion.
    static func rate(image: CGImage,
                     clipModel: MLModel,
                     aestheticModel: MLModel,
                     clipWeight: Float,
                     aestheticWeight: Float) async -> RatingResult {
        do {
            let pixelBuffer = try cgImageToPixelBuffer(image, width: 224, height: 224)

            let clipInput = try MLDictionaryFeatureProvider(dictionary: ["image": pixelBuffer])
            let clipOutput = try await clipModel.prediction(from: clipInput)
            let clipScore = clipOutput.featureValue(for: "score").flatMap { Float($0.doubleValue) } ?? 0.5

            let aestheticInput = try MLDictionaryFeatureProvider(dictionary: ["image": pixelBuffer])
            let aestheticOutput = try await aestheticModel.prediction(from: aestheticInput)
            let aestheticScore = aestheticOutput.featureValue(for: "score").flatMap { Float($0.doubleValue) } ?? 5.0

            return combineScores(clipScore: clipScore, aestheticScore: aestheticScore,
                                  clipWeight: clipWeight, aestheticWeight: aestheticWeight)
        } catch {
            return .unrated
        }
    }

    // MARK: Private

    static func cgImageToPixelBuffer(_ image: CGImage, width: Int, height: Int) throws -> CVPixelBuffer {
        var buffer: CVPixelBuffer?
        let attrs: CFDictionary = [
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true
        ] as CFDictionary
        let status = CVPixelBufferCreate(kCFAllocatorDefault, width, height,
                                         kCVPixelFormatType_32ARGB, attrs, &buffer)
        guard status == kCVReturnSuccess, let pb = buffer else {
            throw RatingError.pixelBufferCreationFailed
        }
        CVPixelBufferLockBaseAddress(pb, [])
        defer { CVPixelBufferUnlockBaseAddress(pb, []) }
        guard let ctx = CGContext(
            data: CVPixelBufferGetBaseAddress(pb),
            width: width, height: height,
            bitsPerComponent: 8,
            bytesPerRow: CVPixelBufferGetBytesPerRow(pb),
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue
        ) else { throw RatingError.pixelBufferCreationFailed }
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        return pb
    }
}
