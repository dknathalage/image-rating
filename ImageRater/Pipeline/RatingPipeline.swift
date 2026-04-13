import CoreML
import CoreImage
import Foundation
import OSLog

private let log = Logger(subsystem: "com.imagerating", category: "RatingPipeline")

enum RatingError: Error {
    case pixelBufferCreationFailed
}

enum RatingPipeline {

    // MARK: - Star mapping

    /// Map a combined NIMA score [1–10] to 1–5 stars.
    ///
    /// Thresholds calibrated against real-session data (median combined ≈ 4.4–4.5);
    /// AVA competition-photo thresholds (4.0/4.8/5.6/6.4) were ~1.2 pts too high
    /// for typical photography sessions, causing nearly all images to score 2★.
    ///
    ///   < 3.8  →  1★  (poor — technically or aesthetically deficient)
    ///   < 4.2  →  2★  (below average)
    ///   < 4.6  →  3★  (average)
    ///   < 5.0  →  4★  (good)
    ///   ≥ 5.0  →  5★  (exceptional)
    static func absoluteStars(combined score: Float) -> Int {
        switch score {
        case ..<3.8: return 1
        case ..<4.2: return 2
        case ..<4.6: return 3
        case ..<5.0: return 4
        default:     return 5
        }
    }

    // MARK: - Inference

    /// Run NIMA aesthetic + technical inference on a CGImage.
    /// Returns .unrated on any failure — never throws.
    static func rate(image: CGImage,
                     nimaAestheticModel: MLModel,
                     nimaTechnicalModel: MLModel) async -> RatingResult {
        do {
            log.debug("Creating pixel buffer \(image.width)×\(image.height) → 224×224")
            let pixelBuffer = try cgImageToPixelBuffer(image, width: 224, height: 224)

            log.debug("Running NIMA aesthetic inference")
            let aestheticInput = try MLDictionaryFeatureProvider(dictionary: ["image": pixelBuffer])
            let aestheticOutput = try await nimaAestheticModel.prediction(from: aestheticInput)
            let aestheticScore = extractScore(from: aestheticOutput)
            log.info("NIMA aesthetic: \(aestheticScore, format: .fixed(precision: 4))")

            log.debug("Running NIMA technical inference")
            let technicalInput = try MLDictionaryFeatureProvider(dictionary: ["image": pixelBuffer])
            let technicalOutput = try await nimaTechnicalModel.prediction(from: technicalInput)
            let technicalScore = extractScore(from: technicalOutput)
            log.info("NIMA technical: \(technicalScore, format: .fixed(precision: 4))")

            let combined = (aestheticScore + technicalScore) / 2.0
            let stars = absoluteStars(combined: combined)
            log.info("Combined \(combined, format: .fixed(precision: 3)) → \(stars)★ (aes \(aestheticScore, format: .fixed(precision: 3)), tech \(technicalScore, format: .fixed(precision: 3)))")

            return RatingResult(stars: stars, aestheticScore: aestheticScore, technicalScore: technicalScore)
        } catch {
            log.error("Rating failed: \(error)")
            return .unrated
        }
    }

    // MARK: - Internal

    /// Extract scalar score from a CoreML output provider.
    /// Tries "score" first, then falls back to whichever key the model exposes
    /// (TF SavedModel conversion often names the output "Identity").
    private static func extractScore(from output: MLFeatureProvider) -> Float {
        let key = output.featureNames.contains("score")
            ? "score"
            : (output.featureNames.first ?? "score")
        return output.featureValue(for: key).flatMap { fv -> Float? in
            if let arr = fv.multiArrayValue { return arr[0].floatValue }
            return Float(fv.doubleValue)
        } ?? 5.0
    }

    static func cgImageToPixelBuffer(_ image: CGImage, width: Int, height: Int) throws -> CVPixelBuffer {
        var buffer: CVPixelBuffer?
        let attrs: CFDictionary = [
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true
        ] as CFDictionary
        let status = CVPixelBufferCreate(kCFAllocatorDefault, width, height,
                                         kCVPixelFormatType_32BGRA, attrs, &buffer)
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
            bitmapInfo: CGBitmapInfo.byteOrder32Little.rawValue | CGImageAlphaInfo.noneSkipFirst.rawValue
        ) else { throw RatingError.pixelBufferCreationFailed }
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        return pb
    }
}
