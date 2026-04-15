import CoreML
import CoreGraphics
@testable import Focal

enum TestImage {
    static func solid(_ w: Int, _ h: Int) -> CGImage {
        let ctx = CGContext(
            data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
            space: CGColorSpace(name: CGColorSpace.sRGB)!,
            bitmapInfo: CGBitmapInfo.byteOrder32Little.rawValue | CGImageAlphaInfo.noneSkipFirst.rawValue
        )!
        ctx.setFillColor(CGColor(red: 0.4, green: 0.6, blue: 0.2, alpha: 1))
        ctx.fill(CGRect(x: 0, y: 0, width: w, height: h))
        return ctx.makeImage()!
    }
}

struct MockMUSIQ: MUSIQInferring {
    let scalar: Float
    func predict(patchTensor: MLMultiArray) async throws -> Float { scalar }
}

enum MockModel {
    static func make(scalar: Float) -> RatingPipeline.BundledModels {
        RatingPipeline.BundledModels(musiq: MockMUSIQ(scalar: scalar))
    }
}
