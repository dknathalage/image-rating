import XCTest
import CoreImage
@testable import ImageRater

final class CullPipelineTests: XCTestCase {

    // MARK: Blur

    func testSharpImageNotRejectedForBlur() {
        let image = makeGradientImage(size: CGSize(width: 100, height: 100))
        // CIEdgeWork Laplacian variance for a sharp gradient image is ~6800; threshold of 5000 keeps it
        let result = CullPipeline.checkBlur(image: image, threshold: 5000.0)
        XCTAssertFalse(result.rejected)
    }

    func testBlurryImageRejected() {
        let blurry = makeBlurredImage(size: CGSize(width: 100, height: 100), radius: 20)
        // CIEdgeWork Laplacian variance for a heavily blurred image is ~2900; threshold of 5000 rejects it
        let result = CullPipeline.checkBlur(image: blurry, threshold: 5000.0)
        XCTAssertTrue(result.rejected)
        XCTAssertEqual(result.reason, .blurry)
    }

    // MARK: Exposure

    func testOverexposedImageRejected() {
        let white = makeSolidImage(size: CGSize(width: 50, height: 50), gray: 1.0)
        let result = CullPipeline.checkExposure(image: white, threshold: 0.9)
        XCTAssertTrue(result.rejected)
        XCTAssertEqual(result.reason, .overexposed)
    }

    func testUnderexposedImageRejected() {
        let black = makeSolidImage(size: CGSize(width: 50, height: 50), gray: 0.0)
        let result = CullPipeline.checkExposure(image: black, threshold: 0.9)
        XCTAssertTrue(result.rejected)
        XCTAssertEqual(result.reason, .underexposed)
    }

    func testNormalExposureKept() {
        let mid = makeSolidImage(size: CGSize(width: 50, height: 50), gray: 0.5)
        let result = CullPipeline.checkExposure(image: mid, threshold: 0.9)
        XCTAssertFalse(result.rejected)
    }

    // MARK: EAR

    func testEARBelowThresholdIndicatesClosedEye() {
        // height: 0.01, width: 0.1 gives EAR exactly 0.2; use height: 0.008 for EAR ≈ 0.16
        let pts = makeEyePoints(height: 0.008, width: 0.1)
        let ear = CullPipeline.eyeAspectRatio(points: pts)
        XCTAssertLessThan(ear, 0.2)
    }

    func testEARAboveThresholdIndicatesOpenEye() {
        let pts = makeEyePoints(height: 0.05, width: 0.1) // EAR ≈ 0.5
        let ear = CullPipeline.eyeAspectRatio(points: pts)
        XCTAssertGreaterThan(ear, 0.2)
    }

    // MARK: Helpers

    private func makeGradientImage(size: CGSize) -> CGImage {
        let ctx = CGContext(data: nil, width: Int(size.width), height: Int(size.height),
                           bitsPerComponent: 8, bytesPerRow: 0,
                           space: CGColorSpaceCreateDeviceGray(),
                           bitmapInfo: CGImageAlphaInfo.none.rawValue)!
        for x in 0..<Int(size.width) {
            ctx.setFillColor(gray: CGFloat(x) / size.width, alpha: 1)
            ctx.fill(CGRect(x: x, y: 0, width: 1, height: Int(size.height)))
        }
        return ctx.makeImage()!
    }

    private func makeBlurredImage(size: CGSize, radius: Double) -> CGImage {
        let src = CIImage(cgImage: makeGradientImage(size: size))
        let blurred = src.applyingFilter("CIGaussianBlur", parameters: ["inputRadius": radius])
        return CIContext().createCGImage(blurred, from: blurred.extent)!
    }

    private func makeSolidImage(size: CGSize, gray: CGFloat) -> CGImage {
        let ctx = CGContext(data: nil, width: Int(size.width), height: Int(size.height),
                           bitsPerComponent: 8, bytesPerRow: 0,
                           space: CGColorSpaceCreateDeviceGray(),
                           bitmapInfo: CGImageAlphaInfo.none.rawValue)!
        ctx.setFillColor(gray: gray, alpha: 1)
        ctx.fill(CGRect(origin: .zero, size: size))
        return ctx.makeImage()!
    }

    private func makeEyePoints(height: CGFloat, width: CGFloat) -> [CGPoint] {
        [
            CGPoint(x: 0, y: 0.5),
            CGPoint(x: width * 0.33, y: 0.5 + height),
            CGPoint(x: width * 0.66, y: 0.5 + height),
            CGPoint(x: width, y: 0.5),
            CGPoint(x: width * 0.66, y: 0.5 - height),
            CGPoint(x: width * 0.33, y: 0.5 - height),
        ]
    }
}
