import XCTest
import CoreImage
import AppKit
@testable import ImageRater

final class CullPipelineTests: XCTestCase {

    // MARK: Blur

    func testSharpImageNotRejectedForBlur() {
        let image = makeGradientImage(size: CGSize(width: 100, height: 100))
        // CIEdges variance for a sharp gradient image is well above 500 (the current threshold)
        let (result, _) = CullPipeline.checkBlur(image: image, threshold: 500.0)
        XCTAssertFalse(result.rejected)
    }

    func testBlurryImageRejected() {
        let blurry = makeBlurredImage(size: CGSize(width: 100, height: 100), radius: 20)
        // CIEdges variance for a heavily blurred image is below 500 (the current threshold)
        let (result, _) = CullPipeline.checkBlur(image: blurry, threshold: 500.0)
        XCTAssertTrue(result.rejected)
        XCTAssertEqual(result.reason, .blurry)
    }

    // MARK: Exposure

    func testOverexposedImageRejected() {
        let white = makeSolidImage(size: CGSize(width: 50, height: 50), gray: 1.0)
        let (result, _) = CullPipeline.checkExposure(image: white, exposureLeniency: 0.9)
        XCTAssertTrue(result.rejected)
        XCTAssertEqual(result.reason, .overexposed)
    }

    func testUnderexposedImageRejected() {
        let black = makeSolidImage(size: CGSize(width: 50, height: 50), gray: 0.0)
        let (result, _) = CullPipeline.checkExposure(image: black, exposureLeniency: 0.9)
        XCTAssertTrue(result.rejected)
        XCTAssertEqual(result.reason, .underexposed)
    }

    func testNormalExposureKept() {
        let mid = makeSolidImage(size: CGSize(width: 50, height: 50), gray: 0.5)
        let (result, _) = CullPipeline.checkExposure(image: mid, exposureLeniency: 0.9)
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

    func testEARWithWrongPointCountReturnsOne() {
        let pts = [CGPoint(x: 0, y: 0), CGPoint(x: 1, y: 0)] // only 2 points
        let ear = CullPipeline.eyeAspectRatio(points: pts)
        XCTAssertEqual(ear, 1.0) // guard returns 1.0 (open eye) on wrong count
    }

    // MARK: CullScores

    func testCullReturnsBlurScoreForSharpImage() async {
        // A synthetic sharp image (high-frequency checkerboard) → high blurScore
        let sharp = makeCheckerboardImage(size: 256)
        let result = await CullPipeline.cull(
            image: sharp, blurThreshold: 100, earThreshold: 0.15, exposureLeniency: 0.95)
        XCTAssertGreaterThan(result.blurScore, 0, "Sharp image must have non-zero blurScore")
    }

    func testCullReturnsMeasurableExposureScoreForWhiteImage() async {
        let white = makeSolidColorImage(size: 256, color: .white)
        let result = await CullPipeline.cull(
            image: white, blurThreshold: 0, earThreshold: 0.15, exposureLeniency: 0.95)
        // All-white = overexposed → positive exposureScore
        XCTAssertGreaterThan(result.exposureScore, 0,
            "All-white image should have positive exposureScore")
    }

    func testCullReturnsMeasurableExposureScoreForBlackImage() async {
        let black = makeSolidColorImage(size: 256, color: .black)
        let result = await CullPipeline.cull(
            image: black, blurThreshold: 0, earThreshold: 0.15, exposureLeniency: 0.95)
        // All-black = underexposed → negative exposureScore
        XCTAssertLessThan(result.exposureScore, 0,
            "All-black image should have negative exposureScore")
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

    private func makeCheckerboardImage(size: Int) -> CGImage {
        let bmi = CGBitmapInfo.byteOrder32Little.rawValue | CGImageAlphaInfo.noneSkipFirst.rawValue
        let ctx = CGContext(data: nil, width: size, height: size, bitsPerComponent: 8,
                            bytesPerRow: 4 * size, space: CGColorSpaceCreateDeviceRGB(),
                            bitmapInfo: bmi)!
        let tileSize = 8
        for row in 0..<(size / tileSize) {
            for col in 0..<(size / tileSize) {
                let isWhite = (row + col) % 2 == 0
                ctx.setFillColor(isWhite ? CGColor.white : CGColor.black)
                ctx.fill(CGRect(x: col * tileSize, y: row * tileSize, width: tileSize, height: tileSize))
            }
        }
        return ctx.makeImage()!
    }

    private func makeSolidColorImage(size: Int, color: NSColor) -> CGImage {
        let bmi = CGBitmapInfo.byteOrder32Little.rawValue | CGImageAlphaInfo.noneSkipFirst.rawValue
        let ctx = CGContext(data: nil, width: size, height: size, bitsPerComponent: 8,
                            bytesPerRow: 4 * size, space: CGColorSpaceCreateDeviceRGB(),
                            bitmapInfo: bmi)!
        ctx.setFillColor(color.cgColor)
        ctx.fill(CGRect(x: 0, y: 0, width: size, height: size))
        return ctx.makeImage()!
    }
}
