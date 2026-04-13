import CoreImage
import Vision
import Foundation

// CullResult and CullReason are defined in ImageRater/Models/CullResult.swift

enum CullPipeline {

    // MARK: - Blur Detection (Laplacian variance)

    /// Returns .reject(.blurry) if Laplacian variance below threshold, else .keep
    static func checkBlur(image: CGImage, threshold: Float) -> CullResult {
        let ci = CIImage(cgImage: image)
        let gray = ci.applyingFilter("CIColorControls", parameters: ["inputSaturation": 0.0])
        let laplacian = gray.applyingFilter("CIEdgeWork", parameters: ["inputRadius": 1.0])
        guard let variance = computeVariance(of: laplacian) else { return .keep }
        return variance < threshold ? .reject(.blurry) : .keep
    }

    // MARK: - Exposure Analysis (histogram)

    /// Returns .reject(.overexposed/.underexposed) if >10% pixels in extreme luminance range
    static func checkExposure(image: CGImage, threshold: Float) -> CullResult {
        let ci = CIImage(cgImage: image)
        let gray = ci.applyingFilter("CIColorControls", parameters: ["inputSaturation": 0.0])
        let binCount = 256
        guard let histogram = computeHistogram(of: gray, binCount: binCount) else { return .keep }
        let total = histogram.reduce(0, +)
        guard total > 0 else { return .keep }
        let topBins = histogram[(binCount - binCount / 20)...].reduce(0, +)
        let bottomBins = histogram[0..<(binCount / 20)].reduce(0, +)
        let topFraction = Float(topBins) / Float(total)
        let bottomFraction = Float(bottomBins) / Float(total)
        if topFraction > (1.0 - threshold) { return .reject(.overexposed) }
        if bottomFraction > (1.0 - threshold) { return .reject(.underexposed) }
        return .keep
    }

    // MARK: - Eyes Closed (EAR)

    /// Eye Aspect Ratio from 6 landmark points. EAR < 0.2 = closed.
    static func eyeAspectRatio(points: [CGPoint]) -> Float {
        guard points.count == 6 else { return 1.0 }
        let p1 = points[0], p2 = points[1], p3 = points[2]
        let p4 = points[3], p5 = points[4], p6 = points[5]
        let vertical = dist(p2, p6) + dist(p3, p5)
        let horizontal = 2.0 * dist(p1, p4)
        guard horizontal > 0 else { return 1.0 }
        return Float(vertical / horizontal)
    }

    static func checkEyesClosed(cgImage: CGImage, earThreshold: Float) async -> CullResult {
        let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
        let request = VNDetectFaceLandmarksRequest()
        try? handler.perform([request])
        guard let faces = request.results, !faces.isEmpty else { return .keep }
        for face in faces {
            guard let landmarks = face.landmarks else { continue }
            if let leftEye = landmarks.leftEye {
                let pts = landmarkPoints(leftEye, in: face.boundingBox)
                if eyeAspectRatio(points: pts) < earThreshold { return .reject(.eyesClosed) }
            }
            if let rightEye = landmarks.rightEye {
                let pts = landmarkPoints(rightEye, in: face.boundingBox)
                if eyeAspectRatio(points: pts) < earThreshold { return .reject(.eyesClosed) }
            }
        }
        return .keep
    }

    /// Full cull — runs all checks, returns first rejection found.
    static func cull(image: CGImage, blurThreshold: Float, earThreshold: Float, exposureThreshold: Float) async -> CullResult {
        let blurResult = checkBlur(image: image, threshold: blurThreshold)
        if blurResult.rejected { return blurResult }
        let exposureResult = checkExposure(image: image, threshold: exposureThreshold)
        if exposureResult.rejected { return exposureResult }
        return await checkEyesClosed(cgImage: image, earThreshold: earThreshold)
    }

    // MARK: - Private helpers

    private static func dist(_ a: CGPoint, _ b: CGPoint) -> CGFloat {
        let dx = a.x - b.x, dy = a.y - b.y
        return sqrt(dx*dx + dy*dy)
    }

    private static func landmarkPoints(_ region: VNFaceLandmarkRegion2D, in box: CGRect) -> [CGPoint] {
        region.normalizedPoints.map { pt in
            CGPoint(x: box.minX + pt.x * box.width, y: box.minY + pt.y * box.height)
        }
    }

    private static func computeVariance(of image: CIImage) -> Float? {
        let context = CIContext()
        let extent = image.extent
        guard let cgImg = context.createCGImage(image, from: extent) else { return nil }
        let width = cgImg.width, height = cgImg.height
        var pixels = [UInt8](repeating: 0, count: width * height)
        let colorSpace = CGColorSpaceCreateDeviceGray()
        guard let ctx = CGContext(data: &pixels, width: width, height: height,
                                  bitsPerComponent: 8, bytesPerRow: width,
                                  space: colorSpace,
                                  bitmapInfo: CGImageAlphaInfo.none.rawValue) else { return nil }
        ctx.draw(cgImg, in: CGRect(x: 0, y: 0, width: width, height: height))
        let count = Float(pixels.count)
        let mean = pixels.reduce(0) { $0 + Float($1) } / count
        let variance = pixels.reduce(0.0) { $0 + pow(Float($1) - mean, 2) } / count
        return variance
    }

    private static func computeHistogram(of image: CIImage, binCount: Int) -> [Int]? {
        let filter = CIFilter(name: "CIAreaHistogram", parameters: [
            "inputImage": image,
            "inputExtent": CIVector(cgRect: image.extent),
            "inputCount": binCount,
            "inputScale": 1.0
        ])
        guard let output = filter?.outputImage else { return nil }
        let context = CIContext()
        var data = [Float](repeating: 0, count: binCount * 4)
        context.render(output, toBitmap: &data,
                       rowBytes: binCount * 4 * MemoryLayout<Float>.size,
                       bounds: CGRect(x: 0, y: 0, width: binCount, height: 1),
                       format: .RGBAf, colorSpace: nil)
        return stride(from: 0, to: data.count, by: 4).map { Int(data[$0] * 1000) }
    }
}
