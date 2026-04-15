// ImageRater/Pipeline/MUSIQPreprocessor.swift
import Foundation

enum MUSIQPreprocessor {

    // MARK: - Bicubic resize (matches PyTorch F.interpolate mode='bicubic', align_corners=False)

    /// Cubic kernel (Keys 1981), a=-0.75. PyTorch default in
    /// aten/src/ATen/native/UpSampleBicubic2d.cpp (not -0.5 as plan stated).
    private static func cubicKernel(_ x: Float) -> Float {
        let a: Float = -0.75
        let ax = abs(x)
        if ax <= 1 {
            return ((a + 2) * ax - (a + 3)) * ax * ax + 1
        } else if ax < 2 {
            return (((ax - 5) * ax + 8) * ax - 4) * a
        }
        return 0
    }

    /// Resample one channel. Pixels stored row-major [h, w].
    /// Uses `align_corners=False`: sample at (out + 0.5) * (in / out) - 0.5.
    private static func resampleChannel(
        src: UnsafePointer<Float>, srcH: Int, srcW: Int,
        dst: UnsafeMutablePointer<Float>, dstH: Int, dstW: Int
    ) {
        let scaleY = Float(srcH) / Float(dstH)
        let scaleX = Float(srcW) / Float(dstW)

        for y in 0..<dstH {
            let srcY = (Float(y) + 0.5) * scaleY - 0.5
            let y0 = Int(floor(srcY)) - 1  // 4-tap kernel needs y0..y0+3

            var wy = [Float](repeating: 0, count: 4)
            for k in 0..<4 {
                wy[k] = cubicKernel(srcY - Float(y0 + k))
            }

            for x in 0..<dstW {
                let srcX = (Float(x) + 0.5) * scaleX - 0.5
                let x0 = Int(floor(srcX)) - 1

                var wx = [Float](repeating: 0, count: 4)
                for k in 0..<4 {
                    wx[k] = cubicKernel(srcX - Float(x0 + k))
                }

                var acc: Float = 0
                for ky in 0..<4 {
                    let yi = max(0, min(srcH - 1, y0 + ky))
                    let row = src + yi * srcW
                    var rowAcc: Float = 0
                    for kx in 0..<4 {
                        let xi = max(0, min(srcW - 1, x0 + kx))
                        rowAcc += row[xi] * wx[kx]
                    }
                    acc += rowAcc * wy[ky]
                }
                dst[y * dstW + x] = acc
            }
        }
    }

    /// Aspect-preserving resize. Longer side becomes `longerSide`. Per-channel.
    /// Input layout: planar [C, H, W] flattened in C-major, H-major, W-minor order.
    static func aspectResize(
        pixels: [Float], h: Int, w: Int, channels: Int, longerSide: Int
    ) -> (pixels: [Float], rh: Int, rw: Int) {
        let ratio = Double(longerSide) / Double(max(h, w))
        let rh = Int((Double(h) * ratio).rounded(.toNearestOrEven))
        let rw = Int((Double(w) * ratio).rounded(.toNearestOrEven))
        var out = [Float](repeating: 0, count: channels * rh * rw)
        pixels.withUnsafeBufferPointer { srcBuf in
            out.withUnsafeMutableBufferPointer { dstBuf in
                for c in 0..<channels {
                    let srcBase = srcBuf.baseAddress! + c * h * w
                    let dstBase = dstBuf.baseAddress! + c * rh * rw
                    resampleChannel(src: srcBase, srcH: h, srcW: w,
                                    dst: dstBase, dstH: rh, dstW: rw)
                }
            }
        }
        return (out, rh, rw)
    }

    // MARK: - Patch unfold (32×32 stride-32 with TF-SAME padding)

    /// TF-SAME padding: output count = ceil(input / stride). Extra pixel on bottom/right.
    static func unfoldPatches(
        pixels: [Float], h: Int, w: Int, channels: Int, patch: Int
    ) -> (patches: [Float], countH: Int, countW: Int) {
        let stride = patch
        let countH = (h + stride - 1) / stride
        let countW = (w + stride - 1) / stride
        let padH = (countH - 1) * stride + patch - h     // ≥ 0
        let padW = (countW - 1) * stride + patch - w
        let top = padH / 2, left = padW / 2              // matches pyiqa F.pad ordering

        let numPatches = countH * countW
        let rowDim = channels * patch * patch
        var out = [Float](repeating: 0, count: numPatches * rowDim)

        pixels.withUnsafeBufferPointer { srcBuf in
            out.withUnsafeMutableBufferPointer { dstBuf in
                for py in 0..<countH {
                    for px in 0..<countW {
                        let patchIdx = py * countW + px
                        let dstBase = dstBuf.baseAddress! + patchIdx * rowDim
                        for c in 0..<channels {
                            let srcChannel = srcBuf.baseAddress! + c * h * w
                            let dstChannel = dstBase + c * patch * patch
                            for dy in 0..<patch {
                                let srcY = py * stride + dy - top
                                for dx in 0..<patch {
                                    let srcX = px * stride + dx - left
                                    let dstIdx = dy * patch + dx
                                    if srcY >= 0 && srcY < h && srcX >= 0 && srcX < w {
                                        dstChannel[dstIdx] = srcChannel[srcY * w + srcX]
                                    } else {
                                        dstChannel[dstIdx] = 0
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        return (out, countH, countW)
    }

    // MARK: - Hash spatial positions (grid_size=10)

    /// Nearest-interp [0..gridSize-1] to `count`, then flatten count_h × count_w
    /// into grid_size-based hash `h * gridSize + w`. Matches pyiqa's
    /// F.interpolate(mode='nearest').
    ///
    /// Nearest-interp formula PyTorch uses (align_corners irrelevant for nearest):
    /// index[i] = floor(i * in_size / out_size) for out_size ≥ in_size.
    /// For grid=10, count=7: indices = [0, 1, 2, 4, 5, 7, 8] (from floor(i * 10 / 7)).
    static func hashSpatialPositions(countH: Int, countW: Int, gridSize: Int) -> [Float] {
        let posH = nearestInterp(count: countH, gridSize: gridSize)
        let posW = nearestInterp(count: countW, gridSize: gridSize)
        var out = [Float](repeating: 0, count: countH * countW)
        for i in 0..<countH {
            for j in 0..<countW {
                out[i * countW + j] = Float(posH[i] * gridSize + posW[j])
            }
        }
        return out
    }

    private static func nearestInterp(count: Int, gridSize: Int) -> [Int] {
        // PyTorch F.interpolate(mode='nearest'): index = floor(i * in / out)
        var out = [Int](repeating: 0, count: count)
        for i in 0..<count {
            out[i] = (i * gridSize) / count
        }
        return out
    }
}
