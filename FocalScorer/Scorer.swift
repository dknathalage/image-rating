// FocalScorer/Scorer.swift
import Foundation
import CoreImage

enum Scorer {

    struct OutputImage: Codable {
        let filename: String
        let topiqTechnical: Float
        let topiqAesthetic: Float
        let clipEmbedding: [Float]
    }

    struct Output: Codable {
        let generatedAt: String
        let modelVersion: String
        let images: [OutputImage]
    }

    static let supportedExts: Set<String> = [
        "jpg", "jpeg", "png", "raf", "nef", "arw", "cr3", "dng"
    ]

    static func scoreDirectory(inputDir: URL, outputURL: URL) async throws {
        let fm = FileManager.default
        let all = try fm.contentsOfDirectory(at: inputDir, includingPropertiesForKeys: nil)
        let files = all.filter { supportedExts.contains($0.pathExtension.lowercased()) }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        let models = try RatingPipeline.loadBundledModels()
        var results: [OutputImage] = []
        results.reserveCapacity(files.count)
        for (i, url) in files.enumerated() {
            FileHandle.standardError.write(Data("[\(i+1)/\(files.count)] \(url.lastPathComponent)\n".utf8))
            guard let cg = LibRawWrapper.decode(url: url) else {
                FileHandle.standardError.write(Data("  skip: decode failed\n".utf8))
                continue
            }
            let r = await RatingPipeline.rate(image: cg, models: models)
            if case .rated(let s) = r {
                results.append(OutputImage(
                    filename: url.lastPathComponent,
                    topiqTechnical: s.topiqTechnicalScore,
                    topiqAesthetic: s.topiqAestheticScore,
                    clipEmbedding:  s.clipEmbedding
                ))
            }
        }
        let out = Output(
            generatedAt: ISO8601DateFormatter().string(from: Date()),
            modelVersion: "topiq-nr, topiq-swin, clip-vision",
            images: results
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(out)
        try data.write(to: outputURL)
        FileHandle.standardError.write(Data("wrote \(results.count) scores -> \(outputURL.path)\n".utf8))
    }
}
