// ImageRater/Pipeline/DiversityScorer.swift
import Foundation
import Accelerate

/// Pure-algorithm diversity scoring. No CoreML, CoreData, or side effects.
/// All functions are static and safe to call from any context.
enum DiversityScorer {

    // MARK: - Public output type

    struct MMRItem {
        let originalIndex: Int
        let clusterRank: Int       // 1-based; 1 = selected first (best quality + diversity)
        let diversityFactor: Float // multiplied against MUSIQ aesthetic score
    }

    // MARK: - Cosine similarity

    /// Dot product of two equal-length Float vectors (L2-normalised embeddings ≡ cosine similarity).
    static func cosineSimilarity(_ a: [Float], _ b: [Float]) -> Float {
        precondition(a.count == b.count, "cosineSimilarity: vector length mismatch \(a.count) vs \(b.count)")
        var result: Float = 0
        vDSP_dotpr(a, 1, b, 1, &result, vDSP_Length(a.count))
        return result
    }

    // MARK: - Threshold clustering

    /// Greedy single-link threshold clustering.
    /// Each image joins the first existing cluster whose seed has sim ≥ threshold,
    /// or seeds a new cluster. O(n²) worst-case; fast in practice for n ≤ 2000.
    /// Returns Int32 cluster ID (≥ 0) for each input index.
    static func clusterByThreshold(embeddings: [[Float]], threshold: Float) -> [Int32] {
        var ids = [Int32](repeating: -1, count: embeddings.count)
        var seeds: [[Float]] = []

        for (i, emb) in embeddings.enumerated() {
            var assigned = false
            for (ci, seed) in seeds.enumerated() {
                if cosineSimilarity(emb, seed) >= threshold {
                    ids[i] = Int32(ci)
                    assigned = true
                    break
                }
            }
            if !assigned {
                ids[i] = Int32(seeds.count)
                seeds.append(emb)
            }
        }
        return ids
    }

    // MARK: - MMR ordering

    /// Maximal Marginal Relevance: greedily selects the next image that maximises
    /// λ × quality - (1-λ) × max_similarity_to_already_selected.
    ///
    /// λ = 0.6: quality counts 60%, diversity 40%.
    /// Returns MMRItems in selection order. clusterRank = position in that order (1-based).
    static func mmrOrder(
        embeddings: [[Float]],
        qualityScores: [Float],
        lambda: Float = 0.6
    ) -> [MMRItem] {
        let n = embeddings.count
        guard n > 0 else { return [] }
        precondition(qualityScores.count == n, "mmrOrder: embeddings/qualityScores count mismatch \(n) vs \(qualityScores.count)")

        var selected: [Int] = []
        var remaining = IndexSet(0..<n)
        var result = [MMRItem]()
        result.reserveCapacity(n)

        while !remaining.isEmpty {
            var bestIdx = remaining.first!
            var bestScore = -Float.infinity

            for idx in remaining {
                let quality = lambda * qualityScores[idx]
                let penalty: Float
                if selected.isEmpty {
                    penalty = 0
                } else {
                    penalty = (1 - lambda) * selected
                        .map { cosineSimilarity(embeddings[idx], embeddings[$0]) }
                        .max()!
                }
                let s = quality - penalty
                if s > bestScore { bestScore = s; bestIdx = idx }
            }

            selected.append(bestIdx)
            remaining.remove(bestIdx)
            let rank = selected.count
            result.append(MMRItem(
                originalIndex:  bestIdx,
                clusterRank:    rank,
                diversityFactor: diversityFactor(rank: rank)
            ))
        }
        return result
    }

    // MARK: - Percentile normalisation

    /// Map finalScores array to 1–5 star ratings using percentile thresholds:
    ///   top 5%     → 5★
    ///   5–20%      → 4★
    ///   20–50%     → 3★
    ///   50–80%     → 2★
    ///   bottom 20% → 1★
    ///
    /// Returns Int16 array aligned with input indices.
    static func percentileToStars(finalScores: [Float]) -> [Int16] {
        let n = finalScores.count
        guard n > 0 else { return [] }

        let sortedDesc = finalScores.indices.sorted { finalScores[$0] > finalScores[$1] }
        var stars = [Int16](repeating: 1, count: n)
        for (rank, originalIdx) in sortedDesc.enumerated() {
            let p = Float(rank) / Float(n)   // 0.0 = best
            stars[originalIdx] = switch p {
            case ..<0.05:      5
            case 0.05..<0.20:  4
            case 0.20..<0.50:  3
            case 0.50..<0.80:  2
            default:           1
            }
        }
        return stars
    }

    // MARK: - Private

    private static func diversityFactor(rank: Int) -> Float {
        switch rank {
        case 1, 2: return 1.0
        case 3:    return 0.85
        case 4:    return 0.70
        default:   return 0.55
        }
    }
}
