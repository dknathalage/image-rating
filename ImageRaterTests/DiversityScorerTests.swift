// ImageRaterTests/DiversityScorerTests.swift
import XCTest
import Accelerate
@testable import ImageRater

final class DiversityScorerTests: XCTestCase {

    // MARK: — Cosine similarity

    func testCosineSimilarityIdenticalVectors() {
        let v: [Float] = [1, 0, 0, 0]
        XCTAssertEqual(DiversityScorer.cosineSimilarity(v, v), 1.0, accuracy: 0.001)
    }

    func testCosineSimilarityOrthogonalVectors() {
        XCTAssertEqual(DiversityScorer.cosineSimilarity([1, 0, 0, 0], [0, 1, 0, 0]), 0.0, accuracy: 0.001)
    }

    func testCosineSimilarityOppositeVectors() {
        XCTAssertEqual(DiversityScorer.cosineSimilarity([1, 0, 0, 0], [-1, 0, 0, 0]), -1.0, accuracy: 0.001)
    }

    // MARK: — Threshold clustering

    func testClusteringGroupsIdenticalEmbeddings() {
        let e: [Float] = [1, 0, 0, 0]
        let ids = DiversityScorer.clusterByThreshold(embeddings: [e, e], threshold: 0.9)
        XCTAssertEqual(ids[0], ids[1], "Identical embeddings must be in same cluster")
    }

    func testClusteringKeepsOrthogonalEmbeddingsApart() {
        let ids = DiversityScorer.clusterByThreshold(
            embeddings: [[1, 0, 0, 0], [0, 1, 0, 0]], threshold: 0.9)
        XCTAssertNotEqual(ids[0], ids[1])
    }

    func testClusteringDoesNotAssignMinusOne() {
        // All images must end up in a valid cluster (>= 0)
        let embs: [[Float]] = (0..<10).map { i in
            var v = [Float](repeating: 0, count: 4)
            v[i % 4] = 1; return v
        }
        let ids = DiversityScorer.clusterByThreshold(embeddings: embs, threshold: 0.9)
        XCTAssertTrue(ids.allSatisfy { $0 >= 0 })
    }

    // MARK: — MMR ordering

    func testMMRSingleImageIsRank1WithFullFactor() {
        let items = DiversityScorer.mmrOrder(
            embeddings: [[1, 0, 0, 0]], qualityScores: [0.9], lambda: 0.6)
        XCTAssertEqual(items.count, 1)
        XCTAssertEqual(items[0].clusterRank, 1)
        XCTAssertEqual(items[0].diversityFactor, 1.0, accuracy: 0.001)
    }

    func testMMRPenalizesRank3WithFactor085() {
        // 5 near-identical vectors: rank 3 should get 0.85
        let same: [Float] = [1, 0, 0, 0]
        let embs  = Array(repeating: same, count: 5)
        let scores = (0..<5).map { Float(5 - $0) / 5.0 }   // descending quality
        let items = DiversityScorer.mmrOrder(embeddings: embs, qualityScores: scores, lambda: 0.6)
        let rank3 = items.first { $0.clusterRank == 3 }
        XCTAssertNotNil(rank3)
        XCTAssertEqual(rank3!.diversityFactor, 0.85, accuracy: 0.001)
    }

    func testMMRPenalizesRank5PlusWithFactor055() {
        let same: [Float] = [1, 0, 0, 0]
        let embs  = Array(repeating: same, count: 6)
        let scores = (0..<6).map { Float(6 - $0) / 6.0 }
        let items = DiversityScorer.mmrOrder(embeddings: embs, qualityScores: scores, lambda: 0.6)
        let rank6 = items.first { $0.clusterRank == 6 }
        XCTAssertNotNil(rank6)
        XCTAssertEqual(rank6!.diversityFactor, 0.55, accuracy: 0.001)
    }

    func testMMRPrefersDiverseOverNearDuplicate() {
        // diverse (c) should rank above near-dup (b) despite lower raw quality
        let a: [Float] = [1, 0, 0, 0]
        let b: [Float] = [1, 0, 0, 0]   // near-dup of a
        let c: [Float] = [0, 1, 0, 0]   // diverse

        let items = DiversityScorer.mmrOrder(
            embeddings: [a, b, c],
            qualityScores: [0.9, 0.88, 0.7],
            lambda: 0.6)

        let rankC = items.first { $0.originalIndex == 2 }!.clusterRank
        let rankB = items.first { $0.originalIndex == 1 }!.clusterRank
        XCTAssertLessThan(rankC, rankB, "Diverse image should rank above near-duplicate")
    }

    // MARK: — Percentile normalization

    func testPercentileTop5PercentGets5Stars() {
        let scores = (1...100).map { Float($0) / 100.0 }  // 0.01...1.00
        let stars = DiversityScorer.percentileToStars(finalScores: scores)
        // Index 99 = highest score → rank 0 → percentile 0.0 → 5★
        XCTAssertEqual(stars[99], 5)
    }

    func testPercentileBottom20PercentGets1Star() {
        let scores = (1...100).map { Float($0) / 100.0 }
        let stars = DiversityScorer.percentileToStars(finalScores: scores)
        // Index 0 = lowest score → rank 99 → percentile 0.99 → 1★
        XCTAssertEqual(stars[0], 1)
    }

    func testPercentileSingleImageGets5Stars() {
        let stars = DiversityScorer.percentileToStars(finalScores: [0.7])
        XCTAssertEqual(stars[0], 5)
    }

    func testPercentileAllValuesAre1Through5() {
        let scores = (0..<20).map { Float($0) / 19.0 }
        let stars = DiversityScorer.percentileToStars(finalScores: scores)
        XCTAssertTrue(stars.allSatisfy { $0 >= 1 && $0 <= 5 })
    }
}
