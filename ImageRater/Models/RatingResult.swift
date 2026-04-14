// ImageRater/Models/RatingResult.swift
import Foundation

/// Scores from a successful rating inference run.
struct RatedScores: Equatable {
    let topiqTechnicalScore: Float      // TOPIQ-NR output [0,1]
    let topiqAestheticScore: Float      // TOPIQ-Swin output [0,1]
    let clipIQAScore: Float             // CLIP-IQA+ antonym softmax [0,1]
    let combinedQualityScore: Float     // weighted ensemble [0,1]
    let clipEmbedding: [Float]          // 512-dim L2-normalised; used for MMR diversity
}

/// Result of rating a single image.
enum RatingResult: Equatable {
    case unrated
    case rated(RatedScores)
}
