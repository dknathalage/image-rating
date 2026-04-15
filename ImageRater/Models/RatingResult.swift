// ImageRater/Models/RatingResult.swift
import Foundation

struct RatedScores: Codable, Equatable {
    let musiqAesthetic: Float
    let stars: Int
}

enum RatingResult {
    case rated(RatedScores)
    case unrated
}
