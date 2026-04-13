import Foundation

struct RatingResult: Equatable {
    /// 1–5 stars. 0 = unrated/failed.
    let stars: Int
    /// NIMA aesthetic quality score [1–10]. Higher = more aesthetically pleasing.
    let aestheticScore: Float
    /// NIMA technical quality score [1–10]. Higher = sharper, less noise, better exposure.
    let technicalScore: Float

    static let unrated = RatingResult(stars: 0, aestheticScore: 0, technicalScore: 0)

    /// Combined score used for star mapping: simple average of both dimensions.
    var combinedScore: Float { (aestheticScore + technicalScore) / 2.0 }
}
