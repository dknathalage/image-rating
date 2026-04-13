import Foundation

struct RatingResult: Equatable {
    /// 1–5 stars. 0 = unrated/failed.
    let stars: Int
    let clipScore: Float
    let aestheticScore: Float

    static let unrated = RatingResult(stars: 0, clipScore: 0, aestheticScore: 0)
}
