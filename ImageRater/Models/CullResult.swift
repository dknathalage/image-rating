import Foundation

enum CullReason: String, Codable {
    case blurry
    case eyesClosed
    case overexposed
    case underexposed
}

struct CullResult: Equatable {
    let rejected: Bool
    let reason: CullReason?

    static let keep = CullResult(rejected: false, reason: nil)

    static func reject(_ reason: CullReason) -> CullResult {
        CullResult(rejected: true, reason: reason)
    }
}
