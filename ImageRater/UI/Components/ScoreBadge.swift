import SwiftUI

struct ScoreBadge: View {
    let stars: Int
    let rejected: Bool
    var isManual: Bool = false

    var body: some View {
        HStack(spacing: 3) {
            if rejected {
                Image(systemName: "xmark.circle.fill")
                    .foregroundColor(.red)
            } else if stars > 0 {
                Text("\(stars)/5")
                    .font(.caption.monospacedDigit())
                Image(systemName: "star.fill")
                    .font(.system(size: 7))
            } else {
                Text("—").font(.caption2)
            }
        }
        .foregroundColor(.secondary)
        .padding(.horizontal, 5)
        .padding(.vertical, 3)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 4))
    }
}
