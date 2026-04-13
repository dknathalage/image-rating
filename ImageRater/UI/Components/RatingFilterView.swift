import CoreData
import SwiftUI

struct RatingFilterView: View {
    let images: [ImageRecord]
    @Binding var ratingFilter: Set<Int>

    private func effectiveRating(_ record: ImageRecord) -> Int {
        if let o = record.userOverride, o.int16Value > 0 { return Int(o.int16Value) }
        if let s = record.ratingStars { return Int(s.int16Value) }
        return 0
    }

    private var ratingCounts: [Int: Int] {
        images.reduce(into: [:]) { counts, record in
            let r = effectiveRating(record)
            counts[r, default: 0] += 1
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            ForEach([5, 4, 3, 2, 1, 0], id: \.self) { rating in
                let count = ratingCounts[rating] ?? 0
                let selected = ratingFilter.contains(rating)

                Button {
                    if selected {
                        ratingFilter.remove(rating)
                    } else {
                        ratingFilter.insert(rating)
                    }
                } label: {
                    HStack {
                        Text(rating == 0 ? "Unrated" : String(repeating: "★", count: rating))
                            .foregroundStyle(selected ? Color.accentColor : (count == 0 ? Color.secondary : Color.primary))
                        Spacer()
                        Text("\(count)")
                            .foregroundStyle(.secondary)
                    }
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
            }
        }
    }
}
