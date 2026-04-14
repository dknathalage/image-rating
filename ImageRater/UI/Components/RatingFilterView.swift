// ImageRater/UI/Components/RatingFilterView.swift
import SwiftUI
import CoreData

struct RatingFilterView: View {
    let images: [ImageRecord]
    @Binding var ratingFilter: Set<Int>

    private var ratingCounts: [Int: Int] {
        var counts = [Int: Int]()
        for img in images {
            let r = effectiveRating(img)
            counts[r, default: 0] += 1
        }
        return counts
    }

    private func effectiveRating(_ r: ImageRecord) -> Int {
        if let o = r.userOverride, o.int16Value > 0 { return Int(o.int16Value) }
        if let s = r.ratingStars { return Int(s.int16Value) }
        return 0
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            ForEach([5, 4, 3, 2, 1, 0], id: \.self) { rating in
                let count    = ratingCounts[rating] ?? 0
                let selected = ratingFilter.contains(rating)
                Button {
                    if selected { ratingFilter.remove(rating) }
                    else        { ratingFilter.insert(rating) }
                } label: {
                    HStack {
                        Text(rating == 0 ? "Unrated" : String(repeating: "★", count: rating))
                            .foregroundStyle(selected ? Color.accentColor
                                           : (count == 0 ? Color.secondary : Color.primary))
                        Spacer()
                        Text("\(count)").foregroundStyle(.secondary)
                    }
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
            }
        }
    }
}
