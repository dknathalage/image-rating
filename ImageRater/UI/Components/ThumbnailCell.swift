import SwiftUI

/// Pure-SwiftUI rotating arc — avoids AppKit NSProgressIndicator layout constraints.
struct SpinnerView: View {
    @State private var rotation: Double = 0
    var size: CGFloat = 16
    var color: Color = .secondary

    var body: some View {
        Circle()
            .trim(from: 0.1, to: 0.9)
            .stroke(color, style: StrokeStyle(lineWidth: max(1, size * 0.12), lineCap: .round))
            .frame(width: size, height: size)
            .rotationEffect(.degrees(rotation))
            .onAppear {
                withAnimation(.linear(duration: 0.8).repeatForever(autoreverses: false)) {
                    rotation = 360
                }
            }
    }
}

struct ThumbnailCell: View {
    @ObservedObject var record: ImageRecord
    let isSelected: Bool
    let onSelect: (NSEvent.ModifierFlags) -> Void

    @State private var thumbnail: NSImage?

    private var isProcessing: Bool {
        record.processState == ProcessState.culling || record.processState == ProcessState.rating
    }

    private var processLabel: String {
        record.processState == ProcessState.culling ? "Culling" : "Rating"
    }

    var body: some View {
        ZStack(alignment: .bottomTrailing) {
            Group {
                if let thumb = thumbnail {
                    Image(nsImage: thumb)
                        .resizable()
                        .aspectRatio(contentMode: .fill)
                } else {
                    Rectangle().fill(Color.secondary.opacity(0.2))
                        .overlay(SpinnerView(size: 20))
                }
            }
            .frame(width: 160, height: 110)
            .clipped()
            .cornerRadius(6)
            .overlay(
                RoundedRectangle(cornerRadius: 6)
                    .stroke(
                        isSelected ? Color.accentColor : (record.cullRejected ? Color.red.opacity(0.6) : Color.clear),
                        lineWidth: 2
                    )
            )
            .overlay {
                if isProcessing {
                    ZStack {
                        Color.black.opacity(0.45)
                        VStack(spacing: 6) {
                            SpinnerView(size: 26, color: .white)
                            Text(processLabel)
                                .font(.caption2)
                                .fontWeight(.semibold)
                                .foregroundStyle(.white)
                        }
                    }
                    .cornerRadius(6)
                }
            }
            .overlay(alignment: .bottomLeading) {
                if record.clusterRank == 1 && record.clusterID >= 0 {
                    Text("C")
                        .font(.system(size: 8, weight: .bold))
                        .foregroundStyle(.white)
                        .padding(.horizontal, 3).padding(.vertical, 1)
                        .background(Color.purple.opacity(0.8))
                        .clipShape(RoundedRectangle(cornerRadius: 2))
                        .padding(4)
                }
            }
            .overlay {
                if record.diversityFactor > 0 && record.diversityFactor < 0.60 {
                    Color.black.opacity(0.30)
                        .cornerRadius(6)
                        .allowsHitTesting(false)
                }
            }

            ScoreBadge(
                stars: Int(record.userOverride?.int16Value ?? record.ratingStars?.int16Value ?? 0),
                rejected: record.cullRejected,
                isManual: (record.userOverride?.int16Value ?? 0) > 0
            )
            .padding(4)
        }
        .onTapGesture {
            let mods = NSApp.currentEvent?.modifierFlags ?? []
            onSelect(mods)
        }
        .task(id: record.objectID) {
            let url = URL(filePath: record.filePath ?? "")
            thumbnail = await ThumbnailCache.shared.thumbnail(for: url, size: CGSize(width: 160, height: 110))
        }
    }
}
