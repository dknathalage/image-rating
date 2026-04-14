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
    let cellSize: CGFloat
    let onSelect: (NSEvent.ModifierFlags) -> Void
    let onDoubleClick: () -> Void

    @State private var thumbnail: NSImage?

    private var isProcessing: Bool {
        record.processState == ProcessState.culling || record.processState == ProcessState.rating
    }

    private var processLabel: String {
        record.processState == ProcessState.culling ? "Culling" : "Rating"
    }

    var body: some View {
        ZStack(alignment: .bottomTrailing) {
            imageLayer
                .frame(width: cellSize, height: cellSize * 0.6875)
                .clipped()
                .cornerRadius(6)
                .overlay(
                    RoundedRectangle(cornerRadius: 6)
                        .stroke(isSelected ? Color.accentColor : Color.clear, lineWidth: 2)
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
                .overlay(alignment: .topLeading) {
                    if record.scoringFilePath != nil {
                        Text("RAW")
                            .font(.system(size: 7, weight: .bold))
                            .foregroundStyle(.white)
                            .padding(.horizontal, 3).padding(.vertical, 1)
                            .background(Color.orange.opacity(0.85))
                            .clipShape(RoundedRectangle(cornerRadius: 2))
                            .padding(4)
                    }
                }
                .overlay(alignment: .bottom) {
                    ratingsOverlay
                }
        }
        .onTapGesture(count: 2) { onDoubleClick() }
        .onTapGesture {
            onSelect(NSApp.currentEvent?.modifierFlags ?? [])
        }
        .task(id: "\(record.objectID.uriRepresentation())-\(cellSize)") {
            let url = URL(filePath: record.filePath ?? "")
            let img = await ThumbnailCache.shared.thumbnail(
                for: url,
                size: CGSize(width: cellSize, height: cellSize * 0.6875)
            )
            guard !Task.isCancelled else { return }
            thumbnail = img
        }
    }

    // MARK: - Subviews

    @ViewBuilder
    private var imageLayer: some View {
        if let thumb = thumbnail {
            Image(nsImage: thumb)
                .resizable()
                .aspectRatio(contentMode: .fill)
        } else {
            Rectangle()
                .fill(Color.secondary.opacity(0.2))
                .overlay(SpinnerView(size: 20))
        }
    }

    @ViewBuilder
    private var ratingsOverlay: some View {
        let aiStars = Int(record.ratingStars?.int16Value ?? 0)
        let manualStars = Int(record.userOverride?.int16Value ?? 0)
        let hasAI = aiStars > 0
        let hasManual = manualStars > 0
        if hasAI || hasManual {
            HStack(spacing: 4) {
                if hasAI {
                    HStack(spacing: 1) {
                        Text("AI")
                            .font(.system(size: max(6, cellSize * 0.055), weight: .semibold))
                            .foregroundStyle(.white.opacity(0.75))
                        Text(String(repeating: "★", count: aiStars))
                            .font(.system(size: max(6, cellSize * 0.055)))
                            .foregroundStyle(.white.opacity(0.85))
                    }
                }
                if hasAI && hasManual {
                    Text("·")
                        .font(.system(size: max(6, cellSize * 0.055)))
                        .foregroundStyle(.white.opacity(0.5))
                }
                if hasManual {
                    HStack(spacing: 1) {
                        Text(String(repeating: "★", count: manualStars))
                            .font(.system(size: max(6, cellSize * 0.055)))
                            .foregroundStyle(.yellow)
                        Text("M")
                            .font(.system(size: max(6, cellSize * 0.055), weight: .semibold))
                            .foregroundStyle(.yellow.opacity(0.9))
                    }
                }
            }
            .padding(.horizontal, 5)
            .padding(.vertical, 2)
            .background(.black.opacity(0.55))
            .clipShape(RoundedRectangle(cornerRadius: 3))
            .padding(.bottom, 4)
        }
    }
}
