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
    let onRate: (Int) -> Void

    @State private var thumbnail: NSImage?
    @State private var isHovering = false
    @State private var hoverStar: Int = 0

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
            // Hover star rating bar
            .overlay(alignment: .bottom) {
                if isHovering {
                    HStack(spacing: 3) {
                        ForEach(1...5, id: \.self) { star in
                            Image(systemName: star <= hoverStar ? "star.fill" : "star")
                                .font(.system(size: max(8, cellSize * 0.07)))
                                .foregroundStyle(.white)
                                .onHover { inside in if inside { hoverStar = star } }
                                .onTapGesture { onRate(star) }
                        }
                    }
                    .padding(.horizontal, 6)
                    .padding(.vertical, 3)
                    .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 4))
                    .padding(.bottom, 6)
                }
            }
            // Rating overlay — bottom strip showing AI + manual ratings
            .overlay(alignment: .bottom) {
                if !isHovering {
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
        }
        .onHover { inside in
            isHovering = inside
            if !inside { hoverStar = 0 }
        }
        .onTapGesture(count: 2) {
            onDoubleClick()
        }
        .onTapGesture {
            let mods = NSApp.currentEvent?.modifierFlags ?? []
            onSelect(mods)
        }
        .task(id: "\(record.objectID)-\(cellSize)") {
            let url = URL(filePath: record.filePath ?? "")
            thumbnail = await ThumbnailCache.shared.thumbnail(for: url, size: CGSize(width: cellSize, height: cellSize * 0.6875))
        }
    }
}
