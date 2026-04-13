import SwiftUI

struct DetailView: View {
    @ObservedObject var record: ImageRecord
    let onPrev: () -> Void
    let onNext: () -> Void
    let onRate: (Int) -> Void

    @State private var fullImage: NSImage?
    @State private var imageMeta: ImageMetadata?
    @State private var zoomScale: CGFloat = 1.0
    @State private var fitTrigger = 0

    var body: some View {
        HStack(spacing: 0) {
            imagePane
            Divider()
            metadataPane
                .frame(width: 260)
        }
        .task(id: record.objectID) {
            fullImage = nil
            imageMeta = nil
            zoomScale = 1.0
            let url = URL(filePath: record.filePath ?? "")
            async let thumb = ThumbnailCache.shared.thumbnail(
                for: url, size: CGSize(width: 2000, height: 2000)
            )
            async let meta = Task { ImageMetadata.read(from: url) }.value
            fullImage = await thumb
            imageMeta = await meta
        }
    }

    // MARK: - Image pane

    private var imagePane: some View {
        ZStack(alignment: .bottomTrailing) {
            ZoomableImageView(image: fullImage, scale: $zoomScale, fitTrigger: fitTrigger)

            HStack(spacing: 2) {
                Button { zoomScale = max(0.05, zoomScale / 1.4) } label: {
                    Image(systemName: "minus.magnifyingglass")
                }
                Button { fitTrigger += 1 } label: {
                    Image(systemName: "arrow.up.left.and.arrow.down.right")
                }
                Button { zoomScale = min(8.0, zoomScale * 1.4) } label: {
                    Image(systemName: "plus.magnifyingglass")
                }
            }
            .buttonStyle(.borderless)
            .padding(.horizontal, 8)
            .padding(.vertical, 5)
            .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 6))
            .padding(10)
        }
    }

    // MARK: - Metadata pane

    private var metadataPane: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 0) {

                // Filename + rating
                VStack(alignment: .leading, spacing: 6) {
                    Text(URL(filePath: record.filePath ?? "").lastPathComponent)
                        .font(.headline)
                        .lineLimit(3)
                        .fixedSize(horizontal: false, vertical: true)
                    ScoreBadge(
                        stars: effectiveStars,
                        rejected: record.cullRejected,
                        isManual: isManuallyRated
                    )
                }
                .padding(.bottom, 12)

                metaSectionDivider("Capture")

                if let d = imageMeta?.dateTakenString {
                    metaRow("Date", d)
                }

                metaSectionDivider("Camera")

                Group {
                    if let make = imageMeta?.cameraMake, let model = imageMeta?.cameraModel {
                        let cam = model.hasPrefix(make) ? model : "\(make) \(model)"
                        metaRow("Body", cam)
                    } else {
                        metaRow("Body", imageMeta?.cameraModel ?? imageMeta?.cameraMake)
                    }
                    metaRow("Lens", imageMeta?.lens)
                    metaRow("Focal", imageMeta?.focalLengthString)
                }

                metaSectionDivider("Exposure")

                Group {
                    metaRow("Aperture", imageMeta?.apertureString)
                    metaRow("Shutter", imageMeta?.shutterString)
                    metaRow("ISO", imageMeta?.isoString)
                    metaRow("Comp.", imageMeta?.exposureBiasString)
                    metaRow("Program", imageMeta?.exposureProgram)
                    metaRow("Metering", imageMeta?.meteringMode)
                    metaRow("Flash", imageMeta?.flash)
                    metaRow("WB", imageMeta?.whiteBalance)
                }

                metaSectionDivider("File")

                Group {
                    metaRow("Size", imageMeta?.dimensionsString)
                    if let w = imageMeta?.pixelWidth, let h = imageMeta?.pixelHeight {
                        metaRow("MP", String(format: "%.1f MP", Double(w * h) / 1_000_000))
                    }
                    metaRow("File", imageMeta?.fileSizeString)
                }

                if record.processState != "done" || record.cullRejected || record.decodeError {
                    metaSectionDivider("Status")
                    if record.processState != "done" {
                        metaRow("State", record.processState == "pending" ? "Awaiting" : "Processing…")
                    }
                    if record.cullRejected {
                        metaRow("Reject", record.cullReason ?? "unknown")
                    }
                    if record.decodeError {
                        metaRow("Error", "Decode failed")
                    }
                }

                if (record.ratingStars?.int16Value ?? 0) > 0 {
                    metaSectionDivider("AI Scores")
                    metaRow("Technical", String(format: "%.2f", record.clipScore?.floatValue ?? 0))
                    metaRow("Aesthetic", String(format: "%.2f", record.aestheticScore?.floatValue ?? 0))
                }

                metaSectionDivider("Rating")

                Picker("", selection: overrideBinding) {
                    Text("AI").tag(Int16(0))
                    ForEach(1...5, id: \.self) { s in
                        Text("\(s) ★").tag(Int16(s))
                    }
                }
                .pickerStyle(.segmented)
                .padding(.bottom, 12)

                Divider().padding(.bottom, 8)

                HStack(spacing: 8) {
                    Button(action: onPrev) {
                        Label("Prev", systemImage: "chevron.left")
                    }
                    .buttonStyle(.bordered)
                    Button(action: onNext) {
                        Label("Next", systemImage: "chevron.right")
                    }
                    .buttonStyle(.bordered)
                }
                .padding(.bottom, 6)

                Text("← → navigate · 1–5 rate · 0 clear")
                    .font(.caption2)
                    .foregroundColor(.secondary)
            }
            .padding(14)
        }
    }

    // MARK: - Helpers

    @ViewBuilder
    private func metaRow(_ label: String, _ value: String?) -> some View {
        if let value, !value.isEmpty {
            HStack(alignment: .top, spacing: 6) {
                Text(label)
                    .font(.caption2)
                    .foregroundColor(.secondary)
                    .frame(width: 68, alignment: .leading)
                Text(value)
                    .font(.caption)
                    .lineLimit(2)
                Spacer(minLength: 0)
            }
            .padding(.vertical, 1)
        }
    }

    private func metaSectionDivider(_ title: String) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Divider()
            Text(title.uppercased())
                .font(.system(size: 9, weight: .semibold))
                .foregroundColor(.secondary)
                .tracking(0.8)
        }
        .padding(.top, 8)
        .padding(.bottom, 4)
    }

    private var effectiveStars: Int {
        if let o = record.userOverride?.int16Value, o > 0 { return Int(o) }
        return Int(record.ratingStars?.int16Value ?? 0)
    }

    private var isManuallyRated: Bool {
        (record.userOverride?.int16Value ?? 0) > 0
    }

    private var overrideBinding: Binding<Int16> {
        Binding(
            get: { record.userOverride?.int16Value ?? 0 },
            set: { onRate(Int($0)) }
        )
    }
}
