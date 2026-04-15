import SwiftUI
import CoreData

/// Returns current wall-clock time as "HH:mm:ss.SSS" for log correlation.
private func ts() -> String {
    var tv = timeval()
    gettimeofday(&tv, nil)
    let ms = tv.tv_usec / 1000
    let s  = Int(tv.tv_sec) % 60
    let m  = (Int(tv.tv_sec) / 60) % 60
    let h  = (Int(tv.tv_sec) / 3600) % 24
    return String(format: "%02d:%02d:%02d.%03d", h, m, s, ms)
}

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
            let t0 = CFAbsoluteTimeGetCurrent()
            let name = URL(filePath: record.filePath ?? "").lastPathComponent
            let url = URL(filePath: record.filePath ?? "")
            print("\(ts()) [detail] \(name) task-start")

            imageMeta = nil
            zoomScale = 1.0

            // Placeholder: largest in-memory variant (usually the grid thumb). No I/O.
            if let cached = await ThumbnailCache.shared.bestAvailableCached(for: url) {
                fullImage = cached
                print("\(ts()) [detail] \(name) placeholder \(Int((CFAbsoluteTimeGetCurrent()-t0)*1000))ms")
            }

            // Hi-res — single decode pass. No intermediate 800px step.
            if let hi = await ThumbnailCache.shared.thumbnail(
                for: url, size: CGSize(width: 1600, height: 1600)) {
                fullImage = hi
                print("\(ts()) [detail] \(name) 1600px \(Int((CFAbsoluteTimeGetCurrent()-t0)*1000))ms")
            }

            imageMeta = ImageMetadata.read(from: url)
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

                if record.musiqAesthetic > 0 {
                    metaSectionDivider("AI Score")
                    metaRow("Aesthetic", String(format: "%.2f", record.musiqAesthetic))
                    if let s = record.ratingStars, s.int16Value > 0 {
                        metaRow("AI stars", String(repeating: "★", count: Int(s.int16Value)))
                    }
                    if let o = record.userOverride, o.int16Value > 0 {
                        metaRow("Manual", String(repeating: "★", count: Int(o.int16Value)))
                    }
                }

                // CHARACTERISTICS — shown once cull + diversity pass have run
                if record.blurScore > 0 || record.clusterRank > 0 {
                    metaSectionDivider("Characteristics")
                    if record.blurScore > 0 {
                        let blurLabel = record.blurScore > 300 ? "Sharp"
                                      : record.blurScore > 100 ? "Soft" : "Blurry"
                        metaRow("Blur", blurLabel)
                    }
                    if record.exposureScore != 0 {
                        metaRow("Exposure", exposureLabel(record.exposureScore))
                    }
                    if record.clusterRank > 0, let ctx = record.managedObjectContext {
                        let size = clusterSize(id: record.clusterID, in: ctx)
                        metaRow("Cluster", "#\(record.clusterID) · rank \(record.clusterRank) of \(size)")
                        metaRow("Diversity", String(format: "%.2f×", record.diversityFactor))
                    }
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

    private func exposureLabel(_ score: Float) -> String {
        switch score {
        case let s where s >  1.5: return String(format: "+%.1f EV (over)", s)
        case let s where s < -1.5: return String(format: "%.1f EV (under)", s)
        case let s where s >  0.3: return String(format: "+%.1f EV", s)
        case let s where s < -0.3: return String(format: "%.1f EV", s)
        default: return "Normal"
        }
    }

    private func clusterSize(id: Int32, in ctx: NSManagedObjectContext) -> Int {
        let req = ImageRecord.fetchRequest()
        req.predicate = NSPredicate(format: "clusterID == %d", id)
        return (try? ctx.count(for: req)) ?? 0
    }
}

