import CoreData
import SwiftUI

private func ts() -> String {
    var tv = timeval()
    gettimeofday(&tv, nil)
    let ms = tv.tv_usec / 1000
    let s  = Int(tv.tv_sec) % 60
    let m  = (Int(tv.tv_sec) / 60) % 60
    let h  = (Int(tv.tv_sec) / 3600) % 24
    return String(format: "%02d:%02d:%02d.%03d", h, m, s, ms)
}

private final class KeyboardHandler: ObservableObject {
    var onPrev: (() -> Void)?
    var onNext: (() -> Void)?
    var onRate: ((Int) -> Void)?
    var onToggleModal: (() -> Void)?
    var onReject: (() -> Void)?
    private var monitor: Any?

    func start() {
        guard monitor == nil else { return }
        monitor = NSEvent.addLocalMonitorForEvents(matching: .keyDown) { [weak self] event in
            guard let self else { return event }
            switch event.keyCode {
            case 123: self.onPrev?(); return nil         // ←
            case 124: self.onNext?(); return nil         // →
            case 49:  self.onToggleModal?(); return nil  // space
            case 7:   self.onReject?(); return nil       // x
            default:
                guard let ch = event.charactersIgnoringModifiers,
                      let n = Int(ch), (0...5).contains(n) else { return event }
                self.onRate?(n)
                return nil
            }
        }
    }

    func stop() {
        if let m = monitor { NSEvent.removeMonitor(m); monitor = nil }
    }
}

struct ContentView: View {
    @Environment(\.managedObjectContext) private var ctx
    @FetchRequest(sortDescriptors: [SortDescriptor(\.createdAt, order: .reverse)])
    private var sessions: FetchedResults<Session>

    @FetchRequest(
        sortDescriptors: [SortDescriptor(\.filePath, order: .forward)],
        predicate: NSPredicate(value: false)
    )
    private var sessionImages: FetchedResults<ImageRecord>

    @State private var selectedSession: Session?
    @State private var selectedIDs: Set<NSManagedObjectID> = []
    @State private var anchorID: NSManagedObjectID?
    @State private var showModelStore = false
    @State private var processingTask: Task<Void, Never>?
    @State private var processingError: String?
    @State private var processingStatus: String?
    @State private var processingDone: Int = 0
    @State private var processingTotal: Int = 0
    @State private var showResetConfirm = false
    @State private var sessionToRemove: Session? = nil
    @State private var ratingFilter: Set<Int> = []
    @State private var showProcessingSheet = false
    @AppStorage(FocalSettings.defaultCellSize) private var cellSizeValue: Double = FocalSettings.defaultCellSizeValue
    @AppStorage(FocalSettings.autoWriteXMP) private var autoWriteXMP: Bool = FocalSettings.defaultAutoWriteXMP
    private var cellSize: CGFloat { CGFloat(cellSizeValue) }
    @State private var detailRecord: ImageRecord? = nil
    @State private var showAIProgressSheet = false
    @State private var showCompareSheet = false
    @StateObject private var keyboard = KeyboardHandler()
    @State private var ratingQueue = RatingQueue()
    @StateObject private var ratingProgress = RatingProgress()

    // MARK: - Sidebar

    @ViewBuilder private var sidebar: some View {
        List(selection: $selectedSession) {
            Section("Sessions") {
                ForEach(sessions) { session in
                    Label(
                        URL(filePath: session.folderPath ?? "").lastPathComponent,
                        systemImage: "folder"
                    )
                    .tag(session)
                    .contextMenu {
                        Button(role: .destructive) {
                            sessionToRemove = session
                        } label: {
                            Label("Remove from Library", systemImage: "trash")
                        }
                    }
                }
            }
            .confirmationDialog(
                "Remove \"\(URL(filePath: sessionToRemove?.folderPath ?? "").lastPathComponent)\"?",
                isPresented: Binding(get: { sessionToRemove != nil }, set: { if !$0 { sessionToRemove = nil } }),
                titleVisibility: .visible
            ) {
                Button("Remove", role: .destructive) {
                    if let s = sessionToRemove { removeSession(s) }
                }
                Button("Cancel", role: .cancel) {}
            } message: {
                Text("Removes the session and all ratings from this app. Files on disk are not affected.")
            }
            Section("Filter by Rating") {
                RatingFilterView(
                    images: Array(sessionImages),
                    ratingFilter: $ratingFilter
                )
                .selectionDisabled()
            }
        }
        .navigationTitle("Sessions")
    }

    // MARK: - Content panel

    @ViewBuilder private var contentPanel: some View {
        if let session = selectedSession {
            GridView(
                images: filteredImages,
                sessionHasImages: !sessionImages.isEmpty,
                selectedIDs: $selectedIDs,
                anchorID: $anchorID,
                cellSize: cellSize,
                onDoubleClick: { record in
                    anchorID = record.objectID
                    selectedIDs = [record.objectID]
                    detailRecord = record
                    prefetchDetailImages(around: record.objectID)
                },
                onRemoveRatings: { removeRatings() },
                onRunAI: { runAIOnSelected() }
            )
            .safeAreaInset(edge: .bottom) {
                VStack(spacing: 0) {
                    if ratingProgress.isActive {
                        RatingProgressBar(progress: ratingProgress)
                    }
                    if let status = processingStatus, !showAIProgressSheet {
                        ProcessingStatusBar(
                            status: status,
                            done: processingDone,
                            total: processingTotal,
                            onCancel: { processingTask?.cancel() }
                        )
                    }
                }
            }
            // Full-session processing setup
            .sheet(isPresented: $showProcessingSheet) {
                ProcessingSetupSheet {
                    runPipeline(session: session)
                }
            }
            // Detail modal (double-click / space)
            .sheet(isPresented: Binding(
                get: { detailRecord != nil },
                set: { if !$0 { detailRecord = nil } }
            )) {
                if let record = detailRecord {
                    DetailView(record: record, onPrev: navigatePrev, onNext: navigateNext, onRate: setRating)
                        .frame(minWidth: 900, minHeight: 600)
                }
            }
            // AI-on-selected progress
            .sheet(isPresented: $showAIProgressSheet) {
                AIProgressSheet(
                    status: processingStatus ?? "Processing…",
                    done: processingDone,
                    total: processingTotal,
                    onCancel: {
                        processingTask?.cancel()
                    }
                )
            }
            // Compare sheet (2–3 selected images)
            .sheet(isPresented: $showCompareSheet) {
                CompareView(records: selectedRecords) { record, stars in
                    rateRecordDirect(record, stars: stars)
                }
            }
            .confirmationDialog("Reset all ratings?",
                                isPresented: $showResetConfirm,
                                titleVisibility: .visible) {
                Button("Reset", role: .destructive) { resetSession(session) }
                Button("Cancel", role: .cancel) {}
            } message: {
                Text("Clears all AI ratings, cull results, and sidecar files. Manual star overrides are preserved.")
            }
        } else {
            ContentUnavailableView("Open a Folder", systemImage: "folder.badge.plus",
                                    description: Text("Use the toolbar button to import images"))
        }
    }

    var body: some View {
        splitView
            .sheet(isPresented: $showModelStore) { ModelStoreView() }
            .alert("Processing Failed", isPresented: processingErrorBinding) {
                Button("OK") { processingError = nil }
            } message: {
                Text(processingError ?? "")
            }
            .onAppear(perform: handleAppear)
            .onDisappear { keyboard.stop(); ratingQueue.flush() }
            .onChange(of: selectedSession) { _, session in
                ratingQueue.flush()
                if let session {
                    loadRatingsFromDisk(for: session)
                    writeRatingsInBackground(for: session)
                }
            }
            .onChange(of: anchorID) { _, id in
                guard detailRecord != nil, let id else { return }
                detailRecord = ctx.object(with: id) as? ImageRecord
                prefetchDetailImages(around: id)
            }
            .modifier(FilterChangeModifier(
                selectedSession: $selectedSession,
                ratingFilter: $ratingFilter,
                anchorID: $anchorID,
                selectedIDs: $selectedIDs,
                filteredImages: filteredImages,
                sessionImages: sessionImages
            ))
    }

    private var processingErrorBinding: Binding<Bool> {
        Binding(
            get: { processingError != nil },
            set: { if !$0 { processingError = nil } }
        )
    }

    @ViewBuilder private var splitView: some View {
        NavigationSplitView {
            sidebar
        } detail: {
            contentPanel
        }
        .toolbar {
            ToolbarItem(placement: .navigation) {
                Button(action: openFolder) {
                    Label("Open Folder", systemImage: "folder.badge.plus")
                }
            }
            ToolbarItem(placement: .navigation) {
                Button { showModelStore = true } label: {
                    Label("Models", systemImage: "square.and.arrow.down")
                }
            }
            if selectedSession != nil {
                ToolbarItem(placement: .primaryAction) {
                    Button(action: { showProcessingSheet = true }) {
                        Label("Process", systemImage: "wand.and.stars")
                    }
                    .disabled(processingStatus != nil)
                }
                ToolbarItem(placement: .primaryAction) {
                    Button(action: { showResetConfirm = true }) {
                        Label("Reset", systemImage: "arrow.counterclockwise")
                    }
                    .disabled(processingStatus != nil)
                }
                ToolbarItem(placement: .primaryAction) {
                    Button(action: { if let s = selectedSession { exportScores(session: s) } }) {
                        Label("Export", systemImage: "square.and.arrow.up")
                    }
                    .disabled(processingStatus != nil)
                }
                if selectedIDs.count >= 2 && selectedIDs.count <= 3 {
                    ToolbarItem(placement: .primaryAction) {
                        Button(action: { showCompareSheet = true }) {
                            Label("Compare", systemImage: "rectangle.split.2x1")
                        }
                    }
                }
                ToolbarItem(placement: .primaryAction) {
                    ControlGroup {
                        Button(action: { cellSizeValue = max(100, cellSizeValue - 30) }) {
                            Image(systemName: "minus")
                        }
                        .disabled(cellSizeValue <= 100)
                        Button(action: { cellSizeValue = min(320, cellSizeValue + 30) }) {
                            Image(systemName: "plus")
                        }
                        .disabled(cellSizeValue >= 320)
                    }
                    .help("Thumbnail size")
                }
            }
        }
    }

    private func handleAppear() {
        ratingQueue.configure(
            context: ctx,
            container: PersistenceController.shared.container,
            progress: ratingProgress
        )
        if let session = selectedSession {
            loadRatingsFromDisk(for: session)
            writeRatingsInBackground(for: session)
        }
        keyboard.onPrev = navigatePrev
        keyboard.onNext = navigateNext
        keyboard.onRate = setRating
        keyboard.onToggleModal = {
            if detailRecord != nil {
                detailRecord = nil
            } else if let record = anchorRecord {
                detailRecord = record
            }
        }
        keyboard.onReject = { setRating(1) }
        keyboard.start()
    }

    // MARK: - Navigation

    private func effectiveRating(_ record: ImageRecord) -> Int {
        if let o = record.userOverride, o.int16Value > 0 { return Int(o.int16Value) }
        if let s = record.ratingStars { return Int(s.int16Value) }
        return 0
    }

    private var filteredImages: [ImageRecord] {
        // sessionImages predicate-filtered to isGroupPrimary == YES at DB level.
        if ratingFilter.isEmpty { return Array(sessionImages) }
        return sessionImages.filter { ratingFilter.contains(effectiveRating($0)) }
    }

    private var anchorRecord: ImageRecord? {
        guard let id = anchorID else { return nil }
        return ctx.object(with: id) as? ImageRecord
    }

    private var selectedRecords: [ImageRecord] {
        selectedIDs.compactMap { ctx.object(with: $0) as? ImageRecord }
    }

    private func navigateNext() {
        let imgs = filteredImages
        guard !imgs.isEmpty else { return }
        guard let cur = anchorID,
              let idx = imgs.firstIndex(where: { $0.objectID == cur }) else {
            let first = imgs.first
            anchorID = first?.objectID
            selectedIDs = first.map { [$0.objectID] } ?? []
            return
        }
        if idx + 1 < imgs.count {
            let next = imgs[idx + 1]
            anchorID = next.objectID
            selectedIDs = [next.objectID]
        }
    }

    private func navigatePrev() {
        let imgs = filteredImages
        guard !imgs.isEmpty else { return }
        guard let cur = anchorID,
              let idx = imgs.firstIndex(where: { $0.objectID == cur }),
              idx > 0 else { return }
        let prev = imgs[idx - 1]
        anchorID = prev.objectID
        selectedIDs = [prev.objectID]
    }

    // MARK: - Detail pre-fetch

    @State private var prefetchTask: Task<Void, Never>?

    private func prefetchDetailImages(around id: NSManagedObjectID) {
        let imgs = filteredImages
        guard let idx = imgs.firstIndex(where: { $0.objectID == id }) else { return }
        let urls: [URL] = [-1, 1, 2].compactMap { offset -> URL? in
            let i = idx + offset
            guard i >= 0, i < imgs.count, let path = imgs[i].filePath else { return nil }
            return URL(filePath: path)
        }
        guard !urls.isEmpty else { return }
        // Cancel previous pre-fetch — stops it from enqueueing more ops into prefetchQueue.
        prefetchTask?.cancel()
        // Routes through prefetchQueue (isolated from decodeQueue) so pre-fetch never
        // blocks user-visible detail-view loads.
        prefetchTask = Task.detached(priority: .background) {
            for url in urls {
                guard !Task.isCancelled else { return }
                _ = await ThumbnailCache.shared.thumbnail(for: url, size: CGSize(width: 800, height: 800), isPrefetch: true)
                guard !Task.isCancelled else { return }
                _ = await ThumbnailCache.shared.thumbnail(for: url, size: CGSize(width: 1600, height: 1600), isPrefetch: true)
            }
        }
    }

    // MARK: - Background XMP sweep

    /// Write XMP metadata for all rated images in the session.
    /// Runs at .background priority — safe to call on every session load.
    private func writeRatingsInBackground(for session: Session) {
        guard let images = session.images?.allObjects as? [ImageRecord] else { return }
        let tasks: [(URL, Int)] = images.compactMap { record in
            guard let path = record.filePath else { return nil }
            let stars: Int
            if let o = record.userOverride, o.int16Value > 0 {
                stars = Int(o.int16Value)
            } else if let s = record.ratingStars, s.int16Value > 0 {
                stars = Int(s.int16Value)
            } else {
                return nil
            }
            return (URL(filePath: path), stars)
        }
        guard !tasks.isEmpty else { return }
        Task.detached(priority: .background) {
            for (url, stars) in tasks {
                try? MetadataWriter.writeSidecar(stars: stars, for: url)
            }
        }
    }

    /// Reverse sweep: read XMP / embedded ratings from disk and seed userOverride for
    /// records that don't have one yet. Runs on background context to avoid main-thread
    /// I/O. Picks up ratings from prior app sessions, external tools (Photomator, Lightroom),
    /// or records imported before the createRecord seed logic existed.
    private func loadRatingsFromDisk(for session: Session) {
        let sessionID = session.objectID
        let container = PersistenceController.shared.container
        let progress = ratingProgress
        Task.detached(priority: .utility) {
            container.performBackgroundTask { bgCtx in
                guard let session = try? bgCtx.existingObject(with: sessionID) as? Session,
                      let images = session.images?.allObjects as? [ImageRecord] else { return }
                let total = images.count
                DispatchQueue.main.async { progress.start(phase: "Loading ratings…", total: total) }
                var changed = 0
                for record in images {
                    defer { DispatchQueue.main.async { progress.tick() } }
                    if let o = record.userOverride, o.int16Value > 0 { continue }
                    guard let path = record.filePath else { continue }
                    let stars = MetadataWriter.readRating(for: URL(filePath: path))
                    if (1...5).contains(stars) {
                        record.userOverride = NSNumber(value: Int16(stars))
                        changed += 1
                    } else if stars == -1 && !record.cullRejected {
                        record.cullRejected = true
                        record.cullReason = "rejected"
                        changed += 1
                    }
                }
                if changed > 0, bgCtx.hasChanges {
                    try? bgCtx.save()
                }
                DispatchQueue.main.async { progress.finish() }
            }
        }
    }

    // MARK: - Rating

    private func setRating(_ stars: Int) {
        guard !selectedIDs.isEmpty else { return }

        // Pre-compute next before in-memory update so filter-disappear doesn't reset to nothing
        let nextID: NSManagedObjectID? = selectedIDs.count == 1 ? {
            let imgs = filteredImages
            guard let cur = anchorID,
                  let idx = imgs.firstIndex(where: { $0.objectID == cur }),
                  idx + 1 < imgs.count else { return nil }
            return imgs[idx + 1].objectID
        }() : nil

        let t0 = CFAbsoluteTimeGetCurrent()
        for id in selectedIDs {
            guard let record = ctx.object(with: id) as? ImageRecord else { continue }
            record.userOverride = stars == 0 ? nil : NSNumber(value: Int16(stars))
            let effective = stars > 0 ? stars : Int(record.ratingStars?.int16Value ?? 0)
            ratingQueue.enqueue(id: id, stars: effective)
        }

        // Auto-advance single selection
        if let next = nextID {
            anchorID = next
            selectedIDs = [next]
        }
    }

    /// Rate a single record directly — used by compare mode, no auto-advance, no selection change.
    private func rateRecordDirect(_ record: ImageRecord, stars: Int) {
        record.userOverride = stars == 0 ? nil : NSNumber(value: Int16(stars))
        let effective = stars > 0 ? stars : Int(record.ratingStars?.int16Value ?? 0)
        ratingQueue.enqueue(id: record.objectID, stars: effective)
    }

    // MARK: - Actions

    private func removeRatings() {
        guard !selectedIDs.isEmpty else { return }
        for id in selectedIDs {
            guard let record = ctx.object(with: id) as? ImageRecord else { continue }
            record.userOverride = nil
            record.ratingStars = nil
            record.clipScore = nil
            record.aestheticScore = nil
            record.musiqAesthetic = 0
            record.cullRejected = false
            record.cullReason = nil
            record.processState = ProcessState.pending
            if let path = record.filePath {
                let xmp = URL(filePath: path).deletingPathExtension().appendingPathExtension("xmp")
                try? FileManager.default.removeItem(at: xmp)
            }
            if let gid = record.groupID,
               let allImages = record.session?.images?.allObjects as? [ImageRecord] {
                for c in allImages where c.groupID == gid && !c.isGroupPrimary {
                    c.userOverride = nil
                    c.ratingStars = nil
                    if let cp = c.filePath {
                        let xmp = URL(filePath: cp).deletingPathExtension().appendingPathExtension("xmp")
                        try? FileManager.default.removeItem(at: xmp)
                    }
                }
            }
        }
        try? ctx.save()
    }

    private func runAIOnSelected() {
        guard !selectedIDs.isEmpty, processingStatus == nil else { return }
        let ids = Array(selectedIDs)
        processingTask?.cancel()
        processingError = nil
        processingStatus = "Preparing…"
        processingDone = 0
        processingTotal = 0
        showAIProgressSheet = true
        processingTask = Task {
            let queue = ProcessingQueue(context: ctx)
            do {
                try await queue.process(imageIDs: ids) { done, total, status in
                    Task { @MainActor in
                        processingDone = done
                        processingTotal = total
                        processingStatus = status
                    }
                }
                await MainActor.run {
                    processingStatus = nil
                    showAIProgressSheet = false
                }
            } catch is CancellationError {
                await MainActor.run {
                    processingStatus = nil
                    showAIProgressSheet = false
                }
            } catch {
                await MainActor.run {
                    processingStatus = nil
                    showAIProgressSheet = false
                    processingError = error.localizedDescription
                }
            }
        }
    }

    private func openFolder() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = false
        if panel.runModal() == .OK, let url = panel.url {
            do {
                let sessionID = try ImageImporter.importFolder(url, context: ctx)
                selectedSession = ctx.object(with: sessionID) as? Session
            } catch {
                NSApp.presentError(error)
            }
        }
    }

    private func runPipeline(session: Session) {
        processingTask?.cancel()
        processingError = nil
        processingStatus = "Preparing…"
        processingDone = 0
        processingTotal = 0
        processingTask = Task {
            let queue = ProcessingQueue(context: ctx)
            do {
                try await queue.process(sessionID: session.objectID) { done, total, status in
                    Task { @MainActor in
                        processingDone = done
                        processingTotal = total
                        processingStatus = status
                    }
                }
                await MainActor.run { processingStatus = nil }
            } catch is CancellationError {
                await MainActor.run { processingStatus = nil }
            } catch {
                await MainActor.run {
                    processingStatus = nil
                    processingError = error.localizedDescription
                }
            }
        }
    }

    private func exportScores(session: Session) {
        guard let images = session.images?.allObjects as? [ImageRecord] else { return }
        let allImages = images

        var companionsByGroup: [String: [String]] = [:]
        for img in allImages where !img.isGroupPrimary {
            guard let gid = img.groupID else { continue }
            let name = URL(filePath: img.filePath ?? "").lastPathComponent
            companionsByGroup[gid, default: []].append(name)
        }

        let sorted = allImages
            .filter { $0.isGroupPrimary }
            .sorted { ($0.filePath ?? "") < ($1.filePath ?? "") }

        var lines = ["filename,companions,musiqAesthetic,ratingStars,userOverride,processState,decodeError"]
        for r in sorted {
            let name = URL(filePath: r.filePath ?? "").lastPathComponent
            let companions = r.groupID.flatMap { companionsByGroup[$0] }?.sorted().joined(separator: "|") ?? ""
            let stars = r.ratingStars?.int16Value.description ?? ""
            let override = r.userOverride?.int16Value.description ?? ""
            lines.append([
                name,
                companions,
                String(r.musiqAesthetic),
                stars,
                override,
                r.processState ?? "",
                r.decodeError ? "true" : "false"
            ].joined(separator: ","))
        }
        let csv = lines.joined(separator: "\n")

        let panel = NSSavePanel()
        panel.allowedContentTypes = [.commaSeparatedText]
        panel.nameFieldStringValue = "scores.csv"
        guard panel.runModal() == .OK, let url = panel.url else { return }
        try? csv.write(to: url, atomically: true, encoding: .utf8)
    }

    private func removeSession(_ session: Session) {
        processingTask?.cancel()
        if selectedSession == session {
            processingStatus = nil
            selectedSession = nil
        }
        ctx.delete(session)
        try? ctx.save()
    }

    private func resetSession(_ session: Session) {
        processingTask?.cancel()
        processingStatus = nil
        guard let images = session.images?.allObjects as? [ImageRecord] else { return }
        for record in images {
            record.processState = ProcessState.pending
            record.cullRejected = false
            record.cullReason = nil
            record.ratingStars = nil
            record.clipScore = nil
            record.aestheticScore = nil
            record.musiqAesthetic = 0
            if let path = record.filePath {
                let xmp = URL(filePath: path).deletingPathExtension().appendingPathExtension("xmp")
                try? FileManager.default.removeItem(at: xmp)
            }
        }
        try? ctx.save()
    }
}

// MARK: - Filter change modifier

private struct FilterChangeModifier: ViewModifier {
    @Binding var selectedSession: Session?
    @Binding var ratingFilter: Set<Int>
    @Binding var anchorID: NSManagedObjectID?
    @Binding var selectedIDs: Set<NSManagedObjectID>
    let filteredImages: [ImageRecord]
    let sessionImages: FetchedResults<ImageRecord>

    func body(content: Content) -> some View {
        content
            .onChange(of: selectedSession) { _, session in
                ratingFilter = []
                selectedIDs = []
                anchorID = nil
                sessionImages.nsPredicate = session.map {
                    NSPredicate(format: "session == %@ AND isGroupPrimary == YES", $0)
                } ?? NSPredicate(value: false)
            }
            .onChange(of: ratingFilter) { _, _ in
                guard let id = anchorID else { return }
                if !filteredImages.contains(where: { $0.objectID == id }) {
                    selectedIDs = []
                    anchorID = nil
                }
            }
    }
}

// MARK: - AI Progress Sheet

private struct AIProgressSheet: View {
    let status: String
    let done: Int
    let total: Int
    let onCancel: () -> Void
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        VStack(spacing: 16) {
            Text("Running AI Rating")
                .font(.headline)

            if total > 0 {
                ProgressView(value: Double(done), total: Double(total))
                    .progressViewStyle(.linear)
                Text("\(done) / \(total)")
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
            } else {
                ProgressView()
            }

            Text(status)
                .font(.caption)
                .foregroundStyle(.secondary)

            Button("Cancel") {
                onCancel()
                dismiss()
            }
            .keyboardShortcut(.cancelAction)
        }
        .padding(24)
        .frame(width: 320)
    }
}

// MARK: - Compare View

private struct CompareView: View {
    let records: [ImageRecord]
    let onRate: (ImageRecord, Int) -> Void
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Text("Compare")
                    .font(.headline)
                Spacer()
                Button("Done") { dismiss() }
                    .keyboardShortcut(.defaultAction)
            }
            .padding()
            Divider()
            HStack(spacing: 1) {
                ForEach(records, id: \.objectID) { record in
                    CompareCell(record: record) { stars in
                        onRate(record, stars)
                    }
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
        .frame(minWidth: CGFloat(records.count) * 400, minHeight: 520)
    }
}

private struct CompareCell: View {
    @ObservedObject var record: ImageRecord
    let onRate: (Int) -> Void
    @State private var image: NSImage?

    var body: some View {
        VStack(spacing: 6) {
            Group {
                if let img = image {
                    Image(nsImage: img)
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                } else {
                    Rectangle()
                        .fill(Color.secondary.opacity(0.2))
                        .overlay(SpinnerView(size: 28))
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)

            Text(URL(filePath: record.filePath ?? "").lastPathComponent)
                .font(.caption)
                .lineLimit(1)
                .truncationMode(.middle)
                .padding(.horizontal, 8)

            let stars = Int(record.userOverride?.int16Value ?? record.ratingStars?.int16Value ?? 0)
            ScoreBadge(stars: stars, rejected: record.cullRejected, isManual: (record.userOverride?.int16Value ?? 0) > 0)

            Picker("", selection: Binding(
                get: { record.userOverride?.int16Value ?? 0 },
                set: { onRate(Int($0)) }
            )) {
                Text("AI").tag(Int16(0))
                ForEach(1...5, id: \.self) { s in
                    Text("\(s) ★").tag(Int16(s))
                }
            }
            .pickerStyle(.segmented)
            .padding(.horizontal, 8)
            .padding(.bottom, 8)
        }
        .task(id: record.objectID) {
            let url = URL(filePath: record.filePath ?? "")
            image = await ThumbnailCache.shared.thumbnail(for: url, size: CGSize(width: 1200, height: 900))
        }
    }
}

// MARK: - Processing Setup Sheet

private struct ProcessingSetupSheet: View {
    let onStart: () -> Void
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        VStack(alignment: .leading, spacing: 20) {
            Text("Process Images")
                .font(.headline)

            Text("Run MUSIQ aesthetic scoring on all images in this session. Each image is rated 1–5 stars based on its aesthetic quality.")
                .font(.caption)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)

            HStack {
                Button("Cancel") { dismiss() }
                    .keyboardShortcut(.cancelAction)
                Spacer()
                Button("Start Processing") {
                    onStart()
                    dismiss()
                }
                .keyboardShortcut(.defaultAction)
                .buttonStyle(.borderedProminent)
            }
        }
        .padding(24)
        .frame(width: 400)
    }
}

// MARK: - Rating Progress Bar

private struct RatingProgressBar: View {
    @ObservedObject var progress: RatingProgress

    var body: some View {
        VStack(spacing: 4) {
            HStack {
                Text(progress.phase)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Spacer()
                Text("\(progress.done) / \(progress.total)")
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
            }
            ProgressView(value: Double(progress.done), total: Double(max(progress.total, 1)))
                .progressViewStyle(.linear)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
        .background(.regularMaterial)
    }
}

// MARK: - Processing Status Bar

private struct ProcessingStatusBar: View {
    let status: String
    let done: Int
    let total: Int
    let onCancel: () -> Void

    var body: some View {
        VStack(spacing: 4) {
            HStack {
                Text(status)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Spacer()
                if total > 0 {
                    Text("\(done) / \(total)")
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.secondary)
                }
                Button("Cancel", action: onCancel)
                    .buttonStyle(.borderless)
                    .font(.caption)
                    .padding(.leading, 8)
            }
            if total > 0 {
                ProgressView(value: Double(done), total: Double(total))
                    .progressViewStyle(.linear)
            } else {
                ProgressView(value: 0.0)
                    .progressViewStyle(.linear)
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .background(.regularMaterial)
    }
}
