import CoreData
import SwiftUI

private final class KeyboardHandler: ObservableObject {
    var onPrev: (() -> Void)?
    var onNext: (() -> Void)?
    var onRate: ((Int) -> Void)?
    private var monitor: Any?

    func start() {
        guard monitor == nil else { return }
        monitor = NSEvent.addLocalMonitorForEvents(matching: .keyDown) { [weak self] event in
            guard let self else { return event }
            switch event.keyCode {
            case 123: self.onPrev?(); return nil  // ←
            case 124: self.onNext?(); return nil  // →
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
    @State private var ratingFilter: Set<Int> = []
    @StateObject private var keyboard = KeyboardHandler()

    var body: some View {
        NavigationSplitView {
            List(selection: $selectedSession) {
                Section("Sessions") {
                    ForEach(sessions) { session in
                        Label(
                            URL(filePath: session.folderPath ?? "").lastPathComponent,
                            systemImage: "folder"
                        )
                        .tag(session)
                    }
                }
                Section("Filter by Rating") {
                    RatingFilterView(images: Array(sessionImages), ratingFilter: $ratingFilter)
                        .selectionDisabled()
                }
            }
            .navigationTitle("Sessions")
            .toolbar {
                ToolbarItem {
                    Button(action: openFolder) {
                        Label("Open Folder", systemImage: "folder.badge.plus")
                    }
                }
                ToolbarItem {
                    Button { showModelStore = true } label: {
                        Label("Models", systemImage: "square.and.arrow.down")
                    }
                }
            }
        } content: {
            if let session = selectedSession {
                GridView(images: filteredImages, sessionHasImages: !sessionImages.isEmpty, selectedIDs: $selectedIDs, anchorID: $anchorID)
                    .safeAreaInset(edge: .bottom) {
                        if let status = processingStatus {
                            ProcessingStatusBar(
                                status: status,
                                done: processingDone,
                                total: processingTotal,
                                onCancel: { processingTask?.cancel() }
                            )
                        }
                    }
                    .toolbar {
                        ToolbarItem {
                            Button(action: { runPipeline(session: session) }) {
                                Label("Process", systemImage: "wand.and.stars")
                            }
                        }
                        ToolbarItem {
                            Button(action: { showResetConfirm = true }) {
                                Label("Reset", systemImage: "arrow.counterclockwise")
                            }
                            .disabled(processingStatus != nil)
                        }
                        ToolbarItem {
                            Button(action: { exportMetadata(session: session) }) {
                                Label("Export XMP", systemImage: "square.and.arrow.up")
                            }
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
        } detail: {
            if let record = anchorRecord {
                DetailView(record: record, onPrev: navigatePrev, onNext: navigateNext, onRate: setRating)
            } else {
                ContentUnavailableView("Select an Image", systemImage: "photo")
            }
        }
        .sheet(isPresented: $showModelStore) { ModelStoreView() }
        .alert("Processing Failed", isPresented: Binding(
            get: { processingError != nil },
            set: { if !$0 { processingError = nil } }
        )) {
            Button("OK") { processingError = nil }
        } message: {
            Text(processingError ?? "")
        }
        .onAppear {
            keyboard.onPrev = navigatePrev
            keyboard.onNext = navigateNext
            keyboard.onRate = setRating
            keyboard.start()
        }
        .onDisappear { keyboard.stop() }
        .onChange(of: selectedSession) { _, session in
            ratingFilter = []
            selectedIDs = []
            anchorID = nil
            sessionImages.nsPredicate = session.map {
                NSPredicate(format: "session == %@", $0)
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

    // MARK: - Navigation

    private func effectiveRating(_ record: ImageRecord) -> Int {
        if let o = record.userOverride, o.int16Value > 0 { return Int(o.int16Value) }
        if let s = record.ratingStars { return Int(s.int16Value) }
        return 0
    }

    private var filteredImages: [ImageRecord] {
        let all = Array(sessionImages)
        guard !ratingFilter.isEmpty else { return all }
        return all.filter { ratingFilter.contains(effectiveRating($0)) }
    }

    private var anchorRecord: ImageRecord? {
        guard let id = anchorID else { return nil }
        return ctx.object(with: id) as? ImageRecord
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

    private func setRating(_ stars: Int) {
        guard !selectedIDs.isEmpty else { return }
        for id in selectedIDs {
            guard let record = ctx.object(with: id) as? ImageRecord else { continue }
            record.userOverride = stars == 0 ? nil : NSNumber(value: Int16(stars))
        }
        try? ctx.save()
    }

    // MARK: - Actions

    private func openFolder() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = false
        if panel.runModal() == .OK, let url = panel.url {
            do {
                try ImageImporter.importFolder(url, context: ctx)
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
            // Delete sidecar so Photomator sees no stale metadata
            if let path = record.filePath {
                let xmp = URL(filePath: path).deletingPathExtension().appendingPathExtension("xmp")
                try? FileManager.default.removeItem(at: xmp)
            }
        }
        try? ctx.save()
    }

    private func exportMetadata(session: Session) {
        guard let images = session.images?.allObjects as? [ImageRecord] else { return }
        for record in images {
            guard let path = record.filePath else { continue }
            let url = URL(filePath: path)
            if record.cullRejected {
                try? MetadataWriter.writeSidecarRejected(for: url)
            } else {
                let effectiveStars: Int
                if let override = record.userOverride?.int16Value, override > 0 {
                    effectiveStars = Int(override)
                } else if let stars = record.ratingStars?.int16Value, stars > 0 {
                    effectiveStars = Int(stars)
                } else {
                    continue
                }
                try? MetadataWriter.writeSidecar(stars: effectiveStars, for: url)
            }
        }
    }
}

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
