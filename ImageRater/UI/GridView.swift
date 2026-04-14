import CoreData
import SwiftUI

// MARK: - Cell frame preference

private struct CellFrame: Equatable {
    let id: NSManagedObjectID
    let frame: CGRect
}

private struct CellFrameKey: PreferenceKey {
    static var defaultValue: [CellFrame] = []
    static func reduce(value: inout [CellFrame], nextValue: () -> [CellFrame]) {
        value.append(contentsOf: nextValue())
    }
}

// MARK: - GridView

struct GridView: View {
    let images: [ImageRecord]
    let sessionHasImages: Bool
    @Binding var selectedIDs: Set<NSManagedObjectID>
    @Binding var anchorID: NSManagedObjectID?
    let cellSize: CGFloat
    let onDoubleClick: (ImageRecord) -> Void
    let onRemoveRatings: () -> Void
    let onRunAI: () -> Void

    var columns: [GridItem] { [GridItem(.adaptive(minimum: cellSize), spacing: 8)] }

    @State private var cellFrames: [NSManagedObjectID: CGRect] = [:]
    @State private var dragRect: CGRect?
    @State private var scrollProxy: ScrollViewProxy?
    // Viewport dimensions tracked for edge-scroll + load window
    @State private var viewportHeight: CGFloat = 0
    @State private var viewportWidth: CGFloat = 0
    @State private var scrollOffset: CGFloat = 0
    /// Debounced scroll offset — only updates after scroll settles. Used for load/evict decisions.
    @State private var stableScrollOffset: CGFloat = 0
    @State private var autoScrollTimer: Timer?
    @State private var prefetchDebounce: Timer?

    var body: some View {
        if images.isEmpty && !sessionHasImages {
            ContentUnavailableView("No Images", systemImage: "photo.on.rectangle")
        } else if images.isEmpty && sessionHasImages {
            ContentUnavailableView(
                "No Matches",
                systemImage: "line.3.horizontal.decrease.circle",
                description: Text("No images match the selected rating filter.")
            )
        } else {
            ScrollViewReader { proxy in
                ScrollView {
                    ZStack(alignment: .topLeading) {
                        LazyVGrid(columns: columns, spacing: 8) {
                            ForEach(Array(images.enumerated()), id: \.element.objectID) { idx, record in
                                ThumbnailCell(
                                    record: record,
                                    isSelected: selectedIDs.contains(record.objectID),
                                    cellSize: cellSize,
                                    shouldLoad: isNearViewport(index: idx)
                                ) { mods in
                                    handleTap(record: record, modifiers: mods)
                                } onDoubleClick: {
                                    onDoubleClick(record)
                                }
                                .id(record.objectID)
                                .background(
                                    GeometryReader { geo in
                                        Color.clear.preference(
                                            key: CellFrameKey.self,
                                            value: [CellFrame(
                                                id: record.objectID,
                                                frame: geo.frame(in: .named("grid"))
                                            )]
                                        )
                                    }
                                )
                            }
                        }
                        .padding()

                        // Rubber band rect
                        Canvas { ctx, _ in
                            guard let rect = dragRect, rect.width > 1 || rect.height > 1 else { return }
                            let path = Path(rect)
                            ctx.fill(path, with: .color(.accentColor.opacity(0.12)))
                            ctx.stroke(path, with: .color(.accentColor.opacity(0.5)), lineWidth: 1)
                        }
                        .allowsHitTesting(false)
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                    }
                    .coordinateSpace(name: "grid")
                    .gesture(
                        DragGesture(minimumDistance: 4, coordinateSpace: .named("grid"))
                            .onChanged { handleDrag($0, proxy: proxy) }
                            .onEnded { _ in
                                dragRect = nil
                                stopAutoScroll()
                            }
                    )
                    // Track scroll offset for edge-scroll calculation
                    .background(
                        GeometryReader { geo in
                            Color.clear.preference(
                                key: ScrollOffsetKey.self,
                                value: -geo.frame(in: .named("scroll")).minY
                            )
                        }
                    )
                }
                .coordinateSpace(name: "scroll")
                .background(
                    GeometryReader { geo in
                        Color.clear
                            .onAppear {
                                // GeometryReader onAppear fires during layout — defer all
                                // state mutations to avoid "Modifying state during view update".
                                Task { @MainActor in
                                    viewportHeight = geo.size.height
                                    viewportWidth = geo.size.width
                                    stableScrollOffset = scrollOffset
                                }
                            }
                            .onChange(of: geo.size) { _, s in viewportHeight = s.height; viewportWidth = s.width }
                    }
                )
                .onPreferenceChange(ScrollOffsetKey.self) { offset in
                    // onPreferenceChange fires during the view update phase.
                    // Defer state mutations to avoid "Modifying state during view update".
                    Task { @MainActor in
                        scrollOffset = offset
                        schedulePrefetch()
                    }
                }
                .contextMenu {
                    if !selectedIDs.isEmpty {
                        Button("Remove Ratings", systemImage: "star.slash") {
                            onRemoveRatings()
                        }
                        Divider()
                        Button("Run AI Rating", systemImage: "wand.and.stars") {
                            onRunAI()
                        }
                    }
                }
                .onPreferenceChange(CellFrameKey.self) { frames in
                    cellFrames = Dictionary(
                        frames.map { ($0.id, $0.frame) },
                        uniquingKeysWith: { _, new in new }
                    )
                }
                .onChange(of: anchorID) { _, id in
                    guard let id else { return }
                    withAnimation(.easeInOut(duration: 0.15)) {
                        proxy.scrollTo(id, anchor: .center)
                    }
                }
                .onAppear { scrollProxy = proxy }
            }

            Button("") { selectedIDs = Set(images.map(\.objectID)) }
                .keyboardShortcut("a", modifiers: .command)
                .frame(width: 0, height: 0)
                .opacity(0)
        }
    }

    // MARK: - Scroll settle debounce

    private func schedulePrefetch() {
        prefetchDebounce?.invalidate()
        prefetchDebounce = Timer.scheduledTimer(withTimeInterval: 0.25, repeats: false) { _ in
            Task { @MainActor in
                // Scroll has settled — commit stable offset so load window re-evaluates
                stableScrollOffset = scrollOffset
            }
        }
    }

    // MARK: - Load window

    /// Returns true if cell at `index` is within 2 rows of the stable (debounced) viewport.
    /// Uses stableScrollOffset so load decisions freeze during scrolling — no task churn.
    private func isNearViewport(index: Int) -> Bool {
        guard viewportWidth > 0, viewportHeight > 0, cellSize > 0 else { return false }
        let colCount = max(1, Int((viewportWidth - 16) / (cellSize + 8)))
        let cellHeight = cellSize * 0.6875 + 8
        let rowIndex = index / colCount
        let cellTop = CGFloat(rowIndex) * cellHeight + 16   // 16 = grid padding
        let cellBottom = cellTop + cellHeight
        let bufferRows: CGFloat = 2
        let loadTop = stableScrollOffset - bufferRows * cellHeight
        let loadBottom = stableScrollOffset + viewportHeight + bufferRows * cellHeight
        return cellBottom > loadTop && cellTop < loadBottom
    }

    // MARK: - Tap logic

    private func handleTap(record: ImageRecord, modifiers: NSEvent.ModifierFlags) {
        let id = record.objectID
        if modifiers.contains(.shift), let anchor = anchorID {
            guard
                let ai = images.firstIndex(where: { $0.objectID == anchor }),
                let ti = images.firstIndex(where: { $0.objectID == id })
            else {
                selectedIDs = [id]; anchorID = id; return
            }
            let lo = min(ai, ti), hi = max(ai, ti)
            selectedIDs = Set(images[lo...hi].map(\.objectID))
        } else if modifiers.contains(.command) {
            if selectedIDs.contains(id) {
                selectedIDs.remove(id)
                if anchorID == id { anchorID = selectedIDs.first }
            } else {
                selectedIDs.insert(id)
                anchorID = id
            }
        } else {
            selectedIDs = [id]
            anchorID = id
        }
    }

    // MARK: - Rubber band + auto-scroll

    private func handleDrag(_ value: DragGesture.Value, proxy: ScrollViewProxy) {
        let s = value.startLocation, c = value.location
        let rect = CGRect(
            x: min(s.x, c.x), y: min(s.y, c.y),
            width: abs(c.x - s.x), height: abs(c.y - s.y)
        )
        dragRect = rect

        let mods = NSApp.currentEvent?.modifierFlags ?? []
        let hit = Set(cellFrames.compactMap { id, frame -> NSManagedObjectID? in
            frame.intersects(rect) ? id : nil
        })
        selectedIDs = mods.contains(.shift) ? selectedIDs.union(hit) : hit

        // Edge-scroll: drag within 40pt of top/bottom triggers auto-scroll
        let edgeZone: CGFloat = 40
        // c.y is in "grid" coordinate space (content coords); convert to viewport
        let viewportY = c.y - scrollOffset
        if viewportY < edgeZone {
            let speed = Double(max(1, edgeZone - viewportY))
            startAutoScroll(direction: -speed, proxy: proxy)
        } else if viewportY > viewportHeight - edgeZone {
            let speed = Double(max(1, viewportY - (viewportHeight - edgeZone)))
            startAutoScroll(direction: speed, proxy: proxy)
        } else {
            stopAutoScroll()
        }
    }

    private func startAutoScroll(direction: Double, proxy: ScrollViewProxy) {
        stopAutoScroll()
        autoScrollTimer = Timer.scheduledTimer(withTimeInterval: 0.016, repeats: true) { _ in
            Task { @MainActor in
                // Scroll by finding the first/last visible image near the edge
                let step = direction * 0.5
                if step < 0, let first = images.first {
                    proxy.scrollTo(first.objectID, anchor: .top)
                } else if step > 0, let last = images.last {
                    proxy.scrollTo(last.objectID, anchor: .bottom)
                }
            }
        }
    }

    private func stopAutoScroll() {
        autoScrollTimer?.invalidate()
        autoScrollTimer = nil
    }
}

// MARK: - Scroll offset preference

private struct ScrollOffsetKey: PreferenceKey {
    static var defaultValue: CGFloat = 0
    static func reduce(value: inout CGFloat, nextValue: () -> CGFloat) {
        value = nextValue()
    }
}
