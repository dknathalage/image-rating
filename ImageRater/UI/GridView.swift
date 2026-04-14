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
    // Viewport dimensions tracked for edge-scroll
    @State private var viewportHeight: CGFloat = 0
    @State private var scrollOffset: CGFloat = 0
    @State private var lastAutoScrollTime: Date = .distantPast

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
                            ForEach(images, id: \.objectID) { record in
                                ThumbnailCell(
                                    record: record,
                                    isSelected: selectedIDs.contains(record.objectID),
                                    cellSize: cellSize
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
                            .onAppear { viewportHeight = geo.size.height }
                            .onChange(of: geo.size) { _, s in viewportHeight = s.height }
                    }
                )
                .onPreferenceChange(ScrollOffsetKey.self) { offset in
                    scrollOffset = offset
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

    // MARK: - Rubber band + edge-scroll

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

        // Edge-scroll: trigger only when cursor exits the viewport boundary.
        // Throttled to 1 row per 0.3s. Reads live state each call so targets are fresh.
        let viewportY = c.y - scrollOffset
        guard viewportY < 0 || viewportY > viewportHeight else { return }
        let now = Date()
        guard now.timeIntervalSince(lastAutoScrollTime) >= 0.3 else { return }

        // Row pitch = cell height + LazyVGrid inter-cell spacing.
        // Target the cell whose top is 1 row above/below the current viewport top,
        // then use .top anchor so the viewport starts exactly there.
        let rowPitch = cellSize * 0.6875 + 8

        if viewportY < 0 {
            // Cursor above viewport — scroll up 1 row.
            let targetTopY = max(0, scrollOffset - rowPitch)
            if let best = cellFrames.min(by: {
                abs($0.value.minY - targetTopY) < abs($1.value.minY - targetTopY)
            }) {
                proxy.scrollTo(best.key, anchor: .top)
                lastAutoScrollTime = now
            }
        } else {
            // Cursor below viewport — scroll down 1 row.
            let targetTopY = scrollOffset + rowPitch
            if let best = cellFrames.min(by: {
                abs($0.value.minY - targetTopY) < abs($1.value.minY - targetTopY)
            }) {
                proxy.scrollTo(best.key, anchor: .top)
                lastAutoScrollTime = now
            }
        }
    }
}

// MARK: - Scroll offset preference

private struct ScrollOffsetKey: PreferenceKey {
    static var defaultValue: CGFloat = 0
    static func reduce(value: inout CGFloat, nextValue: () -> CGFloat) {
        value = nextValue()
    }
}
