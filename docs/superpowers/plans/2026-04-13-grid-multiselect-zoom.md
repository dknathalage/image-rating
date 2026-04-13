# Grid Multi-Select & Cell Size Control Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add rubber-band multi-select with batch rating and a stepped cell-size slider to the image grid.

**Architecture:** `CellFramePreferenceKey` lets each `ThumbnailCell` report its on-screen frame; `GridView` collects these frames, draws a rubber-band rect from a `DragGesture`, and intersects it against stored frames to build `selectedIDs`. Cell size flows as an index into a 5-step preset array from `ContentView` down through `GridView` to `ThumbnailCell`.

**Tech Stack:** SwiftUI (macOS), CoreData, XCTest

---

## File Map

| Action | Path | Responsibility |
|--------|------|---------------|
| **Create** | `ImageRater/UI/Components/CellFramePreferenceKey.swift` | PreferenceKey aggregating `[NSManagedObjectID: CGRect]` |
| **Modify** | `ImageRater/UI/Components/ThumbnailCell.swift` | Accept `cellSize`, `isInSelection`, `onCommandTap`; GeometryReader frame reporting; ⌘-tap detection |
| **Modify** | `ImageRater/UI/GridView.swift` | Multi-select state; drag gesture; rubber-band overlay; stepped slider |
| **Modify** | `ImageRater/App/ContentView.swift` | `cellSizeIndex` + `selectedIDs` state; thread to GridView; batch `setRating` |
| **Test** | `ImageRaterTests/GridSelectionTests.swift` | Unit tests for PreferenceKey reduce and batch rating logic |

---

## Task 1: CellFramePreferenceKey

**Files:**
- Create: `ImageRater/UI/Components/CellFramePreferenceKey.swift`
- Create: `ImageRaterTests/GridSelectionTests.swift`

- [ ] **Step 1: Write the failing test**

Create `ImageRaterTests/GridSelectionTests.swift`:

```swift
import XCTest
import CoreData
@testable import ImageRater

final class GridSelectionTests: XCTestCase {

    func testPreferenceKeyReduceMergesEntries() {
        let ctx = PersistenceController(inMemory: true).container.viewContext
        let r1 = ImageRecord(context: ctx); r1.id = UUID()
        let r2 = ImageRecord(context: ctx); r2.id = UUID()

        var acc = CellFramePreferenceKey.defaultValue
        CellFramePreferenceKey.reduce(value: &acc) {
            [r1.objectID: CGRect(x: 0, y: 0, width: 160, height: 107)]
        }
        CellFramePreferenceKey.reduce(value: &acc) {
            [r2.objectID: CGRect(x: 168, y: 0, width: 160, height: 107)]
        }

        XCTAssertEqual(acc.count, 2)
        XCTAssertEqual(acc[r1.objectID], CGRect(x: 0, y: 0, width: 160, height: 107))
        XCTAssertEqual(acc[r2.objectID], CGRect(x: 168, y: 0, width: 160, height: 107))
    }

    func testPreferenceKeyReduceNewerValueWins() {
        let ctx = PersistenceController(inMemory: true).container.viewContext
        let r1 = ImageRecord(context: ctx); r1.id = UUID()

        var acc = CellFramePreferenceKey.defaultValue
        CellFramePreferenceKey.reduce(value: &acc) {
            [r1.objectID: CGRect(x: 0, y: 0, width: 160, height: 107)]
        }
        CellFramePreferenceKey.reduce(value: &acc) {
            [r1.objectID: CGRect(x: 0, y: 200, width: 160, height: 107)]
        }

        XCTAssertEqual(acc[r1.objectID]?.origin.y, 200)
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

```
xcodebuild test -scheme ImageRater -destination 'platform=macOS' -only-testing:ImageRaterTests/GridSelectionTests 2>&1 | tail -20
```

Expected: compile error — `CellFramePreferenceKey` not defined.

- [ ] **Step 3: Create CellFramePreferenceKey**

Create `ImageRater/UI/Components/CellFramePreferenceKey.swift`:

```swift
import CoreData
import SwiftUI

struct CellFramePreferenceKey: PreferenceKey {
    typealias Value = [NSManagedObjectID: CGRect]
    static var defaultValue: Value = [:]
    static func reduce(value: inout Value, nextValue: () -> Value) {
        value.merge(nextValue()) { _, new in new }
    }
}
```

- [ ] **Step 4: Add file to Xcode project**

Open `ImageRater.xcodeproj` and add `CellFramePreferenceKey.swift` to the `Components` group under `ImageRater/UI/Components/`. Ensure it is in the `ImageRater` target (not test target).

- [ ] **Step 5: Run tests to verify they pass**

```
xcodebuild test -scheme ImageRater -destination 'platform=macOS' -only-testing:ImageRaterTests/GridSelectionTests 2>&1 | tail -20
```

Expected: 2 tests pass.

- [ ] **Step 6: Commit**

```bash
git add ImageRater/UI/Components/CellFramePreferenceKey.swift \
        ImageRaterTests/GridSelectionTests.swift \
        ImageRater.xcodeproj/project.pbxproj
git commit -m "feat: add CellFramePreferenceKey for grid cell frame tracking"
```

---

## Task 2: ThumbnailCell — parameterise and extend

**Files:**
- Modify: `ImageRater/UI/Components/ThumbnailCell.swift`

Current signature: `ThumbnailCell(record:, isSelected:, onSelect:)`
New signature: `ThumbnailCell(record:, cellSize:, isInSelection:, onSelect:, onCommandTap:)`

`isSelected` is renamed `isInSelection` to clarify it means "part of multi-select set."

- [ ] **Step 1: Replace ThumbnailCell.swift**

```swift
import AppKit
import SwiftUI

struct ThumbnailCell: View {
    @ObservedObject var record: ImageRecord
    let cellSize: CGFloat
    let isInSelection: Bool
    let onSelect: () -> Void
    let onCommandTap: () -> Void

    @State private var thumbnail: NSImage?

    private var cellHeight: CGFloat { cellSize * 2 / 3 }

    var body: some View {
        GeometryReader { geo in
            ZStack(alignment: .bottomTrailing) {
                Group {
                    if let thumb = thumbnail {
                        Image(nsImage: thumb)
                            .resizable()
                            .aspectRatio(contentMode: .fill)
                    } else {
                        Rectangle().fill(Color.secondary.opacity(0.2))
                            .overlay(ProgressView())
                    }
                }
                .frame(width: cellSize, height: cellHeight)
                .clipped()
                .cornerRadius(6)
                .overlay(
                    RoundedRectangle(cornerRadius: 6)
                        .stroke(borderColor, lineWidth: 2)
                )

                ScoreBadge(
                    stars: Int(record.userOverride?.int16Value ?? record.ratingStars?.int16Value ?? 0),
                    rejected: record.cullRejected,
                    isManual: (record.userOverride?.int16Value ?? 0) > 0
                )
                .padding(4)
            }
            .frame(width: cellSize, height: cellHeight)
            .preference(
                key: CellFramePreferenceKey.self,
                value: [record.objectID: geo.frame(in: .named("grid"))]
            )
        }
        .frame(width: cellSize, height: cellHeight)
        .onTapGesture {
            if NSEvent.modifierFlags.contains(.command) {
                onCommandTap()
            } else {
                onSelect()
            }
        }
        .task(id: "\(record.objectID)-\(cellSize)") {
            let url = URL(filePath: record.filePath ?? "")
            thumbnail = await ThumbnailCache.shared.thumbnail(
                for: url, size: CGSize(width: cellSize, height: cellHeight))
        }
    }

    private var borderColor: Color {
        if isInSelection { return Color.accentColor }
        if record.cullRejected { return Color.red.opacity(0.6) }
        return .clear
    }
}
```

Note: `GeometryReader` by default sizes to fill available space, so `.frame(width: cellSize, height: cellHeight)` is applied both inside (on the ZStack) and outside (on the GeometryReader) to pin the size.

- [ ] **Step 2: Build to verify compilation**

```
xcodebuild build -scheme ImageRater -destination 'platform=macOS' 2>&1 | grep -E "error:|Build succeeded"
```

Expected: `Build succeeded` (GridView call sites will be broken — that's expected; we fix them in the next task).

- [ ] **Step 3: Commit**

```bash
git add ImageRater/UI/Components/ThumbnailCell.swift
git commit -m "feat: parameterise ThumbnailCell with cellSize and multi-select callbacks"
```

---

## Task 3: GridView — cell size slider

**Files:**
- Modify: `ImageRater/UI/GridView.swift`

Add `cellSizeIndex: Binding<Int>` parameter and toolbar slider. Fix broken `ThumbnailCell` call site.

- [ ] **Step 1: Replace GridView.swift**

```swift
import CoreData
import SwiftUI

let cellSizePresets: [CGFloat] = [120, 160, 220, 300, 400]

struct GridView: View {
    @FetchRequest var images: FetchedResults<ImageRecord>
    @Binding var selectedRecord: ImageRecord?
    @Binding var cellSizeIndex: Int
    @Binding var selectedIDs: Set<NSManagedObjectID>

    init(session: Session,
         selectedRecord: Binding<ImageRecord?>,
         cellSizeIndex: Binding<Int>,
         selectedIDs: Binding<Set<NSManagedObjectID>>) {
        _images = FetchRequest(
            sortDescriptors: [SortDescriptor(\.filePath)],
            predicate: NSPredicate(format: "session == %@", session)
        )
        _selectedRecord = selectedRecord
        _cellSizeIndex = cellSizeIndex
        _selectedIDs = selectedIDs
    }

    // Multi-select state
    @State private var cellFrames: [NSManagedObjectID: CGRect] = [:]
    @State private var dragRect: CGRect? = nil

    private var cellSize: CGFloat { cellSizePresets[cellSizeIndex] }
    private var columns: [GridItem] { [GridItem(.adaptive(minimum: cellSize), spacing: 8)] }

    var body: some View {
        if images.isEmpty {
            ContentUnavailableView("No Images", systemImage: "photo.on.rectangle")
        } else {
            ZStack(alignment: .topLeading) {
                ScrollView {
                    LazyVGrid(columns: columns, spacing: 8) {
                        ForEach(images) { record in
                            ThumbnailCell(
                                record: record,
                                cellSize: cellSize,
                                isInSelection: selectedIDs.contains(record.objectID),
                                onSelect: {
                                    selectedIDs = []
                                    selectedRecord = record
                                },
                                onCommandTap: {
                                    if selectedIDs.contains(record.objectID) {
                                        selectedIDs.remove(record.objectID)
                                    } else {
                                        selectedIDs.insert(record.objectID)
                                    }
                                    selectedRecord = record
                                }
                            )
                            .zIndex(1)
                        }
                    }
                    .padding()
                    .coordinateSpace(name: "grid")
                    .onPreferenceChange(CellFramePreferenceKey.self) { cellFrames = $0 }
                }

                // Rubber-band drag overlay
                Color.clear
                    .contentShape(Rectangle())
                    .gesture(
                        DragGesture(minimumDistance: 4, coordinateSpace: .named("grid"))
                            .onChanged { value in
                                let origin = CGPoint(
                                    x: min(value.startLocation.x, value.location.x),
                                    y: min(value.startLocation.y, value.location.y)
                                )
                                let size = CGSize(
                                    width: abs(value.location.x - value.startLocation.x),
                                    height: abs(value.location.y - value.startLocation.y)
                                )
                                let rect = CGRect(origin: origin, size: size)
                                dragRect = rect
                                selectedIDs = Set(
                                    cellFrames.compactMap { id, frame in
                                        frame.intersects(rect) ? id : nil
                                    }
                                )
                            }
                            .onEnded { _ in dragRect = nil }
                    )

                // Rubber-band rect visual
                if let rect = dragRect {
                    Rectangle()
                        .stroke(Color.accentColor, lineWidth: 1)
                        .background(Color.accentColor.opacity(0.1))
                        .frame(width: rect.width, height: rect.height)
                        .offset(x: rect.minX, y: rect.minY)
                        .allowsHitTesting(false)
                }
            }
            .toolbar {
                ToolbarItem {
                    HStack(spacing: 4) {
                        Image(systemName: "square.grid.3x3")
                            .font(.caption)
                        Slider(
                            value: Binding(
                                get: { Double(cellSizeIndex) },
                                set: { cellSizeIndex = Int($0.rounded()) }
                            ),
                            in: 0...4,
                            step: 1
                        )
                        .frame(width: 100)
                        Image(systemName: "square.grid.2x2")
                            .font(.caption)
                    }
                }
            }
        }
    }
}
```

- [ ] **Step 2: Build to verify compilation**

```
xcodebuild build -scheme ImageRater -destination 'platform=macOS' 2>&1 | grep -E "error:|Build succeeded"
```

Expected: errors only at `GridView` call sites in `ContentView` — that's expected; fixed in the next task.

- [ ] **Step 3: Commit**

```bash
git add ImageRater/UI/GridView.swift
git commit -m "feat: add rubber-band multi-select and cell size slider to GridView"
```

---

## Task 4: ContentView — wire state and batch rating

**Files:**
- Modify: `ImageRater/App/ContentView.swift`
- Modify: `ImageRaterTests/GridSelectionTests.swift`

- [ ] **Step 1: Write failing test for batch setRating logic**

Add to `ImageRaterTests/GridSelectionTests.swift`:

```swift
func testBatchSetRatingAppliesOverrideToAllSelectedIDs() throws {
    let ctx = PersistenceController(inMemory: true).container.viewContext

    let session = Session(context: ctx)
    session.id = UUID()
    session.createdAt = Date()
    session.folderPath = "/tmp"

    var records: [ImageRecord] = []
    for _ in 0..<3 {
        let r = ImageRecord(context: ctx)
        r.id = UUID()
        r.filePath = "/tmp/test.jpg"
        r.session = session
        records.append(r)
    }
    try ctx.save()

    // Simulate batch rating: apply stars=3 to records[0] and records[1]
    let selectedIDs: Set<NSManagedObjectID> = [records[0].objectID, records[1].objectID]
    let stars = 3
    for id in selectedIDs {
        let record = ctx.object(with: id) as! ImageRecord
        record.userOverride = stars == 0 ? nil : NSNumber(value: Int16(stars))
    }
    try ctx.save()

    XCTAssertEqual(records[0].userOverride?.int16Value, 3)
    XCTAssertEqual(records[1].userOverride?.int16Value, 3)
    XCTAssertNil(records[2].userOverride)
}

func testBatchSetRatingZeroClearsOverride() throws {
    let ctx = PersistenceController(inMemory: true).container.viewContext

    let session = Session(context: ctx)
    session.id = UUID()
    session.createdAt = Date()
    session.folderPath = "/tmp"

    let r = ImageRecord(context: ctx)
    r.id = UUID()
    r.filePath = "/tmp/test.jpg"
    r.userOverride = NSNumber(value: Int16(4))
    r.session = session
    try ctx.save()

    // Rating 0 = clear override
    let selectedIDs: Set<NSManagedObjectID> = [r.objectID]
    for id in selectedIDs {
        let record = ctx.object(with: id) as! ImageRecord
        record.userOverride = nil
    }
    try ctx.save()

    XCTAssertNil(r.userOverride)
}
```

- [ ] **Step 2: Run new tests to verify they pass**

(These tests don't depend on ContentView — they test the CoreData logic directly.)

```
xcodebuild test -scheme ImageRater -destination 'platform=macOS' -only-testing:ImageRaterTests/GridSelectionTests 2>&1 | tail -20
```

Expected: 4 tests pass.

- [ ] **Step 3: Update ContentView**

Replace the relevant state declarations and `GridView` call site in `ContentView.swift`:

Add two new `@State` properties after `@State private var selectedRecord`:

```swift
@State private var cellSizeIndex: Int = 1
@State private var selectedIDs: Set<NSManagedObjectID> = []
```

Replace the `GridView(...)` call in the `content:` closure:

```swift
GridView(
    session: session,
    selectedRecord: $selectedRecord,
    cellSizeIndex: $cellSizeIndex,
    selectedIDs: $selectedIDs
)
```

Replace `setRating`:

```swift
private func setRating(_ stars: Int) {
    if !selectedIDs.isEmpty {
        for id in selectedIDs {
            let record = ctx.object(with: id) as! ImageRecord
            record.userOverride = stars == 0 ? nil : NSNumber(value: Int16(stars))
        }
    } else if let record = selectedRecord {
        record.userOverride = stars == 0 ? nil : NSNumber(value: Int16(stars))
    }
    try? ctx.save()
}
```

- [ ] **Step 4: Build and verify all tests pass**

```
xcodebuild test -scheme ImageRater -destination 'platform=macOS' 2>&1 | grep -E "error:|Test Suite.*passed|Test Suite.*failed"
```

Expected: `Build succeeded`, all test suites pass.

- [ ] **Step 5: Commit**

```bash
git add ImageRater/App/ContentView.swift \
        ImageRaterTests/GridSelectionTests.swift \
        ImageRater.xcodeproj/project.pbxproj
git commit -m "feat: wire multi-select state and batch rating in ContentView"
```

---

## Task 5: Manual Smoke Test

These are visual/interaction features — verify by running the app.

- [ ] **Step 1: Build and run**

Open in Xcode (`open ImageRater.xcodeproj`) and run on macOS target, or:

```
xcodebuild build -scheme ImageRater -destination 'platform=macOS' 2>&1 | grep "Build succeeded"
```

- [ ] **Step 2: Verify cell size slider**

- Open a folder with images
- Move the slider left → cells shrink, more columns appear
- Move the slider right → cells grow, fewer columns
- Thumbnails re-render at new size (may take a moment for large folders)
- Default (index 1) matches old 160 px size

- [ ] **Step 3: Verify rubber-band select**

- Drag on empty space between cells → blue rubber-band rect appears
- Cells touched by rect get blue selection border
- Release → rect disappears, selection stays
- Type `3` → all selected images get 3-star rating (check ScoreBadge updates)
- Type `0` → clears ratings on all selected

- [ ] **Step 4: Verify ⌘-click**

- ⌘-click a cell → adds to selection (blue border)
- ⌘-click same cell again → deselects it
- Plain click → clears selection, selects only that cell

- [ ] **Step 5: Verify detail panel still works**

- Single click → detail panel shows that image
- ⌘-click multiple → detail panel shows last tapped image
- Arrow keys still navigate

- [ ] **Step 6: Final commit if any fixups were made**

```bash
git add -p
git commit -m "fix: smoke test fixups for multi-select and cell size"
```
