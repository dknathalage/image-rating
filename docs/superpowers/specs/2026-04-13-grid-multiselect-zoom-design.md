# Grid Multi-Select & Cell Size Control — Design Spec

**Date:** 2026-04-13
**Status:** Approved

---

## Overview

Two features added to the image grid:

1. **Rubber-band multi-select** — drag on empty grid space draws a selection rectangle; all cells it intersects are selected. ⌘-click toggles individual cells. Typing 0–5 rates all selected images at once.
2. **Stepped cell size slider** — a toolbar slider snapping to 5 preset sizes lets the user show more (smaller) or fewer (larger) images.

---

## Feature 1: Rubber-Band Multi-Select

### State (GridView)

| State | Type | Purpose |
|-------|------|---------|
| `selectedIDs` | `Set<NSManagedObjectID>` | Multi-selection set. `NSManagedObjectID` conforms to `Hashable`. |
| `cellFrames` | `[NSManagedObjectID: CGRect]` | Per-cell frames in `"grid"` coordinate space |
| `dragRect` | `CGRect?` | Current rubber-band rect; `nil` when not dragging |

`selectedRecord: Binding<ImageRecord?>` (existing) — retained for detail panel; set to the last tapped cell. When `selectedIDs` is non-empty, `selectedRecord` holds the most recently tapped cell (the one shown in the detail panel). ⌘-tapping a cell updates `selectedRecord` to that cell without clearing `selectedIDs`.

### CellFramePreferenceKey

New file `ImageRater/UI/Components/CellFramePreferenceKey.swift`:

```swift
struct CellFramePreferenceKey: PreferenceKey {
    typealias Value = [NSManagedObjectID: CGRect]
    static var defaultValue: Value = [:]
    static func reduce(value: inout Value, nextValue: () -> Value) {
        value.merge(nextValue()) { _, new in new }
    }
}
```

Each `ThumbnailCell` wraps its entire outer `ZStack` in a `GeometryReader`. Inside, it reads the frame in the `"grid"` coordinate space and emits it via `.preference`:

```swift
GeometryReader { geo in
    ZStack { /* existing content */ }
        .preference(
            key: CellFramePreferenceKey.self,
            value: [record.objectID: geo.frame(in: .named("grid"))]
        )
}
```

The `LazyVGrid`'s enclosing `ScrollView` content layer is tagged `.coordinateSpace(name: "grid")`. `GridView` collects frames:

```swift
.onPreferenceChange(CellFramePreferenceKey.self) { cellFrames = $0 }
```

### Drag Gesture and Overlay

The layout in `GridView.body` is a `ZStack`:

```
ZStack {
    ScrollView {
        LazyVGrid { ... }
            .coordinateSpace(name: "grid")  // ← inside ScrollView content
    }

    // Overlay — also inside the same ZStack so it shares the same origin
    Color.clear
        .gesture(DragGesture(minimumDistance: 4, coordinateSpace: .named("grid")))
        .allowsHitTesting(true)
}
```

Because both the `LazyVGrid` and the `Color.clear` overlay are children of the same `ZStack` (and the coordinate space is named on the grid), drag locations are expressed in the same coordinate system as stored cell frames.

- **onChanged** — compute `dragRect` as the normalized rect from `startLocation` to `startLocation + translation`. Recompute `selectedIDs` as all records whose stored frame **partially overlaps** `dragRect` (`storedFrame.intersects(dragRect)`). Partial overlap counts — touching one pixel selects the cell.
- **onEnded** — clear `dragRect` (hide rubber-band), keep `selectedIDs`.

Cells sit above the overlay via `zIndex(1)` so their tap gestures are not blocked.

### Tap Behaviour and ⌘ Detection

⌘ key state is read via `NSEvent.modifierFlags.contains(.command)` at tap time. `ThumbnailCell` exposes two callbacks — `onTap` and `onCommandTap` — set by `GridView`:

| Gesture | Result |
|---------|--------|
| Plain tap on cell | Clear `selectedIDs`; set `selectedRecord` to this cell |
| ⌘-tap on cell | Toggle `objectID` in `selectedIDs`; set `selectedRecord` to this cell |
| Drag on empty space | Rubber-band select; `selectedRecord` unchanged |

`ThumbnailCell` implements this with a single `.onTapGesture`:

```swift
.onTapGesture {
    if NSEvent.modifierFlags.contains(.command) {
        onCommandTap()
    } else {
        onTap()
    }
}
```

### Visual Feedback

- `ThumbnailCell` selection border priority: **blue** (`Color.accentColor`) when `objectID ∈ selectedIDs`, **red** (`Color.red.opacity(0.6)`) when `cullRejected`, **clear** otherwise. Existing stroke width (2) unchanged.
- Rubber-band rect: `Rectangle().stroke(Color.accentColor, lineWidth: 1).background(Color.accentColor.opacity(0.1))` positioned using `.frame` + `.position` or `.offset` derived from `dragRect`. Drawn as an overlay on the `ZStack`, clipped to the visible area.

### Batch Rating

`ContentView.setRating(_ stars: Int)`. `GridView` exposes `selectedIDs` to `ContentView` via a `Binding<Set<NSManagedObjectID>>`.

```
if selectedIDs is non-empty:
    for id in selectedIDs:
        let record = ctx.object(with: id) as! ImageRecord
        record.userOverride = stars == 0 ? nil : NSNumber(value: Int16(stars))
else if selectedRecord != nil:
    selectedRecord.userOverride = stars == 0 ? nil : NSNumber(value: Int16(stars))
try? ctx.save()
```

Records referenced by `selectedIDs` are guaranteed to be in memory — they are `FetchedResults` objects held by the active `FetchRequest` in `GridView`. Fetching with `ctx.object(with:)` is a no-op memory lookup (no I/O) for already-loaded objects.

Keyboard handler in `ContentView` unchanged — still fires `setRating`.

---

## Feature 2: Stepped Cell Size Slider

### Presets

```swift
let cellSizePresets: [CGFloat] = [120, 160, 220, 300, 400]
```

Default index: `1` (160 px — matches current hardcoded size, no visual change on first launch).

### State

`@State var cellSizeIndex: Int = 1` lives in `ContentView`, passed to `GridView` as `Binding<Int>`.

### Slider

Placed in the `GridView` toolbar alongside Process / Export XMP buttons:

```swift
HStack(spacing: 4) {
    Image(systemName: "square.grid.3x3")
        .font(.caption)
    Slider(value: Binding(
        get: { Double(cellSizeIndex) },
        set: { cellSizeIndex = Int($0.rounded()) }
    ), in: 0...4, step: 1)
    .frame(width: 100)
    Image(systemName: "square.grid.2x2")
        .font(.caption)
}
```

Icons are static `Image` views (not buttons). The slider is 100 pt wide fixed.

### Grid Layout

`GridView` computes `cellSize = cellSizePresets[cellSizeIndex]` and uses:

```swift
let columns = [GridItem(.adaptive(minimum: cellSize), spacing: 8)]
```

### ThumbnailCell

Takes `cellSize: CGFloat` as a parameter. Frame uses 3:2 aspect ratio (appropriate for landscape camera images; portrait images are cropped/filled via `.aspectRatio(contentMode: .fill)` + `.clipped()`, consistent with current behaviour):

```swift
.frame(width: cellSize, height: cellSize * 2 / 3)
```

Thumbnail fetch task ID is keyed by both `objectID` and `cellSize` so it re-fetches at the new resolution when the slider changes:

```swift
.task(id: "\(record.objectID)-\(cellSize)") {
    thumbnail = await ThumbnailCache.shared.thumbnail(
        for: url, size: CGSize(width: cellSize, height: cellSize * 2 / 3))
}
```

---

## Files Changed

| File | Change |
|------|--------|
| `ImageRater/UI/GridView.swift` | Add `selectedIDs`, `cellFrames`, `dragRect`; ZStack overlay with drag gesture; rubber-band rect overlay; `onPreferenceChange`; stepped slider in toolbar; accept `cellSizeIndex: Binding<Int>` and `selectedIDs: Binding<Set<NSManagedObjectID>>` |
| `ImageRater/UI/Components/ThumbnailCell.swift` | Accept `cellSize: CGFloat`, `isSelected: Bool` (from `selectedIDs`), `onCommandTap: () -> Void`; parameterise frame and thumbnail fetch; `GeometryReader` wrapper for frame reporting; ⌘-tap detection |
| `ImageRater/App/ContentView.swift` | Add `cellSizeIndex: Int` and `selectedIDs: Set<NSManagedObjectID>` state; thread both to `GridView`; update `setRating` for batch |
| `ImageRater/UI/Components/CellFramePreferenceKey.swift` | **New file** — `PreferenceKey` aggregating `[NSManagedObjectID: CGRect]` (definition above) |

---

## Non-Goals

- Shift-click range select (not requested)
- Persisting cell size across launches (not requested)
- Scroll-while-dragging rubber-band (out of scope for initial cut)
- Portrait-aware aspect ratio per cell (all cells uniform 3:2, fill-clipped)
