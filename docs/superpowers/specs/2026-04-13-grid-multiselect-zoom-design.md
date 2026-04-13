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
| `selectedIDs` | `Set<NSManagedObjectID>` | Multi-selection set |
| `cellFrames` | `[NSManagedObjectID: CGRect]` | Per-cell frames in grid coordinate space |
| `dragRect` | `CGRect?` | Current rubber-band rect; `nil` when not dragging |

`selectedRecord: Binding<ImageRecord?>` (existing) — retained for detail panel; set to the last tapped cell.

### Cell Frame Reporting

A `CellFramePreferenceKey: PreferenceKey` aggregates `[NSManagedObjectID: CGRect]`.

Each `ThumbnailCell` wraps its content in a `GeometryReader`, reads `.frame(in: .named("grid"))`, and emits the result via `.anchorPreference` / `.transformPreference`. The grid's `ScrollView` content is tagged `.coordinateSpace(name: "grid")`.

`GridView` collects frames via `.onPreferenceChange(CellFramePreferenceKey.self)` and stores them in `cellFrames`.

### Drag Gesture

A transparent `Color.clear` fullscreen overlay sits above the grid content. It carries:

```swift
DragGesture(minimumDistance: 4, coordinateSpace: .named("grid"))
```

- **onChanged** — update `dragRect` from `startLocation` + `translation`. Recompute `selectedIDs` as all records whose stored frame intersects `dragRect`.
- **onEnded** — clear `dragRect` (hide rubber-band), keep `selectedIDs`.

The drag overlay only intercepts events that start on empty space (not on a cell). Cells sit above the overlay via `zIndex`, so `onTapGesture` on cells is unaffected.

### Tap Behaviour

| Gesture | Result |
|---------|--------|
| Plain tap on cell | Clear `selectedIDs`, set `selectedRecord` to this cell |
| ⌘-tap on cell | Toggle `objectID` in `selectedIDs`; `selectedRecord` = this cell |
| Drag on empty space | Rubber-band select |

### Visual Feedback

- `ThumbnailCell` border: blue (`Color.accentColor`) when `objectID ∈ selectedIDs`, red when `cullRejected`, clear otherwise — same stroke width as today.
- Rubber-band rect: `Rectangle().stroke(Color.accentColor, lineWidth: 1).background(Color.accentColor.opacity(0.1))` drawn as an overlay on the grid, positioned at `dragRect`.

### Batch Rating

`ContentView.setRating(_ stars: Int)`:

```
if selectedIDs is non-empty:
    apply userOverride to all records whose objectID ∈ selectedIDs
else if selectedRecord != nil:
    apply userOverride to selectedRecord
save context
```

Keyboard handler in `ContentView` unchanged — still fires `setRating`.

---

## Feature 2: Stepped Cell Size Slider

### Presets

```swift
let cellSizePresets: [CGFloat] = [120, 160, 220, 300, 400]
```

Default index: `1` (160 px — current size, no visual change on first launch).

### State

`@State var cellSizeIndex: Int = 1` lives in `ContentView`, passed to `GridView` as `Binding<Int>`.

### Slider

Placed in the `GridView` toolbar (alongside Process / Export XMP buttons):

```swift
Slider(value: Binding(
    get: { Double(cellSizeIndex) },
    set: { cellSizeIndex = Int($0.rounded()) }
), in: 0...4, step: 1)
.frame(width: 100)
```

Small grid-icon on the left end, large grid-icon on the right end (SF Symbols `square.grid.3x3` / `square.grid.2x2`).

### Grid Layout

`GridView` computes `cellSize = cellSizePresets[cellSizeIndex]` and uses:

```swift
let columns = [GridItem(.adaptive(minimum: cellSize), spacing: 8)]
```

### ThumbnailCell

Takes `cellSize: CGFloat` as a parameter. Frame:

```swift
.frame(width: cellSize, height: cellSize * 2 / 3)
```

Thumbnail fetch task ID changes to `(record.objectID, cellSize)` so it re-fetches at the new size when the slider moves:

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
| `ImageRater/UI/GridView.swift` | Add `selectedIDs`, `cellFrames`, `dragRect`, drag gesture, rubber-band overlay, `onPreferenceChange`, stepped slider in toolbar; accept `cellSizeIndex` binding |
| `ImageRater/UI/Components/ThumbnailCell.swift` | Accept `cellSize`, parameterise frame and thumbnail fetch; add `GeometryReader` + preference emit; update tap/⌘-tap logic |
| `ImageRater/App/ContentView.swift` | Add `cellSizeIndex` state; thread to `GridView`; update `setRating` to handle `selectedIDs` |
| New: `ImageRater/UI/Components/CellFramePreferenceKey.swift` | `PreferenceKey` definition — a small focused file |

---

## Non-Goals

- Shift-click range select (not requested)
- Persisting cell size across launches (not requested)
- Scroll-while-dragging rubber-band (out of scope for initial cut)
