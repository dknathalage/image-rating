# Rating Filter Sidebar Section

**Date:** 2026-04-14
**Status:** Approved

## Overview

Add rating filter section below session list in sidebar. Multi-select 1★–5★ + Unrated. Each row shows count at that rating. Empty selection = show all.

## Effective Rating

`ratingStars` and `userOverride` are both `NSNumber?`. `userOverride` 0 is never persisted.

```swift
func effectiveRating(_ record: ImageRecord) -> Int {
    if let o = record.userOverride, o.int16Value > 0 { return Int(o.int16Value) }
    if let s = record.ratingStars { return Int(s.int16Value) }
    return 0 // Unrated
}
```

## Architecture: Lift @FetchRequest to ContentView

Move `@FetchRequest` from `GridView` to `ContentView`. `GridView` becomes a display view that receives a plain `[ImageRecord]`.

**Declaration** — initial predicate `false` so nothing loads before session selected, sort matches GridView's current filePath ascending order:

```swift
@FetchRequest(
    sortDescriptors: [SortDescriptor(\.filePath, order: .forward)],
    predicate: NSPredicate(value: false)
)
private var sessionImages: FetchedResults<ImageRecord>
```

**Predicate update on session change:**

```swift
.onChange(of: selectedSession) { _, session in
    ratingFilter = []
    selectedRecord = nil  // clear directly; don't rely on ratingFilter onChange chain
    sessionImages.nsPredicate = session.map {
        NSPredicate(format: "session == %@", $0)
    } ?? NSPredicate(value: false)
}
```

## State (ContentView — add to existing state block)

```swift
@State private var ratingFilter: Set<Int> = []
// 0 = Unrated, 1–5 = stars, empty = show all
```

## filteredImages (ContentView — replaces sortedImages)

```swift
var filteredImages: [ImageRecord] {
    let all = Array(sessionImages)
    guard !ratingFilter.isEmpty else { return all }
    return all.filter { ratingFilter.contains(effectiveRating($0)) }
}
```

Remove `sortedImages`. Replace all usages with `filteredImages`, including `navigatePrev`, `navigateNext`, and `setRating`.

## selectedRecord Invalidation

`.onChange(of: ratingFilter)`:

```swift
.onChange(of: ratingFilter) { _, _ in
    guard let record = selectedRecord else { return }
    if !filteredImages.contains(where: { $0.objectID == record.objectID }) {
        selectedRecord = nil
    }
}
```

Guard `selectedRecord != nil` first — session change already cleared it, avoiding stale-data reads while `sessionImages` refreshes. If user re-rates the selected record out of the active filter, keep `selectedRecord` (no additional invalidation beyond `ratingFilter` changes).

## RatingFilterView (new: ImageRater/UI/Components/RatingFilterView.swift)

- Props: `images: [ImageRecord]`, `ratingFilter: Binding<Set<Int>>`
- Computes `ratingCounts: [Int: Int]` in-memory from `images` (counts by `effectiveRating`)
- Rows via `Button` with manual toggle action — no native List selection
- Row order: 5★ → 1★, Unrated last
- Each row: star label left + count right-aligned
- Zero-count rows shown with `.foregroundStyle(.secondary)`

`ratingCounts` is recomputed whenever `images` changes. Since `images` flows from `sessionImages` (a reactive `FetchedResults`), counts update automatically as the pipeline assigns ratings or the user overrides.

## Sidebar Integration (ContentView)

Restructure `List(sessions, selection:)` from flat iteration to `ForEach` + `Section` to allow second section. Apply `.selectionDisabled()` to rating rows to prevent interference with `$selectedSession`:

```swift
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
```

## GridView Changes

- Remove `@FetchRequest`
- Remove `session: Session` parameter
- Accept `images: [ImageRecord]` and `sessionHasImages: Bool`
- `ImageRecord` has no `Identifiable` conformance. Use `ForEach(images, id: \.objectID)` in `LazyVGrid` (or add `extension ImageRecord: Identifiable { var id: NSManagedObjectID { objectID } }` — prefer the `ForEach` id parameter to avoid polluting the model)
- Handle empty state internally based on both parameters:
  - `images.isEmpty && !sessionHasImages` → `ContentUnavailableView("No Images", ...)`
  - `images.isEmpty && sessionHasImages` → `ContentUnavailableView("No Matches", systemImage: "line.3.horizontal.decrease.circle", description: Text("No images match the selected rating filter."))`
  - Otherwise → `LazyVGrid`

## ContentView content pane

Keep existing `if let session = selectedSession` branch. Replace the `GridView` call inside it:

```swift
// Before:
GridView(session: session, selectedRecord: $selectedRecord)

// After:
GridView(
    images: filteredImages,
    sessionHasImages: !sessionImages.isEmpty,
    selectedRecord: $selectedRecord
)
```

All existing modifiers (`.safeAreaInset`, `.toolbar`, `.confirmationDialog`) remain chained on `GridView` unchanged. `session` is still in scope for toolbar actions (`runPipeline`, `exportMetadata`, `resetSession`).

## Touch Points

| File | Change |
|------|--------|
| `ImageRater/UI/Components/RatingFilterView.swift` | New — Button rows, in-memory counts, effectiveRating |
| `ImageRater/App/ContentView.swift` | Add `sessionImages` @FetchRequest, `ratingFilter` state, `effectiveRating` func, `filteredImages` (replaces `sortedImages`), restructure sidebar List, selectedRecord invalidation, update GridView call |
| `ImageRater/UI/GridView.swift` | Remove @FetchRequest + `session` param, accept `images:[ImageRecord]` + `sessionHasImages:Bool`, internal empty-state |

## Out of Scope

- Cull status filtering
- Sorting by rating
- Persisting filter state across sessions
