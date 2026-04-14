# Thumbnail Loading Redesign

**Date:** 2026-04-14  
**Status:** Approved

## Problem

Thumbnails stop loading after the first 1-2 rows. Root cause: the `shouldLoad` mechanism computed per-cell inside `GridView.body` has multiple timing failure modes:

1. `viewportHeight` is set in a deferred `Task { @MainActor in }` — on first render all cells have `shouldLoad=false` because the guard `viewportWidth > 0, viewportHeight > 0` fails
2. `stableScrollOffset` only updates after a 0.25s debounce timer — creates a window where visible cells have `shouldLoad=false` and never start loading
3. `onPreferenceChange` and `onAppear` fire during the view update phase — deferring state mutations to avoid the "Modifying state during view update" warning introduces ordering races
4. Column count formula in `isNearViewport` used `viewportWidth - 16` instead of `viewportWidth - 32` (two sides of `.padding()`), causing off-by-one row errors
5. Disk-cached reads were bottlenecked by `decodeQueue` (max 3 concurrent) — a 50KB JPEG read competed with a 300MB RAW decode for the same slot

## Solution

Remove the `shouldLoad` mechanism entirely. LazyVGrid already manages cell lifecycle — cells enter the render window when visible (plus a natural ~2–3 row buffer) and are destroyed when far away. Swift's structured concurrency handles the rest: `.task` starts when the cell appears, cancels when the cell is evicted.

## Deleted Code

**`GridView`:**
- `@State private var stableScrollOffset`
- `@State private var prefetchDebounce`
- `@State private var viewportWidth`

Note: `scrollOffset` and `viewportHeight` are **not deleted** — only their load-window usage is removed. Both are retained for edge-scroll (see Retained Code section).
- `isNearViewport(index:)` function
- `schedulePrefetch()` function
- `ScrollOffsetKey` preference key
- `onPreferenceChange(ScrollOffsetKey.self)` handler — the GeometryReader that feeds it (in the `ZStack`'s `.background`) is also deleted
- `shouldLoad:` argument passed to `ThumbnailCell`

**Note on edge-scroll state:** `handleDrag` uses both `scrollOffset` (to convert drag coordinates from grid-space to viewport-space) and `viewportHeight` (to detect proximity to the bottom edge). These two `@State` vars must be **retained** for edge-scroll to function. The viewport-size `GeometryReader` (on the `ScrollView`'s `.background`) must also be **retained** — it sets `viewportHeight` and `viewportWidth`. What is deleted is the scroll-offset `GeometryReader` (inside the `ZStack`'s `.background`), `ScrollOffsetKey`, and all load-window usage of `scrollOffset`.

**`ThumbnailCell`:**
- `shouldLoad: Bool` prop
- The 3-state `imageLayer` (`@ViewBuilder`) — replaced by a 2-state version: `if let thumb = thumbnail { Image... } else { spinner }`. The intermediate "idle blank" state (`Rectangle().fill(Color.secondary.opacity(0.08))`) is removed.

**`ThumbnailCache`:**
- `prefetch(urls:size:limit:)` method
- `prefetchQueue` static `OperationQueue` property (lines ~156–161)

## Retained Code

**`GridView`** retains: `@State private var viewportHeight`, `@State private var viewportWidth`, `@State private var scrollOffset`, the viewport-size `GeometryReader` (`.background` on the `ScrollView`), cell frame tracking (`CellFrameKey`), rubber band drag selection, edge-scroll auto-scroll, tap/shift-tap/cmd-tap logic, anchor scroll (`onChange(of: anchorID)`).

**`ThumbnailCell`** retains a simplified `.task`:
```swift
.task(id: "\(record.objectID.uriRepresentation())-\(cellSize)") {
    let url = URL(filePath: record.filePath ?? "")
    let img = await ThumbnailCache.shared.thumbnail(
        for: url,
        size: CGSize(width: cellSize, height: cellSize * 0.6875)
    )
    guard !Task.isCancelled else { return }
    thumbnail = img
}
```
Note: `shouldLoad` is **removed from the task id**. The old id `"...-\(shouldLoad)"` caused the task to restart whenever `shouldLoad` flipped — that was the trigger mechanism. The new id is stable for the cell's lifetime; the task starts on cell appearance and is cancelled by SwiftUI when LazyVGrid evicts the cell.

**`ThumbnailCache`** retains: memory cache (NSCache, 300 items, 150MB), disk cache (JPEG, SHA256-keyed), `diskReadQueue` (8 concurrent), `decodeQueue` (4 concurrent), `invalidate(for:)`, `thumbnail(for:size:)`, `readFromDisk(_:)`.

## Changes to ThumbnailCache

### Continuation safety net

Both `readFromDisk` and the decode path must guarantee the `CheckedContinuation` is resumed exactly once. An Obj-C exception from LibRaw or AppKit mid-operation bypasses all `return` statements and leaves the continuation dangling, permanently holding a queue slot. Fix with a `resumed` flag + `defer`:

```swift
op.addExecutionBlock {
    var resumed = false
    defer { if !resumed { c.resume(returning: nil) } }

    if op.isCancelled { c.resume(returning: nil); resumed = true; return }
    // ... decode work ...
    c.resume(returning: img)
    resumed = true
}
```

Apply the same pattern to `readFromDisk`'s `BlockOperation`.

### Decode concurrency 3 → 4

The IOSurface OOM was caused by `CGImageSourceCreateImageAtIndex` returning full-resolution images (now fixed — using `CGImageSourceCreateThumbnailAtIndex`). Properly-sized thumbnails (~20KB) have negligible GPU footprint. Raising to 4 concurrent decodes improves throughput (~800MB peak LibRaw working memory, safe on 16GB machines).

## Data Flow

```
LazyVGrid renders cell
    → ThumbnailCell .task starts
        → ThumbnailCache.thumbnail(for:size:)
            → Tier 1: NSCache hit → return immediately
            → Tier 2: diskReadQueue (8 concurrent) → disk JPEG hit → cache + return
            → Tier 3: decodeQueue (4 concurrent) → LibRaw/ImageIO decode → write disk → cache + return
        → guard !Task.isCancelled → set thumbnail

LazyVGrid evicts cell (scrolled away)
    → .task cancelled
        → withTaskCancellationHandler → op.cancel()
        → queued op: never starts
        → running op: completes, result discarded by guard !Task.isCancelled
```

## Files Changed

| File | Change |
|------|--------|
| `ImageRater/UI/GridView.swift` | Remove `stableScrollOffset`, `scrollOffset` load-window usage, `isNearViewport`, `schedulePrefetch`, scroll-offset GeometryReader, `ScrollOffsetKey`; retain `viewportHeight`/`viewportWidth`/`scrollOffset` for edge-scroll |
| `ImageRater/UI/Components/ThumbnailCell.swift` | Remove `shouldLoad` prop; simplify task id (no `-\(shouldLoad)`); simplify `imageLayer` to 2-state |
| `ImageRater/Import/ThumbnailCache.swift` | Remove `prefetch` + `prefetchQueue`; add `resumed` flag safety on both continuations; raise `decodeQueue` concurrency to 4 |
