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
- `@State private var scrollOffset`
- `@State private var prefetchDebounce`
- `@State private var viewportHeight`
- `@State private var viewportWidth`
- `isNearViewport(index:)` function
- `schedulePrefetch()` function
- GeometryReader background for viewport size tracking
- GeometryReader background for scroll offset tracking
- `ScrollOffsetKey` preference key
- `onPreferenceChange(ScrollOffsetKey.self)` handler
- `shouldLoad:` argument passed to `ThumbnailCell`

**`ThumbnailCell`:**
- `shouldLoad: Bool` prop
- 3-state `imageLayer` (blank / spinner / image) → back to spinner-until-loaded

**`ThumbnailCache`:**
- `prefetch(urls:size:limit:)` method
- `prefetchQueue` operation queue

## Remaining Code

**`GridView`** retains: cell frame tracking (`CellFrameKey`), rubber band drag selection, edge-scroll auto-scroll, tap/shift-tap/cmd-tap logic, anchor scroll (`onChange(of: anchorID)`).

**`ThumbnailCell`** retains: `.task(id: "\(record.objectID.uriRepresentation())-\(cellSize)")` — loads on appear, auto-cancelled on eviction. `guard !Task.isCancelled else { return }` after the await prevents stale decode results from overwriting nil on eviction.

**`ThumbnailCache`** retains: memory cache (NSCache, 300 items, 150MB), disk cache (JPEG, SHA256-keyed), `diskReadQueue` (8 concurrent — fast disk reads), `decodeQueue` (4 concurrent — throttled RAW/ImageIO decodes), `invalidate(for:)`.

## Changes to ThumbnailCache

**Safety net on continuations:** Both `readFromDisk` and the decode path get a `resumed` flag + `defer` block to ensure `CheckedContinuation` is always resumed exactly once, even if an Obj-C exception fires mid-operation. A stuck continuation permanently holds a queue slot.

**Decode concurrency raised 3 → 4:** The IOSurface OOM was caused by `CGImageSourceCreateImageAtIndex` returning full-resolution images (fixed). Properly-sized thumbnails (~20KB) have negligible GPU footprint, so 4 concurrent decodes is safe on 16GB machines (~600MB peak LibRaw working memory).

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
| `ImageRater/UI/GridView.swift` | Remove scroll tracking, load window, prefetch logic |
| `ImageRater/UI/Components/ThumbnailCell.swift` | Remove `shouldLoad` prop, simplify imageLayer |
| `ImageRater/Import/ThumbnailCache.swift` | Remove prefetch, add continuation safety, raise decode concurrency |
