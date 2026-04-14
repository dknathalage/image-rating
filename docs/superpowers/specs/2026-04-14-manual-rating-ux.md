# Manual Rating UX — Design Spec
**Date:** 2026-04-14  
**Status:** Approved

---

## Overview

Redesign the image culling workflow to maximise manual rating speed. Three themes: layout simplification, per-selection actions, and keyboard/mouse shortcuts that remove friction from rating large batches.

---

## 1. Layout — Grid Maximised

### Current state
3-column `NavigationSplitView`: sidebar | grid | detail panel (always visible).

### Change
Drop to 2-column: sidebar | grid. Detail panel becomes a `.sheet` triggered on demand.

**Double-click** any thumbnail → opens detail modal (full image + metadata + rating picker).  
**Space bar** → toggles detail modal for the anchor image.  
**Escape** → dismisses modal.

`DetailView` is reused as-is inside the sheet. Prev/Next navigation inside the modal still works (←→ keys and buttons). Modal binds to `@State private var detailRecord: ImageRecord?`; nil = closed.

---

## 2. Selection

### Cmd+A — Select All
`GridView` handles `keyboardShortcut("a", modifiers: .command)` → sets `selectedIDs` to all IDs in `filteredImages`.

---

## 3. Context Menu Actions

Both actions appear on right-click anywhere in `GridView` when ≥1 image is selected.

### 3a. Remove Ratings
Clears the following fields on each selected `ImageRecord`:
- `userOverride = nil`
- `ratingStars = nil`
- `topiqTechnicalScore = 0`
- `topiqAestheticScore = 0`
- `clipIQAScore = 0`
- `combinedQualityScore = 0`
- `cullRejected = false`
- `cullReason = nil`
- `processState = ProcessState.pending`
- Deletes XMP sidecar (`<filename>.xmp`) from disk

Saves CoreData context. Also clears companion images in the same group.

### 3b. Run AI Rating
Runs the full pipeline (scoring + cull + star assignment) on selected images only.

**`ProcessingQueue` extension:**
```swift
func process(imageIDs: [NSManagedObjectID], onProgress: ...) async throws
```
- Scores only the supplied `imageIDs` (skips non-primaries, already-done images)
- Resolves `sessionID` from the first image's `session` relationship
- Calls existing `normalizeAndWriteStars(sessionID:)` — normalises across whole session so percentile ranking remains accurate
- Progress reported via existing `onProgress` callback

**UI:** Progress shown as a `.sheet` over the grid (replaces the `ProcessingStatusBar` bottom bar for this flow). Sheet contains status text, `ProgressView`, and Cancel button.

---

## 4. Manual Rating Keyboard & Mouse Shortcuts

### 4a. Auto-Advance After Rating
After `setRating(_:)` saves:
1. Compute `nextID` = next image in current `filteredImages` after `anchorID` (before the save takes effect, so the disappearing image is still in the list).
2. If `nextID` exists, set `anchorID = nextID`, `selectedIDs = [nextID]`.
3. If already at end, stay on last image.

Applies when a single image is selected. Batch rating does not auto-advance.

**Fixes filter-disappear bug:** when a rated image leaves the active filter, cursor is already pointing at the next image rather than resetting to nothing.

### 4b. Space Bar — Toggle Modal
`KeyboardHandler` handles key code 49 (space):
- If modal is closed and `anchorID` is set → open modal for anchor image.
- If modal is open → close modal.

### 4c. X Key — Quick Reject
`KeyboardHandler` handles key `"x"` (key code 7):
- Calls `setRating(1)` on selected images (sets `userOverride = 1`).
- Auto-advance fires (single selection) per §4a.

### 4d. Hover Star Overlay
`ThumbnailCell` adds hover detection via `onHover`:
- On hover, show a 5-star row overlaid at the bottom of the thumbnail (above `ScoreBadge`).
- Each star is a `Button` that calls a new `onRate: (Int) -> Void` callback passed from `GridView`.
- Stars dim/highlight on cursor position (filled up to hovered star).
- Overlay hidden when not hovering.
- Does not interfere with tap/double-click selection.

`GridView` receives `onRate: (ImageRecord, Int) -> Void` from `ContentView` and passes it down to each `ThumbnailCell`.

### 4e. Compare Mode
Available when exactly 2 or 3 images are selected.

**Trigger:** "Compare" button appears in toolbar when selection count is 2 or 3.  
**Modal:** `.sheet` showing selected images side-by-side in an `HStack`. Each column contains:
- `ZoomableImageView` (same zoom controls as detail modal)
- Filename
- Current star rating (`ScoreBadge`)
- 1–5 star rating picker (calls `setRating` for that specific image)

Closing compare modal does not change selection.

---

## 5. Bug Fix — Filter Disappear on Rating

**Root cause:** `filteredImages` recomputes after `setRating` saves `userOverride`. If the new rating no longer matches `ratingFilter`, `anchorID` points to an image not in `filteredImages`. `FilterChangeModifier.onChange(of: ratingFilter)` only fires when the filter itself changes, not when image ratings change, so the stale anchor is never corrected.

**Fix:** Pre-compute `nextID` inside `setRating` before `ctx.save()`, using the still-intact `filteredImages`. Set `anchorID` to `nextID` immediately. By the time SwiftUI re-renders with the updated `filteredImages`, the anchor is already valid.

---

## 6. Thumbnail Size Control

User-adjustable grid cell size via a slider in the grid toolbar.

- `@AppStorage("thumbnailSize") var thumbnailSize: CGFloat = 160` in `ContentView`.
- Slider range: 100–320 pt. Step: continuous with debounce (0.3 s) before triggering re-decode.
- Passed to `GridView` as `cellSize: CGFloat`. `GridItem(.adaptive(minimum: cellSize))`.
- Passed to `ThumbnailCell` which sizes to `frame(width: cellSize, height: cellSize * 0.6875)` (preserves current 160:110 aspect ratio).
- Cache key already includes size, so new size generates fresh cache entries.
- Debounce prevents flooding the decode queue during slider drag.

---

## 7. Fast Thumbnail Loading

### Current bottleneck
`ThumbnailCell.task` triggers decode on scroll-into-view. First load requires disk read → JPEG decode or LibRaw preview extraction → disk write. Subsequent loads hit disk/mem cache and are fast.

### Improvements

**7a. Prefetch on session select**  
When `selectedSession` changes, enqueue background prefetch for all session image URLs at current `cellSize`:
```swift
ThumbnailCache.shared.prefetch(urls: imageURLs, size: CGSize(width: cellSize, height: cellSize * 0.6875))
```
`ThumbnailCache.prefetch` adds ops at `.background` QoS to `decodeQueue` — lower priority than visible cells so on-screen loads always win.

**7b. Priority boost for visible cells**  
`ThumbnailCell.task` already uses `.userInitiated` via `decodeQueue`. No change needed — newly visible cells naturally preempt background prefetch ops.

**7c. Increase decode concurrency**  
Raise `decodeQueue.maxConcurrentOperationCount` from 4 → 6. LibRaw dyld contention only occurs on first-ever decode session; subsequent calls are safe to parallelise more aggressively.

**7d. Mem cache budget**  
Raise `memCache.totalCostLimit` from 200 MB → 400 MB. Modern Macs have ≥16 GB RAM; 400 MB for ~2500 thumbnails at 160pt is safe.

---

## 8. Field Name Clarification (Remove Ratings)

`resetSession()` in ContentView incorrectly uses `clipScore` and `aestheticScore` — these fields do not exist on `ImageRecord`. The correct names (verified from `ProcessingQueue.swift` lines 101–104) are:
- `topiqTechnicalScore`
- `topiqAestheticScore`  
- `clipIQAScore`
- `combinedQualityScore`

`removeRatings()` implementation must use the correct names. `resetSession()` should also be fixed.

---

## 9. File Changes

| File | Change |
|------|--------|
| `ContentView.swift` | 2-column split, detailRecord state, removeRatings(), runAIOnSelected(), compare sheet, keyboard handlers updated, thumbnailSize AppStorage, prefetch call |
| `GridView.swift` | Cmd+A, context menu (Remove Ratings, Run AI), onRate callback, Compare toolbar button, cellSize prop, size slider in toolbar |
| `ThumbnailCell.swift` | Hover star overlay, onRate callback, double-click callback, cellSize-driven frame |
| `ProcessingQueue.swift` | Add `process(imageIDs:onProgress:)` overload |
| `ThumbnailCache.swift` | Add `prefetch(urls:size:)`, raise concurrency to 6, raise mem limit to 400 MB |

No new files required.

---

## 10. Out of Scope

- Changing percentile normalisation logic
- Per-image undo/redo
- Lightroom-style pick flags (separate from star ratings)
