# Thumbnail Loading Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the broken `shouldLoad`/`stableScrollOffset` mechanism and let LazyVGrid + Swift task lifecycle manage thumbnail loading reliably.

**Architecture:** `ThumbnailCell` loads unconditionally via `.task`; LazyVGrid evicts cells (and cancels tasks) when they scroll out of its render window. `ThumbnailCache` gets continuation safety guards and higher decode concurrency. `GridView` loses all load-window state but retains scroll-offset tracking for edge-scroll.

**Tech Stack:** Swift 6, SwiftUI (macOS 15), Swift Concurrency (actors, structured tasks), NSCache, OperationQueue, LibRaw via `LibRawWrapper`, ImageIO (`CGImageSourceCreateThumbnailAtIndex`).

---

## File Map

| File | What changes |
|------|-------------|
| `ImageRater/Import/ThumbnailCache.swift` | Remove `prefetch`/`prefetchQueue`; add `resumed` safety flag to both continuations; raise decode concurrency 3→4 |
| `ImageRater/UI/Components/ThumbnailCell.swift` | Remove `shouldLoad` prop; simplify task id; simplify `imageLayer` to 2-state |
| `ImageRater/UI/GridView.swift` | Remove `stableScrollOffset`, `prefetchDebounce`, `viewportWidth`, `isNearViewport`, `schedulePrefetch`; simplify `onPreferenceChange` and `onAppear`; remove `shouldLoad:` arg from `ThumbnailCell` init |

No new files. No test files for these UI components exist — verification is via build + manual run.

---

## Spec deviation note

The spec states the scroll-offset `GeometryReader` (ZStack `.background`), `ScrollOffsetKey`, and `onPreferenceChange(ScrollOffsetKey)` should be deleted. **This is incorrect** — `handleDrag` converts drag coordinates from grid-space to viewport-space using `scrollOffset` (`let viewportY = c.y - scrollOffset`), so `scrollOffset` must stay live. The scroll-offset GeometryReader, `ScrollOffsetKey`, and `onPreferenceChange` are **retained**. Only `schedulePrefetch()` is removed from the handler body.

---

## Task 1: Fix ThumbnailCache — continuation safety + remove prefetch

**Files:**
- Modify: `ImageRater/Import/ThumbnailCache.swift`

### What to do

Three independent changes in one commit:

**1a — Add `resumed` flag to `readFromDisk` BlockOperation.**

Current `op.addExecutionBlock` body (lines ~123–129):
```swift
op.addExecutionBlock {
    if op.isCancelled { c.resume(returning: nil); return }
    guard let data = try? Data(contentsOf: url),
          let img = NSImage(data: data) else {
        c.resume(returning: nil); return
    }
    c.resume(returning: img)
}
```

Replace with:
```swift
op.addExecutionBlock {
    var resumed = false
    defer { if !resumed { c.resume(returning: nil) } }
    if op.isCancelled { c.resume(returning: nil); resumed = true; return }
    guard let data = try? Data(contentsOf: url),
          let img = NSImage(data: data) else {
        c.resume(returning: nil); resumed = true; return
    }
    c.resume(returning: img)
    resumed = true
}
```

**1b — Add `resumed` flag to the decode BlockOperation** inside `thumbnail(for:)`.

Current `op.addExecutionBlock` body (lines ~55–66):
```swift
op.addExecutionBlock {
    if op.isCancelled { c.resume(returning: nil); return }
    guard let cgImage = ThumbnailCache.decodeThumbnail(url: url, size: size) else {
        c.resume(returning: nil); return
    }
    let actualSize = CGSize(width: cgImage.width, height: cgImage.height)
    let img = NSImage(cgImage: cgImage, size: actualSize)
    let rep = NSBitmapImageRep(cgImage: cgImage)
    if let data = rep.representation(using: .jpeg, properties: [.compressionFactor: 0.8]) {
        try? data.write(to: diskURL)
    }
    c.resume(returning: img)
}
```

Replace with:
```swift
op.addExecutionBlock {
    var resumed = false
    defer { if !resumed { c.resume(returning: nil) } }
    if op.isCancelled { c.resume(returning: nil); resumed = true; return }
    guard let cgImage = ThumbnailCache.decodeThumbnail(url: url, size: size) else {
        c.resume(returning: nil); resumed = true; return
    }
    let actualSize = CGSize(width: cgImage.width, height: cgImage.height)
    let img = NSImage(cgImage: cgImage, size: actualSize)
    let rep = NSBitmapImageRep(cgImage: cgImage)
    if let data = rep.representation(using: .jpeg, properties: [.compressionFactor: 0.8]) {
        try? data.write(to: diskURL)
    }
    c.resume(returning: img)
    resumed = true
}
```

**1c — Raise `decodeQueue.maxConcurrentOperationCount` from 3 to 4.**

Find (lines ~148–152):
```swift
private static let decodeQueue: OperationQueue = {
    let q = OperationQueue()
    q.maxConcurrentOperationCount = 3
    q.qualityOfService = .userInitiated
    return q
}()
```

Change `3` → `4`.

**1d — Delete `prefetch` method and `prefetchQueue` property.**

Delete lines ~82–102 (the `prefetch(urls:size:limit:)` method) and lines ~155–161 (the `prefetchQueue` static property):

```swift
// DELETE THIS METHOD:
func prefetch(urls: [URL], size: CGSize, limit: Int = 20) {
    Self.prefetchQueue.cancelAllOperations()
    // ...
}

// DELETE THIS PROPERTY:
private static let prefetchQueue: OperationQueue = {
    let q = OperationQueue()
    q.maxConcurrentOperationCount = 2
    q.qualityOfService = .background
    return q
}()
```

- [ ] **Step 1: Apply changes 1a–1d to ThumbnailCache.swift**

- [ ] **Step 2: Build and verify no errors**

```bash
xcodebuild -scheme ImageRater -configuration Debug build 2>&1 | grep -E "error:|BUILD SUCCEEDED|BUILD FAILED"
```
Expected: `** BUILD SUCCEEDED **`

- [ ] **Step 3: Commit**

```bash
git add ImageRater/Import/ThumbnailCache.swift
git commit -m "refactor: simplify ThumbnailCache — remove prefetch, add continuation safety, raise decode concurrency"
```

---

## Task 2: Simplify ThumbnailCell — remove shouldLoad

**Files:**
- Modify: `ImageRater/UI/Components/ThumbnailCell.swift`

### What to do

**2a — Remove `shouldLoad: Bool` parameter.**

Delete this line from the struct's stored properties:
```swift
let shouldLoad: Bool
```

**2b — Simplify `.task` block** (lines ~85–100).

Replace the entire `.task(...)` block:
```swift
// OLD — delete this:
.task(id: "\(record.objectID.uriRepresentation())-\(cellSize)-\(shouldLoad)") {
    guard shouldLoad else {
        thumbnail = nil
        return
    }
    let url = URL(filePath: record.filePath ?? "")
    let img = await ThumbnailCache.shared.thumbnail(
        for: url,
        size: CGSize(width: cellSize, height: cellSize * 0.6875)
    )
    guard !Task.isCancelled else { return }
    thumbnail = img
}
```

With:
```swift
// NEW:
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

**2c — Simplify `imageLayer` to 2-state** (lines ~108–122).

Replace the `imageLayer` computed property:
```swift
// OLD — delete this:
@ViewBuilder
private var imageLayer: some View {
    if let thumb = thumbnail {
        Image(nsImage: thumb)
            .resizable()
            .aspectRatio(contentMode: .fill)
    } else if shouldLoad {
        Rectangle()
            .fill(Color.secondary.opacity(0.2))
            .overlay(SpinnerView(size: 20))
    } else {
        Rectangle()
            .fill(Color.secondary.opacity(0.08))
    }
}
```

With:
```swift
// NEW:
@ViewBuilder
private var imageLayer: some View {
    if let thumb = thumbnail {
        Image(nsImage: thumb)
            .resizable()
            .aspectRatio(contentMode: .fill)
    } else {
        Rectangle()
            .fill(Color.secondary.opacity(0.2))
            .overlay(SpinnerView(size: 20))
    }
}
```

- [ ] **Step 1: Apply changes 2a–2c to ThumbnailCell.swift**

- [ ] **Step 2: Build — expect compiler error about missing `shouldLoad:` argument at call site in GridView**

```bash
xcodebuild -scheme ImageRater -configuration Debug build 2>&1 | grep -E "error:|BUILD SUCCEEDED|BUILD FAILED"
```
Expected: build error referencing `shouldLoad` in GridView. This is expected — Task 3 fixes it.

- [ ] **Step 3: Do NOT commit yet — wait for Task 3 to make the build green**

---

## Task 3: Simplify GridView — remove load-window machinery

**Files:**
- Modify: `ImageRater/UI/GridView.swift`

### What to do

**3a — Remove state vars** (lines ~39–42). Delete these three lines:

```swift
@State private var stableScrollOffset: CGFloat = 0
@State private var prefetchDebounce: Timer?
// and remove viewportWidth from the retained vars below:
@State private var viewportWidth: CGFloat = 0
```

Keep `@State private var viewportHeight: CGFloat = 0` and `@State private var scrollOffset: CGFloat = 0`.

**3b — Remove `shouldLoad:` argument from ThumbnailCell init** (line ~63).

Find:
```swift
ThumbnailCell(
    record: record,
    isSelected: selectedIDs.contains(record.objectID),
    cellSize: cellSize,
    shouldLoad: isNearViewport(index: idx)
) { mods in
```

Replace with:
```swift
ThumbnailCell(
    record: record,
    isSelected: selectedIDs.contains(record.objectID),
    cellSize: cellSize
) { mods in
```

**3c — Simplify the viewport-size GeometryReader `onAppear`** (lines ~118–126).

Replace:
```swift
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
```

With:
```swift
.onAppear {
    Task { @MainActor in
        viewportHeight = geo.size.height
    }
}
.onChange(of: geo.size) { _, s in viewportHeight = s.height }
```

**3d — Simplify `onPreferenceChange(ScrollOffsetKey.self)`** (lines ~130–137).

The scroll-offset preference and GeometryReader are **retained** (needed for edge-scroll coordinate conversion). Only `schedulePrefetch()` is removed from the handler.

Replace:
```swift
.onPreferenceChange(ScrollOffsetKey.self) { offset in
    // onPreferenceChange fires during the view update phase.
    // Defer state mutations to avoid "Modifying state during view update".
    Task { @MainActor in
        scrollOffset = offset
        schedulePrefetch()
    }
}
```

With:
```swift
.onPreferenceChange(ScrollOffsetKey.self) { offset in
    Task { @MainActor in
        scrollOffset = offset
    }
}
```

**3e — Delete `schedulePrefetch()` function** (lines ~173–196). Delete the entire method:

```swift
// DELETE THIS METHOD:
private func schedulePrefetch() {
    prefetchDebounce?.invalidate()
    prefetchDebounce = Timer.scheduledTimer(withTimeInterval: 0.25, repeats: false) { _ in
        Task { @MainActor in
            stableScrollOffset = scrollOffset
            // ...
            await ThumbnailCache.shared.prefetch(urls: urls, size: size)
        }
    }
}
```

**3f — Delete `isNearViewport(index:)` function** (lines ~200–213). Delete the entire method:

```swift
// DELETE THIS METHOD:
private func isNearViewport(index: Int) -> Bool {
    // ...
}
```

**3g — Delete `ScrollOffsetKey`** — NO, retain it. It feeds `scrollOffset` for edge-scroll. Leave `ScrollOffsetKey` at the bottom of the file unchanged.

- [ ] **Step 1: Apply changes 3a–3f to GridView.swift**

- [ ] **Step 2: Build and verify clean**

```bash
xcodebuild -scheme ImageRater -configuration Debug build 2>&1 | grep -E "error:|BUILD SUCCEEDED|BUILD FAILED"
```
Expected: `** BUILD SUCCEEDED **`

- [ ] **Step 3: Commit both Task 2 and Task 3 together**

```bash
git add ImageRater/UI/Components/ThumbnailCell.swift ImageRater/UI/GridView.swift
git commit -m "refactor: remove shouldLoad mechanism — let LazyVGrid manage cell lifecycle"
```

---

## Task 4: Manual verification

- [ ] **Step 1: Launch the app**

Open the project in Xcode and run, or:
```bash
open /Users/dknathalage/repos/image-rating/ImageRater.xcodeproj
```

- [ ] **Step 2: Open a session with many images (1000+)**

Expected: thumbnails load progressively as cells appear in LazyVGrid's window. No stall after rows 1-2.

- [ ] **Step 3: Scroll quickly up and down**

Expected: cells outside LazyVGrid's render window evict (task cancelled), cells scrolled back into view reload from disk cache quickly (diskReadQueue, 8 concurrent). No stale images.

- [ ] **Step 4: Verify rubber band selection still works**

Click and drag across multiple cells. Expected: selection rect appears, cells within rect become selected.

- [ ] **Step 5: Verify edge-scroll still works during rubber band drag**

Drag to within 40pt of the top or bottom edge of the grid. Expected: grid auto-scrolls.
