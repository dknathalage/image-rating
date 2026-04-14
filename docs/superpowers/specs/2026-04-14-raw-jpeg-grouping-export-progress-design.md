# RAW+JPEG Grouping & Export Progress Bar — Design Spec
**Date:** 2026-04-14
**Status:** Approved

---

## Overview

Two features:
1. **RAW+JPEG Grouping** — files sharing a base name (e.g. `IMG_001.jpg` + `IMG_001.cr2`) are treated as one unit. One image shown in grid, AI scored from the RAW, metadata exported to all companion files.
2. **Export Progress Bar** — async export with live progress UI replacing the synchronous fire-and-forget loop.

---

## Feature 1: RAW+JPEG Grouping

### Data Model

Three new fields on `ImageRecord` (new CoreData model version, lightweight migration):

| Field | Type | Default | Purpose |
|---|---|---|---|
| `groupID` | String? | nil | UUID string shared by all files in a pair. nil = unpaired. |
| `isGroupPrimary` | Bool | true | true = shown in grid and scored. false = companion (hidden from grid, skipped in pipeline). |
| `scoringFilePath` | String? | nil | When set on a primary, pipeline decodes this path for AI scoring instead of `filePath`. Points from JPEG primary → its first RAW companion (by `localizedStandardCompare`). |

### Pairing Rules (applied at import time)

- Group all scanned URLs by `lowercased(baseName)`.
- **Pair case** (exactly one JPEG + ≥1 RAW with same base name):
  - JPEG → `isGroupPrimary=true`, `scoringFilePath` = first RAW path (by `localizedStandardCompare`, same order as `scanFolder`), `groupID` = new UUID string
  - Each RAW → `isGroupPrimary=false`, `groupID` = same UUID, `scoringFilePath=nil`
- **Multiple JPEGs + RAWs (ambiguous)** → treat all files as unpaired singles.
- **Multiple RAWs with same base name, no JPEG** → all treated as unpaired singles (orphan RAWs).
- **JPEG-only or single-RAW-only** → unpaired single; `groupID=nil`, `isGroupPrimary=true`, `scoringFilePath=nil`.
- **Orphan RAW** (no matching JPEG) → unpaired single; rated normally via RAW decode.

**RAW extension list:** Promote `rawExtensions` in `MetadataWriter.swift` from `private` to `internal` (remove `private` keyword). `ImageImporter` references `MetadataWriter.rawExtensions` directly — no duplication.

### Import Changes (`ImageImporter.swift`)

1. `scanFolder` returns `[URL]` sorted by `localizedStandardCompare` on `lastPathComponent` (unchanged).
2. New `groupByBaseName([URL]) -> [[URL]]` groups URLs by lowercased base name, preserving sort order within each group.
3. For each group, apply pairing rules above before creating `ImageRecord` objects.
4. Companion records (`isGroupPrimary=false`) are created in CoreData but excluded from grid display.
5. `groupID` is a `UUID().uuidString` generated once per pair at import time.

### Pipeline Changes (`ProcessingQueue.swift`)

**Pass 1 (per-image inference):**
- Skip any record where `isGroupPrimary == false`. Companions never enter the pipeline.
- Decode URL = `record.scoringFilePath ?? record.filePath`. Paired JPEG primaries decode from their RAW companion for AI scoring.
- After culling a primary: copy `cullRejected`, `cullReason`, `blurScore`, `exposureScore` to all companion records with matching `groupID`. These fields are available after Pass 1.
- Progress `total` count = primary records only.

**After Pass 2 (diversity/clustering) — companion sync:**
- After `runDiversityPass` completes, fetch all primary records that have a non-nil `groupID`.
- For each such primary, fetch its companions (same `groupID`, `isGroupPrimary=false`) and copy: `ratingStars`, `combinedQualityScore`, `finalScore`, `diversityFactor`, `clusterID`, `clusterRank`.
- Companions do not receive `clipEmbedding` (not displayed, not clustered).
- This is a distinct step after Pass 2, not during Pass 1.

**Pass 2 (diversity/clustering):** Unchanged. Companions were skipped in Pass 1 so they never appear here.

### Grid/UI Changes

- **ContentView** — add `isGroupPrimary == true` filter inside the existing `filteredImages` computed property (not at the `GridView` call site). Companions are excluded here.
- **ThumbnailCell** — show a small `"RAW"` pill badge (bottom-left corner) when `record.scoringFilePath != nil`, indicating this is a JPEG+RAW pair scored from RAW. No other cell changes.
- **GridView** — no changes.

---

## Feature 2: Export Progress Bar

### Export Logic Changes (`ContentView.swift`)

**Companion writes:**
After writing metadata for a primary record, look up all `ImageRecord`s with the same `groupID` and `isGroupPrimary=false`. Write the same star rating to each companion's `filePath`. Companion writes happen inline in the same loop iteration (not separately counted for progress).

**Async export:**
`exportMetadata` becomes `@MainActor async` — keeps all CoreData access on main thread (same as current sync version), but yields between records for UI updates and supports cancellation.

```swift
@MainActor
private func exportMetadata(session: Session) async {
    let primaryRecords = (session.images?.allObjects as? [ImageRecord] ?? [])
        .filter { $0.isGroupPrimary }
    let total = primaryRecords.count
    for (index, record) in primaryRecords.enumerated() {
        guard !Task.isCancelled else { break }
        // write primary + companions
        exportProgress = (done: index + 1, total: total)
    }
    exportProgress = nil
}
```

The export is triggered via:
```swift
exportTask = Task { await exportMetadata(session: session) }
```

`exportTask` is typed `Task<Void, Never>` — no throwing needed since `Task.isCancelled` is checked (not `checkCancellation()`), so the closure body is non-throwing.

### Progress UI (`ContentView.swift`)

New state:
```swift
@State private var exportProgress: (done: Int, total: Int)? = nil
@State private var exportTask: Task<Void, Never>? = nil
```

UI behavior:
- While `exportProgress == nil`: show "Export XMP" toolbar button (existing).
- While `exportProgress != nil`: show export progress in the **same `.safeAreaInset(edge: .bottom)` bar** as `processingStatus`, not in the toolbar. Display `ProgressView(value: Double(done), total: Double(total))`, label `"Exporting \(done) / \(total)…"`, and a Cancel button that calls `exportTask?.cancel()`.
- Export progress bar and processing status bar are mutually exclusive (processing finishes before export is available).
- On cancellation: set `exportProgress = nil`. Partially-written files remain (metadata writes are atomic per file — no rollback needed).

---

## Files Changed

| File | Change |
|---|---|
| `ImageRater.xcdatamodeld` | New model version: add `groupID`, `isGroupPrimary`, `scoringFilePath` to `ImageRecord` |
| `ImageRater/Export/MetadataWriter.swift` | Promote `rawExtensions` from `private` to `internal` |
| `ImageRater/Import/ImageImporter.swift` | Add `groupByBaseName`, apply pairing rules before record creation |
| `ImageRater/Pipeline/ProcessingQueue.swift` | Skip companions in Pass 1, use `scoringFilePath`, copy cull fields in Pass 1, copy diversity fields after Pass 2 |
| `ImageRater/App/ContentView.swift` | Add `isGroupPrimary` filter in `filteredImages`, async export with `exportTask`, progress state + bottom bar UI |
| `ImageRater/UI/Components/ThumbnailCell.swift` | RAW pill badge when `scoringFilePath != nil` |

---

## Out of Scope

- Re-pairing after import (changing which file is primary post-import)
- Manual pairing UI (drag-to-pair)
- Showing companion file count in detail view
- Export rollback on cancellation
- Clearing grouping fields on `resetSession` (pairing is import-time permanent; a new import re-pairs)
