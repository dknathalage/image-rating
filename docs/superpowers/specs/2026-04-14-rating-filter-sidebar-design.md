# Rating Filter Sidebar Section

**Date:** 2026-04-14
**Status:** Approved

## Overview

Add a rating filter section below the session list in the sidebar. Users can multi-select star ratings (1–5) and Unrated to filter the image grid. Each row shows the rating label and count of images at that rating in the current session. Empty selection shows all images.

## Requirements

- Filter by EXIF-compatible ratings: Unrated, 1★–5★
- Multi-select: any combination of ratings can be active simultaneously
- Empty selection = show all images (no filter applied)
- Each filter row displays image count for that rating in the current session
- Zero-count ratings are shown but dimmed
- Layout: list rows below the session list, matching native macOS sidebar style

## Design

### State

`ContentView` owns the filter state:

```swift
@State private var ratingFilter: Set<Int> = []
// 0 = Unrated, 1–5 = star ratings, empty = show all
```

### Effective Rating

Derived per `ImageRecord`:
- If `userOverride != nil && userOverride != 0`: use `Int(userOverride)`
- Else if `ratingStars != nil`: use `Int(ratingStars!)`
- Else: `0` (Unrated)

### New Component: `RatingFilterView`

**File:** `ImageRater/UI/Components/RatingFilterView.swift`

- Props: `session: Session?`, `ratingFilter: Binding<Set<Int>>`
- Own `@FetchRequest` scoped to the session to count images per effective rating
- Renders a `List` with native multi-select (`selection: ratingFilter`)
- Each row: star label (e.g. `★★★★★`) + count right-aligned
- Unrated row at bottom
- Zero-count rows shown, dimmed

### Sidebar Integration (`ContentView.swift`)

Below the existing session `List`, add a `Section("Filter by Rating")` containing `RatingFilterView(session: selectedSession, ratingFilter: $ratingFilter)`.

Reset `ratingFilter` to `[]` when `selectedSession` changes (`.onChange(of: selectedSession)`).

### Grid Filtering (`GridView.swift`)

Accept new parameter `ratingFilter: Set<Int>`.

Add computed property:

```swift
var filteredImages: [ImageRecord] {
    guard !ratingFilter.isEmpty else { return Array(images) }
    return images.filter { record in
        let effective: Int
        if let o = record.userOverride, o != 0 {
            effective = Int(o)
        } else if let s = record.ratingStars {
            effective = Int(s)
        } else {
            effective = 0
        }
        return ratingFilter.contains(effective)
    }
}
```

Use `filteredImages` in `LazyVGrid` instead of `images` directly. Navigation helpers (`navigatePrev`/`navigateNext`) in `ContentView` also use `filteredImages` to stay consistent with what's visible.

## Touch Points

| File | Change |
|------|--------|
| `ImageRater/UI/Components/RatingFilterView.swift` | New file |
| `ImageRater/App/ContentView.swift` | Add `ratingFilter` state, sidebar section, reset on session change, pass to GridView |
| `ImageRater/UI/GridView.swift` | Accept `ratingFilter`, add `filteredImages`, update nav helpers |

## Out of Scope

- Cull status filtering (rejected/accepted)
- Sorting by rating
- Persisting filter state across sessions
