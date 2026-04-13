# Image Rating & Culling App — Design Spec
Date: 2026-04-13

## Overview
macOS Swift desktop app. AI-powered two-phase pipeline: cull bad shots first, rate survivors second. Writes results to XMP `xmp:Rating`. Targets both professional photographers (high-volume) and enthusiasts.

## Tech Stack
- Swift 5.9+, macOS 14+
- SwiftUI + AppKit where needed
- Core ML (HuggingFace models converted to .mlpackage)
- Apple Vision framework
- LibRaw (C++ via bridging header) for RAW decode
- Core Data (with lightweight migration) for persistence
- Swift async/await + actors for concurrency
- Swift Package Manager

## Architecture — 4 Layers

### 1. UI Layer (SwiftUI)
- **GridView** — thumbnail grid, AI score badge overlay, pick/reject flag, bulk select/deselect
- **DetailView** — large image, per-model score breakdown panel, manual override controls
- **ModelStoreView** — download, swap, version HuggingFace models
- Tiered UX: default = auto-process on import; power mode = manual pipeline controls, threshold sliders, per-model score visibility

### 2. Core Layer (Swift actors)
- **ImageImporter** — scan folder/files, enumerate JPEG + RAW, build session file list; use embedded JPEG preview from RAW as fast thumbnail path before full LibRaw decode
- **ThumbnailCache** — NSCache (bounded, respects memory pressure) + disk cache; RAW fast path uses embedded preview; full decode fallback via LibRaw
- **ProcessingQueue** — Swift actor, batches images through pipeline, reports progress via AsyncStream; supports structured cancellation via `Task.checkCancellation()` at each phase boundary; on cancel: flush pending writes, set in-progress records to `.interrupted` state
- **MetadataWriter** — write `xmp:Rating` (integer 0–5, XMP Basic namespace `http://ns.adobe.com/xap/1.0/`) and `MicrosoftPhoto:Rating` (namespace `http://ns.microsoft.com/photo/1.0/`, values 0/1/25/50/75/99) to XMP sidecar `.xmp` file alongside original; implemented by building a `CGImageMetadata` object, serializing to raw XMP bytes via `CGImageMetadataCreateXMPData`, then writing those bytes to `<filename>.xmp` via `Data.write(to:)`; no external XMP library needed

### 3. AI Pipeline — Two Phases
**Phase 1 — Cull (Apple Vision + Core Image, on-device, zero download)**
- **Blur detection** — `CIFilter` Laplacian variance on luminance channel; reject if variance below configurable threshold
- **Eyes-closed detection** — `VNDetectFaceLandmarksRequest` → `VNFaceLandmarks2D` eye points → compute Eye Aspect Ratio (EAR = eye height / eye width); reject if EAR below threshold (e.g. 0.2)
- **Exposure analysis** — `CIAreaHistogram` on luminance channel; reject if >90% pixels in top or bottom 5% of range (overexposed / underexposed)
- Output: `CullResult { rejected: Bool, reason: CullReason }` per image

**Phase 2 — Rate (Core ML, runs only on non-rejected images)**
- **OpenCLIP ViT-B/32** (`mlfoundations/open_clip` architecture, MIT; weights: `laion/CLIP-ViT-B-32-laion2B-s34B-b79K`) — semantic/aesthetic embedding. **Note:** weights are governed by the LAION OpenCLIP license (separate from framework MIT license); commercial use permitted but developer must verify current license on the HuggingFace repo before distribution. Fallback: `google/siglip-base-patch16-224` (Apache 2.0, unconditionally commercial-safe).
- **Aesthetic head** — lightweight linear layer trained on LAION-Aesthetics dataset on top of OpenCLIP features; MIT license (open training data + open weights); outputs score 1–10 normalized to 1–5 stars
- Scores averaged with configurable weights (persisted in Core Data `ModelConfig` entity) → final star rating
- Output: `RatingResult { stars: Int, clipScore: Float, aestheticScore: Float }`

### 4. Foundation
- **LibRaw Bridge** — C++ bridging header wrapping LibRaw; exposes `decodeRAW(path:) -> CGImage`; JPEG fast path: try embedded preview first, fall back to full decode
- **ModelStore** — downloads versioned `.mlpackage` from HuggingFace Hub on first launch; stores in `~/Library/Application Support/ImageRater/models/`; verifies SHA-256 checksum against `models-manifest.json` before loading; manifest URL is hardcoded in app binary (not user-configurable) to prevent redirect attacks; manifest itself is signed with an Ed25519 key (public key bundled in app binary) — verified before trusting any checksum or download URL; supports model swap/update without app update; "no network during processing" constraint met — download happens at session start if model absent, never mid-batch; `MLModelConfiguration.computeUnits = .cpuAndNeuralEngine` on Apple Silicon, `.cpuOnly` on Intel
- **Core Data** — entities: `Session`, `ImageRecord` (path, thumbHash, cullResult, ratingResult, userOverride, processState), `ModelConfig`; lightweight migration enabled; `userOverride` is sticky across re-runs (pipeline skips re-rating images where `userOverride != nil` unless user explicitly resets)

## Data Flow
```
User opens folder
  → ImageImporter scans → Session + ImageRecord list in Core Data
  → ThumbnailCache: RAW embedded preview fast path → disk cache
  → ModelStore: verify models present + checksums valid (download if missing)
  → ProcessingQueue dispatches Phase 1 (Vision + CIFilter) concurrently per image
  → Rejected images flagged, skipped in Phase 2
  → Phase 2 (Core ML) runs on survivors, writes star scores
  → UI observes Core Data → GridView/DetailView update live
  → User reviews, overrides if needed (override sticky across re-runs)
  → Export: MetadataWriter writes xmp:Rating + XMP sidecar per image
```

## Error Handling
- LibRaw decode fail → show placeholder, mark `ImageRecord.decodeError = true`, skip pipeline
- Model download fail → retry 3× with exponential backoff; surface error in ModelStoreView with manual retry button
- Checksum mismatch → delete corrupted download, surface error, prompt re-download
- Core ML inference fail → log, assign null score, show "unrated" badge; do not block rest of batch
- MetadataWriter fail (read-only file) → surface per-image write error in DetailView, allow retry

## Testing
- Unit: blur detection via `CIFilter` Laplacian with fixture images (blurry vs sharp)
- Unit: EAR eyes-closed logic with known landmark point sets
- Unit: exposure histogram rejection with synthetic over/under-exposed CGImages
- Unit: `MetadataWriter` round-trip (write `xmp:Rating` → read back, assert value)
- Unit: `LibRaw` bridge with sample RAW files (CR3, NEF, ARW, RAF)
- Unit: `ProcessingQueue` cancellation — assert `.interrupted` state on cancel mid-batch
- Integration: full pipeline on fixture image folder, assert expected cull/rate outcomes
- No automated UI tests; manual testing on fixture image folder

## HuggingFace Models (Verified Commercial Use)
| Model | License | Task | Notes |
|---|---|---|---|
| laion/CLIP-ViT-B-32-laion2B-s34B-b79K | MIT | Semantic embedding | OpenCLIP, trained on open data |
| custom aesthetic head on OpenCLIP | MIT | Aesthetic score | Linear layer trained on LAION-Aesthetics |

Models converted to Core ML `.mlpackage` via `coremltools` (developer step, not user-facing). SHA-256 checksums stored in `models-manifest.json` hosted remotely (not bundled); fetched on update check.

## Key Constraints
- macOS 14+ required
- Apple Silicon recommended (Neural Engine); Intel fallback via CPU
- No Python runtime bundled in app
- Models stored outside app bundle (swappable without app update)
- All inference on-device; network only for initial model download
- `xmp:Rating` (0–5) is the canonical output tag, readable by Lightroom, Capture One, Photos.app
