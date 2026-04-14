# Focal

AI-powered photo culling and rating for macOS.

Focal is a native macOS app that helps photographers quickly cull bad shots and rate keepers. It uses a two-phase AI pipeline — first removing blurry, over/underexposed, and eyes-closed photos, then ranking survivors by technical and aesthetic quality — all on-device with no cloud upload.

## Features

- **Import RAW + JPEG** — supports Fuji RAF, Nikon NEF, Canon CR3, Sony ARW, and all LibRaw-supported formats
- **Two-phase AI pipeline** — Phase 1 culls bad shots (blur, exposure, eyes-closed); Phase 2 rates survivors using TOPIQ and CLIP-IQA models on Apple Neural Engine
- **Manual rating override** — keyboard-driven workflow: ← → to navigate, 0–5 to rate, X to reject, Space to open full-res detail
- **XMP sidecar export** — writes `xmp:Rating` and `MicrosoftPhoto:Rating` to `.xmp` files compatible with Lightroom, Capture One, Bridge, and Darktable
- **Session history** — Core Data persistence; re-open sessions and re-rate without re-running the pipeline
- **Configurable settings** — cull strictness, model weights, thumbnail size, XMP auto-write via Preferences (⌘,)

## Requirements

- macOS 14.0 (Sonoma) or later
- Apple Silicon recommended (Neural Engine acceleration for ML models)
- LibRaw (for RAW file support) — see Build from Source

## Installation

1. Download the latest `Focal-x.x.x.dmg` from [GitHub Releases](https://github.com/dknathalage/image-rating/releases)
2. Open the DMG and drag **Focal.app** to your Applications folder
3. Launch Focal — on first run it will download the AI models (~200 MB)

> **Note:** Focal is unsigned. On first launch, right-click → Open to bypass Gatekeeper.

## Build from Source

### Prerequisites

```bash
brew install libraw xcodegen
```

### Steps

```bash
git clone https://github.com/dknathalage/image-rating.git
cd image-rating
xcodegen generate
open Focal.xcodeproj
```

Build and run the `Focal` scheme in Xcode.

Alternatively, from the command line:

```bash
xcodebuild build -scheme Focal -destination 'platform=macOS,arch=arm64'
```

### Model Setup

On first launch Focal automatically downloads the required Core ML models from GitHub Releases and stores them in `~/Library/Application Support/ImageRater/models/`. No manual setup needed.

For offline use: download the `.mlpackage` files manually, then import them via **Model Store** in the app sidebar.

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `←` / `→` | Navigate to previous / next image |
| `0`–`5` | Set star rating (0 = unrated) |
| `X` | Reject image (1 star) |
| `Space` | Toggle full-res detail modal |
| `⌘,` | Open Preferences |

## Contributing

Issues and PRs welcome. Please open an issue before starting large changes.

## License

MIT — see [LICENSE](LICENSE).
