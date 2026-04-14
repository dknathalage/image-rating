# Focal — Production Readiness Design Spec
Date: 2026-04-14

## Overview

Prepare the ImageRater macOS app for production release as **Focal** — an AI-powered photo culling and rating tool. Scope: rename, README + docs, persistent Settings/Preferences window, About window, GitHub Actions CI, DMG release pipeline, CHANGELOG, and proper versioning.

Distribution: open source + unsigned DMG via GitHub Releases. No Mac App Store, no notarization for now.

---

## 1. Rename: ImageRater → Focal

**Files to update:**
- `project.yml` — `name: Focal`, bundle ID `com.focal.app`, scheme name `Focal`, test target `FocalTests`
- `Info.plist` — `CFBundleName: Focal`, copyright line
- `ImageRaterApp.swift` — struct `FocalApp`
- All `import`/`@testable import ImageRater` in test files → `FocalApp` (target rename)
- `models-manifest.json` — download URLs remain pointing to `dknathalage/image-rating` repo (no rename needed unless repo is renamed)
- Logger subsystem strings: `com.imagerating` → `com.focal.app`

Source folder `ImageRater/` stays as-is (internal only).

---

## 2. README + Docs

`README.md` at repo root. Sections:

1. **Title + tagline** — "Focal — AI-powered photo culling and rating for macOS"
2. **Screenshot** — placeholder (`docs/screenshots/` dir, note to add)
3. **Features** — import RAW/JPEG folders, two-phase AI pipeline (cull then rate), manual rating override, XMP sidecar export, keyboard-driven workflow, session history
4. **Requirements** — macOS 14.0+, Apple Silicon recommended (Neural Engine acceleration), LibRaw via Homebrew
5. **Installation** — download latest `.dmg` from GitHub Releases (`https://github.com/dknathalage/image-rating/releases`), drag to Applications
6. **Build from source** — step-by-step:
   ```
   brew install libraw xcodegen
   xcodegen generate
   open Focal.xcodeproj
   ```
   Build target `Focal` in Xcode or `xcodebuild`.
7. **Model setup** — first launch auto-downloads models via `models-manifest.json`; for offline use, import `.mlpackage` manually via Model Store
8. **Keyboard shortcuts** — reference table (←/→ navigate, 0–5 rate, X reject, Space detail view)
9. **License** — MIT (or whatever the repo uses)

`CHANGELOG.md` at repo root with initial `## [1.0.0] - 2026-04-14` entry listing all shipped features.

---

## 3. Settings / Preferences Window

Accessible via `Cmd+,` (standard macOS convention). Implemented as `PreferencesView` presented as a `Settings` scene in `FocalApp.swift`.

### Storage

New file `FocalSettings.swift` — a namespace of `@AppStorage` keys as static constants. No new types needed; `ContentView` and pipeline code read `@AppStorage` directly.

```swift
enum FocalSettings {
    static let blurThreshold     = "focal.cull.blurThreshold"      // Double, default 500
    static let exposureLeniency  = "focal.cull.exposureLeniency"   // Double, default 0.9
    static let earThreshold      = "focal.cull.earThreshold"       // Double, default 0.2
    static let weightNimaAesth   = "focal.rating.weightNimaAesth"  // Double, default 0.4
    static let weightNimaTech    = "focal.rating.weightNimaTech"   // Double, default 0.3
    static let weightClip        = "focal.rating.weightClip"       // Double, default 0.3
    static let defaultCellSize   = "focal.ui.defaultCellSize"      // Double, default 160
    static let autoWriteXMP      = "focal.export.autoWriteXMP"     // Bool, default true
}
```

Migrate hardcoded values in `ContentView`, `CullPipeline`, `RatingPipeline` to read from `@AppStorage`.

### Tabs

**Pipeline**
- Blur threshold slider (100–2000, step 50)
- Exposure leniency slider (0.5–1.0, step 0.05) — fraction of pixels allowed near extremes
- EAR threshold slider (0.1–0.4, step 0.01) — eye aspect ratio cutoff
- Model weight sliders: NIMA Aesthetic, NIMA Technical, CLIP (0.0–1.0 each, auto-normalize to sum=1)

**Appearance**
- Default thumbnail size slider (80–320 px)

**Export**
- Auto-write XMP on manual rate toggle (default on)

### Integration

- `FocalApp.swift` gains a `Settings { PreferencesView() }` scene — macOS renders it as standard Preferences window
- `ContentView` `@State var cullStrictness` and `@State var cellSize` replaced by `@AppStorage`
- `CullPipeline.cull()` parameters sourced from `UserDefaults` at call site (passed in, pipeline stays pure/testable)

---

## 4. About Window

Standard macOS About box via `NSApp.orderFrontStandardAboutPanel` with custom info dict:
- App name: Focal
- Version from bundle `CFBundleShortVersionString`
- Copyright: "© 2026 Don Athalage"
- Credits: short description of AI models used

Triggered from `Help` menu (standard macOS location) or `Focal` menu → About Focal.
Implemented in `FocalApp.swift` as a `CommandGroup` replacing `.appInfo`.

---

## 5. CI — GitHub Actions

### `test.yml` (trigger: push + PR to main)

```yaml
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: macos-15
    steps:
      - uses: actions/checkout@v4
      - run: brew install libraw xcodegen xcpretty
      - run: xcodegen generate
      - run: xcodebuild test -scheme Focal -destination 'platform=macOS,arch=arm64' | xcpretty
```

### `release.yml` (trigger: push tag `v*`)

```yaml
on:
  push:
    tags: ['v*']

jobs:
  build-dmg:
    runs-on: macos-15
    steps:
      - uses: actions/checkout@v4
      - run: brew install libraw xcodegen create-dmg
      - run: xcodegen generate
      - name: Extract version from tag
        run: echo "VERSION=${GITHUB_REF_NAME#v}" >> $GITHUB_ENV
      - name: Archive (pass version via xcodebuild vars, not PlistBuddy)
        run: |
          xcodebuild archive \
            -scheme Focal \
            -archivePath $RUNNER_TEMP/Focal.xcarchive \
            -destination 'generic/platform=macOS' \
            MARKETING_VERSION=$VERSION \
            CURRENT_PROJECT_VERSION=$GITHUB_RUN_NUMBER \
            CODE_SIGN_IDENTITY="" CODE_SIGNING_REQUIRED=NO CODE_SIGNING_ALLOWED=NO
      - name: Export app (unsigned)
        run: |
          xcodebuild -exportArchive \
            -archivePath $RUNNER_TEMP/Focal.xcarchive \
            -exportOptionsPlist .github/ExportOptions.plist \
            -exportPath $RUNNER_TEMP/export
      - name: Build DMG
        run: |
          create-dmg \
            --volname "Focal" \
            --window-size 600 400 \
            --app-drop-link 450 200 \
            Focal-$VERSION.dmg \
            $RUNNER_TEMP/export/Focal.app
      - name: Create GitHub Release
        run: gh release create $GITHUB_REF_NAME Focal-$VERSION.dmg --title "Focal $VERSION" --notes-file CHANGELOG.md
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

New file `.github/ExportOptions.plist`:
```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>method</key><string>mac-application</string>
  <key>signingStyle</key><string>manual</string>
  <key>provisioningProfileStyle</key><string>automatic</string>
  <key>stripSwiftSymbols</key><true/>
</dict></plist>
```

Version source of truth: `MARKETING_VERSION` and `CURRENT_PROJECT_VERSION` default to `1.0.0` / `1` in `project.yml` `settings.base`. Release workflow overrides them via xcodebuild command-line variables — no Info.plist patching needed.

Tag format: `v1.0.0`. Release notes pulled from `CHANGELOG.md` top section.

---

## 6. Versioning

- `Info.plist` `CFBundleShortVersionString` set to match git tag (e.g. `1.0.0`)
- `CFBundleVersion` set to CI build number (`GITHUB_RUN_NUMBER`) in release workflow
- `MARKETING_VERSION` and `CURRENT_PROJECT_VERSION` set in `project.yml` `settings.base`

---

## 7. CHANGELOG

`CHANGELOG.md` follows [Keep a Changelog](https://keepachangelog.com) format. Initial entry:

```
## [1.0.0] - 2026-04-14
### Added
- AI two-phase pipeline: blur/exposure/EAR cull + NIMA/CLIP rating
- RAW file support via LibRaw (RAF, NEF, CR3, ARW)
- XMP sidecar export (xmp:Rating + MicrosoftPhoto:Rating)
- Session history with Core Data persistence
- Thumbnail grid with keyboard-driven rating workflow
- Detail modal with zoomable full-res view
- Model Store: auto-download + checksum-verify Core ML models
- Settings window (Cmd+,) for cull thresholds, model weights, export prefs
```

---

## File Checklist

| File | Action |
|------|--------|
| `project.yml` | Rename `ImageRaterTests` → `FocalTests`; name, bundle ID, scheme; add `MARKETING_VERSION: "1.0.0"` + `CURRENT_PROJECT_VERSION: "1"` to `settings.base` |
| `Info.plist` | Update CFBundleName, copyright |
| `ImageRater/App/ImageRaterApp.swift` | Rename struct to `FocalApp`, add `Settings` + About `CommandGroup` scenes |
| `ImageRaterTests/` (all files) | Update `@testable import ImageRater` → `@testable import Focal` |
| `models-manifest.json` | No URL change needed; verify placeholder SHA-256 is noted as "populate before release" |
| `ImageRater/App/FocalSettings.swift` | New — AppStorage key constants |
| `ImageRater/UI/PreferencesView.swift` | New — tabbed prefs window |
| `ImageRater/App/ContentView.swift` | Migrate `@State` prefs → `@AppStorage` |
| `ImageRater/Pipeline/CullPipeline.swift` | Rename Logger subsystem → `com.focal.app` |
| `ImageRater/Pipeline/RatingPipeline.swift` | Rename Logger subsystem → `com.focal.app` |
| `ImageRater/ModelStore/ModelStore.swift` | Rename Logger subsystem → `com.focal.app` |
| `ImageRater/Pipeline/ProcessingQueue.swift` | Rename Logger subsystem → `com.focal.app` |
| `README.md` | New at repo root |
| `CHANGELOG.md` | New at repo root |
| `.github/workflows/test.yml` | New — CI build+test |
| `.github/workflows/release.yml` | New — DMG + GitHub Release |
| `.github/ExportOptions.plist` | New — unsigned export config for xcodebuild |

---

## Out of Scope

- Code signing / notarization (deferred)
- Sparkle auto-update (deferred — needs signing)
- Mac App Store (deferred)
- Repo rename (optional — `image-rating` → `focal`)
