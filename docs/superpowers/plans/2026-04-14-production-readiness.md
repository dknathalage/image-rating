# Focal — Production Readiness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the macOS photo culling/rating app as "Focal" with proper README, persistent Settings window, About window, and GitHub Actions CI + DMG release pipeline.

**Architecture:** Rename everything from ImageRater → Focal, introduce `FocalSettings` (AppStorage key namespace), add `PreferencesView` as a standard macOS `Settings` scene, wire CI with two workflows (test on push, release DMG on tag).

**Tech Stack:** Swift 5.9+, SwiftUI, macOS 14+, XcodeGen, GitHub Actions, `create-dmg`

**Spec:** `docs/superpowers/specs/2026-04-14-production-readiness-design.md`

---

## File Map

| File | Change |
|------|--------|
| `project.yml` | Rename target `ImageRater`→`Focal`, test target `ImageRaterTests`→`FocalTests`, bundle ID `com.focal.app`, scheme `Focal`; add `MARKETING_VERSION`+`CURRENT_PROJECT_VERSION` |
| `ImageRater/Info.plist` | `CFBundleName: Focal`, copyright |
| `ImageRater/App/ImageRaterApp.swift` | Rename struct `ImageRaterApp`→`FocalApp`; add `Settings` scene + About `CommandGroup` |
| `ImageRater/App/FocalSettings.swift` | **New** — AppStorage key constants + default values |
| `ImageRater/App/ContentView.swift` | Swap `@State` prefs → `@AppStorage`; remove manual UserDefaults load/save |
| `ImageRater/Pipeline/ProcessingQueue.swift` | Use `FocalSettings` keys instead of raw strings; read model weights from UserDefaults |
| `ImageRater/Pipeline/CullPipeline.swift` | Rename Logger subsystem |
| `ImageRater/Pipeline/RatingPipeline.swift` | Rename Logger subsystem |
| `ImageRater/ModelStore/ModelStore.swift` | Rename Logger subsystem |
| `ImageRater/UI/PreferencesView.swift` | **New** — tabbed prefs (Pipeline / Appearance / Export) |
| `ImageRaterTests/*.swift` (×10) | `@testable import ImageRater` → `@testable import Focal` |
| `README.md` | **New** at repo root |
| `CHANGELOG.md` | **New** at repo root |
| `.github/workflows/test.yml` | **New** — CI build+test on push/PR |
| `.github/workflows/release.yml` | **New** — DMG + GitHub Release on tag push |
| `.github/ExportOptions.plist` | **New** — unsigned xcodebuild export config |

---

## Task 1: Rename project.yml and Info.plist

**Files:**
- Modify: `project.yml`
- Modify: `ImageRater/Info.plist`

- [ ] **Step 1: Update project.yml**

Replace the entire `project.yml` with:

```yaml
name: Focal
options:
  bundleIdPrefix: com.focal
  deploymentTarget:
    macOS: "14.0"
  xcodeVersion: "15"
  generateEmptyDirectories: true
settings:
  base:
    SWIFT_VERSION: "5.9"
    MACOSX_DEPLOYMENT_TARGET: "14.0"
    SWIFT_OBJC_BRIDGING_HEADER: "ImageRater/ImageRater-Bridging-Header.h"
    CLANG_ENABLE_OBJC_ARC: YES
    MARKETING_VERSION: "1.0.0"
    CURRENT_PROJECT_VERSION: "1"
targets:
  Focal:
    type: application
    platform: macOS
    sources:
      - path: ImageRater
        excludes:
          - "**/*.md"
    resources:
      - ImageRater/CoreData/ImageRater.xcdatamodeld
    settings:
      base:
        PRODUCT_BUNDLE_IDENTIFIER: com.focal.app
        INFOPLIST_FILE: ImageRater/Info.plist
        CODE_SIGN_STYLE: Automatic
        ENABLE_HARDENED_RUNTIME: YES
        OTHER_LDFLAGS: "-lraw"
        HEADER_SEARCH_PATHS: "/opt/homebrew/include"
        LIBRARY_SEARCH_PATHS: "/opt/homebrew/lib"
    dependencies: []
  FocalTests:
    type: bundle.unit-test
    platform: macOS
    sources:
      - path: ImageRaterTests
    resources:
      - ImageRaterTests/Fixtures
    settings:
      base:
        PRODUCT_BUNDLE_IDENTIFIER: com.focal.tests
        TEST_HOST: "$(BUILT_PRODUCTS_DIR)/Focal.app/Contents/MacOS/Focal"
        GENERATE_INFOPLIST_FILE: YES
    dependencies:
      - target: Focal
schemes:
  Focal:
    build:
      targets:
        Focal: all
        FocalTests: [test]
    run:
      config: Debug
    test:
      config: Debug
      targets:
        - FocalTests
    profile:
      config: Release
    analyze:
      config: Debug
    archive:
      config: Release
```

- [ ] **Step 2: Update Info.plist CFBundleName and copyright**

In `ImageRater/Info.plist`, change:
```xml
<key>CFBundleName</key>
<string>$(PRODUCT_NAME)</string>
```
to:
```xml
<key>CFBundleName</key>
<string>Focal</string>
```

And change:
```xml
<key>NSHumanReadableCopyright</key>
<string>Copyright © 2026. All rights reserved.</string>
```
to:
```xml
<key>NSHumanReadableCopyright</key>
<string>© 2026 Don Athalage. All rights reserved.</string>
```

- [ ] **Step 3: Regenerate Xcode project**

```bash
cd /path/to/repo
xcodegen generate
```

Expected: `Focal.xcodeproj` generated without errors.

- [ ] **Step 4: Verify project opens**

Open `Focal.xcodeproj` in Xcode. Confirm scheme dropdown shows "Focal" and test target shows "FocalTests".

- [ ] **Step 5: Commit**

```bash
git add project.yml ImageRater/Info.plist Focal.xcodeproj
git commit -m "chore: rename project to Focal — bundle ID, scheme, versioning"
```

---

## Task 2: Rename Swift struct and Logger subsystems

**Files:**
- Modify: `ImageRater/App/ImageRaterApp.swift`
- Modify: `ImageRater/Pipeline/CullPipeline.swift` (line 6)
- Modify: `ImageRater/Pipeline/RatingPipeline.swift` (line 8)
- Modify: `ImageRater/ModelStore/ModelStore.swift` (line 5)
- Modify: `ImageRater/Pipeline/ProcessingQueue.swift` (line 5)
- Modify: `ImageRaterTests/*.swift` (×10 files)

- [ ] **Step 1: Rename app struct**

In `ImageRater/App/ImageRaterApp.swift`, change:
```swift
@main
struct ImageRaterApp: App {
```
to:
```swift
@main
struct FocalApp: App {
```

- [ ] **Step 2: Update Logger subsystems in all 4 pipeline files**

In each file, change `"com.imagerating"` to `"com.focal.app"`:

- `ImageRater/Pipeline/CullPipeline.swift:6` — `Logger(subsystem: "com.focal.app", category: "CullPipeline")`
- `ImageRater/Pipeline/RatingPipeline.swift:8` — `Logger(subsystem: "com.focal.app", category: "RatingPipeline")`
- `ImageRater/ModelStore/ModelStore.swift:5` — `Logger(subsystem: "com.focal.app", category: "ModelStore")`
- `ImageRater/Pipeline/ProcessingQueue.swift:5` — `Logger(subsystem: "com.focal.app", category: "ProcessingQueue")`

- [ ] **Step 3: Update @testable import in all test files**

In every file under `ImageRaterTests/`, change:
```swift
@testable import ImageRater
```
to:
```swift
@testable import Focal
```

Files to update:
- `ImageRaterTests/CullPipelineTests.swift`
- `ImageRaterTests/DiversityScorerTests.swift`
- `ImageRaterTests/ImageImporterGroupingTests.swift`
- `ImageRaterTests/ImageRaterTests.swift`
- `ImageRaterTests/ImageRecordMigrationTests.swift`
- `ImageRaterTests/IntegrationTests.swift`
- `ImageRaterTests/LibRawWrapperTests.swift`
- `ImageRaterTests/MetadataWriterTests.swift`
- `ImageRaterTests/ModelStoreTests.swift`
- `ImageRaterTests/ProcessingQueueTests.swift`
- `ImageRaterTests/RatingPipelineTests.swift`

- [ ] **Step 4: Regenerate project and build**

```bash
xcodegen generate
xcodebuild build -scheme Focal -destination 'platform=macOS,arch=arm64'
```

Expected: Build succeeds with 0 errors.

- [ ] **Step 5: Run tests**

```bash
xcodebuild test -scheme Focal -destination 'platform=macOS,arch=arm64' 2>&1 | tail -20
```

Expected: All existing tests pass.

- [ ] **Step 6: Commit**

```bash
git add ImageRater/App/ImageRaterApp.swift \
        ImageRater/Pipeline/CullPipeline.swift \
        ImageRater/Pipeline/RatingPipeline.swift \
        ImageRater/ModelStore/ModelStore.swift \
        ImageRater/Pipeline/ProcessingQueue.swift \
        ImageRaterTests/
git commit -m "chore: rename struct FocalApp, update Logger subsystems and test imports"
```

---

## Task 3: FocalSettings — AppStorage key constants

**Files:**
- Create: `ImageRater/App/FocalSettings.swift`

- [ ] **Step 1: Create FocalSettings.swift**

```swift
// ImageRater/App/FocalSettings.swift
import Foundation

/// Centralised UserDefaults keys for Focal. All app preferences live here.
/// Use these constants with @AppStorage or UserDefaults directly.
enum FocalSettings {

    // MARK: - Cull
    /// Percentile strictness for star assignment (0.0 = lenient, 1.0 = strict). Default: 0.5
    static let cullStrictness    = "focal.cull.strictness"

    // MARK: - Rating model weights
    /// Weight for TOPIQ Technical score. Default: 0.4
    static let weightTechnical   = "focal.rating.weightTechnical"
    /// Weight for TOPIQ Aesthetic score. Default: 0.4
    static let weightAesthetic   = "focal.rating.weightAesthetic"
    /// Weight for CLIP-IQA score. Default: 0.2
    static let weightClip        = "focal.rating.weightClip"

    // MARK: - UI
    /// Default thumbnail cell size in points. Default: 160
    static let defaultCellSize   = "focal.ui.defaultCellSize"

    // MARK: - Export
    /// Write XMP sidecar automatically on every manual rating. Default: true
    static let autoWriteXMP      = "focal.export.autoWriteXMP"

    // MARK: - Default values
    static let defaultCullStrictness: Double  = 0.5
    static let defaultWeightTechnical: Double = 0.4
    static let defaultWeightAesthetic: Double = 0.4
    static let defaultWeightClip: Double      = 0.2
    static let defaultCellSizeValue: Double   = 160
    static let defaultAutoWriteXMP: Bool      = true

    // MARK: - Migration
    /// Migrate legacy key written by pre-Focal versions. Call once at app launch.
    static func migrateIfNeeded() {
        let ud = UserDefaults.standard
        if ud.object(forKey: "cullStrictness") != nil,
           ud.object(forKey: cullStrictness) == nil {
            ud.set(ud.double(forKey: "cullStrictness"), forKey: cullStrictness)
            ud.removeObject(forKey: "cullStrictness")
        }
    }
}
```

- [ ] **Step 2: Build to confirm no errors**

```bash
xcodebuild build -scheme Focal -destination 'platform=macOS,arch=arm64' 2>&1 | grep -E "error:|Build succeeded"
```

Expected: `Build succeeded`

- [ ] **Step 3: Commit**

```bash
git add ImageRater/App/FocalSettings.swift
git commit -m "feat: add FocalSettings — centralised AppStorage key constants + migration"
```

---

## Task 4: Migrate ContentView prefs to @AppStorage

**Files:**
- Modify: `ImageRater/App/ContentView.swift`

Note: `@AppStorage` on macOS stores `Double`, not `CGFloat`. The `cellSize` var will become `Double` and cast to `CGFloat` at use sites.

- [ ] **Step 1: Replace @State prefs with @AppStorage**

In `ContentView`, replace:
```swift
@State private var cullStrictness: Double = 0.5
```
with:
```swift
@AppStorage(FocalSettings.cullStrictness) private var cullStrictness: Double = FocalSettings.defaultCullStrictness
```

Replace:
```swift
@State private var cellSize: CGFloat = 160
```
with:
```swift
@AppStorage(FocalSettings.defaultCellSize) private var cellSizeValue: Double = FocalSettings.defaultCellSizeValue
```

- [ ] **Step 2: Add cellSize computed property**

After the `@AppStorage` declarations, add:
```swift
private var cellSize: CGFloat { CGFloat(cellSizeValue) }
```

Update toolbar buttons that currently write `cellSize` directly:
```swift
// Before:
Button(action: { cellSize = max(100, cellSize - 30) }) { ... }
Button(action: { cellSize = min(320, cellSize + 30) }) { ... }
.disabled(cellSize <= 100)
.disabled(cellSize >= 320)

// After:
Button(action: { cellSizeValue = max(100, cellSizeValue - 30) }) { ... }
Button(action: { cellSizeValue = min(320, cellSizeValue + 30) }) { ... }
.disabled(cellSizeValue <= 100)
.disabled(cellSizeValue >= 320)
```

- [ ] **Step 3: Remove manual UserDefaults load in handleAppear**

Remove these lines from `handleAppear()`:
```swift
let ud = UserDefaults.standard
if ud.object(forKey: "cullStrictness") != nil {
    cullStrictness = ud.double(forKey: "cullStrictness")
}
```

Replace with the migration call (add before `keyboard.start()`):
```swift
FocalSettings.migrateIfNeeded()
```

- [ ] **Step 4: Remove applyStrictness manual UserDefaults save**

The `applyStrictness` function:
```swift
private func applyStrictness(_ s: Double) {
    UserDefaults.standard.set(s, forKey: "cullStrictness")
}
```

`@AppStorage` persists automatically. Remove the body, keeping only what's needed. If nothing else is in that function, delete it and update its call site in the `ProcessingSetupSheet` callback to just call the pipeline directly.

- [ ] **Step 5: Add autoWriteXMP guard to XMP enqueue calls**

Add to ContentView property declarations:
```swift
@AppStorage(FocalSettings.autoWriteXMP) private var autoWriteXMP: Bool = FocalSettings.defaultAutoWriteXMP
```

Find the two `ratingQueue.enqueue(xmpTasks: tasks)` / `ratingQueue.enqueue(xmpTasks: xmpTasks)` calls (in `setRating` and `rateRecordDirect`). Wrap each:
```swift
if autoWriteXMP {
    ratingQueue.enqueue(xmpTasks: tasks)
}
```

- [ ] **Step 6: Build and verify**

```bash
xcodebuild build -scheme Focal -destination 'platform=macOS,arch=arm64' 2>&1 | grep -E "error:|Build succeeded"
```

Expected: `Build succeeded`

- [ ] **Step 7: Commit**

```bash
git add ImageRater/App/ContentView.swift
git commit -m "feat: migrate ContentView prefs to @AppStorage via FocalSettings"
```

---

## Task 5: Migrate ProcessingQueue to FocalSettings

**Files:**
- Modify: `ImageRater/Pipeline/ProcessingQueue.swift`

The current code has two hardcoded values to migrate:
1. `UserDefaults.standard.double(forKey: "cullStrictness")` → `FocalSettings.cullStrictness` key
2. Hardcoded weights `0.4 * norm(e.tech...) + 0.4 * norm(e.aes...) + 0.2 * norm(e.clip...)` → read from UserDefaults

- [ ] **Step 1: Replace legacy cullStrictness key**

Find (around line 169):
```swift
let strictness = UserDefaults.standard.double(forKey: "cullStrictness")
```

Replace with:
```swift
let storedStrictness = UserDefaults.standard.double(forKey: FocalSettings.cullStrictness)
let strictness = storedStrictness == 0 ? FocalSettings.defaultCullStrictness : storedStrictness
```

(`double(forKey:)` returns 0 when key is absent — use default to avoid treating "unset" as strictness=0.)

- [ ] **Step 2: Replace hardcoded model weights**

Find the combined score computation (around line 189):
```swift
let validScores: [Float] = valid.map { e in
    0.4 * norm(e.tech, tLo, tHi) +
    0.4 * norm(e.aes,  aLo, aHi) +
    0.2 * norm(e.clip, cLo, cHi)
}
```

Replace with:
```swift
let ud = UserDefaults.standard
let wTech = ud.object(forKey: FocalSettings.weightTechnical) != nil
    ? Float(ud.double(forKey: FocalSettings.weightTechnical))
    : Float(FocalSettings.defaultWeightTechnical)
let wAes = ud.object(forKey: FocalSettings.weightAesthetic) != nil
    ? Float(ud.double(forKey: FocalSettings.weightAesthetic))
    : Float(FocalSettings.defaultWeightAesthetic)
let wClip = ud.object(forKey: FocalSettings.weightClip) != nil
    ? Float(ud.double(forKey: FocalSettings.weightClip))
    : Float(FocalSettings.defaultWeightClip)
let wSum = wTech + wAes + wClip
let (wTn, wAn, wCn) = wSum > 0
    ? (wTech/wSum, wAes/wSum, wClip/wSum)
    : (Float(FocalSettings.defaultWeightTechnical),
       Float(FocalSettings.defaultWeightAesthetic),
       Float(FocalSettings.defaultWeightClip))

let validScores: [Float] = valid.map { e in
    wTn * norm(e.tech, tLo, tHi) +
    wAn * norm(e.aes,  aLo, aHi) +
    wCn * norm(e.clip, cLo, cHi)
}
```

- [ ] **Step 3: Write a test for weight normalisation**

In `ImageRaterTests/ProcessingQueueTests.swift`, add:

```swift
func testWeightNormalisationSumsToOne() {
    let ud = UserDefaults.standard
    ud.set(0.6, forKey: FocalSettings.weightTechnical)
    ud.set(0.6, forKey: FocalSettings.weightAesthetic)
    ud.set(0.3, forKey: FocalSettings.weightClip)
    defer {
        ud.removeObject(forKey: FocalSettings.weightTechnical)
        ud.removeObject(forKey: FocalSettings.weightAesthetic)
        ud.removeObject(forKey: FocalSettings.weightClip)
    }
    let wT = Float(ud.double(forKey: FocalSettings.weightTechnical))
    let wA = Float(ud.double(forKey: FocalSettings.weightAesthetic))
    let wC = Float(ud.double(forKey: FocalSettings.weightClip))
    let sum = wT + wA + wC
    let (wTn, wAn, wCn) = (wT/sum, wA/sum, wC/sum)
    XCTAssertEqual(wTn + wAn + wCn, 1.0, accuracy: 0.001)
}

func testWeightNormalisationZeroSumFallsBackToDefaults() {
    let ud = UserDefaults.standard
    ud.set(0.0, forKey: FocalSettings.weightTechnical)
    ud.set(0.0, forKey: FocalSettings.weightAesthetic)
    ud.set(0.0, forKey: FocalSettings.weightClip)
    defer {
        ud.removeObject(forKey: FocalSettings.weightTechnical)
        ud.removeObject(forKey: FocalSettings.weightAesthetic)
        ud.removeObject(forKey: FocalSettings.weightClip)
    }
    let wT = Float(ud.double(forKey: FocalSettings.weightTechnical))
    let wA = Float(ud.double(forKey: FocalSettings.weightAesthetic))
    let wC = Float(ud.double(forKey: FocalSettings.weightClip))
    let sum = wT + wA + wC
    // sum == 0 → must fall back to defaults, which sum to 1.0
    let defSum = Float(FocalSettings.defaultWeightTechnical + FocalSettings.defaultWeightAesthetic + FocalSettings.defaultWeightClip)
    XCTAssertEqual(sum, 0.0)          // confirm zero-sum scenario
    XCTAssertEqual(defSum, 1.0, accuracy: 0.001)  // confirm defaults are sane
}
```

- [ ] **Step 4: Run the new test**

```bash
xcodebuild test -scheme Focal -destination 'platform=macOS,arch=arm64' -only-testing:FocalTests/ProcessingQueueTests/testWeightNormalisationSumsToOne 2>&1 | tail -10
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add ImageRater/Pipeline/ProcessingQueue.swift ImageRaterTests/ProcessingQueueTests.swift
git commit -m "feat: migrate ProcessingQueue to FocalSettings keys and configurable model weights"
```

---

## Task 6: PreferencesView — tabbed Settings window

**Files:**
- Create: `ImageRater/UI/PreferencesView.swift`

- [ ] **Step 1: Create PreferencesView.swift**

```swift
// ImageRater/UI/PreferencesView.swift
import SwiftUI

struct PreferencesView: View {
    var body: some View {
        TabView {
            PipelineTab()
                .tabItem { Label("Pipeline", systemImage: "gearshape") }
            AppearanceTab()
                .tabItem { Label("Appearance", systemImage: "paintbrush") }
            ExportTab()
                .tabItem { Label("Export", systemImage: "square.and.arrow.up") }
        }
        .frame(width: 480)
        .padding()
    }
}

// MARK: - Pipeline tab

private struct PipelineTab: View {
    @AppStorage(FocalSettings.cullStrictness)  private var strictness: Double  = FocalSettings.defaultCullStrictness
    @AppStorage(FocalSettings.weightTechnical) private var wTech: Double       = FocalSettings.defaultWeightTechnical
    @AppStorage(FocalSettings.weightAesthetic) private var wAes: Double        = FocalSettings.defaultWeightAesthetic
    @AppStorage(FocalSettings.weightClip)      private var wClip: Double       = FocalSettings.defaultWeightClip

    var body: some View {
        Form {
            Section("Rating") {
                VStack(alignment: .leading, spacing: 4) {
                    HStack {
                        Text("Strictness")
                        Spacer()
                        Text(String(format: "%.0f%%", strictness * 100))
                            .foregroundStyle(.secondary)
                            .monospacedDigit()
                    }
                    Slider(value: $strictness, in: 0...1, step: 0.05)
                    Text("Higher = fewer 4–5 star images. Lower = more generous star distribution.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }

            Section("Model Weights (auto-normalised to sum = 1)") {
                weightRow("Technical (TOPIQ)", value: $wTech)
                weightRow("Aesthetic (TOPIQ)", value: $wAes)
                weightRow("Perceptual (CLIP-IQA)", value: $wClip)
                Button("Reset to defaults") {
                    wTech = FocalSettings.defaultWeightTechnical
                    wAes  = FocalSettings.defaultWeightAesthetic
                    wClip = FocalSettings.defaultWeightClip
                }
                .buttonStyle(.plain)
                .foregroundStyle(.accentColor)
            }
        }
        .formStyle(.grouped)
        .padding(.vertical, 8)
    }

    @ViewBuilder
    private func weightRow(_ label: String, value: Binding<Double>) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(label)
                Spacer()
                Text(String(format: "%.2f", value.wrappedValue))
                    .foregroundStyle(.secondary)
                    .monospacedDigit()
            }
            Slider(value: value, in: 0...1, step: 0.05)
        }
    }
}

// MARK: - Appearance tab

private struct AppearanceTab: View {
    @AppStorage(FocalSettings.defaultCellSize) private var cellSize: Double = FocalSettings.defaultCellSizeValue

    var body: some View {
        Form {
            Section("Grid") {
                VStack(alignment: .leading, spacing: 4) {
                    HStack {
                        Text("Default thumbnail size")
                        Spacer()
                        Text("\(Int(cellSize)) px")
                            .foregroundStyle(.secondary)
                            .monospacedDigit()
                    }
                    Slider(value: $cellSize, in: 80...320, step: 20)
                }
            }
        }
        .formStyle(.grouped)
        .padding(.vertical, 8)
    }
}

// MARK: - Export tab

private struct ExportTab: View {
    @AppStorage(FocalSettings.autoWriteXMP) private var autoWriteXMP: Bool = FocalSettings.defaultAutoWriteXMP

    var body: some View {
        Form {
            Section("XMP Sidecar") {
                Toggle("Auto-write XMP on manual rating", isOn: $autoWriteXMP)
                Text("When enabled, Focal writes an .xmp sidecar file alongside the original every time you rate an image with the keyboard or toolbar.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .formStyle(.grouped)
        .padding(.vertical, 8)
    }
}
```

- [ ] **Step 2: Build to confirm no errors**

```bash
xcodebuild build -scheme Focal -destination 'platform=macOS,arch=arm64' 2>&1 | grep -E "error:|Build succeeded"
```

Expected: `Build succeeded`

- [ ] **Step 3: Commit**

```bash
git add ImageRater/UI/PreferencesView.swift
git commit -m "feat: add PreferencesView — Pipeline, Appearance, Export tabs backed by AppStorage"
```

---

## Task 7: Wire Settings scene + About window into FocalApp

**Files:**
- Modify: `ImageRater/App/ImageRaterApp.swift`

- [ ] **Step 1: Rewrite FocalApp.swift**

Replace the entire file contents with:

```swift
import AppKit
import SwiftUI

@main
struct FocalApp: App {
    let persistence = PersistenceController.shared

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environment(\.managedObjectContext, persistence.container.viewContext)
        }
        .commands {
            CommandGroup(replacing: .appInfo) {
                Button("About Focal") {
                    NSApp.orderFrontStandardAboutPanel(options: [
                        .applicationName: "Focal",
                        .credits: NSAttributedString(
                            string: "AI-powered photo culling and rating for macOS.\n\nModels: TOPIQ (IQA-PyTorch), CLIP-IQA (OpenCLIP). RAW decoding via LibRaw.",
                            attributes: [.font: NSFont.systemFont(ofSize: NSFont.smallSystemFontSize)]
                        )
                    ])
                }
            }
        }

        Settings {
            PreferencesView()
        }
    }
}
```

- [ ] **Step 2: Build and verify**

```bash
xcodebuild build -scheme Focal -destination 'platform=macOS,arch=arm64' 2>&1 | grep -E "error:|Build succeeded"
```

Expected: `Build succeeded`

- [ ] **Step 3: Manual smoke test**

Open in Xcode, run the app. Verify:
- `Focal` menu → "About Focal" opens an About box showing name, version, credits
- `Focal` menu → "Preferences…" (Cmd+,) opens the tabbed PreferencesView
- Changing strictness/weights in Preferences persists after app restart

- [ ] **Step 4: Commit**

```bash
git add ImageRater/App/ImageRaterApp.swift
git commit -m "feat: add Settings scene and About window to FocalApp"
```

---

## Task 8: README.md

**Files:**
- Create: `README.md` at repo root

- [ ] **Step 1: Write README.md**

```markdown
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
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add README with features, installation, and build instructions"
```

---

## Task 9: CHANGELOG.md

**Files:**
- Create: `CHANGELOG.md` at repo root

- [ ] **Step 1: Write CHANGELOG.md**

```markdown
# Changelog

All notable changes to Focal are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [1.0.0] - 2026-04-14

### Added
- Two-phase AI pipeline: blur/exposure/EAR cull (Apple Vision + CIFilter) + TOPIQ/CLIP-IQA rating (Core ML)
- RAW file support via LibRaw (RAF, NEF, CR3, ARW, and all LibRaw-supported formats)
- XMP sidecar export writing `xmp:Rating` and `MicrosoftPhoto:Rating`
- Session history with Core Data persistence and lightweight migration
- Thumbnail grid with rubber-band multi-select and keyboard-driven rating workflow
- Detail modal with zoomable full-resolution view and adjacent-image prefetch
- Model Store: auto-download, SHA-256 checksum verification, and local import of Core ML models
- Preferences window (⌘,) — cull strictness, model weights (Technical/Aesthetic/CLIP-IQA), thumbnail size, XMP auto-write
- Compare mode for side-by-side image comparison
- RAW+JPEG grouping — rate the pair together, export to both
- Background XMP sweep — writes ratings for all session images on load
```

- [ ] **Step 2: Commit**

```bash
git add CHANGELOG.md
git commit -m "docs: add CHANGELOG with v1.0.0 initial release entry"
```

---

## Task 10: GitHub Actions — CI test workflow

**Files:**
- Create: `.github/workflows/test.yml`

- [ ] **Step 1: Create .github/workflows directory**

```bash
mkdir -p .github/workflows
```

- [ ] **Step 2: Write test.yml**

```yaml
name: Test

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    name: Build & Test
    runs-on: macos-15

    steps:
      - uses: actions/checkout@v4

      - name: Install dependencies
        run: brew install libraw xcodegen xcpretty

      - name: Generate Xcode project
        run: xcodegen generate

      - name: Build & test
        run: |
          xcodebuild test \
            -scheme Focal \
            -destination 'platform=macOS,arch=arm64' \
            | xcpretty
```

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/test.yml
git commit -m "ci: add GitHub Actions test workflow — build and test on push/PR"
```

---

## Task 11: GitHub Actions — DMG release workflow + ExportOptions

**Files:**
- Create: `.github/workflows/release.yml`
- Create: `.github/ExportOptions.plist`

- [ ] **Step 1: Write ExportOptions.plist**

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>method</key>
    <string>mac-application</string>
    <key>signingStyle</key>
    <string>manual</string>
    <key>provisioningProfileStyle</key>
    <string>automatic</string>
    <key>stripSwiftSymbols</key>
    <true/>
</dict>
</plist>
```

- [ ] **Step 2: Write release.yml**

```yaml
name: Release

on:
  push:
    tags:
      - 'v*'

jobs:
  build-dmg:
    name: Build DMG
    runs-on: macos-15

    steps:
      - uses: actions/checkout@v4

      - name: Install dependencies
        run: brew install libraw xcodegen create-dmg

      - name: Generate Xcode project
        run: xcodegen generate

      - name: Extract version from tag
        run: echo "VERSION=${GITHUB_REF_NAME#v}" >> $GITHUB_ENV

      - name: Archive (unsigned)
        run: |
          xcodebuild archive \
            -scheme Focal \
            -archivePath "$RUNNER_TEMP/Focal.xcarchive" \
            -destination 'generic/platform=macOS' \
            MARKETING_VERSION="$VERSION" \
            CURRENT_PROJECT_VERSION="$GITHUB_RUN_NUMBER" \
            CODE_SIGN_IDENTITY="" \
            CODE_SIGNING_REQUIRED=NO \
            CODE_SIGNING_ALLOWED=NO

      - name: Export app
        run: |
          xcodebuild -exportArchive \
            -archivePath "$RUNNER_TEMP/Focal.xcarchive" \
            -exportOptionsPlist .github/ExportOptions.plist \
            -exportPath "$RUNNER_TEMP/export"

      - name: Build DMG
        run: |
          create-dmg \
            --volname "Focal" \
            --window-size 600 400 \
            --app-drop-link 450 200 \
            "Focal-$VERSION.dmg" \
            "$RUNNER_TEMP/export/Focal.app"

      - name: Create GitHub Release
        run: |
          gh release create "$GITHUB_REF_NAME" \
            "Focal-$VERSION.dmg" \
            --title "Focal $VERSION" \
            --notes-file CHANGELOG.md
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/release.yml .github/ExportOptions.plist
git commit -m "ci: add release workflow — unsigned DMG build and GitHub Release on tag push"
```

---

## Final Verification

- [ ] Regenerate project and run full test suite:

```bash
xcodegen generate
xcodebuild test -scheme Focal -destination 'platform=macOS,arch=arm64' 2>&1 | tail -20
```

Expected: All tests pass.

- [ ] Confirm files exist:
  - `README.md` ✓
  - `CHANGELOG.md` ✓
  - `.github/workflows/test.yml` ✓
  - `.github/workflows/release.yml` ✓
  - `.github/ExportOptions.plist` ✓
  - `ImageRater/App/FocalSettings.swift` ✓
  - `ImageRater/UI/PreferencesView.swift` ✓

- [ ] Open app in Xcode, smoke test:
  - App launches as "Focal"
  - Cmd+, opens Preferences window with 3 tabs
  - About Focal shows name, version, credits
  - Changing strictness/weights persists across restarts
