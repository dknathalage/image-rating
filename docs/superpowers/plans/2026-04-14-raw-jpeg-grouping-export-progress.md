# RAW+JPEG Grouping & Export Progress Bar Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Group JPEG+RAW file pairs into a single grid entry scored from the RAW, and add an async progress bar to the XMP export action.

**Architecture:** Three new optional/defaulted fields on `ImageRecord` (`groupID`, `isGroupPrimary`, `scoringFilePath`) drive pairing. Import groups files by base name, pipeline skips companions and decodes RAW for scoring, export writes metadata to all files in a group. Export becomes an `@MainActor async` Task with live `ProcessingStatusBar`-style progress.

**Tech Stack:** Swift, SwiftUI, CoreData (lightweight migration), ImageIO, XCTest

**Spec:** `docs/superpowers/specs/2026-04-14-raw-jpeg-grouping-export-progress-design.md`

---

## File Map

| File | Change |
|---|---|
| `ImageRater/CoreData/ImageRater.xcdatamodeld/ImageRater 3.xcdatamodel/contents` | **Create** — v3 schema with 3 new fields on ImageRecord |
| `ImageRater/CoreData/ImageRater.xcdatamodeld/.xccurrentversion` | **Modify** — point to v3 |
| `ImageRater/Export/MetadataWriter.swift` | **Modify** — remove `private` from `rawExtensions` |
| `ImageRater/Import/ImageImporter.swift` | **Modify** — add `groupByBaseName`, apply pairing before record creation |
| `ImageRater/Pipeline/ProcessingQueue.swift` | **Modify** — skip companions in Pass 1, use `scoringFilePath`, propagate cull fields in Pass 1, propagate diversity fields after Pass 2, write companion sidecars |
| `ImageRater/App/ContentView.swift` | **Modify** — filter `filteredImages` to primaries, async export with companion writes and progress bar |
| `ImageRater/UI/Components/ThumbnailCell.swift` | **Modify** — add RAW pill badge |
| `ImageRaterTests/ImageRecordMigrationTests.swift` | **Modify** — add v3 field assertions |
| `ImageRaterTests/ImageImporterGroupingTests.swift` | **Create** — unit tests for `groupByBaseName` and pairing logic |

---

## Task 1: CoreData Schema v3 — New Migration Version

**Files:**
- Create: `ImageRater/CoreData/ImageRater.xcdatamodeld/ImageRater 3.xcdatamodel/contents`
- Modify: `ImageRater/CoreData/ImageRater.xcdatamodeld/.xccurrentversion`
- Modify: `ImageRaterTests/ImageRecordMigrationTests.swift`

- [ ] **Step 1: Create the v3 model directory and contents file**

Create directory `ImageRater/CoreData/ImageRater.xcdatamodeld/ImageRater 3.xcdatamodel/` and write `contents`:

```xml
<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<model type="com.apple.IDECoreDataModeler.DataModel" documentVersion="1.0" lastSavedToolsVersion="22222" systemVersion="23A344" minimumToolsVersion="Automatic" sourceLanguage="Swift" userDefinedModelVersionIdentifier="3">
    <entity name="Session" representedClassName="Session" syncable="YES" codeGenerationType="class">
        <attribute name="id" optional="NO" attributeType="UUID" usesScalarValueType="NO"/>
        <attribute name="createdAt" optional="NO" attributeType="Date" usesScalarValueType="NO"/>
        <attribute name="folderPath" optional="NO" attributeType="String"/>
        <relationship name="images" optional="YES" toMany="YES" deletionRule="Cascade" destinationEntity="ImageRecord" inverseName="session" inverseEntity="ImageRecord"/>
    </entity>
    <entity name="ImageRecord" representedClassName="ImageRecord" syncable="YES" codeGenerationType="class">
        <attribute name="id" optional="NO" attributeType="UUID" usesScalarValueType="NO"/>
        <attribute name="filePath" optional="NO" attributeType="String"/>
        <attribute name="thumbHash" optional="YES" attributeType="String"/>
        <attribute name="processState" optional="NO" attributeType="String" defaultValueString="pending"/>
        <attribute name="decodeError" optional="NO" attributeType="Boolean" defaultValueString="NO" usesScalarValueType="YES"/>
        <attribute name="cullRejected" optional="NO" attributeType="Boolean" defaultValueString="NO" usesScalarValueType="YES"/>
        <attribute name="cullReason" optional="YES" attributeType="String"/>
        <attribute name="ratingStars" optional="YES" attributeType="Integer 16" usesScalarValueType="NO"/>
        <attribute name="clipScore" optional="YES" attributeType="Float" usesScalarValueType="NO"/>
        <attribute name="aestheticScore" optional="YES" attributeType="Float" usesScalarValueType="NO"/>
        <attribute name="userOverride" optional="YES" attributeType="Integer 16" usesScalarValueType="NO"/>
        <attribute name="topiqTechnicalScore" attributeType="Float" defaultValueString="0.0" usesScalarValueType="YES"/>
        <attribute name="topiqAestheticScore" attributeType="Float" defaultValueString="0.0" usesScalarValueType="YES"/>
        <attribute name="clipIQAScore" attributeType="Float" defaultValueString="0.0" usesScalarValueType="YES"/>
        <attribute name="combinedQualityScore" attributeType="Float" defaultValueString="0.0" usesScalarValueType="YES"/>
        <attribute name="finalScore" attributeType="Float" defaultValueString="0.0" usesScalarValueType="YES"/>
        <attribute name="diversityFactor" attributeType="Float" defaultValueString="1.0" usesScalarValueType="YES"/>
        <attribute name="clipEmbedding" optional="YES" attributeType="Binary" allowsExternalBinaryDataStorage="YES"/>
        <attribute name="clusterID" attributeType="Integer 32" defaultValueString="-1" usesScalarValueType="YES"/>
        <attribute name="clusterRank" attributeType="Integer 32" defaultValueString="0" usesScalarValueType="YES"/>
        <attribute name="blurScore" attributeType="Float" defaultValueString="0.0" usesScalarValueType="YES"/>
        <attribute name="exposureScore" attributeType="Float" defaultValueString="0.0" usesScalarValueType="YES"/>
        <attribute name="groupID" optional="YES" attributeType="String"/>
        <attribute name="isGroupPrimary" optional="NO" attributeType="Boolean" defaultValueString="YES" usesScalarValueType="YES"/>
        <attribute name="scoringFilePath" optional="YES" attributeType="String"/>
        <relationship name="session" optional="NO" maxCount="1" deletionRule="Nullify" destinationEntity="Session" inverseName="images" inverseEntity="Session"/>
    </entity>
    <entity name="ModelConfig" representedClassName="ModelConfig" syncable="YES" codeGenerationType="class">
        <attribute name="id" optional="NO" attributeType="UUID" usesScalarValueType="NO"/>
        <attribute name="modelName" optional="NO" attributeType="String"/>
        <attribute name="clipWeight" optional="NO" attributeType="Float" defaultValueString="0.5" usesScalarValueType="YES"/>
        <attribute name="aestheticWeight" optional="NO" attributeType="Float" defaultValueString="0.5" usesScalarValueType="YES"/>
        <attribute name="blurThreshold" optional="NO" attributeType="Float" defaultValueString="50.0" usesScalarValueType="YES"/>
        <attribute name="earThreshold" optional="NO" attributeType="Float" defaultValueString="0.2" usesScalarValueType="YES"/>
        <attribute name="exposureLeniency" optional="NO" attributeType="Float" defaultValueString="0.9" usesScalarValueType="YES"/>
    </entity>
</model>
```

- [ ] **Step 2: Update `.xccurrentversion` to point to v3**

Replace contents of `ImageRater/CoreData/ImageRater.xcdatamodeld/.xccurrentversion`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
	<key>_XCCurrentVersionName</key>
	<string>ImageRater 3.xcdatamodel</string>
</dict>
</plist>
```

- [ ] **Step 3: Add v3 field assertions to migration test**

In `ImageRaterTests/ImageRecordMigrationTests.swift`, add a new test after `testNewFieldsReadWrite`:

```swift
func testV3GroupingFieldsReadWrite() throws {
    let ctx = makeInMemoryContext()

    let session = Session(context: ctx)
    session.id = UUID(); session.createdAt = Date(); session.folderPath = "/tmp"

    let jpeg = ImageRecord(context: ctx)
    jpeg.id = UUID(); jpeg.filePath = "/tmp/a.jpg"; jpeg.processState = "pending"; jpeg.session = session
    jpeg.groupID = "test-group-1"
    jpeg.isGroupPrimary = true
    jpeg.scoringFilePath = "/tmp/a.cr2"

    let raw = ImageRecord(context: ctx)
    raw.id = UUID(); raw.filePath = "/tmp/a.cr2"; raw.processState = "pending"; raw.session = session
    raw.groupID = "test-group-1"
    raw.isGroupPrimary = false

    let solo = ImageRecord(context: ctx)
    solo.id = UUID(); solo.filePath = "/tmp/b.jpg"; solo.processState = "pending"; solo.session = session
    // groupID nil, isGroupPrimary defaults to true

    try ctx.save()

    ctx.refresh(jpeg, mergeChanges: false)
    ctx.refresh(raw, mergeChanges: false)
    ctx.refresh(solo, mergeChanges: false)

    XCTAssertEqual(jpeg.groupID, "test-group-1")
    XCTAssertTrue(jpeg.isGroupPrimary)
    XCTAssertEqual(jpeg.scoringFilePath, "/tmp/a.cr2")

    XCTAssertEqual(raw.groupID, "test-group-1")
    XCTAssertFalse(raw.isGroupPrimary)
    XCTAssertNil(raw.scoringFilePath)

    XCTAssertNil(solo.groupID)
    XCTAssertTrue(solo.isGroupPrimary)    // default YES
    XCTAssertNil(solo.scoringFilePath)
}
```

- [ ] **Step 4: Run migration test**

Run: `xcodebuild test -scheme ImageRater -only-testing ImageRaterTests/ImageRecordMigrationTests/testV3GroupingFieldsReadWrite`

Expected: PASS. If FAIL with "keyPath groupID not found", the model version wasn't picked up — verify `.xccurrentversion` points to `ImageRater 3.xcdatamodel`.

- [ ] **Step 5: Commit**

```bash
git add "ImageRater/CoreData/ImageRater.xcdatamodeld/ImageRater 3.xcdatamodel/contents" \
        "ImageRater/CoreData/ImageRater.xcdatamodeld/.xccurrentversion" \
        "ImageRaterTests/ImageRecordMigrationTests.swift"
git commit -m "feat(coredata): add groupID, isGroupPrimary, scoringFilePath to ImageRecord (v3)"
```

---

## Task 2: Expose `rawExtensions` in MetadataWriter

**Files:**
- Modify: `ImageRater/Export/MetadataWriter.swift`

- [ ] **Step 1: Remove `private` from `rawExtensions`**

In `MetadataWriter.swift`, change line:
```swift
private let rawExtensions: Set<String> = [
```
to:
```swift
let rawExtensions: Set<String> = [
```

(Keep it at file scope — `internal` by default, accessible from `ImageImporter`.)

- [ ] **Step 2: Verify existing MetadataWriter tests still pass**

Run: `xcodebuild test -scheme ImageRater -only-testing ImageRaterTests/MetadataWriterTests`

Expected: all 4 tests PASS.

- [ ] **Step 3: Commit**

```bash
git add ImageRater/Export/MetadataWriter.swift
git commit -m "refactor(metadata): expose rawExtensions as internal for ImageImporter use"
```

---

## Task 3: ImageImporter — Grouping Logic

**Files:**
- Modify: `ImageRater/Import/ImageImporter.swift`
- Create: `ImageRaterTests/ImageImporterGroupingTests.swift`

- [ ] **Step 1: Write failing tests for groupByBaseName**

Create `ImageRaterTests/ImageImporterGroupingTests.swift`:

```swift
import XCTest
@testable import ImageRater

final class ImageImporterGroupingTests: XCTestCase {

    // MARK: groupByBaseName

    func testSingleJpegAndRawPaired() {
        let urls = makeURLs(["IMG_001.jpg", "IMG_001.cr2"])
        let groups = ImageImporter.groupByBaseName(urls)
        XCTAssertEqual(groups.count, 1)
        XCTAssertEqual(groups[0].count, 2)
    }

    func testJpegWithMultipleRawsPaired() {
        let urls = makeURLs(["IMG_001.jpg", "IMG_001.cr2", "IMG_001.dng"])
        let groups = ImageImporter.groupByBaseName(urls)
        XCTAssertEqual(groups.count, 1)
        XCTAssertEqual(groups[0].count, 3)
    }

    func testOrphanRawIsOwnGroup() {
        let urls = makeURLs(["IMG_001.cr2"])
        let groups = ImageImporter.groupByBaseName(urls)
        XCTAssertEqual(groups.count, 1)
        XCTAssertEqual(groups[0].count, 1)
    }

    func testJpegOnlyIsOwnGroup() {
        let urls = makeURLs(["IMG_001.jpg"])
        let groups = ImageImporter.groupByBaseName(urls)
        XCTAssertEqual(groups.count, 1)
        XCTAssertEqual(groups[0].count, 1)
    }

    func testMultipleJpegsWithRawsAreAllSingles() {
        // Two JPEGs with same base name (ambiguous) — all singles
        let urls = makeURLs(["IMG_001.jpg", "IMG_001.JPG", "IMG_001.cr2"])
        let groups = ImageImporter.groupByBaseName(urls)
        // Each file is its own group (no pairing)
        XCTAssertEqual(groups.count, 3)
        XCTAssertTrue(groups.allSatisfy { $0.count == 1 })
    }

    func testMultipleRawsNoJpegAllSingles() {
        let urls = makeURLs(["IMG_001.cr2", "IMG_001.dng"])
        let groups = ImageImporter.groupByBaseName(urls)
        XCTAssertEqual(groups.count, 2)
        XCTAssertTrue(groups.allSatisfy { $0.count == 1 })
    }

    func testMixedPairedAndSingles() {
        let urls = makeURLs(["A.jpg", "A.cr2", "B.jpg", "C.nef"])
        let groups = ImageImporter.groupByBaseName(urls)
        // A: paired (2), B: single (1), C: single (1)
        XCTAssertEqual(groups.count, 3)
        let groupSizes = groups.map(\.count).sorted()
        XCTAssertEqual(groupSizes, [1, 1, 2])
    }

    // MARK: pairing fields on records

    func testPrimaryRecordFieldsForPair() throws {
        let ctx = makeInMemoryContext()
        let tmpDir = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmpDir) }

        // Create stub files
        let jpegURL = tmpDir.appendingPathComponent("IMG_001.jpg")
        let rawURL  = tmpDir.appendingPathComponent("IMG_001.cr2")
        FileManager.default.createFile(atPath: jpegURL.path, contents: Data())
        FileManager.default.createFile(atPath: rawURL.path,  contents: Data())

        try ImageImporter.importFolder(tmpDir, context: ctx)

        let req = ImageRecord.fetchRequest()
        let records = try ctx.fetch(req)
        XCTAssertEqual(records.count, 2)

        let primary = records.first { $0.filePath?.hasSuffix(".jpg") == true }
        let companion = records.first { $0.filePath?.hasSuffix(".cr2") == true }

        XCTAssertNotNil(primary)
        XCTAssertNotNil(companion)
        XCTAssertTrue(primary!.isGroupPrimary)
        XCTAssertFalse(companion!.isGroupPrimary)
        XCTAssertNotNil(primary!.groupID)
        XCTAssertEqual(primary!.groupID, companion!.groupID)
        XCTAssertEqual(primary!.scoringFilePath, rawURL.path)
        XCTAssertNil(companion!.scoringFilePath)
    }

    func testOrphanRawRecordIsUnpaired() throws {
        let ctx = makeInMemoryContext()
        let tmpDir = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmpDir) }

        let rawURL = tmpDir.appendingPathComponent("IMG_001.cr2")
        FileManager.default.createFile(atPath: rawURL.path, contents: Data())

        try ImageImporter.importFolder(tmpDir, context: ctx)

        let req = ImageRecord.fetchRequest()
        let records = try ctx.fetch(req)
        XCTAssertEqual(records.count, 1)
        XCTAssertTrue(records[0].isGroupPrimary)
        XCTAssertNil(records[0].groupID)
        XCTAssertNil(records[0].scoringFilePath)
    }

    // MARK: Helpers

    private func makeURLs(_ names: [String]) -> [URL] {
        let base = URL(filePath: "/fake")
        return names.map { base.appendingPathComponent($0) }
            .sorted { $0.lastPathComponent.localizedStandardCompare($1.lastPathComponent) == .orderedAscending }
    }

    private func makeInMemoryContext() -> NSManagedObjectContext {
        let c = NSPersistentContainer(name: "ImageRater")
        let d = NSPersistentStoreDescription()
        d.type = NSInMemoryStoreType
        c.persistentStoreDescriptions = [d]
        c.loadPersistentStores { _, error in
            if let error { XCTFail("Store load failed: \(error)") }
        }
        return c.viewContext
    }
}
```

- [ ] **Step 2: Run tests to confirm they fail**

Run: `xcodebuild test -scheme ImageRater -only-testing ImageRaterTests/ImageImporterGroupingTests`

Expected: compile error or FAIL — `groupByBaseName` doesn't exist yet.

- [ ] **Step 3: Implement `groupByBaseName` and update `importFolder`**

Replace the full content of `ImageRater/Import/ImageImporter.swift`:

```swift
import CoreData
import Foundation

enum ImageImporter {

    static let supportedExtensions: Set<String> = LibRawWrapper.supportedExtensions
        .union(["jpg", "jpeg", "png", "tiff", "tif", "heic", "heif"])

    /// Scan folder, create Session + ImageRecord entities in context, return Session objectID.
    /// Must be called on `context`'s queue (wrap in `context.performAndWait {}` from other threads).
    @discardableResult
    static func importFolder(_ url: URL,
                              context: NSManagedObjectContext) throws -> NSManagedObjectID {
        let session = Session(context: context)
        session.id = UUID()
        session.createdAt = Date()
        session.folderPath = url.path

        let files = try scanFolder(url)
        let groups = groupByBaseName(files)

        for group in groups {
            if group.count == 1 {
                // Unpaired single file
                createRecord(for: group[0], session: session, context: context,
                             groupID: nil, isPrimary: true, scoringPath: nil)
            } else {
                // Paired: exactly one JPEG + ≥1 RAW
                let jpegURL  = group.first { !MetadataWriter.rawExtensions.contains($0.pathExtension.lowercased()) }!
                let rawURLs  = group.filter { MetadataWriter.rawExtensions.contains($0.pathExtension.lowercased()) }
                let gid      = UUID().uuidString
                // JPEG primary: scoringFilePath = first RAW (group is sorted, RAWs already sorted)
                createRecord(for: jpegURL, session: session, context: context,
                             groupID: gid, isPrimary: true, scoringPath: rawURLs[0].path)
                for rawURL in rawURLs {
                    createRecord(for: rawURL, session: session, context: context,
                                 groupID: gid, isPrimary: false, scoringPath: nil)
                }
            }
        }

        try context.save()
        return session.objectID
    }

    /// Recursively scan folder for supported image files. Returns sorted list.
    static func scanFolder(_ url: URL) throws -> [URL] {
        guard let enumerator = FileManager.default.enumerator(
            at: url,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles, .skipsPackageDescendants]
        ) else { return [] }

        return enumerator.compactMap { item -> URL? in
            guard let fileURL = item as? URL,
                  (try? fileURL.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile) == true,
                  supportedExtensions.contains(fileURL.pathExtension.lowercased())
            else { return nil }
            return fileURL
        }.sorted { $0.lastPathComponent.localizedStandardCompare($1.lastPathComponent) == .orderedAscending }
    }

    /// Group URLs by lowercased base name. Returns array of groups sorted by base name.
    /// A group is a pair only when it has exactly one JPEG/non-RAW and ≥1 RAW.
    /// Ambiguous groups (multiple JPEGs, or multiple RAWs with no JPEG) are split into singles.
    static func groupByBaseName(_ urls: [URL]) -> [[URL]] {
        // Group by lowercased base name, preserving sort order within each group
        var byBase: [String: [URL]] = [:]
        for url in urls {
            let base = url.deletingPathExtension().lastPathComponent.lowercased()
            byBase[base, default: []].append(url)
        }

        var result: [[URL]] = []
        for (_, group) in byBase.sorted(by: { $0.key < $1.key }) {
            let nonRaw = group.filter { !MetadataWriter.rawExtensions.contains($0.pathExtension.lowercased()) }
            let raw    = group.filter {  MetadataWriter.rawExtensions.contains($0.pathExtension.lowercased()) }

            if nonRaw.count == 1 && raw.count >= 1 {
                // Valid pair: one JPEG + ≥1 RAW — return as single group
                let sorted = (nonRaw + raw).sorted {
                    $0.lastPathComponent.localizedStandardCompare($1.lastPathComponent) == .orderedAscending
                }
                result.append(sorted)
            } else {
                // Ambiguous or single-type group — each file is its own entry
                for url in group {
                    result.append([url])
                }
            }
        }
        return result
    }

    // MARK: Private

    private static func createRecord(for url: URL,
                                      session: Session,
                                      context: NSManagedObjectContext,
                                      groupID: String?,
                                      isPrimary: Bool,
                                      scoringPath: String?) {
        let record = ImageRecord(context: context)
        record.id = UUID()
        record.filePath = url.path
        record.processState = "pending"
        record.decodeError = false
        record.cullRejected = false
        record.session = session
        record.groupID = groupID
        record.isGroupPrimary = isPrimary
        record.scoringFilePath = scoringPath
    }
}
```

- [ ] **Step 4: Run grouping tests**

Run: `xcodebuild test -scheme ImageRater -only-testing ImageRaterTests/ImageImporterGroupingTests`

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add ImageRater/Import/ImageImporter.swift \
        ImageRaterTests/ImageImporterGroupingTests.swift
git commit -m "feat(import): group JPEG+RAW pairs by base name, set groupID/isGroupPrimary/scoringFilePath"
```

---

## Task 4: ProcessingQueue — Pass 1 Companion Skip + Cull Propagation

**Files:**
- Modify: `ImageRater/Pipeline/ProcessingQueue.swift`

- [ ] **Step 1: Skip companions and use scoringFilePath in Pass 1**

In `ProcessingQueue.process`, the Pass 1 loop reads `filePath` and a `skip` flag. Expand the read block to also fetch `isGroupPrimary`, `groupID`, and `scoringFilePath`:

Replace the `let (filePath, skip)` block (lines ~59–67) with:

```swift
let (filePath, scoringPath, groupID, skip): (String?, String?, String?, Bool) = await context.perform { [self] in
    guard let r = try? self.context.existingObject(with: imageID) as? ImageRecord else {
        return (nil, nil, nil, true)
    }
    // Skip companion records — they inherit scores from their primary
    if !r.isGroupPrimary { return (nil, nil, nil, true) }
    let skip = r.processState == ProcessState.done
            || r.processState == ProcessState.rated
    return (r.filePath, r.scoringFilePath, r.groupID, skip)
}
if skip { continue }
```

Then update the decode line (currently `guard let path = filePath`) to use `scoringPath` when set:

```swift
let decodePath = scoringPath ?? filePath
guard let path = decodePath,
      let cgImage = LibRawWrapper.decode(url: URL(filePath: path)) else {
    decodeErrorCount += 1
    await context.perform { [self] in
        guard let r = try? self.context.existingObject(with: imageID) as? ImageRecord else { return }
        r.decodeError = true
        r.processState = ProcessState.done
        try? self.context.save()
    }
    continue
}
```

- [ ] **Step 2: Propagate cull fields to companions after Pass 1 scoring**

After the `await context.perform { ... r.processState = ProcessState.rated ... }` block, add companion cull propagation:

```swift
// Propagate cull results to companion records (same groupID, isGroupPrimary=false)
if let gid = groupID {
    await context.perform { [self] in
        guard let r = try? self.context.existingObject(with: imageID) as? ImageRecord,
              let session = r.session,
              let companions = session.images?.allObjects as? [ImageRecord] else { return }
        let targets = companions.filter { $0.groupID == gid && !$0.isGroupPrimary }
        for c in targets {
            c.cullRejected  = r.cullRejected
            c.cullReason    = r.cullReason
            c.blurScore     = r.blurScore
            c.exposureScore = r.exposureScore
        }
        try? self.context.save()
    }
}
```

- [ ] **Step 3: Also update `total` count to exclude companions**

In the Pass 1 setup, `total = imageIDs.count` counts all records including companions. Update `total` after the imageIDs fetch to count only primaries:

After `let total = imageIDs.count`, add:

```swift
let primaryTotal = await context.perform { [self] in
    imageIDs.compactMap { try? self.context.existingObject(with: $0) as? ImageRecord }
        .filter { $0.isGroupPrimary }.count
}
```

Replace **all three** `onProgress?` calls that use `total` with `primaryTotal`:
- `onProgress?(i, total, "Scoring \(i + 1) of \(total)")` → `onProgress?(i, primaryTotal, "Scoring \(i + 1) of \(primaryTotal)")`
- `onProgress?(total, total, "Ranking variety…")` → `onProgress?(primaryTotal, primaryTotal, "Ranking variety…")`
- `onProgress?(total, total, "Done")` → `onProgress?(primaryTotal, primaryTotal, "Done")`

The pre-loop call `onProgress?(0, 0, "Compiling models…")` intentionally keeps `0, 0` (models are loading before any images are counted).

- [ ] **Step 4: Run ProcessingQueue tests**

Run: `xcodebuild test -scheme ImageRater -only-testing ImageRaterTests/ProcessingQueueTests`

Expected: existing tests PASS (no regressions).

- [ ] **Step 5: Commit**

```bash
git add ImageRater/Pipeline/ProcessingQueue.swift
git commit -m "feat(pipeline): skip companion records in Pass 1, decode from scoringFilePath, propagate cull fields"
```

---

## Task 5: ProcessingQueue — Post-Pass 2 Diversity Sync + Companion Sidecar Writes

**Files:**
- Modify: `ImageRater/Pipeline/ProcessingQueue.swift`

- [ ] **Step 1: Add companion diversity sync after runDiversityPass**

In `ProcessingQueue.process`, after `try await runDiversityPass(sessionID: sessionID)`, add:

```swift
// Sync diversity fields to companion records
await syncCompanionDiversityFields(sessionID: sessionID)
```

Add the private method at the bottom of `ProcessingQueue`:

```swift
private func syncCompanionDiversityFields(sessionID: NSManagedObjectID) async {
    await context.perform { [self] in
        guard let session = try? self.context.existingObject(with: sessionID) as? Session,
              let images = session.images?.allObjects as? [ImageRecord] else { return }

        // Build a map from groupID → primary record
        var primaryByGroup: [String: ImageRecord] = [:]
        for r in images where r.isGroupPrimary, let gid = r.groupID {
            primaryByGroup[gid] = r
        }

        // Copy diversity fields to each companion
        for r in images where !r.isGroupPrimary, let gid = r.groupID,
            let primary = primaryByGroup[gid] {
            r.ratingStars     = primary.ratingStars
            r.combinedQualityScore = primary.combinedQualityScore
            r.finalScore      = primary.finalScore
            r.diversityFactor = primary.diversityFactor
            r.clusterID       = primary.clusterID
            r.clusterRank     = primary.clusterRank
            r.processState    = ProcessState.done
        }
        try? self.context.save()
    }
}
```

- [ ] **Step 2: Update writeSidecars to include companion files**

The existing `writeSidecars` private method already iterates `imageIDs` (which are all records including companions — but companions were skipped in Pass 1 so they're still in state "pending"/"interrupted"). Since companions now have `processState=done` after the sync step, they'd be picked up.

However, `writeSidecars` is called with the original `imageIDs` array which contains all records (including companions). Companions now have `ratingStars` set. The existing logic in `writeSidecars` will write their metadata correctly.

Verify: `writeSidecars` must not skip companions. Currently it checks `r.ratingStars` or `r.userOverride` — companions will have `ratingStars` after sync. No code change needed, but confirm by reading the method:

```swift
// Confirm this method writes to all imageIDs where stars > 0, including companions.
// No change needed — companions have ratingStars after syncCompanionDiversityFields.
```

- [ ] **Step 3: Run all ProcessingQueue and pipeline tests**

Run: `xcodebuild test -scheme ImageRater -only-testing ImageRaterTests/ProcessingQueueTests`

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add ImageRater/Pipeline/ProcessingQueue.swift
git commit -m "feat(pipeline): sync diversity fields to companion records after Pass 2"
```

---

## Task 6: ContentView — filteredImages + Async Export + Progress Bar

**Files:**
- Modify: `ImageRater/App/ContentView.swift`

- [ ] **Step 1: Filter companions from `filteredImages`**

In `ContentView.filteredImages` computed property (line 215), add a companion filter as the **first** filter operation:

```swift
private var filteredImages: [ImageRecord] {
    // Exclude companion (non-primary) records — they share a groupID with their JPEG primary
    var images = Array(sessionImages).filter { $0.isGroupPrimary }

    if !ratingFilter.isEmpty {
    // ... rest of existing filters unchanged
```

- [ ] **Step 2: Add export state properties**

Add two new `@State` properties to `ContentView` alongside the existing processing state vars (after `processingTotal`):

```swift
@State private var exportProgress: (done: Int, total: Int)? = nil
@State private var exportTask: Task<Void, Never>? = nil
```

- [ ] **Step 3: Replace Export button with conditional in toolbar**

In `contentPanel`, find the `ToolbarItem` for "Export XMP". Replace with:

```swift
ToolbarItem {
    if exportProgress == nil {
        Button(action: { exportTask = Task { await exportMetadata(session: session) } }) {
            Label("Export XMP", systemImage: "square.and.arrow.up")
        }
        .disabled(processingStatus != nil)
    }
}
```

- [ ] **Step 4: Add export progress bar to safeAreaInset**

The `.safeAreaInset(edge: .bottom)` currently shows `ProcessingStatusBar` when `processingStatus != nil`. Extend it to also show export progress. Replace the existing `.safeAreaInset` block with:

```swift
.safeAreaInset(edge: .bottom) {
    if let status = processingStatus {
        ProcessingStatusBar(
            status: status,
            done: processingDone,
            total: processingTotal,
            onCancel: { processingTask?.cancel() }
        )
    } else if let ep = exportProgress {
        ProcessingStatusBar(
            status: "Exporting…",
            done: ep.done,
            total: ep.total,
            onCancel: {
                exportTask?.cancel()
                exportProgress = nil
            }
        )
    }
}
```

- [ ] **Step 5: Rewrite `exportMetadata` as `@MainActor async`**

Replace the existing `exportMetadata` method (lines 360–379) with:

```swift
@MainActor
private func exportMetadata(session: Session) async {
    guard let allImages = session.images?.allObjects as? [ImageRecord] else { return }
    let primaries = allImages.filter { $0.isGroupPrimary }
    let total = primaries.count
    guard total > 0 else { return }

    exportProgress = (done: 0, total: total)

    // Build groupID → companions lookup
    var companionsByGroup: [String: [ImageRecord]] = [:]
    for r in allImages where !r.isGroupPrimary, let gid = r.groupID {
        companionsByGroup[gid, default: []].append(r)
    }

    for (index, record) in primaries.enumerated() {
        guard !Task.isCancelled else { break }
        guard let path = record.filePath else { continue }
        let url = URL(filePath: path)

        // Write primary
        if record.cullRejected {
            try? MetadataWriter.writeSidecarRejected(for: url)
        } else {
            let stars: Int
            if let o = record.userOverride?.int16Value, o > 0 { stars = Int(o) }
            else if let s = record.ratingStars?.int16Value, s > 0 { stars = Int(s) }
            else { exportProgress = (done: index + 1, total: total); continue }
            try? MetadataWriter.writeSidecar(stars: stars, for: url)
        }

        // Write all companions in same group
        if let gid = record.groupID, let companions = companionsByGroup[gid] {
            for companion in companions {
                guard let cpath = companion.filePath else { continue }
                let curl = URL(filePath: cpath)
                if record.cullRejected {
                    try? MetadataWriter.writeSidecarRejected(for: curl)
                } else {
                    let stars: Int
                    if let o = record.userOverride?.int16Value, o > 0 { stars = Int(o) }
                    else if let s = record.ratingStars?.int16Value, s > 0 { stars = Int(s) }
                    else { continue }
                    try? MetadataWriter.writeSidecar(stars: stars, for: curl)
                }
            }
        }

        exportProgress = (done: index + 1, total: total)
    }

    exportProgress = nil
}
```

- [ ] **Step 6: Build and verify no compile errors**

Run: `xcodebuild build -scheme ImageRater`

Expected: BUILD SUCCEEDED. Fix any type errors before proceeding.

- [ ] **Step 7: Commit**

```bash
git add ImageRater/App/ContentView.swift
git commit -m "feat(export): async export with progress bar, write metadata to JPEG+RAW companions"
```

---

## Task 7: ThumbnailCell — RAW Pair Badge

**Files:**
- Modify: `ImageRater/UI/Components/ThumbnailCell.swift`

- [ ] **Step 1: Add RAW pill badge to cell overlay**

In `ThumbnailCell.body`, find the `.overlay(alignment: .bottomLeading)` block that shows the cluster "C" badge (lines 75–85). Add a second `.overlay(alignment: .bottomLeading)` for the RAW badge **after** the cluster badge overlay, OR combine them into one overlay using a VStack. The simplest approach — add a new overlay after the existing cluster one:

```swift
.overlay(alignment: .topLeading) {
    if record.scoringFilePath != nil {
        Text("RAW")
            .font(.system(size: 7, weight: .bold))
            .foregroundStyle(.white)
            .padding(.horizontal, 3).padding(.vertical, 1)
            .background(Color.orange.opacity(0.85))
            .clipShape(RoundedRectangle(cornerRadius: 2))
            .padding(4)
    }
}
```

Place this overlay after the existing `.overlay(alignment: .bottomLeading)` cluster badge block and before the diversity overlay.

- [ ] **Step 2: Build and verify**

Run: `xcodebuild build -scheme ImageRater`

Expected: BUILD SUCCEEDED.

- [ ] **Step 3: Commit**

```bash
git add ImageRater/UI/Components/ThumbnailCell.swift
git commit -m "feat(ui): show RAW badge on paired JPEG+RAW thumbnail cells"
```

---

## Final Verification

- [ ] **Run full test suite**

Run: `xcodebuild test -scheme ImageRater`

Expected: all existing tests PASS plus new grouping and migration tests.

- [ ] **Manual smoke test**

1. Open a folder containing both `.jpg` and `.cr2` files with matching base names
2. Verify: grid shows one cell per unique shot (not two)
3. Verify: "RAW" badge appears on paired cells
4. Run processing — verify AI scores appear on the JPEG cell
5. Click "Export XMP" — verify progress bar appears at bottom, counts up, then disappears
6. Verify: both `.jpg` and `.cr2` files have updated metadata (check in Photomator or with `exiftool`)
7. Open a folder with orphan RAW files — verify they appear and are rated normally
