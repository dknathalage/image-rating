# CoreML Model Integration Design

**Date:** 2026-04-13  
**Status:** Approved  
**Topic:** Getting CLIP + Aesthetic Predictor models working on-device via GitHub Releases

---

## Problem

The app's AI rating pipeline (`RatingPipeline.swift`) requires two CoreML models:
- `clip` — outputs a `score` Float (0–1) representing photographic quality via CLIP similarity
- `aesthetic` — outputs a `score` Float (1–10) representing aesthetic quality via LAION predictor

The manifest URL and Ed25519 public key in `ManifestFetcher.swift` are placeholders. No models are installed. The "Download / Update Models" button fails with a DNS error.

---

## Architecture

Three independent phases:

```
Phase 1: Convert (Python, run once locally)
  HuggingFace → scripts/convert_models.py → models/clip.mlpackage
                                           → models/aesthetic.mlpackage

Phase 2: Release (shell script, run once per model version)
  .mlpackage dirs → zip (path-stripped) → GitHub Release assets
                 → SHA-256 of each zip → models-manifest.json committed to repo main

Phase 3: Runtime download (existing app flow, two fixes)
  ManifestFetcher → real raw GitHub URL
  → ModelDownloader → download zip → verify SHA-256 → unzip → .mlpackage dir
  → ModelStore → write .sha256 sidecar → move bundle → CoreML inference (unchanged)
```

---

## Models

| Slot | Architecture | Input | Output | Source |
|------|-------------|-------|--------|--------|
| `clip` | CLIP ViT-B/32 image encoder + cosine similarity to baked text embedding | `image` CVPixelBuffer 224×224 `kCVPixelFormatType_32ARGB` | `score` Float 0–1 | `openai/clip-vit-base-patch32` |
| `aesthetic` | CLIP ViT-B/32 encoder + LAION Aesthetic Predictor v2.5 MLP, fused into single CoreML pipeline | `image` CVPixelBuffer 224×224 `kCVPixelFormatType_32ARGB` | `score` Float 1–10 | `christos-c/aesthetic-predictor` |

Both output a single CoreML feature named exactly `"score"` — matches `RatingPipeline.rate()` with no changes.

**Pixel format note:** `RatingPipeline.cgImageToPixelBuffer` creates buffers as `kCVPixelFormatType_32ARGB`. The conversion script must declare this exact format as the CoreML model's input type. This must be verified by running a test prediction with a sample buffer before uploading models.

---

## Phase 1: Python Conversion Script

**File:** `scripts/convert_models.py`  
**Dependencies:** `scripts/requirements.txt`

```
torch>=2.0
transformers>=4.35
coremltools>=7.0
Pillow>=9.0
requests
```

### CLIP model conversion
1. Load `openai/clip-vit-base-patch32` from HuggingFace
2. Pre-compute text embedding for `"a high quality, aesthetically pleasing photograph"`
3. Bake text embedding as constant — model takes only image at inference time
4. Trace image encoder with `torch.jit.trace` on dummy 224×224 input
5. Append cosine similarity op, normalize to 0–1 (add 1, divide by 2)
6. Convert with `coremltools.convert()`:
   - Input name: `"image"`, pixel format: `kCVPixelFormatType_32ARGB` (declared via `coremltools.ImageType`)
   - Output name: `"score"`
7. Save to `models/clip.mlpackage`

### Aesthetic model conversion
1. Reuse CLIP ViT-B/32 image encoder (loaded once)
2. Load LAION Aesthetic Predictor v2.5 MLP weights from HuggingFace
3. Chain: image → CLIP encoder → L2 normalize → MLP head → scalar
4. Trace full pipeline as single `torch.nn.Module`
5. Convert with `coremltools.convert()`:
   - Input name: `"image"`, pixel format: `kCVPixelFormatType_32ARGB`
   - Output name: `"score"`
6. Save to `models/aesthetic.mlpackage`

Script prints SHA-256 of each **unzipped directory** for informational purposes only. The authoritative SHA-256 values used in the manifest are computed by `release_models.sh` in Phase 2 after zipping.

---

## Phase 2: GitHub Release Script

**File:** `scripts/release_models.sh`

### Steps
1. **Preflight:** verify `gh` CLI authenticated, git remote exists, both `.mlpackage` dirs present in `models/`
2. **Zip each model** — strip path prefix so unzip produces `clip.mlpackage` at root, not `models/clip.mlpackage`:
   ```bash
   (cd models && zip -r ../clip-1.0.0.mlpackage.zip clip.mlpackage)
   (cd models && zip -r ../aesthetic-1.0.0.mlpackage.zip aesthetic.mlpackage)
   ```
3. **Compute SHA-256** of each zip file
4. **Create GitHub release:** `gh release create v1.0.0 clip-1.0.0.mlpackage.zip aesthetic-1.0.0.mlpackage.zip`
5. **Generate `models-manifest.json`** at repo root:

```json
{
  "models": [
    {
      "name": "clip",
      "version": "1.0.0",
      "url": "https://github.com/USER/REPO/releases/download/v1.0.0/clip-1.0.0.mlpackage.zip",
      "sha256": "<sha256-of-zip>"
    },
    {
      "name": "aesthetic",
      "version": "1.0.0",
      "url": "https://github.com/USER/REPO/releases/download/v1.0.0/aesthetic-1.0.0.mlpackage.zip",
      "sha256": "<sha256-of-zip>"
    }
  ],
  "signature": "0000000000000000000000000000000000000000000000000000000000000000"
}
```

6. **Commit and push** `models-manifest.json` to `main`

Manifest stable URL: `https://raw.githubusercontent.com/USER/REPO/main/models-manifest.json`

### Ed25519 Signing
Placeholder signature `"0000...0000"` (64 hex chars) used for now. In DEBUG builds the app skips verification when `publicKeyHex` equals the placeholder string — the guard short-circuits before parsing the signature.

**RELEASE BUILDS ARE BROKEN until real Ed25519 signing is added.** The placeholder public key causes the guard to hit `throw manifestVerificationFailed` in release mode. Do not ship to users without completing `scripts/sign_manifest.py`. This is a hard blocker for production.

**Developer warning — mixed credentials:** If a developer sets a real `IMAGERATING_PUBKEY_HEX` env var while the manifest still carries the all-zeros placeholder signature, the guard proceeds past the public key check and calls `pubKey.isValidSignature(sigData, for: canonical)` with 32 zero bytes as the signature. CryptoKit will return false, surfacing as `manifestVerificationFailed` — not a DNS or parsing error. If you see this during development, either clear `IMAGERATING_PUBKEY_HEX` or use a properly signed manifest.

---

## Phase 3: App Changes

### `ImageRater/ModelStore/ManifestFetcher.swift`
Replace one placeholder:
```swift
// Before:
"https://REPLACE_WITH_REAL_MANIFEST_HOST/models-manifest.json"

// After:
"https://raw.githubusercontent.com/USER/REPO/main/models-manifest.json"
```
Public key placeholder stays — DEBUG mode bypasses signature verification.

### `ImageRater/ModelStore/ModelDownloader.swift`
Add unzip step after SHA-256 verification. `.mlpackage` is a directory bundle — must be downloaded as zip and extracted.

After verifying SHA-256 of the downloaded zip:
1. Create a temp subdirectory: `FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)` — on standard macOS configurations this is on the same APFS volume as `~/Library/Application Support`, so `moveItem` succeeds. If `moveItem` throws (e.g. user has Application Support on a different volume via symlink), fall back to `copyItem` + `removeItem`.
2. Run `/usr/bin/unzip -q <zipPath> -d <tempSubdir>` via `Process`
3. Find `.mlpackage` at `tempSubdir/<name>.mlpackage` (one level deep — guaranteed by the path-stripped zip format from Phase 2)
4. Return that URL — caller moves it to final destination unchanged

### `ImageRater/ModelStore/ModelStore.swift`
Two changes needed to handle directory bundles correctly:

**Change 1 — Replace zip-SHA-256 re-verification with sidecar check:**

Current code tries to call `verify(fileAt: dest, expectedSHA256: entry.sha256)` where `dest` is an installed `.mlpackage` directory. This hashes the directory path (not contents), producing a SHA-256 that will never match the zip's SHA-256. Every launch would detect mismatch, delete the model, and re-download.

Fix: write a sidecar file `<name>-<version>.mlpackage.sha256` containing the zip's SHA-256 when installing. On subsequent launches, read and compare the sidecar instead of re-hashing the directory:

```swift
// Writing sidecar (after moveItem):
let sidecar = dest.appendingPathExtension("sha256")
try? entry.sha256.write(to: sidecar, atomically: true, encoding: .utf8)

// Re-verification (replace verify(fileAt:dest...) call):
let sidecarURL = dest.appendingPathExtension("sha256")
let stored = try? String(contentsOf: sidecarURL, encoding: .utf8)
let valid = stored?.trimmingCharacters(in: .whitespacesAndNewlines) == entry.sha256
```

**Change 2 — Detect directory bundle existence correctly:**

`FileManager.default.fileExists(atPath:)` returns `true` for both files and directories, so the existing existence check works correctly for `.mlpackage` bundles. No change needed here.

**Change 3 — `importModel` must also write the sidecar:**

`ModelStore.importModel(from:name:)` copies a local `.mlpackage` into the models directory. Without a sidecar write, the next launch's `prepareModels` will find no sidecar, treat the model as corrupt, and re-download from the manifest — silently overwriting the locally-imported model. Add a sidecar write after `copyItem` in `importModel`. Use `"local"` as the SHA-256 value (a sentinel distinguishing locally-imported models from manifest-managed ones, which `prepareModels` can skip re-verification for).

---

## Error Handling

| Failure | Behaviour |
|---------|-----------|
| Manifest fetch fails | `try?` in `ProcessingQueue` — silently skipped if models already local |
| Download fails | 4× retry with exponential backoff (existing) |
| SHA-256 mismatch | No retry, throws `checksumMismatch` (existing) |
| Unzip fails (non-zero exit) | Throw new error, surface via existing error path |
| `.mlpackage` not found after unzip | Throw `modelNotFound`, surfaces in UI |
| CoreML inference fails | Returns `.unrated` (existing silent fallback) |

---

## Testing

- **Unit:** `ModelDownloader` unzip logic with a fixture `.mlpackage.zip`
- **Unit:** sidecar write/read round-trip in `ModelStore` (install → relaunch → no re-download)
- **Unit:** sidecar sentinel `"local"` written by `importModel` prevents re-download on next launch
- **Pixel format verification:** run `RatingPipeline.rate()` with converted models on a sample ARGB buffer before uploading — confirm non-default `score` value returned
- **Manual smoke:** import via "Import CLIP Model..." + "Import Aesthetic Model..." buttons, confirm models appear in Installed Models list and persist across relaunch
- **Integration:** full download → unzip → rate cycle on a test image with manifest pointing at real GitHub Release

---

## Prerequisites

- Python 3.9+ with pip (~8 GB free disk during conversion for PyTorch + weights)
- GitHub repo with remote configured and `gh auth login` completed
- Xcode 15 / macOS 14 target (existing)
- **For production/release:** Ed25519 key pair generated, manifest signed, public key embedded in app
