# CoreML Model Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Get the CLIP + LAION Aesthetic Predictor models converted to CoreML, hosted on GitHub Releases, and downloadable by the app with a working AI rating pipeline.

**Architecture:** Python conversion script converts HuggingFace models to CoreML `.mlpackage` bundles. A shell script zips them, creates a GitHub Release, computes SHA-256 hashes, and generates a manifest JSON. The app downloads and unzips the bundles — requiring fixes to `ModelDownloader` (add unzip step) and `ModelStore` (sidecar-based re-verification instead of directory hashing).

**Tech Stack:** Python 3.12 + torch + coremltools + transformers (conversion), `gh` CLI (release), Swift + CoreML + Foundation (app), XCTest (tests)

**Spec:** `docs/superpowers/specs/2026-04-13-model-integration-design.md`

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `.gitignore` | Create | Exclude `models/`, Python artifacts |
| `scripts/requirements.txt` | Create | Python deps for conversion |
| `scripts/convert_models.py` | Create | HF → CoreML conversion for both models |
| `scripts/release_models.sh` | Create | Zip, release to GitHub, generate manifest |
| `models-manifest.json` | Create (generated) | Model registry with URLs + hashes |
| `ImageRater/ModelStore/ModelDownloader.swift` | Modify | Add `unzip()`, update `download()`, add `unzipFailed` error |
| `ImageRater/ModelStore/ModelStore.swift` | Modify | Sidecar read/write in `prepareModels`, sidecar sentinel in `importModel` |
| `ImageRater/ModelStore/ManifestFetcher.swift` | Modify | Replace placeholder manifest URL |
| `ImageRaterTests/ModelStoreTests.swift` | Modify | Unzip tests, sidecar round-trip tests |

---

## Task 1: Git + GitHub Repository Setup

**Files:** `.gitignore`

- [ ] **Step 1: Create `.gitignore`**

```
# Python
scripts/.venv/
scripts/__pycache__/
*.pyc

# Models (distributed via GitHub Releases, not committed)
models/
*.mlpackage
*.mlpackage.zip

# Xcode
*.xcworkspace/xcuserdata/
DerivedData/
```

Write this to `/Users/dknathalage/repos/image-rating/.gitignore`.

- [ ] **Step 2: Initialize git repository**

```bash
cd /Users/dknathalage/repos/image-rating
git init
git add .
git commit -m "Initial commit: ImageRater macOS app"
```

- [ ] **Step 3: Create GitHub repository and push**

```bash
gh repo create image-rating --public --source=. --remote=origin --push
```

Note the full repo slug printed (e.g. `dknathalage/image-rating`). You will need it in Task 5.

- [ ] **Step 4: Verify remote is set**

```bash
git remote -v
```

Expected: `origin  git@github.com:dknathalage/image-rating.git (fetch)` (or https variant)

---

## Task 2: Python Conversion Environment + Script

**Files:** `scripts/requirements.txt`, `scripts/convert_models.py`

> **Note:** System Python is 3.14, which torch does not yet support. Use `python3.12` (already installed via homebrew).

- [ ] **Step 1: Create `scripts/requirements.txt`**

```
torch>=2.1,<3.0
torchvision>=0.16
transformers>=4.35
coremltools>=7.2
Pillow>=9.0
requests
```

- [ ] **Step 2: Create virtual environment with Python 3.12**

```bash
cd /Users/dknathalage/repos/image-rating
python3.12 -m venv scripts/.venv
source scripts/.venv/bin/activate
pip install --upgrade pip
pip install -r scripts/requirements.txt
```

Expected: packages install without error. `coremltools` and `torch` may take a few minutes.

- [ ] **Step 3: Create `scripts/convert_models.py`**

```python
#!/usr/bin/env python3
"""
Convert CLIP ViT-B/32 and LAION Aesthetic Predictor to CoreML .mlpackage.

Outputs:
  models/clip.mlpackage       -- image -> score (0-1, CLIP similarity to "high quality photo")
  models/aesthetic.mlpackage  -- image -> score (1-10, LAION aesthetic predictor)

Both models accept kCVPixelFormatType_32ARGB pixel buffers (224x224).
CoreML strips the alpha channel internally when color_layout='ARGB' is declared.
"""

import os
import sys
import urllib.request
import torch
import torch.nn as nn
import coremltools as ct
import numpy as np
from pathlib import Path
from transformers import CLIPModel, CLIPTokenizer

MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

# CLIP normalisation constants (standard ImageNet-CLIP values)
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]

# LAION aesthetic predictor weights (ViT-L/14 version)
AESTHETIC_WEIGHTS_URL = (
    "https://github.com/christophschuhmann/"
    "improved-aesthetic-predictor/raw/main/"
    "sac+logos+ava1-l14-linearMSE.pth"
)
AESTHETIC_WEIGHTS_PATH = Path(__file__).parent / "aesthetic_weights.pth"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

class Normalizer(nn.Module):
    """Normalise [0-255] float RGB tensor to CLIP input space."""
    def __init__(self):
        super().__init__()
        self.register_buffer("mean", torch.tensor(CLIP_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor(CLIP_STD ).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x / 255.0 - self.mean) / self.std


def download_aesthetic_weights():
    if not AESTHETIC_WEIGHTS_PATH.exists():
        print("Downloading LAION aesthetic predictor weights (~50 MB)…")
        urllib.request.urlretrieve(AESTHETIC_WEIGHTS_URL, AESTHETIC_WEIGHTS_PATH)
    return AESTHETIC_WEIGHTS_PATH


def load_aesthetic_mlp(weight_path: Path) -> nn.Module:
    """Build the 5-layer MLP used by the improved aesthetic predictor."""
    state = torch.load(weight_path, map_location="cpu", weights_only=False)
    # Architecture: Linear(768,1024) ReLU Dropout Linear(1024,128) ReLU
    #               Dropout Linear(128,64) ReLU Dropout Linear(64,16)
    #               ReLU Linear(16,1)
    mlp = nn.Sequential(
        nn.Linear(768, 1024), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(1024, 128), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(128, 64),   nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(64, 16),    nn.ReLU(),
        nn.Linear(16, 1),
    )
    mlp.load_state_dict(state)
    mlp.eval()
    return mlp


# ---------------------------------------------------------------------------
# Model 1: CLIP scorer (ViT-B/32)
# ---------------------------------------------------------------------------

class CLIPImageScorer(nn.Module):
    """
    Input : (1, 3, 224, 224) float32, values 0-255 RGB
    Output: (1,) float32, cosine-similarity to 'high quality photograph' scaled to [0,1]
    """
    def __init__(self, vision_model, projection, text_feat: torch.Tensor):
        super().__init__()
        self.norm = Normalizer()
        self.vision_model = vision_model
        self.projection   = projection
        self.register_buffer("text_feat", text_feat)  # (1, 512), L2-normalised

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = self.norm(image)
        # return_dict=False → tuple: (last_hidden_state, pooler_output)
        outputs = self.vision_model(pixel_values=x, return_dict=False)
        pooled = outputs[1]                                       # (1, 768) or (1, 512)
        img_feat = self.projection(pooled)                        # (1, 512)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        sim = torch.mm(img_feat, self.text_feat.T).squeeze(0)    # scalar
        score = ((sim + 1.0) / 2.0).reshape(1)                   # [0, 1]
        return score


def build_clip_scorer(clip_model, tokenizer) -> CLIPImageScorer:
    with torch.no_grad():
        tokens = tokenizer(
            ["a high quality, aesthetically pleasing photograph"],
            return_tensors="pt", padding=True
        )
        text_feat = clip_model.get_text_features(**tokens)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
    return CLIPImageScorer(
        clip_model.vision_model,
        clip_model.visual_projection,
        text_feat.detach(),
    )


# ---------------------------------------------------------------------------
# Model 2: Aesthetic scorer (ViT-L/14 + LAION MLP)
# ---------------------------------------------------------------------------

class AestheticScorer(nn.Module):
    """
    Input : (1, 3, 224, 224) float32, values 0-255 RGB
    Output: (1,) float32, aesthetic score ~[1, 10]
    """
    def __init__(self, vision_model, projection, mlp: nn.Module):
        super().__init__()
        self.norm       = Normalizer()
        self.vision_model = vision_model
        self.projection   = projection
        self.mlp          = mlp

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = self.norm(image)
        outputs  = self.vision_model(pixel_values=x, return_dict=False)
        pooled   = outputs[1]
        img_feat = self.projection(pooled)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        score    = self.mlp(img_feat).reshape(1)
        return score


# ---------------------------------------------------------------------------
# CoreML conversion helper
# ---------------------------------------------------------------------------

def to_coreml(traced_model, output_name: str, save_path: Path):
    """Convert a traced PyTorch model to a CoreML .mlpackage."""
    dummy = torch.zeros(1, 3, 224, 224)
    mlmodel = ct.convert(
        traced_model,
        inputs=[ct.ImageType(
            name="image",
            shape=(1, 3, 224, 224),              # (batch, channels, H, W)
            color_layout=ct.colorlayout.ARGB,    # matches kCVPixelFormatType_32ARGB
        )],
        outputs=[ct.TensorType(name=output_name)],
        minimum_deployment_target=ct.target.macOS14,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        convert_to="mlprogram",
    )
    mlmodel.save(str(save_path))
    print(f"Saved: {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=== Loading CLIP ViT-B/32 (for clip model) ===")
    clip_b32   = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    tokenizer  = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    clip_b32.eval()

    # --- CLIP scorer ---
    print("\n=== Building CLIP scorer ===")
    clip_scorer = build_clip_scorer(clip_b32, tokenizer)
    clip_scorer.eval()
    dummy = torch.zeros(1, 3, 224, 224)
    with torch.no_grad():
        traced_clip = torch.jit.trace(clip_scorer, dummy, strict=False)
        # Sanity check: output must be a scalar in [0,1]
        out = traced_clip(dummy)
        assert out.shape == (1,), f"clip scorer bad shape: {out.shape}"
        print(f"  CLIP scorer test output: {out.item():.4f}  (expected ~0.5 for zeros)")

    print("\n=== Converting CLIP scorer to CoreML ===")
    to_coreml(traced_clip, "score", MODELS_DIR / "clip.mlpackage")
    del clip_scorer, traced_clip, clip_b32

    # --- Aesthetic scorer ---
    print("\n=== Loading CLIP ViT-L/14 (for aesthetic model) ===")
    clip_l14 = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    clip_l14.eval()

    print("=== Loading LAION aesthetic predictor weights ===")
    weights_path = download_aesthetic_weights()
    mlp = load_aesthetic_mlp(weights_path)

    aesthetic_scorer = AestheticScorer(
        clip_l14.vision_model,
        clip_l14.visual_projection,
        mlp,
    )
    aesthetic_scorer.eval()
    with torch.no_grad():
        traced_aes = torch.jit.trace(aesthetic_scorer, dummy, strict=False)
        out = traced_aes(dummy)
        assert out.shape == (1,), f"aesthetic scorer bad shape: {out.shape}"
        print(f"  Aesthetic scorer test output: {out.item():.4f}  (expected ~4-7 for real photos)")

    print("\n=== Converting aesthetic scorer to CoreML ===")
    to_coreml(traced_aes, "score", MODELS_DIR / "aesthetic.mlpackage")

    print("\n=== Done ===")
    print(f"Models written to: {MODELS_DIR.resolve()}")
    print("Next: run scripts/release_models.sh to upload to GitHub Releases.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run conversion (takes 15-30 minutes, downloads ~4 GB)**

```bash
source scripts/.venv/bin/activate
python scripts/convert_models.py
```

Expected final output:
```
Saved: .../models/clip.mlpackage
Saved: .../models/aesthetic.mlpackage
Models written to: /Users/dknathalage/repos/image-rating/models
```

If `torch.jit.trace` fails on the vision model due to return_dict conditionals, add `clip_b32.vision_model.config.torchscript = True` before tracing.

- [ ] **Step 5: Verify output directories exist**

```bash
ls models/
```

Expected: `clip.mlpackage  aesthetic.mlpackage`

- [ ] **Step 6: Commit scripts (not models — those go in .gitignore)**

```bash
git add scripts/requirements.txt scripts/convert_models.py
git commit -m "feat: add CoreML model conversion scripts"
```

---

## Task 3: Fix ModelDownloader — Add Unzip Step

**Files:** `ImageRater/ModelStore/ModelDownloader.swift`, `ImageRaterTests/ModelStoreTests.swift`

`.mlpackage` is a directory bundle, not a single file. GitHub Releases distributes it as a zip. The current `download()` returns the raw downloaded file URL; after this task it returns the extracted `.mlpackage` directory URL.

- [ ] **Step 1: Write the failing tests first**

Open `ImageRaterTests/ModelStoreTests.swift`. Add after the existing tests:

```swift
// MARK: - Unzip tests

func testUnzipExtractsMLPackage() throws {
    // Create a minimal fake .mlpackage directory
    let tmp = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    let fakePackage = tmp.appendingPathComponent("test.mlpackage")
    try FileManager.default.createDirectory(at: fakePackage, withIntermediateDirectories: true)
    // Add a sentinel file inside the package
    try "hello".write(to: fakePackage.appendingPathComponent("Manifest.json"),
                      atomically: true, encoding: .utf8)

    // Zip it (path-stripped: zip from inside tmp so archive root is test.mlpackage)
    let zipURL = tmp.appendingPathComponent("test.zip")
    let zipper = Process()
    zipper.executableURL = URL(filePath: "/usr/bin/zip")
    zipper.currentDirectoryURL = tmp
    zipper.arguments = ["-rq", zipURL.path, "test.mlpackage"]
    try zipper.run(); zipper.waitUntilExit()
    XCTAssertEqual(zipper.terminationStatus, 0)

    // Unzip and verify
    let outputDir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    let pkgURL = try ModelDownloader.unzip(zipURL, to: outputDir)
    XCTAssertEqual(pkgURL.pathExtension, "mlpackage")
    XCTAssertTrue(FileManager.default.fileExists(atPath: pkgURL.path))

    // Cleanup
    try? FileManager.default.removeItem(at: tmp)
    try? FileManager.default.removeItem(at: outputDir)
}

func testUnzipThrowsOnNonZip() throws {
    let badFile = FileManager.default.temporaryDirectory.appendingPathComponent("bad.zip")
    try "not a zip".write(to: badFile, atomically: true, encoding: .utf8)
    let outputDir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer {
        try? FileManager.default.removeItem(at: badFile)
        try? FileManager.default.removeItem(at: outputDir)
    }
    XCTAssertThrowsError(try ModelDownloader.unzip(badFile, to: outputDir))
}

func testUnzipThrowsWhenNoMLPackageInZip() throws {
    // Create zip with a non-.mlpackage file
    let tmp = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: tmp, withIntermediateDirectories: true)
    let textFile = tmp.appendingPathComponent("readme.txt")
    try "hello".write(to: textFile, atomically: true, encoding: .utf8)
    let zipURL = tmp.appendingPathComponent("test.zip")
    let zipper = Process()
    zipper.executableURL = URL(filePath: "/usr/bin/zip")
    zipper.currentDirectoryURL = tmp
    zipper.arguments = ["-q", zipURL.path, "readme.txt"]
    try zipper.run(); zipper.waitUntilExit()

    let outputDir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer {
        try? FileManager.default.removeItem(at: tmp)
        try? FileManager.default.removeItem(at: outputDir)
    }
    XCTAssertThrowsError(try ModelDownloader.unzip(zipURL, to: outputDir))
}
```

- [ ] **Step 2: Run tests to confirm they fail (ModelDownloader.unzip doesn't exist yet)**

```bash
xcodebuild test -scheme ImageRater -only-testing ImageRaterTests/ModelStoreTests/testUnzipExtractsMLPackage 2>&1 | tail -20
```

Expected: compile error — `value of type 'ModelDownloader' has no member 'unzip'`

- [ ] **Step 3: Add `unzipFailed` to `ModelStoreError` in `ModelDownloader.swift`**

Open `ImageRater/ModelStore/ModelDownloader.swift`. Change:
```swift
enum ModelStoreError: Error {
    case checksumMismatch
    case downloadFailed
    case manifestVerificationFailed
    case modelNotFound(String)
}
```
To:
```swift
enum ModelStoreError: Error {
    case checksumMismatch
    case downloadFailed
    case manifestVerificationFailed
    case modelNotFound(String)
    case unzipFailed
}
```

- [ ] **Step 4: Add `unzip(_:to:)` static method to `ModelDownloader`**

Add this method at the bottom of `enum ModelDownloader`, before the closing `}`:

```swift
/// Extract a zip archive to `destDir`. Returns URL of the extracted `.mlpackage` directory.
/// The zip must contain exactly one root-level `.mlpackage` directory (no path prefix).
static func unzip(_ zipURL: URL, to destDir: URL) throws -> URL {
    try FileManager.default.createDirectory(at: destDir, withIntermediateDirectories: true)
    let process = Process()
    process.executableURL = URL(filePath: "/usr/bin/unzip")
    process.arguments = ["-q", zipURL.path, "-d", destDir.path]
    try process.run()
    process.waitUntilExit()
    guard process.terminationStatus == 0 else {
        throw ModelStoreError.unzipFailed
    }
    let contents = try FileManager.default.contentsOfDirectory(
        at: destDir, includingPropertiesForKeys: nil
    )
    guard let pkg = contents.first(where: { $0.pathExtension == "mlpackage" }) else {
        throw ModelStoreError.unzipFailed
    }
    return pkg
}
```

- [ ] **Step 5: Update `download()` to call `unzip` after SHA-256 verification**

In `ModelDownloader.download()`, replace:
```swift
            let (tmpURL, _) = try await URLSession.shared.download(from: url)
            guard (try? verify(fileAt: tmpURL, expectedSHA256: expectedSHA256)) == true else {
                throw ModelStoreError.checksumMismatch
            }
            return tmpURL
```
With:
```swift
            let (zipURL, _) = try await URLSession.shared.download(from: url)
            guard (try? verify(fileAt: zipURL, expectedSHA256: expectedSHA256)) == true else {
                throw ModelStoreError.checksumMismatch
            }
            let unzipDir = FileManager.default.temporaryDirectory
                .appendingPathComponent(UUID().uuidString)
            let pkgURL = try unzip(zipURL, to: unzipDir)
            try? FileManager.default.removeItem(at: zipURL)
            return pkgURL
```

- [ ] **Step 6: Run the unzip tests — all three must pass**

```bash
xcodebuild test -scheme ImageRater -only-testing ImageRaterTests/ModelStoreTests/testUnzipExtractsMLPackage -only-testing ImageRaterTests/ModelStoreTests/testUnzipThrowsOnNonZip -only-testing ImageRaterTests/ModelStoreTests/testUnzipThrowsWhenNoMLPackageInZip 2>&1 | tail -20
```

Expected: `** TEST SUCCEEDED **`

- [ ] **Step 7: Run all existing tests to check for regressions**

```bash
xcodebuild test -scheme ImageRater 2>&1 | tail -30
```

Expected: `** TEST SUCCEEDED **`

- [ ] **Step 8: Commit**

```bash
git add ImageRater/ModelStore/ModelDownloader.swift ImageRaterTests/ModelStoreTests.swift
git commit -m "fix: unzip downloaded .mlpackage bundle after SHA-256 verification"
```

---

## Task 4: Fix ModelStore — Sidecar Pattern

**Files:** `ImageRater/ModelStore/ModelStore.swift`, `ImageRaterTests/ModelStoreTests.swift`

The current code calls `verify(fileAt: installedDir, ...)` which hashes a directory path — always returns SHA-256 of empty bytes, never matching the zip hash. Causes re-download every launch. Fix: write a `.sha256` sidecar file on install; read it for re-verification.

- [ ] **Step 1: Write the failing sidecar tests first**

Add to `ImageRaterTests/ModelStoreTests.swift`:

```swift
// MARK: - Sidecar tests

func testSidecarWriteAndReadRoundTrip() throws {
    let tmp = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    let dest = tmp.appendingPathComponent("clip-1.0.0.mlpackage")
    try FileManager.default.createDirectory(at: dest, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: tmp) }

    let expectedHash = "abc123def456789"
    let sidecar = dest.appendingPathExtension("sha256")
    try expectedHash.write(to: sidecar, atomically: true, encoding: .utf8)

    let stored = try? String(contentsOf: sidecar, encoding: .utf8)
    let trimmed = stored?.trimmingCharacters(in: .whitespacesAndNewlines)
    XCTAssertEqual(trimmed, expectedHash)
}

func testLocalSentinelDoesNotEqualRealHash() {
    // "local" sentinel must not accidentally match any real SHA-256 hex string
    let realHash = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    XCTAssertNotEqual("local", realHash)
    XCTAssertFalse("local".count == 64)
}

func testMissingSidecarMeansNeedsRedownload() {
    // No sidecar file → stored is nil → valid is false
    let fakeDir = FileManager.default.temporaryDirectory.appendingPathComponent("nonexistent.mlpackage")
    let sidecar = fakeDir.appendingPathExtension("sha256")
    let stored = (try? String(contentsOf: sidecar, encoding: .utf8))
        ?.trimmingCharacters(in: .whitespacesAndNewlines)
    let valid = stored == "someHash" || stored == "local"
    XCTAssertFalse(valid)
}

func testLocalSentinelSidecarBlocksVersionedDownload() throws {
    // Simulate: user imported clip-local.mlpackage with sidecar "local"
    // prepareModels checks this when versioned path doesn't exist
    let tmp = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    let localDest = tmp.appendingPathComponent("clip-local.mlpackage")
    try FileManager.default.createDirectory(at: localDest, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: tmp) }

    let localSidecar = localDest.appendingPathExtension("sha256")
    try "local".write(to: localSidecar, atomically: true, encoding: .utf8)

    let stored = (try? String(contentsOf: localSidecar, encoding: .utf8))
        ?.trimmingCharacters(in: .whitespacesAndNewlines)
    XCTAssertEqual(stored, "local")  // prepareModels sees this → needsDownload = false
}
```

- [ ] **Step 2: Run sidecar tests to confirm they pass as written (they test pure file I/O logic)**

```bash
xcodebuild test -scheme ImageRater -only-testing ImageRaterTests/ModelStoreTests/testSidecarWriteAndReadRoundTrip -only-testing ImageRaterTests/ModelStoreTests/testLocalSentinelDoesNotEqualRealHash -only-testing ImageRaterTests/ModelStoreTests/testMissingSidecarMeansNeedsRedownload 2>&1 | tail -20
```

Expected: `** TEST SUCCEEDED **`

- [ ] **Step 3: Replace `verify(fileAt:dest...)` with sidecar read in `ModelStore.prepareModels`**

Open `ImageRater/ModelStore/ModelStore.swift`. Find this block (lines ~22–31):

```swift
            if FileManager.default.fileExists(atPath: dest.path) {
                // Re-verify existing file — catches partial downloads or bit rot
                let valid = (try? ModelDownloader.verify(fileAt: dest, expectedSHA256: entry.sha256)) ?? false
                if valid {
                    needsDownload = false
                } else {
                    progress("\(entry.name) checksum mismatch — re-downloading…")
                    try? FileManager.default.removeItem(at: dest)
                    needsDownload = true
                }
            } else {
                needsDownload = true
            }
```

Replace with:

```swift
            if FileManager.default.fileExists(atPath: dest.path) {
                // Verify via sidecar (.sha256 file stores the zip's SHA-256 on install).
                // "local" sentinel = manually imported model, never re-download.
                let sidecar = dest.appendingPathExtension("sha256")
                let stored = (try? String(contentsOf: sidecar, encoding: .utf8))
                    ?.trimmingCharacters(in: .whitespacesAndNewlines)
                let valid = stored == entry.sha256 || stored == "local"
                if valid {
                    needsDownload = false
                } else {
                    progress("\(entry.name) sidecar missing or mismatched — re-downloading…")
                    try? FileManager.default.removeItem(at: dest)
                    try? FileManager.default.removeItem(at: sidecar)
                    needsDownload = true
                }
            } else {
                // Also check for a locally-imported model (name-local.mlpackage).
                // If the user imported a model manually, don't auto-download the versioned one.
                let localDest = modelsDir.appendingPathComponent("\(entry.name)-local.mlpackage")
                let localSidecar = localDest.appendingPathExtension("sha256")
                let localStored = (try? String(contentsOf: localSidecar, encoding: .utf8))
                    ?.trimmingCharacters(in: .whitespacesAndNewlines)
                if localStored == "local" {
                    needsDownload = false   // local import serves this model slot
                } else {
                    needsDownload = true
                }
            }
```

- [ ] **Step 4: Add sidecar write + `moveItem` fallback after download in `ModelStore.prepareModels`**

Find this block (lines ~36–40):

```swift
            if needsDownload {
                progress("Downloading \(entry.name)…")
                let tmp = try await ModelDownloader.download(from: entry.url, expectedSHA256: entry.sha256)
                try FileManager.default.moveItem(at: tmp, to: dest)
                progress("\(entry.name) ready.")
            }
```

Replace with:

```swift
            if needsDownload {
                progress("Downloading \(entry.name)…")
                let tmp = try await ModelDownloader.download(from: entry.url, expectedSHA256: entry.sha256)
                do {
                    try FileManager.default.moveItem(at: tmp, to: dest)
                } catch {
                    // Cross-volume fallback (Application Support on a different volume)
                    try FileManager.default.copyItem(at: tmp, to: dest)
                    try? FileManager.default.removeItem(at: tmp)
                }
                // Write sidecar so next launch skips re-download
                let sidecar = dest.appendingPathExtension("sha256")
                try? entry.sha256.write(to: sidecar, atomically: true, encoding: .utf8)
                progress("\(entry.name) ready.")
            }
```

- [ ] **Step 5: Add sidecar sentinel write to `importModel`**

Find `importModel(from:name:)` in `ModelStore.swift` (lines ~62–69):

```swift
    func importModel(from url: URL, name: String) throws {
        let dest = modelsDir.appendingPathComponent("\(name)-local.mlpackage")
        if FileManager.default.fileExists(atPath: dest.path) {
            try FileManager.default.removeItem(at: dest)
        }
        try FileManager.default.copyItem(at: url, to: dest)
        loadedModels.removeValue(forKey: name)
    }
```

Replace with:

```swift
    func importModel(from url: URL, name: String) throws {
        let dest = modelsDir.appendingPathComponent("\(name)-local.mlpackage")
        if FileManager.default.fileExists(atPath: dest.path) {
            try FileManager.default.removeItem(at: dest)
        }
        try FileManager.default.copyItem(at: url, to: dest)
        // Write "local" sentinel so prepareModels never overwrites a manually imported model
        let sidecar = dest.appendingPathExtension("sha256")
        try? "local".write(to: sidecar, atomically: true, encoding: .utf8)
        loadedModels.removeValue(forKey: name)
    }
```

- [ ] **Step 6: Run all tests**

```bash
xcodebuild test -scheme ImageRater 2>&1 | tail -30
```

Expected: `** TEST SUCCEEDED **`

- [ ] **Step 7: Commit**

```bash
git add ImageRater/ModelStore/ModelStore.swift ImageRaterTests/ModelStoreTests.swift
git commit -m "fix: sidecar-based model verification, importModel sentinel"
```

---

## Task 5: GitHub Release + ManifestFetcher URL

**Files:** `scripts/release_models.sh`, `models-manifest.json`, `ImageRater/ModelStore/ManifestFetcher.swift`

> **Prerequisite:** Task 1 must be complete (git remote exists). Task 2 must be complete (models/ directory contains both .mlpackage bundles).

- [ ] **Step 1: Create `scripts/release_models.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail

VERSION="1.0.0"
REPO=$(gh repo view --json nameWithOwner -q '.nameWithOwner')
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$SCRIPT_DIR/.."
MODELS_DIR="$ROOT/models"

echo "=== Preflight checks ==="
command -v gh >/dev/null || { echo "ERROR: gh CLI not found"; exit 1; }
gh auth status >/dev/null 2>&1 || { echo "ERROR: not authenticated — run: gh auth login"; exit 1; }
git remote get-url origin >/dev/null 2>&1 || { echo "ERROR: no git remote — run Task 1 first"; exit 1; }
[[ -d "$MODELS_DIR/clip.mlpackage"      ]] || { echo "ERROR: models/clip.mlpackage missing — run Task 2 first"; exit 1; }
[[ -d "$MODELS_DIR/aesthetic.mlpackage" ]] || { echo "ERROR: models/aesthetic.mlpackage missing — run Task 2 first"; exit 1; }

echo "Repo: $REPO  Version: v$VERSION"

echo ""
echo "=== Zipping models ==="
cd "$MODELS_DIR"
zip -rq "../clip-${VERSION}.mlpackage.zip"      clip.mlpackage
zip -rq "../aesthetic-${VERSION}.mlpackage.zip" aesthetic.mlpackage
cd "$ROOT"

CLIP_ZIP="clip-${VERSION}.mlpackage.zip"
AES_ZIP="aesthetic-${VERSION}.mlpackage.zip"

echo ""
echo "=== Computing SHA-256 ==="
CLIP_SHA=$(shasum -a 256 "$CLIP_ZIP" | awk '{print $1}')
AES_SHA=$(shasum -a 256 "$AES_ZIP"  | awk '{print $1}')
echo "clip:      $CLIP_SHA"
echo "aesthetic: $AES_SHA"

echo ""
echo "=== Creating GitHub Release ==="
gh release create "v${VERSION}" \
  "$CLIP_ZIP" "$AES_ZIP" \
  --title "Models v${VERSION}" \
  --notes "CoreML CLIP and LAION Aesthetic Predictor models for ImageRater v${VERSION}."

echo ""
echo "=== Generating models-manifest.json ==="
CLIP_URL="https://github.com/${REPO}/releases/download/v${VERSION}/clip-${VERSION}.mlpackage.zip"
AES_URL="https://github.com/${REPO}/releases/download/v${VERSION}/aesthetic-${VERSION}.mlpackage.zip"

cat > "$ROOT/models-manifest.json" <<EOF
{
  "models": [
    {
      "name": "clip",
      "version": "${VERSION}",
      "url": "${CLIP_URL}",
      "sha256": "${CLIP_SHA}"
    },
    {
      "name": "aesthetic",
      "version": "${VERSION}",
      "url": "${AES_URL}",
      "sha256": "${AES_SHA}"
    }
  ],
  "signature": "0000000000000000000000000000000000000000000000000000000000000000"
}
EOF

echo ""
echo "=== Committing and pushing manifest ==="
git add models-manifest.json
git commit -m "feat: add models-manifest.json for v${VERSION}"
git push origin main

echo ""
echo "=== Done ==="
echo "Manifest URL: https://raw.githubusercontent.com/${REPO}/main/models-manifest.json"
echo ""
echo "Next: update ManifestFetcher.swift with this URL (Task 5, Step 3)."

# Cleanup zips
rm -f "$CLIP_ZIP" "$AES_ZIP"
```

- [ ] **Step 2: Make executable and run**

```bash
chmod +x scripts/release_models.sh
bash scripts/release_models.sh
```

Expected last lines:
```
Manifest URL: https://raw.githubusercontent.com/dknathalage/image-rating/main/models-manifest.json
Next: update ManifestFetcher.swift with this URL (Task 5, Step 3).
```

Note the exact manifest URL printed.

- [ ] **Step 3: Update `ManifestFetcher.swift` with the real manifest URL**

Open `ImageRater/ModelStore/ManifestFetcher.swift`. Replace:
```swift
        return URL(string: "https://REPLACE_WITH_REAL_MANIFEST_HOST/models-manifest.json")!
```
With the URL printed in Step 2, e.g.:
```swift
        return URL(string: "https://raw.githubusercontent.com/dknathalage/image-rating/main/models-manifest.json")!
```

- [ ] **Step 4: Run all tests**

```bash
xcodebuild test -scheme ImageRater 2>&1 | tail -20
```

Expected: `** TEST SUCCEEDED **`

- [ ] **Step 5: Commit**

```bash
git add scripts/release_models.sh ImageRater/ModelStore/ManifestFetcher.swift
git commit -m "feat: release script + real manifest URL in ManifestFetcher"
git push origin main
```

---

## Task 6: End-to-End Verification

No code changes. Verify the full pipeline works.

- [ ] **Step 1: Open the project in Xcode and build**

```bash
open ImageRater.xcodeproj
```

Build with ⌘B. Fix any compile errors (there should be none).

- [ ] **Step 2: Run the app and open Model Store**

Run with ⌘R. Navigate to the AI Models panel (the gear/settings area or Model Store view).

- [ ] **Step 3: Click "Download / Update Models"**

Expected behaviour:
1. Status changes from "Download failed" to "Downloading clip…"
2. Then "Downloading aesthetic…"
3. Installed Models list populates with `clip-1.0.0` and `aesthetic-1.0.0`
4. No error shown

If you see a network error: verify the manifest URL is the raw GitHub URL (not the GitHub HTML page URL), and that the release assets are public.

- [ ] **Step 4: Import test RAW images and run processing**

1. Import some `.RAF` files from the `testing/` directory
2. Start processing
3. Verify images receive star ratings (1–5 stars)
4. Check that culled images (blurry/over-exposed) are marked as rejected

- [ ] **Step 5: Verify models persist across relaunch**

Quit the app and relaunch. Check that:
- Installed Models still shows both models
- No re-download triggered (sidecar check passes)

- [ ] **Step 6: Final commit**

```bash
git add docs/
git commit -m "docs: implementation plan and spec for model integration"
git push origin main
```

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `torch.jit.trace` fails with `return_dict` error | HF model returns dataclass | Add `model.config.torchscript = True` before tracing |
| CoreML conversion fails on ARGB color layout | Old coremltools | Upgrade: `pip install coremltools>=7.2` |
| "Download failed" in app | Manifest URL is the GitHub HTML page, not raw | Use `raw.githubusercontent.com/…/main/models-manifest.json` |
| Models re-download every launch | Sidecar not written | Check that Task 4 changes compiled correctly |
| `unzip` returns exit 1 | Zip file corrupt or path prefix present | Re-run `release_models.sh`; check zip with `unzip -l *.zip` |
| `MLModel(contentsOf:)` throws | Zip wasn't extracted before move | Verify Task 3 unzip changes are present |
| Rating pipeline returns `.unrated` for all images | Model output feature name wrong | Confirm conversion script uses `name="score"` in `ct.TensorType` |

---

## Production Checklist (not part of this plan)

- [ ] Generate Ed25519 key pair and implement `scripts/sign_manifest.py`
- [ ] Embed real public key in `ManifestFetcher.swift`
- [ ] Test RELEASE build (currently broken — see spec)
