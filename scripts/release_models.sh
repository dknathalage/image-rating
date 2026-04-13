#!/usr/bin/env bash
# Release CoreML models to GitHub Releases and update models-manifest.json.
#
# Usage: bash scripts/release_models.sh [VERSION]
#   VERSION defaults to 1.0.0
#
# Prerequisites:
#   - gh CLI authenticated (gh auth login)
#   - git remote configured (origin)
#   - models/clip.mlpackage and models/aesthetic.mlpackage present
set -euo pipefail

VERSION="${1:-1.0.0}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="$REPO_ROOT/models"
TAG="v$VERSION"

# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------
echo "=== Preflight checks ==="

if ! gh auth status &>/dev/null; then
    echo "ERROR: gh CLI not authenticated. Run: gh auth login" >&2
    exit 1
fi

if ! git -C "$REPO_ROOT" remote get-url origin &>/dev/null; then
    echo "ERROR: git remote 'origin' not configured." >&2
    exit 1
fi

for pkg in clip aesthetic; do
    if [ ! -d "$MODELS_DIR/$pkg.mlpackage" ]; then
        echo "ERROR: $MODELS_DIR/$pkg.mlpackage not found. Run scripts/convert_models.py first." >&2
        exit 1
    fi
done

echo "  gh:         OK"
echo "  git remote: OK"
echo "  models:     OK (clip + aesthetic)"

# ---------------------------------------------------------------------------
# Zip models (path-stripped so unzip produces clip.mlpackage at root)
# ---------------------------------------------------------------------------
echo ""
echo "=== Zipping models ==="

CLIP_ZIP="$REPO_ROOT/clip-${VERSION}.mlpackage.zip"
AES_ZIP="$REPO_ROOT/aesthetic-${VERSION}.mlpackage.zip"

rm -f "$CLIP_ZIP" "$AES_ZIP"

(cd "$MODELS_DIR" && zip -qr "$CLIP_ZIP" clip.mlpackage)
echo "  clip zip:      $(du -sh "$CLIP_ZIP" | cut -f1)"

(cd "$MODELS_DIR" && zip -qr "$AES_ZIP" aesthetic.mlpackage)
echo "  aesthetic zip: $(du -sh "$AES_ZIP" | cut -f1)"

# ---------------------------------------------------------------------------
# SHA-256 of each zip
# ---------------------------------------------------------------------------
echo ""
echo "=== Computing SHA-256 ==="

CLIP_SHA=$(shasum -a 256 "$CLIP_ZIP" | awk '{print $1}')
AES_SHA=$(shasum -a 256 "$AES_ZIP" | awk '{print $1}')

echo "  clip:      $CLIP_SHA"
echo "  aesthetic: $AES_SHA"

# ---------------------------------------------------------------------------
# Create GitHub Release
# ---------------------------------------------------------------------------
echo ""
echo "=== Creating GitHub Release $TAG ==="

# Delete existing tag/release if present (idempotent re-run)
if gh release view "$TAG" &>/dev/null; then
    echo "  Release $TAG already exists — deleting and recreating."
    gh release delete "$TAG" --yes --cleanup-tag 2>/dev/null || true
fi

gh release create "$TAG" \
    "$CLIP_ZIP" \
    "$AES_ZIP" \
    --title "CoreML Models $VERSION" \
    --notes "CLIP ViT-B/32 scorer and LAION Aesthetic Predictor v2.5 (ViT-L/14) as CoreML .mlpackage bundles.

Models built with:
- openai/clip-vit-base-patch32 (clip slot)
- openai/clip-vit-large-patch14 + sac+logos+ava1-l14-linearMSE.pth (aesthetic slot)

Both accept 224×224 RGB pixel buffers and output a single \`score\` float."

echo "  Release created: $TAG"

# ---------------------------------------------------------------------------
# Build asset URLs
# ---------------------------------------------------------------------------
REPO_SLUG=$(gh repo view --json nameWithOwner -q .nameWithOwner)
CLIP_URL="https://github.com/${REPO_SLUG}/releases/download/${TAG}/clip-${VERSION}.mlpackage.zip"
AES_URL="https://github.com/${REPO_SLUG}/releases/download/${TAG}/aesthetic-${VERSION}.mlpackage.zip"

# ---------------------------------------------------------------------------
# Write models-manifest.json
# ---------------------------------------------------------------------------
echo ""
echo "=== Writing models-manifest.json ==="

cat > "$REPO_ROOT/models-manifest.json" <<MANIFEST
{
  "models": [
    {
      "name": "clip",
      "version": "$VERSION",
      "url": "$CLIP_URL",
      "sha256": "$CLIP_SHA"
    },
    {
      "name": "aesthetic",
      "version": "$VERSION",
      "url": "$AES_URL",
      "sha256": "$AES_SHA"
    }
  ],
  "signature": "0000000000000000000000000000000000000000000000000000000000000000"
}
MANIFEST

echo "  Written: $REPO_ROOT/models-manifest.json"
cat "$REPO_ROOT/models-manifest.json"

# ---------------------------------------------------------------------------
# Commit and push manifest
# ---------------------------------------------------------------------------
echo ""
echo "=== Committing and pushing manifest ==="

git -C "$REPO_ROOT" add models-manifest.json
git -C "$REPO_ROOT" commit -m "release: add models-manifest.json for v$VERSION

clip sha256:      $CLIP_SHA
aesthetic sha256: $AES_SHA"
git -C "$REPO_ROOT" push origin main

echo ""
echo "=== Done ==="
echo "Manifest URL: https://raw.githubusercontent.com/${REPO_SLUG}/main/models-manifest.json"
echo ""
echo "Next: update ManifestFetcher.swift with this URL."

# Cleanup zips
rm -f "$CLIP_ZIP" "$AES_ZIP"
