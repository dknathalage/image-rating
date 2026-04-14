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
