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

    // MARK: - Star bucket edges (percentile cut-points in warped space)
    static let bucketEdge1       = "focal.rating.bucketEdge1"
    static let bucketEdge2       = "focal.rating.bucketEdge2"
    static let bucketEdge3       = "focal.rating.bucketEdge3"
    static let bucketEdge4       = "focal.rating.bucketEdge4"

    // MARK: - CLIP-IQA softmax temperature
    static let clipLogitScale    = "focal.rating.clipLogitScale"

    // MARK: - UI
    /// Default thumbnail cell size in points. Default: 160
    static let defaultCellSize   = "focal.ui.defaultCellSize"

    // MARK: - Export
    /// Write XMP sidecar automatically on every manual rating. Default: true
    static let autoWriteXMP      = "focal.export.autoWriteXMP"

    // MARK: - Defaults (backed by FocalSettings+Generated.swift)
    static var defaultCullStrictness: Double   { generatedCullStrictness }
    static var defaultWeightTechnical: Double  { generatedWeightTechnical }
    static var defaultWeightAesthetic: Double  { generatedWeightAesthetic }
    static var defaultWeightClip: Double       { generatedWeightClip }
    static var defaultBucketEdge1: Double      { generatedBucketEdge1 }
    static var defaultBucketEdge2: Double      { generatedBucketEdge2 }
    static var defaultBucketEdge3: Double      { generatedBucketEdge3 }
    static var defaultBucketEdge4: Double      { generatedBucketEdge4 }
    static var defaultClipLogitScale: Double   { generatedClipLogitScale }
    static let defaultCellSizeValue: Double    = 160
    static let defaultAutoWriteXMP: Bool       = true

    // MARK: - Resolved accessors (UserDefaults override → default)

    static func resolvedBucketEdges() -> (Double, Double, Double, Double) {
        let ud = UserDefaults.standard
        func r(_ key: String, _ d: Double) -> Double {
            ud.object(forKey: key) != nil ? ud.double(forKey: key) : d
        }
        return (
            r(bucketEdge1, defaultBucketEdge1),
            r(bucketEdge2, defaultBucketEdge2),
            r(bucketEdge3, defaultBucketEdge3),
            r(bucketEdge4, defaultBucketEdge4)
        )
    }

    static func resolvedClipLogitScale() -> Double {
        let ud = UserDefaults.standard
        return ud.object(forKey: clipLogitScale) != nil
            ? ud.double(forKey: clipLogitScale)
            : defaultClipLogitScale
    }

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
