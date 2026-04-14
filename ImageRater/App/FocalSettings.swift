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
