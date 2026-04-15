// ImageRater/App/FocalSettings.swift
import Foundation

/// Centralised UserDefaults keys for Focal. All app preferences live here.
/// Use these constants with @AppStorage or UserDefaults directly.
enum FocalSettings {

    // MARK: - UI
    /// Default thumbnail cell size in points. Default: 160
    static let defaultCellSize   = "focal.ui.defaultCellSize"

    // MARK: - Export
    /// Write XMP sidecar automatically on every manual rating. Default: true
    static let autoWriteXMP      = "focal.export.autoWriteXMP"

    // MARK: - Defaults (backed by FocalSettings+Generated.swift)
    static var defaultMUSIQThreshold1: Float { generatedMUSIQThreshold1 }
    static var defaultMUSIQThreshold2: Float { generatedMUSIQThreshold2 }
    static var defaultMUSIQThreshold3: Float { generatedMUSIQThreshold3 }
    static var defaultMUSIQThreshold4: Float { generatedMUSIQThreshold4 }
    static let defaultCellSizeValue: Double = 160
    static let defaultAutoWriteXMP: Bool    = true
}
