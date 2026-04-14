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
                .foregroundStyle(Color.accentColor)
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
