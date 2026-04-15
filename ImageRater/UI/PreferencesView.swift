// ImageRater/UI/PreferencesView.swift
import SwiftUI

struct PreferencesView: View {
    var body: some View {
        TabView {
            AppearanceTab()
                .tabItem { Label("Appearance", systemImage: "paintbrush") }
            ExportTab()
                .tabItem { Label("Export", systemImage: "square.and.arrow.up") }
        }
        .frame(width: 480)
        .padding()
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
