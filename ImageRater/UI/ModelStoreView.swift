import AppKit
import SwiftUI

struct ModelStoreView: View {
    @State private var status: String = "Ready"
    @State private var isDownloading = false
    @State private var downloadProgress: Double = 0
    @State private var errorMessage: String?
    @State private var installedModels: [String] = []

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("AI Models").font(.title2).bold()

            Text("Models stored in ~/Library/Application Support/ImageRater/models/")
                .font(.caption).foregroundColor(.secondary)

            // Installed models status
            GroupBox("Installed Models") {
                if installedModels.isEmpty {
                    Text("No models installed")
                        .font(.caption).foregroundColor(.secondary)
                        .padding(.vertical, 4)
                } else {
                    VStack(alignment: .leading, spacing: 4) {
                        ForEach(installedModels, id: \.self) { name in
                            Label(name, systemImage: "checkmark.circle.fill")
                                .foregroundColor(.green)
                                .font(.caption)
                        }
                    }
                    .padding(.vertical, 4)
                }
            }

            Divider()

            // Download from manifest
            HStack {
                Button("Download / Update Models") {
                    Task { await downloadModels() }
                }
                .disabled(isDownloading)
            }

            if isDownloading {
                ProgressView(value: downloadProgress)
                    .progressViewStyle(.linear)
            }

            Text(status).font(.caption).foregroundColor(.secondary)

            if let err = errorMessage {
                Label(err, systemImage: "exclamationmark.triangle")
                    .foregroundColor(.red)
                    .font(.caption)
                Button("Retry") { Task { await downloadModels() } }
            }

            Divider()

            // Local import
            Text("Import Local Models").font(.headline)
            Text("Select a .mlpackage file exported from coremltools.")
                .font(.caption).foregroundColor(.secondary)

            HStack(spacing: 12) {
                Button("Import CLIP Model…") { importModel(named: "clip") }
                Button("Import Aesthetic Model…") { importModel(named: "aesthetic") }
            }
        }
        .padding()
        .frame(minWidth: 420)
        .task { await refreshInstalledModels() }
    }

    @MainActor private func downloadModels() async {
        isDownloading = true
        downloadProgress = 0
        errorMessage = nil
        do {
            try await ModelStore.shared.prepareModels(
                progress: { msg in Task { @MainActor in status = msg } },
                downloadProgress: { p in Task { @MainActor in downloadProgress = p } }
            )
            status = "Models ready."
        } catch {
            errorMessage = error.localizedDescription
            status = "Download failed."
        }
        isDownloading = false
        await refreshInstalledModels()
    }

    @MainActor private func refreshInstalledModels() async {
        installedModels = await ModelStore.shared.installedModelNames()
    }

    private func importModel(named name: String) {
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        panel.message = "Select the \(name).mlpackage directory"
        panel.prompt = "Import"
        guard panel.runModal() == .OK, let url = panel.url else { return }
        Task {
            do {
                try await ModelStore.shared.importModel(from: url, name: name)
                await MainActor.run { status = "\(name) imported." }
                await refreshInstalledModels()
            } catch {
                await MainActor.run {
                    errorMessage = "Import failed: \(error.localizedDescription)"
                }
            }
        }
    }
}
