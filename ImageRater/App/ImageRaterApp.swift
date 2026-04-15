import AppKit
import SwiftUI

@main
struct FocalApp: App {
    let persistence = PersistenceController.shared

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environment(\.managedObjectContext, persistence.container.viewContext)
        }
        .commands {
            CommandGroup(replacing: .appInfo) {
                Button("About Focal") {
                    NSApp.orderFrontStandardAboutPanel(options: [
                        .applicationName: "Focal",
                        .credits: NSAttributedString(
                            string: "AI-powered photo culling and rating for macOS.\n\nModels: TOPIQ (IQA-PyTorch), CLIP-IQA (OpenCLIP). RAW decoding via LibRaw.",
                            attributes: [.font: NSFont.systemFont(ofSize: NSFont.smallSystemFontSize)]
                        )
                    ])
                }
            }
        }

        Settings {
            PreferencesView()
        }
    }
}
