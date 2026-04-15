import XCTest
@testable import Focal

/// One-time exporter: writes CLIP prompt embeddings to a JSON file the Python
/// bench harness can read. Skip in normal test runs; invoke explicitly when
/// prompt list changes.
final class ExportPromptEmbeddings: XCTestCase {
    func testExportPromptEmbeddings() throws {
        let destPath = ProcessInfo.processInfo.environment["PROMPT_EMBEDDING_EXPORT_PATH"]
        try XCTSkipIf(destPath == nil,
                      "Set PROMPT_EMBEDDING_EXPORT_PATH to export prompt embeddings JSON.")
        let payload: [String: [[Float]]] = [
            "positive": CLIPTextEmbeddings.positivePrompts,
            "negative": CLIPTextEmbeddings.negativePrompts,
        ]
        let data = try JSONSerialization.data(
            withJSONObject: payload.mapValues { $0.map { $0.map { Double($0) } } },
            options: [.prettyPrinted, .sortedKeys]
        )
        try data.write(to: URL(fileURLWithPath: destPath!))
    }
}
