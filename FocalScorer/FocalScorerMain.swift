// FocalScorer/main.swift
import Foundation

@main
struct FocalScorerMain {
    static func main() async {
        let args = CommandLine.arguments
        guard args.count >= 3 else {
            FileHandle.standardError.write(Data("""
            usage: FocalScorer <input-dir> <output-json>
              input-dir:   directory of .jpg/.jpeg/.raf/.nef/.arw/.cr3 files
              output-json: path to write the scores JSON
            """.utf8))
            exit(2)
        }
        let inputDir  = URL(fileURLWithPath: args[1])
        let outputURL = URL(fileURLWithPath: args[2])
        do {
            try await Scorer.scoreDirectory(inputDir: inputDir, outputURL: outputURL)
        } catch {
            FileHandle.standardError.write(Data("error: \(error)\n".utf8))
            exit(1)
        }
    }
}
