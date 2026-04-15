import Foundation

/// Observable progress for rating I/O operations (load sweep + XMP write queue).
/// Updated from background contexts via DispatchQueue.main.async so the UI binding
/// stays on the main actor.
@MainActor
final class RatingProgress: ObservableObject {
    @Published private(set) var phase: String = ""
    @Published private(set) var done: Int = 0
    @Published private(set) var total: Int = 0

    var isActive: Bool { total > 0 }

    func start(phase: String, total: Int) {
        self.phase = phase
        self.done = 0
        self.total = total
    }

    func tick(by n: Int = 1) {
        done = min(done + n, total)
    }

    func finish() {
        phase = ""
        done = 0
        total = 0
    }
}
