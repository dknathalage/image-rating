import CoreData

// ModelConfig is a global singleton row (one config per install).
// ProcessingQueue uses fetchOrCreate to ensure exactly one instance.
final class PersistenceController {
    static let shared = PersistenceController()

    let container: NSPersistentContainer

    init(inMemory: Bool = false) {
        container = NSPersistentContainer(name: "ImageRater")
        if inMemory {
            container.persistentStoreDescriptions.first?.url = URL(filePath: "/dev/null")
        }
        // Enable lightweight migration so schema changes don't crash existing stores
        let description = container.persistentStoreDescriptions.first!
        description.setOption(true as NSNumber, forKey: NSMigratePersistentStoresAutomaticallyOption)
        description.setOption(true as NSNumber, forKey: NSInferMappingModelAutomaticallyOption)
        container.loadPersistentStores { _, error in
            if let error { fatalError("Core Data load failed: \(error)") }
        }
        container.viewContext.automaticallyMergesChangesFromParent = true
    }
}
