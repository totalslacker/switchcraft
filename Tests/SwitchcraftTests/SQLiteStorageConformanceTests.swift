import Foundation
import Testing
import SwitchcraftCore
import SwitchcraftSQLite
import SwitchcraftStorageTesting

@Suite("SQLiteStorage Conformance")
struct SQLiteStorageConformanceTests {

    @Test("In-memory SQLiteStorage satisfies the SwitchcraftStorage contract")
    func inMemoryConformance() async throws {
        try await StorageConformance.runAll {
            SQLiteStorage(path: ":memory:")
        }
    }

    @Test("On-disk SQLiteStorage satisfies the SwitchcraftStorage contract")
    func onDiskConformance() async throws {
        let path = NSTemporaryDirectory().appending("switchcraft-conformance-\(UUID().uuidString).sqlite3")
        defer { try? FileManager.default.removeItem(atPath: path) }

        try await StorageConformance.runAll {
            SQLiteStorage(path: path)
        }
    }

    @Test("Reopening an on-disk store preserves committed data")
    func onDiskPersistence() async throws {
        let path = NSTemporaryDirectory().appending("switchcraft-persist-\(UUID().uuidString).sqlite3")
        defer { try? FileManager.default.removeItem(atPath: path) }

        do {
            let storage = SQLiteStorage(path: path)
            try await storage.open()
            try await storage.upsertDocument(DocumentRecord(
                uuid: "persist-1",
                date: Date(timeIntervalSince1970: 12345),
                metadata: Data(),
                hash: "h",
                body: "persisted body",
                lens: [3, 4]
            ))
            try await storage.close()
        }

        let storage = SQLiteStorage(path: path)
        try await storage.open()
        let fetched = try await storage.document(uuid: "persist-1")
        #expect(fetched?.body == "persisted body")
        #expect(fetched?.lens == [3, 4])
        try await storage.close()
    }
}
