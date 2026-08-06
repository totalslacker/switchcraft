// SPDX-License-Identifier: Apache-2.0
import Foundation
import Testing
@testable import Switchcraft
@testable import SwitchcraftCore
@testable import SwitchcraftSQLite

/// Store-level tests for the shutdown → reopen snapshot fast path
/// (issue #136 / ADR 034). Exercises the full `SwitchcraftStore.shutdown()`
/// write point and the `Indexer.init` fast path across a real reopen of a
/// file-backed SQLite database.
@Suite("Store Snapshot Shutdown")
struct StoreSnapshotShutdownTests {

    private static func makeTempDatabasePath() -> (path: String, cleanup: () -> Void) {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("switchcraft-snapshot-tests-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let path = dir.appendingPathComponent("store.db").path
        return (path, { try? FileManager.default.removeItem(at: dir) })
    }

    private static let storeConfig = StoreConfig(
        indexer: IndexerConfig(l0Capacity: 4, lsmFanout: 2)
    )

    // MARK: - Reproducer (acceptance criteria 5 + 7)

    @Test("shutdown writes a snapshot; reopen takes the fast path with recoveredConflictCount == 0")
    func shutdownThenReopenUsesFastPath() async throws {
        let (path, cleanup) = Self.makeTempDatabasePath()
        defer { cleanup() }

        // First lifetime: add documents, index, clean shutdown.
        let docs = [
            ("doc-1", "the quick brown fox jumps over"),
            ("doc-2", "a slow green turtle crawls under"),
            ("doc-3", "bright red apples fall from trees"),
            ("doc-4", "cold blue rivers flow to the sea"),
            ("doc-5", "warm yellow suns rise each morning"),
        ]
        do {
            let store = try await SwitchcraftStore.sqlite(
                databasePath: path, embedder: MockEmbedder(dims: 32), config: Self.storeConfig
            )
            for (id, body) in docs { try await store.add(id: id, body: body) }
            try await store.index()
            // First lifetime saw no snapshot at startup (empty DB).
            #expect(await store.indexerUsedSnapshotFastPath == false)
            try await store.shutdown()
        }

        // The clean shutdown must have left a snapshot on disk.
        do {
            let probe = SQLiteStorage(path: path)
            try await probe.open()
            #expect(try await probe.loadLedgerSnapshot() != nil, "shutdown() must persist a snapshot")
            try await probe.close()
        }

        // Second lifetime: reopen. The indexer must take the fast path.
        do {
            let store = try await SwitchcraftStore.sqlite(
                databasePath: path, embedder: MockEmbedder(dims: 32), config: Self.storeConfig
            )
            #expect(await store.indexerUsedSnapshotFastPath == true)

            // Search still works and returns the expected document.
            let hits = try await store.search(query: "quick brown fox", topK: 5).hits
            #expect(hits.contains { $0.uuid == "doc-1" })

            try await store.shutdown()
        }
    }

    // MARK: - Idle/read-only session still (re)writes a snapshot (Research gap)

    @Test("read-only session: shutdown rewrites the snapshot even with nothing pending")
    func idleSessionRewritesSnapshotOnShutdown() async throws {
        let (path, cleanup) = Self.makeTempDatabasePath()
        defer { cleanup() }

        // Lifetime 1: populate + clean shutdown (writes snapshot).
        do {
            let store = try await SwitchcraftStore.sqlite(
                databasePath: path, embedder: MockEmbedder(dims: 32), config: Self.storeConfig
            )
            for i in 1...5 { try await store.add(id: "d\(i)", body: "document number \(i) has words") }
            try await store.index()
            try await store.shutdown()
        }

        // Lifetime 2: reopen (consumes + invalidates the snapshot on load), do
        // NOT add anything (read-only/idle), then shut down. The shutdown must
        // rewrite the snapshot despite nothing being pending, or lifetime 3
        // would silently regress to a full walk.
        do {
            let store = try await SwitchcraftStore.sqlite(
                databasePath: path, embedder: MockEmbedder(dims: 32), config: Self.storeConfig
            )
            #expect(await store.indexerUsedSnapshotFastPath == true)
            _ = try await store.search(query: "document", topK: 3).hits  // read-only work
            try await store.shutdown()
        }

        // Lifetime 3: the idle shutdown's rewritten snapshot must let the fast
        // path fire again.
        do {
            let store = try await SwitchcraftStore.sqlite(
                databasePath: path, embedder: MockEmbedder(dims: 32), config: Self.storeConfig
            )
            #expect(await store.indexerUsedSnapshotFastPath == true,
                    "idle-session shutdown must rewrite the snapshot so the next startup stays fast")
            try await store.shutdown()
        }
    }
}
