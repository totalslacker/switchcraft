import Foundation
import Testing
@testable import Switchcraft
@testable import SwitchcraftCore
@testable import SwitchcraftSQLite

@Suite("SwitchcraftStore")
struct SwitchcraftStoreTests {

    // MARK: - Helpers

    /// Spawn a fresh SQLite-backed store with a deterministic
    /// `MockEmbedder`. `path` defaults to `:memory:`.
    static func makeStore(
        path: String = ":memory:",
        dims: Int = 32
    ) async throws -> SwitchcraftStore {
        try await SwitchcraftStore.sqlite(
            databasePath: path,
            embedder: MockEmbedder(dims: dims)
        )
    }

    /// Allocate (and reserve cleanup of) a temporary SQLite path. The
    /// caller MUST call the returned `cleanup` closure once it is done.
    static func makeTempDatabasePath() -> (path: String, cleanup: () -> Void) {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("switchcraft-store-tests-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(
            at: dir, withIntermediateDirectories: true
        )
        let path = dir.appendingPathComponent("store.db").path
        let cleanup: () -> Void = {
            try? FileManager.default.removeItem(at: dir)
        }
        return (path, cleanup)
    }

    // MARK: - add → search round-trip

    @Test("add three docs; search returns the matching doc at rank 1; topK=1 returns one hit")
    func roundTripTopHit() async throws {
        let store = try await Self.makeStore()
        try await store.add(id: "doc-a", body: "the apple is red and round")
        try await store.add(id: "doc-b", body: "the banana is yellow and curved")
        try await store.add(id: "doc-c", body: "the cherry is dark and small")

        let hits = try await store.search(query: "banana", topK: 1)
        #expect(hits.count == 1)
        #expect(hits[0].uuid == "doc-b")

        try await store.shutdown()
    }

    // MARK: - filter

    @Test("filter via uuidIn restricts results to the named documents")
    func filterRespected() async throws {
        let store = try await Self.makeStore()
        try await store.add(id: "doc-a", body: "alpha shared keyword")
        try await store.add(id: "doc-b", body: "beta shared keyword")
        try await store.add(id: "doc-c", body: "gamma shared keyword")

        let hits = try await store.search(
            query: "shared keyword",
            topK: 10,
            filter: .uuidIn(["doc-a", "doc-b"])
        )
        let uuids = Set(hits.map(\.uuid))
        #expect(uuids.isSubset(of: ["doc-a", "doc-b"]))
        #expect(!uuids.contains("doc-c"))
        #expect(hits.count >= 1)

        try await store.shutdown()
    }

    // MARK: - remove

    @Test("remove drops the document from subsequent search results; missing id is a no-op")
    func removeWorksAndIsNoOpForMissing() async throws {
        let store = try await Self.makeStore()
        try await store.add(id: "doc-a", body: "alpha keyword")
        try await store.add(id: "doc-b", body: "beta keyword")

        // No-op for missing id.
        try await store.remove(id: "does-not-exist")

        try await store.remove(id: "doc-a")
        let hits = try await store.search(query: "keyword", topK: 10)
        #expect(!hits.contains(where: { $0.uuid == "doc-a" }))
        #expect(hits.contains(where: { $0.uuid == "doc-b" }))

        try await store.shutdown()
    }

    // MARK: - clear

    @Test("clear empties documents and the index; subsequent search returns []")
    func clearEmptiesEverything() async throws {
        let store = try await Self.makeStore()
        try await store.add(id: "doc-a", body: "alpha keyword")
        try await store.add(id: "doc-b", body: "beta keyword")
        // Force a flush before clearing to make sure clear handles
        // persisted generations too, not just the in-memory ledger.
        try await store.index()
        try await store.clear()

        let hits = try await store.search(query: "keyword", topK: 10)
        #expect(hits.isEmpty)

        // Adding new content after clear works.
        try await store.add(id: "doc-z", body: "zeta keyword")
        let postClear = try await store.search(query: "keyword", topK: 10)
        #expect(postClear.contains(where: { $0.uuid == "doc-z" }))

        try await store.shutdown()
    }

    // MARK: - idempotent add (same id, different body)

    @Test("adding the same id twice with different bodies; search reflects only the second body")
    func idempotentAddReplacesBody() async throws {
        let store = try await Self.makeStore()
        // Add a control doc that contains "apple" so the FTS path has a
        // genuine match for the first-body keyword; this lets us tell
        // replacement apart from "doc-x is the only doc in storage".
        try await store.add(id: "doc-control", body: "apple original document")
        try await store.add(id: "doc-x", body: "apple variant")
        try await store.add(id: "doc-x", body: "banana variant")

        // Second-body keyword finds doc-x via FTS.
        let bananaHits = try await store.search(query: "banana", topK: 10)
        let bananaDocX = bananaHits.first(where: { $0.uuid == "doc-x" })
        #expect(bananaDocX != nil)
        #expect(bananaDocX?.ftsRank != nil)

        // First-body keyword no longer finds doc-x via FTS — its FTS row
        // was rewritten by the upsert. (Vector source may still return
        // it as a candidate via random MockEmbedder vectors; the
        // determinism we care about for replacement is that the FTS
        // index reflects the new body only.)
        let appleHits = try await store.search(query: "apple", topK: 10)
        let appleDocX = appleHits.first(where: { $0.uuid == "doc-x" })
        #expect(appleDocX?.ftsRank == nil)
        #expect(appleHits.contains(where: { $0.uuid == "doc-control" }))

        try await store.shutdown()
    }

    // MARK: - score

    @Test("score returns one float per passage; values reproducible across runs")
    func scoreReproducible() async throws {
        let store1 = try await Self.makeStore()
        let store2 = try await Self.makeStore()

        let passages = [
            "the apple is red",
            "the banana is yellow",
            "this passage is unrelated and longer than the others",
        ]
        let scores1 = try await store1.score(query: "banana", passages: passages)
        let scores2 = try await store2.score(query: "banana", passages: passages)

        #expect(scores1.count == passages.count)
        #expect(scores1 == scores2)

        try await store1.shutdown()
        try await store2.shutdown()
    }

    // MARK: - auto-flush

    @Test("search auto-flushes pending L0 embeddings; no explicit index() needed")
    func searchAutoFlushes() async throws {
        let store = try await Self.makeStore()
        try await store.add(id: "doc-a", body: "alpha distinctivekeyword")
        try await store.add(id: "doc-b", body: "beta")
        try await store.add(id: "doc-c", body: "gamma")

        // No explicit `index()` call.
        let hits = try await store.search(query: "distinctivekeyword", topK: 10)
        #expect(hits.contains(where: { $0.uuid == "doc-a" }))

        try await store.shutdown()
    }

    // MARK: - shutdown

    @Test("post-shutdown calls throw alreadyShutDown; second shutdown is a no-op")
    func shutdownIsIdempotentAndGated() async throws {
        let store = try await Self.makeStore()
        try await store.add(id: "doc-a", body: "alpha")

        try await store.shutdown()
        // Second shutdown is a no-op.
        try await store.shutdown()

        await #expect(throws: SwitchcraftStoreError.alreadyShutDown) {
            try await store.add(id: "doc-b", body: "beta")
        }
        await #expect(throws: SwitchcraftStoreError.alreadyShutDown) {
            _ = try await store.search(query: "alpha", topK: 1)
        }
        await #expect(throws: SwitchcraftStoreError.alreadyShutDown) {
            try await store.remove(id: "doc-a")
        }
        await #expect(throws: SwitchcraftStoreError.alreadyShutDown) {
            try await store.index()
        }
        await #expect(throws: SwitchcraftStoreError.alreadyShutDown) {
            try await store.clear()
        }
        await #expect(throws: SwitchcraftStoreError.alreadyShutDown) {
            _ = try await store.score(query: "alpha", passages: ["x"])
        }
    }

    // MARK: - determinism

    @Test("two identical runs produce identical hit ordering and float-equal scores")
    func deterministicAcrossRuns() async throws {
        let docs: [(String, String)] = [
            ("doc-a", "the apple is red"),
            ("doc-b", "the banana is yellow"),
            ("doc-c", "the cherry is dark"),
            ("doc-d", "the durian is pungent"),
            ("doc-e", "the elderberry is small"),
        ]

        func run() async throws -> [HybridHit] {
            let store = try await Self.makeStore()
            for (id, body) in docs {
                try await store.add(id: id, body: body)
            }
            let hits = try await store.search(
                query: "apple banana cherry",
                topK: 10
            )
            try await store.shutdown()
            return hits
        }

        let a = try await run()
        let b = try await run()
        #expect(a == b)
    }

    // MARK: - open / reopen

    @Test("reopening the same SQLite database resumes existing documents")
    func reopenResumesDocuments() async throws {
        let (path, cleanup) = Self.makeTempDatabasePath()
        defer { cleanup() }

        do {
            let store = try await SwitchcraftStore.sqlite(
                databasePath: path,
                embedder: MockEmbedder(dims: 32)
            )
            try await store.add(id: "persisted", body: "alpha keyword uniquemarker")
            try await store.add(id: "transient", body: "beta keyword")
            try await store.shutdown()
        }

        let reopened = try await SwitchcraftStore.sqlite(
            databasePath: path,
            embedder: MockEmbedder(dims: 32)
        )
        let hits = try await reopened.search(query: "uniquemarker", topK: 10)
        #expect(hits.contains(where: { $0.uuid == "persisted" }))
        try await reopened.shutdown()
    }

    // MARK: - dim validation at init

    @Test("invalid embedder dims throws invalidEmbeddingDimensions at init")
    func invalidDimsRejected() async throws {
        struct OddDimsEmbedder: Embedder {
            let dims = 33
            let modelIdentifier = "odd"
            func encode(_ text: String) async throws -> [Float] { [] }
        }

        await #expect(throws: SwitchcraftStoreError.invalidEmbeddingDimensions(33)) {
            _ = try await SwitchcraftStore.sqlite(
                databasePath: ":memory:",
                embedder: OddDimsEmbedder()
            )
        }
    }
}
