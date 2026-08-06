import Foundation
import Testing
@testable import Switchcraft
@testable import SwitchcraftCore
@testable import SwitchcraftSQLite

#if canImport(CoreML)
@testable import SwitchcraftCoreML
#endif

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
        do {
            try FileManager.default.createDirectory(
                at: dir, withIntermediateDirectories: true
            )
        } catch {
            fatalError(
                "Failed to create temporary database directory at \(dir.path): \(error)"
            )
        }
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

        let hits = try await store.search(query: "banana", topK: 1).hits
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
        ).hits
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
        let hits = try await store.search(query: "keyword", topK: 10).hits
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

        let hits = try await store.search(query: "keyword", topK: 10).hits
        #expect(hits.isEmpty)

        // Adding new content after clear works.
        try await store.add(id: "doc-z", body: "zeta keyword")
        let postClear = try await store.search(query: "keyword", topK: 10).hits
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
        let bananaHits = try await store.search(query: "banana", topK: 10).hits
        let bananaDocX = bananaHits.first(where: { $0.uuid == "doc-x" })
        #expect(bananaDocX != nil)
        #expect(bananaDocX?.ftsRank != nil)

        // First-body keyword no longer finds doc-x via FTS — its FTS row
        // was rewritten by the upsert. (Vector source may still return
        // it as a candidate via random MockEmbedder vectors; the
        // determinism we care about for replacement is that the FTS
        // index reflects the new body only.)
        let appleHits = try await store.search(query: "apple", topK: 10).hits
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
        let hits = try await store.search(query: "distinctivekeyword", topK: 10).hits
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
            _ = try await store.search(query: "alpha", topK: 1).hits
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
            ).hits
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
        let hits = try await reopened.search(query: "uniquemarker", topK: 10).hits
        #expect(hits.contains(where: { $0.uuid == "persisted" }))
        try await reopened.shutdown()
    }

    // MARK: - ledger rehydration after actor restart

    /// Regression test for issue #80: after reopening a store against a
    /// database that already has committed generations, `add` + `search`
    /// must succeed without `Indexer.Error.ledgerOutOfSync`.
    ///
    /// Pre-fix: step 6 (`add` after reopen) causes `flush` on the next
    /// `search` to find only the new doc's rows in the ledger while the
    /// cascade walk counts prior-session rows too → `ledgerOutOfSync`.
    /// Post-fix: the Indexer rehydrates its ledger from bucket data at
    /// construction, so ledger row counts always match storage.
    @Test("reopen then add succeeds: ledger rehydrated from prior-session generations")
    func reopenThenAddSucceeds() async throws {
        let (path, cleanup) = Self.makeTempDatabasePath()
        defer { cleanup() }

        // Session 1: add documents and force a flush so at least one
        // generation is committed to the SQLite file.
        do {
            let store = try await SwitchcraftStore.sqlite(
                databasePath: path,
                embedder: MockEmbedder(dims: 32)
            )
            try await store.add(id: "doc-a", body: "apple fruit sweet red")
            try await store.add(id: "doc-b", body: "banana fruit yellow curved")
            try await store.add(id: "doc-c", body: "cherry fruit dark small")
            // Explicit flush — ensures an L0 generation exists in storage
            // before we close the store.
            try await store.index()
            try await store.shutdown()
        }

        // Session 2: open a *new* store against the same file.
        let reopened = try await SwitchcraftStore.sqlite(
            databasePath: path,
            embedder: MockEmbedder(dims: 32)
        )
        defer { Task { try? await reopened.shutdown() } }

        // R6: search-after-restart must return hits from the prior session.
        let hits1 = try await reopened.search(query: "apple fruit", topK: 5).hits
        #expect(hits1.count > 0, "search after restart returned no hits")

        // R7: add-then-search-after-restart must NOT throw ledgerOutOfSync.
        // Before the fix this throws; after the fix it succeeds.
        try await reopened.add(id: "doc-d", body: "grape fruit purple cluster")
        let hits2 = try await reopened.search(query: "fruit", topK: 10).hits
        #expect(hits2.count > 0, "search after add-then-restart returned no hits")
    }

    // MARK: - dim validation at init

    @Test("invalid embedder dims throws invalidEmbeddingDimensions at init")
    func invalidDimsRejected() async throws {
        struct OddDimsEmbedder: Embedder {
            let dims = 33
            let modelIdentifier = "odd"
            let maxInputTokens: Int = Int.max
            func encode(_ text: String) async throws -> [Float] { [] }
        }

        await #expect(throws: SwitchcraftStoreError.invalidEmbeddingDimensions(33)) {
            _ = try await SwitchcraftStore.sqlite(
                databasePath: ":memory:",
                embedder: OddDimsEmbedder()
            )
        }
    }

    // MARK: - WAL checkpoint bounds regression

    /// Verifies that shutdown() keeps the WAL file bounded across multiple
    /// open/write/shutdown cycles, and that walCheckpoint() flushes pending
    /// writes before checkpointing (durability invariant).
    @Test("WAL file stays bounded across multiple shutdown/reopen cycles")
    func testWALBoundsAcrossShutdownCycles() async throws {
        let (dbPath, cleanupDB) = Self.makeTempDatabasePath()
        let walPath = dbPath + "-wal"
        let shmPath = dbPath + "-shm"
        defer {
            cleanupDB()
            try? FileManager.default.removeItem(atPath: walPath)
            try? FileManager.default.removeItem(atPath: shmPath)
        }

        func walFileSize() -> Int {
            (try? FileManager.default.attributesOfItem(atPath: walPath)[.size] as? Int) ?? 0
        }

        let batchCount = 4
        let batchSize  = 5

        // Measure WAL size after one batch of writes but BEFORE shutdown().
        // This is the single-batch baseline; with shutdown() checkpointing the
        // WAL after each cycle, the N-th batch's pre-shutdown WAL size should
        // be approximately the same — not N times larger.
        var singleBatchWALSize = 0
        for batch in 0..<batchCount {
            let store = try await SwitchcraftStore.sqlite(
                databasePath: dbPath,
                embedder: MockEmbedder(dims: 32)
            )
            for i in 0..<batchSize {
                try await store.add(
                    id: "batch-\(batch)-doc-\(i)",
                    body: "batch \(batch) document \(i) content for WAL bounds test"
                )
            }
            // Capture baseline AFTER writes, BEFORE checkpoint-on-shutdown.
            if batch == 0 {
                singleBatchWALSize = walFileSize()
                #expect(singleBatchWALSize > 0, "WAL should be non-empty after first batch")
            }
            try await store.shutdown()
        }

        // After the last batch's shutdown(), the WAL should have been truncated
        // back to a bounded size (≤ 2× one-batch baseline, not N× it).
        // If shutdown() were NOT checkpointing, the WAL would grow to
        // ≈ batchCount × singleBatchWALSize.
        let finalWALSize = walFileSize()
        #expect(
            finalWALSize <= 2 * singleBatchWALSize,
            "WAL size should stay bounded across multiple shutdown/reopen cycles"
        )

        // walCheckpoint() durability invariant: add a doc, then explicitly
        // checkpoint; the write must survive (be readable through a fresh store).
        let checkpointStore = try await SwitchcraftStore.sqlite(
            databasePath: dbPath,
            embedder: MockEmbedder(dims: 32)
        )
        try await checkpointStore.add(
            id: "checkpoint-doc",
            body: "document added before explicit walCheckpoint call"
        )
        let result = try await checkpointStore.walCheckpoint()
        #expect(result == .complete, "explicit walCheckpoint() should return .complete")
        try await checkpointStore.shutdown()

        // A fresh store must be able to read the doc added before the checkpoint.
        let verifyStore = try await SwitchcraftStore.sqlite(
            databasePath: dbPath,
            embedder: MockEmbedder(dims: 32)
        )
        let count = try await verifyStore.documentCount()
        let expectedCount = batchCount * batchSize + 1
        #expect(
            count == expectedCount,
            "all documents should be readable after explicit walCheckpoint"
        )
        try await verifyStore.shutdown()
    }

    // MARK: - End-to-end with the real T5/CoreML embedder

#if canImport(CoreML)
    /// Asset-gated end-to-end smoke test: exercises the full
    /// `T5CoreMLEmbedder + SwitchcraftStore.sqlite(:memory:)` pipeline
    /// on three real-text documents, then verifies that searching with
    /// the body of a fourth (held-out) document returns the
    /// semantically-closest doc at rank 1.
    ///
    /// Skipped when `SWITCHCRAFT_XTR_MLPACKAGE` is unset.
    @Test(
        "T5CoreMLEmbedder end-to-end: closest semantic match wins at rank 1",
        .enabled(if: CoreMLAsset.isAvailable)
    )
    func t5CoreMLEndToEnd() async throws {
        let modelURL = CoreMLAsset.url!
        let tokenizerURL = Bundle.module.url(
            forResource: "xtr-base-en.tokenizer",
            withExtension: "json",
            subdirectory: "Fixtures"
        )!
        let tokenizer = try Tokenizer(contentsOf: tokenizerURL.path)
        let embedder = try await T5CoreMLEmbedder(
            modelURL: modelURL,
            tokenizer: tokenizer
        )
        let store = try await SwitchcraftStore.sqlite(
            databasePath: ":memory:",
            embedder: embedder
        )

        try await store.add(
            id: "fruit",
            body: "Apples and bananas are popular fruits eaten worldwide."
        )
        try await store.add(
            id: "weather",
            body: "Heavy rainfall and thunderstorms are expected this evening."
        )
        try await store.add(
            id: "code",
            body: "Refactoring legacy code requires careful test coverage."
        )
        try await store.index()

        // The query talks about fruit; "fruit" should top the ranking.
        let hits = try await store.search(
            query: "I picked some red apples from the orchard.",
            topK: 3
        ).hits
        #expect(hits.first?.uuid == "fruit")

        try await store.shutdown()
    }
#endif
}
