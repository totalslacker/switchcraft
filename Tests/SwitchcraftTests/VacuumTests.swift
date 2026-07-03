// Tests for SwitchcraftStore.vacuum() (issue #134).
//
// vacuum() removes chunks with no owning document ("abandoned" chunks),
// cascades empty buckets/generations, keeps the Indexer ledger consistent,
// and reports batch/idempotency metadata via VacuumResult. This is
// orthogonal to findOrphanedChunks()/add()'s R1 recovery path, which
// targets chunks that DO have an owning document but are incompletely
// indexed — vacuum must never remove those.

import Foundation
import Testing
@testable import Switchcraft
@testable import SwitchcraftCore
@testable import SwitchcraftSQLite

@Suite("Vacuum")
struct VacuumTests {

    // MARK: - Backend parameterization

    enum Backend: String, CaseIterable, Sendable {
        case inMemory
        case sqlite
    }

    private static func makeStore(
        _ backend: Backend,
        config: StoreConfig = StoreConfig(indexer: IndexerConfig(l0Capacity: 4, lsmFanout: 2)),
        dims: Int = 32
    ) async throws -> (SwitchcraftStore, any SwitchcraftStorage) {
        let storage: any SwitchcraftStorage
        switch backend {
        case .inMemory: storage = InMemoryStorage()
        case .sqlite: storage = SQLiteStorage(path: ":memory:")
        }
        let store = try await SwitchcraftStore(
            storage: storage,
            embedder: MockEmbedder(dims: dims),
            config: config
        )
        return (store, storage)
    }

    /// Allocate (and reserve cleanup of) a temporary SQLite path.
    private static func makeTempDatabasePath() -> (path: String, cleanup: () -> Void) {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("switchcraft-vacuum-tests-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let path = dir.appendingPathComponent("store.db").path
        return (path, { try? FileManager.default.removeItem(at: dir) })
    }

    // MARK: - 16(a)-(d): basic removal correctness, parameterized over both backends

    @Test(
        "vacuum removes abandoned chunks, excludes them from findOrphanedChunks, leaves documentCount and search results for surviving docs unchanged",
        arguments: Backend.allCases
    )
    func vacuumRemovesAbandonedChunks(_ backend: Backend) async throws {
        let (store, storage) = try await Self.makeStore(backend)

        // Corpus: 4 documents, each with distinct content.
        let docs = [
            ("keep-1", "the quick brown fox jumps"),
            ("keep-2", "a slow green turtle crawls"),
            ("gone-1", "an old red bicycle rusts"),
            ("gone-2", "a new blue umbrella opens"),
        ]
        for (id, body) in docs {
            try await store.add(id: id, body: body)
        }
        try await store.index()

        let goneHashes = ["gone-1", "gone-2"].map { id -> String in
            SwitchcraftStore.contentHash(docs.first { $0.0 == id }!.1)
        }

        // Abandon gone-1 / gone-2 by deleting the owning documents.
        try await store.remove(id: "gone-1")
        try await store.remove(id: "gone-2")

        // Capture search results for surviving docs after remove() but
        // before vacuum() — this is the baseline vacuum() must not disturb.
        let beforeHitsKeep1 = try await store.search(query: "quick brown fox", topK: 10)
        let beforeHitsKeep2 = try await store.search(query: "slow green turtle", topK: 10)
        let countBefore = try await store.documentCount()

        let result = try await store.vacuum()

        #expect(result.chunksRemoved == 2)
        #expect(result.remainingCandidates == 0)

        // (a) abandoned chunk rows are gone.
        for hash in goneHashes {
            #expect(try await storage.chunk(hash: hash) == nil, "abandoned chunk must be deleted")
        }

        // (b) findOrphanedChunks() has no entries for the removed chunks.
        let orphans = try await store.findOrphanedChunks()
        #expect(orphans.allSatisfy { !goneHashes.contains($0.hash) })

        // (c) documentCount() unchanged by vacuum() itself.
        let countAfter = try await store.documentCount()
        #expect(countAfter == countBefore, "vacuum() must not change documentCount() beyond what remove() already did")

        // (d) search() results for surviving docs are unchanged.
        let afterHitsKeep1 = try await store.search(query: "quick brown fox", topK: 10)
        let afterHitsKeep2 = try await store.search(query: "slow green turtle", topK: 10)
        #expect(afterHitsKeep1.map(\.uuid) == beforeHitsKeep1.map(\.uuid))
        #expect(afterHitsKeep2.map(\.uuid) == beforeHitsKeep2.map(\.uuid))
        #expect(afterHitsKeep1.contains { $0.uuid == "keep-1" })
        #expect(afterHitsKeep2.contains { $0.uuid == "keep-2" })

        try await store.shutdown()
    }

    // MARK: - 12: idempotent on a clean store

    @Test("vacuum on a clean store (no abandoned chunks) returns all-zero counts", arguments: Backend.allCases)
    func vacuumOnCleanStoreIsIdempotent(_ backend: Backend) async throws {
        let (store, _) = try await Self.makeStore(backend)

        // Empty store.
        let emptyResult = try await store.vacuum()
        #expect(emptyResult.chunksRemoved == 0)
        #expect(emptyResult.bucketPairsRemoved == 0)
        #expect(emptyResult.approximateDiskReclaimed == 0)
        #expect(emptyResult.generationsAffected.isEmpty)
        #expect(emptyResult.remainingCandidates == 0)

        // Fully-owned corpus, nothing abandoned.
        try await store.add(id: "solo", body: "nothing abandoned here")
        try await store.index()
        let noneAbandonedResult = try await store.vacuum()
        #expect(noneAbandonedResult.chunksRemoved == 0)
        #expect(noneAbandonedResult.bucketPairsRemoved == 0)
        #expect(noneAbandonedResult.remainingCandidates == 0)

        try await store.shutdown()
    }

    // MARK: - 16(e): cascade deletion + ledger consistency after vacuum

    @Test("emptied buckets/generations are cascade-deleted; subsequent add()/index()/search() does not throw ledgerOutOfSync", arguments: Backend.allCases)
    func vacuumCascadeAndLedgerConsistency(_ backend: Backend) async throws {
        let (store, storage) = try await Self.makeStore(backend)

        // Small l0Capacity means every doc below lands in its own generation
        // (or a small cascade), so abandoning all of them empties buckets and
        // generations entirely.
        let abandonedDocs = (0..<6).map { ("cascade-\($0)", "cascade content number \($0) words here") }
        for (id, body) in abandonedDocs {
            try await store.add(id: id, body: body)
            try await store.index()
        }

        // A separate, never-abandoned document that must survive vacuum and
        // remain searchable afterward.
        try await store.add(id: "survivor", body: "the survivor document remains searchable")
        try await store.index()

        for (id, _) in abandonedDocs {
            try await store.remove(id: id)
        }

        let gensBefore = try await storage.generations()
        #expect(!gensBefore.isEmpty)

        let result = try await store.vacuum()
        #expect(result.chunksRemoved == abandonedDocs.count)
        #expect(!result.generationsAffected.isEmpty, "at least one generation must be reported as affected")

        // Every generation that still exists must have at least one bucket
        // (fully-emptied generations must be deleted, not left as husks).
        let gensAfter = try await storage.generations()
        for gen in gensAfter {
            let buckets = try await storage.buckets(forGeneration: gen.id)
            #expect(!buckets.isEmpty, "surviving generation \(gen.id) must not be empty")
        }

        // No ledgerOutOfSync on subsequent add()/index()/search().
        try await store.add(id: "post-vacuum-doc", body: "post vacuum content words")
        try await store.index()
        let hits = try await store.search(query: "survivor document remains", topK: 10)
        #expect(hits.contains { $0.uuid == "survivor" })
        let hits2 = try await store.search(query: "post vacuum content", topK: 10)
        #expect(hits2.contains { $0.uuid == "post-vacuum-doc" })

        try await store.shutdown()
    }

    // MARK: - 16(f)/5/14: maxBatch, remainingCandidates, and looped vs. unbounded parity

    @Test("bounded maxBatch reports remainingCandidates correctly; looping to remainingCandidates == 0 matches a single unbounded call", arguments: Backend.allCases)
    func vacuumBatchingAndLoopParity(_ backend: Backend) async throws {
        let totalAbandoned = 9
        let batchSize = 4

        // Run A: single unbounded vacuum() call.
        let (storeA, storageA) = try await Self.makeStore(backend)
        for i in 0..<totalAbandoned {
            try await storeA.add(id: "batch-doc-\(i)", body: "batch content variant \(i) extra words padding")
        }
        try await storeA.index()
        for i in 0..<totalAbandoned {
            try await storeA.remove(id: "batch-doc-\(i)")
        }
        let unboundedResult = try await storeA.vacuum(maxBatch: nil)
        #expect(unboundedResult.chunksRemoved == totalAbandoned)
        #expect(unboundedResult.remainingCandidates == 0)
        let chunkCountA = try await storageA.chunkCount()
        try await storeA.shutdown()

        // Run B: identical corpus, drained via repeated bounded maxBatch calls.
        let (storeB, storageB) = try await Self.makeStore(backend)
        for i in 0..<totalAbandoned {
            try await storeB.add(id: "batch-doc-\(i)", body: "batch content variant \(i) extra words padding")
        }
        try await storeB.index()
        for i in 0..<totalAbandoned {
            try await storeB.remove(id: "batch-doc-\(i)")
        }

        var totalRemoved = 0
        var callCount = 0
        var lastRemaining = Int.max
        while true {
            let r = try await storeB.vacuum(maxBatch: batchSize)
            totalRemoved += r.chunksRemoved
            lastRemaining = r.remainingCandidates
            callCount += 1
            if r.remainingCandidates == 0 { break }
            precondition(callCount < 100, "vacuum loop did not converge")
        }
        #expect(totalRemoved == totalAbandoned, "batched calls must not double-process or drop chunks")
        #expect(lastRemaining == 0)
        #expect(callCount >= 2, "batchSize < totalAbandoned must require more than one call")

        // First call's remainingCandidates must reflect the unprocessed backlog.
        let (storeC, _) = try await Self.makeStore(backend)
        for i in 0..<totalAbandoned {
            try await storeC.add(id: "batch-doc-\(i)", body: "batch content variant \(i) extra words padding")
        }
        try await storeC.index()
        for i in 0..<totalAbandoned {
            try await storeC.remove(id: "batch-doc-\(i)")
        }
        let firstBatch = try await storeC.vacuum(maxBatch: batchSize)
        #expect(firstBatch.chunksRemoved == batchSize)
        #expect(firstBatch.remainingCandidates == totalAbandoned - batchSize)
        try await storeC.shutdown()

        let chunkCountB = try await storageB.chunkCount()
        #expect(chunkCountA == chunkCountB, "looped and unbounded vacuum runs must converge to the same end state")

        try await storeB.shutdown()
    }

    // MARK: - 15: regression — chunks with owning documents are never removed

    @Test("a partially-indexed chunk with an owning document is never removed by vacuum()", arguments: Backend.allCases)
    func vacuumNeverRemovesPartialOrphansWithOwningDocuments(_ backend: Backend) async throws {
        let dims = 32
        let config = StoreConfig(indexer: IndexerConfig(l0Capacity: 4, lsmFanout: 2))
        let (store, storage) = try await Self.makeStore(backend, config: config, dims: dims)

        let body = "partial orphan vacuum regression content"
        let expectedTokenCount = body.split(whereSeparator: { $0.isWhitespace }).count

        try await store.add(id: "partial-doc", body: body)
        try await store.index()

        let hash = SwitchcraftStore.contentHash(body)
        guard let chunk = try await storage.chunk(hash: hash) else {
            Issue.record("chunk not found")
            return
        }
        let chunkID = chunk.id

        // Simulate partial indexing: replace all generations with one whose
        // single bucket only has 1 of the expected pairs.
        let existingGens = try await storage.generations()
        for gen in existingGens {
            try await storage.deleteGeneration(id: gen.id)
        }
        let partialGen = try await storage.insertGeneration(
            GenerationRecord(level: 0, numEmbeddings: 1, minChunkID: chunkID, maxChunkID: chunkID, created: Date())
        )
        let dummyCenter = Data(repeating: 0, count: dims * MemoryLayout<Float>.size)
        let dummyResiduals = Q4Codec.encodeResiduals([Float](repeating: 0.0, count: dims))
        _ = try await storage.insertBucket(
            BucketRecord(
                generationID: partialGen.id,
                center: dummyCenter,
                indices: IndicesCodec.encode([IndexPair(chunkID: UInt32(chunkID), tokenOffset: 0)]),
                residuals: dummyResiduals
            )
        )

        // The document still exists and still owns this chunk's hash.
        #expect(try await storage.document(uuid: "partial-doc") != nil)

        let result = try await store.vacuum()
        #expect(result.chunksRemoved == 0, "a chunk with an owning document must never be removed by vacuum(), even if partially indexed")
        #expect(try await storage.chunk(id: chunkID) != nil)

        // findOrphanedChunks() must still report it as a partial orphan —
        // vacuum() must not interfere with recovery-path visibility.
        let orphans = try await store.findOrphanedChunks()
        #expect(orphans.contains { $0.chunkID == chunkID })
        _ = expectedTokenCount
    }

    @Test("mixed corpus: true orphans, partial orphans, and abandoned chunks are each handled by the correct API", arguments: Backend.allCases)
    func vacuumMixedCorpusNoCrossContamination(_ backend: Backend) async throws {
        let dims = 32
        let config = StoreConfig(indexer: IndexerConfig(l0Capacity: 4, lsmFanout: 2))
        let (store, storage) = try await Self.makeStore(backend, config: config, dims: dims)

        // 1. Normally indexed + later abandoned (document removed).
        try await store.add(id: "abandoned-doc", body: "abandoned mixed corpus content")
        // 2. Normally indexed, kept (control).
        try await store.add(id: "kept-doc", body: "kept mixed corpus content words")
        try await store.index()
        try await store.remove(id: "abandoned-doc")

        // 3. True orphan: chunk row planted directly, with an owning document,
        //    zero bucket refs.
        let orphanBody = "true orphan mixed corpus content"
        let orphanHash = SwitchcraftStore.contentHash(orphanBody)
        let orphanTokenCount = orphanBody.split(whereSeparator: { $0.isWhitespace }).count
        _ = try await storage.upsertChunk(
            ChunkRecord(hash: orphanHash, model: "mock-embedder-v1-d32", embeddings: Data(), counts: [orphanTokenCount])
        )
        try await storage.upsertDocument(
            DocumentRecord(uuid: "orphan-owner", date: Date(), hash: orphanHash, body: orphanBody, lens: [orphanTokenCount])
        )

        let abandonedHash = SwitchcraftStore.contentHash("abandoned mixed corpus content")

        let result = try await store.vacuum()

        // Only the abandoned chunk (no owning document) was removed.
        #expect(result.chunksRemoved == 1)
        #expect(try await storage.chunk(hash: abandonedHash) == nil)

        // The true orphan (has an owning document) must survive.
        #expect(try await storage.chunk(hash: orphanHash) != nil)
        let orphansAfter = try await store.findOrphanedChunks()
        #expect(orphansAfter.contains { $0.hash == orphanHash })

        // The kept doc's chunk must survive and remain searchable.
        let hits = try await store.search(query: "kept mixed corpus content", topK: 10)
        #expect(hits.contains { $0.uuid == "kept-doc" })

        try await store.shutdown()
    }

    // MARK: - 16(g)/(h): SQLite-only PRAGMA measurement contract

    @Test("approximateDiskReclaimed tracks the freelist_count delta, and file size (page_count) is unchanged immediately after vacuum()")
    func vacuumFreelistAndFileSizeContract() async throws {
        let (dbPath, cleanup) = Self.makeTempDatabasePath()
        defer { cleanup() }

        let store = try await SwitchcraftStore.sqlite(databasePath: dbPath, embedder: MockEmbedder(dims: 32))

        // Enough abandoned chunks with enough token content that removal
        // frees at least one page's worth of residual/index bytes.
        let bigBody = (0..<40).map { "word\($0)" }.joined(separator: " ")
        for i in 0..<40 {
            try await store.add(id: "freelist-doc-\(i)", body: "\(bigBody) unique\(i)")
        }
        try await store.index()
        for i in 0..<40 {
            try await store.remove(id: "freelist-doc-\(i)")
        }

        let verifyConn = try SQLiteConnection(path: dbPath)
        func pragmaInt64(_ sql: String) throws -> Int64 {
            let stmt = try verifyConn.prepare(sql)
            guard try stmt.step() else { return 0 }
            return stmt.columnInt64(0)
        }

        let pageSize = try pragmaInt64("PRAGMA page_size")
        let pageCountBefore = try pragmaInt64("PRAGMA page_count")
        let freelistBefore = try pragmaInt64("PRAGMA freelist_count")

        let result = try await store.vacuum()
        #expect(result.chunksRemoved == 40)

        let pageCountAfter = try pragmaInt64("PRAGMA page_count")
        let freelistAfter = try pragmaInt64("PRAGMA freelist_count")

        // (h) File size (page_count × page_size) is unchanged immediately
        // after vacuum() — DELETE + wal_checkpoint(TRUNCATE) frees pages to
        // the internal free-list, it does not shrink the file.
        #expect(pageCountAfter == pageCountBefore,
                "file size must not shrink from vacuum() alone (requires a full PRAGMA vacuum)")

        // (g) approximateDiskReclaimed roughly matches the freelist delta.
        let expectedReclaimed = max(0, (freelistAfter - freelistBefore) * pageSize)
        let tolerance = pageSize * 4
        #expect(
            abs(result.approximateDiskReclaimed - expectedReclaimed) <= tolerance,
            "approximateDiskReclaimed (\(result.approximateDiskReclaimed)) should roughly match the externally observed freelist delta (\(expectedReclaimed)) within \(tolerance) bytes"
        )
        #expect(result.approximateDiskReclaimed > 0, "vacuuming 40 non-trivial abandoned chunks should free at least one page")

        try await store.shutdown()
    }

    // MARK: - 17: sustained workload (≥1,000 abandoned chunks)

    @Test("vacuum drains ≥1,000 abandoned chunks across multiple generations without ledgerOutOfSync or unbounded WAL growth")
    func vacuumSustainedWorkload() async throws {
        let (dbPath, cleanup) = Self.makeTempDatabasePath()
        let walPath = dbPath + "-wal"
        defer {
            cleanup()
            try? FileManager.default.removeItem(atPath: walPath)
        }

        let config = StoreConfig(indexer: IndexerConfig(l0Capacity: 64, lsmFanout: 4))
        let store = try await SwitchcraftStore.sqlite(
            databasePath: dbPath, embedder: MockEmbedder(dims: 32), config: config
        )

        let total = 1_200
        for i in 0..<total {
            try await store.add(id: "sustained-\(i)", body: "sustained workload document number \(i) with padding words")
            if i % 100 == 99 {
                try await store.index()
            }
        }
        try await store.index()

        for i in 0..<total {
            try await store.remove(id: "sustained-\(i)")
        }

        func walFileSize() -> Int {
            (try? FileManager.default.attributesOfItem(atPath: walPath)[.size] as? Int) ?? 0
        }

        let start = ContinuousClock.now
        var totalRemoved = 0
        var calls = 0
        var walSizesAfterEachBatch: [Int] = []
        while true {
            let r = try await store.vacuum(maxBatch: 150)
            totalRemoved += r.chunksRemoved
            calls += 1
            walSizesAfterEachBatch.append(walFileSize())
            if r.remainingCandidates == 0 { break }
            precondition(calls < 200, "sustained vacuum loop did not converge")
        }
        let elapsed = ContinuousClock.now - start

        #expect(totalRemoved == total)

        // WAL must stay bounded: since vacuum() checkpoints at the end of
        // every non-empty batch, later batches' post-checkpoint WAL size
        // should not grow unboundedly relative to the first batch's.
        if let first = walSizesAfterEachBatch.first, let last = walSizesAfterEachBatch.last {
            #expect(last <= max(first, 1) * 4, "WAL size must not grow unboundedly across sustained vacuum batches")
        }

        // No ledgerOutOfSync: a fresh add()/index()/search() cycle must work.
        try await store.add(id: "post-sustained-doc", body: "post sustained vacuum content")
        try await store.index()
        let hits = try await store.search(query: "post sustained vacuum content", topK: 5)
        #expect(hits.contains { $0.uuid == "post-sustained-doc" })

        // Document the observed throughput (not asserted, informational).
        let perChunkMicros = Double(elapsed.components.seconds) * 1_000_000
            / Double(max(totalRemoved, 1))
        _ = perChunkMicros // see stage report for the recorded figure

        try await store.shutdown()
    }
}
