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

    // MARK: - AC1/AC2 (issue #142): shared-bucket compaction must preserve
    // live chunks' `.bucketRef` ledger pointers

    /// Directly hand-plants a single bucket containing `pairs` (in that
    /// exact order) with `residualRows[i]` as pair `i`'s residual row, then
    /// constructs an `Indexer` over the resulting storage — mirroring the
    /// `LazyLedgerMaterializationTests`/`vacuumNeverRemovesPartialOrphansWithOwningDocuments`
    /// pattern of driving `Indexer` + `VacuumPlanBuilder` directly against
    /// synthetic fixtures, since k-means cluster assignment isn't
    /// controllable from the public `SwitchcraftStore` API and both AC1/AC2
    /// require an exact, deliberate bucket layout.
    private static func plantSharedBucket(
        dims: Int, pairs: [IndexPair], residualRows: [[Float]]
    ) async throws -> (storage: any SwitchcraftStorage, decodedResiduals: [Float]) {
        let storage = InMemoryStorage()
        try await storage.open()

        let chunkIDs = pairs.map { Int64($0.chunkID) }
        let gen = try await storage.insertGeneration(
            GenerationRecord(
                level: 0, numEmbeddings: pairs.count,
                minChunkID: chunkIDs.min()!, maxChunkID: chunkIDs.max()!, created: Date()
            )
        )
        let residualsEncoded = Q4Codec.encodeResiduals(residualRows.flatMap { $0 })
        _ = try await storage.insertBucket(
            BucketRecord(
                generationID: gen.id,
                center: Indexer.encodeFloat32LE([Float](repeating: 0, count: dims)),
                indices: IndicesCodec.encode(pairs),
                residuals: residualsEncoded
            )
        )
        // Golden reference: decode the exact bytes just written, matching
        // LazyLedgerMaterializationTests' pattern — Q4 is lossy, so the
        // golden value must come from the same decode path the ledger
        // itself uses, not from the pre-quantization input floats.
        return (storage, Q4Codec.decodeResiduals(residualsEncoded))
    }

    @Test("shared-bucket reproducer: a live chunk's .bucketRef resolves to its exact pre-vacuum embedding after vacuum compacts a bucket it shares with an abandoned chunk (AC1)")
    func sharedBucketLiveChunkResolvesAfterVacuum() async throws {
        let dims = 4
        let abandonedChunkID: Int64 = 100
        let liveChunkID: Int64 = 1

        // Two pairs in one bucket: abandoned first, live second. After
        // vacuum removes the abandoned pair, the live pair's offset shifts
        // from 1 (still in-range for the pre-compaction 2-pair bucket) to
        // 0 — pre-fix, the live chunk's stale ledger ref would still say
        // offset 1, which is out of range for the post-compaction 1-pair
        // bucket, throwing `bucketRefUnresolvable`.
        let pairs = [
            IndexPair(chunkID: UInt32(abandonedChunkID), tokenOffset: 0),
            IndexPair(chunkID: UInt32(liveChunkID), tokenOffset: 0),
        ]
        let residualRows: [[Float]] = [
            [0.05, -0.05, 0.1, -0.1],
            [0.2, -0.2, 0.15, -0.15],
        ]
        let (storage, decodedResiduals) = try await Self.plantSharedBucket(
            dims: dims, pairs: pairs, residualRows: residualRows
        )
        let goldenLive = Array(decodedResiduals[dims..<(2 * dims)])

        let config = IndexerConfig.testing(rehydrationConflictBehavior: .throwError)
        let indexer = try await Indexer(storage: storage, config: config)

        let gens = try await storage.generations()
        let planResult = try await VacuumPlanBuilder.buildPlan(
            abandonedChunkIDs: [abandonedChunkID],
            generations: gens,
            fetchBuckets: { genID in try await storage.buckets(forGeneration: genID) }
        )
        _ = try await storage.applyVacuumPlan(planResult.plan)
        try await indexer.applyVacuumLedgerUpdates(
            abandonedChunkIDs: [abandonedChunkID], remaps: planResult.bucketRefRemaps
        )

        let ledgerContents = try await indexer.ledgerContents()
        let liveRows = try #require(ledgerContents[liveChunkID], "live chunk must resolve, not throw bucketRefUnresolvable")
        #expect(liveRows == [goldenLive], "live chunk must resolve to the exact same embedding it referenced before vacuum ran")
        #expect(ledgerContents[abandonedChunkID] == nil, "abandoned chunk must be gone from the ledger")
    }

    @Test("adversarial silent-wrong-value regression: a live chunk's remapped .bucketRef never resolves to a different chunk's embedding after vacuum compacts a shared bucket (AC2)")
    func sharedBucketAdversarialWrongValueNeverResolves() async throws {
        let dims = 4
        let abandonedChunkID: Int64 = 100
        let live1ChunkID: Int64 = 1
        let live2ChunkID: Int64 = 2

        // Three pairs in one bucket: abandoned, live1, live2. After vacuum
        // removes the abandoned pair, live1 shifts from offset 1 -> 0 and
        // live2 shifts from offset 2 -> 1. Pre-fix, live1's stale ref
        // (still pointing at offset 1) would resolve to what is now
        // live2's row — an in-range, silently wrong embedding, never a
        // thrown error. This is the load-bearing adversarial case (AC2):
        // the fixture is built so a coincidentally-passing fix is ruled
        // out by asserting inequality with the known-wrong value, not just
        // equality with the correct one.
        let pairs = [
            IndexPair(chunkID: UInt32(abandonedChunkID), tokenOffset: 0),
            IndexPair(chunkID: UInt32(live1ChunkID), tokenOffset: 0),
            IndexPair(chunkID: UInt32(live2ChunkID), tokenOffset: 0),
        ]
        let residualRows: [[Float]] = [
            [0.05, -0.05, 0.1, -0.1],
            [0.2, -0.2, 0.15, -0.15],
            [-0.22, 0.22, -0.18, 0.18],
        ]
        let (storage, decodedResiduals) = try await Self.plantSharedBucket(
            dims: dims, pairs: pairs, residualRows: residualRows
        )
        let goldenLive1 = Array(decodedResiduals[dims..<(2 * dims)])
        let goldenLive2 = Array(decodedResiduals[(2 * dims)..<(3 * dims)])
        #expect(goldenLive1 != goldenLive2, "fixture rows must be distinguishable or the adversarial assertion below proves nothing")

        let config = IndexerConfig.testing(rehydrationConflictBehavior: .throwError)
        let indexer = try await Indexer(storage: storage, config: config)

        let gens = try await storage.generations()
        let planResult = try await VacuumPlanBuilder.buildPlan(
            abandonedChunkIDs: [abandonedChunkID],
            generations: gens,
            fetchBuckets: { genID in try await storage.buckets(forGeneration: genID) }
        )
        _ = try await storage.applyVacuumPlan(planResult.plan)
        try await indexer.applyVacuumLedgerUpdates(
            abandonedChunkIDs: [abandonedChunkID], remaps: planResult.bucketRefRemaps
        )

        let ledgerContents = try await indexer.ledgerContents()
        let live1Rows = try #require(ledgerContents[live1ChunkID])
        let live2Rows = try #require(ledgerContents[live2ChunkID])

        #expect(live1Rows == [goldenLive1], "live1 must resolve to its own embedding")
        #expect(live1Rows != [goldenLive2], "live1 must not silently resolve to live2's embedding (the pre-fix bug)")
        #expect(live2Rows == [goldenLive2], "live2 must resolve to its own embedding")
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

    // MARK: - AC3 (issue #142): sustained shared-bucket workload

    /// Unlike AC1/AC2 (which hand-plant an exact bucket layout to prove the
    /// remap mechanism directly), this drives the real end-to-end
    /// `SwitchcraftStore.vacuum()` orchestration path repeatedly, since
    /// sustained-cycle draining is about that real path staying consistent
    /// over many iterations, not about a single deterministic bucket
    /// layout. A small `l0Capacity`/`lsmFanout` forces frequent cascades,
    /// merging each cohort's chunks together with more-recently-added
    /// still-live chunks' — since buckets are k-means clusters, not
    /// chunkID-range partitions, this reliably produces buckets shared
    /// across live and abandoned chunks without needing to control k-means
    /// assignment directly.
    ///
    /// Critically, a stale/corrupt ledger `.bucketRef` left behind by a
    /// buggy vacuum is only ever *resolved* (and so only ever surfaces,
    /// via `Indexer.resolveBucketRefRow`) the next time a cascade merges
    /// that specific chunk's generation back in — `SwitchcraftStorage`-level
    /// counts (`chunksRemoved`/`remainingCandidates`) and `search()` (which
    /// reads bucket blobs directly, bypassing the ledger entirely) can't
    /// observe it. So each cycle forces one more such cascade — a batch
    /// large enough to merge every still-live generation back together —
    /// immediately after vacuum(), guaranteeing every surviving chunk's
    /// bucket-ref gets re-resolved while its cohort neighbors are still
    /// fresh from being vacuumed. Reproduces the issue's observed symptom
    /// ("18 of 20 abandoned chunks stuck", `bucketRefUnresolvable` thrown
    /// from `resolveBucketRefRow`) as a converged-to-zero regression: every
    /// cycle's `vacuum()` must fully drain the cohort it just abandoned,
    /// and the forced cascade immediately after must not throw.
    @Test("sustained workload: 20+ vacuum cycles against a corpus with buckets shared across live and abandoned chunks each fully drain, with no bucketRefUnresolvable and no ledgerOutOfSync (AC3)")
    func vacuumSustainedSharedBucketWorkloadFullyDrains() async throws {
        let config = StoreConfig(indexer: IndexerConfig(l0Capacity: 4, lsmFanout: 2))
        let (store, _) = try await Self.makeStore(.inMemory, config: config)

        let cycles = 25
        let cohortSize = 4
        var cohorts: [[String]] = []

        for cycle in 0..<cycles {
            let ids = (0..<cohortSize).map { "shared-cycle\(cycle)-doc\($0)" }
            for (j, id) in ids.enumerated() {
                try await store.add(id: id, body: "shared bucket workload cycle \(cycle) variant \(j) document content padding words")
            }
            try await store.index()
            cohorts.append(ids)

            // Abandon a cohort from a few cycles back — by now, later
            // cascades have very likely merged its chunks' buckets
            // together with still-live chunks' from more recent cohorts.
            if cycle >= 3 {
                let toAbandon = cohorts[cycle - 3]
                for id in toAbandon {
                    try await store.remove(id: id)
                }
                let result = try await store.vacuum()
                #expect(result.chunksRemoved == cohortSize, "cycle \(cycle) must fully drain its just-abandoned cohort in one call")
                #expect(result.remainingCandidates == 0, "no abandoned chunk may be left stuck across cycles (the issue's observed symptom)")

                // Force a cascade wide enough to merge every still-live
                // generation back together, so every surviving chunk's
                // (possibly just-remapped) bucket-ref gets re-resolved
                // right away rather than staying dormant at a high LSM
                // level where this bug would otherwise hide indefinitely.
                for k in 0..<20 {
                    try await store.add(id: "cascade-force-\(cycle)-\(k)", body: "cascade force cycle \(cycle) item \(k) padding words here")
                }
                try await store.index()
            }
        }

        // Drain the final cohorts not yet abandoned by the loop above,
        // leaving the store fully vacuumed.
        for ids in cohorts.suffix(3) {
            for id in ids {
                try await store.remove(id: id)
            }
        }
        let finalResult = try await store.vacuum()
        #expect(finalResult.remainingCandidates == 0)

        // No ledgerOutOfSync / no lingering corruption: a fresh
        // add()/index()/search() cycle must still work after 20+
        // shared-bucket vacuum cycles.
        try await store.add(id: "post-sustained-shared-doc", body: "post sustained shared bucket vacuum content")
        try await store.index()
        let hits = try await store.search(query: "post sustained shared bucket vacuum", topK: 5)
        #expect(hits.contains { $0.uuid == "post-sustained-shared-doc" })

        try await store.shutdown()
    }
}
