// SPDX-License-Identifier: Apache-2.0
import Foundation
import SwitchcraftCore
import Testing

/// A reusable conformance suite for `SwitchcraftStorage` implementations.
///
/// Backends import this module and run `StorageConformance.runAll(makeStorage:)`
/// from a `@Test` function. Every assertion uses Swift Testing's `#expect`
/// macros so failures attribute to the calling test.
///
/// Usage:
/// ```swift
/// @Test func myBackendIsCompliant() async throws {
///     try await StorageConformance.runAll {
///         MyBackend(path: ":memory:")
///     }
/// }
/// ```
public enum StorageConformance {

    /// Run every conformance check against a fresh storage instance produced
    /// by `makeStorage`. The closure is called once at the start of the run;
    /// the suite performs `clear()` between scenarios so a single instance
    /// is reused.
    public static func runAll(
        makeStorage: () async throws -> any SwitchcraftStorage
    ) async throws {
        let storage = try await makeStorage()
        try await storage.open()

        try await runDocumentRoundTrip(storage)
        try await runDocumentUpsertReplaces(storage)
        try await runDocumentDelete(storage)
        try await runDocumentFilters(storage)
        try await runDocumentsForChunkHash(storage)

        try await runChunkInsertAndLookup(storage)
        try await runChunkDedupByHash(storage)

        try await runGenerationLifecycle(storage)
        try await runBucketLifecycle(storage)
        try await runBucketScanLifecycle(storage)
        try await runBucketsByIDs(storage)
        try await runBucketsByIDsExceedsBatchSize(storage)
        try await runDeleteGenerationCascadesToBuckets(storage)
        try await runReplaceGeneration(storage)
        try await runUpdateGenerationEmbeddingCount(storage)

        try await runClearEmptiesEverything(storage)

        try await runFullTextRetrieval(storage)
        try await runFullTextRespectsFilter(storage)
        try await runFullTextDeterministicTieBreak(storage)
        try await runFullTextHandlesPunctuation(storage)
        try await runFullTextTitleWeighting(storage)

        try await runAllChunks(storage)
        try await runChunkBucketRefCount(storage)

        try await runDocumentHashes(storage)
        try await runApplyVacuumPlan(storage)
        try await runFreeListByteCount(storage)

        try await runSnapshotLifecycle(storage)

        try await storage.close()
    }

    // MARK: - Documents

    static func runDocumentRoundTrip(_ storage: any SwitchcraftStorage) async throws {
        try await storage.clear()
        let doc = makeDocument(uuid: "a", body: "hello world")
        try await storage.upsertDocument(doc)

        let fetched = try await storage.document(uuid: "a")
        #expect(fetched == doc)
        #expect(try await storage.documentCount() == 1)
    }

    static func runDocumentUpsertReplaces(_ storage: any SwitchcraftStorage) async throws {
        try await storage.clear()
        try await storage.upsertDocument(makeDocument(uuid: "a", body: "v1"))
        try await storage.upsertDocument(makeDocument(uuid: "a", body: "v2"))

        let fetched = try await storage.document(uuid: "a")
        #expect(fetched?.body == "v2")
        #expect(try await storage.documentCount() == 1)
    }

    static func runDocumentDelete(_ storage: any SwitchcraftStorage) async throws {
        try await storage.clear()
        try await storage.upsertDocument(makeDocument(uuid: "a", body: "x"))
        try await storage.deleteDocument(uuid: "a")
        #expect(try await storage.document(uuid: "a") == nil)
        #expect(try await storage.documentCount() == 0)

        // Deleting a missing document is a no-op, not an error.
        try await storage.deleteDocument(uuid: "missing")
    }

    static func runDocumentFilters(_ storage: any SwitchcraftStorage) async throws {
        try await storage.clear()
        let early = makeDocument(
            uuid: "early",
            body: "early text",
            date: Date(timeIntervalSince1970: 1_000_000),
            metadata: ["tag": "alpha"]
        )
        let late = makeDocument(
            uuid: "late",
            body: "late text",
            date: Date(timeIntervalSince1970: 2_000_000),
            metadata: ["tag": "beta"]
        )
        try await storage.upsertDocument(early)
        try await storage.upsertDocument(late)

        let all = try await storage.documents(matching: .all)
        #expect(Set(all.map(\.uuid)) == ["early", "late"])

        let onlyLate = try await storage.documents(matching: .uuidIn(["late"]))
        #expect(onlyLate.map(\.uuid) == ["late"])

        let beforeMid = try await storage.documents(
            matching: .dateRange(start: nil, end: Date(timeIntervalSince1970: 1_500_000))
        )
        #expect(beforeMid.map(\.uuid) == ["early"])

        let onlyAlpha = try await storage.documents(
            matching: .metadataEquals(key: "tag", value: "alpha")
        )
        #expect(onlyAlpha.map(\.uuid) == ["early"])

        let notAlpha = try await storage.documents(
            matching: .not(.metadataEquals(key: "tag", value: "alpha"))
        )
        #expect(notAlpha.map(\.uuid) == ["late"])
    }

    static func runDocumentsForChunkHash(_ storage: any SwitchcraftStorage) async throws {
        try await storage.clear()

        // Empty store: no matches.
        let none = try await storage.documents(forChunkHash: "missing")
        #expect(none.isEmpty)

        // Single document with hash "h-a".
        let solo = makeDocument(uuid: "solo", body: "x")
        try await storage.upsertDocument(solo)
        let onlySolo = try await storage.documents(forChunkHash: solo.hash)
        #expect(onlySolo.map(\.uuid) == ["solo"])

        // Two documents sharing a chunk hash both come back.
        var shared1 = makeDocument(uuid: "shared1", body: "x")
        var shared2 = makeDocument(uuid: "shared2", body: "y")
        shared1.hash = "shared-hash"
        shared2.hash = "shared-hash"
        try await storage.upsertDocument(shared1)
        try await storage.upsertDocument(shared2)
        let shared = try await storage.documents(forChunkHash: "shared-hash")
        #expect(Set(shared.map(\.uuid)) == ["shared1", "shared2"])

        // Deleting one of the sharers removes only that one.
        try await storage.deleteDocument(uuid: "shared1")
        let remaining = try await storage.documents(forChunkHash: "shared-hash")
        #expect(remaining.map(\.uuid) == ["shared2"])

        // Unknown hash still returns empty.
        let stillNone = try await storage.documents(forChunkHash: "no-such-hash")
        #expect(stillNone.isEmpty)
    }

    // MARK: - Chunks

    static func runChunkInsertAndLookup(_ storage: any SwitchcraftStorage) async throws {
        try await storage.clear()
        let inserted = try await storage.upsertChunk(
            ChunkRecord(hash: "h1", model: "xtr", embeddings: Data([0x01, 0x02]), counts: [3])
        )
        #expect(inserted.id != ChunkRecord.unassigned)
        #expect(inserted.hash == "h1")

        let byHash = try await storage.chunk(hash: "h1")
        #expect(byHash == inserted)

        let byID = try await storage.chunk(id: inserted.id)
        #expect(byID == inserted)

        #expect(try await storage.chunk(hash: "missing") == nil)
        #expect(try await storage.chunkCount() == 1)
    }

    static func runChunkDedupByHash(_ storage: any SwitchcraftStorage) async throws {
        try await storage.clear()
        let first = try await storage.upsertChunk(
            ChunkRecord(hash: "h1", model: "xtr", embeddings: Data([0x01]))
        )
        let second = try await storage.upsertChunk(
            ChunkRecord(hash: "h1", model: "xtr", embeddings: Data([0x99]))
        )
        #expect(first.id == second.id)
        #expect(first.embeddings == second.embeddings)
        #expect(try await storage.chunkCount() == 1)
    }

    // MARK: - Generations & Buckets

    static func runGenerationLifecycle(_ storage: any SwitchcraftStorage) async throws {
        try await storage.clear()

        let g1 = try await storage.insertGeneration(
            GenerationRecord(level: 0, numEmbeddings: 100, minChunkID: 1, maxChunkID: 10, created: Date())
        )
        let g2 = try await storage.insertGeneration(
            GenerationRecord(level: 1, numEmbeddings: 500, minChunkID: 1, maxChunkID: 50, created: Date())
        )
        #expect(g1.id != GenerationRecord.unassigned)
        #expect(g2.id > g1.id)

        let listed = try await storage.generations()
        #expect(listed.map(\.id) == [g1.id, g2.id])

        try await storage.deleteGeneration(id: g1.id)
        let remaining = try await storage.generations()
        #expect(remaining.map(\.id) == [g2.id])

        // Deleting a missing generation is a no-op.
        try await storage.deleteGeneration(id: 9999)
    }

    static func runBucketLifecycle(_ storage: any SwitchcraftStorage) async throws {
        try await storage.clear()

        let gen = try await storage.insertGeneration(
            GenerationRecord(level: 0, numEmbeddings: 0, minChunkID: 0, maxChunkID: 0, created: Date())
        )

        let b1 = try await storage.insertBucket(
            BucketRecord(generationID: gen.id, center: Data([0x01]), indices: Data([0x02]), residuals: Data([0x03]))
        )
        let b2 = try await storage.insertBucket(
            BucketRecord(generationID: gen.id, center: Data([0x04]), indices: Data([0x05]), residuals: Data([0x06]))
        )
        #expect(b1.id != BucketRecord.unassigned)
        #expect(b2.id != b1.id)

        let buckets = try await storage.buckets(forGeneration: gen.id)
        #expect(buckets.map(\.id) == [b1.id, b2.id])

        let none = try await storage.buckets(forGeneration: 9999)
        #expect(none.isEmpty)
    }

    /// Covers the scan-shaped query added for issue #151 / ADR 041:
    /// `id ASC` order matches `buckets(forGeneration:)` exactly (load-bearing
    /// for `cblas_sgemm` summation order), `center` round-trips byte-for-byte,
    /// and `residualByteCount` matches the full record's `residuals.count`
    /// without the backend ever materializing the `residuals`/`indices`
    /// payload to compute it.
    static func runBucketScanLifecycle(_ storage: any SwitchcraftStorage) async throws {
        try await storage.clear()

        let gen = try await storage.insertGeneration(
            GenerationRecord(level: 0, numEmbeddings: 0, minChunkID: 0, maxChunkID: 0, created: Date())
        )

        let b1 = try await storage.insertBucket(
            BucketRecord(
                generationID: gen.id,
                center: Data([0x01, 0x02]),
                indices: Data([0xAA]),
                residuals: Data([0x03, 0x04, 0x05])
            )
        )
        let b2 = try await storage.insertBucket(
            BucketRecord(
                generationID: gen.id,
                center: Data([0x06, 0x07]),
                indices: Data([0xBB, 0xCC]),
                residuals: Data([0x08])
            )
        )

        let scanned = try await storage.scanBuckets(forGeneration: gen.id)
        #expect(
            scanned.map(\.id) == [b1.id, b2.id],
            "scan order must match id ASC, same as buckets(forGeneration:)"
        )
        #expect(scanned[0].center == b1.center)
        #expect(scanned[0].residualByteCount == b1.residuals.count)
        #expect(scanned[1].center == b2.center)
        #expect(scanned[1].residualByteCount == b2.residuals.count)

        let emptyGen = try await storage.scanBuckets(forGeneration: 9999)
        #expect(emptyGen.isEmpty)
    }

    /// Covers `buckets(ids:)` added for issue #151 / ADR 041: only the
    /// requested ids come back (with full `center`/`indices`/`residuals`),
    /// ids may span multiple generations in one call, missing ids are
    /// silently omitted rather than causing an error, and an empty request
    /// returns an empty result.
    static func runBucketsByIDs(_ storage: any SwitchcraftStorage) async throws {
        try await storage.clear()

        #expect(try await storage.buckets(ids: []).isEmpty)

        let gen1 = try await storage.insertGeneration(
            GenerationRecord(level: 0, numEmbeddings: 0, minChunkID: 0, maxChunkID: 0, created: Date())
        )
        let gen2 = try await storage.insertGeneration(
            GenerationRecord(level: 1, numEmbeddings: 0, minChunkID: 0, maxChunkID: 0, created: Date())
        )

        let b1 = try await storage.insertBucket(
            BucketRecord(generationID: gen1.id, center: Data([0x01]), indices: Data([0x02]), residuals: Data([0x03]))
        )
        let b2 = try await storage.insertBucket(
            BucketRecord(generationID: gen1.id, center: Data([0x04]), indices: Data([0x05]), residuals: Data([0x06]))
        )
        let b3 = try await storage.insertBucket(
            BucketRecord(generationID: gen2.id, center: Data([0x07]), indices: Data([0x08]), residuals: Data([0x09]))
        )

        // Exact requested ids only, spanning multiple generations.
        let fetched = try await storage.buckets(ids: [b1.id, b3.id])
        #expect(fetched.count == 2)
        let byID = Dictionary(uniqueKeysWithValues: fetched.map { ($0.id, $0) })
        #expect(byID[b1.id]?.center == b1.center)
        #expect(byID[b1.id]?.indices == b1.indices)
        #expect(byID[b1.id]?.residuals == b1.residuals)
        #expect(byID[b3.id]?.generationID == gen2.id)

        // b2 was not requested and must not appear.
        #expect(!fetched.map(\.id).contains(b2.id))

        // Missing ids are silently omitted, not an error.
        let withMissing = try await storage.buckets(ids: [b1.id, 987_654_321])
        #expect(withMissing.map(\.id) == [b1.id])

        let allMissing = try await storage.buckets(ids: [987_654_321, 987_654_322])
        #expect(allMissing.isEmpty)
    }

    /// `SQLiteStorage`'s `buckets(ids:)` batches its `IN (...)` fetch in
    /// chunks (ADR 041) to stay under SQLite's bound-parameter limit. This
    /// exercises a request that spans more than one batch, so every backend
    /// must reassemble results across the boundary correctly.
    static func runBucketsByIDsExceedsBatchSize(_ storage: any SwitchcraftStorage) async throws {
        try await storage.clear()

        let gen = try await storage.insertGeneration(
            GenerationRecord(level: 0, numEmbeddings: 0, minChunkID: 0, maxChunkID: 0, created: Date())
        )

        var ids: [Int64] = []
        ids.reserveCapacity(550)
        for i in 0..<550 {
            let b = try await storage.insertBucket(
                BucketRecord(
                    generationID: gen.id,
                    center: Data([UInt8(i % 256)]),
                    indices: Data(),
                    residuals: Data()
                )
            )
            ids.append(b.id)
        }

        let fetched = try await storage.buckets(ids: ids)
        #expect(
            Set(fetched.map(\.id)) == Set(ids),
            "fetch must return every requested id even across a >500-id batch boundary"
        )
        #expect(fetched.count == 550)
    }

    static func runDeleteGenerationCascadesToBuckets(_ storage: any SwitchcraftStorage) async throws {
        try await storage.clear()
        let gen = try await storage.insertGeneration(
            GenerationRecord(level: 0, numEmbeddings: 0, minChunkID: 0, maxChunkID: 0, created: Date())
        )
        _ = try await storage.insertBucket(
            BucketRecord(generationID: gen.id, center: Data(), indices: Data(), residuals: Data())
        )
        try await storage.deleteGeneration(id: gen.id)

        let buckets = try await storage.buckets(forGeneration: gen.id)
        #expect(buckets.isEmpty)
    }

    static func runReplaceGeneration(_ storage: any SwitchcraftStorage) async throws {
        // Scenario 1: Partial prune — keep one of two buckets in the loser gen.
        try await storage.clear()
        let loser = try await storage.insertGeneration(
            GenerationRecord(level: 0, numEmbeddings: 10, minChunkID: 1, maxChunkID: 5, created: Date())
        )
        let bucketA = try await storage.insertBucket(
            BucketRecord(generationID: loser.id, center: Data([0xAA]), indices: Data([0x01]), residuals: Data([0x02]))
        )
        _ = try await storage.insertBucket(
            BucketRecord(generationID: loser.id, center: Data([0xBB]), indices: Data([0x03]), residuals: Data([0x04]))
        )

        // Surviving record: same level/created as loser, but updated stats.
        let survivingRecord = GenerationRecord(
            level: loser.level,
            numEmbeddings: 5,
            minChunkID: 1,
            maxChunkID: 3,
            created: loser.created
        )
        // Keep only bucketA's data as a surviving bucket.
        let survivingBucket = BucketRecord(
            generationID: BucketRecord.unassigned,
            center: bucketA.center,
            indices: bucketA.indices,
            residuals: bucketA.residuals
        )
        let newGen = try await storage.replaceGeneration(
            losingGenerationID: loser.id,
            survivingRecord: survivingRecord,
            survivingBuckets: [survivingBucket]
        )
        #expect(newGen != nil)
        #expect(newGen!.id != loser.id)
        #expect(newGen!.id != GenerationRecord.unassigned)
        #expect(newGen!.numEmbeddings == 5)

        // Old generation must be gone.
        let allGens = try await storage.generations()
        #expect(!allGens.map(\.id).contains(loser.id))

        // New generation must have exactly 1 bucket with bucketA's data.
        let newBuckets = try await storage.buckets(forGeneration: newGen!.id)
        #expect(newBuckets.count == 1)
        #expect(newBuckets[0].center == bucketA.center)
        #expect(newBuckets[0].indices == bucketA.indices)
        #expect(newBuckets[0].residuals == bucketA.residuals)

        // Scenario 2: Full prune — no survivors.
        try await storage.clear()
        let loser2 = try await storage.insertGeneration(
            GenerationRecord(level: 0, numEmbeddings: 5, minChunkID: 1, maxChunkID: 2, created: Date())
        )
        _ = try await storage.insertBucket(
            BucketRecord(generationID: loser2.id, center: Data([0xCC]), indices: Data([0x05]), residuals: Data([0x06]))
        )
        let nilResult = try await storage.replaceGeneration(
            losingGenerationID: loser2.id,
            survivingRecord: GenerationRecord(level: 0, numEmbeddings: 0, minChunkID: 0, maxChunkID: 0, created: Date()),
            survivingBuckets: []
        )
        #expect(nilResult == nil)

        let allGens2 = try await storage.generations()
        #expect(!allGens2.map(\.id).contains(loser2.id))

        let leftoverBuckets = try await storage.buckets(forGeneration: loser2.id)
        #expect(leftoverBuckets.isEmpty)
    }

    static func runUpdateGenerationEmbeddingCount(_ storage: any SwitchcraftStorage) async throws {
        try await storage.clear()

        let gen = try await storage.insertGeneration(
            GenerationRecord(level: 0, numEmbeddings: 10, minChunkID: 1, maxChunkID: 5, created: Date())
        )
        #expect(gen.numEmbeddings == 10)

        try await storage.updateGenerationEmbeddingCount(id: gen.id, count: 5)

        let updated = try await storage.generations()
        let found = updated.first { $0.id == gen.id }
        #expect(found != nil)
        #expect(found?.numEmbeddings == 5)
        #expect(found?.id == gen.id)

        // No-op for an absent id — must not throw.
        try await storage.updateGenerationEmbeddingCount(id: 9999, count: 0)
    }

    // MARK: - Clear

    static func runClearEmptiesEverything(_ storage: any SwitchcraftStorage) async throws {
        try await storage.clear()
        try await storage.upsertDocument(makeDocument(uuid: "a", body: "x"))
        _ = try await storage.upsertChunk(ChunkRecord(hash: "h", model: "m", embeddings: Data()))
        let gen = try await storage.insertGeneration(
            GenerationRecord(level: 0, numEmbeddings: 0, minChunkID: 0, maxChunkID: 0, created: Date())
        )
        _ = try await storage.insertBucket(
            BucketRecord(generationID: gen.id, center: Data(), indices: Data(), residuals: Data())
        )

        try await storage.clear()
        #expect(try await storage.documentCount() == 0)
        #expect(try await storage.chunkCount() == 0)
        #expect(try await storage.generations().isEmpty)
        #expect(try await storage.buckets(forGeneration: gen.id).isEmpty)
    }

    // MARK: - Full-text Search

    static func runFullTextRetrieval(_ storage: any SwitchcraftStorage) async throws {
        try await storage.clear()
        try await storage.upsertDocument(makeDocument(uuid: "fruit", body: "apples and bananas are fruit"))
        try await storage.upsertDocument(makeDocument(uuid: "veg",   body: "carrots and broccoli are vegetables"))

        let hits = try await storage.searchFullText(query: "bananas", limit: 10, filter: .all)
        #expect(hits.first?.uuid == "fruit")
        #expect(hits.contains { $0.uuid == "veg" } == false)
    }

    /// FTS results with tied scores must be ordered by uuid ascending so
    /// hybrid fusion is deterministic across runs (ADR 008).
    static func runFullTextDeterministicTieBreak(_ storage: any SwitchcraftStorage) async throws {
        try await storage.clear()
        // Three documents with bodies that all match "common" once.
        // BM25 scores will be identical (or near-identical), and the
        // in-memory backend's overlap score is exactly equal.
        try await storage.upsertDocument(makeDocument(uuid: "c-zeta",  body: "common term zeta"))
        try await storage.upsertDocument(makeDocument(uuid: "a-alpha", body: "common term alpha"))
        try await storage.upsertDocument(makeDocument(uuid: "b-beta",  body: "common term beta"))

        let first  = try await storage.searchFullText(query: "common", limit: 10, filter: .all)
        let second = try await storage.searchFullText(query: "common", limit: 10, filter: .all)

        // (a) Two consecutive runs must produce byte-identical ordering.
        #expect(first.map(\.uuid) == second.map(\.uuid))
        #expect(Set(first.map(\.uuid)) == ["a-alpha", "b-beta", "c-zeta"])

        // (b) Within any group of equal scores, uuids must be ascending.
        //     We don't require a globally uuid-sorted list because some
        //     backends (e.g. SQLite's `bm25()`) can return scores that
        //     differ by a tiny amount across SQLite/FTS5 versions even
        //     for symmetric documents. The contract is "uuid ascending
        //     under tied scores", not "uuid ascending unconditionally".
        let epsilon: Float = 1e-6
        var i = 0
        while i < first.count {
            var j = i + 1
            while j < first.count && abs(first[j].score - first[i].score) <= epsilon {
                j += 1
            }
            let groupUuids = first[i..<j].map(\.uuid)
            #expect(groupUuids == groupUuids.sorted())
            i = j
        }
    }

    /// Natural-language queries with punctuation (periods, commas, quotes,
    /// parentheses, exclamation marks) must not be passed through to the
    /// FTS backend's query grammar verbatim. Backends that delegate to
    /// FTS5 used to raise `SQLITE_ERROR: fts5: syntax error near "."` on
    /// input like `"I picked some apples."` — see issue #31. This suite
    /// asserts that every backend tolerates such inputs without throwing.
    /// Because FTS5's default tokenizer treats space-separated terms as an
    /// implicit AND, this is a smoke test only; ranked-quality assertions
    /// for multi-term queries belong in higher-level hybrid-search tests.
    static func runFullTextHandlesPunctuation(_ storage: any SwitchcraftStorage) async throws {
        try await storage.clear()
        try await storage.upsertDocument(
            makeDocument(uuid: "fruit", body: "apples and bananas are fruit")
        )

        let punctuationQueries = [
            "I picked some red apples from the orchard.",
            "apples; bananas, and pears.",
            "(apples) \"and\" bananas?",
            "Are you serious?!?",
            "what about \"phrase\" search?",
        ]
        for query in punctuationQueries {
            // Either returns hits or doesn't; the contract is "must not throw".
            _ = try await storage.searchFullText(query: query, limit: 10, filter: .all)
        }
    }

    static func runFullTextRespectsFilter(_ storage: any SwitchcraftStorage) async throws {
        try await storage.clear()
        try await storage.upsertDocument(
            makeDocument(uuid: "a", body: "the quick brown fox", metadata: ["lang": "en"])
        )
        try await storage.upsertDocument(
            makeDocument(uuid: "b", body: "the quick green frog", metadata: ["lang": "en"])
        )

        let onlyA = try await storage.searchFullText(
            query: "quick", limit: 10, filter: .uuidIn(["a"])
        )
        #expect(onlyA.map(\.uuid) == ["a"])
    }

    // MARK: - Full-text title weighting

    static func runFullTextTitleWeighting(_ storage: any SwitchcraftStorage) async throws {
        try await storage.clear()

        // Document with title "Bartleby" but generic body (mirrors the production failure case).
        let titled = makeDocument(uuid: "bartleby-com", body: "homework help literature", title: "Bartleby")
        // Document without a title whose body mentions "bartleby".
        let bodyMatch = makeDocument(uuid: "bartleby-body", body: "bartleby the scrivener melville")

        try await storage.upsertDocument(titled)
        try await storage.upsertDocument(bodyMatch)

        // Round-trip: title must be stored and returned.
        let fetched = try await storage.document(uuid: "bartleby-com")
        #expect(fetched?.title == "Bartleby")

        // FTS for "bartleby" must surface the titled document (rank ≤ 2).
        let hits = try await storage.searchFullText(query: "bartleby", limit: 10, filter: .all)
        let uuids = hits.map(\.uuid)
        #expect(uuids.contains("bartleby-com"), "titled document must appear in FTS results")
        if let idx = uuids.firstIndex(of: "bartleby-com") {
            #expect(idx < 2, "titled document must rank in top 2")
        }
    }

    // MARK: - allChunks / chunkBucketRefCount

    static func runAllChunks(_ storage: any SwitchcraftStorage) async throws {
        try await storage.clear()

        let c1 = try await storage.upsertChunk(
            ChunkRecord(hash: "all-h1", model: "m1", embeddings: Data(), counts: [3])
        )
        let c2 = try await storage.upsertChunk(
            ChunkRecord(hash: "all-h2", model: "m2", embeddings: Data(), counts: [5])
        )

        let all = try await storage.allChunks()
        let ids = Set(all.map(\.id))
        #expect(ids.contains(c1.id))
        #expect(ids.contains(c2.id))
        #expect(all.count >= 2)

        let hashes = Set(all.map(\.hash))
        #expect(hashes.contains("all-h1"))
        #expect(hashes.contains("all-h2"))
    }

    static func runChunkBucketRefCount(_ storage: any SwitchcraftStorage) async throws {
        try await storage.clear()

        // Chunk with no bucket assignment — orphan.
        let orphan = try await storage.upsertChunk(
            ChunkRecord(hash: "bpa-orphan", model: "m", embeddings: Data(), counts: [1])
        )

        // Chunk with one bucket assignment.
        let indexed = try await storage.upsertChunk(
            ChunkRecord(hash: "bpa-indexed", model: "m", embeddings: Data(), counts: [1])
        )
        let gen = try await storage.insertGeneration(
            GenerationRecord(
                level: 0,
                numEmbeddings: 1,
                minChunkID: indexed.id,
                maxChunkID: indexed.id,
                created: Date()
            )
        )
        let indicesBlob = IndicesCodec.encode([
            IndexPair(chunkID: UInt32(indexed.id), tokenOffset: 0)
        ])
        _ = try await storage.insertBucket(
            BucketRecord(
                generationID: gen.id,
                center: Data(repeating: 0, count: 4),
                indices: indicesBlob,
                residuals: Data()
            )
        )

        let orphanCount = try await storage.chunkBucketRefCount(orphan.id)
        let indexedCount = try await storage.chunkBucketRefCount(indexed.id)
        #expect(orphanCount == 0, "orphan chunk must have 0 bucket ref count")
        #expect(indexedCount >= 1, "indexed chunk must have at least 1 bucket ref count")
    }

    // MARK: - documentHashes / applyVacuumPlan / freeListByteCount

    static func runDocumentHashes(_ storage: any SwitchcraftStorage) async throws {
        try await storage.clear()
        #expect(try await storage.documentHashes().isEmpty)

        try await storage.upsertDocument(makeDocument(uuid: "dh-a", body: "x"))
        try await storage.upsertDocument(makeDocument(uuid: "dh-b", body: "y"))
        let hashes = try await storage.documentHashes()
        #expect(hashes == ["hash-dh-a", "hash-dh-b"])

        try await storage.deleteDocument(uuid: "dh-a")
        let afterDelete = try await storage.documentHashes()
        #expect(afterDelete == ["hash-dh-b"])
    }

    static func runApplyVacuumPlan(_ storage: any SwitchcraftStorage) async throws {
        try await storage.clear()

        // Insert chunks first to get real backend-assigned ids, matching
        // the pattern already used by runChunkBucketRefCount.
        let chunkA = try await storage.upsertChunk(
            ChunkRecord(hash: "vac-a", model: "m", embeddings: Data(), counts: [1])
        )
        let chunkB = try await storage.upsertChunk(
            ChunkRecord(hash: "vac-b", model: "m", embeddings: Data(), counts: [1])
        )
        let chunkC = try await storage.upsertChunk(
            ChunkRecord(hash: "vac-c", model: "m", embeddings: Data(), counts: [1])
        )
        let chunkD = try await storage.upsertChunk(
            ChunkRecord(hash: "vac-d", model: "m", embeddings: Data(), counts: [1])
        )
        let chunkE = try await storage.upsertChunk(
            ChunkRecord(hash: "vac-e", model: "m", embeddings: Data(), counts: [1])
        )

        // Only chunkE keeps an owning document; the rest are abandoned.
        try await storage.upsertDocument(
            DocumentRecord(uuid: "vac-owner", date: Date(), hash: "vac-e", body: "x", lens: [1])
        )

        let dims = 4
        func onePairBucket(generationID: Int64, chunkID: Int64) -> BucketRecord {
            BucketRecord(
                generationID: generationID,
                center: Data(repeating: 0, count: dims * 4),
                indices: IndicesCodec.encode([IndexPair(chunkID: UInt32(chunkID), tokenOffset: 0)]),
                residuals: Q4Codec.encodeResiduals([Float](repeating: 0, count: dims))
            )
        }
        func twoPairBucket(generationID: Int64, chunkID1: Int64, chunkID2: Int64) -> BucketRecord {
            BucketRecord(
                generationID: generationID,
                center: Data(repeating: 0, count: dims * 4),
                indices: IndicesCodec.encode([
                    IndexPair(chunkID: UInt32(chunkID1), tokenOffset: 0),
                    IndexPair(chunkID: UInt32(chunkID2), tokenOffset: 0),
                ]),
                residuals: Q4Codec.encodeResiduals([Float](repeating: 0, count: dims * 2))
            )
        }

        // Generation A: one bucket holding chunkA (to remove) and chunkE
        // (survives) — bucket update, generation survives with corrected count.
        let genA = try await storage.insertGeneration(
            GenerationRecord(level: 0, numEmbeddings: 2, minChunkID: min(chunkA.id, chunkE.id), maxChunkID: max(chunkA.id, chunkE.id), created: Date())
        )
        let bucketA = try await storage.insertBucket(
            twoPairBucket(generationID: genA.id, chunkID1: chunkA.id, chunkID2: chunkE.id)
        )

        // Generation B: two buckets — bucketB1 holds only chunkB (to remove,
        // bucket becomes empty and is deleted outright); bucketB2 holds
        // chunkD (survives) — generation survives with one bucket gone.
        let genB = try await storage.insertGeneration(
            GenerationRecord(level: 0, numEmbeddings: 2, minChunkID: min(chunkB.id, chunkD.id), maxChunkID: max(chunkB.id, chunkD.id), created: Date())
        )
        let bucketB1 = try await storage.insertBucket(onePairBucket(generationID: genB.id, chunkID: chunkB.id))
        let bucketB2 = try await storage.insertBucket(onePairBucket(generationID: genB.id, chunkID: chunkD.id))

        // Generation C: single bucket holding only chunkC (to remove) — the
        // whole generation is emptied and must be deleted wholesale.
        let genC = try await storage.insertGeneration(
            GenerationRecord(level: 0, numEmbeddings: 1, minChunkID: chunkC.id, maxChunkID: chunkC.id, created: Date())
        )
        let bucketC = try await storage.insertBucket(onePairBucket(generationID: genC.id, chunkID: chunkC.id))

        var updatedBucketA = bucketA
        updatedBucketA.indices = IndicesCodec.encode([IndexPair(chunkID: UInt32(chunkE.id), tokenOffset: 0)])
        updatedBucketA.residuals = Q4Codec.encodeResiduals([Float](repeating: 0, count: dims))

        // chunkD is included in the delete set as the guarded case: a
        // document referencing its hash is inserted below (simulating a
        // race between vacuum's detection scan and this call), so the
        // guarded delete must skip it.
        let plan = VacuumPlan(
            chunkIDsToDelete: [chunkA.id, chunkB.id, chunkC.id, chunkD.id],
            bucketUpdates: [updatedBucketA],
            bucketIDsToDelete: [bucketB1.id],
            generationIDsToDelete: [genC.id],
            generationEmbeddingCountUpdates: [genA.id: 1, genB.id: 1]
        )

        // Simulate a race: a document was (re-)added referencing chunkD's
        // hash after vacuum's detection scan ran but before this call.
        try await storage.upsertDocument(
            DocumentRecord(uuid: "vac-raced-owner", date: Date(), hash: "vac-d", body: "y", lens: [1])
        )

        let deletedCount = try await storage.applyVacuumPlan(plan)

        // chunkA, chunkB, chunkC deleted; chunkD survives (guard fired).
        #expect(deletedCount == 3, "exactly chunkA/B/C should be deleted; chunkD is guarded by its raced document")
        #expect(try await storage.chunk(id: chunkA.id) == nil)
        #expect(try await storage.chunk(id: chunkB.id) == nil)
        #expect(try await storage.chunk(id: chunkC.id) == nil)
        #expect(try await storage.chunk(id: chunkD.id) != nil, "chunkD must survive: a document now references its hash")
        #expect(try await storage.chunk(id: chunkE.id) != nil)

        // Generation A survives with one bucket, updated pair count.
        let gensAfter = try await storage.generations()
        let genAAfter = gensAfter.first { $0.id == genA.id }
        #expect(genAAfter != nil)
        #expect(genAAfter?.numEmbeddings == 1)
        let bucketsA = try await storage.buckets(forGeneration: genA.id)
        #expect(bucketsA.count == 1)
        let survivingPairsA = try IndicesCodec.decode(bucketsA[0].indices)
        #expect(survivingPairsA.map { Int64($0.chunkID) } == [chunkE.id])

        // Generation B survives with only bucketB2 (bucketB1 deleted outright).
        let genBAfter = gensAfter.first { $0.id == genB.id }
        #expect(genBAfter != nil)
        #expect(genBAfter?.numEmbeddings == 1)
        let bucketsB = try await storage.buckets(forGeneration: genB.id)
        #expect(bucketsB.map(\.id) == [bucketB2.id])

        // Generation C fully deleted (its only bucket is gone with it).
        #expect(!gensAfter.map(\.id).contains(genC.id))
        let bucketsC = try await storage.buckets(forGeneration: genC.id)
        #expect(bucketsC.isEmpty)
        _ = bucketC // silence unused-variable warning; existence already asserted via bucketsC above.
    }

    static func runFreeListByteCount(_ storage: any SwitchcraftStorage) async throws {
        try await storage.clear()
        // Smoke test: must not throw, must be non-negative for any backend.
        let count = try await storage.freeListByteCount()
        #expect(count >= 0)
    }

    // MARK: - Ledger snapshot lifecycle (issue #136 / ADR 034)

    static func runSnapshotLifecycle(_ storage: any SwitchcraftStorage) async throws {
        try await storage.clear()

        // Empty store: no snapshot present.
        #expect(try await storage.loadLedgerSnapshot() == nil)

        // Save → load round-trips every field byte-for-byte.
        let payload = Data([0x01, 0x02, 0x03, 0xFF, 0x00, 0x42])
        let snapshot = LedgerSnapshotRecord(
            dims: 128,
            chunkCount: 7,
            maxChunkID: 9_001,
            totalEmbeddings: 123_456,
            maxGenerationID: 42,
            generationCount: 3,
            payload: payload
        )
        try await storage.saveLedgerSnapshot(snapshot)
        let loaded = try await storage.loadLedgerSnapshot()
        #expect(loaded == snapshot)

        // Save again overwrites the single slot (does not accumulate).
        let payload2 = Data([0xAA, 0xBB])
        let snapshot2 = LedgerSnapshotRecord(
            dims: 64,
            chunkCount: 1,
            maxChunkID: 1,
            totalEmbeddings: 5,
            maxGenerationID: 1,
            generationCount: 1,
            payload: payload2
        )
        try await storage.saveLedgerSnapshot(snapshot2)
        #expect(try await storage.loadLedgerSnapshot() == snapshot2)

        // An empty payload is a valid snapshot (empty-ledger case).
        let emptyPayloadSnapshot = LedgerSnapshotRecord(
            dims: 2, chunkCount: 0, maxChunkID: 0, totalEmbeddings: 0,
            maxGenerationID: 0, generationCount: 0, payload: Data()
        )
        try await storage.saveLedgerSnapshot(emptyPayloadSnapshot)
        #expect(try await storage.loadLedgerSnapshot() == emptyPayloadSnapshot)

        // Clear removes it; load returns nil.
        try await storage.clearLedgerSnapshot()
        #expect(try await storage.loadLedgerSnapshot() == nil)

        // Clearing again is a no-op (must not throw).
        try await storage.clearLedgerSnapshot()

        // storage.clear() also wipes any persisted snapshot.
        try await storage.saveLedgerSnapshot(snapshot)
        #expect(try await storage.loadLedgerSnapshot() != nil)
        try await storage.clear()
        #expect(try await storage.loadLedgerSnapshot() == nil)
    }

    // MARK: - Helpers

    static func makeDocument(
        uuid: String,
        body: String,
        date: Date = Date(timeIntervalSince1970: 0),
        metadata: [String: String] = [:],
        title: String? = nil
    ) -> DocumentRecord {
        let metadataData: Data
        if metadata.isEmpty {
            metadataData = Data()
        } else {
            metadataData = (try? JSONSerialization.data(withJSONObject: metadata)) ?? Data()
        }
        return DocumentRecord(
            uuid: uuid,
            date: date,
            metadata: metadataData,
            hash: "hash-\(uuid)",
            body: body,
            title: title,
            lens: []
        )
    }
}
