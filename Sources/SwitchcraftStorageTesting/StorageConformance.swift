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
