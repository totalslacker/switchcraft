// Tests for orphan chunk detection and recovery (issues #120, #130).
//
// R4: a chunk inserted via upsertChunk() without indexer.add() is
//     automatically recovered when store.add() is called with the same content.
// R9: findOrphanedChunks() identifies orphans correctly and excludes
//     properly-indexed chunks.
// R6: a chunk with partial bucket assignments (0 < refs < expected) is
//     detected by findOrphanedChunks() and recovered by re-adding the document.

import Foundation
import Testing
@testable import Switchcraft
@testable import SwitchcraftCore

@Suite("Orphan Detection and Recovery")
struct OrphanTests {

    // MARK: - Helpers

    /// Construct a fresh (SwitchcraftStore, InMemoryStorage) pair with a
    /// small l0Capacity so that store.index() reliably flushes to buckets.
    private static func makeStoreWithStorage(
        config: StoreConfig = StoreConfig(
            indexer: IndexerConfig(l0Capacity: 4, lsmFanout: 2)
        )
    ) async throws -> (SwitchcraftStore, InMemoryStorage) {
        let storage = InMemoryStorage()
        let store = try await SwitchcraftStore(
            storage: storage,
            embedder: MockEmbedder(dims: 32),
            config: config
        )
        return (store, storage)
    }

    // MARK: - R4: orphan recovery via add()

    @Test("orphan chunk is recovered when store.add() is called with matching content")
    func orphanRecoveredOnAdd() async throws {
        let (store, storage) = try await Self.makeStoreWithStorage()

        let body = "orphan recovery content"
        // Compute the content hash that store.add() will compute for this body.
        let hash = SwitchcraftStore.contentHash(body)
        // MockEmbedder splits on whitespace: "orphan", "recovery", "content" = 3 tokens.
        let tokenCount = body.split(whereSeparator: { $0.isWhitespace }).count

        // Plant an orphan chunk with the correct hash but no bucket assignments.
        _ = try await storage.upsertChunk(
            ChunkRecord(hash: hash, model: "mock-embedder-v1-d32", embeddings: Data(), counts: [tokenCount])
        )

        // Verify the orphan has no bucket assignments before recovery.
        let chunk = try await storage.chunk(hash: hash)
        #expect(chunk != nil)
        let refCount = try await storage.chunkBucketRefCount(chunk!.id)
        #expect(refCount == 0, "orphan chunk must start with no bucket assignments")

        // Call store.add() — the orphan detection path (R1) should fire.
        try await store.add(id: "orphan-doc", body: body)

        // Flush: this compacts the re-buffered embeddings into bucket blobs.
        try await store.index()

        // The document must now appear in vector search.
        let hits = try await store.search(query: body, topK: 10)
        #expect(hits.contains(where: { $0.uuid == "orphan-doc" }),
                "orphan-doc must be retrievable after orphan recovery + flush")

        try await store.shutdown()
    }

    // MARK: - R2: pending chunk is not double-buffered

    @Test("re-adding pending-but-unflushed content does not double-buffer")
    func pendingChunkNotDoubleBuffered() async throws {
        let (store, storage) = try await Self.makeStoreWithStorage()

        let body = "pending dedup content"

        // First add: puts the chunk into the indexer ledger.
        try await store.add(id: "doc-a", body: body)
        // Second add with identical content before any flush.
        // This must not throw and must not double-buffer (R2).
        try await store.add(id: "doc-b", body: body)

        // Both documents are present, but still only one chunk.
        let chunkCount = try await storage.chunkCount()
        #expect(chunkCount == 1)

        try await store.index()

        // Both docs must be findable.
        let hits = try await store.search(query: body, topK: 10)
        let uuids = Set(hits.map(\.uuid))
        #expect(uuids.contains("doc-a"))
        #expect(uuids.contains("doc-b"))

        try await store.shutdown()
    }

    // MARK: - R9: findOrphanedChunks()

    @Test("findOrphanedChunks returns orphans and excludes indexed chunks")
    func findOrphanedChunksCorrect() async throws {
        let (store, storage) = try await Self.makeStoreWithStorage()

        let orphanBody = "orphan scan body text"
        let orphanHash = SwitchcraftStore.contentHash(orphanBody)
        let orphanTokenCount = orphanBody.split(whereSeparator: { $0.isWhitespace }).count

        // Plant orphan chunk directly (no indexer involvement).
        let orphanChunk = try await storage.upsertChunk(
            ChunkRecord(
                hash: orphanHash,
                model: "mock-embedder-v1-d32",
                embeddings: Data(),
                counts: [orphanTokenCount]
            )
        )
        // Plant a document pointing to the orphan chunk so owningDocuments is non-empty.
        try await storage.upsertDocument(
            DocumentRecord(
                uuid: "orphan-owner",
                date: Date(),
                metadata: Data(),
                hash: orphanHash,
                body: orphanBody,
                lens: [orphanTokenCount]
            )
        )

        // Add a normally indexed document via the store.
        try await store.add(id: "indexed-doc", body: "indexed document body words")
        // Flush so the indexed chunk has committed bucket assignments.
        try await store.index()

        // Run orphan scan.
        let orphans = try await store.findOrphanedChunks()

        // The orphan chunk must appear.
        let foundOrphan = orphans.first(where: { $0.chunkID == orphanChunk.id })
        #expect(foundOrphan != nil, "orphan chunk must be returned by findOrphanedChunks()")
        if let fo = foundOrphan {
            #expect(fo.hash == orphanHash)
            #expect(fo.expectedTokenCount == orphanTokenCount)
            #expect(fo.bucketReferenceCount == 0)
            #expect(fo.owningDocuments.contains("orphan-owner"),
                    "owningDocuments must include the document pointing to this chunk")
        }

        // The indexed chunk must NOT appear.
        let indexedBody = "indexed document body words"
        let indexedHash = SwitchcraftStore.contentHash(indexedBody)
        let indexedChunk = try await storage.chunk(hash: indexedHash)
        #expect(indexedChunk != nil)
        let foundIndexed = orphans.first(where: { $0.chunkID == indexedChunk!.id })
        #expect(foundIndexed == nil, "correctly indexed chunk must not appear in orphan list")

        try await store.shutdown()
    }

    // MARK: - R6: partial-orphan recovery via add()

    @Test("chunk with partial bucket assignments is recovered when store.add() is called with matching content")
    func partialOrphanRecoveredOnAdd() async throws {
        let dims = 32
        let config = StoreConfig(indexer: IndexerConfig(l0Capacity: 4, lsmFanout: 2))
        let (store, storage) = try await Self.makeStoreWithStorage(config: config)

        let body = "partial orphan test content"
        // MockEmbedder splits on whitespace: 4 tokens.
        let expectedTokenCount = body.split(whereSeparator: { $0.isWhitespace }).count

        // 1. Index the document fully.
        try await store.add(id: "partial-doc", body: body)
        try await store.index()

        // 2. Confirm fully indexed: no orphans.
        let orphansBefore = try await store.findOrphanedChunks()
        #expect(orphansBefore.isEmpty, "chunk must be fully indexed after first flush")

        // 3. Retrieve chunk to get its ID.
        let hash = SwitchcraftStore.contentHash(body)
        let chunk = try await storage.chunk(hash: hash)
        #expect(chunk != nil)
        let chunkID = chunk!.id

        // 4. Simulate partial indexing: delete all existing generations and
        //    replace with one generation that has a single-pair bucket blob with
        //    numEmbeddings=1 (the actual pair count). This exercises the real
        //    production path — unlike the previous numEmbeddings=0 workaround.
        let existingGens = try await storage.generations()
        for gen in existingGens {
            try await storage.deleteGeneration(id: gen.id)
        }
        let partialGen = try await storage.insertGeneration(
            GenerationRecord(
                level: 0,
                numEmbeddings: 1,
                minChunkID: chunkID,
                maxChunkID: chunkID,
                created: Date()
            )
        )
        let partialIndices = IndicesCodec.encode([
            IndexPair(chunkID: UInt32(chunkID), tokenOffset: 0)
        ])
        // Center: all-zero float vector. Residuals: Q4-encoded zeros for 1 pair.
        // Both must be valid for Indexer rehydration to succeed.
        let dummyCenter = Data(repeating: 0, count: dims * MemoryLayout<Float>.size)
        let dummyResiduals = Q4Codec.encodeResiduals([Float](repeating: 0.0, count: dims))
        _ = try await storage.insertBucket(
            BucketRecord(
                generationID: partialGen.id,
                center: dummyCenter,
                indices: partialIndices,
                residuals: dummyResiduals
            )
        )

        // 5. Simulate process restart: close the first store (flush is a no-op
        //    since pendingCount = 0 after step 1's store.index()) then open a
        //    new store over the same storage so the Indexer rehydrates exactly
        //    P=1 rows from the planted partial gen (not the M=4 rows that
        //    remained in the first store's in-memory ledger). Without the restart
        //    simulation, removeFromLedger would remove M rows (not P), causing
        //    the correction to overcount.
        try await store.shutdown()
        let store2 = try await SwitchcraftStore(
            storage: storage,
            embedder: MockEmbedder(dims: dims),
            config: config
        )

        // 6. Verify findOrphanedChunks() detects the partial orphan.
        let orphansPartial = try await store2.findOrphanedChunks()
        let partialOrphan = orphansPartial.first(where: { $0.chunkID == chunkID })
        #expect(partialOrphan != nil, "partial orphan must be detected by findOrphanedChunks()")
        if let po = partialOrphan {
            #expect(po.bucketReferenceCount == 1,
                    "partial orphan must have 1 bucket ref (from planted blob)")
            #expect(po.expectedTokenCount == expectedTokenCount,
                    "expectedTokenCount must equal the chunk's token count")
            #expect(po.bucketReferenceCount < po.expectedTokenCount,
                    "partial orphan must have fewer refs than expected")
        }

        // 7. Re-add the same document — the partial-recovery path (R1) must fire
        //    and must NOT throw ledgerOutOfSync (regression for issue #132).
        try await store2.add(id: "partial-doc", body: body)

        // 8. Flush: re-buffered embeddings are written to a new full generation.
        try await store2.index()

        // 9. Verify fully indexed: no orphans.
        let orphansAfter = try await store2.findOrphanedChunks()
        #expect(orphansAfter.isEmpty, "chunk must be fully indexed after recovery flush")

        // 10. Verify bucket ref count equals expected token count.
        let refCount = try await storage.chunkBucketRefCount(chunkID)
        #expect(refCount == expectedTokenCount,
                "bucket ref count must equal expected token count after recovery")

        try await store2.shutdown()
    }

    // MARK: - AC6: batch partial-orphan recovery completes without ledgerOutOfSync

    @Test("batch partial-orphan recovery loop completes without ledgerOutOfSync (issue #132)")
    func partialOrphanBatchRecoveryNoLedgerSync() async throws {
        let dims = 32
        let config = StoreConfig(indexer: IndexerConfig(l0Capacity: 4, lsmFanout: 2))
        let (store, storage) = try await Self.makeStoreWithStorage(config: config)

        // Three documents with distinct multi-token bodies (5, 4, 5 tokens each).
        let docs: [(id: String, body: String)] = [
            ("batch-doc-0", "batch recovery alpha beta gamma"),
            ("batch-doc-1", "batch recovery delta epsilon"),
            ("batch-doc-2", "batch recovery zeta eta theta"),
        ]

        // 1. Index all three documents fully.
        for doc in docs {
            try await store.add(id: doc.id, body: doc.body)
        }
        try await store.index()

        // 2. Confirm all chunks fully indexed.
        #expect(try await store.findOrphanedChunks().isEmpty,
                "all chunks must be fully indexed before partial-orphan simulation")

        // 3. Delete all existing generations and plant one partial gen per chunk
        //    (P=1 pair, numEmbeddings=1) so each chunk becomes a partial orphan.
        let existingGens = try await storage.generations()
        for gen in existingGens {
            try await storage.deleteGeneration(id: gen.id)
        }
        let dummyCenter = Data(repeating: 0, count: dims * MemoryLayout<Float>.size)
        let dummyResiduals = Q4Codec.encodeResiduals([Float](repeating: 0.0, count: dims))
        for doc in docs {
            let hash = SwitchcraftStore.contentHash(doc.body)
            guard let chunk = try await storage.chunk(hash: hash) else {
                Issue.record("chunk not found for body: \(doc.body)")
                continue
            }
            let chunkID = chunk.id
            let partialGen = try await storage.insertGeneration(
                GenerationRecord(
                    level: 0,
                    numEmbeddings: 1,
                    minChunkID: chunkID,
                    maxChunkID: chunkID,
                    created: Date()
                )
            )
            _ = try await storage.insertBucket(
                BucketRecord(
                    generationID: partialGen.id,
                    center: dummyCenter,
                    indices: IndicesCodec.encode([IndexPair(chunkID: UInt32(chunkID), tokenOffset: 0)]),
                    residuals: dummyResiduals
                )
            )
        }

        // 4. Simulate process restart: Indexer rehydrates exactly P=1 rows per
        //    chunk from the planted partial gens.
        try await store.shutdown()
        let store2 = try await SwitchcraftStore(
            storage: storage,
            embedder: MockEmbedder(dims: dims),
            config: config
        )

        // 5. Find orphans: all 3 chunks must be detected as partial.
        let orphans = try await store2.findOrphanedChunks()
        #expect(orphans.count == docs.count,
                "all 3 chunks must be detected as partial orphans")

        // 6. Run recovery loop — must complete without throwing ledgerOutOfSync
        //    for any document in the batch (regression for issue #132).
        for orphan in orphans {
            for docUUID in orphan.owningDocuments {
                guard let doc = try await storage.document(uuid: docUUID) else { continue }
                try await store2.add(id: docUUID, body: doc.body)
            }
        }

        // 7. Flush: all re-buffered chunks compacted into new full generation.
        try await store2.index()

        // 8. Verify all orphans resolved.
        let orphansAfter = try await store2.findOrphanedChunks()
        #expect(orphansAfter.isEmpty,
                "all partial orphans must be fully indexed after batch recovery")

        try await store2.shutdown()
    }

    // MARK: - AC7: partial-orphan recovery is idempotent

    @Test("partial-orphan recovery is idempotent on a partially-repaired corpus (issue #132)")
    func partialOrphanRecoveryIdempotent() async throws {
        let dims = 32
        // Large l0Capacity prevents cascade merging between the two sequential
        // flushes, keeping full-doc and partial-doc in separate generations so
        // we can surgically make only partial-doc's gen partial.
        let config = StoreConfig(indexer: IndexerConfig(l0Capacity: 100, lsmFanout: 2))
        let (store, storage) = try await Self.makeStoreWithStorage(config: config)

        // Two docs with 2 tokens each.
        let fullBody = "full content"   // 2 tokens → M = 2
        let partialBody = "partial doc" // 2 tokens → M = 2

        // 1. Index "full-doc" and flush → gen1 at level 0 with 2 pairs.
        try await store.add(id: "full-doc", body: fullBody)
        try await store.index()

        // 2. Index "partial-doc" and flush → with l0Capacity=100, total pending=2
        //    and levelSums[0]=2 gives total=4 ≤ 100, so NO cascade; gen2 at level 0.
        //    NOTE: the cascade walk MERGES gen1 into the new flush even at level 0 —
        //    both chunks end up in a single gen. We'll split them surgically below.
        try await store.add(id: "partial-doc", body: partialBody)
        try await store.index()

        // 3. Confirm all chunks fully indexed: no orphans.
        #expect(try await store.findOrphanedChunks().isEmpty,
                "both chunks must be fully indexed before partial-orphan simulation")

        // 4. Retrieve chunk IDs.
        let fullHash = SwitchcraftStore.contentHash(fullBody)
        let partialHash = SwitchcraftStore.contentHash(partialBody)
        guard let fullChunk = try await storage.chunk(hash: fullHash),
              let partialChunk = try await storage.chunk(hash: partialHash) else {
            Issue.record("chunks not found")
            return
        }
        let fullChunkID = fullChunk.id
        let partialChunkID = partialChunk.id

        // 5. Surgically replace the single merged gen with two separate gens:
        //    - gen A (fully indexed): full-doc's chunk with M=2 pairs
        //    - gen B (partial):       partial-doc's chunk with P=1 pair
        //    This simulates a "partially repaired corpus" where one chunk is at
        //    full count and another is still partial.
        let existingGens = try await storage.generations()
        for gen in existingGens {
            try await storage.deleteGeneration(id: gen.id)
        }

        let dummyCenter = Data(repeating: 0, count: dims * MemoryLayout<Float>.size)
        let fullResiduals = Q4Codec.encodeResiduals([Float](repeating: 0.0, count: dims * 2))
        let partialResiduals = Q4Codec.encodeResiduals([Float](repeating: 0.0, count: dims))

        // gen A: full-doc fully indexed (2 pairs, numEmbeddings=2)
        let genA = try await storage.insertGeneration(
            GenerationRecord(
                level: 0,
                numEmbeddings: 2,
                minChunkID: fullChunkID,
                maxChunkID: fullChunkID,
                created: Date()
            )
        )
        _ = try await storage.insertBucket(
            BucketRecord(
                generationID: genA.id,
                center: dummyCenter,
                indices: IndicesCodec.encode([
                    IndexPair(chunkID: UInt32(fullChunkID), tokenOffset: 0),
                    IndexPair(chunkID: UInt32(fullChunkID), tokenOffset: 1),
                ]),
                residuals: fullResiduals
            )
        )

        // gen B: partial-doc partially indexed (1 pair, numEmbeddings=1)
        let genB = try await storage.insertGeneration(
            GenerationRecord(
                level: 0,
                numEmbeddings: 1,
                minChunkID: partialChunkID,
                maxChunkID: partialChunkID,
                created: Date()
            )
        )
        _ = try await storage.insertBucket(
            BucketRecord(
                generationID: genB.id,
                center: dummyCenter,
                indices: IndicesCodec.encode([
                    IndexPair(chunkID: UInt32(partialChunkID), tokenOffset: 0),
                ]),
                residuals: partialResiduals
            )
        )

        // 6. Simulate process restart: Indexer rehydrates full-doc (2 rows) and
        //    partial-doc (1 row) from the two planted gens.
        try await store.shutdown()
        let store2 = try await SwitchcraftStore(
            storage: storage,
            embedder: MockEmbedder(dims: dims),
            config: config
        )

        // 7. findOrphanedChunks() must return only partial-doc (1 orphan).
        let orphansBeforeRecovery = try await store2.findOrphanedChunks()
        #expect(orphansBeforeRecovery.count == 1,
                "only partial-doc must appear as an orphan; full-doc is already fully indexed")
        let initialOrphanCount = orphansBeforeRecovery.count

        // 8. Pass 1: run recovery — must not throw; partial count must not increase.
        for orphan in orphansBeforeRecovery {
            for docUUID in orphan.owningDocuments {
                guard let doc = try await storage.document(uuid: docUUID) else { continue }
                try await store2.add(id: docUUID, body: doc.body)
            }
        }
        try await store2.index()
        let orphansAfterPass1 = try await store2.findOrphanedChunks()
        #expect(orphansAfterPass1.count <= initialOrphanCount,
                "partial-orphan count must not increase after recovery pass 1")
        #expect(orphansAfterPass1.isEmpty,
                "all orphans must be resolved after pass 1")

        // 9. Pass 2: re-run on an already-recovered corpus — must be a no-op
        //    with no errors and no increase in orphan count.
        let orphansBeforePass2 = try await store2.findOrphanedChunks()
        for orphan in orphansBeforePass2 {
            for docUUID in orphan.owningDocuments {
                guard let doc = try await storage.document(uuid: docUUID) else { continue }
                try await store2.add(id: docUUID, body: doc.body)
            }
        }
        try await store2.index()
        let orphansAfterPass2 = try await store2.findOrphanedChunks()
        #expect(orphansAfterPass2.count <= orphansBeforePass2.count,
                "partial-orphan count must not increase on second recovery pass")

        try await store2.shutdown()
    }
}
