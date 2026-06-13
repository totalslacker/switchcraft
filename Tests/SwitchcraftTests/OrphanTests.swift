// Tests for orphan chunk detection and recovery (issue #120).
//
// R4: a chunk inserted via upsertChunk() without indexer.add() is
//     automatically recovered when store.add() is called with the same content.
// R9: findOrphanedChunks() identifies orphans correctly and excludes
//     properly-indexed chunks.

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
        let hasBuckets = try await storage.chunkHasBucketAssignments(chunk!.id)
        #expect(!hasBuckets, "orphan chunk must start with no bucket assignments")

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
}
