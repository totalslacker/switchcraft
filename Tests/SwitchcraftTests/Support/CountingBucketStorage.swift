// SPDX-License-Identifier: Apache-2.0
import Foundation
import SwitchcraftCore

/// A test-only `SwitchcraftStorage` actor that wraps `InMemoryStorage` and
/// counts calls to `buckets(forGeneration:)` — the storage-layer boundary
/// crossed by both the Requirement 13 init-time integrity walk and lazy
/// bucket-ref materialization (issue #137).
///
/// Used by `LazyLedgerMaterializationTests` to verify that rehydrating the
/// ledger costs exactly one `buckets(forGeneration:)` call per active
/// generation (the cheap integrity walk — no residual decode, no per-token
/// work), and that later materialization (at `ledgerContents()` or a real
/// compaction) batches its own bucket reads the same way — one call per
/// generation regardless of how many tokens within it are referenced —
/// rather than one call per token.
public actor CountingBucketStorage: SwitchcraftStorage {

    private let inner: InMemoryStorage

    /// Number of `buckets(forGeneration:)` calls observed since the last
    /// `resetBucketReadCount()`.
    public private(set) var bucketReadCount: Int = 0

    /// Direct access to the underlying `InMemoryStorage` for setup/teardown
    /// that doesn't need counting (e.g. building the initial corpus).
    public var underlyingStorage: InMemoryStorage { inner }

    public init(inner: InMemoryStorage = InMemoryStorage()) {
        self.inner = inner
    }

    public func resetBucketReadCount() {
        bucketReadCount = 0
    }

    // MARK: - Lifecycle

    public func open() async throws { try await inner.open() }
    public func close() async throws { try await inner.close() }
    public func clear() async throws { try await inner.clear() }

    // MARK: - Documents

    public func upsertDocument(_ document: DocumentRecord) async throws {
        try await inner.upsertDocument(document)
    }
    public func deleteDocument(uuid: String) async throws {
        try await inner.deleteDocument(uuid: uuid)
    }
    public func document(uuid: String) async throws -> DocumentRecord? {
        try await inner.document(uuid: uuid)
    }
    public func documents(matching filter: StorageFilter) async throws -> [DocumentRecord] {
        try await inner.documents(matching: filter)
    }
    public func documents(forChunkHash hash: String) async throws -> [DocumentRecord] {
        try await inner.documents(forChunkHash: hash)
    }
    public func documentCount() async throws -> Int { try await inner.documentCount() }
    public func indexedURLs() async throws -> Set<String> { try await inner.indexedURLs() }

    // MARK: - Chunks

    public func upsertChunk(_ chunk: ChunkRecord) async throws -> ChunkRecord {
        try await inner.upsertChunk(chunk)
    }
    public func chunk(hash: String) async throws -> ChunkRecord? {
        try await inner.chunk(hash: hash)
    }
    public func chunk(id: Int64) async throws -> ChunkRecord? {
        try await inner.chunk(id: id)
    }
    public func chunkCount() async throws -> Int { try await inner.chunkCount() }

    // MARK: - Generations

    public func insertGeneration(_ generation: GenerationRecord) async throws -> GenerationRecord {
        try await inner.insertGeneration(generation)
    }
    public func generations() async throws -> [GenerationRecord] { try await inner.generations() }
    public func deleteGeneration(id: Int64) async throws { try await inner.deleteGeneration(id: id) }
    public func updateGenerationEmbeddingCount(id: Int64, count: Int) async throws {
        try await inner.updateGenerationEmbeddingCount(id: id, count: count)
    }
    public func replaceGeneration(
        losingGenerationID: Int64,
        survivingRecord: GenerationRecord,
        survivingBuckets: [BucketRecord]
    ) async throws -> GenerationRecord? {
        try await inner.replaceGeneration(
            losingGenerationID: losingGenerationID,
            survivingRecord: survivingRecord,
            survivingBuckets: survivingBuckets
        )
    }

    // MARK: - Buckets

    public func insertBucket(_ bucket: BucketRecord) async throws -> BucketRecord {
        try await inner.insertBucket(bucket)
    }

    /// The instrumented call: counts every invocation before delegating.
    public func buckets(forGeneration generationID: Int64) async throws -> [BucketRecord] {
        bucketReadCount += 1
        return try await inner.buckets(forGeneration: generationID)
    }

    // MARK: - Full-text Search

    public func searchFullText(
        query: String,
        limit: Int,
        filter: StorageFilter
    ) async throws -> [FullTextHit] {
        try await inner.searchFullText(query: query, limit: limit, filter: filter)
    }
}
