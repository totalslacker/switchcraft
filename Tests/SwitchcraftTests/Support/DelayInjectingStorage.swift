// SPDX-License-Identifier: Apache-2.0
import Foundation
import SwitchcraftCore

/// A test-only `SwitchcraftStorage` actor that wraps `InMemoryStorage` and
/// injects a configurable `Task.sleep` before `buckets(forGeneration:)`
/// (the vector/XTR sub-stage's storage call) and/or `searchFullText(...)`
/// (the BM25/FTS sub-stage's storage call), independently.
///
/// Used by `SearchTimingTests` (issue #148) to prove that a slow vector
/// stage doesn't inflate `bm25Duration` and vice versa, without a
/// hardcoded seconds-scale sleep — delays here are millisecond-scale and
/// injected explicitly by the test.
public actor DelayInjectingStorage: SwitchcraftStorage {

    private let inner: InMemoryStorage
    private var bucketDelay: Duration = .zero
    private var fullTextDelay: Duration = .zero

    /// Direct access to the underlying `InMemoryStorage` for setup/teardown
    /// that doesn't need delay injection (e.g. building the initial corpus).
    public var underlyingStorage: InMemoryStorage { inner }

    public init(inner: InMemoryStorage = InMemoryStorage()) {
        self.inner = inner
    }

    /// Delay injected before every subsequent `buckets(forGeneration:)` call.
    public func setBucketDelay(_ delay: Duration) {
        bucketDelay = delay
    }

    /// Delay injected before every subsequent `searchFullText(...)` call.
    public func setFullTextDelay(_ delay: Duration) {
        fullTextDelay = delay
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

    public func buckets(forGeneration generationID: Int64) async throws -> [BucketRecord] {
        try await inner.buckets(forGeneration: generationID)
    }

    /// The instrumented call: sleeps for `bucketDelay` before delegating.
    /// This is the call `SearchEngine`'s centroid scan actually makes
    /// (issue #151 / ADR 041's two-phase scan/fetch split), so this is
    /// where the delay belongs to preserve the original "slow vector
    /// stage" semantics.
    public func scanBuckets(forGeneration generationID: Int64) async throws -> [BucketScanRecord] {
        if bucketDelay > .zero {
            try await Task.sleep(for: bucketDelay)
        }
        return try await inner.scanBuckets(forGeneration: generationID)
    }

    public func buckets(ids: [Int64]) async throws -> [BucketRecord] {
        try await inner.buckets(ids: ids)
    }

    // MARK: - Full-text Search

    /// The instrumented call: sleeps for `fullTextDelay` before delegating.
    public func searchFullText(
        query: String,
        limit: Int,
        filter: StorageFilter
    ) async throws -> [FullTextHit] {
        if fullTextDelay > .zero {
            try await Task.sleep(for: fullTextDelay)
        }
        return try await inner.searchFullText(query: query, limit: limit, filter: filter)
    }
}
