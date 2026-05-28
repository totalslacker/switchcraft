// SPDX-License-Identifier: Apache-2.0
import Foundation
import SwitchcraftCore

/// A test-only `SwitchcraftStorage` actor that wraps `InMemoryStorage` and
/// can inject a configurable error on `replaceGeneration` **before** mutating
/// the underlying storage. All other protocol methods forward transparently.
///
/// Used to verify the atomicity guarantee in requirement #6: when
/// `replaceGeneration` throws, storage must be left unchanged.
public actor FailingReplaceStorage: SwitchcraftStorage {

    /// Errors injectable by this stub.
    public enum ReplaceError: Error {
        case injected
    }

    private let inner: InMemoryStorage

    /// When `true`, `replaceGeneration` throws `ReplaceError.injected`
    /// **before** delegating to the underlying storage.
    public var shouldFailOnReplace: Bool = false

    /// Direct access to the underlying `InMemoryStorage` for post-failure
    /// state assertions in tests.
    public var underlyingStorage: InMemoryStorage { inner }

    public init(inner: InMemoryStorage = InMemoryStorage()) {
        self.inner = inner
    }

    // MARK: - Lifecycle

    public func open() async throws {
        try await inner.open()
    }

    public func close() async throws {
        try await inner.close()
    }

    public func clear() async throws {
        try await inner.clear()
    }

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

    public func documentCount() async throws -> Int {
        try await inner.documentCount()
    }

    public func indexedURLs() async throws -> Set<String> {
        try await inner.indexedURLs()
    }

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

    public func chunkCount() async throws -> Int {
        try await inner.chunkCount()
    }

    // MARK: - Generations

    public func insertGeneration(_ generation: GenerationRecord) async throws -> GenerationRecord {
        try await inner.insertGeneration(generation)
    }

    public func generations() async throws -> [GenerationRecord] {
        try await inner.generations()
    }

    public func deleteGeneration(id: Int64) async throws {
        try await inner.deleteGeneration(id: id)
    }

    public func updateGenerationEmbeddingCount(id: Int64, count: Int) async throws {
        try await inner.updateGenerationEmbeddingCount(id: id, count: count)
    }

    /// Injects `ReplaceError.injected` **before** any mutation when
    /// `shouldFailOnReplace == true`. Otherwise delegates transparently.
    public func replaceGeneration(
        losingGenerationID: Int64,
        survivingRecord: GenerationRecord,
        survivingBuckets: [BucketRecord]
    ) async throws -> GenerationRecord? {
        // Guard must be first — no mutations happen before this check.
        guard !shouldFailOnReplace else {
            throw ReplaceError.injected
        }
        return try await inner.replaceGeneration(
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

    // MARK: - Full-text Search

    public func searchFullText(
        query: String,
        limit: Int,
        filter: StorageFilter
    ) async throws -> [FullTextHit] {
        try await inner.searchFullText(query: query, limit: limit, filter: filter)
    }
}
