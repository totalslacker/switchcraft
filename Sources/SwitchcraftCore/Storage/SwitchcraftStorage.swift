// SPDX-License-Identifier: Apache-2.0
import Foundation

/// The pluggable storage layer that all Switchcraft backends conform to.
///
/// SQLite is the reference implementation; other embedded or server-side
/// stores can implement this protocol and slot in without touching the
/// search/index/scoring engine.
///
/// Backends are responsible for:
/// - Storing and retrieving the four record types (document, chunk,
///   generation, bucket).
/// - Maintaining a full-text index over document `body` so that
///   `searchFullText` returns BM25-ish results.
/// - Lowering `StorageFilter` to the backend's native query form.
///
/// Conformance is verified by `StorageConformance` in the
/// `SwitchcraftStorageTesting` module. Any implementation that passes the
/// conformance suite can be used with the engine.
public protocol SwitchcraftStorage: Sendable {
    // MARK: - Lifecycle

    /// Open the underlying store. Idempotent.
    func open() async throws

    /// Close the underlying store. Idempotent.
    func close() async throws

    /// Remove all documents, chunks, generations, and buckets, and reset
    /// any auto-assigned identifiers.
    func clear() async throws

    // MARK: - Documents

    /// Insert or replace a document, keyed by `uuid`. Backends update the
    /// full-text index as part of this call.
    func upsertDocument(_ document: DocumentRecord) async throws

    /// Delete the document with the given uuid. No-op if absent.
    func deleteDocument(uuid: String) async throws

    /// Return the document with the given uuid, or nil if absent.
    func document(uuid: String) async throws -> DocumentRecord?

    /// Return all documents matching `filter`, in unspecified order.
    func documents(matching filter: StorageFilter) async throws -> [DocumentRecord]

    /// Return every document whose `hash` equals the supplied chunk hash,
    /// in unspecified order. Used by the search pipeline to map a scored
    /// chunk back to the documents that share it (chunks are deduplicated
    /// by content hash; multiple documents may reference the same chunk).
    /// Returns an empty array if no documents match.
    func documents(forChunkHash hash: String) async throws -> [DocumentRecord]

    /// Return the total number of documents.
    func documentCount() async throws -> Int

    // MARK: - Chunks

    /// Insert a chunk if no chunk with the same `hash` exists. Returns the
    /// resulting record (with assigned id) regardless of whether it was
    /// newly inserted or already present.
    func upsertChunk(_ chunk: ChunkRecord) async throws -> ChunkRecord

    /// Return the chunk with the given hash, or nil if absent.
    func chunk(hash: String) async throws -> ChunkRecord?

    /// Return the chunk with the given backend-assigned id, or nil if absent.
    func chunk(id: Int64) async throws -> ChunkRecord?

    /// Return the total number of chunks.
    func chunkCount() async throws -> Int

    // MARK: - Generations

    /// Insert a generation record. The provided `id` is ignored; the backend
    /// assigns a new monotonically increasing id and returns the inserted
    /// record.
    func insertGeneration(_ generation: GenerationRecord) async throws -> GenerationRecord

    /// Return all generations, ordered by ascending id.
    func generations() async throws -> [GenerationRecord]

    /// Delete the generation with the given id and any buckets that
    /// reference it. No-op if absent.
    func deleteGeneration(id: Int64) async throws

    // MARK: - Buckets

    /// Insert a bucket record. The provided `id` is ignored; the backend
    /// assigns a new monotonically increasing id and returns the inserted
    /// record.
    func insertBucket(_ bucket: BucketRecord) async throws -> BucketRecord

    /// Return all buckets for the given generation, in insertion order.
    func buckets(forGeneration generationID: Int64) async throws -> [BucketRecord]

    // MARK: - Full-text Search

    /// Run a BM25-style full-text query over document bodies. Backends MAY
    /// apply `filter` natively; the in-memory backend evaluates it in Swift.
    /// Returns at most `limit` hits, highest-scoring first.
    func searchFullText(
        query: String,
        limit: Int,
        filter: StorageFilter
    ) async throws -> [FullTextHit]
}
