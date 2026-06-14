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

    /// Return the set of all document UUIDs present in the store.
    ///
    /// Used by the backfill machinery to skip pages that are already indexed
    /// without re-processing them. Returns an empty set if the store has no
    /// documents or if the backend does not support enumeration.
    func indexedURLs() async throws -> Set<String>

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

    /// Return every chunk record in the store, in unspecified order.
    ///
    /// Used by `SwitchcraftStore.findOrphanedChunks()` to enumerate chunks
    /// that may lack bucket assignments. Backends that do not implement this
    /// method return an empty array via the default extension, so
    /// `findOrphanedChunks()` returns an empty result rather than an error.
    func allChunks() async throws -> [ChunkRecord]

    /// Return `true` if the given chunk has at least one
    /// `(chunkID, tokenOffset)` pair across all committed bucket blobs.
    ///
    /// Used by `SwitchcraftStore.add()` to detect orphaned chunks in the
    /// content-hash dedup branch.  Returning `true` (the default) is the
    /// safe conservative choice: it treats unknown chunks as fully indexed,
    /// so third-party backends opt out of orphan recovery rather than
    /// unconditionally re-indexing on every `add()` call.
    ///
    /// - Throws: `IndicesCodec.Error` if a bucket blob is corrupt.
    func chunkHasBucketAssignments(_ chunkID: Int64) async throws -> Bool

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

    /// Atomically delete `losingGenerationID` (and all its buckets) and,
    /// if `survivingBuckets` is non-empty, insert a new generation record
    /// (assigned a fresh monotonically-increasing id by the backend) with
    /// each surviving bucket. Surviving buckets may arrive with any
    /// `generationID`; the backend overwrites it with the newly-assigned
    /// generation id before persisting.
    ///
    /// The entire operation — delete + optional insert — must execute in a
    /// single atomic transaction. On any error the transaction is rolled
    /// back, the storage state is unchanged, and the error is re-thrown.
    ///
    /// Used by `Indexer.init` when `rehydrationConflictBehavior == .autoRecover`
    /// to prune conflicting (losing) generation data and optionally replace it
    /// with a filtered survivor set, all without losing any data on failure.
    ///
    /// - Returns: The newly-inserted `GenerationRecord` (with assigned id),
    ///   or `nil` if `survivingBuckets` is empty (generation fully pruned).
    func replaceGeneration(
        losingGenerationID: Int64,
        survivingRecord: GenerationRecord,
        survivingBuckets: [BucketRecord]
    ) async throws -> GenerationRecord?

    /// Update the `numEmbeddings` field for the generation with the given id.
    /// No-op if the id is absent.
    ///
    /// Used by `Indexer.init` auto-recovery (step 3.5) to correct stale
    /// `numEmbeddings` values left on surviving (winner) generations when a
    /// crash occurred between the generation-row insert (step 9 of
    /// `performFlush`) and the completion of bucket inserts (step 10). A
    /// targeted UPDATE is used rather than `replaceGeneration` so that the
    /// generation's id is preserved — which is required by the cascade walk
    /// and by existing test assertions.
    func updateGenerationEmbeddingCount(id: Int64, count: Int) async throws

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

    // MARK: - Search deadline (optional, backend-specific)

    /// Configure a per-search-call deadline on the backend.
    ///
    /// Called at the top of `SearchEngine.searchHybrid` with the
    /// `SearchDeadlineContext` for the current call. Backends that support
    /// in-flight interruption (e.g. `SQLiteStorage` via
    /// `sqlite3_progress_handler`) override this to arm their interrupt
    /// mechanism with the remaining budget. Backends that cannot interrupt
    /// in-flight queries may leave the default no-op implementation.
    ///
    /// Pass `nil` to disarm any previously configured deadline.
    func configureSearchDeadline(_ ctx: SearchDeadlineContext?) async

    // MARK: - WAL checkpoint (optional, backend-specific)

    /// Flush any pending WAL writes to the main database file and truncate
    /// the WAL.
    ///
    /// Backends that use SQLite in WAL mode override this to run
    /// `PRAGMA wal_checkpoint(TRUNCATE)`. Non-WAL backends (in-memory,
    /// non-SQLite) may leave the default no-op.
    ///
    /// Called during graceful shutdown to ensure the database file is in a
    /// clean state before process exit.
    func walCheckpoint() async throws -> CheckpointResult
}

extension SwitchcraftStorage {
    public func configureSearchDeadline(_ ctx: SearchDeadlineContext?) async {}
    public func walCheckpoint() async throws -> CheckpointResult { .complete }
    public func allChunks() async throws -> [ChunkRecord] { [] }
    public func chunkHasBucketAssignments(_ chunkID: Int64) async throws -> Bool { true }
}
