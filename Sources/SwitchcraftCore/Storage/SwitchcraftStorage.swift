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

    /// Return the total number of `(chunkID, tokenOffset)` pairs for the
    /// given chunk across all committed bucket blobs.
    ///
    /// Used by `SwitchcraftStore.add()` to determine whether a chunk is
    /// *completely* indexed (actualRefs == expectedTokenCount) in the
    /// content-hash dedup branch.  Returning `Int.max` (the default) is the
    /// safe conservative choice: it treats unknown chunks as fully indexed,
    /// so third-party backends opt out of orphan recovery rather than
    /// unconditionally re-indexing on every `add()` call.
    ///
    /// - Throws: `IndicesCodec.Error` if a bucket blob is corrupt.
    func chunkBucketRefCount(_ chunkID: Int64) async throws -> Int

    /// Return the set of all `document.hash` values present in the store.
    ///
    /// Used by `SwitchcraftStore.vacuum()` to compute the abandoned-chunk
    /// candidate set (`chunk.hash` values with no owning document) via one
    /// set-difference instead of one storage round trip per chunk. The
    /// default implementation derives the set from `documents(matching:
    /// .all)`, which is correct but materializes every full
    /// `DocumentRecord` (including `body`); a backend that can run
    /// `SELECT DISTINCT hash FROM document` should override this method
    /// for production-scale corpora.
    func documentHashes() async throws -> Set<String>

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

    /// Return a lightweight scan-phase view (`center` + residual byte count,
    /// no `indices`/`residuals` payload bytes) of every bucket in the given
    /// generation, ordered by ascending `id`.
    ///
    /// Used by `SearchEngine`'s centroid scan, which only needs the centroid
    /// and residual token count to score and select candidate buckets —
    /// `indices`/`residuals` are fetched afterward, only for the subset of
    /// buckets selection picks, via `buckets(ids:)`. Implementations MUST NOT
    /// read the `indices`/`residuals` payload bytes to satisfy this call
    /// (e.g. a SQL backend should read a blob's length header, not its
    /// content). The `id ASC` order is load-bearing: it feeds
    /// `cblas_sgemm`'s summation order and must match `buckets(forGeneration:)`
    /// exactly for determinism (ADR 006(f), ADR 035, ADR 041).
    func scanBuckets(forGeneration generationID: Int64) async throws -> [BucketScanRecord]

    /// Return full `BucketRecord`s (`center`, `indices`, `residuals`) for the
    /// given bucket ids. Ids that no longer exist (e.g. deleted by a
    /// concurrent vacuum/compaction between scan and fetch) are silently
    /// omitted from the result rather than causing an error. Return order is
    /// unspecified — callers that need a specific order must reconstruct it
    /// themselves (e.g. from the id sequence produced by `scanBuckets`).
    func buckets(ids: [Int64]) async throws -> [BucketRecord]

    // MARK: - Vacuum

    /// Atomically apply a pre-computed `VacuumPlan`: update or delete
    /// bucket rows, delete emptied generation rows, update surviving
    /// generations' `numEmbeddings`, and delete the plan's abandoned
    /// chunk rows.
    ///
    /// A chunk-row delete is guarded — a chunk is only actually deleted
    /// if no document row currently references its hash — so a
    /// concurrent `add()` racing between `SwitchcraftStore.vacuum()`'s
    /// detection scan and this call cannot leave a document pointing at
    /// a deleted chunk row. Returns the number of chunk rows *actually*
    /// deleted, which may be less than `plan.chunkIDsToDelete.count` if
    /// the guard fired for some chunks.
    ///
    /// The entire operation must execute in a single atomic transaction;
    /// on any error the transaction is rolled back and storage state is
    /// unchanged.
    ///
    /// Default implementation is a no-op that returns `0` — a safe
    /// opt-out for external conformers, matching the ADR 029/033
    /// default-extension pattern.
    func applyVacuumPlan(_ plan: VacuumPlan) async throws -> Int

    /// Approximate bytes currently sitting on the backend's internal
    /// free-list, sampled as `PRAGMA freelist_count × PRAGMA page_size`
    /// for SQLite backends.
    ///
    /// Used by `SwitchcraftStore.vacuum()` to compute
    /// `VacuumResult.approximateDiskReclaimed` as a before/after delta
    /// around the vacuum write phase. Meaningful only for SQLite-backed
    /// connections; the default implementation returns `0`.
    func freeListByteCount() async throws -> Int64

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

    // MARK: - Ledger snapshot (optional, fast-path startup)

    /// Persist a single ledger snapshot, replacing any previously stored one.
    ///
    /// Called by `Indexer` at points where the in-memory ledger is known to
    /// be consistent with storage (after a successful `performFlush()` commit,
    /// and at the end of a clean `SwitchcraftStore.shutdown()`), so that the
    /// next `Indexer.init` can skip the full rehydration walk. Backends store
    /// exactly one snapshot slot; a second save overwrites the first.
    ///
    /// The default implementation is a no-op — a backend that does not
    /// implement this simply never gets the startup fast path and always
    /// falls back to full rehydration. Never incorrect, just slower. See
    /// ADR 034.
    func saveLedgerSnapshot(_ snapshot: LedgerSnapshotRecord) async throws

    /// Load the persisted ledger snapshot, or `nil` if none is stored.
    ///
    /// Called once at `Indexer.init`. The default implementation returns
    /// `nil`, so backends without snapshot support always fall back to the
    /// full rehydration walk.
    func loadLedgerSnapshot() async throws -> LedgerSnapshotRecord?

    /// Clear the persisted ledger snapshot. No-op if none is stored.
    ///
    /// Called immediately after a successful fast-path load (invalidate-on-
    /// load, ADR 034) so that a crash after load but before the next
    /// successful flush/shutdown correctly falls back to full rehydration on
    /// the following startup. Also called when the whole index is cleared.
    /// The default implementation is a no-op.
    func clearLedgerSnapshot() async throws
}

extension SwitchcraftStorage {
    public func configureSearchDeadline(_ ctx: SearchDeadlineContext?) async {}
    public func walCheckpoint() async throws -> CheckpointResult { .complete }
    public func allChunks() async throws -> [ChunkRecord] { [] }
    public func chunkBucketRefCount(_ chunkID: Int64) async throws -> Int { Int.max }
    public func documentHashes() async throws -> Set<String> {
        Set(try await documents(matching: .all).map { $0.hash })
    }
    public func applyVacuumPlan(_ plan: VacuumPlan) async throws -> Int { 0 }
    public func freeListByteCount() async throws -> Int64 { 0 }
    public func saveLedgerSnapshot(_ snapshot: LedgerSnapshotRecord) async throws {}
    public func loadLedgerSnapshot() async throws -> LedgerSnapshotRecord? { nil }
    public func clearLedgerSnapshot() async throws {}
}
