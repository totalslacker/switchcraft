// SPDX-License-Identifier: Apache-2.0
import Foundation
import CryptoKit
import os
import SwitchcraftCore

/// Public, async, document-management surface over a `SwitchcraftStorage`
/// backend, an `Embedder`, and the index/search engines.
///
/// Library consumers normally interact with Switchcraft through this
/// actor:
///
/// ```swift
/// let store = try await SwitchcraftStore.sqlite(
///     databasePath: "/tmp/x.db",
///     embedder: myEmbedder
/// )
/// try await store.add(id: "doc-1", body: "some text")
/// let hits = try await store.search(query: "what is some text about?", topK: 10)
/// ```
///
/// The store performs four jobs on every `add`: embed text via the
/// supplied `Embedder`, persist a `DocumentRecord` and a `ChunkRecord`,
/// buffer per-token embeddings in the `Indexer`'s ledger, and (later)
/// flush before serving queries.
///
/// Determinism is preserved end-to-end: same `Embedder` + same `add`
/// sequence ⇒ byte-identical `HybridHit` orderings and float-equal scores.
public actor SwitchcraftStore {

    // MARK: - State

    private let logger = Logger(subsystem: "com.switchcraft", category: "Store")
    private let storage: any SwitchcraftStorage
    private let embedder: any Embedder
    private var indexer: Indexer!
    private let searchEngine: SearchEngine
    public let config: StoreConfig

    private var isShutDown: Bool = false

    private var onCompactionEvent: (@Sendable (CompactionEvent) async -> Void)?

    /// Register or replace the compaction event callback.
    ///
    /// The callback fires once per compaction that produces at least one output
    /// generation, invoked from the `SwitchcraftStore` actor after the flush
    /// completes. Fields on the delivered `CompactionEvent` reflect the final
    /// post-recovery state (see ADR 030 and ADR 032): if mid-operation
    /// divergence recovery absorbed additional generations, `inputBytes` and
    /// `inputSegments` will be larger than the initial cascade-walk estimates.
    ///
    /// The callback does **not** fire for no-op flushes (flush called when the
    /// L0 buffer is empty and no cascade is triggered).
    ///
    /// Pass `nil` to clear a previously registered callback.
    ///
    /// > Warning: Do not call store methods that themselves trigger a flush
    /// > (e.g. `index()`) from within this callback. Doing so is safe (calls
    /// > queue behind the current actor turn via Swift actor re-entrancy), but
    /// > can cause unexpected recursive compaction bursts.
    public func setOnCompactionEvent(_ callback: (@Sendable (CompactionEvent) async -> Void)?) {
        onCompactionEvent = callback
    }

    // Tracks chunkIDs passed to `indexer.add()` since the last flush.
    // Guards against double-buffering when a pending-but-unflushed chunk
    // is re-added with matching content (R2 of issue #120).
    private var pendingChunkIDs: Set<Int64> = []

    // MARK: - Init

    /// Build a store over any `SwitchcraftStorage`.
    ///
    /// Opens storage and rehydrates the indexer's in-memory ledger from any
    /// existing generations, so a reopened store can immediately serve both
    /// `search` and `add` without `Indexer.Error.ledgerOutOfSync`.
    ///
    /// - Parameters:
    ///   - storage: backend implementation of `SwitchcraftStorage`.
    ///   - embedder: model used for both write-time and query-time encodes.
    ///     `dims` must be positive and even.
    ///   - config: tunables for the indexer, search engine, and hybrid
    ///     fusion stages. Defaults match upstream Witchcraft.
    /// - Throws: `SwitchcraftStoreError.invalidEmbeddingDimensions` if the
    ///   embedder reports a non-positive or odd `dims`; any error thrown
    ///   by `storage.open()` or `Indexer.init`.
    public init(
        storage: any SwitchcraftStorage,
        embedder: any Embedder,
        config: StoreConfig = .default
    ) async throws {
        guard embedder.dims > 0, embedder.dims % 2 == 0 else {
            throw SwitchcraftStoreError.invalidEmbeddingDimensions(embedder.dims)
        }
        self.storage = storage
        self.embedder = embedder
        self.config = config
        self.searchEngine = SearchEngine(storage: storage, config: config.search)
        // Open storage before constructing the Indexer so that rehydration
        // queries (generations, buckets) find an open connection.
        try await storage.open()
        self.indexer = try await Indexer(storage: storage, config: config.indexer)
    }

    // MARK: - Document management

    /// Upsert a document. Calling `add` twice with the same `id` replaces
    /// the previous document; the new body's embeddings supersede the
    /// old ones for search purposes (the old chunk's embeddings are left
    /// in storage as orphans — see the package's known-limitations notes).
    ///
    /// `metadata` is JSON-encoded with sorted keys so byte-identical input
    /// produces byte-identical `DocumentRecord.metadata`.
    ///
    /// When `title` is supplied, the embedder sees a title-prefixed text rather
    /// than `body` alone. For rich titles (`title.count ≥ 20`) the formula is
    /// `"\(title)\n\(body)"` (ADR 025 §a). For sparse titles (`title.count < 20`)
    /// a 200-character body excerpt is also prepended — `"\(title)\n\(body.prefix(200))\n\(body)"` —
    /// so body tokens appear near T5 position 0 rather than being shifted right
    /// by a short title (ADR 025 §h). See ADR 025 for the full rationale and
    /// hash-semantics.
    ///
    /// The chunk dedup key is `SHA-256(embeddingText)` — so two documents
    /// with identical bodies but different titles produce distinct chunks with
    /// correct per-title embeddings. When `title` is nil, `embeddingText ==
    /// body` and the hash is identical to the pre-025 formula; existing callers
    /// are unaffected.
    ///
    /// `DocumentRecord.body` always stores the caller-supplied `body` (never
    /// the title-prefixed embedding text). The title is stored in
    /// `DocumentRecord.title` and indexed in the FTS pipeline with a
    /// configurable BM25 column weight (`HybridConfig.ftsTitleWeight`). See
    /// ADR 026 for the schema and migration details.
    ///
    /// ## Content-hash dedup and orphan recovery
    ///
    /// `add()` deduplicates chunks by `SHA-256(embeddingText)`. When a chunk
    /// row already exists for that hash, the dedup branch applies a three-way
    /// check before deciding whether to call `indexer.add()`:
    ///
    /// 1. **Pending (R2):** if the existing chunk was already passed to
    ///    `indexer.add()` in this store's lifetime but not yet flushed, skip
    ///    `indexer.add()` — it is already buffered. Double-buffering the same
    ///    chunkID violates the Indexer's single-reference invariant.
    /// 2. **Indexed (R3):** if the existing chunk's committed bucket reference
    ///    count equals its expected token count, skip `indexer.add()` — this
    ///    is the normal, fully-indexed dedup case.
    /// 3. **Incomplete (R1):** if the existing chunk has fewer bucket refs
    ///    than its expected token count (including zero), and is not pending,
    ///    clear any stale rehydrated ledger rows via `indexer.removeFromLedger()`
    ///    and call `indexer.add()` with the freshly computed embeddings to
    ///    recover it. This handles both full orphans (zero refs, caused by a
    ///    crash between `upsertChunk` and `indexer.add()`) and partial-indexing
    ///    anomalies (some refs but not all). The recovery is always-on; no
    ///    opt-in flag is required.
    ///
    /// Use `SwitchcraftStore.findOrphanedChunks()` to enumerate all orphans
    /// already present in storage that pre-date this fix.
    ///
    /// See ADR 029 for the full orphan failure-mode analysis and recovery
    /// contract.
    ///
    /// - Note: **Bulk-indexing callers** (e.g. indexing thousands of documents
    ///   in a single process) should call `embedder.resetState()` periodically
    ///   between batches to release the ANE IOSurface pool and prevent
    ///   throughput degradation. The store does not expose the embedder
    ///   directly — callers hold the `Embedder` reference they supplied at
    ///   `SwitchcraftStore.init` and call `resetState()` on it. See ADR 031.
    ///
    /// - Parameters:
    ///   - id: caller-supplied document identifier (UUID, slug, etc.).
    ///   - date: date associated with the document. Defaults to `now`.
    ///   - metadata: opaque key/value pairs filterable via `StorageFilter`.
    ///   - title: optional document title. Prepended to `body` at embedding
    ///     time and stored in `DocumentRecord.title` for FTS title-weight
    ///     boosting. See ADR 025 and ADR 026.
    ///   - body: searchable text. Embedded by the configured `Embedder`.
    /// - Throws: `SwitchcraftStoreError.alreadyShutDown` if the store has
    ///   been shut down; `SwitchcraftStoreError.embeddingMismatch` if the
    ///   embedder returns a buffer length that is not a multiple of `dims`;
    ///   storage- or embedder-originated errors otherwise.
    public func add(
        id: String,
        date: Date = Date(),
        metadata: [String: String] = [:],
        title: String? = nil,
        body: String
    ) async throws {
        try ensureRunning()

        // ADR 025 §(h): For sparse titles (< 20 chars), prepend a body excerpt so
        // body tokens appear near T5 position 0 — restoring the near-context shape
        // they had before the title shift introduced by §(a).
        let embeddingText: String
        if let title {
            embeddingText = title.count < 20
                ? "\(title)\n\(body.prefix(200))\n\(body)"
                : "\(title)\n\(body)"
        } else {
            embeddingText = body
        }

        // TODO: paragraph splitter — single chunk per body for v1.
        let embeddings = try await embedder.encode(embeddingText)
        let dims = embedder.dims
        guard embeddings.count % dims == 0 else {
            throw SwitchcraftStoreError.embeddingMismatch(
                count: embeddings.count, dims: dims
            )
        }
        let tokenCount = embeddings.count / dims

        let hash = Self.contentHash(embeddingText)
        let metadataData = try Self.encodeMetadata(metadata)

        // Pre-check chunk existence to decide whether embeddings need
        // buffering. Three-way dedup logic (see doc comment above):
        let existing = try await storage.chunk(hash: hash)
        let chunkID: Int64
        if let existing {
            chunkID = existing.id
            let expectedRefs = existing.counts.reduce(0, +)
            if pendingChunkIDs.contains(chunkID) {
                // R2: already buffered in the indexer ledger — skip.
            } else if try await storage.chunkBucketRefCount(chunkID) >= expectedRefs {
                // R3: fully indexed (actualRefs == expectedTokenCount) — skip.
            } else if tokenCount > 0 {
                // R1: incomplete — zero or partial bucket refs; clear stale
                // rehydrated ledger rows then re-buffer to recover.
                try await indexer.removeFromLedger(chunkID)
                try await indexer.add(chunkID: chunkID, embeddings: embeddings, dims: dims)
                pendingChunkIDs.insert(chunkID)
            }
        } else {
            let inserted = try await storage.upsertChunk(
                ChunkRecord(
                    hash: hash,
                    model: embedder.modelIdentifier,
                    embeddings: Data(),
                    counts: [tokenCount]
                )
            )
            chunkID = inserted.id
            if tokenCount > 0 {
                try await indexer.add(
                    chunkID: chunkID,
                    embeddings: embeddings,
                    dims: dims
                )
                pendingChunkIDs.insert(chunkID)
            }
        }

        try await storage.upsertDocument(
            DocumentRecord(
                uuid: id,
                date: date,
                metadata: metadataData,
                hash: hash,
                body: body,
                title: title,
                lens: [tokenCount]
            )
        )
    }

    /// Remove the document identified by `id`. No-op if absent. The
    /// content chunk and its buffered embeddings are left in place — the
    /// search engine drops orphan chunks because no document maps to the
    /// chunk's hash.
    ///
    /// - Parameter id: document identifier supplied to `add(id:body:)`.
    /// - Throws: `SwitchcraftStoreError.alreadyShutDown`; storage errors.
    public func remove(id: String) async throws {
        try ensureRunning()
        try await storage.deleteDocument(uuid: id)
    }

    /// Flush any pending L0 embeddings into the LSM tree. Called
    /// automatically before every `search` and `score`, so most callers
    /// never need to invoke it directly.
    ///
    /// - Throws: `SwitchcraftStoreError.alreadyShutDown`; indexer errors.
    public func index() async throws {
        try ensureRunning()
        try await flushAndClearPending()
    }

    /// Return all chunk rows that have no, or only partial, bucket
    /// assignments across all committed generations.
    ///
    /// A chunk is an orphan when `storage.upsertChunk()` succeeded but a
    /// subsequent crash or error prevented `indexer.add()` from running.
    /// Orphaned chunks are visible in FTS results but invisible to vector
    /// search.
    ///
    /// ## Recovery contract
    ///
    /// For each returned `OrphanedChunkInfo`, re-add any document in
    /// `owningDocuments` through `store.add(id:body:)`.  `add()` detects
    /// the missing bucket assignments and calls `indexer.add()` automatically;
    /// the next `store.search()` or explicit `store.index()` call flushes
    /// the recovered embeddings into the bucket index.
    ///
    /// This method operates on committed storage state only and does not
    /// require a flush before being called.
    ///
    /// - Throws: `SwitchcraftStoreError.alreadyShutDown`; storage errors;
    ///   `IndicesCodec.Error` if a bucket blob is corrupt.
    public func findOrphanedChunks() async throws -> [OrphanedChunkInfo] {
        try ensureRunning()

        let chunks = try await storage.allChunks()
        guard !chunks.isEmpty else { return [] }

        // Build a map of chunkID → bucket reference count by decoding all blobs.
        var refCounts: [Int64: Int] = [:]
        let gens = try await storage.generations()
        for gen in gens {
            let buckets = try await storage.buckets(forGeneration: gen.id)
            for bucket in buckets {
                let pairs = try IndicesCodec.decode(bucket.indices)
                for pair in pairs {
                    let cid = Int64(pair.chunkID)
                    refCounts[cid, default: 0] += 1
                }
            }
        }

        var orphans: [OrphanedChunkInfo] = []
        for chunk in chunks {
            let expected = chunk.counts.reduce(0, +)
            let refs = refCounts[chunk.id] ?? 0
            if refs < expected {
                let docs = try await storage.documents(forChunkHash: chunk.hash)
                let uuids = docs.map { $0.uuid }
                orphans.append(OrphanedChunkInfo(
                    chunkID: chunk.id,
                    hash: chunk.hash,
                    model: chunk.model,
                    expectedTokenCount: expected,
                    bucketReferenceCount: refs,
                    owningDocuments: uuids
                ))
            }
        }
        return orphans
    }

    /// Remove chunks that have no owning document ("abandoned" chunks) and
    /// reclaim the disk they occupy.
    ///
    /// A chunk becomes abandoned through natural re-indexing (a page's
    /// content changes, producing a new hash; the old chunk is left
    /// behind), through `remove(id:)` (which deletes the document row but
    /// leaves the chunk and its bucket entries in place), or through bulk
    /// delete-then-readd workflows (e.g. URL canonicalization passes).
    /// Unlike `findOrphanedChunks()` — which targets chunks with
    /// insufficient bucket assignments but a still-existing owning
    /// document, and are recovery candidates — `vacuum()` targets chunks
    /// whose `hash` does not appear as any `document.hash`, which are
    /// removal candidates. A chunk with an owning document is never
    /// removed, even if it is only partially indexed (see
    /// `findOrphanedChunks()` for that recovery path).
    ///
    /// ## Batch semantics
    ///
    /// `vacuum(maxBatch: N)` processes at most `N` abandoned chunks and
    /// returns — it does not internally loop to process the full backlog.
    /// `vacuum(maxBatch: nil)` processes every abandoned chunk currently
    /// known. `VacuumResult.remainingCandidates` reports how many
    /// abandoned chunks this call left unprocessed; call `vacuum()` again
    /// (repeatedly, if needed) until it reaches `0` to drain a large
    /// backlog. Repeated calls are idempotent: a chunk already deleted by
    /// a prior call is simply absent from the next call's candidate scan.
    ///
    /// ## Cascade deletion
    ///
    /// If removing a chunk's bucket entries leaves a bucket with zero
    /// surviving `(chunkID, tokenOffset)` pairs, that bucket row is
    /// deleted. If deleting a bucket leaves its generation with zero
    /// buckets, the generation row is deleted too. Surviving generations
    /// have their `numEmbeddings` corrected to match the post-removal
    /// bucket state. `VacuumResult.generationsAffected` includes both
    /// partially-rewritten and fully-deleted generation ids.
    ///
    /// ## Ledger consistency
    ///
    /// Before this method returns, the in-memory `Indexer` ledger is
    /// updated to match the post-vacuum committed storage state — vacuum
    /// owns this itself rather than relying on the mid-add/mid-compaction
    /// self-recovery mechanisms from ADR 029/030, which target different
    /// drift classes. A subsequent `add()`, `index()`, or `search()` call
    /// will not throw `Indexer.Error.ledgerOutOfSync` as a result of a
    /// prior `vacuum()` call.
    ///
    /// ## Concurrency contract
    ///
    /// `vacuum()` flushes pending writes first (mirroring
    /// `walCheckpoint()`'s contract), so a chunk added-then-immediately
    /// document-deleted before its own flush is never misclassified as
    /// abandoned. Do not call `add()`/`index()` concurrently with
    /// `vacuum()` from the same process: a document re-add racing between
    /// vacuum's detection scan and its write phase could match a candidate
    /// chunk. The chunk-row delete is guarded against this (a raced re-add
    /// cannot leave a document pointing at a deleted chunk row — the
    /// worst-case outcome is the chunk surviving as a fresh true-orphan,
    /// self-healing via the existing `findOrphanedChunks()` → `add()`
    /// recovery path), but bucket-blob rewrites are not re-validated after
    /// the scan, so avoid concurrent same-process writers as a rule.
    ///
    /// - Parameter maxBatch: maximum number of abandoned chunks to process
    ///   in this call. `nil` (the default) processes all currently-known
    ///   abandoned chunks.
    /// - Returns: counts of what this call removed/updated, plus
    ///   `remainingCandidates` for backlog-draining loops.
    /// - Throws: `SwitchcraftStoreError.alreadyShutDown`; storage errors;
    ///   `IndicesCodec.Error` if a bucket blob is corrupt.
    public func vacuum(maxBatch: Int? = nil) async throws -> VacuumResult {
        try ensureRunning()
        try await flushAndClearPending()

        let allChunkRecords = try await storage.allChunks()
        guard !allChunkRecords.isEmpty else {
            return VacuumResult(
                chunksRemoved: 0, bucketPairsRemoved: 0, approximateDiskReclaimed: 0,
                generationsAffected: [], remainingCandidates: 0, checkpoint: .complete
            )
        }

        let docHashes = try await storage.documentHashes()
        let abandonedChunks = allChunkRecords
            .filter { !docHashes.contains($0.hash) }
            .sorted { $0.id < $1.id }

        guard !abandonedChunks.isEmpty else {
            return VacuumResult(
                chunksRemoved: 0, bucketPairsRemoved: 0, approximateDiskReclaimed: 0,
                generationsAffected: [], remainingCandidates: 0, checkpoint: .complete
            )
        }

        let effectiveBatchSize = maxBatch.map { max(0, $0) } ?? abandonedChunks.count
        let batch = Array(abandonedChunks.prefix(effectiveBatchSize))
        let remainingCandidates = abandonedChunks.count - batch.count

        guard !batch.isEmpty else {
            return VacuumResult(
                chunksRemoved: 0, bucketPairsRemoved: 0, approximateDiskReclaimed: 0,
                generationsAffected: [], remainingCandidates: remainingCandidates, checkpoint: .complete
            )
        }

        let batchChunkIDs = Set(batch.map(\.id))
        let gens = try await storage.generations()
        let planResult = try await VacuumPlanBuilder.buildPlan(
            abandonedChunkIDs: batchChunkIDs,
            generations: gens,
            fetchBuckets: { [storage] genID in try await storage.buckets(forGeneration: genID) }
        )

        let preBytes = try await storage.freeListByteCount()
        let deletedCount = try await storage.applyVacuumPlan(planResult.plan)
        let postBytes = try await storage.freeListByteCount()
        let reclaimed = max(0, postBytes - preBytes)

        try await indexer.removeAbandonedFromLedger(batchChunkIDs)

        let checkpoint = try await storage.walCheckpoint()

        return VacuumResult(
            chunksRemoved: deletedCount,
            bucketPairsRemoved: planResult.bucketPairsRemoved,
            approximateDiskReclaimed: reclaimed,
            generationsAffected: planResult.generationsAffected.sorted(),
            remainingCandidates: remainingCandidates,
            checkpoint: checkpoint
        )
    }

    /// Wipe all documents, chunks, and LSM generations. The backing
    /// storage file is left in place (tables are emptied).
    ///
    /// - Throws: `SwitchcraftStoreError.alreadyShutDown`; storage errors.
    public func clear() async throws {
        try ensureRunning()
        try await indexer.clearIndex()
        try await storage.clear()
    }

    /// Return the total number of documents in the store.
    ///
    /// - Throws: `SwitchcraftStoreError.alreadyShutDown`; storage errors.
    public func documentCount() async throws -> Int {
        try ensureRunning()
        return try await storage.documentCount()
    }

    /// Return the set of all document UUIDs present in the store.
    ///
    /// Used by external backfill machinery to identify which pages have
    /// already been indexed so they can be skipped during catch-up runs.
    ///
    /// - Throws: `SwitchcraftStoreError.alreadyShutDown`; storage errors.
    public func indexedURLs() async throws -> Set<String> {
        try ensureRunning()
        return try await storage.indexedURLs()
    }

    /// Flush pending indexer writes, then run `wal_checkpoint(TRUNCATE)`.
    ///
    /// Flushing the indexer first ensures that all `add()` calls issued before
    /// this call are durable on disk when the method returns — matching the
    /// intuitive "I just checkpointed, my data is safe" contract. Without the
    /// flush, buffered indexer writes would survive a call to this method but
    /// be lost on a crash that followed.
    ///
    /// Returns `.partial(framesRemaining:)` when a reader connection blocked
    /// WAL truncation. Data is durable; only the WAL file compaction is
    /// incomplete. Retry after the reader closes, or let the next
    /// `shutdown()` drain it. No-op for in-memory stores (always `.complete`).
    ///
    /// - Returns: `.complete` when the WAL was truncated; `.partial` when
    ///   a reader blocked truncation.
    /// - Throws: `SwitchcraftStoreError.alreadyShutDown`; indexer or storage errors.
    public func walCheckpoint() async throws -> CheckpointResult {
        try ensureRunning()
        try await flushAndClearPending()
        return try await storage.walCheckpoint()
    }

    // MARK: - Search

    /// Hybrid vector + BM25 search via Reciprocal Rank Fusion. Calls
    /// `index()` internally so callers never need to flush manually.
    ///
    /// - Parameters:
    ///   - query: free-text query.
    ///   - topK: maximum number of fused hits to return. Defaults to 10.
    ///   - filter: document predicate applied before fusion.
    ///   - deadline: per-call wall-time budget measured from the instant
    ///     this method is entered. Overrides `config.search.searchDeadline`
    ///     when non-nil. Pass `.seconds(0)` to enforce a pre-SQLite-only
    ///     check (useful in tests). When the budget is exhausted the call
    ///     throws `SwitchcraftStoreError.searchTimedOut(elapsed:)`.
    /// - Returns: hits sorted by `(score DESC, uuid ASC)`.
    /// - Throws: `SwitchcraftStoreError.alreadyShutDown` /
    ///   `embeddingMismatch` / `searchTimedOut`; storage and embedder errors.
    public func search(
        query: String,
        topK: Int = 10,
        filter: StorageFilter = .all,
        deadline: Duration? = nil
    ) async throws -> [HybridHit] {
        try ensureRunning()

        let searchStart = ContinuousClock.now
        let effectiveDeadline = deadline ?? config.search.searchDeadline
        let deadlineCtx = SearchDeadlineContext(
            searchStart: searchStart,
            deadline: effectiveDeadline
        )

        try await flushAndClearPending()

        // Pre-SQLite checkpoint: flush may have consumed the budget.
        if deadlineCtx.isExpired {
            throw SwitchcraftStoreError.searchTimedOut(elapsed: deadlineCtx.elapsed)
        }

        let queryEmbeddings = try await embedder.encodeQuery(
            query,
            minSurfaceFormLength: config.search.queryMinSurfaceFormLength
        )
        let dims = embedder.dims
        guard queryEmbeddings.count % dims == 0 else {
            throw SwitchcraftStoreError.embeddingMismatch(
                count: queryEmbeddings.count, dims: dims
            )
        }

        // Pre-SQLite checkpoint: encode may have consumed the budget.
        if deadlineCtx.isExpired {
            throw SwitchcraftStoreError.searchTimedOut(elapsed: deadlineCtx.elapsed)
        }

        // Arm the progress handler with the remaining budget just before
        // entering the SQL phase. Disarmed in all branches below so that
        // write operations after search never see a stale armed handler.
        await storage.configureSearchDeadline(deadlineCtx)

        do {
            let hits = try await searchEngine.searchHybrid(
                queryEmbeddings: queryEmbeddings,
                dims: dims,
                queryText: query,
                topK: topK,
                filter: filter,
                config: config.hybrid,
                deadlineContext: deadlineCtx
            )
            await storage.configureSearchDeadline(nil)
            return hits
        } catch let e as SearchEngine.Error {
            await storage.configureSearchDeadline(nil)
            if case .deadlineExceeded(let elapsed) = e {
                throw SwitchcraftStoreError.searchTimedOut(elapsed: elapsed)
            }
            throw e
        } catch {
            await storage.configureSearchDeadline(nil)
            throw error
        }
    }

    /// Score `passages` against `query`. Returns one MaxSim score per
    /// passage in input order. Reproducible for the same `Embedder`.
    ///
    /// - Parameters:
    ///   - query: free-text query.
    ///   - passages: candidate passages, in caller-controlled order.
    /// - Returns: per-passage MaxSim scores aligned with `passages`.
    /// - Throws: `SwitchcraftStoreError.alreadyShutDown` /
    ///   `embeddingMismatch`; embedder errors.
    public func score(query: String, passages: [String]) async throws -> [Float] {
        try ensureRunning()
        try await flushAndClearPending()
        let queryEmbeddings = try await embedder.encode(query)
        let dims = embedder.dims
        guard queryEmbeddings.count % dims == 0 else {
            throw SwitchcraftStoreError.embeddingMismatch(
                count: queryEmbeddings.count, dims: dims
            )
        }
        var passageEmbeddings: [[Float]] = []
        passageEmbeddings.reserveCapacity(passages.count)
        for passage in passages {
            let emb = try await embedder.encode(passage)
            guard emb.count % dims == 0 else {
                throw SwitchcraftStoreError.embeddingMismatch(
                    count: emb.count, dims: dims
                )
            }
            passageEmbeddings.append(emb)
        }
        return try await searchEngine.score(
            queryEmbeddings: queryEmbeddings,
            dims: dims,
            passages: passageEmbeddings
        )
    }

    // MARK: - Lifecycle

    /// Flush pending writes, close the underlying storage, and mark the
    /// store as shut down. Idempotent: a second call is a no-op. After
    /// shutdown, every other public method throws
    /// `SwitchcraftStoreError.alreadyShutDown`.
    ///
    /// - Throws: indexer or storage errors raised during flush/close.
    public func shutdown() async throws {
        if isShutDown { return }
        // Set the flag before any `await` so calls that arrive while
        // the actor is suspended in flush/close observe shutdown and
        // throw `alreadyShutDown` instead of racing through gates.
        isShutDown = true
        try await flushAndClearPending()
        // Checkpoint the WAL before closing so that the next launch finds
        // a clean database file. Skipped on error — a partial WAL is still
        // better than a half-checkpointed file, and `close()` handles final
        // cleanup regardless. Calls storage directly (not self.walCheckpoint())
        // to avoid re-entering the isShutDown guard and to skip a second flush.
        if case .partial(let frames) = try? await storage.walCheckpoint() {
            logger.error(
                "WAL checkpoint partial on shutdown: \(frames) frames remaining"
            )
        }
        try await storage.close()
    }

    // MARK: - Helpers

    private func ensureRunning() throws {
        if isShutDown { throw SwitchcraftStoreError.alreadyShutDown }
    }

    private func flushAndClearPending() async throws {
        let event = try await indexer.flush()
        pendingChunkIDs.removeAll()
        if let event {
            await onCompactionEvent?(event)
        }
    }

    /// SHA-256 of the body's UTF-8 bytes, hex-encoded. Used as the chunk
    /// dedup key. Stable across processes and Swift versions.
    static func contentHash(_ body: String) -> String {
        let digest = SHA256.hash(data: Data(body.utf8))
        var out = ""
        out.reserveCapacity(SHA256.Digest.byteCount * 2)
        for byte in digest {
            let hi = byte >> 4
            let lo = byte & 0x0F
            out.append(Self.hexDigit(hi))
            out.append(Self.hexDigit(lo))
        }
        return out
    }

    private static func hexDigit(_ nibble: UInt8) -> Character {
        nibble < 10
            ? Character(UnicodeScalar(0x30 + nibble))
            : Character(UnicodeScalar(0x61 + (nibble - 10)))
    }

    /// JSON-encode a `[String: String]` metadata dict using sorted keys
    /// so equal dicts always produce byte-identical `Data`. Required for
    /// the determinism acceptance criterion.
    static func encodeMetadata(_ metadata: [String: String]) throws -> Data {
        if metadata.isEmpty { return Data() }
        return try JSONSerialization.data(
            withJSONObject: metadata,
            options: [.sortedKeys]
        )
    }
}
