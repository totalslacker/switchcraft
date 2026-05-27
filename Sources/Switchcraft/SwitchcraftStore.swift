// SPDX-License-Identifier: Apache-2.0
import Foundation
import CryptoKit
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

    private let storage: any SwitchcraftStorage
    private let embedder: any Embedder
    private var indexer: Indexer!
    private let searchEngine: SearchEngine
    public let config: StoreConfig

    private var isShutDown: Bool = false

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
    /// - Parameters:
    ///   - id: caller-supplied document identifier (UUID, slug, etc.).
    ///   - date: date associated with the document. Defaults to `now`.
    ///   - metadata: opaque key/value pairs filterable via `StorageFilter`.
    ///   - body: searchable text. Embedded by the configured `Embedder`.
    /// - Throws: `SwitchcraftStoreError.alreadyShutDown` if the store has
    ///   been shut down; `SwitchcraftStoreError.embeddingMismatch` if the
    ///   embedder returns a buffer length that is not a multiple of `dims`;
    ///   storage- or embedder-originated errors otherwise.
    public func add(
        id: String,
        date: Date = Date(),
        metadata: [String: String] = [:],
        body: String
    ) async throws {
        try ensureRunning()

        // TODO: paragraph splitter — single chunk per body for v1.
        let embeddings = try await embedder.encode(body)
        let dims = embedder.dims
        guard embeddings.count % dims == 0 else {
            throw SwitchcraftStoreError.embeddingMismatch(
                count: embeddings.count, dims: dims
            )
        }
        let tokenCount = embeddings.count / dims

        let hash = Self.contentHash(body)
        let metadataData = try Self.encodeMetadata(metadata)

        // Pre-check chunk existence so we only buffer embeddings into the
        // indexer ledger when the chunk is genuinely new. `upsertChunk`
        // alone returns the existing record on hash collision but does
        // not signal whether an insert happened, so we'd risk
        // double-buffering identical embeddings under the same chunkID.
        let existing = try await storage.chunk(hash: hash)
        let chunkID: Int64
        if let existing {
            chunkID = existing.id
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
            }
        }

        try await storage.upsertDocument(
            DocumentRecord(
                uuid: id,
                date: date,
                metadata: metadataData,
                hash: hash,
                body: body,
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
        try await indexer.flush()
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

    /// Flush any pending WAL writes to the main database file and truncate
    /// the WAL. No-op for in-memory stores.
    ///
    /// Call this during graceful shutdown to ensure the database is in a
    /// clean state before process exit. If the WAL is not checkpointed and
    /// the process is killed, the next launch may encounter a rehydration
    /// conflict during index reconstruction.
    ///
    /// - Throws: `SwitchcraftStoreError.alreadyShutDown`; storage errors.
    public func walCheckpoint() async throws {
        try ensureRunning()
        try await storage.walCheckpoint()
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

        try await indexer.flush()

        // Pre-SQLite checkpoint: flush may have consumed the budget.
        if deadlineCtx.isExpired {
            throw SwitchcraftStoreError.searchTimedOut(elapsed: deadlineCtx.elapsed)
        }

        let queryEmbeddings = try await embedder.encode(query)
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
        try await indexer.flush()
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
        try await indexer.flush()
        // Checkpoint the WAL before closing so that the next launch finds
        // a clean database file. Skipped on error — a partial WAL is still
        // better than a half-checkpointed file, and `close()` handles final
        // cleanup regardless.
        try? await storage.walCheckpoint()
        try await storage.close()
    }

    // MARK: - Helpers

    private func ensureRunning() throws {
        if isShutDown { throw SwitchcraftStoreError.alreadyShutDown }
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
