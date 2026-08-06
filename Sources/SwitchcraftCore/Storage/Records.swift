// SPDX-License-Identifier: Apache-2.0
import Foundation

/// A document stored in the index.
///
/// The `body` is the searchable text. `title` is the optional document title,
/// stored separately and indexed for BM25 matching with configurable weight
/// (see `HybridConfig.ftsTitleWeight` and ADR 026). `metadata` is opaque
/// JSON-encoded data that callers can attach for filtering or display. `hash`
/// is a content hash used for chunk deduplication. `lens` records the chunk
/// lengths produced when the body was split for embedding.
public struct DocumentRecord: Sendable, Hashable {
    public var uuid: String
    public var date: Date
    public var metadata: Data
    public var hash: String
    public var body: String
    public var title: String?
    public var lens: [Int]

    public init(
        uuid: String,
        date: Date,
        metadata: Data = Data(),
        hash: String,
        body: String,
        title: String? = nil,
        lens: [Int] = []
    ) {
        self.uuid = uuid
        self.date = date
        self.metadata = metadata
        self.hash = hash
        self.body = body
        self.title = title
        self.lens = lens
    }
}

/// A deduplicated content chunk with its packed token embeddings.
///
/// `id` is assigned by the backend on insert and is monotonically increasing.
/// `hash` is the dedup key — inserting a chunk with an existing hash returns
/// the existing record. `embeddings` is the packed per-token embedding blob.
public struct ChunkRecord: Sendable, Hashable {
    public var id: Int64
    public var hash: String
    public var model: String
    public var embeddings: Data
    public var counts: [Int]

    /// Sentinel id used when constructing a chunk for insert.
    public static let unassigned: Int64 = 0

    public init(
        id: Int64 = Self.unassigned,
        hash: String,
        model: String,
        embeddings: Data,
        counts: [Int] = []
    ) {
        self.id = id
        self.hash = hash
        self.model = model
        self.embeddings = embeddings
        self.counts = counts
    }
}

/// One LSM-tree generation (index level).
///
/// Generations track ranges of chunk IDs that have been clustered into
/// buckets at a particular level. New writes accumulate at L0 and cascade
/// upward as levels fill.
public struct GenerationRecord: Sendable, Hashable {
    public var id: Int64
    public var level: Int
    public var numEmbeddings: Int
    public var minChunkID: Int64
    public var maxChunkID: Int64
    public var created: Date

    /// Sentinel id used when constructing a generation for insert.
    public static let unassigned: Int64 = 0

    public init(
        id: Int64 = Self.unassigned,
        level: Int,
        numEmbeddings: Int,
        minChunkID: Int64,
        maxChunkID: Int64,
        created: Date
    ) {
        self.id = id
        self.level = level
        self.numEmbeddings = numEmbeddings
        self.minChunkID = minChunkID
        self.maxChunkID = maxChunkID
        self.created = created
    }
}

/// One k-means cluster in a generation.
///
/// `center` holds the centroid (typically 128 × F32). `indices` holds the
/// compressed list of chunk pointers that fell into this cluster.
/// `residuals` holds the Q4-quantised residual vectors for those pointers.
public struct BucketRecord: Sendable, Hashable {
    public var id: Int64
    public var generationID: Int64
    public var center: Data
    public var indices: Data
    public var residuals: Data

    /// Sentinel id used when constructing a bucket for insert.
    public static let unassigned: Int64 = 0

    public init(
        id: Int64 = Self.unassigned,
        generationID: Int64,
        center: Data,
        indices: Data,
        residuals: Data
    ) {
        self.id = id
        self.generationID = generationID
        self.center = center
        self.indices = indices
        self.residuals = residuals
    }
}

/// A lightweight, scan-phase view of a bucket used during the centroid scan.
///
/// The centroid scan (see `SearchEngine`) only needs the centroid (to score
/// against the query via `cblas_sgemm`) and the residual token count (to
/// drive the cumulative-budget `tPrime` selection logic) for every bucket in
/// a generation — `indices` and `residuals` are only needed for the small
/// subset of buckets selection actually picks. `residualByteCount` is the
/// raw byte length of the `residuals` blob (not divided by dims/2 into a
/// token count) because the storage layer does not track `dims`; callers
/// derive the token count themselves. See ADR 041.
public struct BucketScanRecord: Sendable, Hashable {
    public var id: Int64
    public var generationID: Int64
    public var center: Data
    public var residualByteCount: Int

    public init(
        id: Int64,
        generationID: Int64,
        center: Data,
        residualByteCount: Int
    ) {
        self.id = id
        self.generationID = generationID
        self.center = center
        self.residualByteCount = residualByteCount
    }
}

/// A persisted point-in-time snapshot of the indexer's in-memory ledger.
///
/// Written at points where the ledger is known-consistent with storage
/// (after a successful `performFlush()` commit, and at the end of a clean
/// `SwitchcraftStore.shutdown()`), this lets `Indexer.init` skip the
/// O(all embeddings) rehydration walk on the next clean-shutdown startup.
///
/// The `payload` blob encodes the full `[Int64: [[Float]]]` ledger (see
/// `LedgerSnapshotCodec`). The five fingerprint fields (`chunkCount`,
/// `maxChunkID`, `totalEmbeddings`, `maxGenerationID`, `generationCount`)
/// are a cheap staleness check computed from storage metadata that is
/// already available without decoding buckets — if any of them disagree
/// with freshly-computed storage state at the next `init`, the snapshot is
/// treated as stale and the full rehydration walk runs instead. This is a
/// fast staleness/corruption check, not a cryptographic integrity
/// guarantee; correctness is backstopped by the full-walk fallback. See
/// ADR 034.
public struct LedgerSnapshotRecord: Sendable, Hashable {
    /// Vector dimensionality locked in by the ledger at snapshot time.
    public var dims: Int
    /// `storage.chunkCount()` at snapshot time.
    public var chunkCount: Int
    /// `max(gen.maxChunkID)` across all generations at snapshot time.
    public var maxChunkID: Int64
    /// Sum of `gen.numEmbeddings` across all generations at snapshot time.
    public var totalEmbeddings: Int
    /// `max(gen.id)` across all generations at snapshot time.
    public var maxGenerationID: Int64
    /// Number of generations at snapshot time.
    public var generationCount: Int
    /// The encoded ledger payload (see `LedgerSnapshotCodec.encode`).
    public var payload: Data

    public init(
        dims: Int,
        chunkCount: Int,
        maxChunkID: Int64,
        totalEmbeddings: Int,
        maxGenerationID: Int64,
        generationCount: Int,
        payload: Data
    ) {
        self.dims = dims
        self.chunkCount = chunkCount
        self.maxChunkID = maxChunkID
        self.totalEmbeddings = totalEmbeddings
        self.maxGenerationID = maxGenerationID
        self.generationCount = generationCount
        self.payload = payload
    }
}

/// A scored full-text search hit returned by a storage backend.
public struct FullTextHit: Sendable, Hashable {
    public var uuid: String
    public var score: Float

    public init(uuid: String, score: Float) {
        self.uuid = uuid
        self.score = score
    }
}
