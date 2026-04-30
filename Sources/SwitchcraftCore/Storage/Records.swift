// SPDX-License-Identifier: Apache-2.0
import Foundation

/// A document stored in the index.
///
/// The `body` is the searchable text. `metadata` is opaque JSON-encoded data
/// that callers can attach for filtering or display. `hash` is a content hash
/// used for chunk deduplication. `lens` records the chunk lengths produced
/// when the body was split for embedding.
public struct DocumentRecord: Sendable, Hashable {
    public var uuid: String
    public var date: Date
    public var metadata: Data
    public var hash: String
    public var body: String
    public var lens: [Int]

    public init(
        uuid: String,
        date: Date,
        metadata: Data = Data(),
        hash: String,
        body: String,
        lens: [Int] = []
    ) {
        self.uuid = uuid
        self.date = date
        self.metadata = metadata
        self.hash = hash
        self.body = body
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

/// A scored full-text search hit returned by a storage backend.
public struct FullTextHit: Sendable, Hashable {
    public var uuid: String
    public var score: Float

    public init(uuid: String, score: Float) {
        self.uuid = uuid
        self.score = score
    }
}
