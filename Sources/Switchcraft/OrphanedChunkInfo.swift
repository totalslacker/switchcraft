// SPDX-License-Identifier: Apache-2.0

/// Describes a chunk whose tokens are absent from, or only partially
/// present in, committed bucket assignments.
///
/// A chunk becomes orphaned when `storage.upsertChunk()` succeeds but
/// `indexer.add()` is interrupted before compaction writes the
/// corresponding bucket entries — for example, by a process kill, OOM,
/// or mid-write disk error.  An orphaned chunk is visible to FTS but
/// invisible to vector search; re-adding any of its owning documents
/// through `SwitchcraftStore.add()` triggers automatic recovery.
///
/// ## Recovery contract
///
/// For each entry returned by `SwitchcraftStore.findOrphanedChunks()`,
/// re-add any document listed in `owningDocuments` through
/// `store.add(id:body:)`.  `add()` detects the missing bucket
/// assignments and calls `indexer.add()` to re-buffer the embeddings;
/// the next flush (or the next `store.search()` call) compacts them
/// into the bucket index, making the document retrievable through
/// vector search.
///
/// A future `reindex(id:)` API will provide a more targeted recovery
/// path; for now, re-adding the full document body is the supported
/// mechanism.
public struct OrphanedChunkInfo: Sendable, Equatable {
    /// Backend-assigned chunk identifier.
    public let chunkID: Int64
    /// SHA-256 content hash used as the dedup key.
    public let hash: String
    /// Embedder model identifier recorded at ingest time.
    public let model: String
    /// Total token count recorded in the chunk row at ingest time.
    public let expectedTokenCount: Int
    /// Number of `(chunkID, tokenOffset)` pairs found across all bucket
    /// blobs.  Zero for a true orphan; greater than zero but less than
    /// `expectedTokenCount` for a partially-indexed anomaly.
    public let bucketReferenceCount: Int
    /// UUIDs of every document row whose `hash` maps to this chunk.
    /// Re-adding any of these documents triggers automatic recovery.
    public let owningDocuments: [String]

    public init(
        chunkID: Int64,
        hash: String,
        model: String,
        expectedTokenCount: Int,
        bucketReferenceCount: Int,
        owningDocuments: [String]
    ) {
        self.chunkID = chunkID
        self.hash = hash
        self.model = model
        self.expectedTokenCount = expectedTokenCount
        self.bucketReferenceCount = bucketReferenceCount
        self.owningDocuments = owningDocuments
    }
}
