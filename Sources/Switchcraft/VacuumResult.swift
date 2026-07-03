// SPDX-License-Identifier: Apache-2.0
import SwitchcraftCore

/// The outcome of one `SwitchcraftStore.vacuum()` call.
public struct VacuumResult: Sendable, Equatable {
    /// Number of abandoned chunk rows actually deleted by this call. May be
    /// less than the number of candidates in this call's batch if a
    /// concurrent re-add raced the vacuum's guarded chunk delete (see
    /// `vacuum()`'s doc comment).
    public let chunksRemoved: Int

    /// Number of `(chunkID, tokenOffset)` index pairs removed from bucket
    /// `indices`/`residuals` blobs by this call.
    public let bucketPairsRemoved: Int

    /// Approximate bytes reclaimed to the backend's internal free-list,
    /// measured as the delta in `PRAGMA freelist_count × PRAGMA page_size`
    /// immediately before and after this call's write phase.
    ///
    /// This reflects space returned to SQLite's internal free-list for
    /// reuse by future inserts — it does **not** reflect a reduction in
    /// on-disk file size. `DELETE` plus `wal_checkpoint(TRUNCATE)` does not
    /// return pages to the OS; the file only shrinks after a full `PRAGMA
    /// vacuum`, which is out of scope for this operation (it can require
    /// up to 2× the database size in temporary disk and is incompatible
    /// with `maxBatch` batching).
    ///
    /// A consumer that wants an actual file-size reduction should run
    /// `PRAGMA vacuum` on the underlying connection themselves, after
    /// looping `vacuum()` calls until `remainingCandidates == 0`.
    ///
    /// Always `0` for non-SQLite backends (the default
    /// `SwitchcraftStorage.freeListByteCount()` implementation returns `0`).
    public let approximateDiskReclaimed: Int64

    /// Ids of every generation that was either partially rewritten (at
    /// least one bucket updated or deleted, but not the whole generation)
    /// or fully deleted by this call.
    public let generationsAffected: [Int64]

    /// Count of abandoned chunks not yet processed by this call. Always
    /// `0` when `maxBatch` is `nil`. When greater than `0`, call
    /// `vacuum()` again to continue processing the backlog.
    public let remainingCandidates: Int

    /// Outcome of the `wal_checkpoint(TRUNCATE)` run at the end of this
    /// call's write phase. `.complete` for non-SQLite backends and for
    /// calls that found no abandoned chunks to process (no checkpoint is
    /// run in that case).
    public let checkpoint: CheckpointResult

    public init(
        chunksRemoved: Int,
        bucketPairsRemoved: Int,
        approximateDiskReclaimed: Int64,
        generationsAffected: [Int64],
        remainingCandidates: Int,
        checkpoint: CheckpointResult
    ) {
        self.chunksRemoved = chunksRemoved
        self.bucketPairsRemoved = bucketPairsRemoved
        self.approximateDiskReclaimed = approximateDiskReclaimed
        self.generationsAffected = generationsAffected
        self.remainingCandidates = remainingCandidates
        self.checkpoint = checkpoint
    }
}
