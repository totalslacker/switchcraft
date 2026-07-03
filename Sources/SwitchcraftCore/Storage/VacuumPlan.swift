// SPDX-License-Identifier: Apache-2.0
import Foundation

/// A pre-computed, already-encoded set of storage mutations for one
/// `SwitchcraftStore.vacuum()` batch.
///
/// Built by `VacuumPlanBuilder` in the engine layer (where the
/// `IndicesCodec`/`Q4Codec` blob format is understood) and executed
/// atomically by a backend's `SwitchcraftStorage.applyVacuumPlan(_:)`,
/// which knows nothing about bucket-blob internals — it only updates or
/// deletes rows by id.
public struct VacuumPlan: Sendable, Equatable {
    /// Chunk ids to delete. A backend must guard each delete against a
    /// concurrent re-add: only delete a chunk row if no document row
    /// currently references its hash.
    public var chunkIDsToDelete: Set<Int64>

    /// Buckets whose `indices`/`residuals` were re-encoded after filtering
    /// out abandoned pairs. `id` and `generationID` identify the existing
    /// row to update; the row is never re-inserted.
    public var bucketUpdates: [BucketRecord]

    /// Buckets left with zero surviving pairs, to be deleted outright.
    /// Only used for buckets whose generation survives (has at least one
    /// other surviving bucket) — a generation with no surviving buckets is
    /// deleted wholesale via `generationIDsToDelete` instead.
    public var bucketIDsToDelete: Set<Int64>

    /// Generations left with zero surviving buckets, to be deleted
    /// wholesale (cascading to any of their bucket rows).
    public var generationIDsToDelete: Set<Int64>

    /// New `numEmbeddings` values for generations that survive (are not
    /// in `generationIDsToDelete`) but had at least one pair removed.
    public var generationEmbeddingCountUpdates: [Int64: Int]

    public init(
        chunkIDsToDelete: Set<Int64> = [],
        bucketUpdates: [BucketRecord] = [],
        bucketIDsToDelete: Set<Int64> = [],
        generationIDsToDelete: Set<Int64> = [],
        generationEmbeddingCountUpdates: [Int64: Int] = [:]
    ) {
        self.chunkIDsToDelete = chunkIDsToDelete
        self.bucketUpdates = bucketUpdates
        self.bucketIDsToDelete = bucketIDsToDelete
        self.generationIDsToDelete = generationIDsToDelete
        self.generationEmbeddingCountUpdates = generationEmbeddingCountUpdates
    }

    public var isEmpty: Bool {
        chunkIDsToDelete.isEmpty
            && bucketUpdates.isEmpty
            && bucketIDsToDelete.isEmpty
            && generationIDsToDelete.isEmpty
            && generationEmbeddingCountUpdates.isEmpty
    }
}
