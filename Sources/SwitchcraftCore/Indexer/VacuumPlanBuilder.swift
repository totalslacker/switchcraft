// SPDX-License-Identifier: Apache-2.0
import Foundation

/// Pure algorithm that turns a set of abandoned chunk ids into a
/// `VacuumPlan`: which bucket blobs to re-encode, which buckets/generations
/// to delete outright, and which surviving generations need a corrected
/// `numEmbeddings`.
///
/// Has no storage or actor dependency beyond the injected `fetchBuckets`
/// closure, so it is directly unit-testable against synthetic generations
/// and buckets.
public enum VacuumPlanBuilder {

    /// One live chunk's `.bucketRef` ledger token that must be repointed at
    /// its surviving pair's new offset after a bucket is compacted.
    ///
    /// Computed in the same per-bucket loop that builds the compacted
    /// `indices`/`residuals` blobs (see `buildPlan`), from the same
    /// `survivingPairs` accumulation — never re-derived from a separate
    /// decode — so the remap can never desync from the bytes actually
    /// written to storage (issue #142).
    public struct BucketRefRemap: Sendable, Equatable {
        public let chunkID: Int64
        public let tokenOffset: UInt32
        public let genID: Int64
        public let bucketID: Int64
        public let newPairOffset: Int

        public init(chunkID: Int64, tokenOffset: UInt32, genID: Int64, bucketID: Int64, newPairOffset: Int) {
            self.chunkID = chunkID
            self.tokenOffset = tokenOffset
            self.genID = genID
            self.bucketID = bucketID
            self.newPairOffset = newPairOffset
        }
    }

    /// Result of building a vacuum plan: the plan itself plus the metrics
    /// `SwitchcraftStore.vacuum()` folds into `VacuumResult`.
    public struct Result: Sendable {
        public let plan: VacuumPlan
        public let bucketPairsRemoved: Int
        public let generationsAffected: Set<Int64>
        /// Every surviving pair's new `.bucketRef` offset, for every bucket
        /// this plan touches (`bucketUpdates`). A bucket queued for deletion
        /// (`bucketIDsToDelete`/`generationIDsToDelete`) never contributes a
        /// remap, since by construction it has zero surviving pairs — no
        /// live ledger ref can point there. See issue #142.
        public let bucketRefRemaps: [BucketRefRemap]

        public init(
            plan: VacuumPlan, bucketPairsRemoved: Int, generationsAffected: Set<Int64>,
            bucketRefRemaps: [BucketRefRemap] = []
        ) {
            self.plan = plan
            self.bucketPairsRemoved = bucketPairsRemoved
            self.generationsAffected = generationsAffected
            self.bucketRefRemaps = bucketRefRemaps
        }
    }

    /// Build a `VacuumPlan` that removes every `(chunkID, tokenOffset)`
    /// pair whose `chunkID` is in `abandonedChunkIDs` from every bucket in
    /// `generations`.
    ///
    /// Generations whose `[minChunkID, maxChunkID]` range does not overlap
    /// `[abandonedChunkIDs.min, abandonedChunkIDs.max]` are skipped
    /// entirely — their buckets are never fetched or decoded — turning the
    /// common case (abandoned ids clustered together) into O(chunks
    /// actually affected) rather than O(total indexed tokens).
    ///
    /// - Parameters:
    ///   - abandonedChunkIDs: chunk ids to remove from bucket blobs and to
    ///     include in `VacuumPlan.chunkIDsToDelete`. Every id is added to
    ///     `chunkIDsToDelete` regardless of whether it turns up in any
    ///     bucket — a chunk that was never (or only partially) indexed
    ///     must still be deleted, since the removal criterion is "no
    ///     owning document", independent of indexing completeness.
    ///   - generations: every generation currently in storage.
    ///   - fetchBuckets: fetches the buckets for one generation. Injected
    ///     so this algorithm has no storage dependency.
    /// - Throws: `IndicesCodec.Error` if a bucket's `indices` blob is
    ///   corrupt; whatever `fetchBuckets` throws.
    public static func buildPlan(
        abandonedChunkIDs: Set<Int64>,
        generations: [GenerationRecord],
        fetchBuckets: (Int64) async throws -> [BucketRecord]
    ) async throws -> Result {
        guard !abandonedChunkIDs.isEmpty else {
            return Result(plan: VacuumPlan(), bucketPairsRemoved: 0, generationsAffected: [])
        }
        let minAbandoned = abandonedChunkIDs.min()!
        let maxAbandoned = abandonedChunkIDs.max()!

        var bucketUpdates: [BucketRecord] = []
        var bucketIDsToDelete: Set<Int64> = []
        var generationIDsToDelete: Set<Int64> = []
        var generationEmbeddingCountUpdates: [Int64: Int] = [:]
        var generationsAffected: Set<Int64> = []
        var totalPairsRemoved = 0
        var bucketRefRemaps: [BucketRefRemap] = []

        for gen in generations {
            // Range-prune: this generation cannot contain any abandoned
            // chunkID, so skip fetching/decoding its buckets entirely.
            guard gen.minChunkID <= maxAbandoned, gen.maxChunkID >= minAbandoned else { continue }

            let buckets = try await fetchBuckets(gen.id)
            var survivingBucketCount = 0
            var pairsRemovedInGen = 0
            var genChanged = false

            for bucket in buckets {
                let pairs = try IndicesCodec.decode(bucket.indices)
                let hasAbandonedPair = pairs.contains { abandonedChunkIDs.contains(Int64($0.chunkID)) }
                guard hasAbandonedPair else {
                    survivingBucketCount += 1
                    continue
                }

                genChanged = true
                let residuals = Q4Codec.decodeResiduals(bucket.residuals)
                let dims = residuals.count / pairs.count

                var survivingPairs: [IndexPair] = []
                var survivingResiduals: [Float] = []
                survivingPairs.reserveCapacity(pairs.count)
                survivingResiduals.reserveCapacity(residuals.count)

                for (i, pair) in pairs.enumerated() {
                    if abandonedChunkIDs.contains(Int64(pair.chunkID)) {
                        pairsRemovedInGen += 1
                        continue
                    }
                    // Every surviving pair gets a remap entry, whether or
                    // not its offset actually shifted — cheaper to always
                    // emit than to track "did this shift" (issue #142).
                    bucketRefRemaps.append(BucketRefRemap(
                        chunkID: Int64(pair.chunkID), tokenOffset: pair.tokenOffset,
                        genID: gen.id, bucketID: bucket.id, newPairOffset: survivingPairs.count
                    ))
                    survivingPairs.append(pair)
                    let base = i * dims
                    survivingResiduals.append(contentsOf: residuals[base..<(base + dims)])
                }

                if survivingPairs.isEmpty {
                    bucketIDsToDelete.insert(bucket.id)
                } else {
                    survivingBucketCount += 1
                    var updated = bucket
                    updated.indices = IndicesCodec.encode(survivingPairs)
                    updated.residuals = Q4Codec.encodeResiduals(survivingResiduals)
                    bucketUpdates.append(updated)
                }
            }

            guard genChanged else { continue }
            generationsAffected.insert(gen.id)
            totalPairsRemoved += pairsRemovedInGen

            if survivingBucketCount == 0 {
                generationIDsToDelete.insert(gen.id)
                // Any bucket-level delete queued for this generation's
                // buckets is superseded by the wholesale generation delete.
                for bucket in buckets {
                    bucketIDsToDelete.remove(bucket.id)
                }
            } else {
                generationEmbeddingCountUpdates[gen.id] = gen.numEmbeddings - pairsRemovedInGen
            }
        }

        let plan = VacuumPlan(
            chunkIDsToDelete: abandonedChunkIDs,
            bucketUpdates: bucketUpdates,
            bucketIDsToDelete: bucketIDsToDelete,
            generationIDsToDelete: generationIDsToDelete,
            generationEmbeddingCountUpdates: generationEmbeddingCountUpdates
        )
        return Result(
            plan: plan, bucketPairsRemoved: totalPairsRemoved, generationsAffected: generationsAffected,
            bucketRefRemaps: bucketRefRemaps
        )
    }
}
