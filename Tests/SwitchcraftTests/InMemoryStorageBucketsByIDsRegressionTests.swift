// SPDX-License-Identifier: Apache-2.0
import Foundation
import Testing
import SwitchcraftCore

/// `InMemoryStorage.buckets(ids:)` (issue #151 / ADR 041) is a linear scan
/// over `bucketsByGeneration.values`, not a separately-maintained index —
/// see Plan stage decision. These tests prove it stays correct across every
/// mutation path that touches `bucketsByGeneration`: `deleteGeneration`,
/// `replaceGeneration`, and `applyVacuumPlan`.
@Suite("InMemoryStorage buckets(ids:) regression")
struct InMemoryStorageBucketsByIDsRegressionTests {

    @Test("buckets(ids:) omits buckets removed by deleteGeneration")
    func bucketsByIDsAfterDeleteGeneration() async throws {
        let storage = InMemoryStorage()
        try await storage.open()

        let gen = try await storage.insertGeneration(
            GenerationRecord(level: 0, numEmbeddings: 0, minChunkID: 0, maxChunkID: 0, created: Date())
        )
        let bucket = try await storage.insertBucket(
            BucketRecord(generationID: gen.id, center: Data([0x01]), indices: Data(), residuals: Data())
        )

        #expect(try await storage.buckets(ids: [bucket.id]).map(\.id) == [bucket.id])

        try await storage.deleteGeneration(id: gen.id)

        #expect(try await storage.buckets(ids: [bucket.id]).isEmpty)
    }

    @Test("buckets(ids:) reflects replaceGeneration's old-id removal and new-id insertion")
    func bucketsByIDsAfterReplaceGeneration() async throws {
        let storage = InMemoryStorage()
        try await storage.open()

        let loser = try await storage.insertGeneration(
            GenerationRecord(level: 0, numEmbeddings: 1, minChunkID: 1, maxChunkID: 1, created: Date())
        )
        let losingBucket = try await storage.insertBucket(
            BucketRecord(generationID: loser.id, center: Data([0xAA]), indices: Data([0x01]), residuals: Data([0x02]))
        )

        let survivor = BucketRecord(
            generationID: BucketRecord.unassigned,
            center: losingBucket.center,
            indices: losingBucket.indices,
            residuals: losingBucket.residuals
        )
        let newGen = try await storage.replaceGeneration(
            losingGenerationID: loser.id,
            survivingRecord: GenerationRecord(level: 0, numEmbeddings: 1, minChunkID: 1, maxChunkID: 1, created: Date()),
            survivingBuckets: [survivor]
        )
        #expect(newGen != nil)

        // The old bucket id is gone — replaceGeneration assigns fresh ids.
        #expect(try await storage.buckets(ids: [losingBucket.id]).isEmpty)

        // The surviving bucket, under its fresh id, carries the same payload.
        let newBuckets = try await storage.buckets(forGeneration: newGen!.id)
        #expect(newBuckets.count == 1)
        let fetched = try await storage.buckets(ids: [newBuckets[0].id])
        #expect(fetched.count == 1)
        #expect(fetched[0].center == losingBucket.center)
        #expect(fetched[0].indices == losingBucket.indices)
        #expect(fetched[0].residuals == losingBucket.residuals)
    }

    @Test("buckets(ids:) reflects applyVacuumPlan's bucket update and delete")
    func bucketsByIDsAfterApplyVacuumPlan() async throws {
        let storage = InMemoryStorage()
        try await storage.open()

        let gen = try await storage.insertGeneration(
            GenerationRecord(level: 0, numEmbeddings: 2, minChunkID: 1, maxChunkID: 2, created: Date())
        )
        let bucketToUpdate = try await storage.insertBucket(
            BucketRecord(generationID: gen.id, center: Data([0x01]), indices: Data([0xAA]), residuals: Data([0xBB]))
        )
        let bucketToDelete = try await storage.insertBucket(
            BucketRecord(generationID: gen.id, center: Data([0x02]), indices: Data([0xCC]), residuals: Data([0xDD]))
        )

        var updated = bucketToUpdate
        updated.indices = Data([0xEE])
        updated.residuals = Data([0xFF])

        let plan = VacuumPlan(
            bucketUpdates: [updated],
            bucketIDsToDelete: [bucketToDelete.id],
            generationEmbeddingCountUpdates: [gen.id: 1]
        )
        _ = try await storage.applyVacuumPlan(plan)

        // Deleted bucket no longer fetchable by id.
        #expect(try await storage.buckets(ids: [bucketToDelete.id]).isEmpty)

        // Updated bucket keeps its id; buckets(ids:) returns the new payload.
        let fetched = try await storage.buckets(ids: [bucketToUpdate.id])
        #expect(fetched.count == 1)
        #expect(fetched[0].indices == Data([0xEE]))
        #expect(fetched[0].residuals == Data([0xFF]))
    }
}
