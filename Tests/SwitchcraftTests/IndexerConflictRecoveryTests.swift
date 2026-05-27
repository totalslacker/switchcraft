// SPDX-License-Identifier: Apache-2.0
import Foundation
import Testing
@testable import SwitchcraftCore

/// Tests for `Indexer.init` auto-recovery of `rehydrationConflict` storage
/// corruption (issue #101). Covers requirements 11–15 from the spec.
@Suite("Indexer Conflict Recovery")
struct IndexerConflictRecoveryTests {

    // MARK: - Dimensions

    /// Embedding dimensionality for all synthetic data in this suite.
    /// Must be even (Q4Codec requirement) and small for speed.
    static let dims = 4

    // MARK: - Helpers

    /// Build a synthetic 2-generation conflict in `InMemoryStorage`.
    ///
    /// Both generations claim `sharedChunkID` (tokenOffset=0).
    /// Gen1 uses center `[1, 0, 0, 0]` and gen2 uses center `[0, 1, 0, 0]`,
    /// both with zero residuals — so the reconstructed embeddings equal the
    /// centers exactly (within Q4 precision), making winner verification easy.
    ///
    /// Returns the storage plus the gen1/gen2 IDs so tests can assert on
    /// post-recovery state.
    @discardableResult
    static func makeConflictingStorage(
        gen1Level: Int,
        gen1Created: Date,
        gen2Level: Int,
        gen2Created: Date,
        sharedChunkID: Int64
    ) async throws -> (storage: InMemoryStorage, gen1ID: Int64, gen2ID: Int64) {
        let storage = InMemoryStorage()
        try await storage.open()

        let zeroResiduals = [Float](repeating: 0.0, count: dims)

        // Gen1: center=[1,0,0,0], tokenOffset=0 for sharedChunkID.
        let center1: [Float] = [1.0, 0.0, 0.0, 0.0]
        let gen1 = try await storage.insertGeneration(GenerationRecord(
            level: gen1Level,
            numEmbeddings: 1,
            minChunkID: sharedChunkID,
            maxChunkID: sharedChunkID,
            created: gen1Created
        ))
        let pairs1 = [IndexPair(chunkID: UInt32(sharedChunkID), tokenOffset: 0)]
        _ = try await storage.insertBucket(BucketRecord(
            generationID: gen1.id,
            center: Indexer.encodeFloat32LE(center1),
            indices: IndicesCodec.encode(pairs1),
            residuals: Q4Codec.encodeResiduals(zeroResiduals)
        ))

        // Gen2: center=[0,1,0,0], tokenOffset=0 for sharedChunkID (conflict).
        let center2: [Float] = [0.0, 1.0, 0.0, 0.0]
        let gen2 = try await storage.insertGeneration(GenerationRecord(
            level: gen2Level,
            numEmbeddings: 1,
            minChunkID: sharedChunkID,
            maxChunkID: sharedChunkID,
            created: gen2Created
        ))
        let pairs2 = [IndexPair(chunkID: UInt32(sharedChunkID), tokenOffset: 0)]
        _ = try await storage.insertBucket(BucketRecord(
            generationID: gen2.id,
            center: Indexer.encodeFloat32LE(center2),
            indices: IndicesCodec.encode(pairs2),
            residuals: Q4Codec.encodeResiduals(zeroResiduals)
        ))

        return (storage, gen1.id, gen2.id)
    }

    // MARK: - Test 1 (req 11): autoRecover with 2-gen conflict

    @Test("autoRecover: 2-gen conflict rehydrates successfully, correct winner selected")
    func autoRecover_2genConflict_rehydratesSuccessfully() async throws {
        let sharedChunkID: Int64 = 10
        let now = Date()
        let (storage, gen1ID, _) = try await Self.makeConflictingStorage(
            gen1Level: 2,
            gen1Created: now,
            gen2Level: 0,
            gen2Created: now,
            sharedChunkID: sharedChunkID
        )

        // Gen1 (level=2) wins over gen2 (level=0).
        // Init must complete without throwing.
        let indexer = try await Indexer(
            storage: storage,
            config: .testing(rehydrationConflictBehavior: .autoRecover)
        )

        #expect(await indexer.recoveredConflictCount == 1)

        // The ledger must have exactly 1 entry for sharedChunkID (from the winner).
        let ledger = await indexer.ledgerSnapshot
        let entries = ledger[sharedChunkID]
        #expect(entries != nil)
        #expect(entries?.count == 1)

        // Winner (gen1) has center=[1,0,0,0] with zero residuals.
        // Embedding ≈ [1.0, 0.0, 0.0, 0.0].
        // Loser (gen2) would produce ≈ [0.0, 1.0, 0.0, 0.0].
        if let e = entries?.first {
            // First element should be close to 1.0 (gen1), not 0.0 (gen2).
            #expect(e[0] > 0.9, "expected winner gen1 (level=2) embedding[0]≈1.0, got \(e[0])")
            #expect(e[1] < 0.1, "expected winner gen1 (level=2) embedding[1]≈0.0, got \(e[1])")
        }

        // The winner generation must still be present in storage.
        let gens = try await storage.generations()
        #expect(gens.map(\.id).contains(gen1ID))
    }

    // MARK: - Test 2 (req 12): loser pruned from storage

    @Test("autoRecover: loser generation and its buckets are absent from storage after init")
    func autoRecover_prunedLoserAbsentFromStorage() async throws {
        let sharedChunkID: Int64 = 20
        let now = Date()
        let (storage, gen1ID, gen2ID) = try await Self.makeConflictingStorage(
            gen1Level: 2,
            gen1Created: now,
            gen2Level: 0,
            gen2Created: now,
            sharedChunkID: sharedChunkID
        )

        _ = try await Indexer(
            storage: storage,
            config: .testing(rehydrationConflictBehavior: .autoRecover)
        )

        let gens = try await storage.generations()

        // The loser (gen2, level=0) must be gone entirely — it had only the
        // conflicted chunkID, so survivingBuckets was empty → fully pruned.
        #expect(!gens.map(\.id).contains(gen2ID), "loser gen2 should be removed from storage")

        // The winner (gen1) must still be present.
        #expect(gens.map(\.id).contains(gen1ID), "winner gen1 should remain in storage")

        // No buckets should remain for the loser generation.
        let loserBuckets = try await storage.buckets(forGeneration: gen2ID)
        #expect(loserBuckets.isEmpty, "loser gen2 should have no buckets")

        // Winner's buckets should still be present.
        let winnerBuckets = try await storage.buckets(forGeneration: gen1ID)
        #expect(!winnerBuckets.isEmpty, "winner gen1 should still have buckets")
    }

    // MARK: - Test 3 (req 13): storage error rolls back and rethrows

    @Test("autoRecover: storage error during prune rolls back and rethrows")
    func autoRecover_storageError_rollsBackAndRethrows() async throws {
        let sharedChunkID: Int64 = 30
        let now = Date()
        let (inner, gen1ID, gen2ID) = try await Self.makeConflictingStorage(
            gen1Level: 2,
            gen1Created: now,
            gen2Level: 0,
            gen2Created: now,
            sharedChunkID: sharedChunkID
        )

        let stub = FailingReplaceStorage(inner: inner)
        await stub.set(shouldFailOnReplace: true)

        // Init must throw the injected error.
        await #expect(throws: FailingReplaceStorage.ReplaceError.injected) {
            _ = try await Indexer(
                storage: stub,
                config: .testing(rehydrationConflictBehavior: .autoRecover)
            )
        }

        // Storage must be unchanged: both conflicting generations still present.
        let underlying = await stub.underlyingStorage
        let gens = try await underlying.generations()
        let ids = gens.map(\.id)
        #expect(ids.contains(gen1ID), "gen1 must still be present after failed prune")
        #expect(ids.contains(gen2ID), "gen2 must still be present after failed prune")
    }

    // MARK: - Test 4 (req 14): throwError mode still throws

    @Test("throwError: still throws rehydrationConflict for the same synthetic conflict")
    func throwError_stillThrows_forSameConflict() async throws {
        let sharedChunkID: Int64 = 40
        let now = Date()
        let (storage, _, _) = try await Self.makeConflictingStorage(
            gen1Level: 2,
            gen1Created: now,
            gen2Level: 0,
            gen2Created: now,
            sharedChunkID: sharedChunkID
        )

        await #expect(throws: Indexer.Error.rehydrationConflict(chunkID: sharedChunkID)) {
            _ = try await Indexer(
                storage: storage,
                config: .testing(rehydrationConflictBehavior: .throwError)
            )
        }
    }

    // MARK: - Test 5 (req 15): N-way conflict (3 generations)

    @Test("autoRecover: 3-gen conflict picks single correct winner (highest level wins)")
    func autoRecover_3genConflict_picksCorrectWinner() async throws {
        let sharedChunkID: Int64 = 42
        let storage = InMemoryStorage()
        try await storage.open()
        let now = Date()
        let zeroResiduals = [Float](repeating: 0.0, count: Self.dims)

        // Gen1: level=0, created=T-100. Center=[1,0,0,0]. Loser.
        let t100ago = now.addingTimeInterval(-100)
        let center1: [Float] = [1.0, 0.0, 0.0, 0.0]
        let gen1 = try await storage.insertGeneration(GenerationRecord(
            level: 0, numEmbeddings: 1,
            minChunkID: sharedChunkID, maxChunkID: sharedChunkID,
            created: t100ago
        ))
        _ = try await storage.insertBucket(BucketRecord(
            generationID: gen1.id,
            center: Indexer.encodeFloat32LE(center1),
            indices: IndicesCodec.encode([IndexPair(chunkID: UInt32(sharedChunkID), tokenOffset: 0)]),
            residuals: Q4Codec.encodeResiduals(zeroResiduals)
        ))

        // Gen2: level=0, created=T-50 (more recent same-level). Center=[0,1,0,0]. Loser.
        let t50ago = now.addingTimeInterval(-50)
        let center2: [Float] = [0.0, 1.0, 0.0, 0.0]
        let gen2 = try await storage.insertGeneration(GenerationRecord(
            level: 0, numEmbeddings: 1,
            minChunkID: sharedChunkID, maxChunkID: sharedChunkID,
            created: t50ago
        ))
        _ = try await storage.insertBucket(BucketRecord(
            generationID: gen2.id,
            center: Indexer.encodeFloat32LE(center2),
            indices: IndicesCodec.encode([IndexPair(chunkID: UInt32(sharedChunkID), tokenOffset: 0)]),
            residuals: Q4Codec.encodeResiduals(zeroResiduals)
        ))

        // Gen3: level=1 (highest level, wins regardless of created). Center=[0,0,1,0]. Winner.
        let center3: [Float] = [0.0, 0.0, 1.0, 0.0]
        let gen3 = try await storage.insertGeneration(GenerationRecord(
            level: 1, numEmbeddings: 1,
            minChunkID: sharedChunkID, maxChunkID: sharedChunkID,
            created: now
        ))
        _ = try await storage.insertBucket(BucketRecord(
            generationID: gen3.id,
            center: Indexer.encodeFloat32LE(center3),
            indices: IndicesCodec.encode([IndexPair(chunkID: UInt32(sharedChunkID), tokenOffset: 0)]),
            residuals: Q4Codec.encodeResiduals(zeroResiduals)
        ))

        let indexer = try await Indexer(
            storage: storage,
            config: .testing(rehydrationConflictBehavior: .autoRecover)
        )

        // One distinct chunkID was conflicted.
        #expect(await indexer.recoveredConflictCount == 1)

        // Verify the winner's embedding is in the ledger.
        // Gen3 center=[0,0,1,0] → embedding[2] ≈ 1.0.
        let ledger = await indexer.ledgerSnapshot
        let entries = ledger[sharedChunkID]
        #expect(entries != nil)
        #expect(entries?.count == 1)
        if let e = entries?.first {
            #expect(e[2] > 0.9, "expected winner gen3 (level=1) embedding[2]≈1.0, got \(e[2])")
            #expect(e[0] < 0.1, "not gen1: embedding[0] should be ≈0.0, got \(e[0])")
            #expect(e[1] < 0.1, "not gen2: embedding[1] should be ≈0.0, got \(e[1])")
        }

        // Both gen1 and gen2 must be pruned from storage.
        let gens = try await storage.generations()
        let ids = gens.map(\.id)
        #expect(!ids.contains(gen1.id), "gen1 (loser, level=0) should be pruned")
        #expect(!ids.contains(gen2.id), "gen2 (loser, level=0) should be pruned")
        #expect(ids.contains(gen3.id), "gen3 (winner, level=1) should remain")
    }
}

// MARK: - FailingReplaceStorage actor mutation helper

extension FailingReplaceStorage {
    /// Set `shouldFailOnReplace` — helper for test clarity.
    func set(shouldFailOnReplace value: Bool) {
        shouldFailOnReplace = value
    }
}
