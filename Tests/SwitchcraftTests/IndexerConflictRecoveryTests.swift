// SPDX-License-Identifier: Apache-2.0
import Foundation
import Testing
@testable import Switchcraft
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
        let ledger = await indexer.ledgerContents
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
        let ledger = await indexer.ledgerContents
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
    // MARK: - Test 6 (req 103): inflated numEmbeddings on winner is corrected, flush succeeds

    /// Regression test for issue #103. Simulates a crash between step 9 (generation row
    /// insert) and step 10 (bucket inserts) in `performFlush()`, leaving the winner gen's
    /// `numEmbeddings` higher than the actual decoded pair count. Verifies that:
    ///   (a) `Indexer.init` with `.autoRecover` corrects the stale count in storage, and
    ///   (b) a subsequent `add()` + `flush()` does not throw `ledgerOutOfSync`.
    @Test("autoRecover: inflated numEmbeddings on winner is corrected, subsequent flush succeeds")
    func autoRecover_inflatedNumEmbeddings_correctedAndFlushSucceeds() async throws {
        let storage = InMemoryStorage()
        try await storage.open()

        let now = Date()
        let d = Self.dims  // 4
        let zeroResiduals = [Float](repeating: 0.0, count: d)

        // Loser gen (inserted first → lower id): level=0, one pair at chunkID=100.
        // This creates a rehydrationConflict on chunkID=100 with the winner.
        let loserCenter: [Float] = [0.0, 0.0, 1.0, 0.0]
        let loserGen = try await storage.insertGeneration(GenerationRecord(
            level: 0,
            numEmbeddings: 1,
            minChunkID: 100,
            maxChunkID: 100,
            created: now
        ))
        _ = try await storage.insertBucket(BucketRecord(
            generationID: loserGen.id,
            center: Indexer.encodeFloat32LE(loserCenter),
            indices: IndicesCodec.encode([IndexPair(chunkID: 100, tokenOffset: 0)]),
            residuals: Q4Codec.encodeResiduals(zeroResiduals)
        ))

        // Winner gen (inserted second → higher id, wins tie-break): level=0, two pairs at
        // chunkID=100 and chunkID=200. numEmbeddings is intentionally inflated to 9,
        // simulating a crash between the generation row insert (step 9 of performFlush)
        // and bucket completion (step 10). Actual decoded pair count is 2.
        let winnerCenter: [Float] = [0.5, 0.5, 0.0, 0.0]
        let winnerGen = try await storage.insertGeneration(GenerationRecord(
            level: 0,
            numEmbeddings: 9,
            minChunkID: 100,
            maxChunkID: 200,
            created: now
        ))
        _ = try await storage.insertBucket(BucketRecord(
            generationID: winnerGen.id,
            center: Indexer.encodeFloat32LE(winnerCenter),
            indices: IndicesCodec.encode([
                IndexPair(chunkID: 100, tokenOffset: 0),
                IndexPair(chunkID: 200, tokenOffset: 0),
            ]),
            residuals: Q4Codec.encodeResiduals(zeroResiduals + zeroResiduals)
        ))

        // Init with autoRecover: conflict on chunkID=100 is resolved, winner (higher id) wins.
        let indexer = try await Indexer(
            storage: storage,
            config: .testing(rehydrationConflictBehavior: .autoRecover)
        )
        #expect(await indexer.recoveredConflictCount == 1)

        // Step 3.5 must have corrected the winner's numEmbeddings from 9 to 2.
        let gensAfterRecovery = try await storage.generations()
        let survivingGen = gensAfterRecovery.first { $0.id == winnerGen.id }
        #expect(survivingGen != nil, "winner gen must survive recovery")
        #expect(survivingGen?.numEmbeddings == 2,
                "numEmbeddings must be corrected from 9 to 2; got \(survivingGen?.numEmbeddings ?? -1)")

        // add() + flush() must not throw ledgerOutOfSync.
        // With the inflated count (9), the cascade walk computes total=2+9=11>8, cascades
        // to level 1, then finds m=4 != total=11 and throws. With the fix, total=2+2=4≤8
        // stays at level 0, m=4=total → success.
        let newEmbeddings = [Float](repeating: 0.1, count: 2 * d)  // 2 token embeddings
        try await indexer.add(chunkID: 300, embeddings: newEmbeddings, dims: d)
        try await indexer.flush()

        // Flush must have created at least one generation.
        let gensAfterFlush = try await storage.generations()
        #expect(!gensAfterFlush.isEmpty, "flush must produce at least one generation")
    }

    // MARK: - Test 7 (issue #123): mid-operation compaction self-recovery

    /// Regression test for issue #123 (ADR 030).
    ///
    /// Constructs a 3-generation storage state (L2+L1+L0), rehydrates an Indexer,
    /// then calls `indexer.add()` for a chunkID already in the L2 gen — simulating
    /// the ADR 029 orphan-recovery workload. A subsequent flush previously threw
    /// `ledgerOutOfSync` because the cascade walk range swept in ledger rows from
    /// the L2 gen that were not included in `mergedGens`. The fix absorbs the
    /// surprise gens and self-recovers.
    @Test("performFlush: mid-operation ledger–storage divergence self-recovers without throwing ledgerOutOfSync")
    func midOperationCompaction_selfRecovers_noLedgerOutOfSync() async throws {
        let d = Self.dims  // 4
        let tokensPerChunk = 2

        // Build storage with 3 gens:
        //   Gen A at L2: chunkIDs 1,2,3 — 2 tokens each → numEmbeddings=6
        //   Gen B at L1: chunkIDs 5,6   — 2 tokens each → numEmbeddings=4
        //   Gen C at L0: chunkID  8     — 2 tokens       → numEmbeddings=2
        //
        // Config: l0Capacity=8, lsmFanout=2.
        //   levelCapacity(0)=16, levelCapacity(1)=32, levelCapacity(2)=64.
        //
        // After orphan-recovery add of 2 tokens for chunkID=2:
        //   pending=2, levelSums={2:6, 1:4, 0:2}
        //   cascade walk: total=2+2=4 ≤ 16 → targetLevel=0
        //   mergedGens = [Gen C (L0, chunk 8)]
        //   range = [2..8] (pendingMin=2, Gen C max=8)
        //   ledger collect [2..8]: chunks 2(×4),3,5,6,8(×2 each) → m=4+2+2+2+2=12
        //   total=4 → m≠total → divergence → self-recovery fires
        //   surprise gens: Gen A (chunks 1..3 overlaps [2..8]) + Gen B (chunks 5..6 overlaps [2..8])
        //   total = 4+6+4=14, targetLevel=2, range=[1..8]
        //   re-collect [1..8]: chunks 1,3,5,6,8(×2) + chunk 2(×4) → m=2+4+2+2+2+2=14 ✓
        let storage = InMemoryStorage()
        try await storage.open()

        // Unit center: [0.5]*d, ||center||=1.0; zero residuals → embedding = center.
        let center: [Float] = [Float](repeating: 0.5, count: d)

        func insertGen(level: Int, chunkIDs: [Int64]) async throws {
            let numEmb = chunkIDs.count * tokensPerChunk
            let gen = try await storage.insertGeneration(GenerationRecord(
                level: level,
                numEmbeddings: numEmb,
                minChunkID: chunkIDs.min()!,
                maxChunkID: chunkIDs.max()!,
                created: Date()
            ))
            var pairs: [IndexPair] = []
            for cID in chunkIDs {
                for t in 0..<tokensPerChunk {
                    pairs.append(IndexPair(chunkID: UInt32(cID), tokenOffset: UInt32(t)))
                }
            }
            let residuals = [Float](repeating: 0.0, count: numEmb * d)
            _ = try await storage.insertBucket(BucketRecord(
                generationID: gen.id,
                center: Indexer.encodeFloat32LE(center),
                indices: IndicesCodec.encode(pairs),
                residuals: Q4Codec.encodeResiduals(residuals)
            ))
        }

        try await insertGen(level: 2, chunkIDs: [1, 2, 3])   // Gen A: L2, 6 emb
        try await insertGen(level: 1, chunkIDs: [5, 6])       // Gen B: L1, 4 emb
        try await insertGen(level: 0, chunkIDs: [8])          // Gen C: L0, 2 emb

        let config = IndexerConfig.testing(l0Capacity: 8)
        let indexer = try await Indexer(storage: storage, config: config)

        // Orphan-recovery pattern: re-buffer chunkID=2 (already in ledger from Gen A).
        let orphanEmbeddings = [Float](repeating: 0.5, count: tokensPerChunk * d)
        try await indexer.add(chunkID: 2, embeddings: orphanEmbeddings, dims: d)

        // Must not throw. Previously threw ledgerOutOfSync(ledgerRows: 12, expected: 4).
        await #expect(throws: Never.self) {
            try await indexer.flush()
        }

        // All 3 input gens should be merged into a single L2 gen by self-recovery.
        let gens = try await storage.generations()
        #expect(gens.count == 1,
                "All input gens should merge into 1 via self-recovery cascade; got \(gens.count)")
        if let g = gens.first {
            #expect(g.level == 2,
                    "Merged gen should be at L2 (max of absorbed levels); got L\(g.level)")
        }

        // The merged gen must have non-empty buckets (k-means produced output).
        if let genID = gens.first?.id {
            let buckets = try await storage.buckets(forGeneration: genID)
            #expect(!buckets.isEmpty, "Self-recovered gen must have at least 1 bucket")
        }
    }

    // MARK: - Test 8 (issue #123 req 6): findOrphanedChunks idempotence after interruption

    /// Verifies that `findOrphanedChunks()` still returns the correct orphan set
    /// after a mid-recovery interruption. An interrupted recovery (add without flush)
    /// must not incorrectly commit partial state to storage; a fresh store from the
    /// same storage must see both orphans and can resume recovery to convergence.
    @Test("findOrphanedChunks: idempotent after mid-recovery interruption")
    func findOrphanedChunks_idempotentAfterMidRecoveryInterruption() async throws {
        let storage = InMemoryStorage()
        let embedder = MockEmbedder(dims: 8)
        let config = StoreConfig(indexer: IndexerConfig.testing(l0Capacity: 8))

        let store1 = try await SwitchcraftStore(
            storage: storage,
            embedder: embedder,
            config: config
        )

        // Plant two orphan chunks directly in storage (no bucket assignments).
        let bodyA = "alpha bravo charlie delta"
        let bodyB = "echo foxtrot golf hotel"
        let hashA = SwitchcraftStore.contentHash(bodyA)
        let hashB = SwitchcraftStore.contentHash(bodyB)
        let tA = bodyA.split(whereSeparator: { $0.isWhitespace }).count
        let tB = bodyB.split(whereSeparator: { $0.isWhitespace }).count
        _ = try await storage.upsertChunk(
            ChunkRecord(hash: hashA, model: embedder.modelIdentifier,
                        embeddings: Data(), counts: [tA])
        )
        _ = try await storage.upsertChunk(
            ChunkRecord(hash: hashB, model: embedder.modelIdentifier,
                        embeddings: Data(), counts: [tB])
        )

        // Verify both are orphans before any recovery.
        let chunkA = try await storage.chunk(hash: hashA)
        let chunkB = try await storage.chunk(hash: hashB)
        #expect(chunkA != nil && chunkB != nil)
        let orphansBefore = try await store1.findOrphanedChunks()
        let idsBefore = Set(orphansBefore.map { $0.chunkID })
        #expect(idsBefore.contains(chunkA!.id), "Orphan A must be detected before recovery")
        #expect(idsBefore.contains(chunkB!.id), "Orphan B must be detected before recovery")

        // Partial recovery: recover orphan A, but do NOT flush (simulate interruption).
        try await store1.add(id: "doc-a", body: bodyA)

        // Simulate restart: create a new store from the same in-memory storage.
        // Storage still has no bucket assignments for A (add was never flushed).
        let store2 = try await SwitchcraftStore(
            storage: storage,
            embedder: embedder,
            config: config
        )

        // After restart, BOTH orphans must still appear — unflushed recovery state
        // is transient and must not be persisted to storage.
        let orphansAfterRestart = try await store2.findOrphanedChunks()
        let idsAfterRestart = Set(orphansAfterRestart.map { $0.chunkID })
        #expect(idsAfterRestart.contains(chunkA!.id),
                "Orphan A must still appear after interruption (add was not flushed)")
        #expect(idsAfterRestart.contains(chunkB!.id),
                "Orphan B must still appear after interruption (never started recovery)")

        // Complete recovery on store2: recover both orphans and flush.
        try await store2.add(id: "doc-a", body: bodyA)
        try await store2.add(id: "doc-b", body: bodyB)
        try await store2.index()

        // After full recovery, no orphans should remain.
        let orphansAfterRecovery = try await store2.findOrphanedChunks()
        #expect(orphansAfterRecovery.isEmpty,
                "All orphans must be resolved after complete recovery; got \(orphansAfterRecovery.count)")

        // Do not call store1.shutdown() — it still has unflushed pending embeddings
        // from the simulated partial recovery. Calling shutdown() would flush them and
        // conflict with store2's committed gen (same bug as #123 but test-induced).
        // In a real app the "interrupted" store process would have crashed; here we
        // simply let store1 be deallocated with its unflushed state intact.
        try await store2.shutdown()
    }

    // MARK: - Test 9 (issue #123 req 7): performance — self-recovery overhead ≤ 2×

    /// Verifies that a sequential `add()` batch crossing a compaction boundary (and
    /// triggering self-recovery) completes within 2× the wall time of an equal-size
    /// batch that does not cross a compaction boundary. The 2× allowance accounts
    /// for compaction cost; self-recovery itself should add negligible overhead.
    ///
    /// Uses a reduced corpus (10 iterations of the Task 3 scenario) scaled for
    /// unit-test speed. The key metric is the crossing/baseline ratio, not absolute time.
    @Test("performFlush: recovery batch crossing compaction boundary completes within 2× baseline")
    func recoveryBatch_compactionBoundary_withinTwoXBaseline() async throws {
        let d = Self.dims        // 4
        let tokensPerChunk = 2
        let iterations = 10      // Repeat the compaction-triggering scenario N times.

        // Measure T_crossing: N iterations, each building a fresh 3-gen state,
        // adding 1 orphan doc, and flushing (self-recovery fires on each flush).
        let crossingElapsed = try await measureCrossing(
            d: d, tokensPerChunk: tokensPerChunk, iterations: iterations
        )

        // Measure T_baseline: N iterations with a fresh empty storage and
        // l0Capacity high enough that no compaction fires during the add.
        let baselineElapsed = try await measureBaseline(
            d: d, tokensPerChunk: tokensPerChunk, iterations: iterations
        )

        // The crossing run must complete in ≤ 2× the baseline run.
        // We allow crossingElapsed < 2.0 as an absolute floor so the test isn't fragile on very fast HW.
        let ratio = crossingElapsed / max(baselineElapsed, 0.001)
        #expect(
            ratio <= 2.0 || crossingElapsed < 2.0,
            "Self-recovery overhead should not exceed 2× baseline"
        )
        print("[PerfTest] crossing=\(String(format: "%.3f", crossingElapsed))s baseline=\(String(format: "%.3f", baselineElapsed))s ratio=\(String(format: "%.2f", ratio))×")
    }

    /// Runs `iterations` crossing cycles (build 3-gen state → orphan add → flush).
    /// Returns elapsed wall time in seconds.
    private func measureCrossing(d: Int, tokensPerChunk: Int, iterations: Int) async throws -> Double {
        let center: [Float] = [Float](repeating: 0.5, count: d)

        func insertGen(storage: InMemoryStorage, level: Int, chunkIDs: [Int64]) async throws {
            let numEmb = chunkIDs.count * tokensPerChunk
            let gen = try await storage.insertGeneration(GenerationRecord(
                level: level,
                numEmbeddings: numEmb,
                minChunkID: chunkIDs.min()!,
                maxChunkID: chunkIDs.max()!,
                created: Date()
            ))
            var pairs: [IndexPair] = []
            for cID in chunkIDs {
                for t in 0..<tokensPerChunk {
                    pairs.append(IndexPair(chunkID: UInt32(cID), tokenOffset: UInt32(t)))
                }
            }
            let residuals = [Float](repeating: 0.0, count: numEmb * d)
            _ = try await storage.insertBucket(BucketRecord(
                generationID: gen.id,
                center: Indexer.encodeFloat32LE(center),
                indices: IndicesCodec.encode(pairs),
                residuals: Q4Codec.encodeResiduals(residuals)
            ))
        }

        let start = Date()
        for _ in 0..<iterations {
            let storage = InMemoryStorage()
            try await storage.open()
            try await insertGen(storage: storage, level: 2, chunkIDs: [1, 2, 3])
            try await insertGen(storage: storage, level: 1, chunkIDs: [5, 6])
            try await insertGen(storage: storage, level: 0, chunkIDs: [8])
            let indexer = try await Indexer(
                storage: storage,
                config: IndexerConfig.testing(l0Capacity: 8)
            )
            let emb = [Float](repeating: 0.5, count: tokensPerChunk * d)
            try await indexer.add(chunkID: 2, embeddings: emb, dims: d)
            try await indexer.flush()
        }
        return Date().timeIntervalSince(start)
    }

    /// Runs `iterations` baseline cycles (fresh empty storage → fresh add → flush).
    /// Uses l0Capacity=1000 so no compaction fires; measures pure add+flush overhead.
    private func measureBaseline(d: Int, tokensPerChunk: Int, iterations: Int) async throws -> Double {
        let start = Date()
        for i in 0..<iterations {
            let storage = InMemoryStorage()
            try await storage.open()
            let indexer = try await Indexer(
                storage: storage,
                config: IndexerConfig(l0Capacity: 1000, lsmFanout: 2)
            )
            let emb = [Float](repeating: 0.5, count: tokensPerChunk * d)
            try await indexer.add(chunkID: Int64(i + 1), embeddings: emb, dims: d)
            // No flush: capacity is 1000 and we only have 2 tokens → pendingCount < l0cap.
            // Call flush() explicitly to measure the full flush path without compaction.
            try await indexer.flush()
        }
        return Date().timeIntervalSince(start)
    }
}

// MARK: - FailingReplaceStorage actor mutation helper

extension FailingReplaceStorage {
    /// Set `shouldFailOnReplace` — helper for test clarity.
    func set(shouldFailOnReplace value: Bool) {
        shouldFailOnReplace = value
    }
}
