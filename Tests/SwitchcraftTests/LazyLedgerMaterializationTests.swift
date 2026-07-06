// SPDX-License-Identifier: Apache-2.0
import Foundation
import Testing
@testable import SwitchcraftCore

/// Reproducer + parity tests for issue #137 (lazy embedding materialization).
///
/// **Requirement 9** (functional reproducer): a 10K-chunk synthetic corpus
/// that forces multiple compactions, instrumented via `CountingBucketStorage`
/// to count storage-layer bucket reads. Asserts:
///   - Rehydrating the ledger (`Indexer.init`) costs exactly one
///     `buckets(forGeneration:)` call per active generation — the cheap
///     Requirement 13 integrity walk — regardless of how many tokens each
///     generation holds. No residual-value decode happens at this point.
///   - Materializing the rehydrated (all-`.bucketRef`) ledger — via
///     `ledgerContents()`, standing in for any compaction that samples these
///     rows — happens strictly *after* rehydration, batches to the same
///     one-call-per-generation cost (Requirement 6), and reproduces a golden
///     `[Float]` reference computed independently, directly from raw storage
///     bytes via the pre-#137 `center + dequantized residual` formula.
///
/// A second, small deterministic test (`compactionReadsBucketRefDataAtCascadeTime`)
/// confirms a *real* compaction — not just the `ledgerContents()` test
/// accessor — also triggers bucket reads at the moment it runs: decode work
/// happens at compaction time, not only when a test asks for it.
///
/// **Requirement 10** (parity): the golden-reference comparison above is
/// exactly the "batched lazy decode of a bucket == per-token eager decode of
/// the same bucket" claim — the golden reference is computed per-bucket
/// directly from raw bytes, independent of any batching, and must equal
/// `ledgerContents()`'s batched-materialization output bit-for-bit. Separately,
/// single-flush parity (nothing is a bucket-ref yet on a first-ever flush, so
/// the lazy and pre-#137 eager code paths are trivially identical — both
/// just copy `.materialized` rows with zero storage I/O) is already exercised
/// by every existing single-flush Indexer test (e.g.
/// `IndexerTests.flushWritesKBucketsAssignsEveryToken`, the NFCorpus and
/// 33-fact cross-implementation benchmarks) and isn't duplicated here.
@Suite("Lazy Ledger Materialization (issue #137)")
struct LazyLedgerMaterializationTests {

    static let dims = 8
    static let totalChunks = 10_000
    static let batchSize = 500

    /// Directly reconstructs every token's embedding from raw storage bytes
    /// using the pre-#137 eager formula (`center[j] + dequantized
    /// residual[base+j]`), independent of `Indexer`/`materialize` entirely.
    /// This is the "golden reference" — a test-local copy of the deleted
    /// reconstruction loop, not shared production code. Also reused by
    /// `LazyLedgerBenchmarkTests` as the "pre-refactor" timing/memory
    /// baseline, since it exercises the same O(pairs×dims) work the old
    /// `Indexer.init` eager rehydration loop did.
    static func eagerGoldenReconstruction(
        storage: any SwitchcraftStorage
    ) async throws -> [Int64: [[Float]]] {
        var out: [Int64: [(tokenOffset: UInt32, embedding: [Float])]] = [:]
        for gen in try await storage.generations() {
            for bucket in try await storage.buckets(forGeneration: gen.id) {
                let center = Indexer.decodeFloat32LE(bucket.center)
                let d = center.count
                let pairs = try IndicesCodec.decode(bucket.indices)
                let residuals = Q4Codec.decodeResiduals(bucket.residuals)
                for (i, pair) in pairs.enumerated() {
                    var embedding = [Float]()
                    embedding.reserveCapacity(d)
                    let base = i * d
                    for j in 0..<d {
                        embedding.append(center[j] + residuals[base + j])
                    }
                    out[Int64(pair.chunkID), default: []].append(
                        (tokenOffset: pair.tokenOffset, embedding: embedding)
                    )
                }
            }
        }
        var result: [Int64: [[Float]]] = [:]
        result.reserveCapacity(out.count)
        for (chunkID, entries) in out {
            result[chunkID] = entries.sorted { $0.tokenOffset < $1.tokenOffset }.map { $0.embedding }
        }
        return result
    }

    @Test("10K-chunk corpus: rehydrate is O(generations) with no decode; materialization happens later, batched, and matches the golden reference")
    func reproducer_lazyMaterializationTimingAndParity() async throws {
        let storage = CountingBucketStorage()
        try await storage.open()
        // .throwError avoids any conflict-recovery bookkeeping (irrelevant
        // here — a clean sequential ingest never produces a chunkID
        // conflict) so the read-count assertions below are unambiguous.
        let config = IndexerConfig.testing(l0Capacity: 500, lsmFanout: 4, rehydrationConflictBehavior: .throwError)
        var indexer: Indexer? = try await Indexer(storage: storage, config: config)
        var rng = SplitMix64(seed: 137)

        // Ingest in batches, flushing after each — forces multiple
        // compactions/cascades across levels well before all 10K chunks are
        // in, so later batches' flushes must merge (and thus, under this
        // issue's design, lazily re-materialize) earlier-flushed chunks.
        var nextChunkID: Int64 = 1
        for _ in stride(from: 0, to: Self.totalChunks, by: Self.batchSize) {
            for _ in 0..<Self.batchSize {
                let row = IndexerTests.randomRow(dims: Self.dims, rng: &rng)
                try await indexer!.add(chunkID: nextChunkID, embeddings: row, dims: Self.dims)
                nextChunkID += 1
            }
            try await indexer!.flush()
        }
        let gensAfterIngest = try await storage.generations()
        #expect(gensAfterIngest.count >= 1)
        let totalEmbeddingsAfterIngest = gensAfterIngest.reduce(0) { $0 + $1.numEmbeddings }
        #expect(totalEmbeddingsAfterIngest == Self.totalChunks)

        // Drop the in-process indexer and reopen fresh — the only way to
        // observe the rehydration walk in isolation.
        indexer = nil
        await storage.resetBucketReadCount()
        let reopened = try await Indexer(storage: storage, config: config)

        // Rehydrate cost: exactly one buckets(forGeneration:) call per
        // active generation (the Requirement 13 integrity walk), regardless
        // of how many of the 10,000 tokens live in each generation. This is
        // the "not at rehydrate time" half of the claim.
        let rehydrateReads = await storage.bucketReadCount
        #expect(rehydrateReads == gensAfterIngest.count,
                "rehydrate should read each active generation's buckets exactly once (integrity walk only)")

        // Golden reference: reconstruct every token directly from raw
        // storage bytes, completely independent of Indexer/materialize.
        let golden = try await Self.eagerGoldenReconstruction(storage: storage)
        #expect(golden.count == Self.totalChunks)

        // Materialize the rehydrated (100% bucket-ref) ledger. This is the
        // "at compaction time" half: decode work happens here, strictly
        // after the rehydrate-only phase above, and batches to one
        // buckets(forGeneration:) call per generation — not one per token,
        // even though there are 10,000 tokens spread across (likely) far
        // fewer generations.
        await storage.resetBucketReadCount()
        let materialized = try await reopened.ledgerContents()
        let materializationReads = await storage.bucketReadCount
        #expect(materializationReads == gensAfterIngest.count,
                "materialization should batch to one buckets(forGeneration:) call per generation, not per token")
        #expect(materializationReads > 0, "materialization must actually touch storage — proves decode isn't a no-op")

        // Parity (Requirement 10b): batched lazy decode reproduces exactly
        // the same values as the independent per-bucket golden reference.
        #expect(materialized.count == golden.count)
        for (chunkID, expectedRows) in golden {
            let actualRows = try #require(materialized[chunkID], "chunk \(chunkID) missing from materialized ledger")
            #expect(actualRows == expectedRows, "chunk \(chunkID) materialized value diverges from golden reference")
        }
    }

    /// Confirms a *real* compaction (not the `ledgerContents()` test
    /// accessor used above) triggers bucket reads at the moment it runs —
    /// decode work occurs at compaction time, not only when a test asks for
    /// it. Deterministic by construction (mirrors
    /// `IndexerTests.cascadeCollapsesLowerLevels`'s flush-2 case): with
    /// `l0Capacity=4, lsmFanout=2`, `cap(0) == 8`. The first flush creates
    /// one level-0 generation of 4 (all its tokens become bucket-refs); the
    /// second flush's `pending(4) + existing L0 sum(4) == 8 <= cap(0)`, so it
    /// *must* merge that existing bucket-ref generation into the new one —
    /// unlike appending to the big 10K corpus in the test above, whose exact
    /// end-of-ingest cascade state (and therefore whether one more batch
    /// happens to touch existing data) isn't something a test should assert
    /// on without controlling it directly.
    @Test("a real cascade compaction reads and materializes existing bucket-ref generations")
    func compactionReadsBucketRefDataAtCascadeTime() async throws {
        let storage = CountingBucketStorage()
        try await storage.open()
        let config = IndexerConfig.testing(l0Capacity: 4, lsmFanout: 2, rehydrationConflictBehavior: .throwError)
        let indexer = try await Indexer(storage: storage, config: config)
        var rng = SplitMix64(seed: 7)

        for i in 0..<4 {
            let row = IndexerTests.randomRow(dims: Self.dims, rng: &rng)
            try await indexer.add(chunkID: Int64(i + 1), embeddings: row, dims: Self.dims)
        }
        try await indexer.flush()
        let gensAfterFirstFlush = try await storage.generations()
        #expect(gensAfterFirstFlush.count == 1)
        #expect(gensAfterFirstFlush.first?.level == 0)

        await storage.resetBucketReadCount()
        for i in 4..<8 {
            let row = IndexerTests.randomRow(dims: Self.dims, rng: &rng)
            try await indexer.add(chunkID: Int64(i + 1), embeddings: row, dims: Self.dims)
        }
        try await indexer.flush()

        let gensAfterSecondFlush = try await storage.generations()
        #expect(gensAfterSecondFlush.count == 1)
        #expect(gensAfterSecondFlush.first?.numEmbeddings == 8)

        let compactionReads = await storage.bucketReadCount
        #expect(compactionReads > 0, "the second flush must read/materialize the first flush's bucket-ref generation")
    }

    /// Regression test for a gap in the Requirement 13 integrity walk found
    /// during review: `checkedResidualsSize` computed `expectedBytes` as
    /// `pairsCount * dims / 2` and compared it directly against the blob
    /// size, but never checked that `pairsCount * dims` is itself even.
    /// `Q4Codec.encodeResiduals` requires an even value count (two 4-bit
    /// levels packed per byte), so a bucket with an odd `pairsCount * dims`
    /// can only exist via corruption — and integer division would silently
    /// floor that odd product, letting a residuals blob truncated by
    /// exactly one packed value slip past the check undetected.
    ///
    /// Plants exactly that: 1 pair × dims=3 (odd product = 3) with a
    /// residuals blob of the floor-divided 1 byte, bypassing `Indexer.add`/
    /// `flush` entirely so the corruption is under direct control.
    @Test("odd pairsCount×dims with a floor-divided-matching residuals blob is still rejected as corrupt")
    func oddPairsTimesDimsIsRejectedAsCorrupt() async throws {
        let storage = InMemoryStorage()
        try await storage.open()

        let dims = 3
        let gen = try await storage.insertGeneration(GenerationRecord(
            level: 0, numEmbeddings: 1, minChunkID: 1, maxChunkID: 1, created: Date()
        ))
        let pairs = [IndexPair(chunkID: 1, tokenOffset: 0)]
        // 1 pair × 3 dims = 3 values (odd) — never producible by
        // Q4Codec.encodeResiduals, which preconditions an even count. A
        // corrupt/truncated blob could plausibly still be 1 byte (the
        // floor of 3/2), which the pre-fix check accepted.
        let corruptResiduals = Data([0x00])
        let bucket = BucketRecord(
            generationID: gen.id,
            center: Indexer.encodeFloat32LE([Float](repeating: 0, count: dims)),
            indices: IndicesCodec.encode(pairs),
            residuals: corruptResiduals
        )
        _ = try await storage.insertBucket(bucket)

        let config = IndexerConfig.testing(rehydrationConflictBehavior: .throwError)
        await #expect(throws: Indexer.Error.rehydrationBucketCorrupt(generationID: gen.id)) {
            _ = try await Indexer(storage: storage, config: config)
        }
    }
}
