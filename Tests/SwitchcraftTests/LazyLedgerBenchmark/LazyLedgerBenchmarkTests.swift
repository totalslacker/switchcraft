// SPDX-License-Identifier: Apache-2.0
//
// Sustained-workload acceptance benchmark for issue #137 (lazy embedding
// materialization), Requirement 11.
//
// # What this tests
//
// `LazyLedgerMaterializationTests` (Requirement 9) proves functional
// correctness at a small (10K-chunk) scale that runs quickly in CI. This
// suite is the separate, larger, opt-in benchmark demonstrating the actual
// perf improvement at a scale where the pre-refactor eager rehydration walk
// is measurably slow and memory-hungry: 100,000 chunks / ~1,000,000
// embeddings, dims=768 (matching the production scale quoted in the issue,
// one order of magnitude below the observed 11.1M-embedding corpus).
//
// The fixture is synthesized directly into storage (bypassing k-means/flush
// entirely — one bucket per generation, containing every pair for that
// generation) so corpus construction costs O(pairs×dims) once (to encode the
// synthetic residuals) rather than paying k-means training cost on top; this
// suite is about rehydration cost, not compaction/clustering cost.
//
// "Pre-refactor" is `LazyLedgerMaterializationTests.eagerGoldenReconstruction`
// — a test-local copy of the deleted `center[j] + residuals[base+j]`
// reconstruction loop the old `Indexer.init` ran for every bucket. The real
// production code for that loop no longer exists (this issue replaced it),
// so the "before" comparator is necessarily a test-local reimplementation,
// not shared production code.
//
// "Post-refactor" is the real `Indexer.init` full-walk path, with the
// on-disk ledger snapshot (ADR 034 / issue #136) explicitly cleared first —
// otherwise `Indexer.init` would take the unrelated snapshot fast path
// instead of the full rehydration walk this issue changes.
//
// # Acceptance
//
// Memory is asserted as a ratio in both build configurations (it holds
// reliably either way): steady-state ledger memory (phys_footprint delta)
// improves by >= 30×.
//
// Time is asserted differently per build configuration, because Release's
// `-O` auto-vectorizes the deleted eager loop's per-dimension scalar
// arithmetic far more than it helps the lazy path's bookkeeping-dominated
// cost (dictionary/array operations, not vectorizable arithmetic) — so the
// *ratio* between them shrinks substantially in Release even though the
// lazy path is still, in absolute terms, dramatically faster:
//   - Debug: rehydrate time improves by >= 50× (confirms the algorithmic
//     complexity change — O(pairs×dims) → O(pairs) — actually took effect;
//     Debug's unvectorized arithmetic makes this ratio a faithful proxy for
//     the underlying big-O win).
//   - Release: rehydrate completes within an absolute wall-clock ceiling
//     (see `releaseAbsoluteTimeCeiling` below for the measured basis this
//     was picked from). This is what users actually experience. The
//     Release ratio is still logged for observability but is intentionally
//     NOT asserted — see ADR 035 §8 for why chasing a higher Release ratio
//     would be optimizing the wrong metric.
//
// # How to run
//
//   SWITCHCRAFT_LAZY_LEDGER_BENCHMARK=1 swift test \
//     --filter LazyLedgerBenchmarkTests
//
// Requires several GB free RAM (dims=768 × 1M embeddings nominal ~3 GB for
// the eager/"pre-refactor" ledger alone). Fixture synthesis itself skips
// k-means entirely and is fast; the "pre-refactor" eager reconstruction it
// times is not — expect roughly 2 minutes total in a debug build (the eager
// loop's per-dimension Swift arithmetic is unvectorized in debug), well
// under a minute in `-c release`. macOS-only (uses task_vm_info.phys_footprint,
// matching `CompactionMemoryRegressionTests`); on other platforms the test
// prints "[SKIP]" and returns without assertion.

import Foundation
import Testing
@testable import SwitchcraftCore

#if os(macOS)
import Darwin.Mach
#endif

private let benchmarkEnabled =
    ProcessInfo.processInfo.environment["SWITCHCRAFT_LAZY_LEDGER_BENCHMARK"] != nil

@Suite("LazyLedgerBenchmark", .serialized,
       .enabled(if: benchmarkEnabled,
                "Set SWITCHCRAFT_LAZY_LEDGER_BENCHMARK=1 to run the sustained-workload benchmark"))
struct LazyLedgerBenchmarkTests {

    // MARK: - Memory measurement (mirrors CompactionMemoryRegressionTests)

#if os(macOS)
    static func physFootprint() -> UInt64 {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(
            MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size
        )
        let kr: kern_return_t = withUnsafeMutablePointer(to: &info) { ptr in
            ptr.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { intPtr in
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), intPtr, &count)
            }
        }
        guard kr == KERN_SUCCESS else { return 0 }
        return info.phys_footprint
    }
#else
    static func physFootprint() -> UInt64 { 0 }
#endif

    // MARK: - Fixture synthesis

    /// Directly synthesize `chunkCount` chunks × `tokensPerChunk` tokens into
    /// `storage`, split across `genCount` generations (one bucket per
    /// generation, containing every pair for that generation). Bypasses
    /// `Indexer.add`/`flush`/k-means entirely — this benchmark measures
    /// rehydration cost, not compaction/clustering cost, and a single
    /// synthetic bucket per generation is just as valid a fixture for
    /// exercising O(pairs×dims) reconstruction as many small ones.
    static func synthesizeFixture(
        storage: any SwitchcraftStorage,
        chunkCount: Int, tokensPerChunk: Int, dims: Int, genCount: Int,
        rng: inout SplitMix64
    ) async throws {
        precondition(chunkCount % genCount == 0)
        let chunksPerGen = chunkCount / genCount
        var nextChunkID: Int64 = 1
        for _ in 0..<genCount {
            let center = IndexerTests.randomRow(dims: dims, rng: &rng)
            var pairs: [IndexPair] = []
            pairs.reserveCapacity(chunksPerGen * tokensPerChunk)
            var residualValues: [Float] = []
            residualValues.reserveCapacity(chunksPerGen * tokensPerChunk * dims)
            let minChunkID = nextChunkID
            for _ in 0..<chunksPerGen {
                for t in 0..<tokensPerChunk {
                    pairs.append(IndexPair(chunkID: UInt32(nextChunkID), tokenOffset: UInt32(t)))
                    let embedding = IndexerTests.randomRow(dims: dims, rng: &rng)
                    for j in 0..<dims {
                        residualValues.append(embedding[j] - center[j])
                    }
                }
                nextChunkID += 1
            }
            let maxChunkID = nextChunkID - 1
            let gen = try await storage.insertGeneration(GenerationRecord(
                level: 0, numEmbeddings: pairs.count,
                minChunkID: minChunkID, maxChunkID: maxChunkID, created: Date()
            ))
            let bucket = BucketRecord(
                generationID: gen.id,
                center: Indexer.encodeFloat32LE(center),
                indices: IndicesCodec.encode(pairs),
                residuals: Q4Codec.encodeResiduals(residualValues)
            )
            _ = try await storage.insertBucket(bucket)
        }
    }

    // MARK: - Benchmark

    /// Release-mode absolute ceiling for post-refactor rehydrate wall-clock
    /// time at this benchmark's 100K-chunk/1M-embedding/dims=768 fixture.
    ///
    /// Picked from evidence, not guessed: measured Release post-refactor
    /// time on the reference machine was ~0.20s (well under the "much less
    /// than 5s" case called out during the Requirement 11 acceptance
    /// discussion), so per that discussion's own guidance the ceiling is
    /// tightened rather than left at a generic 5s. 1.0s is ~5× the observed
    /// figure — enough headroom for slower CI hardware while still being a
    /// meaningful gate against a real Release-mode regression.
    private static let releaseAbsoluteTimeCeiling: TimeInterval = 1.0

    @Test("post-refactor rehydrate clears the Debug ratio bar and the Release absolute-time ceiling; memory clears its ratio bar in both")
    func sustainedWorkloadRehydrateSpeedupAndMemory() async throws {
#if !os(macOS)
        print("[SKIP] LazyLedgerBenchmarkTests requires macOS (task_vm_info).")
        return
#endif

        let dims = 768
        let chunkCount = 100_000
        let tokensPerChunk = 10
        let genCount = 20

        let storage = InMemoryStorage()
        try await storage.open()
        var rng = SplitMix64(seed: 0xBEEF_CAFE)

        print("[LazyLedgerBenchmark] Synthesizing fixture: \(chunkCount) chunks × \(tokensPerChunk) tokens, dims=\(dims) …")
        try await Self.synthesizeFixture(
            storage: storage, chunkCount: chunkCount, tokensPerChunk: tokensPerChunk,
            dims: dims, genCount: genCount, rng: &rng
        )
        let totalEmbeddings = chunkCount * tokensPerChunk
        print("[LazyLedgerBenchmark] Fixture ready: \(totalEmbeddings) embeddings across \(genCount) generations.")

        // No snapshot exists yet (fixture was synthesized directly, bypassing
        // flush()'s writeLedgerSnapshot), so both phases below already take
        // the full rehydration walk — but clear defensively in case that
        // ever changes, so this benchmark keeps measuring what it claims to.
        try await storage.clearLedgerSnapshot()

        // `.throwError` avoids `.autoRecover`'s conflict-bookkeeping overhead
        // (chunkGenSets, decodedBuckets metadata, winner/loser maps) — all
        // unused here since a freshly-synthesized fixture with disjoint
        // chunkID ranges per generation never conflicts. This keeps the
        // measured "post-refactor" cost representative of the lazy
        // bucket-ref population this issue is actually about, rather than
        // mixing in unrelated auto-recovery bookkeeping this fixture never
        // exercises.
        let config = IndexerConfig(l0Capacity: 1024, lsmFanout: 16, rehydrationConflictBehavior: .throwError)

        // Warm up one-time async-runtime costs (executor pools, Foundation
        // lazy singletons, type metadata) against a throwaway empty store,
        // so they land in neither phase's measured RSS delta below — with
        // multi-GB "pre" deltas and tens-of-MB "post" deltas, misattributing
        // even a modest fixed cost could meaningfully distort the ratio.
        let warmupStorage = InMemoryStorage()
        try await warmupStorage.open()
        _ = try await Indexer(storage: warmupStorage, config: config)
        _ = try await LazyLedgerMaterializationTests.eagerGoldenReconstruction(storage: warmupStorage)

        // Phase 1 ("pre-refactor"): the deleted eager reconstruction loop,
        // run directly against storage.
        let preRSSBefore = Self.physFootprint()
        let preStart = Date()
        var eagerLedger: [Int64: [[Float]]]? = try await LazyLedgerMaterializationTests.eagerGoldenReconstruction(storage: storage)
        let preElapsed = Date().timeIntervalSince(preStart)
        let preRSSAfter = Self.physFootprint()
        let preRSSDelta = preRSSAfter > preRSSBefore ? preRSSAfter - preRSSBefore : 0
        #expect(eagerLedger?.count == chunkCount)
        eagerLedger = nil // release before phase 2 so its footprint doesn't linger

        // Phase 2 ("post-refactor"): the real Indexer.init full-walk path.
        let postRSSBefore = Self.physFootprint()
        let postStart = Date()
        let reopened = try await Indexer(storage: storage, config: config)
        let postElapsed = Date().timeIntervalSince(postStart)
        let postRSSAfter = Self.physFootprint()
        let postRSSDelta = postRSSAfter > postRSSBefore ? postRSSAfter - postRSSBefore : 0
        #expect(await reopened.didUseSnapshotFastPath == false)

        let timeRatio = preElapsed / max(postElapsed, 1e-9)
        let rssRatio = Double(preRSSDelta) / Double(max(postRSSDelta, 1))

        print("[LazyLedgerBenchmark] Pre-refactor  rehydrate: \(String(format: "%.4f", preElapsed))s, RSS delta \(preRSSDelta >> 20) MB")
        print("[LazyLedgerBenchmark] Post-refactor rehydrate: \(String(format: "%.4f", postElapsed))s, RSS delta \(postRSSDelta >> 20) MB")
        print("[LazyLedgerBenchmark] Speedup: \(String(format: "%.1f", timeRatio))×  Memory improvement: \(String(format: "%.1f", rssRatio))×")

        // Memory: the ratio bar holds reliably in both build configurations
        // (Requirement 11), so it's asserted unconditionally.
        #expect(rssRatio >= 30,
                "steady-state ledger memory should improve by >= 30x; measured \(String(format: "%.1f", rssRatio))x (pre=\(preRSSDelta) bytes, post=\(postRSSDelta) bytes)")

        // Time: Debug asserts the ratio (a faithful proxy for the O(pairs×dims)
        // → O(pairs) algorithmic change, since Debug's unvectorized arithmetic
        // doesn't compress it); Release asserts absolute wall-clock instead,
        // since that's what users experience and the ratio metric shrinks in
        // Release for reasons unrelated to whether the optimization worked
        // (see the file-header comment and ADR 035 §8).
#if DEBUG
        #expect(timeRatio >= 50,
                "Debug rehydrate time should improve by >= 50x; measured \(String(format: "%.1f", timeRatio))x (pre=\(preElapsed)s, post=\(postElapsed)s)")
#else
        print("[LazyLedgerBenchmark] Release ratio (informational, not asserted): \(String(format: "%.1f", timeRatio))×")
        #expect(postElapsed < Self.releaseAbsoluteTimeCeiling,
                "Release post-refactor rehydrate should complete within \(Self.releaseAbsoluteTimeCeiling)s; measured \(postElapsed)s")
#endif
    }
}
