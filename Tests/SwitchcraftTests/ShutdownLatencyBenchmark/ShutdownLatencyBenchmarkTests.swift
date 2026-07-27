// SPDX-License-Identifier: Apache-2.0
//
// Corpus-scale acceptance benchmark for issue #144 (persistSnapshot/WAL-
// checkpoint shutdown cost), Requirement 6.
//
// # What this tests
//
// `LedgerSnapshotCodecTests` proves `LedgerSnapshotCodec`'s correctness at
// small (single/double-digit chunk) scale. This suite is the separate,
// larger, opt-in benchmark demonstrating that `SwitchcraftStore.shutdown()`
// meets its <2s p95 target at a scale representative of the operator corpus
// that motivated this issue: ~7,300 active chunks / ~11.1M embeddings,
// dims=768 (the production scale quoted in ADR 034 / WebBrain#254).
//
// # Fixture construction
//
// Unlike `LazyLedgerBenchmarkTests` (which synthesizes real Q4-quantized
// bucket blobs to exercise bucket *rehydration*), this benchmark targets
// `persistSnapshot()`/`writeLedgerSnapshot()`/`LedgerSnapshotCodec.encode()`
// — the cost is driven purely by the ledger's row count and token kind, not
// by whether real bucket blobs exist on disk. Synthesizing real residual
// data at this scale (~11.1M × 768 Float32 values ≈ 34GB before Q4
// quantization) is infeasible for a test fixture.
//
// Instead, the fixture is built directly as the *target* on-disk ledger
// snapshot: a pure `.bucketRef`-only ledger (matching the common case per
// ADR 036 — a post-flush/rehydration ledger is overwhelmingly bucket-refs,
// which is also the exact "idle session still pays the cost" case this
// issue targets) is encoded once via `LedgerSnapshotCodec.encode` and
// written directly to storage's ledger-snapshot slot, alongside matching
// (but otherwise-empty) `ChunkRecord`/`GenerationRecord` rows so the
// snapshot's fingerprint matches on load. `Indexer.init`'s existing snapshot
// fast path (ADR 034) then loads this ledger in O(chunks) — no bucket
// decode, no k-means, no residual synthesis — so fixture construction
// itself stays cheap (~277MB payload, matching the bucket-ref row size ×
// row count) even though the resulting fixture is production-scale.
//
// # How to run
//
//   SWITCHCRAFT_SHUTDOWN_LATENCY_BENCHMARK=1 swift test \
//     --filter ShutdownLatencyBenchmarkTests
//
// Several hundred MB of RAM; expect a few seconds total (fixture
// construction is O(chunks), not O(embeddings×dims)).

import Foundation
import Testing
@testable import Switchcraft
@testable import SwitchcraftCore

private let benchmarkEnabled =
    ProcessInfo.processInfo.environment["SWITCHCRAFT_SHUTDOWN_LATENCY_BENCHMARK"] != nil

/// `true` when compiled with optimisations (`swift test -c release`); mirrors
/// `IndexerSnapshotSpeedupTests.isReleaseBuild` / `PerformanceTests.isReleaseBuild`.
/// An absolute wall-clock ceiling at this fixture's ~11.1M-row scale is only
/// meaningful in Release: Debug's unoptimized `inout`/exclusivity-checking
/// overhead on the encode loop's hundreds of millions of small buffer writes
/// is on the order of 100× slower than Release and is not representative of
/// what users experience.
private let isReleaseBuild: Bool = {
    var debug = false
    assert({ debug = true; return true }())
    return !debug
}()

@Suite("ShutdownLatencyBenchmark", .serialized,
       .enabled(if: benchmarkEnabled && isReleaseBuild,
                "Set SWITCHCRAFT_SHUTDOWN_LATENCY_BENCHMARK=1 and run with `swift test -c release` to run the corpus-scale shutdown benchmark"))
struct ShutdownLatencyBenchmarkTests {

    /// Issue #144's documented acceptance target (REQ-6). Measured Release
    /// time on the reference machine at this fixture's scale was ~0.4s (post
    /// REQ-3's bulk-buffer encode rewrite), so the 2s ceiling itself already
    /// carries ~5× headroom for slower CI hardware while still being a
    /// meaningful gate against a real regression.
    private static let shutdownCeiling: TimeInterval = 2.0

    /// Build a `chunkCount`-chunk / `chunkCount * tokensPerChunk`-embedding
    /// bucket-ref-only ledger snapshot directly in `storage`, bypassing
    /// k-means/flush/residual synthesis entirely (see file header). Matching
    /// `ChunkRecord`/`GenerationRecord` rows are inserted so the snapshot's
    /// fingerprint matches what `Indexer.init` computes on load, letting the
    /// existing ADR 034 fast path populate the ledger in O(chunks).
    static func synthesizeSnapshotFixture(
        storage: any SwitchcraftStorage,
        chunkCount: Int, tokensPerChunk: Int, dims: Int, genCount: Int
    ) async throws {
        precondition(chunkCount % genCount == 0)
        let chunksPerGen = chunkCount / genCount

        var ledger: [Int64: [Indexer.LedgerToken]] = [:]
        ledger.reserveCapacity(chunkCount)

        var nextChunkID: Int64 = 1
        var genSpans: [(minChunkID: Int64, maxChunkID: Int64, numEmbeddings: Int)] = []
        genSpans.reserveCapacity(genCount)

        for _ in 0..<genCount {
            let minChunkID = nextChunkID
            var numEmbeddings = 0
            for _ in 0..<chunksPerGen {
                var tokens: [Indexer.LedgerToken] = []
                tokens.reserveCapacity(tokensPerChunk)
                for t in 0..<tokensPerChunk {
                    tokens.append(.bucketRef(genID: 1, bucketID: 1, pairOffset: t))
                }
                ledger[nextChunkID] = tokens
                numEmbeddings += tokensPerChunk
                _ = try await storage.upsertChunk(ChunkRecord(
                    hash: "synthetic-chunk-\(nextChunkID)", model: "synthetic", embeddings: Data()
                ))
                nextChunkID += 1
            }
            genSpans.append((minChunkID, nextChunkID - 1, numEmbeddings))
        }

        for span in genSpans {
            _ = try await storage.insertGeneration(GenerationRecord(
                level: 0, numEmbeddings: span.numEmbeddings,
                minChunkID: span.minChunkID, maxChunkID: span.maxChunkID, created: Date()
            ))
        }

        let gens = try await storage.generations()
        let chunkCountActual = try await storage.chunkCount()
        let fingerprint = LedgerSnapshotFingerprint.compute(chunkCount: chunkCountActual, generations: gens)

        let payload = await LedgerSnapshotCodec.encode(ledger, dims: dims)
        let record = LedgerSnapshotRecord(
            dims: dims,
            chunkCount: fingerprint.chunkCount,
            maxChunkID: fingerprint.maxChunkID,
            totalEmbeddings: fingerprint.totalEmbeddings,
            maxGenerationID: fingerprint.maxGenerationID,
            generationCount: fingerprint.generationCount,
            payload: payload
        )
        try await storage.saveLedgerSnapshot(record)
    }

    @Test("shutdown() completes within the 2s p95 target at production scale (~7,300 chunks / ~11.1M embeddings)")
    func shutdownMeetsLatencyTarget() async throws {
        let dims = 768
        let chunkCount = 7_300
        let tokensPerChunk = 1_521 // 7,300 × 1,521 ≈ 11.1M, matching ADR 034's observed corpus.
        let genCount = 20

        let storage = InMemoryStorage()
        try await storage.open()

        print("[ShutdownLatencyBenchmark] Synthesizing fixture: \(chunkCount) chunks × \(tokensPerChunk) tokens, dims=\(dims) …")
        try await Self.synthesizeSnapshotFixture(
            storage: storage, chunkCount: chunkCount, tokensPerChunk: tokensPerChunk,
            dims: dims, genCount: genCount
        )
        let totalEmbeddings = chunkCount * tokensPerChunk
        print("[ShutdownLatencyBenchmark] Fixture ready: \(totalEmbeddings) embeddings across \(genCount) generations.")

        let store = try await SwitchcraftStore(storage: storage, embedder: MockEmbedder(dims: dims))
        #expect(await store.indexerUsedSnapshotFastPath == true,
                "fixture construction must produce a valid fast-path snapshot, or this benchmark measures the full rehydration walk instead of persistSnapshot()")

        let start = Date()
        try await store.shutdown()
        let elapsed = Date().timeIntervalSince(start)

        print("[ShutdownLatencyBenchmark] shutdown() completed in \(String(format: "%.3f", elapsed))s (target < \(Self.shutdownCeiling)s)")
        #expect(elapsed < Self.shutdownCeiling,
                "shutdown() should complete within \(Self.shutdownCeiling)s at production scale; measured \(elapsed)s")
    }
}
