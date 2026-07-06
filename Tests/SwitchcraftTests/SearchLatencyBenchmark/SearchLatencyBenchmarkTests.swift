// SPDX-License-Identifier: Apache-2.0
//
// Search-latency benchmark suite (issue #140 / ADR 035).
//
// Two gates, both against the same realistic-scale synthetic fixture
// (see `SearchLatencyFixture.swift`):
//
//  1. Ratio: `searchOptimized` (default, parallel decode + query-token
//     dedup) must be >= 3x faster than `searchLegacy` (the frozen
//     pre-#140 sequential baseline, reachable via
//     `SearchConfig.legacySequentialSearch`), measured in the same test
//     invocation against the same fixture so hardware/run variance
//     cancels out. This is a ratio assertion, not an absolute-time
//     threshold — it holds across Debug and Release builds even though
//     absolute times differ ~10-100x between them (Specify-stage Q2
//     decision).
//  2. Sustained: 100 distinct queries back-to-back must show no
//     throughput degradation, defined as `lastQuartileP95 <=
//     firstQuartileMedian * 2.0` (Specify-stage Q3 decision).
//
// Both are opt-in (`SWITCHCRAFT_SEARCH_LATENCY_BENCHMARK=1`) since fixture
// construction is heavy (hundreds of thousands of 768-dim embeddings,
// Q4-encoded) — matching the existing `CompactionMemoryRegressionTests`
// convention for large synthetic fixtures. Unlike that suite, this one is
// NOT release-gated (`PerformanceTests`' convention) — the ratio
// assertion is specifically designed to hold in both Debug and Release.
//
// Run:
//
//     SWITCHCRAFT_SEARCH_LATENCY_BENCHMARK=1 swift test --filter SearchLatencyBenchmark
//     SWITCHCRAFT_SEARCH_LATENCY_BENCHMARK=1 swift test -c release --filter SearchLatencyBenchmark

import Foundation
import Testing
@testable import SwitchcraftCore

private let searchLatencyBenchmarkEnabled =
    ProcessInfo.processInfo.environment["SWITCHCRAFT_SEARCH_LATENCY_BENCHMARK"] != nil

// `.serialized`: the ratio test and the sustained-workload test are both
// latency-sensitive and CPU-bound (parallel decode competes for cores).
// Swift Testing runs `@Test`s in a suite concurrently by default, which
// would let the sustained test's 100 back-to-back searches contend for
// CPU with the ratio test's timed measurements running at the same
// moment — undermining the "same run, same hardware" comparison the
// ratio assertion depends on. Serializing matches `PerformanceTests`'
// existing convention for the same reason.
@Suite("SearchLatencyBenchmark", .serialized,
       .enabled(if: searchLatencyBenchmarkEnabled,
                "Set SWITCHCRAFT_SEARCH_LATENCY_BENCHMARK=1 to run the realistic-scale search-latency benchmark"))
struct SearchLatencyBenchmarkTests {

    /// Minimum required `preElapsed / postElapsed` speedup. Parallel
    /// bucket decode + query-token dedup are both provably output-
    /// equivalent optimisations expected to comfortably clear 3x per the
    /// issue's own guidance; see ADR 035 for the measured ratio.
    static let minimumSpeedupRatio: Double = 3.0

    /// Sustained-workload throughput-degradation tolerance (Specify-stage
    /// Q3 decision). Tighten to 1.5 if real degradation ever slips
    /// through; loosen to 2.5 if this proves flaky under thermal or
    /// background-load noise.
    static let throughputDegradationTolerance: Double = 2.0

    /// One-time fixture build, shared across both tests in this suite
    /// (mirrors `PerformanceTests.buildOnce`).
    static let sharedFixture: Task<SearchLatencyFixture.Built, Error> = Task {
        try await SearchLatencyFixture.build()
    }

    // MARK: - 1. Ratio benchmark

    @Test("ratio: searchOptimized is >= 3x faster than searchLegacy on the same fixture (Debug+Release)")
    func optimizedIsAtLeastMinimumRatioFasterThanLegacy() async throws {
        let built = try await Self.sharedFixture.value
        print("""
        [SearchLatencyBenchmark] fixture: \(built.totalChunks) chunks, \
        \(built.totalEmbeddings) embeddings, dims=\(SearchLatencyFixture.dims), \
        bucket counts by generation=\(built.bucketCountsByGeneration)
        """)

        let query = SearchLatencyFixture.makeQuery(dims: SearchLatencyFixture.dims, seed: 999)

        let legacyEngine = SearchEngine(
            storage: built.storage, config: SearchConfig(legacySequentialSearch: true)
        )
        let optimizedEngine = SearchEngine(
            storage: built.storage, config: SearchConfig()
        )

        // Warm-up each path once so first-call one-time costs (lazy
        // allocations, page faults on the fixture's backing arrays)
        // don't pollute the timed measurement.
        _ = try await legacyEngine.search(queryEmbeddings: query, dims: SearchLatencyFixture.dims, topK: 10)
        _ = try await optimizedEngine.search(queryEmbeddings: query, dims: SearchLatencyFixture.dims, topK: 10)

        let clock = ContinuousClock()

        let preStart = clock.now
        let legacyHits = try await legacyEngine.search(queryEmbeddings: query, dims: SearchLatencyFixture.dims, topK: 10)
        let preElapsed = Self.seconds(clock.now - preStart)

        let postStart = clock.now
        let optimizedHits = try await optimizedEngine.search(queryEmbeddings: query, dims: SearchLatencyFixture.dims, topK: 10)
        let postElapsed = Self.seconds(clock.now - postStart)

        let ratio = preElapsed / postElapsed
        print("[SearchLatencyBenchmark] pre-refactor (searchLegacy):   \(preElapsed)s")
        print("[SearchLatencyBenchmark] post-refactor (searchOptimized): \(postElapsed)s (\(ratio)x)")
        print("""
        [SearchLatencyBenchmark] downstream consumer's own <2.5s target \
        (reported for visibility, not asserted — see issue #140): \
        pre=\(preElapsed < 2.5 ? "under" : "OVER"), post=\(postElapsed < 2.5 ? "under" : "OVER")
        """)

        #expect(!legacyHits.isEmpty, "fixture/query produced no hits from searchLegacy — check fixture construction")
        #expect(legacyHits.map(\.uuid) == optimizedHits.map(\.uuid),
                "legacy/optimized diverged in ranked document order on the benchmark fixture")

        #expect(postElapsed < preElapsed / Self.minimumSpeedupRatio, """
            Expected >= \(Self.minimumSpeedupRatio)x speedup; \
            pre=\(preElapsed)s post=\(postElapsed)s (\(ratio)x)
            """)
    }

    // MARK: - 2. Sustained workload

    @Test("sustained: 100 distinct queries back-to-back show no throughput degradation")
    func sustainedHundredQueriesNoThroughputDegradation() async throws {
        let built = try await Self.sharedFixture.value
        // Direct SearchEngine calls over a hand-built fixture never touch
        // Indexer/rehydration, so `ledgerOutOfSync` (an Indexer-specific
        // error) cannot occur here by construction — the issue's "no
        // ledgerOutOfSync" requirement is trivially satisfied.
        let engine = SearchEngine(storage: built.storage)

        var rng = SplitMix64(seed: 555)
        var timings: [Double] = []
        timings.reserveCapacity(100)
        let clock = ContinuousClock()

        for i in 0..<100 {
            let query = SearchLatencyFixture.makeQuery(dims: SearchLatencyFixture.dims, rng: &rng)
            let start = clock.now
            let hits = try await engine.search(queryEmbeddings: query, dims: SearchLatencyFixture.dims, topK: 10)
            timings.append(Self.seconds(clock.now - start))
            #expect(!hits.isEmpty, "query \(i) of the sustained run produced no hits")
        }

        print("[SearchLatencyBenchmark] sustained 100-query timing series (s): \(timings)")

        let firstQuartile = Array(timings.prefix(25))
        let lastQuartile = Array(timings.suffix(25))
        let firstMedian = Self.median(firstQuartile)
        let lastP95 = Self.percentile(lastQuartile, 0.95)

        print("""
        [SearchLatencyBenchmark] first-quartile median=\(firstMedian)s, \
        last-quartile p95=\(lastP95)s, ratio=\(lastP95 / firstMedian)
        """)

        #expect(lastP95 <= firstMedian * Self.throughputDegradationTolerance, """
            Throughput degraded across sustained run: \
            first-quartile median=\(firstMedian)s, last-quartile p95=\(lastP95)s \
            (tolerance \(Self.throughputDegradationTolerance)x)
            """)
    }

    // MARK: - Helpers

    static func seconds(_ duration: ContinuousClock.Duration) -> Double {
        let parts = duration.components
        return Double(parts.seconds) + Double(parts.attoseconds) / 1e18
    }

    static func median(_ values: [Double]) -> Double {
        let sorted = values.sorted()
        let n = sorted.count
        precondition(n > 0)
        if n % 2 == 1 { return sorted[n / 2] }
        return (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    }

    /// Nearest-rank percentile, matching `PerformanceTests.percentileIndex`'s
    /// convention: `clamp(ceil(p * n) - 1, 0, n - 1)`.
    static func percentile(_ values: [Double], _ p: Double) -> Double {
        let sorted = values.sorted()
        let n = sorted.count
        precondition(n > 0)
        let idx = min(max(Int((p * Double(n)).rounded(.up)) - 1, 0), n - 1)
        return sorted[idx]
    }
}
