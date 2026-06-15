// Performance regression suite for SwitchcraftStore.
//
// These tests are floor guards, not production targets. They exist to
// catch obvious regressions in search latency, indexing throughput, and
// resident memory before Phase 2 optimisations (Metal kernels, parallel
// bucket search) silently degrade them. The thresholds are deliberately
// 2–4× the observed release-mode numbers on a current dev box so they
// remain stable on the macos-15 GitHub Actions runner (~7 GB RAM, 3
// cores). See `adrs/012-performance-regression-thresholds.md` for the
// rationale, the recalibration of the spec's original 50 / 100 ms
// latency numbers, and the policy for tightening once CI history
// accrues.
//
// Run in release mode for representative numbers:
//
//     swift test -c release --filter PerformanceTests
//
// Debug builds pay for stdlib bounds checks, ARC traffic, and unoptimised
// inlining, which makes the latency floors flap on shared CI runners. The
// suite is therefore **gated to release builds only** via an `assert`-based
// runtime detection (see `isReleaseBuild`); under `swift test` (debug) the
// three tests are skipped with a clear reason. The release-mode CI job in
// `.github/workflows/ci.yml` is the authoritative perf gate.
//
// ## Corpus tuning
//
// The shared fixture is 5,000 documents × ≤ 50 whitespace-separated
// tokens × 32-dim float embeddings via `MockEmbedder(dims: 32)`. Worst
// case in-memory float ledger:
//
//     5_000 docs × 50 tokens × 32 dims × 4 bytes = 32_000_000 bytes ≈ 32 MB
//
// That sits well under the 300 MB peak-RSS ceiling even after per-doc
// `DocumentRecord` / `ChunkRecord` overhead, the BM25 index, and the LSM
// generations produced by `IndexerConfig.production` (l0Capacity: 1024,
// lsmFanout: 16). A `dims: 128` × 700-token corpus would be ~1.8 GB —
// hence the reduced parameters. This matches the `dims: 32` pattern in
// `ConcurrencyTests`.
//
// `InMemoryStorage` is used (not SQLite) so disk I/O does not confound
// the latency or throughput measurements.

import Foundation
import Testing
@testable import Switchcraft
@testable import SwitchcraftCore

import Darwin.Mach

/// `true` when compiled with optimisations (`swift test -c release`),
/// `false` under a debug build. Uses the standard `assert` trick: the
/// closure runs only when assertions are enabled, which the optimiser
/// strips out in release. Used by the perf suite to skip itself in debug
/// builds where measurements are too noisy to defend a floor against.
private let isReleaseBuild: Bool = {
    var debug = false
    assert({ debug = true; return true }())
    return !debug
}()

@Suite("PerformanceTests", .serialized,
       .enabled(if: isReleaseBuild,
                "Performance suite is release-only (run with `swift test -c release`)"))
struct PerformanceTests {

    // MARK: - Fixture

    /// Captured corpus + store handle returned by the shared builder.
    /// `static let buildOnce` initialises this exactly once across the
    /// whole suite so the latency and memory tests don't pay the
    /// 5,000-doc indexing cost more than once.
    fileprivate struct Fixture: Sendable {
        let store: SwitchcraftStore
        let storage: InMemoryStorage
    }

    /// Dimensionality of the mock embeddings. Kept small so the in-memory
    /// ledger fits inside the 300 MB ceiling — see header comment.
    fileprivate static let embedDims: Int = 32

    /// Number of documents in the shared corpus. Matches the spec.
    fileprivate static let corpusSize: Int = 5_000

    /// Lazy, thread-safe single-build of the 5,000-doc corpus. Each test
    /// `await`s the same `Task.value`. Throughput cannot use this fixture
    /// because it must measure the build itself.
    fileprivate static let buildOnce: Task<Fixture, Error> = Task {
        try await buildCorpus(size: Self.corpusSize)
    }

    /// Build a `SwitchcraftStore` backed by `InMemoryStorage`,
    /// `MockEmbedder(dims: 32)`, and `StoreConfig.default`. Adds `size`
    /// documents whose body embeds the loop index so chunk hashes are
    /// unique (otherwise dedup collapses the corpus into a single chunk),
    /// then flushes the LSM tree with `index()` so the first searches
    /// don't pay flush cost.
    fileprivate static func buildCorpus(size: Int) async throws -> Fixture {
        let storage = InMemoryStorage()
        let store = try await SwitchcraftStore(
            storage: storage,
            embedder: MockEmbedder(dims: Self.embedDims),
            config: .default
        )
        for i in 0..<size {
            try await store.add(id: "doc-\(i)", body: Self.body(i))
        }
        try await store.index()
        return Fixture(store: store, storage: storage)
    }

    /// Per-doc body. ~10 whitespace tokens — well under the 50-token cap
    /// the header comment cites — keeping the in-memory float ledger near
    /// the lower end of the bound. The loop index makes each body's
    /// SHA-256 chunk hash unique.
    fileprivate static func body(_ i: Int) -> String {
        "document \(i) about apples and oranges and grapes and pears"
    }

    /// Query word that is present in every doc body so the search engine
    /// returns real BM25 + vector hits (not the empty-result fast path).
    fileprivate static let probeQuery: String = "apples"

    // MARK: - Tests

    @Test("search latency: p50 < 300 ms and p95 < 500 ms over ≥ 50 iterations")
    func searchLatencyP50AndP95UnderFloor() async throws {
        let fixture = try await Self.buildOnce.value
        let store = fixture.store

        // Warm-up: discount lazy initialisation costs (cold caches, first
        // BM25 lookup, etc.) from the timed loop.
        _ = try await store.search(query: Self.probeQuery, topK: 10)

        let iterations = 50
        var samplesNs: [UInt64] = []
        samplesNs.reserveCapacity(iterations)

        let clock = ContinuousClock()
        for _ in 0..<iterations {
            let start = clock.now
            _ = try await store.search(query: Self.probeQuery, topK: 10)
            let elapsed = clock.now - start
            samplesNs.append(Self.nanoseconds(elapsed))
        }

        samplesNs.sort()
        // Nearest-rank percentile, 0-based: idx = clamp(ceil(p * n) - 1, 0, n-1).
        // For n=50: p50 → idx 24 (25th-ranked sample), p95 → idx 47 (48th-ranked).
        let p50ns = samplesNs[Self.percentileIndex(0.50, count: iterations)]
        let p95ns = samplesNs[Self.percentileIndex(0.95, count: iterations)]

        let p50ms = Double(p50ns) / 1_000_000.0
        let p95ms = Double(p95ns) / 1_000_000.0

        // Floor values are deliberately loose (~2–4× the observed
        // release-mode numbers on M1 Ultra: ~126 ms p50 / ~137 ms p95).
        // The spec's original 50 / 100 ms numbers were derived from the
        // Witchcraft Rust target on M2 Max and do not reflect the
        // current Swift port's hybrid-search cost; ADR 012 documents the
        // recalibration. Tighten only after CI history accrues.
        #expect(p50ms < 300.0,
                "search p50 \(p50ms) ms exceeded 300 ms floor")
        #expect(p95ms < 500.0,
                "search p95 \(p95ms) ms exceeded 500 ms floor")
    }

    @Test("indexing throughput: > 10 documents/second for 5,000-doc add + index() cycle")
    func indexingThroughputAbove10DocsPerSecond() async throws {
        // Build a fresh corpus so the throughput measurement is not
        // amortised across the shared fixture. This intentionally
        // duplicates the work in `buildCorpus(size:)` rather than reusing
        // the cached `buildOnce` task.
        let storage = InMemoryStorage()
        let store = try await SwitchcraftStore(
            storage: storage,
            embedder: MockEmbedder(dims: Self.embedDims),
            config: .default
        )

        let clock = ContinuousClock()
        let start = clock.now
        for i in 0..<Self.corpusSize {
            try await store.add(id: "doc-\(i)", body: Self.body(i))
        }
        try await store.index()
        let elapsedNs = Self.nanoseconds(clock.now - start)

        let elapsedSeconds = Double(elapsedNs) / 1_000_000_000.0
        let throughput = Double(Self.corpusSize) / elapsedSeconds

        #expect(throughput > 10.0,
                "indexing throughput \(throughput) docs/s below 10 docs/s floor")
    }

    // MARK: - R14: dedup-hit overhead of chunkBucketRefCount

    /// Measures the per-call cost of the bucket-completeness check introduced by
    /// issue #120 (updated to a count predicate in issue #130).  The check fires
    /// on every `add()` call whose content hash matches an existing indexed chunk
    /// (the common dedup case).
    ///
    /// Methodology: build a 200-doc corpus, flush to buckets, then re-add the
    /// same 200 bodies 5× each (1,000 calls total).  Every re-add hits the
    /// R3 path (`chunkBucketRefCount >= expectedTokenCount`, skip `indexer.add()`).
    ///
    /// Threshold: 100 ms for 1,000 dedup-hits (≤ 100 µs/call) — generous
    /// relative to InMemoryStorage's O(total_indexed_tokens) scan cost.
    @Test("dedup-hit: 1000 re-adds with bucket-presence check complete < 100 ms (release)")
    func dedupHitBucketPresenceCheckOverhead() async throws {
        let dedupCorpusSize = 200
        let repeatCount = 5

        let storage = InMemoryStorage()
        let store = try await SwitchcraftStore(
            storage: storage,
            embedder: MockEmbedder(dims: Self.embedDims),
            config: .default
        )
        for i in 0..<dedupCorpusSize {
            try await store.add(id: "dedup-\(i)", body: Self.body(i))
        }
        try await store.index()

        // Warm-up pass.
        for i in 0..<dedupCorpusSize {
            try await store.add(id: "dedup-\(i)", body: Self.body(i))
        }

        // Timed pass.
        let clock = ContinuousClock()
        let start = clock.now
        for _ in 0..<repeatCount {
            for i in 0..<dedupCorpusSize {
                try await store.add(id: "dedup-\(i)", body: Self.body(i))
            }
        }
        let elapsedNs = Self.nanoseconds(clock.now - start)
        let elapsedMs = Double(elapsedNs) / 1_000_000.0
        let totalCalls = dedupCorpusSize * repeatCount

        #expect(elapsedMs < 100.0,
                "\(totalCalls) dedup-hits took \(elapsedMs) ms; expected < 100 ms")
    }

    @Test("memory peak: resident-size delta < 300 MB during repeated search")
    func memoryPeakUnder300MBDuringSearch() async throws {
        let fixture = try await Self.buildOnce.value
        let store = fixture.store

        // Warm-up: paged-in caches stabilise the baseline reading.
        _ = try await store.search(query: Self.probeQuery, topK: 10)

        let baselineOpt = Self.currentResidentBytes()
        try #require(baselineOpt != nil,
                     "baseline RSS measurement (task_info) failed")
        let baseline = baselineOpt!
        var peak: UInt64 = baseline

        let iterations = 50
        for _ in 0..<iterations {
            _ = try await store.search(query: Self.probeQuery, topK: 10)
            let nowOpt = Self.currentResidentBytes()
            try #require(nowOpt != nil,
                         "RSS measurement (task_info) failed mid-loop")
            let now = nowOpt!
            if now > peak { peak = now }
        }

        let delta = peak >= baseline ? peak - baseline : 0
        let limit: UInt64 = 300 * 1024 * 1024
        let deltaMB = Double(delta) / (1024.0 * 1024.0)

        #expect(delta < limit,
                "peak RSS delta \(deltaMB) MB exceeded 300 MB ceiling")
    }

    // MARK: - Helpers

    /// Total nanoseconds in a `ContinuousClock.Duration`. The duration's
    /// internal representation is `(seconds, attoseconds)`; we read it via
    /// `components` and clamp into `UInt64` (a single test will never run
    /// for hundreds of years).
    fileprivate static func nanoseconds(_ duration: ContinuousClock.Duration) -> UInt64 {
        let parts = duration.components
        let secNs = UInt64(max(parts.seconds, 0)) * 1_000_000_000
        let attoNs = UInt64(max(parts.attoseconds, 0)) / 1_000_000_000
        return secNs &+ attoNs
    }

    /// Nearest-rank percentile index over a 0-based sorted array of `count`
    /// samples. Returns `clamp(ceil(p * n) - 1, 0, n - 1)`. For `n = 50`:
    /// p50 → index 24 (25th-ranked), p95 → index 47 (48th-ranked). Defined
    /// once so the same definition holds if `iterations` ever changes.
    fileprivate static func percentileIndex(_ p: Double, count: Int) -> Int {
        precondition(count > 0, "percentileIndex requires count > 0")
        let raw = Int((p * Double(count)).rounded(.up)) - 1
        return min(max(raw, 0), count - 1)
    }

    /// Resident set size in bytes via `task_info(MACH_TASK_BASIC_INFO)`.
    /// Returns `nil` on `task_info` failure so the caller can fail the
    /// test explicitly with `#require`, rather than silently treating a
    /// failed measurement as a passing 0-byte reading. Darwin-only; the
    /// package targets Apple platforms exclusively (no `#if canImport`
    /// guard needed).
    fileprivate static func currentResidentBytes() -> UInt64? {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(
            MemoryLayout<mach_task_basic_info>.size / MemoryLayout<natural_t>.size
        )
        let result: kern_return_t = withUnsafeMutablePointer(to: &info) { ptr in
            ptr.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { rebound in
                task_info(
                    mach_task_self_,
                    task_flavor_t(MACH_TASK_BASIC_INFO),
                    rebound,
                    &count
                )
            }
        }
        guard result == KERN_SUCCESS else { return nil }
        return UInt64(info.resident_size)
    }
}
