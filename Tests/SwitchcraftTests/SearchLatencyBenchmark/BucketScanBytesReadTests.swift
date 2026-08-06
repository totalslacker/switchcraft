// SPDX-License-Identifier: Apache-2.0
//
// Bucket-scan bytes-read benchmark (issue #151 / ADR 041).
//
// The centroid scan used to materialize every bucket's `indices` and
// `residuals` payload bytes just to read `center` and a token count derived
// from `residuals.count` — 95%+ of the bytes copied out of SQLite were
// discarded unread. `scanBuckets(forGeneration:)` fixes this by reading the
// residual blob's length header (SQL `length(residuals)`) instead of its
// payload.
//
// This test proves the fix at the timing level (no SQLite I/O-byte
// instrumentation exists in this codebase — see Research stage's Technical
// Questions): two on-disk generations with equal bucket counts but very
// different per-bucket residual sizes. `scanBuckets` must stay flat across
// them; `buckets(forGeneration:)` (the full fetch, used as a control proving
// the methodology can actually detect what `scanBuckets` is free of) must
// measurably scale with residual size.
//
// Unlike `SearchLatencyBenchmarkTests`, this suite is NOT gated behind
// `SWITCHCRAFT_SEARCH_LATENCY_BENCHMARK=1` — the fixture is small (a few
// hundred buckets, a few MB of residual bytes total), cheap enough to run on
// every `swift test`. Per the issue's own risk section, this is "the only
// thing that will actually catch a regression of this specific waste going
// forward"; gating it behind an opt-in env var most contributors never set
// would defeat that purpose.

import Foundation
import Testing
import SwitchcraftCore
import SwitchcraftSQLite

@Suite("BucketScanBytesRead", .serialized)
struct BucketScanBytesReadTests {

    /// Buckets per generation. Equal across both generations — only the
    /// per-bucket residual size differs.
    static let bucketCount = 300

    /// Small generation: residual bytes per bucket.
    static let smallResidualBytes = 64

    /// Large generation: residual bytes per bucket — 200x the small
    /// generation's, matching the ~200x order of magnitude the issue's own
    /// measured live-index numbers show between `center` and `residuals`.
    static let largeResidualBytes = smallResidualBytes * 200

    /// Timed trials per measurement, median-reduced to damp scheduler/OS
    /// noise at these small (sub-millisecond to low-millisecond) scales.
    static let trials = 7

    @Test("scanBuckets bytes-read is flat across bucket residual size; full fetch scales (control)")
    func scanBucketsStaysFlatWhileFullFetchScales() async throws {
        let path = NSTemporaryDirectory().appending("switchcraft-scan-bytes-read-\(UUID().uuidString).sqlite3")
        defer { try? FileManager.default.removeItem(atPath: path) }

        let storage = SQLiteStorage(path: path)
        try await storage.open()

        let smallGen = try await Self.makeGeneration(
            storage: storage, residualBytesPerBucket: Self.smallResidualBytes
        )
        let largeGen = try await Self.makeGeneration(
            storage: storage, residualBytesPerBucket: Self.largeResidualBytes
        )

        // Warm up the connection/page cache once per generation so the
        // first timed trial isn't penalized by cold-cache effects that have
        // nothing to do with the byte-count difference under test.
        _ = try await storage.scanBuckets(forGeneration: smallGen)
        _ = try await storage.scanBuckets(forGeneration: largeGen)
        _ = try await storage.buckets(forGeneration: smallGen)
        _ = try await storage.buckets(forGeneration: largeGen)

        let scanSmall = try await Self.medianElapsed(trials: Self.trials) {
            _ = try await storage.scanBuckets(forGeneration: smallGen)
        }
        let scanLarge = try await Self.medianElapsed(trials: Self.trials) {
            _ = try await storage.scanBuckets(forGeneration: largeGen)
        }
        let fetchSmall = try await Self.medianElapsed(trials: Self.trials) {
            _ = try await storage.buckets(forGeneration: smallGen)
        }
        let fetchLarge = try await Self.medianElapsed(trials: Self.trials) {
            _ = try await storage.buckets(forGeneration: largeGen)
        }

        // Guard against a degenerate all-zero measurement (e.g. a clock
        // resolution issue) before dividing.
        let epsilon = 1e-9
        let scanRatio = scanLarge / max(scanSmall, epsilon)
        let fetchRatio = fetchLarge / max(fetchSmall, epsilon)

        print("""
        [BucketScanBytesRead] bucketCount=\(Self.bucketCount), \
        small residual=\(Self.smallResidualBytes)B/bucket, large residual=\(Self.largeResidualBytes)B/bucket
        [BucketScanBytesRead] scanBuckets:            small=\(scanSmall)s large=\(scanLarge)s ratio=\(scanRatio)x
        [BucketScanBytesRead] buckets(forGeneration:): small=\(fetchSmall)s large=\(fetchLarge)s ratio=\(fetchRatio)x
        """)

        // Control: the full fetch must actually scale with residual size —
        // otherwise this test's methodology couldn't detect the thing
        // `scanBuckets` is supposed to be free of, and a pass would be
        // meaningless.
        #expect(fetchRatio > 3.0, """
            Control failed: buckets(forGeneration:) (full fetch) did not scale with residual \
            size (ratio=\(fetchRatio)x) — this benchmark's methodology can't detect the \
            regression it's meant to catch.
            """)

        // The actual assertion: scanBuckets must stay flat (its cost is
        // driven by bucket *count*, not residual *payload size*) while the
        // control clearly scales.
        #expect(scanRatio < fetchRatio, """
            scanBuckets scaled with residual size (scanRatio=\(scanRatio)x) almost as much as \
            the full fetch (fetchRatio=\(fetchRatio)x) — scanBuckets may be reading the \
            residuals/indices payload instead of just the length header.
            """)
        #expect(scanRatio < 3.0, """
            scanBuckets elapsed time scaled by \(scanRatio)x between small- and large-residual \
            generations of equal bucket count — expected it to stay flat.
            """)

        try await storage.close()
    }

    // MARK: - Helpers

    private static func makeGeneration(
        storage: SQLiteStorage, residualBytesPerBucket: Int
    ) async throws -> Int64 {
        let gen = try await storage.insertGeneration(
            GenerationRecord(level: 0, numEmbeddings: 0, minChunkID: 0, maxChunkID: 0, created: Date())
        )
        for i in 0..<bucketCount {
            _ = try await storage.insertBucket(
                BucketRecord(
                    generationID: gen.id,
                    center: Data(repeating: UInt8(i % 256), count: 128 * 4),
                    indices: Data(repeating: UInt8(i % 256), count: residualBytesPerBucket / 2),
                    residuals: Data(repeating: UInt8(i % 256), count: residualBytesPerBucket)
                )
            )
        }
        return gen.id
    }

    private static func medianElapsed(
        trials: Int, _ body: () async throws -> Void
    ) async rethrows -> Double {
        let clock = ContinuousClock()
        var samples: [Double] = []
        samples.reserveCapacity(trials)
        for _ in 0..<trials {
            let start = clock.now
            try await body()
            samples.append(seconds(clock.now - start))
        }
        samples.sort()
        return samples[samples.count / 2]
    }

    private static func seconds(_ duration: ContinuousClock.Duration) -> Double {
        let parts = duration.components
        return Double(parts.seconds) + Double(parts.attoseconds) / 1e18
    }
}
