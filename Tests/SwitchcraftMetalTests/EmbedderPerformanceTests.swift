// SPDX-License-Identifier: Apache-2.0
//
// Informational performance measurements for `T5MetalEmbedder`. Issue
// #65 (umbrella #57). Release-only, asset-gated, **no fail floor** —
// the suite reports observed numbers via the ADR 012 percentile harness
// so an operator can compare against the umbrella's informational
// targets (~50 ms / 512-token encode, end-to-end search p95 ≤ 25 ms).
// If observed numbers are far off, that is a signal for a follow-up
// performance issue, not a gate on #65.
//
// Triple-gated: release build + GGUF asset + Metal-capable host. Fresh
// checkouts and CI skip cleanly.

#if canImport(Metal)

import Foundation
import Metal
import Testing
@testable import Switchcraft
@_spi(SwitchcraftMetal) @testable import SwitchcraftCore
@_spi(SwitchcraftMetal) import SwitchcraftMetal

// ASSET-GATED: requires SWITCHCRAFT_XTR_GGUF + release build — skips in CI and on fresh checkouts
@Suite(
    "T5MetalEmbedder Performance (release-only, asset-gated)",
    .serialized,
    .enabled(if: isReleaseBuildMetal
                 && GGUFAsset.isAvailable
                 && MetalAvailability.isAvailable,
             "Release-only suite; also requires SWITCHCRAFT_XTR_GGUF + Metal.")
)
struct EmbedderPerformanceTests {

    // MARK: - Fixed encode input set

    /// Fixed input set for encode-latency measurement. Mix of short and
    /// long inputs so the p50/p95 reflects realistic query/document
    /// lengths the embedder will see in production.
    private static let encodeInputs: [String] = [
        "apples",
        "ascorbic acid",
        "vitamin c reduces cancer risk",
        "high-fat ketogenic diet effects on glucose tolerance",
        "the quick brown fox jumps over the lazy dog",
        "long-term effects of vegetarian diets on cardiovascular outcomes in older adults",
        // Filler prose to exercise multi-window paths.
        String(repeating: "broccoli sprouts contain sulforaphane and protect against cancer ", count: 16),
        String(repeating: "low-fat dietary patterns improve metabolic markers among middle-aged participants ", count: 16),
    ]

    /// Number of timed iterations after warm-up. Matches the `MetalPerf`
    /// percentile harness shape used by `MetalPerformanceTests`.
    private static let encodeIterations: Int = 32

    // MARK: - Encode-latency measurement

    @Test("T5MetalEmbedder.encode p50/p95 over fixed input set")
    func encodeLatencyPercentiles() async throws {
        let modelURL = GGUFAsset.url!
        let tokenizerURL = try #require(
            TokenizerFixture.url,
            "tokenizer fixture missing — Tests/Fixtures/xtr-base-en.tokenizer.json absent"
        )
        let tokenizer = try Tokenizer(contentsOf: tokenizerURL.path)
        let embedder = try await T5MetalEmbedder(
            modelURL: modelURL,
            tokenizer: tokenizer
        )

        // Warm-up — first kernel touch absorbs shader-pipeline binding
        // cost. Every input runs once before timing starts so the
        // p50/p95 numbers reflect steady-state, not first-touch.
        for input in Self.encodeInputs {
            _ = try await embedder.encode(input)
        }

        var samplesNs: [UInt64] = []
        samplesNs.reserveCapacity(Self.encodeIterations * Self.encodeInputs.count)
        let clock = ContinuousClock()
        for _ in 0..<Self.encodeIterations {
            for input in Self.encodeInputs {
                let start = clock.now
                _ = try await embedder.encode(input)
                let elapsed = clock.now - start
                samplesNs.append(MetalPerf.nanoseconds(elapsed))
            }
        }
        samplesNs.sort()
        let count = samplesNs.count
        let p50 = samplesNs[MetalPerf.percentileIndex(0.50, count: count)]
        let p95 = samplesNs[MetalPerf.percentileIndex(0.95, count: count)]

        let p50Ms = Double(p50) / 1_000_000.0
        let p95Ms = Double(p95) / 1_000_000.0
        print("""
              EmbedderPerformanceTests.encode \
              n=\(count), inputs=\(Self.encodeInputs.count) \
              p50=\(p50Ms) ms, p95=\(p95Ms) ms \
              (informational target ~50 ms / 512-token encode)
              """)

        // Sanity assertions only — informational suite per the issue spec.
        #expect(p50 > 0, "encode p50 was 0 ns — the timer didn't fire")
        #expect(p95 >= p50, "encode p95 (\(p95) ns) less than p50 (\(p50) ns)")
    }

    // MARK: - End-to-end search p95 over a 5k-doc corpus

    /// Number of documents indexed through `T5MetalEmbedder` for the
    /// end-to-end p95 measurement. 5,000 per the issue spec; one-time
    /// build cost is multi-minute so this is amortised through
    /// `static let buildOnce`.
    fileprivate static let corpusSize: Int = 5_000

    fileprivate static let probeQuery: String = "apples"

    fileprivate struct Fixture: Sendable {
        let store: SwitchcraftStore
    }

    fileprivate static let buildOnce: Task<Fixture, Error> = Task {
        try await buildCorpus()
    }

    fileprivate static func buildCorpus() async throws -> Fixture {
        let modelURL = GGUFAsset.url!
        let tokenizerURL = try #require(
            TokenizerFixture.url,
            "tokenizer fixture missing — Tests/Fixtures/xtr-base-en.tokenizer.json absent"
        )
        let tokenizer = try Tokenizer(contentsOf: tokenizerURL.path)
        let embedder = try await T5MetalEmbedder(
            modelURL: modelURL,
            tokenizer: tokenizer
        )
        let storage = InMemoryStorage()
        let store = try await SwitchcraftStore(
            storage: storage,
            embedder: embedder,
            config: .default
        )
        for i in 0..<Self.corpusSize {
            try await store.add(id: "doc-\(i)", body: Self.body(i))
        }
        try await store.index()
        return Fixture(store: store)
    }

    fileprivate static func body(_ i: Int) -> String {
        "document \(i) about apples and oranges and grapes and pears"
    }

    @Test("end-to-end search p95 over 5k-doc corpus indexed via T5MetalEmbedder")
    func endToEndSearchPercentiles() async throws {
        let fixture = try await Self.buildOnce.value
        let store = fixture.store

        // Warm-up.
        _ = try await store.search(query: Self.probeQuery, topK: 10)

        let iterations = 50
        var samplesNs: [UInt64] = []
        samplesNs.reserveCapacity(iterations)
        let clock = ContinuousClock()
        for _ in 0..<iterations {
            let start = clock.now
            _ = try await store.search(query: Self.probeQuery, topK: 10)
            let elapsed = clock.now - start
            samplesNs.append(MetalPerf.nanoseconds(elapsed))
        }
        samplesNs.sort()
        let p50 = samplesNs[MetalPerf.percentileIndex(0.50, count: iterations)]
        let p95 = samplesNs[MetalPerf.percentileIndex(0.95, count: iterations)]

        let p50Ms = Double(p50) / 1_000_000.0
        let p95Ms = Double(p95) / 1_000_000.0
        print("""
              EmbedderPerformanceTests.search \
              corpus=\(Self.corpusSize), iters=\(iterations) \
              p50=\(p50Ms) ms, p95=\(p95Ms) ms \
              (informational target end-to-end p95 ≤ 25 ms)
              """)

        // Sanity assertions only — informational suite per the issue spec.
        #expect(p50 > 0, "search p50 was 0 ns — the timer didn't fire")
        #expect(p95 >= p50, "search p95 (\(p95) ns) less than p50 (\(p50) ns)")
    }
}

#endif // canImport(Metal)
