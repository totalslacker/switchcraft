// SPDX-License-Identifier: Apache-2.0
import Foundation
import Testing
import SwitchcraftCore
@testable import SwitchcraftCoreML

#if canImport(CoreML)
import CoreML

/// Stress and lifecycle tests for `T5CoreMLEmbedder`'s ANE IOSurface pool
/// exhaustion mitigation: autoreleasepool discipline, proactive model reload,
/// and reactive CPU fallback.
///
/// No CoreML model asset is required — `CountingStubPredictor` is injected
/// via the factory-based internal init.
@Suite("T5CoreMLEmbedder stress and reload")
struct T5CoreMLEmbedderStressTests {

    private static func makeTokenizer() throws -> Tokenizer {
        let url = try #require(
            Bundle.module.url(
                forResource: "xtr-base-en.tokenizer",
                withExtension: "json",
                subdirectory: "Fixtures"
            ),
            "tokenizer fixture missing from test bundle"
        )
        return try Tokenizer(contentsOf: url.path)
    }

    // MARK: - Stress test

    /// 2,000 encode calls with varied input sizes must all succeed without error.
    ///
    /// Uses a fast `CountingStubPredictor` (no delay, no failures). The stub seam
    /// validates the call-counter / reload-trigger / actor-isolation mechanism.
    /// Inputs are kept short (≤ ~15 tokens) so tokenizer overhead stays sub-ms and
    /// the 2,000-iteration loop completes in a few seconds in CI.
    ///
    /// The spec target is 5,000 iterations; 2,000 is used here because the BPE
    /// tokenizer (not the stub predictor) dominates per-call cost and long inputs
    /// push total CI time into minutes. The real IOSurface leak validation lives in
    /// the asset-gated ≥10k test in `T5CoreMLEmbedderTests`, which correctly tests
    /// against the actual CoreML allocator.
    @Test("2,000 encode calls with varied short inputs succeed without error")
    func testStress2000IterationsVariedInputSizes() async throws {
        let tokenizer = try Self.makeTokenizer()

        // Small dims/windowSize to keep per-iteration MLMultiArray allocations tiny
        // (64×16 = 1 KB per array vs. 512×128 = 256 KB). The test validates reload
        // cadence and absence of errors, not embedding quality.
        let dims = 16
        let embedder = try T5CoreMLEmbedder(
            predictorFactory: { CountingStubPredictor(dims: dims) },
            tokenizer: tokenizer,
            dims: dims,
            windowSize: 64,
            stride: 32,
            minNorm: 1.0,
            reloadInterval: 500
        )

        // Four short, varied inputs (all ≤ ~15 tokens) so tokenization is fast.
        // Different lengths still exercise the sliding-window planner distinctly.
        let inputs: [String] = [
            "cat",
            "semantic search",
            "neural information retrieval system",
            "the quick brown fox jumps over the lazy dog",
        ]

        for i in 0..<2_000 {
            let text = inputs[i % inputs.count]
            let result = try await embedder.encode(text)
            if text.count > 3 {
                #expect(!result.isEmpty, "encode returned empty for input index \(i)")
            }
        }
    }

    // MARK: - Reload counter test

    /// After 31 encode calls with `reloadInterval: 10`, the factory must have
    /// been called at init + at encode calls 10, 20, and 30 (≥ 4 times total).
    @Test("Proactive reload triggers predictorFactory at each reloadInterval boundary")
    func testReloadTriggeredAtInterval() async throws {
        let tokenizer = try Self.makeTokenizer()

        let counter = FactoryCallCounter()
        let dims = 16
        let embedder = try T5CoreMLEmbedder(
            predictorFactory: {
                counter.increment()
                return CountingStubPredictor(dims: dims)
            },
            tokenizer: tokenizer,
            dims: dims,
            windowSize: 64,
            stride: 32,
            minNorm: 1.0,
            reloadInterval: 10
        )

        // 1 factory call happened at init already.
        #expect(counter.count == 1, "factory should be called once at init, got \(counter.count)")

        for _ in 0..<31 {
            _ = try await embedder.encode("reload interval test")
        }

        // Reloads fire at encode calls 10, 20, 30 → 3 additional factory calls.
        #expect(counter.count >= 4, "expected ≥4 factory calls (1 init + 3 reloads), got \(counter.count)")
    }

    // MARK: - IOSurface fallback test

    /// When every prediction raises an IOSurface-like exception, the CPU fallback
    /// must succeed and log a `"recovered_iosurface_exhaustion"` JSONL row with
    /// `"category": "warning"` for each encode call.
    @Test("IOSurface exhaustion triggers CPU fallback and logs recovery row")
    func testIOSurfaceFallbackLogsRecovery() async throws {
        let tokenizer = try Self.makeTokenizer()

        let logURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("switchcraft-stress-recovery-\(UUID().uuidString).jsonl")
        defer { try? FileManager.default.removeItem(at: logURL) }

        // Main predictor: always raises IOSurface-like exception.
        // CPU fallback predictor: always succeeds.
        let dims = 16
        let embedder = try T5CoreMLEmbedder(
            predictorFactory: { CountingStubPredictor(failInterval: 1, dims: dims) },
            cpuPredictorFactory: { CountingStubPredictor(failInterval: nil, dims: dims) },
            tokenizer: tokenizer,
            dims: dims,
            windowSize: 64,
            stride: 32,
            minNorm: 1.0,
            failureLogURL: logURL,
            reloadInterval: 500
        )

        // "test input" (10 chars, 1 window) → each encode = 1 IOSurface hit → 1 recovery row.
        let inputText = "test input"
        let encodeCount = 10

        for _ in 0..<encodeCount {
            // Must NOT throw — the CPU fallback should recover.
            let result = try await embedder.encode(inputText)
            #expect(!result.isEmpty, "CPU-fallback encode should return non-empty embeddings")
        }

        // Verify recovery rows were written.
        let logData = try Data(contentsOf: logURL)
        let rawText = try #require(String(data: logData, encoding: .utf8), "log not UTF-8")
        let lines = rawText.split(separator: "\n", omittingEmptySubsequences: true)
        #expect(lines.count == encodeCount,
                "expected \(encodeCount) recovery rows, got \(lines.count)")

        for (idx, line) in lines.enumerated() {
            let rowData = Data(line.utf8)
            let json = try #require(
                try JSONSerialization.jsonObject(with: rowData) as? [String: Any],
                "row \(idx) is not a JSON object"
            )
            #expect((json["name"]     as? String) == "recovered_iosurface_exhaustion",
                    "row \(idx): unexpected name")
            #expect((json["category"] as? String) == "warning",
                    "row \(idx): category must be \"warning\"")
        }
    }
}

#endif
