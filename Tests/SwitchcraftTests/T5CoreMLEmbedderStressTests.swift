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

    /// 5,000 encode calls with varied input sizes must all succeed without error.
    ///
    /// Uses a fast `CountingStubPredictor` (no delay, no failures). The varied
    /// sizes — single word, short phrase, medium paragraph, long passage — exercise
    /// the same allocation pressure pattern that triggered the real IOSurface
    /// exhaustion, where uniform-size loops can mask leaks.
    @Test("5,000 encode calls with varied input sizes succeed without error")
    func testStress5000IterationsVariedInputSizes() async throws {
        let tokenizer = try Self.makeTokenizer()

        let embedder = try T5CoreMLEmbedder(
            predictorFactory: { CountingStubPredictor() },
            tokenizer: tokenizer,
            dims: 128,
            windowSize: 512,
            stride: 256,
            minNorm: 1.0,
            reloadInterval: 500
        )

        // Four representative input sizes (approximate token counts).
        let inputs: [String] = [
            "cat",                                           // ~1 token
            String(repeating: "semantic search query ", count: 10),  // ~50 tokens
            String(repeating: "the quick brown fox jumps over the lazy dog ", count: 30), // ~300 tokens
            String(repeating: "neural information retrieval with token-level embeddings ", count: 60), // ~600 tokens
        ]

        for i in 0..<5_000 {
            let text = inputs[i % inputs.count]
            let result = try await embedder.encode(text)
            // Non-trivial inputs must produce a non-empty embedding.
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
        let embedder = try T5CoreMLEmbedder(
            predictorFactory: {
                counter.increment()
                return CountingStubPredictor()
            },
            tokenizer: tokenizer,
            dims: 128,
            windowSize: 512,
            stride: 256,
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
        let embedder = try T5CoreMLEmbedder(
            predictorFactory: { CountingStubPredictor(failInterval: 1) },
            cpuPredictorFactory: { CountingStubPredictor(failInterval: nil) },
            tokenizer: tokenizer,
            dims: 128,
            windowSize: 512,
            stride: 256,
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
