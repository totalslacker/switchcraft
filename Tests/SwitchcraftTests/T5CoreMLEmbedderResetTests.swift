// SPDX-License-Identifier: Apache-2.0
import Foundation
import Testing
import SwitchcraftCore
@testable import SwitchcraftCoreML

#if canImport(CoreML)
import CoreML

/// Tests for `T5CoreMLEmbedder.resetState()`.
///
/// These tests do NOT require the CoreML model asset — stub predictors are
/// injected via the internal factory-based init. The tokenizer fixture is
/// always present in the test bundle.
@Suite("T5CoreMLEmbedder resetState()")
struct T5CoreMLEmbedderResetTests {

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

    // MARK: - Determinism test

    /// `encode(s)` must return byte-identical output before and after
    /// `resetState()` for the same input string `s`. The factory must be
    /// called exactly once more (for the reset reload).
    @Test("encode output is byte-identical before and after resetState()")
    func resetStateIsTransparent() async throws {
        let tokenizer = try Self.makeTokenizer()
        let dims = 16
        let counter = FactoryCallCounter()
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
            reloadInterval: 500
        )

        let countAfterInit = counter.count
        #expect(countAfterInit == 1, "factory should be called once at init, got \(countAfterInit)")

        let input = "determinism test for resetState"
        let resultA = try await embedder.encode(input)
        #expect(!resultA.isEmpty, "encode before reset should return embeddings")

        try await embedder.resetState()

        // Factory must have been called exactly once more for the reload.
        #expect(counter.count == countAfterInit + 1,
                "resetState() should call predictorFactory once, got \(counter.count - countAfterInit) calls")

        let resultB = try await embedder.encode(input)
        #expect(!resultB.isEmpty, "encode after reset should return embeddings")

        // CountingStubPredictor returns deterministic output (1/√dims per element),
        // so results must be element-wise identical.
        #expect(resultA.count == resultB.count,
                "result count changed: \(resultA.count) → \(resultB.count)")
        #expect(resultA == resultB, "encode output must be byte-identical after resetState()")
    }

    // MARK: - Factory-failure test

    /// When the factory throws during `resetState()`, the error must propagate
    /// and the inFlight guard must drain correctly so subsequent encodes can proceed
    /// against the retained (stale) predictor.
    ///
    /// Factory call sequence:
    ///   1 (init) → succeeds, returns stub predictor
    ///   2 (resetState) → throws `intentionalFactoryFailure`
    /// After the failed reset, the stale stub predictor is still held and encode
    /// must succeed.
    @Test("resetState() propagates factory errors and drains the inFlight guard")
    func resetStateFactoryFailurePropagates() async throws {
        let tokenizer = try Self.makeTokenizer()
        let dims = 16
        let counter = FactoryCallCounter()

        let embedder = try T5CoreMLEmbedder(
            predictorFactory: {
                counter.increment()
                // Call 1 (init): succeed. Call 2 (resetState): fail.
                if counter.count > 1 {
                    throw TestError.intentionalFactoryFailure
                }
                return CountingStubPredictor(dims: dims)
            },
            tokenizer: tokenizer,
            dims: dims,
            windowSize: 64,
            stride: 32,
            minNorm: 1.0,
            reloadInterval: 500
        )

        // Pre-reset encode succeeds.
        let before = try await embedder.encode("before reset")
        #expect(!before.isEmpty)

        // resetState() must throw when the factory fails.
        do {
            try await embedder.resetState()
            Issue.record("Expected resetState() to throw but it succeeded")
        } catch TestError.intentionalFactoryFailure {
            // Expected — error propagated correctly.
        }

        // After the failed reset, the embedder's inFlight guard must be cleared
        // so subsequent encodes can proceed against the stale (original) predictor.
        let after = try await embedder.encode("after failed reset")
        #expect(!after.isEmpty, "encode must succeed after a failed resetState() drains the guard")
    }

    // MARK: - Re-entrancy test

    /// `resetState()` called concurrently with an in-flight `encode()` call
    /// must wait for the encode to complete, then reset, and allow subsequent
    /// encodes to succeed. No deadlock.
    @Test("resetState() serialises correctly with concurrent in-flight encode()")
    func resetStateSerialisesConcurrently() async throws {
        let tokenizer = try Self.makeTokenizer()
        let dims = 16
        let counter = FactoryCallCounter()

        // Use a slow predictor so the encode is still in-flight when resetState() arrives.
        let embedder = try T5CoreMLEmbedder(
            predictorFactory: {
                counter.increment()
                // First call (init): slow predictor to make concurrent test reliable.
                // Subsequent calls (reset): fast predictor.
                return counter.count == 1
                    ? SlowStubPredictor(delay: .milliseconds(150), dims: dims)
                    : CountingStubPredictor(dims: dims)
            },
            tokenizer: tokenizer,
            dims: dims,
            windowSize: 64,
            stride: 32,
            minNorm: 1.0,
            reloadInterval: 500
        )

        // Launch encode and resetState() concurrently. The encode starts first
        // (the slow predictor ensures it is in-flight when resetState() arrives).
        async let encodeResult = embedder.encode("concurrent encode during reset")
        async let resetResult: () = embedder.resetState()

        // Both must complete without deadlock or error.
        let result = try await encodeResult
        try await resetResult

        #expect(!result.isEmpty, "concurrent encode should return embeddings")

        // Factory should have been called: 1 (init) + 1 (reset) = 2.
        #expect(counter.count == 2,
                "factory should be called once at init and once at reset, got \(counter.count)")

        // After the reset, a fresh encode must also succeed.
        let postReset = try await embedder.encode("encode after reset")
        #expect(!postReset.isEmpty, "encode after reset should return embeddings")
    }

    // MARK: - callCount reset test

    /// After `resetState()`, `callCount` is reset to 0, so the Layer 2
    /// proactive reload fires after `reloadInterval` more encodes (not sooner).
    @Test("resetState() resets the callCount proactive-reload counter to 0")
    func resetStateClearsCallCount() async throws {
        let tokenizer = try Self.makeTokenizer()
        let dims = 16
        let counter = FactoryCallCounter()
        let reloadInterval = 5

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
            reloadInterval: reloadInterval
        )

        let afterInit = counter.count  // 1
        #expect(afterInit == 1)

        // Encode 3 times (< reloadInterval) — no proactive reload expected.
        for _ in 0..<3 {
            _ = try await embedder.encode("call count test")
        }
        #expect(counter.count == 1, "no proactive reload expected for 3 encodes with interval=5")

        // Explicit reset: callCount goes to 0, factory called once.
        try await embedder.resetState()
        #expect(counter.count == 2, "resetState() should call factory once")

        // Encode 4 more times. callCount is now 0+4=4, still below reloadInterval=5.
        for _ in 0..<4 {
            _ = try await embedder.encode("post-reset call count test")
        }
        #expect(counter.count == 2, "no proactive reload expected for 4 encodes after reset (callCount=4 < 5)")

        // One more encode → callCount=5 → proactive reload fires.
        _ = try await embedder.encode("trigger reload")
        #expect(counter.count == 3, "proactive reload should fire at encode #5 after reset")
    }
}

private enum TestError: Error {
    case intentionalFactoryFailure
}

#endif
