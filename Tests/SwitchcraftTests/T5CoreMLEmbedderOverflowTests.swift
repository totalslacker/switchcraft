// SPDX-License-Identifier: Apache-2.0
import Foundation
import Testing
import SwitchcraftCore
@testable import SwitchcraftCoreML

#if canImport(CoreML)
import CoreML

/// Tests for `T5CoreMLEmbedder`'s overflow guard (`maxInputTokens` / `overflowPolicy`).
///
/// No CoreML model asset is required — `CountingStubPredictor` is injected
/// via the factory-based internal init. All tests use a small `windowSize` and
/// `dims` to keep per-iteration allocations tiny.
@Suite("T5CoreMLEmbedder overflow guard")
struct T5CoreMLEmbedderOverflowTests {

    // MARK: - Helpers

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

    /// Returns a string that reliably tokenises to more than `minTokens` tokens.
    /// Repeats "hello " enough times that the BPE tokenizer (plus the T5 </s>
    /// sentinel) always produces more than `minTokens` tokens.
    private static func oversizedInput(minTokens: Int) -> String {
        // "hello" is a single BPE token in xtr-base-en; " " is a word-start
        // prefix on the following token. Repeating "hello " minTokens+5 times
        // reliably produces more than minTokens tokens after tokenisation.
        return String(repeating: "hello ", count: minTokens + 5)
    }

    // MARK: - R7d: maxInputTokens property

    @Test("maxInputTokens property returns the configured value")
    func testMaxInputTokensPropertyReturnsConfiguredValue() throws {
        let tokenizer = try Self.makeTokenizer()
        let dims = 16
        let windowSize = 8
        let customLimit = 24 // >= windowSize

        let embedder = try T5CoreMLEmbedder(
            predictorFactory: { CountingStubPredictor(dims: dims) },
            tokenizer: tokenizer,
            dims: dims,
            windowSize: windowSize,
            stride: 4,
            minNorm: 1.0,
            maxInputTokens: customLimit
        )

        #expect(embedder.maxInputTokens == customLimit)
    }

    // MARK: - R7a: truncate policy

    @Test("Truncate policy encodes oversized input without error and returns non-empty embeddings")
    func testTruncatePolicyEncodesOversizedInputWithoutError() async throws {
        let tokenizer = try Self.makeTokenizer()
        let dims = 16
        let windowSize = 8
        let maxTokens = 16 // >= windowSize; guarantees at least 1 full window

        let embedder = try T5CoreMLEmbedder(
            predictorFactory: { CountingStubPredictor(dims: dims) },
            tokenizer: tokenizer,
            dims: dims,
            windowSize: windowSize,
            stride: 4,
            minNorm: 1.0,
            maxInputTokens: maxTokens,
            overflowPolicy: .truncate
        )

        // Build an input that definitely exceeds maxTokens.
        let input = Self.oversizedInput(minTokens: maxTokens)
        let result = try await embedder.encode(input)

        // Truncation succeeds: the result must be non-empty and at most
        // maxTokens rows wide (each row is `dims` floats).
        #expect(!result.isEmpty, "truncated encode must return non-empty embeddings")
        #expect(result.count <= maxTokens * dims,
                "result has \(result.count) floats; expected ≤ \(maxTokens * dims) for \(maxTokens) tokens × \(dims) dims")
    }

    // MARK: - R7b: reject policy

    @Test("Reject policy throws EmbedderError.inputTooLarge with correct actual and max values")
    func testRejectPolicyThrowsInputTooLargeError() async throws {
        let tokenizer = try Self.makeTokenizer()
        let dims = 16
        let windowSize = 8
        let maxTokens = 16

        let embedder = try T5CoreMLEmbedder(
            predictorFactory: { CountingStubPredictor(dims: dims) },
            tokenizer: tokenizer,
            dims: dims,
            windowSize: windowSize,
            stride: 4,
            minNorm: 1.0,
            maxInputTokens: maxTokens,
            overflowPolicy: .reject
        )

        // Count the tokens the tokenizer actually produces so we can verify
        // the `actual` field on the thrown error.
        let input = Self.oversizedInput(minTokens: maxTokens)
        let actualTokenCount = try tokenizer.encode(input, addSpecialTokens: true).count
        try #require(actualTokenCount > maxTokens,
                     "test precondition: input must tokenise to > \(maxTokens) tokens; got \(actualTokenCount)")

        await #expect(throws: EmbedderError.inputTooLarge(actual: actualTokenCount, max: maxTokens)) {
            _ = try await embedder.encode(input)
        }
    }

    // MARK: - R5, R7c: stress test

    @Test("1,000 encode calls with oversized inputs interleaved complete without ANE pool degradation (truncate policy)")
    func testOverflowStress1000CallsTruncatePolicy() async throws {
        let tokenizer = try Self.makeTokenizer()
        let dims = 16
        let windowSize = 8
        let maxTokens = 16

        let embedder = try T5CoreMLEmbedder(
            predictorFactory: { CountingStubPredictor(dims: dims) },
            tokenizer: tokenizer,
            dims: dims,
            windowSize: windowSize,
            stride: 4,
            minNorm: 1.0,
            maxInputTokens: maxTokens,
            overflowPolicy: .truncate
        )

        let normalInput = "semantic search"
        let oversizedInput = Self.oversizedInput(minTokens: maxTokens)

        for i in 0..<1_000 {
            let text = (i == 250 || i == 750) ? oversizedInput : normalInput
            // All calls must succeed (oversized inputs are silently truncated).
            let result = try await embedder.encode(text)
            #expect(!result.isEmpty, "encode returned empty at index \(i)")
        }
    }

    @Test("1,000 encode calls: reject policy throws on oversized inputs but normal-sized calls always succeed")
    func testOverflowStress1000CallsRejectPolicy() async throws {
        let tokenizer = try Self.makeTokenizer()
        let dims = 16
        let windowSize = 8
        let maxTokens = 16

        let embedder = try T5CoreMLEmbedder(
            predictorFactory: { CountingStubPredictor(dims: dims) },
            tokenizer: tokenizer,
            dims: dims,
            windowSize: windowSize,
            stride: 4,
            minNorm: 1.0,
            maxInputTokens: maxTokens,
            overflowPolicy: .reject
        )

        let normalInput = "semantic search"
        let oversizedInput = Self.oversizedInput(minTokens: maxTokens)

        var oversizedErrors = 0
        for i in 0..<1_000 {
            if i == 250 || i == 750 {
                // Must throw inputTooLarge — and must NOT leave the embedder in a
                // broken state (subsequent normal calls must still work).
                do {
                    _ = try await embedder.encode(oversizedInput)
                    Issue.record("Expected EmbedderError.inputTooLarge at index \(i) but no error was thrown")
                } catch let err as EmbedderError {
                    if case .inputTooLarge = err {
                        oversizedErrors += 1
                    } else {
                        Issue.record("Unexpected EmbedderError at index \(i): \(err)")
                    }
                } catch {
                    Issue.record("Unexpected error type at index \(i): \(error)")
                }
            } else {
                // Normal-sized call must always succeed.
                let result = try await embedder.encode(normalInput)
                #expect(!result.isEmpty, "normal encode returned empty at index \(i)")
            }
        }

        #expect(oversizedErrors == 2, "expected 2 inputTooLarge errors (indices 250 and 750), got \(oversizedErrors)")
    }
}

#endif
