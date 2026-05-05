// SPDX-License-Identifier: Apache-2.0
import Foundation
import Testing
import SwitchcraftCore
@testable import SwitchcraftCoreML

#if canImport(CoreML)
import CoreML

/// Acceptance tests for the `T5CoreMLEmbedder` re-entrancy guard.
///
/// These tests do NOT require the CoreML model asset — a `SlowStubPredictor`
/// is injected via the internal init. The tokenizer fixture is always present
/// in the test bundle (no env-var gate required).
@Suite("T5CoreMLEmbedder re-entrancy guard")
struct T5CoreMLEmbedderConcurrencyTests {

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

    // MARK: - Serialisation test

    @Test("Two concurrent encode calls are serialised and both succeed")
    func concurrentEncodesAreSerialised() async throws {
        let tokenizer = try Self.makeTokenizer()
        let stub = SlowStubPredictor(delay: .milliseconds(50))
        let embedder = T5CoreMLEmbedder(
            predictor: stub,
            tokenizer: tokenizer,
            dims: 128,
            windowSize: 512,
            stride: 256,
            minNorm: 1.0
        )

        // Launch two encodes concurrently. The re-entrancy guard must serialise
        // them so only one prediction call is in-flight at any time.
        async let resultA = embedder.encode("hello world semantic search")
        async let resultB = embedder.encode("another query for the embedder")

        let (a, b) = try await (resultA, resultB)

        // Both must return non-empty embeddings.
        #expect(!a.isEmpty, "encode A should return embeddings")
        #expect(!b.isEmpty, "encode B should return embeddings")

        // The stub must have been called exactly twice.
        let log = stub.callLog
        #expect(log.count == 2, "predictor should have been called exactly twice, got \(log.count)")

        // Verify serialisation: the call that started first must have finished
        // before the second call started.
        let (first, second) = log[0].start <= log[1].start
            ? (log[0], log[1])
            : (log[1], log[0])

        #expect(
            first.end <= second.start,
            "encode calls overlap: first ended \(first.end), second started \(second.start)"
        )
    }
}

#endif
