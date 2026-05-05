// SPDX-License-Identifier: Apache-2.0
import Foundation
import Testing
import SwitchcraftCore
@testable import SwitchcraftCoreML

#if canImport(CoreML)
import CoreML

/// Acceptance tests for the `T5CoreMLEmbedder` failure-log surface.
///
/// Verifies that a caught `CoreMLNativeError.nativeException`:
/// - Does NOT abort the process (implicit — if the test returns, it didn't)
/// - Results in exactly one valid JSONL row appended to `failureLogURL`
/// - Has all required fields (`name`, `reason`, `inputLength`, `callStack`,
///   `timestamp`)
///
/// No CoreML model asset is required — a `ThrowingStubPredictor` is injected
/// via the internal init.
@Suite("T5CoreMLEmbedder failure-log")
struct T5CoreMLEmbedderFailureLogTests {

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

    // MARK: - Failure-log test

    @Test("ThrowingStubPredictor: no abort, CoreMLNativeError thrown, JSONL row appended")
    func failureLogIsAppended() async throws {
        let tokenizer = try Self.makeTokenizer()
        let stub = ThrowingStubPredictor()

        // Write logs to a temp file that doesn't pre-exist (the embedder must create it).
        let logURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("switchcraft-coreml-test-\(UUID().uuidString).jsonl")
        defer { try? FileManager.default.removeItem(at: logURL) }

        let embedder = T5CoreMLEmbedder(
            predictor: stub,
            tokenizer: tokenizer,
            dims: 128,
            windowSize: 512,
            stride: 256,
            minNorm: 1.0,
            failureLogURL: logURL
        )

        // "test input" has exactly 10 characters; the spec uses this to verify inputLength.
        let inputText = "test input"
        #expect(inputText.count == 10)

        // (a) The encode must throw CoreMLNativeError.nativeException — not abort.
        do {
            _ = try await embedder.encode(inputText)
            Issue.record("Expected CoreMLNativeError to be thrown but encode returned normally")
            return
        } catch let error as CoreMLNativeError {
            guard case .nativeException(let name, let reason, _) = error else {
                Issue.record("Expected .nativeException, got \(error)")
                return
            }
            #expect(name   == ThrowingStubPredictor.exceptionName)
            #expect(reason == ThrowingStubPredictor.exceptionReason)
        }

        // (b) Exactly one JSONL row must be present.
        let logData = try Data(contentsOf: logURL)
        let rawText = try #require(String(data: logData, encoding: .utf8), "log not UTF-8")
        let lines = rawText.split(separator: "\n", omittingEmptySubsequences: true)
        #expect(lines.count == 1, "expected exactly 1 JSONL row, got \(lines.count)")

        // (c) Parse and validate all fields.
        let rowData = try #require(lines.first.map { Data($0.utf8) })
        let json = try #require(
            try JSONSerialization.jsonObject(with: rowData) as? [String: Any],
            "JSONL row is not a JSON object"
        )

        #expect((json["name"]   as? String) == ThrowingStubPredictor.exceptionName)
        #expect((json["reason"] as? String) == ThrowingStubPredictor.exceptionReason)
        #expect((json["inputLength"] as? Int) == inputText.count)

        let callStack = try #require(json["callStack"] as? [String], "callStack must be [String]")
        #expect(!callStack.isEmpty, "callStack should contain at least one frame")

        let timestampStr = try #require(json["timestamp"] as? String, "timestamp must be String")
        let formatter = ISO8601DateFormatter()
        #expect(
            formatter.date(from: timestampStr) != nil,
            "timestamp '\(timestampStr)' must parse as ISO 8601"
        )
    }
}

#endif
