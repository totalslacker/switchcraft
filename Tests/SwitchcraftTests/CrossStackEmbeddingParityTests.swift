import Foundation
import Testing
@testable import SwitchcraftCore

#if canImport(CoreML)
import CoreML
@testable import SwitchcraftCoreML

/// Cross-stack embedding parity (issue #28 / ADR 013(h-bis)):
/// Witchcraft GGUF (Q4K) per-token embeddings vs Switchcraft CoreML
/// FP32 per-token embeddings, on the same input strings.
///
/// Tolerance: ±0.025 absolute per dimension, per ADR 010(h). The bound
/// reflects measured FP32-vs-Q4K drift on the XTR T5 graph; matched
/// precision is not achievable on this graph (T5 encoder body NaNs
/// under FP16 compute — see ADR 010(c)).
///
/// Doubly gated:
///   * `SWITCHCRAFT_XTR_MLPACKAGE` must point at a CoreML `.mlpackage`
///     produced by `scripts/convert-xtr-to-coreml.py`.
///   * `Tests/Fixtures/reference_embeddings.{bin,json}` must be
///     committed (regenerated via `scripts/witchcraft-fixture-export.patch`).
///
/// Both gates absent → suite skips cleanly. Fresh checkouts stay green.
@Suite(
    "Cross-stack embedding parity (asset-gated)",
    .enabled(if: CoreMLAsset.isAvailable && ReferenceEmbeddingsFixture.isAvailable)
)
struct CrossStackEmbeddingParityTests {

    actor SharedEmbedder {
        static let shared = SharedEmbedder()
        private var cached: T5CoreMLEmbedder?

        func get() async throws -> T5CoreMLEmbedder {
            if let e = cached { return e }
            let modelURL = CoreMLAsset.url!
            let tokenizerURL = Bundle.module.url(
                forResource: "xtr-base-en.tokenizer",
                withExtension: "json",
                subdirectory: "Fixtures"
            )!
            let tokenizer = try Tokenizer(contentsOf: tokenizerURL.path)
            let embedder = try await T5CoreMLEmbedder(
                modelURL: modelURL,
                tokenizer: tokenizer
            )
            cached = embedder
            return embedder
        }
    }

    @Test(
        "Switchcraft CoreML matches Witchcraft GGUF per-token embeddings within ±0.025"
    )
    func referenceEmbeddingsParity() async throws {
        let embedder = try await SharedEmbedder.shared.get()
        let index = try ReferenceEmbeddingsFixture.loadIndex()
        let blob = try ReferenceEmbeddingsFixture.loadBlob()

        for fixture in index.fixtures {
            let referenceRows = ReferenceEmbeddingsFixture.rows(
                for: fixture, blob: blob, dims: index.dims
            )
            let actualFlat = try await embedder.encode(fixture.input)
            let actualRows = actualFlat.count / index.dims
            #expect(
                actualRows == fixture.rows,
                "\(fixture.name): row count mismatch — Swift \(actualRows), Witchcraft \(fixture.rows)"
            )
            if actualRows == 0 || referenceRows.count == 0 { continue }

            // Per-row, per-dim absolute diff. ±0.025 per ADR 010(h).
            var maxAbs: Float = 0
            let rowsToCompare = min(actualRows, referenceRows.count)
            for r in 0..<rowsToCompare {
                let ref = referenceRows[r]
                for d in 0..<index.dims {
                    let diff = abs(actualFlat[r * index.dims + d] - ref[d])
                    if diff > maxAbs { maxAbs = diff }
                }
            }
            #expect(
                maxAbs <= 0.025,
                "\(fixture.name): max per-token absolute diff \(maxAbs) > 0.025"
            )
        }
    }
}
#endif
