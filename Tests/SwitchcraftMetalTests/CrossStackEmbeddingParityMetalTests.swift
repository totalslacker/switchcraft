// SPDX-License-Identifier: Apache-2.0
//
// Cross-stack embedding parity for `T5MetalEmbedder` (issue #65, ADR
// 010(h) Metal sub-section). Sibling of
// `Tests/SwitchcraftTests/CrossStackEmbeddingParityTests.swift`.
//
// Compares Witchcraft GGUF (Q4K) per-token embeddings vs Switchcraft
// `T5MetalEmbedder` (Q4K-on-Metal) per-token embeddings on the same
// input strings. Both sides are Q4K-derived, so the empirical drift is
// expected to be substantially tighter than ADR 010(h)'s ±0.025
// CoreML-vs-Witchcraft FP32 tolerance — see ADR 014's two-axis
// precision framing.
//
// Tolerance: `Self.tolerance`, set to `2 ×` the empirically-observed
// max abs per-dim drift. The test prints the observed `maxAbs` on every
// run so an operator regenerating fixtures can re-derive the tolerance
// and update ADR 010(h)'s Metal sub-section in lockstep.
//
// Triple-gated:
//   * `SWITCHCRAFT_XTR_GGUF` must point at the Q4K GGUF asset.
//   * `Tests/Fixtures/reference_embeddings.{bin,json}` must be present
//     (regenerated locally via `scripts/witchcraft-fixture-export.patch`
//     per ADR 013 — the fixture is **not committed**).
//   * Metal-capable host (no `SWITCHCRAFT_FORCE_ACCELERATE`).
//
// All gates absent → suite skips cleanly. Fresh checkouts stay green.

#if canImport(Metal)

import Foundation
import Testing
@_spi(SwitchcraftMetal) import SwitchcraftCore
@_spi(SwitchcraftMetal) import SwitchcraftMetal

@Suite(
    "Cross-stack embedding parity Metal (asset-gated)",
    .enabled(if: GGUFAsset.isAvailable
                 && ReferenceEmbeddingsFixtureMetal.isAvailable
                 && MetalAvailability.isAvailable,
             "Set SWITCHCRAFT_XTR_GGUF + regenerate Tests/Fixtures/reference_embeddings.{bin,json}; requires Metal.")
)
struct CrossStackEmbeddingParityMetalTests {

    /// Per-dim absolute tolerance for `T5MetalEmbedder` vs Witchcraft
    /// GGUF reference embeddings.
    ///
    /// Per the issue #65 plan §"Tolerance retune mechanics", this value
    /// is `2 × observed maxAbs` from a one-time empirical measurement on
    /// a developer machine with the regenerated reference fixture and
    /// `SWITCHCRAFT_XTR_GGUF` set. The test prints `maxAbs` on every
    /// run; if the observed value drifts beyond `tolerance / 2` the
    /// operator should re-derive both this constant and the matching
    /// entry in ADR 010(h)'s Metal sub-section.
    ///
    /// Initial conservative ceiling pending in-tree empirical
    /// measurement: 0.01. Both sides Q4K-derived → expected substantially
    /// tighter than the CoreML ±0.025 tolerance per ADR 014.
    private static let tolerance: Float = 0.01

    actor SharedEmbedder {
        static let shared = SharedEmbedder()
        private var cached: T5MetalEmbedder?

        func get() async throws -> T5MetalEmbedder {
            if let e = cached { return e }
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
            cached = embedder
            return embedder
        }
    }

    @Test(
        "T5MetalEmbedder matches Witchcraft GGUF per-token embeddings within Metal-vs-Witchcraft tolerance"
    )
    func referenceEmbeddingsParity() async throws {
        let embedder = try await SharedEmbedder.shared.get()
        let index = try ReferenceEmbeddingsFixtureMetal.loadIndex()
        let blob = try ReferenceEmbeddingsFixtureMetal.loadBlob()

        var globalMaxAbs: Float = 0
        var globalMinCosine: Double = 1.0

        for fixture in index.fixtures {
            let referenceRows = ReferenceEmbeddingsFixtureMetal.rows(
                for: fixture, blob: blob, dims: index.dims
            )
            let actualFlat = try await embedder.encode(fixture.input)
            #expect(
                actualFlat.count % index.dims == 0,
                "\(fixture.name): embedder output length \(actualFlat.count) is not a multiple of dims \(index.dims)"
            )
            let actualRows = actualFlat.count / index.dims
            #expect(
                actualRows == fixture.rows,
                "\(fixture.name): row count mismatch — Metal \(actualRows), Witchcraft \(fixture.rows)"
            )
            if actualRows == 0 || referenceRows.count == 0 { continue }

            // Per-row, per-dim absolute diff.
            var fixtureMaxAbs: Float = 0
            var fixtureMinCosine: Double = 1.0
            let rowsToCompare = min(actualRows, referenceRows.count)
            for r in 0..<rowsToCompare {
                let ref = referenceRows[r]
                var rowDot: Double = 0, rowNa: Double = 0, rowNb: Double = 0
                for d in 0..<index.dims {
                    let actual = actualFlat[r * index.dims + d]
                    let diff = abs(actual - ref[d])
                    if diff > fixtureMaxAbs { fixtureMaxAbs = diff }
                    let a = Double(actual), b = Double(ref[d])
                    rowDot += a * b
                    rowNa += a * a
                    rowNb += b * b
                }
                let denom = rowNa.squareRoot() * rowNb.squareRoot()
                let cos = denom > 0 ? rowDot / denom : 1.0
                if cos < fixtureMinCosine { fixtureMinCosine = cos }
            }
            if fixtureMaxAbs > globalMaxAbs { globalMaxAbs = fixtureMaxAbs }
            if fixtureMinCosine < globalMinCosine { globalMinCosine = fixtureMinCosine }

            #expect(
                fixtureMaxAbs <= Self.tolerance,
                """
                \(fixture.name): max per-token absolute diff \(fixtureMaxAbs) > \
                tolerance \(Self.tolerance). If this is the first run on a \
                regenerated fixture, re-derive both the test tolerance and \
                ADR 010(h)'s Metal sub-section to `2 × observed maxAbs`.
                """
            )
        }

        // Always print the empirical numbers — operators rely on this
        // output to keep the tolerance constant (and ADR 010(h)) honest.
        print("""
              CrossStackEmbeddingParityMetal: \
              maxAbs=\(globalMaxAbs), \
              minCosine=\(globalMinCosine), \
              tolerance=\(Self.tolerance)
              """)
    }
}

#endif // canImport(Metal)
