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
//   * `Tests/Fixtures/reference_embeddings.{bin,json}` — committed to the
//     repo (regenerated via `scripts/witchcraft-fixture-export.patch` per
//     ADR 013; regenerate locally if the Metal pipeline changes).
//   * Metal-capable host (no `SWITCHCRAFT_FORCE_ACCELERATE`).
//
// All gates absent → suite skips cleanly. Fresh checkouts stay green.

#if canImport(Metal)

import Foundation
import Testing
@_spi(SwitchcraftMetal) import SwitchcraftCore
@_spi(SwitchcraftMetal) import SwitchcraftMetal

// ASSET-GATED: requires SWITCHCRAFT_XTR_GGUF + reference_embeddings fixture — skips in CI
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
    /// **Calibrated (issue #75, 2026-05-02).** Observed `maxAbs = 0.000216`
    /// on a developer machine with `xtr-v3.gguf` (Q4K) and the committed
    /// `reference_embeddings.bin` (Witchcraft Q4K reference). Both sides are
    /// Q4K-derived so the drift is arithmetic-order rounding only.
    ///
    /// Value = `2 × observed maxAbs` per ADR 010(h) tolerance policy:
    ///   `2 × 0.000216 = 0.000432 → 0.0005` (rounded to 1 sig fig).
    ///
    /// If a kernel revision pushes `observed maxAbs` past `tolerance / 2`
    /// (= 0.00025), re-derive and update ADR 010(h)'s Metal sub-section
    /// in the same commit.
    private static let tolerance: Float = 0.0005

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
