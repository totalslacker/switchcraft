import Foundation
import Testing
@testable import SwitchcraftCore

#if canImport(CoreML)
import CoreML
@testable import SwitchcraftCoreML

/// Strict-parity integration tests for the real T5 + projection CoreML
/// model.
///
/// This suite enforces the **`≥ 0.999` mean cosine similarity** bar from
/// ADR 010(e) — the same threshold the conversion script's parity gate
/// uses — over the full tokenizer → encoder pipeline. It is a stricter
/// counterpart to `T5CoreMLEmbedderTests.fixtureParity` (which keeps the
/// looser `≥ 0.99` bar as a smoke gate). Both suites coexist; the
/// stricter suite is the one that protects against regressions that
/// would silently pass the smoke gate but fail the conversion gate.
///
/// Asset gating mirrors `T5CoreMLEmbedderTests`: when
/// `SWITCHCRAFT_XTR_MLPACKAGE` is unset or invalid, the suite skips
/// cleanly so a fresh checkout stays green.
///
/// To run locally:
///   1. `pip install -r scripts/requirements-coreml.txt`
///   2. `python3 scripts/convert-xtr-to-coreml.py --revision <sha>`
///   3. `export SWITCHCRAFT_XTR_MLPACKAGE=$PWD/Tests/Fixtures/xtr-base-en.mlpackage`
///   4. `swift test --filter CoreMLInference`
@Suite(
    "CoreMLInference (asset-gated)",
    .enabled(if: CoreMLAsset.isAvailable)
)
struct CoreMLInferenceTests {

    // MARK: - Suite-level shared embedder

    /// One MLModel load per test run. ANE compilation can take 1-3s on
    /// first load; sharing avoids paying that cost N times. Duplicated
    /// from `T5CoreMLEmbedderTests.SharedEmbedder` because the spec
    /// forbids modifying that file (extracting it to `Support/` would
    /// require changing the existing reference).
    actor SharedEmbedder {
        static let shared = SharedEmbedder()
        private var cached: T5CoreMLEmbedder?

        func get() async throws -> T5CoreMLEmbedder {
            if let e = cached { return e }
            let modelURL = CoreMLAsset.url!  // gate ensures non-nil
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

    // MARK: - Strict full-pipeline parity

    @Test(
        "fixture parity: mean cosine similarity ≥ 0.999 vs PyTorch reference",
        .enabled(if: CoreMLEmbeddingFixtures.isAvailable)
    )
    func fixtureParity999() async throws {
        let embedder = try await SharedEmbedder.shared.get()
        let index = try CoreMLEmbeddingFixtures.loadIndex()
        let blob = try CoreMLEmbeddingFixtures.loadBlob()

        for fixture in index.fixtures {
            let referenceRows = CoreMLEmbeddingFixtures.rows(
                for: fixture, blob: blob, dims: index.dims
            )
            let actualFlat = try await embedder.encode(fixture.input)
            let actualRows = actualFlat.count / index.dims
            #expect(
                actualRows == fixture.rows,
                "\(fixture.name): row count mismatch (got \(actualRows), expected \(fixture.rows))"
            )
            if actualRows == 0 { continue }

            var simSum: Float = 0
            for r in 0..<actualRows {
                let ref = referenceRows[r]
                var dot: Float = 0
                var na: Float = 0
                var nb: Float = 0
                for d in 0..<index.dims {
                    let a = actualFlat[r * index.dims + d]
                    let b = ref[d]
                    dot += a * b
                    na += a * a
                    nb += b * b
                }
                let sim = dot / (na.squareRoot() * nb.squareRoot() + 1e-12)
                simSum += sim
            }
            let meanSim = simSum / Float(actualRows)
            #expect(
                meanSim >= 0.999,
                "\(fixture.name): mean cosine similarity \(meanSim) < 0.999"
            )
        }
    }

    // MARK: - Sliding-window end-to-end correctness

    /// Exercises the live `T5CoreMLEmbedder.encode` path through multiple
    /// sliding-window inferences (the Frankenstein opening tokenises to
    /// > 512 tokens, which forces `windowSize=512, stride=256` to plan
    /// multiple windows and merge their overlap region).
    ///
    /// `SlidingWindowTests` covers the planner+merger in isolation with
    /// synthetic data; this is the end-to-end sibling that runs the same
    /// path through the real model and validates against the PyTorch
    /// reference at the strict `≥ 0.999` bar.
    @Test(
        "sliding-window end-to-end: long_frankenstein_opening parity ≥ 0.999",
        .enabled(if: CoreMLEmbeddingFixtures.isAvailable)
    )
    func slidingWindowEndToEndParity999() async throws {
        let embedder = try await SharedEmbedder.shared.get()
        let index = try CoreMLEmbeddingFixtures.loadIndex()
        let blob = try CoreMLEmbeddingFixtures.loadBlob()

        guard
            let fixture = index.fixtures.first(where: { $0.name == "long_frankenstein_opening" })
        else {
            Issue.record("long_frankenstein_opening fixture missing from xtr-base-en.embeddings.json")
            return
        }

        let referenceRows = CoreMLEmbeddingFixtures.rows(
            for: fixture, blob: blob, dims: index.dims
        )
        let actualFlat = try await embedder.encode(fixture.input)
        let actualRows = actualFlat.count / index.dims
        #expect(
            actualRows == fixture.rows,
            "\(fixture.name): row count mismatch (got \(actualRows), expected \(fixture.rows))"
        )
        guard actualRows > 0 else { return }

        var simSum: Float = 0
        for r in 0..<actualRows {
            let ref = referenceRows[r]
            var dot: Float = 0
            var na: Float = 0
            var nb: Float = 0
            for d in 0..<index.dims {
                let a = actualFlat[r * index.dims + d]
                let b = ref[d]
                dot += a * b
                na += a * a
                nb += b * b
            }
            let sim = dot / (na.squareRoot() * nb.squareRoot() + 1e-12)
            simSum += sim
        }
        let meanSim = simSum / Float(actualRows)
        #expect(
            meanSim >= 0.999,
            "\(fixture.name): sliding-window mean cosine similarity \(meanSim) < 0.999"
        )
    }

    // MARK: - MIN_NORM row-count filter verification

    /// Verifies that `T5CoreMLEmbedder.encode` drops exactly the rows the
    /// PyTorch reference dropped via the `MIN_NORM` filter, by row count,
    /// for every fixture — including the whitespace fixture (`rows == 0`)
    /// which must produce an empty result.
    @Test(
        "MIN_NORM row-count filter matches PyTorch reference for every fixture",
        .enabled(if: CoreMLEmbeddingFixtures.isAvailable)
    )
    func rowCountFilterMatchesReference() async throws {
        let embedder = try await SharedEmbedder.shared.get()
        let index = try CoreMLEmbeddingFixtures.loadIndex()

        for fixture in index.fixtures {
            let actualFlat = try await embedder.encode(fixture.input)
            #expect(
                actualFlat.count % index.dims == 0,
                "\(fixture.name): output length \(actualFlat.count) is not a multiple of dims \(index.dims)"
            )
            let actualRows = actualFlat.count / index.dims
            #expect(
                actualRows == fixture.rows,
                "\(fixture.name): MIN_NORM-filtered row count mismatch (got \(actualRows), expected \(fixture.rows))"
            )
        }
    }
}
#endif
