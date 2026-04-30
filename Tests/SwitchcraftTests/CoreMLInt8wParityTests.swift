import Foundation
import Testing
@testable import SwitchcraftCore

#if canImport(CoreML)
import CoreML
@testable import SwitchcraftCoreML

/// Parity tests for the optional INT8 weight-only `.mlpackage` variant
/// produced by `scripts/quantize-mlpackage-int8w.py` (rung 1 of the
/// "If SmoothQuant fails" ladder — see `docs/Plan.md` and ADR 010(i)).
///
/// The INT8w variant stores Linear-op weights as INT8 with per-channel
/// scales; compute precision stays at FP32 throughout. The within-stack
/// parity contract is **mean cosine similarity ≥ 0.998 vs the committed
/// PyTorch FP32 reference** (`Tests/Fixtures/xtr-base-en.embeddings.{bin,json}`).
/// Per ADR 010(i), the threshold may be ratcheted to 0.999 once the
/// INT8w asset is produced and measured drift is reliably below 0.001;
/// until then 0.998 is the floor.
///
/// Cross-stack INT8w-vs-Witchcraft Q4K parity reuses ADR 010(h)'s
/// existing ±0.025 tolerance unchanged — INT8w weight rounding adds
/// small noise on the Switchcraft side but does not meaningfully close
/// the gap with Q4K.
///
/// All tests are gated on `SWITCHCRAFT_XTR_MLPACKAGE_INT8W` resolving
/// to an existing path; when unset, the suite is `.disabled` via the
/// `.enabled(if:)` trait and CI on a fresh checkout stays green. The
/// fixture-dependent tests are doubly gated on
/// `CoreMLEmbeddingFixtures.isAvailable`. The existing FP32 suite
/// (`T5CoreMLEmbedderTests`, `CoreMLInferenceTests`) gated on
/// `SWITCHCRAFT_XTR_MLPACKAGE` continues to run unchanged.
///
/// To run locally:
///   1. `pip install -r scripts/requirements-coreml.txt`
///   2. `python3 scripts/convert-xtr-to-coreml.py --revision <sha>`
///   3. `python3 scripts/quantize-mlpackage-int8w.py`
///   4. `export SWITCHCRAFT_XTR_MLPACKAGE_INT8W=$PWD/Tests/Fixtures/xtr-base-en-int8w.mlpackage`
///   5. `swift test --filter CoreMLInt8wParity`
@Suite(
    "CoreMLInt8wParity (asset-gated)",
    .enabled(if: CoreMLAssetInt8w.isAvailable)
)
struct CoreMLInt8wParityTests {

    // MARK: - Suite-level shared embedder

    /// One MLModel load per test run. ANE compilation can take 1-3s on
    /// first load; sharing avoids paying that cost N times. Duplicated
    /// from `T5CoreMLEmbedderTests.SharedEmbedder` because the ADR 010
    /// distribution model treats variant selection as consumer-side
    /// (extracting a shared embedder to `Support/` would require
    /// changing the existing FP32 test reference). Same precedent as
    /// `CoreMLInferenceTests.SharedEmbedder`.
    actor SharedEmbedder {
        static let shared = SharedEmbedder()
        private var cached: T5CoreMLEmbedder?

        func get() async throws -> T5CoreMLEmbedder {
            if let e = cached { return e }
            let modelURL = CoreMLAssetInt8w.url!  // gate ensures non-nil
            let tokenizerURL = Bundle.module.url(
                forResource: "xtr-base-en.tokenizer",
                withExtension: "json",
                subdirectory: "Fixtures"
            )!
            let tokenizer = try Tokenizer(contentsOf: tokenizerURL.path)
            let embedder = try await T5CoreMLEmbedder(
                modelURL: modelURL,
                tokenizer: tokenizer,
                modelIdentifier: "google/xtr-base-en@v1-int8w"
            )
            cached = embedder
            return embedder
        }
    }

    // MARK: - Cosine-similarity helper

    /// Mean cosine similarity between row-major actual embeddings and
    /// the reference rows. Bounded by `min(actualRows, referenceRows.count)`
    /// so a row-count mismatch reported by `#expect` upstream cannot
    /// turn into an out-of-bounds crash here.
    private func meanCosineSimilarity(
        actualFlat: [Float],
        referenceRows: [[Float]],
        dims: Int
    ) -> Float {
        let actualRows = actualFlat.count / dims
        let n = min(actualRows, referenceRows.count)
        guard n > 0 else { return 0 }
        var simSum: Float = 0
        for r in 0..<n {
            let ref = referenceRows[r]
            var dot: Float = 0
            var na: Float = 0
            var nb: Float = 0
            for d in 0..<dims {
                let a = actualFlat[r * dims + d]
                let b = ref[d]
                dot += a * b
                na += a * a
                nb += b * b
            }
            let sim = dot / (na.squareRoot() * nb.squareRoot() + 1e-12)
            simSum += sim
        }
        return simSum / Float(n)
    }

    // MARK: - Within-stack parity (INT8w vs PyTorch FP32 reference)

    @Test(
        "INT8w fixture parity: mean cosine similarity ≥ 0.998 vs PyTorch reference",
        .enabled(if: CoreMLEmbeddingFixtures.isAvailable)
    )
    func fixtureParity998() async throws {
        let embedder = try await SharedEmbedder.shared.get()
        let index = try CoreMLEmbeddingFixtures.loadIndex()
        let blob = try CoreMLEmbeddingFixtures.loadBlob()

        for fixture in index.fixtures {
            let referenceRows = CoreMLEmbeddingFixtures.rows(
                for: fixture, blob: blob, dims: index.dims
            )
            let actualFlat = try await embedder.encode(fixture.input)
            #expect(
                actualFlat.count % index.dims == 0,
                "\(fixture.name): output length \(actualFlat.count) is not a multiple of dims \(index.dims)"
            )
            let actualRows = actualFlat.count / index.dims
            #expect(
                actualRows == fixture.rows,
                "\(fixture.name): row count mismatch (got \(actualRows), expected \(fixture.rows))"
            )
            if actualRows == 0 { continue }

            let meanSim = meanCosineSimilarity(
                actualFlat: actualFlat,
                referenceRows: referenceRows,
                dims: index.dims
            )
            #expect(
                meanSim >= 0.998,
                "\(fixture.name): INT8w mean cosine similarity \(meanSim) < 0.998"
            )
        }
    }

    // MARK: - Smoke-level invariants

    @Test("INT8w: dims returns 128")
    func dimsIs128() async throws {
        let embedder = try await SharedEmbedder.shared.get()
        #expect(embedder.dims == 128)
    }

    @Test("INT8w: identical inputs produce float-equal output")
    func deterministic() async throws {
        let embedder = try await SharedEmbedder.shared.get()
        let a = try await embedder.encode("the apple is red and round")
        let b = try await embedder.encode("the apple is red and round")
        #expect(a == b)
        #expect(!a.isEmpty)
        #expect(a.count % embedder.dims == 0)
    }

    @Test("INT8w: rows are unit-length within FP tolerance")
    func rowsAreUnitLength() async throws {
        let embedder = try await SharedEmbedder.shared.get()
        let v = try await embedder.encode("the apple is red and round")
        let dims = embedder.dims
        let rows = v.count / dims
        #expect(rows > 0)
        for r in 0..<rows {
            var sumSq: Float = 0
            for d in 0..<dims {
                let x = v[r * dims + d]
                sumSq += x * x
            }
            let len = sumSq.squareRoot()
            #expect(abs(len - 1) < 1e-3)
        }
    }
}
#endif
