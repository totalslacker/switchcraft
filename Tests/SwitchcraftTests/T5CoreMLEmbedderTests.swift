import Foundation
import Testing
@testable import SwitchcraftCore

#if canImport(CoreML)
import CoreML
@testable import SwitchcraftCoreML

/// Integration tests for the real T5 + projection CoreML model.
///
/// All tests are gated on `SWITCHCRAFT_XTR_MLPACKAGE` resolving to an
/// existing path; when unset, the suite is `.disabled` via the
/// `.enabled(if:)` trait and CI on a fresh checkout stays green.
///
/// To run locally:
///   1. `pip install -r scripts/requirements-coreml.txt`
///   2. `python3 scripts/convert-xtr-to-coreml.py --revision <sha>`
///   3. `export SWITCHCRAFT_XTR_MLPACKAGE=$PWD/Tests/Fixtures/xtr-base-en.mlpackage`
///   4. `swift test --filter T5CoreMLEmbedder`
@Suite(
    "T5CoreMLEmbedder (asset-gated)",
    .enabled(if: CoreMLAsset.isAvailable)
)
struct T5CoreMLEmbedderTests {

    // MARK: - Suite-level shared embedder

    /// One MLModel load per test run. ANE compilation can take 1-3s on
    /// first load; sharing avoids paying that cost N times.
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

    // MARK: - dims

    @Test("dims returns 128")
    func dimsIs128() async throws {
        let embedder = try await SharedEmbedder.shared.get()
        #expect(embedder.dims == 128)
    }

    // MARK: - Empty / whitespace

    @Test("empty string returns []")
    func emptyReturnsEmpty() async throws {
        let embedder = try await SharedEmbedder.shared.get()
        let v = try await embedder.encode("")
        #expect(v.isEmpty)
    }

    @Test("whitespace-only string returns []")
    func whitespaceReturnsEmpty() async throws {
        let embedder = try await SharedEmbedder.shared.get()
        let v = try await embedder.encode("   \t\n  ")
        #expect(v.isEmpty)
    }

    // MARK: - Determinism

    @Test("two identical encodes produce float-equal output")
    func deterministic() async throws {
        let embedder = try await SharedEmbedder.shared.get()
        let a = try await embedder.encode("the apple is red and round")
        let b = try await embedder.encode("the apple is red and round")
        #expect(a == b)
        #expect(a.count % embedder.dims == 0)
        #expect(a.count > 0)
    }

    // MARK: - Long input (sliding window)

    @Test("long input runs sliding-window without crashing and returns at least one row")
    func longInputSlidingWindow() async throws {
        let embedder = try await SharedEmbedder.shared.get()
        // Generate a synthetic long input that tokenises to > 512 tokens.
        // The Frankenstein opening already does this in the conversion-script
        // fixtures; here we just need volume, so repeat a small phrase enough
        // times to clear the threshold.
        let unit = "the quick brown fox jumps over the lazy dog "
        let text = String(repeating: unit, count: 200)
        let v = try await embedder.encode(text)
        #expect(v.count > 0)
        #expect(v.count % embedder.dims == 0)
        let rows = v.count / embedder.dims
        // Each row must be (approximately) unit length.
        for r in 0..<rows {
            var sumSq: Float = 0
            for d in 0..<embedder.dims {
                let x = v[r * embedder.dims + d]
                sumSq += x * x
            }
            let len = sumSq.squareRoot()
            #expect(abs(len - 1) < 1e-3)
        }
    }

    // MARK: - Reference-fixture parity

    @Test(
        "fixture inputs produce embeddings with mean cosine similarity ≥ 0.99 vs PyTorch reference",
        .enabled(if: CoreMLEmbeddingFixtures.isAvailable)
    )
    func fixtureParity() async throws {
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

            // Mean cosine similarity per row.
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
                meanSim >= 0.99,
                "\(fixture.name): mean cosine similarity \(meanSim) < 0.99"
            )
        }
    }
}
#endif
