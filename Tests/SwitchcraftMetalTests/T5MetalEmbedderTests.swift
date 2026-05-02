// SPDX-License-Identifier: Apache-2.0
//
// Parity + edge-case tests for `T5MetalEmbedder` (issue #64).
//
// The cosine ≥ 0.99999 gate against the committed PyTorch FP32 reference
// (`Tests/Fixtures/xtr-base-en.embeddings.bin`) is the primary
// correctness contract. All asset-gated tests skip cleanly when
// `SWITCHCRAFT_XTR_GGUF` is unset — fresh checkouts stay green.

#if canImport(Metal)

import Foundation
import Metal
import Testing
import SwitchcraftCore
@_spi(SwitchcraftMetal) import SwitchcraftCore
@_spi(SwitchcraftMetal) import SwitchcraftMetal

// MARK: - Helpers

private struct EmbedderTestEnvironment {
    let embedder: T5MetalEmbedder
    let index: CoreMLEmbeddingFixtures.Index
    let blob: Data
}

private func makeEmbedder() async throws -> EmbedderTestEnvironment {
    let url = try #require(GGUFAsset.url, "GGUFAsset.url was nil")
    let tokenizerURL = try #require(
        TokenizerFixture.url,
        "tokenizer fixture missing — Tests/Fixtures/xtr-base-en.tokenizer.json absent"
    )
    let tokenizer = try Tokenizer(contentsOf: tokenizerURL.path)
    let index = try CoreMLEmbeddingFixtures.loadIndex()
    let blob = try CoreMLEmbeddingFixtures.loadBlob()
    let embedder = try await T5MetalEmbedder(
        modelURL: url,
        tokenizer: tokenizer,
        dims: index.dims,
        windowSize: index.windowSize,
        stride: index.stride,
        minNorm: index.minNorm
    )
    return EmbedderTestEnvironment(embedder: embedder, index: index, blob: blob)
}

private func cosine(_ a: [Float], _ b: [Float]) -> Double {
    precondition(a.count == b.count)
    var dot: Double = 0, nA: Double = 0, nB: Double = 0
    for i in 0..<a.count {
        let ai = Double(a[i]), bi = Double(b[i])
        dot += ai * bi
        nA += ai * ai
        nB += bi * bi
    }
    let denom = nA.squareRoot() * nB.squareRoot()
    return denom > 0 ? dot / denom : 1.0
}

private func maxAbs(_ a: [Float], _ b: [Float]) -> Double {
    precondition(a.count == b.count)
    var m: Double = 0
    for i in 0..<a.count {
        let d = abs(Double(a[i]) - Double(b[i]))
        if d > m { m = d }
    }
    return m
}

// MARK: - Init failure

@Suite("T5MetalEmbedder init failures")
struct T5MetalEmbedderInitFailureTests {

    /// `init` throws `metalUnavailable` cleanly when Metal is forced
    /// off via `SWITCHCRAFT_FORCE_ACCELERATE` (or the host has no GPU).
    /// Gated to only run when the env var is set so default `swift test`
    /// runs with Metal available aren't impacted.
    @Test("init throws metalUnavailable when SWITCHCRAFT_FORCE_ACCELERATE is set",
          .enabled(if: !MetalAvailability.isAvailable,
                   "Run only when Metal is unavailable (SWITCHCRAFT_FORCE_ACCELERATE=1 or no GPU)"))
    func metalUnavailable() async throws {
        // Use any path — the throw happens before the GGUF read.
        let bogusURL = URL(fileURLWithPath: "/nonexistent/path.gguf")
        // Use a placeholder tokenizer; if Metal is unavailable, init
        // throws before the tokenizer is consulted.
        let tokenizerURL = try #require(TokenizerFixture.url)
        let tokenizer = try Tokenizer(contentsOf: tokenizerURL.path)
        do {
            _ = try await T5MetalEmbedder(
                modelURL: bogusURL,
                tokenizer: tokenizer
            )
            Issue.record("Expected metalUnavailable, got success")
        } catch T5MetalEmbedderError.metalUnavailable {
            // ✓
        } catch {
            Issue.record("Expected metalUnavailable, got \(error)")
        }
    }
}

// MARK: - Asset-gated parity tests

// ASSET-GATED: requires SWITCHCRAFT_XTR_GGUF — skips in CI and on fresh checkouts
@Suite("T5MetalEmbedder PyTorch FP32 parity",
       .enabled(if: GGUFAsset.isAvailable
                    && CoreMLEmbeddingFixtures.isAvailable
                    && MetalAvailability.isAvailable,
                "Set SWITCHCRAFT_XTR_GGUF=<path>; requires Metal."))
struct T5MetalEmbedderParityTests {

    /// Per-token cosine ≥ 0.99999 vs PyTorch FP32 reference for every
    /// fixture in `xtr-base-en.embeddings.bin`. This is issue #64's
    /// primary correctness gate.
    @Test("Per-token cosine ≥ 0.99999 vs PyTorch FP32 reference")
    func perTokenCosine() async throws {
        let env = try await makeEmbedder()
        for fixture in env.index.fixtures {
            // Whitespace-only fixture has zero rows; skip — the embedder
            // returns [] which is correct, but there's nothing to compare.
            if fixture.rows == 0 { continue }

            let expected = CoreMLEmbeddingFixtures.rows(
                for: fixture, blob: env.blob, dims: env.index.dims
            )
            let actualFlat = try await env.embedder.encode(fixture.input)
            let dims = env.index.dims
            #expect(
                actualFlat.count == fixture.rows * dims,
                "Fixture '\(fixture.name)': got \(actualFlat.count / dims) rows, expected \(fixture.rows)"
            )
            for r in 0..<fixture.rows {
                let actual = Array(actualFlat[(r * dims)..<((r + 1) * dims)])
                let cos = cosine(actual, expected[r])
                let abs = maxAbs(actual, expected[r])
                #expect(
                    cos >= 0.99999,
                    "Fixture '\(fixture.name)' row \(r): cosine \(cos) < 0.99999"
                )
                #expect(
                    abs < 1e-4,
                    "Fixture '\(fixture.name)' row \(r): maxAbsError \(abs) ≥ 1e-4"
                )
            }
        }
    }

    /// Two consecutive calls with the same input produce bit-equal
    /// output (per the `Embedder` doc contract).
    @Test("Determinism: same input → bit-equal output across two calls")
    func determinism() async throws {
        let env = try await makeEmbedder()
        // Pick the first non-empty fixture as the input.
        let fixture = try #require(
            env.index.fixtures.first(where: { $0.rows > 0 }),
            "No non-empty fixture available"
        )
        let a = try await env.embedder.encode(fixture.input)
        let b = try await env.embedder.encode(fixture.input)
        #expect(a == b, "Two consecutive encodes were not bit-equal")
    }

    /// Sliding-window correctness: `long_frankenstein_opening` is 554
    /// tokens which exceeds the 512 window, so the merger runs.
    @Test("Sliding-window correctness on long Frankenstein fixture")
    func slidingWindow() async throws {
        let env = try await makeEmbedder()
        guard let fixture = env.index.fixtures.first(where: {
            $0.name == "long_frankenstein_opening"
        }) else {
            Issue.record("long_frankenstein_opening fixture missing")
            return
        }
        let expected = CoreMLEmbeddingFixtures.rows(
            for: fixture, blob: env.blob, dims: env.index.dims
        )
        let actualFlat = try await env.embedder.encode(fixture.input)
        let dims = env.index.dims
        #expect(
            actualFlat.count == fixture.rows * dims,
            "Sliding-window output row count mismatch: got \(actualFlat.count / dims), expected \(fixture.rows)"
        )
        // Spot-check a handful of rows — the parity test above already
        // covers all rows; this guarantees the merge path itself is
        // exercised even if the parity test name is changed.
        let sampleRows = [0, fixture.rows / 2, fixture.rows - 1]
        for r in sampleRows {
            let actual = Array(actualFlat[(r * dims)..<((r + 1) * dims)])
            let cos = cosine(actual, expected[r])
            #expect(cos >= 0.99999, "Frankenstein row \(r): cosine \(cos)")
        }
    }
}

// MARK: - Tensor-name regression (issue #74)

/// Verifies that GGUFReader correctly surfaces a tensor named
/// "linear.weight" — the bare name Witchcraft's `quantize-tool` writes
/// for the 2_Dense projection at the pinned Candle rev `5bd5618`.
/// This regression test ensures T5MetalEmbedder's candidate-name probe
/// can find the tensor if the GGUF asset uses this naming convention.
@Suite("T5MetalEmbedder linear.weight tensor-name regression")
struct T5MetalEmbedderLinearWeightTests {

    @Test("GGUFReader surfaces a tensor named linear.weight")
    func linearWeightTensorName() async throws {
        guard let dev = MTLCreateSystemDefaultDevice() else {
            throw GGUFError.ioError("no Metal device")
        }
        // Build a minimal GGUF fixture with one F32 tensor named
        // "linear.weight". GGUF shape is fastest-varying-first (GGML
        // convention): for the 2_Dense projection [out=128, in=768] the
        // on-disk shape is [768, 128]. 98304 elements = 393216 bytes.
        var fixture = GGUFFixture()
        let byteCount = 768 * 128 * 4  // F32
        fixture.tensors = [
            .init(
                name: "linear.weight",
                shape: [768, 128],  // fastest-varying-first: in_features=768, out_features=128
                typeRaw: 0,  // F32
                bytes: Data(count: byteCount)
            )
        ]
        let url = try writeFixture(fixture, named: "linear-weight.gguf")
        defer { try? FileManager.default.removeItem(at: url) }

        let reader = try GGUFReader(url: url, device: dev)
        let names = await reader.tensorNames()
        #expect(names.contains("linear.weight"), "tensor named linear.weight not found; got: \(names)")

        // Confirm the tensor resolves with the correct dtype and shape.
        let tensor = try await reader.tensor("linear.weight")
        #expect(tensor.dtype == .f32)
        #expect(tensor.shape == [768, 128])
        #expect(tensor.elementCount == 768 * 128)
    }
}

#endif // canImport(Metal)
