// SPDX-License-Identifier: Apache-2.0
//
// Parity + edge-case tests for `SoftmaxKernel` (issue #62). Validates
// that the Metal `kernel_soft_max_f32` produces output matching a
// pure-Swift FP32 softmax reference at the T5-base attention shape
// (B = 1 × 12 × 512 = 6144, N = 512 — one softmax per (batch, head,
// query-token) triple) and at synthetic edge-case inputs.
//
// Reference path
// --------------
//
// Pure Swift, FP32 throughout, computed row-wise over the flat
// `B × N` layout the kernel sees:
//   row_i  = scale * x_i + bias_i           (bias optional)
//   m_i    = max(row_i)                     (subtract-max stabiliser)
//   e_i    = exp(row_i - m_i)
//   y_i    = e_i / sum(e_i)
//
// Edge cases asserted explicitly:
//   - all-equal input row → output uniform `1/N` (the post-softmax
//     row of any constant input must be exactly the uniform
//     distribution, regardless of the constant value).
//   - single very large value at a known position → output near
//     one-hot at that position (the dominant element drives every
//     other position's exp to underflow).
//   - strongly negative bias on selected positions → output exactly
//     zero at those positions (FP32 `exp(very_negative)` underflows
//     to hard zero; tolerance must allow exact-zero, not just a
//     positive epsilon floor — that's the mask-semantics contract).
//
// All tests gate on `MetalAvailability.isAvailable` so a no-GPU host
// (or a `SWITCHCRAFT_FORCE_ACCELERATE`-set run) skips cleanly.

#if canImport(Metal)

import Foundation
import Metal
import Testing
@_spi(SwitchcraftMetal) import SwitchcraftCore
@_spi(SwitchcraftMetal) import SwitchcraftMetal

// MARK: - PRNG + reference

/// Deterministic LCG. Matches the shape used in `Q4KMatMulKernelTests`
/// and `RMSNormKernelTests` so seeded runs reproduce across hosts.
private struct LCG {
    var state: UInt64
    init(seed: UInt64) { self.state = seed &* 0x9E3779B97F4A7C15 &+ 1 }

    mutating func nextU32() -> UInt32 {
        state = state &* 6364136223846793005 &+ 1442695040888963407
        return UInt32(truncatingIfNeeded: state >> 32)
    }

    /// Centred float in `[-1, 1)`.
    mutating func nextFloat() -> Float {
        let f = Float(nextU32()) / Float(UInt32.max)
        return f * 2 - 1
    }
}

private enum SoftmaxReference {
    /// Pure-Swift FP32 row-wise softmax with optional additive bias.
    /// Row layout matches the kernel's flat `B × N` view; if `bias`
    /// is non-nil it must also be `B × N`.
    static func compute(
        x: [Float], bias: [Float]?,
        B: Int, N: Int, scale: Float
    ) -> [Float] {
        precondition(x.count == B * N)
        if let bias = bias { precondition(bias.count == B * N) }
        var y = [Float](repeating: 0, count: B * N)
        for r in 0..<B {
            // FP64 accumulator on the reference side so a kernel using
            // a clean FP32 accumulator can still match within tolerance
            // — and so the reference itself is not the source of any
            // precision drift in the comparison.
            var maxVal: Double = -.infinity
            for k in 0..<N {
                let v = Double(x[r * N + k]) * Double(scale)
                  + (bias.map { Double($0[r * N + k]) } ?? 0)
                if v > maxVal { maxVal = v }
            }
            var sum: Double = 0
            var rowExp = [Double](repeating: 0, count: N)
            for k in 0..<N {
                let v = Double(x[r * N + k]) * Double(scale)
                  + (bias.map { Double($0[r * N + k]) } ?? 0)
                let e = (v - maxVal).exp()
                rowExp[k] = e
                sum += e
            }
            let inv = 1.0 / sum
            for k in 0..<N {
                y[r * N + k] = Float(rowExp[k] * inv)
            }
        }
        return y
    }
}

// Double extension confined to this file — `Foundation` provides
// `exp(_:)` as a free function, but a method form keeps the reference
// loop's expression short and consistent with `.squareRoot()` usage
// in the sibling RMSNorm reference.
private extension Double {
    func exp() -> Double { Foundation.exp(self) }
}

// MARK: - Kernel runner

private enum SoftmaxRunner {
    /// One-shot dispatch helper. Builds Metal buffers, encodes the
    /// kernel into a fresh command buffer, commits + waits, returns
    /// the FP32 output array.
    static func run(
        ctx: MetalContext,
        kernel: SoftmaxKernel,
        x: [Float], bias: [Float]?,
        B: Int, N: Int, scale: Float
    ) throws -> [Float] {
        let inBuf = try makeBuffer(ctx: ctx, floats: x, label: "input")
        let biasBuf: MTLBuffer? = try bias.map {
            try makeBuffer(ctx: ctx, floats: $0, label: "bias")
        }
        let outLen = B * N * MemoryLayout<Float>.size
        guard let outBuf = ctx.device.makeBuffer(
            length: outLen,
            options: .storageModeShared
        ) else {
            throw MetalContextError.pipelineCreationFailed("output buffer alloc returned nil")
        }
        memset(outBuf.contents(), 0, outLen)

        guard let cmd = ctx.queue.makeCommandBuffer() else {
            throw MetalContextError.pipelineCreationFailed("makeCommandBuffer returned nil")
        }
        kernel.encode(
            commandBuffer: cmd,
            input: inBuf, bias: biasBuf, output: outBuf,
            B: B, N: N, scale: scale
        )
        cmd.commit()
        cmd.waitUntilCompleted()
        if let err = cmd.error {
            throw MetalContextError.pipelineCreationFailed("command buffer error: \(err)")
        }

        let raw = outBuf.contents().bindMemory(to: Float.self, capacity: B * N)
        return Array(UnsafeBufferPointer(start: raw, count: B * N))
    }

    private static func makeBuffer(
        ctx: MetalContext, floats: [Float], label: String
    ) throws -> MTLBuffer {
        return try floats.withUnsafeBufferPointer { ptr -> MTLBuffer in
            guard
                let base = ptr.baseAddress,
                let buf = ctx.device.makeBuffer(
                    bytes: base,
                    length: ptr.count * MemoryLayout<Float>.size,
                    options: .storageModeShared
                )
            else {
                throw MetalContextError.pipelineCreationFailed("\(label) buffer alloc returned nil")
            }
            return buf
        }
    }
}

// MARK: - Parity suite

@Suite("SoftmaxKernel parity",
       .enabled(if: MetalAvailability.isAvailable,
                "Metal unavailable on this host (or SWITCHCRAFT_FORCE_ACCELERATE is set)"))
struct SoftmaxKernelTests {

    /// T5 attention shape, no bias. `B = batch × heads × M = 1 × 12 × 512
    /// = 6144`, `N = 512`. The kernel's `tptg = 512` exercises the
    /// multi-simdgroup branch (`tptg.x > N_SIMDWIDTH`).
    @Test("T5 attention parity, no bias (B=6144, N=512)")
    func parityT5NoBias() throws {
        try runParity(B: 6144, N: 512, withBias: false, seed: 0xDA7A_C0DE_0501)
    }

    /// T5 attention shape, with synthetic relative-position bias. The
    /// bias has the upstream `(nheads, M, N) = (12, 512, 512)` shape;
    /// it is flattened into `B × N = 6144 × 512` order before being
    /// handed to the kernel. The flattening rule is
    /// `flat[batch*nheads*M + head*M + m, k] = bias3d[head, m, k]`,
    /// so for `batch = 1` (the parity-test framing) the row index is
    /// `head * M + m`.
    @Test("T5 attention parity, with bias (B=6144, N=512)")
    func parityT5WithBias() throws {
        try runParity(B: 6144, N: 512, withBias: true, seed: 0xDA7A_C0DE_0502)
    }

    /// Small parity case to exercise the single-simdgroup branch
    /// (`tptg.x = 32 ≤ N_SIMDWIDTH` so the inter-simdgroup reduction
    /// is skipped).
    @Test("synthetic parity, no bias, small (B=8, N=32)")
    func paritySmallNoBias() throws {
        try runParity(B: 8, N: 32, withBias: false, seed: 0xDA7A_C0DE_0503)
    }

    /// Non-power-of-two N to verify the `i00 += tptg.x` stride loop
    /// handles tptg > N correctly (extra threads contribute -INF / 0
    /// to the simd reductions, which are the identities).
    @Test("synthetic parity, no bias, non-aligned N (B=4, N=100)")
    func parityNonAlignedN() throws {
        try runParity(B: 4, N: 100, withBias: false, seed: 0xDA7A_C0DE_0504)
    }

    private func runParity(B: Int, N: Int, withBias: Bool, seed: UInt64) throws {
        let ctx = try #require(MetalContext.shared,
                               "MetalContext.shared was nil on a Metal-capable host")
        let kernel = try SoftmaxKernel(context: ctx)

        var rng = LCG(seed: seed)
        var x = [Float](repeating: 0, count: B * N)
        for i in 0..<(B * N) { x[i] = rng.nextFloat() }

        // Synthetic relpos-style bias: structured per-head, per-(query,
        // key) values whose magnitude swings around 0.5 — within FP32
        // range, large enough to perturb the softmax distribution
        // visibly so the parity assertion has signal.
        var bias: [Float]? = nil
        if withBias {
            // T5-base relpos bias is per-head (nheads = 12). For B = 6144
            // and N = 512 this corresponds to (1, 12, 512, 512) → a
            // 12×512×512 buffer flattened over (head, m, k). For other
            // shapes we still construct a B×N bias to exercise the
            // generic flat-layout contract.
            let heads = 12
            let M = (B == 6144) ? 512 : B
            var b = [Float](repeating: 0, count: B * N)
            for r in 0..<B {
                let h = r / M
                let m = r % M
                for k in 0..<N {
                    // Deterministic mix of (h, m, k) with mild magnitude.
                    let a = Float(h % heads) * 0.05
                    let c = Float(m - k) * 0.001
                    b[r * N + k] = a + c + 0.25 * rng.nextFloat()
                }
            }
            bias = b
        }

        let scale: Float = 1.0
        let kernelOut = try SoftmaxRunner.run(
            ctx: ctx, kernel: kernel,
            x: x, bias: bias, B: B, N: N, scale: scale
        )
        let referenceOut = SoftmaxReference.compute(
            x: x, bias: bias, B: B, N: N, scale: scale
        )

        var minCos: Double = 1.0
        var maxAbs: Double = 0
        for r in 0..<B {
            let kRow = Array(kernelOut[(r * N)..<((r + 1) * N)])
            let rRow = Array(referenceOut[(r * N)..<((r + 1) * N)])
            let cos = MetalCorrectness.cosineSimilarity(kRow, rRow)
            let err = MetalCorrectness.maxAbsError(kRow, rRow)
            if cos < minCos { minCos = cos }
            if err > maxAbs { maxAbs = err }
        }
        print("[Softmax] B=\(B) N=\(N) bias=\(withBias) — min row cosine \(minCos), max abs err \(maxAbs)")
        #expect(minCos >= 0.99999,
                "Softmax parity at (B=\(B), N=\(N), bias=\(withBias)): row cosine \(minCos) below 0.99999")

        // Sum-to-one is the structural softmax invariant — kernel
        // output rows must each sum to 1 within FP32 round-off.
        for r in 0..<B {
            var s: Double = 0
            for k in 0..<N { s += Double(kernelOut[r * N + k]) }
            #expect(abs(s - 1.0) < 1e-5,
                    "row \(r) sum \(s) deviates from 1.0 beyond 1e-5")
        }
    }
}

// MARK: - Edge case suite

@Suite("SoftmaxKernel edge cases",
       .enabled(if: MetalAvailability.isAvailable,
                "Metal unavailable on this host (or SWITCHCRAFT_FORCE_ACCELERATE is set)"))
struct SoftmaxKernelEdgeCaseTests {

    @Test("all-equal input yields uniform 1/N")
    func allEqualUniform() throws {
        let ctx = try #require(MetalContext.shared)
        let kernel = try SoftmaxKernel(context: ctx)

        let B = 4, N = 256
        // Pick a non-zero constant to defeat any "input was already
        // uniform-by-luck" interpretation.
        let x = [Float](repeating: 3.14159, count: B * N)
        let out = try SoftmaxRunner.run(
            ctx: ctx, kernel: kernel,
            x: x, bias: nil, B: B, N: N, scale: 1.0
        )
        let expected: Float = 1.0 / Float(N)
        var maxAbs: Float = 0
        for v in out { maxAbs = max(maxAbs, abs(v - expected)) }
        #expect(maxAbs < 1e-6,
                "all-equal-input softmax diverged from uniform \(expected) (max abs err \(maxAbs))")
    }

    @Test("single very large value yields near one-hot")
    func singleLargeValueOneHot() throws {
        let ctx = try #require(MetalContext.shared)
        let kernel = try SoftmaxKernel(context: ctx)

        let B = 2, N = 64
        let hotIndex = 42
        var x = [Float](repeating: 0, count: B * N)
        for r in 0..<B { x[r * N + hotIndex] = 50.0 }

        let out = try SoftmaxRunner.run(
            ctx: ctx, kernel: kernel,
            x: x, bias: nil, B: B, N: N, scale: 1.0
        )
        for r in 0..<B {
            // Hot position should be ≈ 1.
            let hot = out[r * N + hotIndex]
            #expect(hot > 0.999_999,
                    "row \(r) hot position \(hotIndex) = \(hot), expected ≈ 1")
            // Every other position should underflow to ~0.
            for k in 0..<N where k != hotIndex {
                #expect(out[r * N + k] < 1e-15,
                        "row \(r) cold position \(k) = \(out[r * N + k]), expected ≈ 0")
            }
        }
    }

    /// Strongly-negative bias is the mask-semantics contract: a bias
    /// of e.g. `-1e9` at position k drives `exp(...) → 0` (FP32
    /// underflows to hard zero well before reaching the global max),
    /// so the softmax output at that position is exactly 0. This is
    /// how T5 attention masks padding and how downstream callers
    /// would mask future tokens. Tolerance must allow exact-zero —
    /// not just `< epsilon` — to certify the mask is total, not
    /// merely small.
    @Test("strongly negative bias produces exact-zero output (mask semantics)")
    func strongNegativeBiasMask() throws {
        let ctx = try #require(MetalContext.shared)
        let kernel = try SoftmaxKernel(context: ctx)

        let B = 3, N = 32
        // Random input — the mask must dominate regardless of input
        // magnitude.
        var rng = LCG(seed: 0xFEED_FACE_BABE_0001)
        var x = [Float](repeating: 0, count: B * N)
        for i in 0..<(B * N) { x[i] = rng.nextFloat() }

        // Mask the first half of every row.
        var bias = [Float](repeating: 0, count: B * N)
        for r in 0..<B {
            for k in 0..<(N / 2) { bias[r * N + k] = -1e9 }
        }

        let out = try SoftmaxRunner.run(
            ctx: ctx, kernel: kernel,
            x: x, bias: bias, B: B, N: N, scale: 1.0
        )
        for r in 0..<B {
            // Masked positions exactly zero.
            for k in 0..<(N / 2) {
                #expect(out[r * N + k] == 0,
                        "row \(r) masked position \(k) = \(out[r * N + k]), expected exact 0")
            }
            // Unmasked half must sum to ≈ 1 (the masked half's sum is
            // 0, so the row total still satisfies the softmax invariant).
            var s: Double = 0
            for k in (N / 2)..<N { s += Double(out[r * N + k]) }
            #expect(abs(s - 1.0) < 1e-5,
                    "row \(r) unmasked sum \(s) deviates from 1.0")
        }
    }
}

#endif // canImport(Metal)
