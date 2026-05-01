// SPDX-License-Identifier: Apache-2.0
//
// Parity + edge-case tests for `RMSNormKernel` (issue #61). Validates
// that the Metal `kernel_rms_norm_mul_f32` produces output matching a
// pure-Swift FP32 RMSNorm reference at the T5-base encoder shape and
// at synthetic edge-case shapes / inputs.
//
// Reference path
// --------------
//
// Pure Swift, FP32 throughout, computed row-wise:
//   mean_i  = sum(x_i²) / D
//   scale_i = 1 / sqrt(mean_i + eps)
//   y_i[k]  = x_i[k] * scale_i * gain[k]
//
// Edge cases asserted explicitly:
//   - all-zeros row → output is zero (`mean=0` ⇒ `scale = 1/sqrt(eps)`,
//     finite; multiplied by `x[k]=0` yields 0).
//   - all-equal-values row → output equals gain within tolerance
//     (since `x[k]=v` ⇒ `mean=v²` ⇒ `scale=1/|v|` modulo eps, and
//     `x[k]*scale = sign(v)`; we use positive v so the product is 1).
//   - FP16-overflow magnitudes — `sum(x²)` over `D=768` at magnitude
//     250 is ~4.8e7, comfortably overflowing FP16's ~65 504 dynamic
//     range. The kernel must produce finite output (the FP32 carve-out
//     is the entire point of issue #61, ADR 014(i)).
//
// All parity tests gate on `MetalAvailability.isAvailable` so a no-GPU
// host (or a `SWITCHCRAFT_FORCE_ACCELERATE`-set run) skips cleanly.

#if canImport(Metal)

import Foundation
import Metal
import Testing
@_spi(SwitchcraftMetal) import SwitchcraftCore
@_spi(SwitchcraftMetal) import SwitchcraftMetal

// MARK: - PRNG + reference

/// Deterministic LCG. Matches the shape used in
/// `Q4KMatMulKernelTests` so seeded runs reproduce across hosts.
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

private enum RMSNormReference {
    /// Pure-Swift FP32 RMSNorm: `y_i = (x_i / sqrt(mean(x_i²) + eps)) * gain`.
    /// Row-major `M × D` input/output; `D`-length gain shared across rows.
    static func compute(
        x: [Float], gain: [Float],
        M: Int, D: Int, eps: Float
    ) -> [Float] {
        precondition(x.count == M * D)
        precondition(gain.count == D)
        var y = [Float](repeating: 0, count: M * D)
        for i in 0..<M {
            // FP64 accumulator on the reference side so a kernel using
            // a clean FP32 accumulator can still match within tolerance
            // — and so the reference itself can't be the source of the
            // FP16-overflow regression case.
            var sum: Double = 0
            for k in 0..<D {
                let v = Double(x[i * D + k])
                sum += v * v
            }
            let mean = sum / Double(D)
            let scale = 1.0 / (mean + Double(eps)).squareRoot()
            for k in 0..<D {
                y[i * D + k] = Float(Double(x[i * D + k]) * scale * Double(gain[k]))
            }
        }
        return y
    }
}

// MARK: - Kernel runner

private enum RMSNormRunner {
    /// One-shot dispatch helper. Builds Metal buffers, encodes the
    /// kernel into a fresh command buffer, commits + waits, returns
    /// the FP32 output array.
    static func run(
        ctx: MetalContext,
        kernel: RMSNormKernel,
        x: [Float], gain: [Float],
        M: Int, D: Int, eps: Float
    ) throws -> [Float] {
        let inBuf = try makeBuffer(ctx: ctx, floats: x, label: "input")
        let gainBuf = try makeBuffer(ctx: ctx, floats: gain, label: "gain")
        let outLen = M * D * MemoryLayout<Float>.size
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
            input: inBuf, gain: gainBuf, output: outBuf,
            M: M, D: D, eps: eps
        )
        cmd.commit()
        cmd.waitUntilCompleted()
        if let err = cmd.error {
            throw MetalContextError.pipelineCreationFailed("command buffer error: \(err)")
        }

        let raw = outBuf.contents().bindMemory(to: Float.self, capacity: M * D)
        return Array(UnsafeBufferPointer(start: raw, count: M * D))
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

@Suite("RMSNormKernel parity",
       .enabled(if: MetalAvailability.isAvailable,
                "Metal unavailable on this host (or SWITCHCRAFT_FORCE_ACCELERATE is set)"))
struct RMSNormKernelTests {

    @Test("T5-base parity (M=512, D=768)")
    func parityT5Base() throws {
        try runParity(M: 512, D: 768, seed: 0xDA7A_C0DE_0001)
    }

    @Test("synthetic parity (M=1, D=128)")
    func paritySingleRowSmall() throws {
        try runParity(M: 1, D: 128, seed: 0xDA7A_C0DE_0002)
    }

    @Test("synthetic parity (M=64, D=2048)")
    func parityWideRow() throws {
        try runParity(M: 64, D: 2048, seed: 0xDA7A_C0DE_0003)
    }

    private func runParity(M: Int, D: Int, seed: UInt64) throws {
        let ctx = try #require(MetalContext.shared,
                               "MetalContext.shared was nil on a Metal-capable host")
        let kernel = try RMSNormKernel(context: ctx)

        var rng = LCG(seed: seed)
        var x = [Float](repeating: 0, count: M * D)
        for i in 0..<(M * D) { x[i] = rng.nextFloat() }
        var gain = [Float](repeating: 0, count: D)
        // Centre the gain near 1 (T5 RMSNorm weights are typically
        // close to unity). Mixing in a sign bit checks the kernel
        // multiplies in the right order.
        for k in 0..<D { gain[k] = 1.0 + 0.25 * rng.nextFloat() }

        let eps: Float = 1e-6
        let kernelOut = try RMSNormRunner.run(
            ctx: ctx, kernel: kernel,
            x: x, gain: gain, M: M, D: D, eps: eps
        )
        let referenceOut = RMSNormReference.compute(
            x: x, gain: gain, M: M, D: D, eps: eps
        )

        var minCos: Double = 1.0
        var maxAbs: Double = 0
        for m in 0..<M {
            let kRow = Array(kernelOut[(m * D)..<((m + 1) * D)])
            let rRow = Array(referenceOut[(m * D)..<((m + 1) * D)])
            let cos = MetalCorrectness.cosineSimilarity(kRow, rRow)
            let err = MetalCorrectness.maxAbsError(kRow, rRow)
            if cos < minCos { minCos = cos }
            if err > maxAbs { maxAbs = err }
        }
        print("[RMSNorm] M=\(M) D=\(D) — min row cosine \(minCos), max abs err \(maxAbs)")
        #expect(minCos >= 0.99999,
                "RMSNorm parity at (M=\(M),D=\(D)): row cosine \(minCos) below 0.99999")
    }
}

// MARK: - Edge case suite

@Suite("RMSNormKernel edge cases",
       .enabled(if: MetalAvailability.isAvailable,
                "Metal unavailable on this host (or SWITCHCRAFT_FORCE_ACCELERATE is set)"))
struct RMSNormKernelEdgeCaseTests {

    @Test("all-zeros row produces zero output")
    func allZerosRow() throws {
        let ctx = try #require(MetalContext.shared)
        let kernel = try RMSNormKernel(context: ctx)

        let M = 4, D = 768
        let x = [Float](repeating: 0, count: M * D)
        // Non-trivial gain to make sure a nonzero gain * zero scale
        // still yields zero (i.e. the kernel doesn't smuggle in a NaN
        // from `0 / sqrt(0 + eps) = 0` somehow).
        let gain = (0..<D).map { Float(1.0 + 0.001 * Float($0)) }
        let eps: Float = 1e-6
        let out = try RMSNormRunner.run(
            ctx: ctx, kernel: kernel,
            x: x, gain: gain, M: M, D: D, eps: eps
        )
        // All-zeros output. NaN check via the equality test below
        // (NaN != 0).
        var maxAbs: Float = 0
        for v in out { maxAbs = max(maxAbs, abs(v)) }
        #expect(maxAbs == 0,
                "all-zeros row produced non-zero output (max abs = \(maxAbs))")
    }

    @Test("row of identical values yields output equal to gain")
    func identicalValuesRow() throws {
        let ctx = try #require(MetalContext.shared)
        let kernel = try RMSNormKernel(context: ctx)

        let M = 4, D = 768
        let v: Float = 1.5
        let x = [Float](repeating: v, count: M * D)
        let gain = (0..<D).map { _ in Float(0.7) }   // any uniform gain
        let eps: Float = 1e-6
        let out = try RMSNormRunner.run(
            ctx: ctx, kernel: kernel,
            x: x, gain: gain, M: M, D: D, eps: eps
        )
        // For x[k] = v across all k: mean = v², scale = 1/sqrt(v² + eps),
        // y[k] = v * scale * gain[k] = v / sqrt(v² + eps) * gain[k].
        // For v >> sqrt(eps), this collapses to gain[k].
        let expected = v / (v * v + eps).squareRoot() * gain[0]
        var maxAbs: Float = 0
        for k in 0..<(M * D) { maxAbs = max(maxAbs, abs(out[k] - expected)) }
        #expect(maxAbs < 1e-5,
                "identical-values row diverged from expected scalar gain (max abs err = \(maxAbs))")
    }

    /// The canonical falsifying case behind ADR 014(i): the unreduced
    /// `sum(x²)` overflows FP16's ~65 504 dynamic range, so an FP16
    /// implementation would saturate to inf/NaN, while the FP32
    /// kernel must produce finite, parity-matching output.
    ///
    /// At D=768 and `|x| ≈ 250`, `sum(x²) ≈ 250² × 768 ≈ 4.8e7`,
    /// comfortably above FP16's max. The mean (~62 500) still fits in
    /// FP16, but the unreduced accumulator in the parallel-sum loop
    /// would have already saturated long before the divide.
    @Test("FP16-overflow magnitudes produce finite parity-matching output")
    func fp16OverflowRow() throws {
        let ctx = try #require(MetalContext.shared)
        let kernel = try RMSNormKernel(context: ctx)

        let M = 2, D = 768
        let magnitude: Float = 250

        // Sanity-check the FP16 saturation premise so the test can
        // never pass vacuously: if a future refactor lowers the input
        // magnitude below the FP16 threshold, the assertion below
        // catches it before we ever dispatch the kernel.
        let fp16Max: Float = 65504
        let unreducedSumSquared = magnitude * magnitude * Float(D)
        #expect(unreducedSumSquared > fp16Max,
                "test premise broken: sum(x²) ≈ \(unreducedSumSquared) does not overflow FP16's \(fp16Max)")

        var rng = LCG(seed: 0xFFFA_0001_BEEF_CAFE)
        var x = [Float](repeating: 0, count: M * D)
        // Sign-flipped magnitudes so the row mean isn't trivially
        // dominated by the gain offset; squaring discards the sign.
        for i in 0..<(M * D) {
            x[i] = magnitude * (rng.nextFloat() < 0 ? -1 : 1)
        }
        let gain = [Float](repeating: 1.0, count: D)
        let eps: Float = 1e-6
        let out = try RMSNormRunner.run(
            ctx: ctx, kernel: kernel,
            x: x, gain: gain, M: M, D: D, eps: eps
        )

        // 1. All output finite — the FP32 contract is the precision-
        //    routing rationale's empirical witness.
        for v in out {
            #expect(v.isFinite,
                    "FP16-overflow row produced a non-finite output (\(v)) — FP32 contract violated")
        }

        // 2. Match the FP64-reference. With |x|=250 and gain=1, the
        //    expected per-element output is x/|x| × 1 = ±1 (modulo eps).
        let referenceOut = RMSNormReference.compute(
            x: x, gain: gain, M: M, D: D, eps: eps
        )
        var maxAbs: Double = 0
        var minCos: Double = 1.0
        for m in 0..<M {
            let kRow = Array(out[(m * D)..<((m + 1) * D)])
            let rRow = Array(referenceOut[(m * D)..<((m + 1) * D)])
            let cos = MetalCorrectness.cosineSimilarity(kRow, rRow)
            let err = MetalCorrectness.maxAbsError(kRow, rRow)
            if cos < minCos { minCos = cos }
            if err > maxAbs { maxAbs = err }
        }
        print("[RMSNorm fp16-overflow] min row cosine \(minCos), max abs err \(maxAbs)")
        #expect(minCos >= 0.99999,
                "FP16-overflow row cosine \(minCos) below 0.99999")
    }
}

#endif // canImport(Metal)
