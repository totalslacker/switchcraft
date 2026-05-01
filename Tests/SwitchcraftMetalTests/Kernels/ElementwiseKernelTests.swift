// SPDX-License-Identifier: Apache-2.0
//
// Parity + edge-case tests for the three element-wise FP32 Metal
// kernels added in issue #63: `ResidualAddKernel` (`kernel_add_f32`),
// `GatedGELUKernel` (`kernel_gated_gelu_f32`), and `L2NormKernel`
// (`kernel_l2_norm_f32`). Each kernel is validated at the xtr-base-en
// shape called out in the issue spec and against an FP64-accumulator
// pure-Swift reference; cosine ≥ 0.99999 is the parity threshold.
//
// All parity / edge-case suites gate on `MetalAvailability.isAvailable`
// so a no-GPU host (or a `SWITCHCRAFT_FORCE_ACCELERATE`-set run) skips
// cleanly.

#if canImport(Metal)

import Foundation
import Metal
import Testing
@_spi(SwitchcraftMetal) import SwitchcraftCore
@_spi(SwitchcraftMetal) import SwitchcraftMetal

// MARK: - PRNG

/// Deterministic LCG. Matches the shape used in
/// `RMSNormKernelTests` so seeded runs reproduce across hosts.
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

// MARK: - Buffer helpers

private enum BufferHelpers {
    static func makeBuffer(
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

    static func makeOutputBuffer(
        ctx: MetalContext, count: Int
    ) throws -> MTLBuffer {
        let len = count * MemoryLayout<Float>.size
        guard let buf = ctx.device.makeBuffer(
            length: len,
            options: .storageModeShared
        ) else {
            throw MetalContextError.pipelineCreationFailed("output buffer alloc returned nil")
        }
        memset(buf.contents(), 0, len)
        return buf
    }

    static func readFloats(_ buf: MTLBuffer, count: Int) -> [Float] {
        let raw = buf.contents().bindMemory(to: Float.self, capacity: count)
        return Array(UnsafeBufferPointer(start: raw, count: count))
    }
}

// MARK: - References

private enum ResidualAddReference {
    static func compute(a: [Float], b: [Float]) -> [Float] {
        precondition(a.count == b.count)
        var out = [Float](repeating: 0, count: a.count)
        for i in 0..<a.count {
            // FP64 accumulator on the reference side — kernel is FP32.
            out[i] = Float(Double(a[i]) + Double(b[i]))
        }
        return out
    }
}

private enum GatedGELUReference {
    /// HuggingFace `gelu_new` in FP64 to match the kernel's FP32 path
    /// within cosine ≥ 0.99999 (RMSNorm/Softmax precedent: FP64
    /// reference vs FP32 kernel is the established parity contract).
    static func geluNew(_ x: Double) -> Double {
        let sqrt2OverPi = (2.0 / Double.pi).squareRoot()
        let inner = sqrt2OverPi * (x + 0.044715 * x * x * x)
        return 0.5 * x * (1.0 + tanh(inner))
    }

    static func compute(gate: [Float], up: [Float]) -> [Float] {
        precondition(gate.count == up.count)
        var out = [Float](repeating: 0, count: gate.count)
        for i in 0..<gate.count {
            let g = geluNew(Double(gate[i]))
            out[i] = Float(g * Double(up[i]))
        }
        return out
    }
}

private enum L2NormReference {
    /// Per-row L2: `y_i = x_i / max(eps, sqrt(Σ x_i²))`.
    /// FP64 accumulators on the reference side so the kernel's clean
    /// FP32 accumulator can still match within tolerance — and so the
    /// reference itself can't be the source of an FP16-overflow
    /// regression case.
    static func compute(
        x: [Float], rows: Int, cols: Int, eps: Float
    ) -> [Float] {
        precondition(x.count == rows * cols)
        var y = [Float](repeating: 0, count: rows * cols)
        for r in 0..<rows {
            var sum: Double = 0
            for k in 0..<cols {
                let v = Double(x[r * cols + k])
                sum += v * v
            }
            let norm = sum.squareRoot()
            let denom = max(Double(eps), norm)
            // For all-zeros rows: denom = eps, scale = 1/eps, finite;
            // multiplied by x[k]=0 yields 0 — matches kernel contract.
            let scale = 1.0 / denom
            for k in 0..<cols {
                y[r * cols + k] = Float(Double(x[r * cols + k]) * scale)
            }
        }
        return y
    }
}

// MARK: - Runners

private enum ResidualAddRunner {
    static func run(
        ctx: MetalContext,
        kernel: ResidualAddKernel,
        a: [Float], b: [Float]
    ) throws -> [Float] {
        precondition(a.count == b.count)
        let aBuf = try BufferHelpers.makeBuffer(ctx: ctx, floats: a, label: "a")
        let bBuf = try BufferHelpers.makeBuffer(ctx: ctx, floats: b, label: "b")
        let outBuf = try BufferHelpers.makeOutputBuffer(ctx: ctx, count: a.count)

        guard let cmd = ctx.queue.makeCommandBuffer() else {
            throw MetalContextError.pipelineCreationFailed("makeCommandBuffer returned nil")
        }
        kernel.encode(commandBuffer: cmd, a: aBuf, b: bBuf, output: outBuf, count: a.count)
        cmd.commit()
        cmd.waitUntilCompleted()
        if let err = cmd.error {
            throw MetalContextError.pipelineCreationFailed("command buffer error: \(err)")
        }
        return BufferHelpers.readFloats(outBuf, count: a.count)
    }
}

private enum GatedGELURunner {
    static func run(
        ctx: MetalContext,
        kernel: GatedGELUKernel,
        gate: [Float], up: [Float]
    ) throws -> [Float] {
        precondition(gate.count == up.count)
        let gBuf = try BufferHelpers.makeBuffer(ctx: ctx, floats: gate, label: "gate")
        let uBuf = try BufferHelpers.makeBuffer(ctx: ctx, floats: up, label: "up")
        let outBuf = try BufferHelpers.makeOutputBuffer(ctx: ctx, count: gate.count)

        guard let cmd = ctx.queue.makeCommandBuffer() else {
            throw MetalContextError.pipelineCreationFailed("makeCommandBuffer returned nil")
        }
        kernel.encode(commandBuffer: cmd, gate: gBuf, up: uBuf, output: outBuf, count: gate.count)
        cmd.commit()
        cmd.waitUntilCompleted()
        if let err = cmd.error {
            throw MetalContextError.pipelineCreationFailed("command buffer error: \(err)")
        }
        return BufferHelpers.readFloats(outBuf, count: gate.count)
    }
}

private enum L2NormRunner {
    static func run(
        ctx: MetalContext,
        kernel: L2NormKernel,
        x: [Float], rows: Int, cols: Int, eps: Float
    ) throws -> [Float] {
        precondition(x.count == rows * cols)
        let inBuf = try BufferHelpers.makeBuffer(ctx: ctx, floats: x, label: "input")
        let outBuf = try BufferHelpers.makeOutputBuffer(ctx: ctx, count: rows * cols)

        guard let cmd = ctx.queue.makeCommandBuffer() else {
            throw MetalContextError.pipelineCreationFailed("makeCommandBuffer returned nil")
        }
        kernel.encode(
            commandBuffer: cmd, input: inBuf, output: outBuf,
            rows: rows, cols: cols, epsilon: eps
        )
        cmd.commit()
        cmd.waitUntilCompleted()
        if let err = cmd.error {
            throw MetalContextError.pipelineCreationFailed("command buffer error: \(err)")
        }
        return BufferHelpers.readFloats(outBuf, count: rows * cols)
    }
}

// MARK: - ResidualAdd suites

@Suite("ResidualAddKernel parity",
       .enabled(if: MetalAvailability.isAvailable,
                "Metal unavailable on this host (or SWITCHCRAFT_FORCE_ACCELERATE is set)"))
struct ResidualAddKernelTests {

    @Test("T5-base parity (M=512, D=768)")
    func parityT5Base() throws {
        let ctx = try #require(MetalContext.shared,
                               "MetalContext.shared was nil on a Metal-capable host")
        let kernel = try ResidualAddKernel(context: ctx)

        let M = 512, D = 768
        let count = M * D
        var rng = LCG(seed: 0xADD0_C0DE_0001)
        var a = [Float](repeating: 0, count: count)
        var b = [Float](repeating: 0, count: count)
        for i in 0..<count {
            a[i] = rng.nextFloat()
            b[i] = rng.nextFloat()
        }

        let kernelOut = try ResidualAddRunner.run(ctx: ctx, kernel: kernel, a: a, b: b)
        let referenceOut = ResidualAddReference.compute(a: a, b: b)

        let cos = MetalCorrectness.cosineSimilarity(kernelOut, referenceOut)
        let err = MetalCorrectness.maxAbsError(kernelOut, referenceOut)
        print("[ResidualAdd] M=\(M) D=\(D) — cosine \(cos), max abs err \(err)")
        #expect(cos >= 0.99999,
                "ResidualAdd parity at (M=\(M),D=\(D)): cosine \(cos) below 0.99999")
    }
}

@Suite("ResidualAddKernel edge cases",
       .enabled(if: MetalAvailability.isAvailable,
                "Metal unavailable on this host (or SWITCHCRAFT_FORCE_ACCELERATE is set)"))
struct ResidualAddKernelEdgeCaseTests {

    @Test("zero + zero produces zero")
    func zeroPlusZero() throws {
        let ctx = try #require(MetalContext.shared)
        let kernel = try ResidualAddKernel(context: ctx)

        let count = 4 * 768
        let a = [Float](repeating: 0, count: count)
        let b = [Float](repeating: 0, count: count)
        let out = try ResidualAddRunner.run(ctx: ctx, kernel: kernel, a: a, b: b)
        var maxAbs: Float = 0
        for v in out { maxAbs = max(maxAbs, abs(v)) }
        #expect(maxAbs == 0,
                "zero+zero produced non-zero output (max abs = \(maxAbs))")
    }

    /// Large-positive + large-negative cancellation. The kernel must
    /// preserve FP32 arithmetic precision: at magnitudes around 1e6,
    /// a near-cancelling pair should yield the exact FP32 sum, not
    /// drift catastrophically.
    @Test("large-positive + large-negative cancellation")
    func largeOppositeMagnitudes() throws {
        let ctx = try #require(MetalContext.shared)
        let kernel = try ResidualAddKernel(context: ctx)

        let count = 4 * 768
        var rng = LCG(seed: 0xADD0_C0DE_0002)
        var a = [Float](repeating: 0, count: count)
        var b = [Float](repeating: 0, count: count)
        // a = +1e6 + small noise; b = -1e6 + small noise; sum ~ noise
        for i in 0..<count {
            let na = rng.nextFloat() * 0.5
            let nb = rng.nextFloat() * 0.5
            a[i] = 1.0e6 + na
            b[i] = -1.0e6 + nb
        }
        let kernelOut = try ResidualAddRunner.run(ctx: ctx, kernel: kernel, a: a, b: b)
        let referenceOut = ResidualAddReference.compute(a: a, b: b)

        // All outputs finite (no NaN / inf from weird codegen).
        for v in kernelOut {
            #expect(v.isFinite,
                    "ResidualAdd produced non-finite output (\(v))")
        }
        let cos = MetalCorrectness.cosineSimilarity(kernelOut, referenceOut)
        let err = MetalCorrectness.maxAbsError(kernelOut, referenceOut)
        print("[ResidualAdd cancellation] cosine \(cos), max abs err \(err)")
        #expect(cos >= 0.99999,
                "ResidualAdd cancellation parity: cosine \(cos) below 0.99999")
    }
}

// MARK: - GatedGELU suites

@Suite("GatedGELUKernel parity",
       .enabled(if: MetalAvailability.isAvailable,
                "Metal unavailable on this host (or SWITCHCRAFT_FORCE_ACCELERATE is set)"))
struct GatedGELUKernelTests {

    /// xtr-base-en FFN intermediate dimension is `d_ff = 2048`
    /// (T5 v1.1, **not** classic T5-base's 3072 — locked by the
    /// catalogue's read of the live `xtr-base-en/config.json`).
    @Test("xtr-base-en FFN parity (M=512, D=2048)")
    func parityFFNIntermediate() throws {
        let ctx = try #require(MetalContext.shared,
                               "MetalContext.shared was nil on a Metal-capable host")
        let kernel = try GatedGELUKernel(context: ctx)

        let M = 512, D = 2048
        let count = M * D
        var rng = LCG(seed: 0xCE10_C0DE_0001)
        var gate = [Float](repeating: 0, count: count)
        var up = [Float](repeating: 0, count: count)
        // T5 FFN activations are roughly unit-scale post-RMSNorm-and-
        // matmul; ±2 covers the typical evaluation range without
        // pushing tanh into either saturation regime.
        for i in 0..<count {
            gate[i] = rng.nextFloat() * 2.0
            up[i] = rng.nextFloat() * 2.0
        }

        let kernelOut = try GatedGELURunner.run(ctx: ctx, kernel: kernel, gate: gate, up: up)
        let referenceOut = GatedGELUReference.compute(gate: gate, up: up)

        let cos = MetalCorrectness.cosineSimilarity(kernelOut, referenceOut)
        let err = MetalCorrectness.maxAbsError(kernelOut, referenceOut)
        print("[GatedGELU] M=\(M) D=\(D) — cosine \(cos), max abs err \(err)")
        #expect(cos >= 0.99999,
                "GatedGELU parity at (M=\(M),D=\(D)): cosine \(cos) below 0.99999")
    }
}

@Suite("GatedGELUKernel edge cases",
       .enabled(if: MetalAvailability.isAvailable,
                "Metal unavailable on this host (or SWITCHCRAFT_FORCE_ACCELERATE is set)"))
struct GatedGELUKernelEdgeCaseTests {

    /// All-negative-large gate ⇒ `gelu_new` saturates near zero,
    /// regardless of `up` sign / magnitude. Output should be near zero.
    @Test("all-negative-large gate yields near-zero output")
    func allNegativeGate() throws {
        let ctx = try #require(MetalContext.shared)
        let kernel = try GatedGELUKernel(context: ctx)

        let count = 4 * 2048
        let gate = [Float](repeating: -10.0, count: count)
        var rng = LCG(seed: 0xCE10_C0DE_0002)
        var up = [Float](repeating: 0, count: count)
        for i in 0..<count { up[i] = rng.nextFloat() * 5.0 }

        let out = try GatedGELURunner.run(ctx: ctx, kernel: kernel, gate: gate, up: up)
        var maxAbs: Float = 0
        for v in out { maxAbs = max(maxAbs, abs(v)) }
        // gelu_new(-10) ≈ -1.5e-22 · ... — astronomically small. The
        // kernel result should be well below 1e-4 once multiplied by
        // bounded up. Use a generous bound to avoid flaking on FP32
        // tanh-saturation differences.
        #expect(maxAbs < 1e-4,
                "all-negative-large gate produced non-trivial output (max abs = \(maxAbs))")
    }

    /// All-positive gate + all-positive up: result must be positive
    /// throughout (sanity for sign-correctness of `gelu_new` * up).
    @Test("all-positive gate × all-positive up yields all-positive")
    func allPositiveBoth() throws {
        let ctx = try #require(MetalContext.shared)
        let kernel = try GatedGELUKernel(context: ctx)

        let count = 4 * 2048
        var rng = LCG(seed: 0xCE10_C0DE_0003)
        var gate = [Float](repeating: 0, count: count)
        var up = [Float](repeating: 0, count: count)
        for i in 0..<count {
            gate[i] = 0.5 + 0.5 * abs(rng.nextFloat())  // [0.5, 1.0)
            up[i] = 0.5 + 0.5 * abs(rng.nextFloat())    // [0.5, 1.0)
        }
        let out = try GatedGELURunner.run(ctx: ctx, kernel: kernel, gate: gate, up: up)
        for v in out {
            #expect(v > 0, "all-positive inputs produced non-positive output (\(v))")
        }
    }

    /// Gate near zero, large up: gate dominates, output near zero.
    /// `gelu_new(0) = 0` exactly in the formula, so output is exactly
    /// zero up to FP32 rounding regardless of up's magnitude.
    @Test("gate near zero with large up yields near-zero output")
    func gateNearZero() throws {
        let ctx = try #require(MetalContext.shared)
        let kernel = try GatedGELUKernel(context: ctx)

        let count = 4 * 2048
        let gate = [Float](repeating: 0.0, count: count)
        var rng = LCG(seed: 0xCE10_C0DE_0004)
        var up = [Float](repeating: 0, count: count)
        for i in 0..<count { up[i] = rng.nextFloat() * 100.0 }

        let out = try GatedGELURunner.run(ctx: ctx, kernel: kernel, gate: gate, up: up)
        var maxAbs: Float = 0
        for v in out { maxAbs = max(maxAbs, abs(v)) }
        #expect(maxAbs == 0,
                "gate=0 should produce exact-zero output regardless of up (max abs = \(maxAbs))")
    }
}

// MARK: - L2Norm suites

@Suite("L2NormKernel parity",
       .enabled(if: MetalAvailability.isAvailable,
                "Metal unavailable on this host (or SWITCHCRAFT_FORCE_ACCELERATE is set)"))
struct L2NormKernelTests {

    /// Post-`2_Dense` projection dimension on xtr-base-en is D=128 —
    /// the L2-norm shape that feeds the `MIN_NORM` filter.
    @Test("xtr-base-en post-projection parity (M=512, D=128)")
    func parityPostProjection() throws {
        try runParity(M: 512, D: 128, seed: 0x12C0_DE00_0001)
    }

    /// Final encoder L2 dimension is D=768 (T5 hidden), used by the
    /// orchestration layer (#64) before the absorbed projection.
    @Test("T5 hidden parity (M=64, D=768)")
    func parityT5Hidden() throws {
        try runParity(M: 64, D: 768, seed: 0x12C0_DE00_0002)
    }

    private func runParity(M: Int, D: Int, seed: UInt64) throws {
        let ctx = try #require(MetalContext.shared,
                               "MetalContext.shared was nil on a Metal-capable host")
        let kernel = try L2NormKernel(context: ctx)

        var rng = LCG(seed: seed)
        var x = [Float](repeating: 0, count: M * D)
        for i in 0..<(M * D) { x[i] = rng.nextFloat() }

        let eps: Float = 1e-12
        let kernelOut = try L2NormRunner.run(
            ctx: ctx, kernel: kernel,
            x: x, rows: M, cols: D, eps: eps
        )
        let referenceOut = L2NormReference.compute(
            x: x, rows: M, cols: D, eps: eps
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
        print("[L2Norm] M=\(M) D=\(D) — min row cosine \(minCos), max abs err \(maxAbs)")
        #expect(minCos >= 0.99999,
                "L2Norm parity at (M=\(M),D=\(D)): row cosine \(minCos) below 0.99999")
    }
}

@Suite("L2NormKernel edge cases",
       .enabled(if: MetalAvailability.isAvailable,
                "Metal unavailable on this host (or SWITCHCRAFT_FORCE_ACCELERATE is set)"))
struct L2NormKernelEdgeCaseTests {

    /// All-zeros row → output must be exactly zero, not NaN. The
    /// `epsilon` floor (`1 / max(eps, √Σ)`) keeps the scale finite at
    /// Σ=0; `0 × finite = 0`. This is the canonical falsifying case
    /// the issue spec calls out — assert exact zero directly rather
    /// than via cosine (cosine of zero-vs-zero is undefined and our
    /// helper returns 1.0 for it, which would mask a NaN).
    @Test("all-zeros row produces exact zero (not NaN)")
    func allZerosRow() throws {
        let ctx = try #require(MetalContext.shared)
        let kernel = try L2NormKernel(context: ctx)

        let M = 4, D = 128
        let x = [Float](repeating: 0, count: M * D)
        let eps: Float = 1e-12
        let out = try L2NormRunner.run(
            ctx: ctx, kernel: kernel,
            x: x, rows: M, cols: D, eps: eps
        )
        // Direct zero check (also catches NaN — NaN != 0).
        var sawNaN = false
        var maxAbs: Float = 0
        for v in out {
            if v.isNaN { sawNaN = true }
            maxAbs = max(maxAbs, abs(v))
        }
        #expect(!sawNaN,
                "all-zeros L2-norm row produced NaN — epsilon floor failed")
        #expect(maxAbs == 0,
                "all-zeros L2-norm row produced non-zero output (max abs = \(maxAbs))")
    }

    /// Single large outlier per row. `sum(x²)` at FP32 magnitudes
    /// like 250² × 128 ≈ 8e6 overflows FP16's ~65 504 dynamic range —
    /// the canonical falsifying case behind ADR 014(b) point 2's
    /// FP32 carve-out for L2-norm. The kernel must produce finite
    /// parity-matching output.
    @Test("FP16-overflow magnitudes produce finite parity-matching output")
    func fp16OverflowRow() throws {
        let ctx = try #require(MetalContext.shared)
        let kernel = try L2NormKernel(context: ctx)

        let M = 2, D = 128
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
        // Sign-flipped magnitudes so squaring still cancels the sign
        // but the row-sum doesn't trivially correlate with row index.
        for i in 0..<(M * D) {
            x[i] = magnitude * (rng.nextFloat() < 0 ? -1 : 1)
        }
        let eps: Float = 1e-12
        let out = try L2NormRunner.run(
            ctx: ctx, kernel: kernel,
            x: x, rows: M, cols: D, eps: eps
        )

        // 1. All output finite — the FP32 contract is the precision-
        //    routing rationale's empirical witness.
        for v in out {
            #expect(v.isFinite,
                    "FP16-overflow L2 row produced a non-finite output (\(v)) — FP32 contract violated")
        }

        // 2. Match the FP64-reference. With |x|=250 and D=128, the
        //    expected per-element output is `±250 / sqrt(250² × 128)`
        //    = `±1/sqrt(128)` ≈ ±0.0884.
        let referenceOut = L2NormReference.compute(
            x: x, rows: M, cols: D, eps: eps
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
        print("[L2Norm fp16-overflow] min row cosine \(minCos), max abs err \(maxAbs)")
        #expect(minCos >= 0.99999,
                "FP16-overflow row cosine \(minCos) below 0.99999")
    }

    /// Already-unit-norm input row: output ≈ input within tolerance.
    @Test("unit-norm input is preserved")
    func unitNormInput() throws {
        let ctx = try #require(MetalContext.shared)
        let kernel = try L2NormKernel(context: ctx)

        let M = 4, D = 128
        var rng = LCG(seed: 0x0017_C0DE_0001)
        var x = [Float](repeating: 0, count: M * D)
        // Build each row, then renormalise to unit-L2 in FP64 so the
        // input is as close to unit-norm as FP32 storage permits.
        for r in 0..<M {
            var raw = [Double](repeating: 0, count: D)
            var sum: Double = 0
            for k in 0..<D {
                let v = Double(rng.nextFloat())
                raw[k] = v
                sum += v * v
            }
            let inv = 1.0 / sum.squareRoot()
            for k in 0..<D {
                x[r * D + k] = Float(raw[k] * inv)
            }
        }

        let eps: Float = 1e-12
        let out = try L2NormRunner.run(
            ctx: ctx, kernel: kernel,
            x: x, rows: M, cols: D, eps: eps
        )

        // Per-row max-abs error: should be tight (kernel is essentially
        // a no-op for unit-norm input modulo FP32 division round-off).
        var maxAbs: Float = 0
        for i in 0..<(M * D) {
            maxAbs = max(maxAbs, abs(out[i] - x[i]))
        }
        #expect(maxAbs < 1e-5,
                "unit-norm input drifted (max abs err = \(maxAbs))")
    }
}

#endif // canImport(Metal)
