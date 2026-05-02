// SPDX-License-Identifier: Apache-2.0
//
// Parity + edge-case tests for `Q4KMatMulKernel` (issue #60). Validates
// that the Metal `kernel_mul_mm_q4_K_f32` produces output matching a
// CPU reference (`Q4KDecode.dequantise` + `cblas_sgemm`) on the four
// xtr-base-en (dFF=2048) Q4_K-stored shapes plus M=1 and a non-tile-
// aligned N. NOTE: classic T5-base uses dFF=3072; xtr-base-en is T5 v1.1
// with dFF=2048 — prior tests used the wrong shape (issue #75 gap fix).
//
// Reference path
// --------------
//
// For each shape (M, K, N):
//   1. Build a row-major `[N, K]` weight in raw Q4_K bytes by stacking
//      N rows × (K/256) randomly-seeded super-blocks via the existing
//      `Q4KDecodeTests.makeBlock(...)` helper from #59. This avoids
//      porting ggml's quantiser (Q5/Q6 resolution).
//   2. Dequantise the same bytes via `Q4KDecode.dequantise(...)` to
//      get the reference `[N, K]` FP32 weight matrix.
//   3. Generate a deterministic-seeded `[M, K]` FP32 activation matrix.
//   4. Compute the CPU reference `C_ref = A @ W^T` via `cblas_sgemm`,
//      shape `[M, N]`.
//   5. Encode the kernel against the raw Q4_K bytes + FP32 activations
//      and read back the kernel's output `C_kernel` (shape `[M, N]`).
//   6. Assert `cosineSimilarity(C_kernel.row(m), C_ref.row(m)) >= 0.99999`
//      for every row, and print the row-wise max-abs-error for
//      ratchet visibility.
//
// All parity tests gate on `MetalAvailability.isAvailable` so a no-GPU
// host (or a `SWITCHCRAFT_FORCE_ACCELERATE`-set run) skips cleanly.

#if canImport(Metal)

import Accelerate
import Foundation
import Metal
import Testing
@_spi(SwitchcraftMetal) import SwitchcraftCore
@_spi(SwitchcraftMetal) import SwitchcraftMetal

// MARK: - Test inputs

/// Deterministic linear-congruential PRNG. Test-internal so seeded runs
/// are reproducible across machines / CI without depending on Swift's
/// `SystemRandomNumberGenerator`.
private struct LCG {
    var state: UInt64
    init(seed: UInt64) { self.state = seed &* 0x9E3779B97F4A7C15 &+ 1 }

    mutating func nextU32() -> UInt32 {
        state = state &* 6364136223846793005 &+ 1442695040888963407
        return UInt32(truncatingIfNeeded: state >> 32)
    }

    /// Uniform `[0, bound)` integer.
    mutating func nextInt(below bound: Int) -> Int {
        precondition(bound > 0)
        return Int(nextU32() % UInt32(bound))
    }

    /// Centred float in `[-1, 1)`.
    mutating func nextFloat() -> Float {
        let f = Float(nextU32()) / Float(UInt32.max)
        return f * 2 - 1
    }
}

private enum Q4KMatMulFixtures {
    /// Build N×(K/256) random Q4_K super-blocks concatenated into a
    /// single `[N, K]`-shaped weight buffer. Each row's super-block
    /// header (`d`, `dmin`) is randomised so any cross-row addressing
    /// bug is visible. Returns the raw bytes plus the dequantised FP32
    /// reference.
    ///
    /// `seed` controls all randomness so a failing test on a CI host
    /// can be reproduced locally.
    static func buildWeight(N: Int, K: Int, seed: UInt64) -> (bytes: Data, dequant: [Float]) {
        precondition(K % 256 == 0, "K must be a multiple of 256")
        let blocksPerRow = K / 256
        var rng = LCG(seed: seed)
        var bytes = Data()
        bytes.reserveCapacity(N * blocksPerRow * 144)

        for _ in 0..<N {
            for _ in 0..<blocksPerRow {
                // Centre `d` and `dmin` near 1e-2..1e-1 so dequantised
                // weights span a useful magnitude (~ 0.0..1.5) without
                // overflowing FP16 storage. Q4_K's dequant formula is
                // `d * sub_scale * nibble - dmin * sub_min`.
                let dRaw  = 0.005 + abs(rng.nextFloat()) * 0.05
                let dmRaw = 0.001 + abs(rng.nextFloat()) * 0.02
                let d  = Float16(dRaw)
                let dm = Float16(dmRaw)

                var subScales = [UInt8](repeating: 0, count: 8)
                var subMins   = [UInt8](repeating: 0, count: 8)
                for i in 0..<8 {
                    subScales[i] = UInt8(rng.nextInt(below: 64))
                    subMins[i]   = UInt8(rng.nextInt(below: 64))
                }

                var weightsLow  = [UInt8](repeating: 0, count: 128)
                var weightsHigh = [UInt8](repeating: 0, count: 128)
                for i in 0..<128 {
                    weightsLow[i]  = UInt8(rng.nextInt(below: 16))
                    weightsHigh[i] = UInt8(rng.nextInt(below: 16))
                }

                bytes.append(Q4KDecodeTests.makeBlock(
                    d: d, dmin: dm,
                    subScales: subScales, subMins: subMins,
                    weightsLow: weightsLow, weightsHigh: weightsHigh
                ))
            }
        }

        let dequant = Q4KDecode.dequantise(blocks: bytes)
        precondition(dequant.count == N * K, "dequant size mismatch")
        return (bytes, dequant)
    }

    /// Build an M×K row-major FP32 activation matrix with deterministic
    /// values in `[-1, 1)`.
    static func buildActivation(M: Int, K: Int, seed: UInt64) -> [Float] {
        var rng = LCG(seed: seed)
        var out = [Float](repeating: 0, count: M * K)
        for i in 0..<(M * K) { out[i] = rng.nextFloat() }
        return out
    }
}

// MARK: - Reference matmul

private enum Q4KMatMulReference {
    /// `C = A @ W^T` via `cblas_sgemm`. `A` is `[M, K]` row-major;
    /// `W` is `[N, K]` row-major; `C` is `[M, N]` row-major.
    static func reference(
        activations: [Float],
        weights: [Float],
        M: Int, K: Int, N: Int
    ) -> [Float] {
        precondition(activations.count == M * K)
        precondition(weights.count == N * K)
        var c = [Float](repeating: 0, count: M * N)
        activations.withUnsafeBufferPointer { aPtr in
            weights.withUnsafeBufferPointer { wPtr in
                c.withUnsafeMutableBufferPointer { cPtr in
                    cblas_sgemm(
                        CblasRowMajor,
                        CblasNoTrans, CblasTrans,
                        Int32(M), Int32(N), Int32(K),
                        1.0,
                        aPtr.baseAddress, Int32(K),
                        wPtr.baseAddress, Int32(K),
                        0.0,
                        cPtr.baseAddress, Int32(N)
                    )
                }
            }
        }
        return c
    }
}

// MARK: - Kernel runner

private enum Q4KMatMulRunner {
    /// One-shot dispatch helper. Builds Metal buffers, encodes the
    /// kernel into a fresh command buffer, commits + waits, returns the
    /// FP32 output array.
    static func run(
        ctx: MetalContext,
        kernel: Q4KMatMulKernel,
        weightBytes: Data,
        activations: [Float],
        M: Int, K: Int, N: Int
    ) throws -> [Float] {
        // Weight buffer (raw Q4_K bytes).
        let weightBuf: MTLBuffer = try weightBytes.withUnsafeBytes { raw -> MTLBuffer in
            guard
                let base = raw.baseAddress,
                let buf = ctx.device.makeBuffer(
                    bytes: base,
                    length: raw.count,
                    options: .storageModeShared
                )
            else {
                throw MetalContextError.pipelineCreationFailed("weight buffer alloc returned nil")
            }
            return buf
        }
        // Activation buffer (FP32).
        let actBuf: MTLBuffer = try activations.withUnsafeBufferPointer { ptr -> MTLBuffer in
            guard
                let base = ptr.baseAddress,
                let buf = ctx.device.makeBuffer(
                    bytes: base,
                    length: ptr.count * MemoryLayout<Float>.size,
                    options: .storageModeShared
                )
            else {
                throw MetalContextError.pipelineCreationFailed("activation buffer alloc returned nil")
            }
            return buf
        }
        // Output buffer (FP32). Cleared to zero so a no-op kernel is
        // visibly wrong.
        let outLen = M * N * MemoryLayout<Float>.size
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
            weight: weightBuf,
            weightShape: (N, K),
            input: actBuf,
            output: outBuf,
            M: M
        )
        cmd.commit()
        cmd.waitUntilCompleted()
        if let err = cmd.error {
            throw MetalContextError.pipelineCreationFailed("command buffer error: \(err)")
        }

        let raw = outBuf.contents().bindMemory(to: Float.self, capacity: M * N)
        return Array(UnsafeBufferPointer(start: raw, count: M * N))
    }
}

// MARK: - Suite

@Suite("Q4KMatMulKernel parity",
       .enabled(if: MetalAvailability.isAvailable,
                "Metal unavailable on this host (or SWITCHCRAFT_FORCE_ACCELERATE is set)"))
struct Q4KMatMulKernelTests {

    // xtr-base-en (T5 v1.1) shapes — Q/K/V/O projections share
    // (M=512, K=768, N=768); FFN gate/up share (M=512, K=768, N=2048);
    // FFN down is (M=512, K=2048, N=768).
    // NOTE: dFF=2048 is the xtr-base-en / T5 v1.1 value; classic T5-base
    // uses dFF=3072. The prior N=3072 tests were the wrong shape and gave
    // no coverage for the actual asset (issue #75 coverage gap).

    @Test("qkv_proj parity (M=512, K=768, N=768)")
    func parityQKVProj() throws {
        try runParity(M: 512, K: 768, N: 768, seed: 0xDA7A_C0DE_0001)
    }

    @Test("ffn_wi parity (M=512, K=768, N=2048)")
    func parityFFNWi() throws {
        try runParity(M: 512, K: 768, N: 2048, seed: 0xDA7A_C0DE_0002)
    }

    @Test("ffn_wo parity (M=512, K=2048, N=768)")
    func parityFFNWo() throws {
        try runParity(M: 512, K: 2048, N: 768, seed: 0xDA7A_C0DE_0003)
    }

    // MARK: - Helper

    private func runParity(M: Int, K: Int, N: Int, seed: UInt64) throws {
        let ctx = try #require(MetalContext.shared,
                               "MetalContext.shared was nil on a Metal-capable host")
        let kernel = try Q4KMatMulKernel(context: ctx)

        let (weightBytes, weightDequant) = Q4KMatMulFixtures.buildWeight(
            N: N, K: K, seed: seed
        )
        let activations = Q4KMatMulFixtures.buildActivation(
            M: M, K: K, seed: seed ^ 0xA5A5_5A5A_A5A5_5A5A
        )

        let kernelOut = try Q4KMatMulRunner.run(
            ctx: ctx, kernel: kernel,
            weightBytes: weightBytes, activations: activations,
            M: M, K: K, N: N
        )
        let referenceOut = Q4KMatMulReference.reference(
            activations: activations, weights: weightDequant,
            M: M, K: K, N: N
        )

        var minCos: Double = 1.0
        var maxAbs: Double = 0.0
        for m in 0..<M {
            let kernelRow = Array(kernelOut[(m * N)..<((m + 1) * N)])
            let refRow    = Array(referenceOut[(m * N)..<((m + 1) * N)])
            let cos = MetalCorrectness.cosineSimilarity(kernelRow, refRow)
            let err = MetalCorrectness.maxAbsError(kernelRow, refRow)
            if cos < minCos { minCos = cos }
            if err > maxAbs { maxAbs = err }
        }

        // Surface ratchet metrics on every run so a near-miss is visible
        // even when the test passes.
        print("[Q4KMatMul] M=\(M) K=\(K) N=\(N) — min row cosine \(minCos), max abs err \(maxAbs)")

        #expect(minCos >= 0.99999,
                "Q4KMatMul parity for (M=\(M),K=\(K),N=\(N)): row cosine \(minCos) below 0.99999")
    }
}

// MARK: - Edge cases (separate suite for clarity)

@Suite("Q4KMatMulKernel edge cases",
       .enabled(if: MetalAvailability.isAvailable,
                "Metal unavailable on this host (or SWITCHCRAFT_FORCE_ACCELERATE is set)"))
struct Q4KMatMulKernelEdgeCaseTests {

    @Test("M=1 routes through boundary path with cosine ≥ 0.99999")
    func edgeMEqualsOne() throws {
        let ctx = try #require(MetalContext.shared)
        let kernel = try Q4KMatMulKernel(context: ctx)

        let M = 1, K = 768, N = 768
        let (weightBytes, weightDequant) = Q4KMatMulFixtures.buildWeight(
            N: N, K: K, seed: 0xED9E_001
        )
        let activations = Q4KMatMulFixtures.buildActivation(
            M: M, K: K, seed: 0xED9E_002
        )

        let kernelOut = try Q4KMatMulRunner.run(
            ctx: ctx, kernel: kernel,
            weightBytes: weightBytes, activations: activations,
            M: M, K: K, N: N
        )
        let referenceOut = Q4KMatMulReference.reference(
            activations: activations, weights: weightDequant,
            M: M, K: K, N: N
        )
        let cos = MetalCorrectness.cosineSimilarity(kernelOut, referenceOut)
        let err = MetalCorrectness.maxAbsError(kernelOut, referenceOut)
        print("[Q4KMatMul edge M=1] cosine \(cos), max abs err \(err)")
        #expect(cos >= 0.99999,
                "M=1 cosine \(cos) below 0.99999")
    }

    @Test("non-tile-aligned N exercises bc_out boundary path")
    func edgeNNonTileAligned() throws {
        let ctx = try #require(MetalContext.shared)
        let kernel = try Q4KMatMulKernel(context: ctx)

        // N=33 is not a multiple of NR0=64 — forces the staged-write
        // path in the kernel's PHASE 3.
        let M = 512, K = 768, N = 33
        let (weightBytes, weightDequant) = Q4KMatMulFixtures.buildWeight(
            N: N, K: K, seed: 0xED9E_003
        )
        let activations = Q4KMatMulFixtures.buildActivation(
            M: M, K: K, seed: 0xED9E_004
        )

        let kernelOut = try Q4KMatMulRunner.run(
            ctx: ctx, kernel: kernel,
            weightBytes: weightBytes, activations: activations,
            M: M, K: K, N: N
        )
        let referenceOut = Q4KMatMulReference.reference(
            activations: activations, weights: weightDequant,
            M: M, K: K, N: N
        )

        var minCos: Double = 1.0
        var maxAbs: Double = 0.0
        for m in 0..<M {
            let kernelRow = Array(kernelOut[(m * N)..<((m + 1) * N)])
            let refRow    = Array(referenceOut[(m * N)..<((m + 1) * N)])
            let cos = MetalCorrectness.cosineSimilarity(kernelRow, refRow)
            let err = MetalCorrectness.maxAbsError(kernelRow, refRow)
            if cos < minCos { minCos = cos }
            if err > maxAbs { maxAbs = err }
        }
        print("[Q4KMatMul edge N=33] min row cosine \(minCos), max abs err \(maxAbs)")
        #expect(minCos >= 0.99999,
                "non-tile-aligned N=33 row cosine \(minCos) below 0.99999")
    }

    /// The kernel's register-tile dequantiser (`dequantize_q4_K(xb, il,
    /// thread half4x4 &)`) traverses Q4_K bytes in a different order
    /// than `Q4KDecode.dequantise` (whole-row form) but must produce
    /// the same FP32 output. We can't call the kernel-internal
    /// dequant directly, but we can dispatch the matmul with an
    /// identity activation matrix and recover the per-row dequantised
    /// weight: `out[m, n] = sum_k I[m, k] * W[n, k] = W[n, m]` when
    /// `I[m, k] = 1 if m==k else 0` and M == K.
    ///
    /// This is a transpose of the dequantised weight; comparing
    /// element-wise against `Q4KDecode.dequantise` confirms both
    /// dequant traversals agree on a fixed canonical super-block.
    @Test("kernel inline dequant matches Q4KDecode.dequantise on a canonical super-block")
    func dequantFormAgreement() throws {
        let ctx = try #require(MetalContext.shared)
        let kernel = try Q4KMatMulKernel(context: ctx)

        // K must be a multiple of 256 (one super-block) and we want
        // M == K so an identity activation recovers the weight. So
        // M = K = 256, N pinned to a tile-aligned 64.
        let M = 256, K = 256, N = 64
        let (weightBytes, weightDequant) = Q4KMatMulFixtures.buildWeight(
            N: N, K: K, seed: 0xCAFE_BABE_DEAD_BEEF
        )

        // Identity [M, K] = [256, 256]: one-hot per row.
        var I = [Float](repeating: 0, count: M * K)
        for i in 0..<M { I[i * K + i] = 1.0 }

        let kernelOut = try Q4KMatMulRunner.run(
            ctx: ctx, kernel: kernel,
            weightBytes: weightBytes, activations: I,
            M: M, K: K, N: N
        )
        // kernelOut[m, n] should equal weightDequant[n, m] (transpose),
        // modulo the kernel's FP16 staging precision (the dequantised
        // weight tile is materialised as `half4x4` then stored to `half`
        // shmem before the simdgroup multiply). Compute both an absolute
        // error and a relative error so the assertion handles wide
        // dequantised magnitudes (e.g. `d ≈ 0.05 × sub_scale ≈ 63 ×
        // nibble ≈ 15` produces values up to ~47 with FP16 ULPs around
        // 0.05).
        var maxAbs: Double = 0
        var maxRel: Double = 0
        for n in 0..<N {
            for m in 0..<M {
                let got = kernelOut[m * N + n]
                let expected = weightDequant[n * K + m]
                let absErr = abs(Double(got) - Double(expected))
                if absErr > maxAbs { maxAbs = absErr }
                let denom = max(abs(Double(expected)), 1.0)  // avoid /0
                let relErr = absErr / denom
                if relErr > maxRel { maxRel = relErr }
            }
        }
        // Bound consistent with FP16 representation of the dequant
        // result range we synthesised in `buildWeight` (peak ~47):
        // half ULP at that magnitude ≈ 47 × 2^-10 ≈ 0.046, so 0.1 max
        // abs error and 1e-3 max rel error both comfortably accommodate
        // FP16 staging without masking real per-byte traversal bugs.
        print("[Q4KMatMul dequant agreement] max abs err \(maxAbs), max rel err \(maxRel)")
        #expect(maxAbs < 0.1,
                "kernel inline dequant abs error \(maxAbs) exceeds FP16-staging tolerance")
        #expect(maxRel < 1e-3,
                "kernel inline dequant rel error \(maxRel) exceeds 1e-3")
    }
}

#endif // canImport(Metal)
