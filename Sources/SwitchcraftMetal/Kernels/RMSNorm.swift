// SPDX-License-Identifier: Apache-2.0
//
// Swift host wrapper for the gain-fused RMSNorm Metal kernel
// (`kernel_rms_norm_mul_f32` — see
// `Sources/SwitchcraftMetal/Shaders/RMSNorm.metal`). Issue #61.
//
// Public surface intentionally minimal:
//   - `init(context:)` resolves the pipeline state once and caches it.
//   - `encode(commandBuffer:input:gain:output:M:D:eps:)` records a
//     single dispatch into the caller-supplied command buffer; no
//     commit, no wait. Orchestration / batching is the
//     T5MetalEmbedder's responsibility (issue #65).
//
// `@_spi(SwitchcraftMetal) public` — visible to the SwitchcraftMetal
// target's tests and to the future T5MetalEmbedder (also in
// `SwitchcraftMetal`) but not part of the stable Switchcraft public
// API surface (ADR 009 / ADR 016).

#if canImport(Metal)

import Foundation
import Metal
@_spi(SwitchcraftMetal) import SwitchcraftCore

/// Single-dispatch encoder for `kernel_rms_norm_mul_f32`.
///
/// Computes `y = (x / sqrt(mean(x²) + eps)) * gain` row-by-row over a
/// row-major `M × D` FP32 input. `gain` is a `D`-length FP32 vector
/// shared across all rows. Output is FP32 row-major `M × D`.
///
/// Precision contract: FP32 throughout (per ADR 014(b)+(i) — the FP16
/// saturation point in `mean(x²)` for T5 encoder activations is the
/// canonical falsifying case behind the precision-routing decision).
@_spi(SwitchcraftMetal)
public struct RMSNormKernel {
    /// Function name in MSL — matches the upstream gain-fused
    /// specialisation's `[[host_name(...)]]`. The issue body's
    /// `kernel_rms_norm_f32` shorthand referred to the upstream symbol
    /// family; the F==2 (gain-fused) instantiation that this wrapper
    /// dispatches is named `kernel_rms_norm_mul_f32` upstream. See
    /// `RMSNorm.metal` §"Deviations from upstream" point 5.
    public static let kernelFunctionName = "kernel_rms_norm_mul_f32"

    /// Maximum threads per threadgroup. Apple Silicon's max is 1024;
    /// the kernel's two-stage simdgroup reduction also requires the
    /// number of simdgroups (`tptg / simdwidth`) to fit in the first
    /// simdgroup so `simd_sum(shmem[tiisg])` covers them all. With
    /// simdwidth=32 that bounds `tptg ≤ 32 * 32 = 1024` regardless.
    public static let maxThreadsPerThreadgroup: Int = 1024

    /// SIMD-group width on Apple Silicon GPUs.
    public static let simdWidth: Int = 32

    /// Threadgroup memory size in bytes — one float per simdgroup
    /// slot. The kernel zeroes `shmem_f32[0..simdWidth)` in the first
    /// simdgroup and writes `shmem_f32[sgitg]` in each subsequent
    /// simdgroup, so we need at least `simdWidth * sizeof(float)`
    /// bytes (= 128) regardless of the chosen `tptg`.
    public static let threadgroupMemoryBytes: Int = simdWidth * MemoryLayout<Float>.size

    private let context: MetalContext
    private let pipeline: MTLComputePipelineState

    public init(context: MetalContext) throws {
        // Trap immediately if the Swift mirror struct's layout drifts
        // from the MSL `ggml_metal_kargs_norm` definition. The MSL side
        // has a `static_assert(sizeof(...) == 144)` for the same reason;
        // the Swift side needs an explicit check because layout drift
        // would otherwise surface as silent garbage outputs / failed
        // parity tests rather than a clear init-time error.
        precondition(
            MemoryLayout<RMSNormArgs>.size == 144,
            "RMSNormArgs layout drifted from MSL ggml_metal_kargs_norm "
            + "(expected 144 bytes, got \(MemoryLayout<RMSNormArgs>.size))"
        )
        try registerSwitchcraftMetalShaders(with: context)
        self.context = context
        self.pipeline = try context.pipeline(for: Self.kernelFunctionName)
    }

    /// Encode a single RMSNorm dispatch into `commandBuffer`. Does not
    /// commit or wait; the caller (typically `T5MetalEmbedder`, issue
    /// #65) batches dispatches into a per-encoder command buffer and
    /// commits once.
    ///
    /// - Parameters:
    ///   - commandBuffer: the command buffer to encode into.
    ///   - input: FP32 row-major `M × D` activation matrix.
    ///   - gain: FP32 `D`-length gain vector (shared across rows; the
    ///     T5 RMSNorm `weight` parameter).
    ///   - output: FP32 row-major `M × D` output buffer.
    ///   - M: number of rows.
    ///   - D: row length (the inner dimension).
    ///   - eps: RMSNorm epsilon. Must be `>= 0`. `1e-6` is T5's
    ///     default.
    public func encode(
        commandBuffer: MTLCommandBuffer,
        input: MTLBuffer,
        gain: MTLBuffer,
        output: MTLBuffer,
        M: Int,
        D: Int,
        eps: Float
    ) {
        precondition(M > 0, "RMSNorm: M must be positive (got \(M))")
        precondition(D > 0, "RMSNorm: D must be positive (got \(D))")
        precondition(eps >= 0,
                     "RMSNorm: eps must be non-negative (got \(eps))")

        let floatBytes = UInt64(MemoryLayout<Float>.size)
        let rowBytes = UInt64(D) * floatBytes
        let planeBytes = rowBytes * UInt64(M)

        precondition(input.length >= Int(planeBytes),
                     "RMSNorm: input buffer length \(input.length) < required \(planeBytes)")
        precondition(output.length >= Int(planeBytes),
                     "RMSNorm: output buffer length \(output.length) < required \(planeBytes)")
        precondition(gain.length >= Int(rowBytes),
                     "RMSNorm: gain buffer length \(gain.length) < required \(rowBytes)")

        // Build the args struct verbatim per the MSL definition. Field
        // order, types, and array sizes match `ggml_metal_kargs_norm`
        // byte-for-byte (see `RMSNorm.metal` §"verbatim port"). Layout:
        //
        //   [0..0] input x          — single source, never broadcast,
        //                              sizes irrelevant; strides per row
        //                              of M × D.
        //   [..1] gain (f0)         — D-length vector broadcast across
        //                              every row: nef1[1]=1 collapses
        //                              `i01 % 1 = 0`, so all rows hit
        //                              gain[0..D]. nbf1[1]=0 keeps the
        //                              row offset at zero.
        //   [..2] residual (f1)     — UNREAD in F==2; the kernel still
        //                              computes the f1 pointer, so we
        //                              set nef*[2]=1 (avoid div-by-zero
        //                              in `i01 % nef1[2]`) and nbf*[2]=0
        //                              (pointer stays at base + 0).
        //
        // `nef*[i] == 0` is a div-by-zero bug in upstream's broadcast
        // index math; populate every modulus with `1` even where it
        // doesn't logically apply (cheap, defensive).
        var args = RMSNormArgs(
            ne00:    Int32(D),
            ne00_t:  Int32(D),       // scalar T=float instantiation
            nb1:     rowBytes,
            nb2:     planeBytes,
            nb3:     planeBytes,
            eps:     eps,
            nef1_0: Int32(M), nef1_1: 1, nef1_2: 1,
            nef2_0: 1,        nef2_1: 1, nef2_2: 1,
            nef3_0: 1,        nef3_1: 1, nef3_2: 1,
            nbf1_0: rowBytes, nbf1_1: 0, nbf1_2: 0,
            nbf2_0: planeBytes, nbf2_1: 0, nbf2_2: 0,
            nbf3_0: planeBytes, nbf3_1: 0, nbf3_2: 0
        )

        // Threads per threadgroup: round D up to a multiple of 32
        // (simdwidth) and clamp at 1024. The kernel's parallel-sum loop
        // (`for i00 = tpitg.x; i00 < ne00_t; i00 += ntg.x`) handles any
        // tptg correctly — extra threads (D not a multiple of 32) just
        // skip the loop body and contribute 0 to the simdgroup sum.
        let roundedUpD = ((D + Self.simdWidth - 1) / Self.simdWidth) * Self.simdWidth
        let tptg = min(roundedUpD, Self.maxThreadsPerThreadgroup)

        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            preconditionFailure("RMSNorm: failed to create compute command encoder")
        }
        encoder.setComputePipelineState(pipeline)
        encoder.setBytes(&args, length: MemoryLayout<RMSNormArgs>.size, index: 0)
        encoder.setBuffer(input,  offset: 0, index: 1)
        encoder.setBuffer(gain,   offset: 0, index: 2)
        // `src1_1` (residual) is unread in F==2 but still needs a non-
        // null binding. Reuse the gain buffer as a stand-in so we don't
        // require the caller to allocate an unused buffer.
        encoder.setBuffer(gain,   offset: 0, index: 3)
        encoder.setBuffer(output, offset: 0, index: 4)
        encoder.setThreadgroupMemoryLength(Self.threadgroupMemoryBytes, index: 0)

        // One threadgroup per row (i01 = tgpig.x). i02 / i03 are 1 in
        // 2-D layout — broadcast modulus and zero strides keep the
        // f0 / f1 / src0 indexing well-defined regardless.
        encoder.dispatchThreadgroups(
            MTLSize(width: M, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: tptg, height: 1, depth: 1)
        )
        encoder.endEncoding()
    }

    /// Mirrors the MSL `ggml_metal_kargs_norm` byte-for-byte. Field
    /// order is load-bearing — changes must be mirrored in
    /// `RMSNorm.metal`. Inline-arrays-as-individual-fields keeps the
    /// layout obvious across Swift's tuple-vs-array-vs-fixed-size-buffer
    /// storage rules; `MemoryLayout<RMSNormArgs>.size == 144` is
    /// asserted in `init(context:)` so a layout drift traps immediately
    /// with a clear message rather than surfacing as silent garbage.
    struct RMSNormArgs {
        var ne00: Int32       // offset 0
        var ne00_t: Int32     // offset 4
        var nb1: UInt64       // offset 8
        var nb2: UInt64       // offset 16
        var nb3: UInt64       // offset 24
        var eps: Float        // offset 32
        var nef1_0: Int32     // offset 36
        var nef1_1: Int32     // offset 40
        var nef1_2: Int32     // offset 44
        var nef2_0: Int32     // offset 48
        var nef2_1: Int32     // offset 52
        var nef2_2: Int32     // offset 56
        var nef3_0: Int32     // offset 60
        var nef3_1: Int32     // offset 64
        var nef3_2: Int32     // offset 68
        var nbf1_0: UInt64    // offset 72
        var nbf1_1: UInt64    // offset 80
        var nbf1_2: UInt64    // offset 88
        var nbf2_0: UInt64    // offset 96
        var nbf2_1: UInt64    // offset 104
        var nbf2_2: UInt64    // offset 112
        var nbf3_0: UInt64    // offset 120
        var nbf3_1: UInt64    // offset 128
        var nbf3_2: UInt64    // offset 136
        // total: 144
    }
}

#endif // canImport(Metal)
