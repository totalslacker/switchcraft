// SPDX-License-Identifier: Apache-2.0
//
// Swift host wrapper for the per-row L2 normalisation Metal kernel
// (`kernel_l2_norm_f32` — see
// `Sources/SwitchcraftMetal/Shaders/L2Norm.metal`). Issue #63.
//
// Public surface intentionally minimal:
//   - `init(context:)` resolves the pipeline state once and caches it.
//   - `encode(commandBuffer:input:output:rows:cols:epsilon:)` records
//     a single dispatch into the caller-supplied command buffer; no
//     commit, no wait. Orchestration / batching is the
//     T5MetalEmbedder's responsibility (issue #64).
//
// `@_spi(SwitchcraftMetal) public` — visible to the SwitchcraftMetal
// target's tests and to the future T5MetalEmbedder (also in
// `SwitchcraftMetal`) but not part of the stable Switchcraft public
// API surface (ADR 009 / ADR 016).

#if canImport(Metal)

import Foundation
import Metal
@_spi(SwitchcraftMetal) import SwitchcraftCore

/// Single-dispatch encoder for `kernel_l2_norm_f32`.
///
/// Computes `output[i, k] = input[i, k] / max(epsilon, sqrt(Σ_k
/// input[i, k]²))` row-by-row over a row-major `rows × cols` FP32
/// matrix.
///
/// This kernel applies after the absorbed `2_Dense` projection (issue
/// #64) and before the `MIN_NORM` filter, matching
/// `T5CoreMLEmbedder`'s output contract. The xtr-base-en projection
/// dimension is `D=128`; the final-encoder L2 (also in #64) operates
/// at `D=768`.
///
/// Precision contract: FP32 throughout. **Sensitive op** per ADR
/// 014(b) point 2: `sum(x²)` over 768 elements has overflowed FP16
/// historically. This is the canonical falsifying case behind the FP32
/// carve-out for L2-norm.
///
/// All-zeros row contract: an input row that is identically zero
/// produces an identically-zero output (not NaN). The `epsilon` floor
/// on the denominator (`1 / max(epsilon, √Σ)`) ensures the scale is
/// finite at Σ=0; `0 × finite = 0`.
@_spi(SwitchcraftMetal)
public struct L2NormKernel {
    /// Function name in MSL.
    public static let kernelFunctionName = "kernel_l2_norm_f32"

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
        precondition(
            MemoryLayout<L2NormArgs>.size == 24,
            "L2NormArgs layout drifted from MSL L2NormArgs "
            + "(expected 24 bytes, got \(MemoryLayout<L2NormArgs>.size))"
        )
        try registerSwitchcraftMetalShaders(with: context)
        self.context = context
        self.pipeline = try context.pipeline(for: Self.kernelFunctionName)
    }

    /// Encode a single L2-norm dispatch into `commandBuffer`. Does not
    /// commit or wait; the caller (typically `T5MetalEmbedder`, issue
    /// #64) batches dispatches into a per-encoder command buffer and
    /// commits once.
    ///
    /// - Parameters:
    ///   - commandBuffer: the command buffer to encode into.
    ///   - input: FP32 row-major `rows × cols` input matrix.
    ///   - output: FP32 row-major `rows × cols` output buffer. May
    ///     alias `input` for in-place normalisation; the kernel reads
    ///     the entire row before writing back, so a per-row alias is
    ///     safe with the threadgroup barriers in place.
    ///   - rows: number of rows (one threadgroup per row).
    ///   - cols: row length (the reduction inner dim).
    ///   - epsilon: floor on the denominator (`1 / max(eps, √Σ)`).
    ///     Must be `> 0` so the all-zeros row produces zero rather
    ///     than `inf`/`NaN`. `1e-12` is a reasonable default for FP32
    ///     residual streams.
    public func encode(
        commandBuffer: MTLCommandBuffer,
        input: MTLBuffer,
        output: MTLBuffer,
        rows: Int,
        cols: Int,
        epsilon: Float
    ) {
        precondition(rows > 0, "L2Norm: rows must be positive (got \(rows))")
        precondition(cols > 0, "L2Norm: cols must be positive (got \(cols))")
        precondition(epsilon > 0,
                     "L2Norm: epsilon must be positive to keep all-zeros rows finite (got \(epsilon))")

        let floatBytes = UInt64(MemoryLayout<Float>.size)
        let rowBytes = UInt64(cols) * floatBytes
        let planeBytes = rowBytes * UInt64(rows)

        precondition(input.length >= Int(planeBytes),
                     "L2Norm: input buffer length \(input.length) < required \(planeBytes)")
        precondition(output.length >= Int(planeBytes),
                     "L2Norm: output buffer length \(output.length) < required \(planeBytes)")

        var args = L2NormArgs(
            cols: Int32(cols),
            _pad0: 0,
            row_pitch_bytes: rowBytes,
            eps: epsilon,
            _pad1: 0
        )

        // Threads per threadgroup: round cols up to a multiple of 32
        // (simdwidth) and clamp at 1024. The kernel's parallel-sum
        // loop handles any tptg correctly — extra threads (cols not a
        // multiple of 32) just skip the loop body and contribute 0.
        let roundedUpCols = ((cols + Self.simdWidth - 1) / Self.simdWidth) * Self.simdWidth
        let tptg = min(roundedUpCols, Self.maxThreadsPerThreadgroup)

        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            preconditionFailure("L2Norm: failed to create compute command encoder")
        }
        encoder.setComputePipelineState(pipeline)
        encoder.setBytes(&args, length: MemoryLayout<L2NormArgs>.size, index: 0)
        encoder.setBuffer(input, offset: 0, index: 1)
        encoder.setBuffer(output, offset: 0, index: 2)
        encoder.setThreadgroupMemoryLength(Self.threadgroupMemoryBytes, index: 0)

        // One threadgroup per row.
        encoder.dispatchThreadgroups(
            MTLSize(width: rows, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: tptg, height: 1, depth: 1)
        )
        encoder.endEncoding()
    }

    /// Mirrors the MSL `L2NormArgs` byte-for-byte. Field order is
    /// load-bearing — changes must be mirrored in `L2Norm.metal`.
    /// Two interior 4-byte pads handle the `int32 → uint64` and
    /// trailing struct alignment transitions.
    /// `MemoryLayout<L2NormArgs>.size == 24` is asserted in
    /// `init(context:)` so a layout drift traps immediately with a
    /// clear message rather than surfacing as silent garbage.
    struct L2NormArgs {
        var cols: Int32              // offset 0
        var _pad0: Int32             // offset 4 — align row_pitch_bytes to 8
        var row_pitch_bytes: UInt64  // offset 8
        var eps: Float               // offset 16
        var _pad1: Int32             // offset 20 — round struct to 8-byte alignment
        // total: 24
    }
}

#endif // canImport(Metal)
