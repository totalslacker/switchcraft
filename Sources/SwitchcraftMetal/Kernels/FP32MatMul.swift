// SPDX-License-Identifier: Apache-2.0
//
// Swift host wrapper for the batched FP32 matmul Metal kernel
// (`kernel_fp32_matmul` — see
// `Sources/SwitchcraftMetal/Shaders/FP32MatMul.metal`). Issue #64.
//
// Three callers in `T5MetalEmbedder`:
//   - `Q · Kᵀ`           — per-head, batch=12, M=N=512, K=64, transposeB=true
//   - `softmax · V`      — per-head, batch=12, M=512, N=64, K=512, transposeB=false
//   - `2_Dense` (768→128) — batch=1, M=512, N=128, K=768, transposeB=true
//                           (FP16 weight widened to FP32 once at init).
//
// Public surface intentionally minimal:
//   - `init(context:)` resolves the pipeline state once and caches it.
//   - `encode(...)` records a single dispatch into the caller-supplied
//     command buffer; no commit, no wait. Orchestration / batching is
//     `T5MetalEmbedder`'s responsibility.
//
// `@_spi(SwitchcraftMetal) public` — visible to the SwitchcraftMetal
// target's tests and to `T5MetalEmbedder` (also in `SwitchcraftMetal`)
// but not part of the stable Switchcraft public API surface.

#if canImport(Metal)

import Foundation
import Metal
@_spi(SwitchcraftMetal) import SwitchcraftCore

/// Single-dispatch encoder for `kernel_fp32_matmul`.
///
/// Computes `C = A · B` (or `C = A · Bᵀ` when `transposeB == true`)
/// over a batch of independent matrices, all FP32. See the MSL header
/// for the shape model and deviations from upstream.
///
/// Precision contract: FP32 throughout — used on the activation paths
/// where ADR 014(b)/(i) require FP32 (Q·Kᵀ feeds the FP32 softmax;
/// softmax·V feeds the FP32 residual stream; the 2_Dense projection
/// feeds the L2-norm sensitive op).
@_spi(SwitchcraftMetal)
public struct FP32MatMulKernel {
    /// Function name in MSL.
    public static let kernelFunctionName = "kernel_fp32_matmul"

    private let context: MetalContext
    private let pipeline: MTLComputePipelineState

    public init(context: MetalContext) throws {
        precondition(
            MemoryLayout<FP32MatMulArgs>.size == 32,
            "FP32MatMulArgs layout drifted from MSL FP32MatMulArgs "
            + "(expected 32 bytes, got \(MemoryLayout<FP32MatMulArgs>.size))"
        )
        try registerSwitchcraftMetalShaders(with: context)
        self.context = context
        self.pipeline = try context.pipeline(for: Self.kernelFunctionName)
    }

    /// Encode a single FP32 matmul dispatch into `commandBuffer`. Does
    /// not commit or wait.
    ///
    /// - Parameters:
    ///   - commandBuffer: the command buffer to encode into.
    ///   - a: FP32 row-major `[batch, M, K]` activation matrix.
    ///   - b: FP32 row-major `[batch, N, K]` if `transposeB == true`,
    ///     else `[batch, K, N]`.
    ///   - output: FP32 row-major `[batch, M, N]` output buffer.
    ///   - M: rows of `A` and `output`.
    ///   - N: cols of `output` (and rows of `B` when transposed).
    ///   - K: contraction dim.
    ///   - batch: number of independent matmuls. Pass `1` for the
    ///     non-batched 2_Dense projection.
    ///   - transposeB: when `true`, treat `B` as `[batch, N, K]` and
    ///     compute `C = A · Bᵀ`.
    ///   - aStrideBatch: floats per batch in `a`. Pass `nil` for the
    ///     default `M*K` (contiguous).
    ///   - bStrideBatch: floats per batch in `b`. Pass `nil` for the
    ///     default (`N*K` if transposed, `K*N` otherwise).
    ///   - cStrideBatch: floats per batch in `output`. Pass `nil` for
    ///     the default `M*N`.
    public func encode(
        commandBuffer: MTLCommandBuffer,
        a: MTLBuffer,
        b: MTLBuffer,
        output: MTLBuffer,
        M: Int,
        N: Int,
        K: Int,
        batch: Int = 1,
        transposeB: Bool,
        aStrideBatch: Int? = nil,
        bStrideBatch: Int? = nil,
        cStrideBatch: Int? = nil
    ) {
        precondition(M > 0, "FP32MatMul: M must be positive (got \(M))")
        precondition(N > 0, "FP32MatMul: N must be positive (got \(N))")
        precondition(K > 0, "FP32MatMul: K must be positive (got \(K))")
        precondition(batch > 0, "FP32MatMul: batch must be positive (got \(batch))")

        let aStride = aStrideBatch ?? (M * K)
        let bStride = bStrideBatch ?? (N * K)
        let cStride = cStrideBatch ?? (M * N)

        precondition(aStride >= 0, "FP32MatMul: aStrideBatch must be non-negative")
        precondition(bStride >= 0, "FP32MatMul: bStrideBatch must be non-negative")
        precondition(cStride >= 0, "FP32MatMul: cStrideBatch must be non-negative")

        let floatBytes = MemoryLayout<Float>.size
        let aRequired = (batch * aStride + M * K) * floatBytes // upper bound
        let bRequired = (batch * bStride + N * K) * floatBytes
        let cRequired = (batch * cStride + M * N) * floatBytes
        // Loose checks — they can be slack when caller passes a custom
        // stride. Use a stricter form for the all-default case.
        if aStrideBatch == nil {
            precondition(a.length >= batch * M * K * floatBytes,
                         "FP32MatMul: a buffer length \(a.length) < required \(batch * M * K * floatBytes)")
        } else {
            precondition(a.length >= aRequired,
                         "FP32MatMul: a buffer length \(a.length) < required \(aRequired)")
        }
        if bStrideBatch == nil {
            precondition(b.length >= batch * N * K * floatBytes,
                         "FP32MatMul: b buffer length \(b.length) < required \(batch * N * K * floatBytes)")
        } else {
            precondition(b.length >= bRequired,
                         "FP32MatMul: b buffer length \(b.length) < required \(bRequired)")
        }
        if cStrideBatch == nil {
            precondition(output.length >= batch * M * N * floatBytes,
                         "FP32MatMul: output buffer length \(output.length) < required \(batch * M * N * floatBytes)")
        } else {
            precondition(output.length >= cRequired,
                         "FP32MatMul: output buffer length \(output.length) < required \(cRequired)")
        }

        var args = FP32MatMulArgs(
            M: UInt32(M),
            N: UInt32(N),
            K: UInt32(K),
            batch: UInt32(batch),
            transpose_b: transposeB ? 1 : 0,
            a_stride_batch: UInt32(aStride),
            b_stride_batch: UInt32(bStride),
            c_stride_batch: UInt32(cStride)
        )

        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            preconditionFailure("FP32MatMul: failed to create compute command encoder")
        }
        encoder.setComputePipelineState(pipeline)
        encoder.setBytes(&args, length: MemoryLayout<FP32MatMulArgs>.size, index: 0)
        encoder.setBuffer(a, offset: 0, index: 1)
        encoder.setBuffer(b, offset: 0, index: 2)
        encoder.setBuffer(output, offset: 0, index: 3)

        // Threadgroup choice: 8×8×1 keeps each TG modest (64 threads) and
        // tiles cleanly across the M / N grid. `dispatchThreads` with
        // non-uniform threadgroups handles `M % 8 != 0` / `N % 8 != 0`
        // tails directly; the kernel still has a defensive bounds check.
        let tgWidth = 8
        let tgHeight = 8
        let tgDepth = 1

        encoder.dispatchThreads(
            MTLSize(width: M, height: N, depth: batch),
            threadsPerThreadgroup: MTLSize(width: tgWidth, height: tgHeight, depth: tgDepth)
        )
        encoder.endEncoding()
    }

    /// Mirrors the MSL `FP32MatMulArgs` byte-for-byte. Field order is
    /// load-bearing — changes must be mirrored in `FP32MatMul.metal`.
    /// All fields are `UInt32` so there is no padding;
    /// `MemoryLayout<FP32MatMulArgs>.size == 32` is asserted in
    /// `init(context:)` so a layout drift traps immediately with a clear
    /// message rather than surfacing as silent garbage.
    struct FP32MatMulArgs {
        var M: UInt32              // offset 0
        var N: UInt32              // offset 4
        var K: UInt32              // offset 8
        var batch: UInt32          // offset 12
        var transpose_b: UInt32    // offset 16
        var a_stride_batch: UInt32 // offset 20
        var b_stride_batch: UInt32 // offset 24
        var c_stride_batch: UInt32 // offset 28
        // total: 32
    }
}

#endif // canImport(Metal)
