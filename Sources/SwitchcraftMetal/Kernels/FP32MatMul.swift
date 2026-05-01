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
//     command buffer; no commit, no wait.

#if canImport(Metal)

import Foundation
import Metal
@_spi(SwitchcraftMetal) import SwitchcraftCore

/// Single-dispatch encoder for `kernel_fp32_matmul`.
///
/// Computes `C = A · B` (or `C = A · Bᵀ` when `transposeB == true`)
/// over a batch of independent matrices, all FP32. Per-head views of
/// an interleaved `[M, nHeads*dHead]` buffer are addressed by passing
/// explicit row and batch strides.
///
/// Precision contract: FP32 throughout — used on the activation paths
/// where ADR 014(b)/(i) require FP32.
@_spi(SwitchcraftMetal)
public struct FP32MatMulKernel {
    /// Function name in MSL.
    public static let kernelFunctionName = "kernel_fp32_matmul"

    private let context: MetalContext
    private let pipeline: MTLComputePipelineState

    public init(context: MetalContext) throws {
        precondition(
            MemoryLayout<FP32MatMulArgs>.size == 44,
            "FP32MatMulArgs layout drifted from MSL FP32MatMulArgs "
            + "(expected 44 bytes, got \(MemoryLayout<FP32MatMulArgs>.size))"
        )
        try registerSwitchcraftMetalShaders(with: context)
        self.context = context
        self.pipeline = try context.pipeline(for: Self.kernelFunctionName)
    }

    /// Encode a single FP32 matmul dispatch into `commandBuffer`. Does
    /// not commit or wait.
    ///
    /// Default strides assume contiguous row-major layouts:
    ///   - `a_row_stride = K`
    ///   - `b_row_stride = K`  (if `transposeB == true`)
    ///   - `b_row_stride = N`  (if `transposeB == false`)
    ///   - `c_row_stride = N`
    ///   - `a_stride_batch = M*K`, `b_stride_batch = N*K` or `K*N`,
    ///     `c_stride_batch = M*N`
    /// Pass non-default strides to slice per-head views from an
    /// interleaved buffer.
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
        cStrideBatch: Int? = nil,
        aRowStride: Int? = nil,
        bRowStride: Int? = nil,
        cRowStride: Int? = nil
    ) {
        precondition(M > 0, "FP32MatMul: M must be positive (got \(M))")
        precondition(N > 0, "FP32MatMul: N must be positive (got \(N))")
        precondition(K > 0, "FP32MatMul: K must be positive (got \(K))")
        precondition(batch > 0, "FP32MatMul: batch must be positive (got \(batch))")

        let aStride = aStrideBatch ?? (M * K)
        let bStride = bStrideBatch ?? (transposeB ? N * K : K * N)
        let cStride = cStrideBatch ?? (M * N)
        let aRow = aRowStride ?? K
        let bRow = bRowStride ?? (transposeB ? K : N)
        let cRow = cRowStride ?? N

        precondition(aStride >= 0, "FP32MatMul: aStrideBatch must be non-negative")
        precondition(bStride >= 0, "FP32MatMul: bStrideBatch must be non-negative")
        precondition(cStride >= 0, "FP32MatMul: cStrideBatch must be non-negative")
        precondition(aRow > 0, "FP32MatMul: aRowStride must be positive")
        precondition(bRow > 0, "FP32MatMul: bRowStride must be positive")
        precondition(cRow > 0, "FP32MatMul: cRowStride must be positive")

        var args = FP32MatMulArgs(
            M: UInt32(M),
            N: UInt32(N),
            K: UInt32(K),
            batch: UInt32(batch),
            transpose_b: transposeB ? 1 : 0,
            a_stride_batch: UInt32(aStride),
            b_stride_batch: UInt32(bStride),
            c_stride_batch: UInt32(cStride),
            a_row_stride: UInt32(aRow),
            b_row_stride: UInt32(bRow),
            c_row_stride: UInt32(cRow)
        )

        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            preconditionFailure("FP32MatMul: failed to create compute command encoder")
        }
        encoder.setComputePipelineState(pipeline)
        encoder.setBytes(&args, length: MemoryLayout<FP32MatMulArgs>.size, index: 0)
        encoder.setBuffer(a, offset: 0, index: 1)
        encoder.setBuffer(b, offset: 0, index: 2)
        encoder.setBuffer(output, offset: 0, index: 3)

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
    /// `MemoryLayout<FP32MatMulArgs>.size == 44` is asserted in
    /// `init(context:)`.
    struct FP32MatMulArgs {
        var M: UInt32              // offset 0
        var N: UInt32              // offset 4
        var K: UInt32              // offset 8
        var batch: UInt32          // offset 12
        var transpose_b: UInt32    // offset 16
        var a_stride_batch: UInt32 // offset 20
        var b_stride_batch: UInt32 // offset 24
        var c_stride_batch: UInt32 // offset 28
        var a_row_stride: UInt32   // offset 32
        var b_row_stride: UInt32   // offset 36
        var c_row_stride: UInt32   // offset 40
        // total: 44
    }
}

#endif // canImport(Metal)
