// SPDX-License-Identifier: Apache-2.0
//
// Swift host wrapper for the Q4_K × FP32 matmul Metal kernel
// (`kernel_mul_mm_q4_K_f32` — see
// `Sources/SwitchcraftMetal/Shaders/Q4KMatMul.metal`). Issue #60.
//
// Public surface intentionally minimal:
//   - `init(context:)` resolves the pipeline state once and caches it.
//   - `encode(commandBuffer:weight:weightShape:input:output:M:)`
//     records a single dispatch into the caller-supplied command
//     buffer; no commit, no wait. Orchestration / batching is the
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

/// Single-dispatch encoder for `kernel_mul_mm_q4_K_f32`.
///
/// Computes `C = A * B^T` where `A` is the Q4_K-stored weight matrix
/// in GGUF's `[N, K]` storage layout and `B` is an FP32 row-major
/// activation matrix `[M, K]`. The output `C` is FP32 row-major
/// `[M, N]`.
///
/// `weightShape` is `(N, K)` matching GGUF's `[N, K]` storage layout —
/// K is the contiguous axis; super-blocks iterate along K within a row
/// of N. K must be a multiple of 256 by Q4_K block layout (precondition).
@_spi(SwitchcraftMetal)
public struct Q4KMatMulKernel {
    /// Function name in MSL — matches upstream's `[[host_name(...)]]`.
    public static let kernelFunctionName = "kernel_mul_mm_q4_K_f32"

    /// Threadgroup tile width along `ne0` (N). Mirrors `NR0` in the MSL.
    public static let tileN: Int = 64
    /// Threadgroup tile width along `ne1` (M). Mirrors `NR1` in the MSL.
    public static let tileM: Int = 32
    /// Threads per threadgroup. Mirrors upstream's
    /// `N_SIMDWIDTH * N_MM_SIMD_GROUP_X * N_MM_SIMD_GROUP_Y = 32 * 2 * 2 = 128`.
    public static let threadsPerThreadgroup: Int = 128
    /// Threadgroup memory size in bytes. Sized for the dequantised A
    /// tile (`NR0 * NK * sizeof(half) = 64 * 32 * 2 = 4096`) + the half
    /// B tile (`NR1 * NK * sizeof(half) = 32 * 32 * 2 = 2048`), with
    /// extra slack for the bounds-checked output staging path
    /// (`temp_str` reuses the same shmem space, max
    /// `NR0 * NR1 * sizeof(float) = 64 * 32 * 4 = 8192`).
    public static let threadgroupMemoryBytes: Int = 8192

    /// Bytes per Q4_K super-block (4 bytes `dm` + 12 bytes `scales` +
    /// 128 bytes `qs`). Matches `Q4KDecode.bytesPerSuperBlock`.
    public static let bytesPerQ4KSuperBlock: Int = 144
    /// Elements per Q4_K super-block.
    public static let elementsPerQ4KSuperBlock: Int = 256

    private let context: MetalContext
    private let pipeline: MTLComputePipelineState

    public init(context: MetalContext) throws {
        try registerSwitchcraftMetalShaders(with: context)
        self.context = context
        self.pipeline = try context.pipeline(for: Self.kernelFunctionName)
    }

    /// Encode a single Q4_K × FP32 matmul into `commandBuffer`. Does
    /// not commit or wait; the caller (typically `T5MetalEmbedder`,
    /// issue #65) batches dispatches into a per-encoder command buffer
    /// and commits once.
    ///
    /// - Parameters:
    ///   - commandBuffer: the command buffer to encode into.
    ///   - weight: Q4_K-packed weight bytes — `(N/1) * (K/256) * 144`
    ///     bytes. Must be a Metal-resident `MTLBuffer` produced by the
    ///     GGUFReader (or by tests via `Q4KDecode`-shaped helpers).
    ///   - weightShape: `(N, K)` in elements. K must be a multiple of 256.
    ///   - input: FP32 row-major activation matrix, `M * K` floats.
    ///   - output: FP32 row-major output, `M * N` floats. Must be
    ///     `.storageModeShared` so the caller can read the result
    ///     after the command buffer completes.
    ///   - M: number of activation rows (== number of output rows).
    public func encode(
        commandBuffer: MTLCommandBuffer,
        weight: MTLBuffer,
        weightShape: (Int, Int),
        input: MTLBuffer,
        output: MTLBuffer,
        M: Int
    ) {
        let (N, K) = weightShape

        precondition(
            K % 256 == 0,
            "Q4_K matmul: K (\(K)) must be a multiple of 256 by Q4_K block layout"
        )
        precondition(N > 0, "Q4_K matmul: N must be positive (got \(N))")
        precondition(M > 0, "Q4_K matmul: M must be positive (got \(M))")

        // Build the args struct verbatim per the MSL definition. Field
        // order and types match `ggml_metal_kargs_mul_mm` byte-for-byte.
        // Batch fields are populated to safe non-trapping defaults
        // (`r2 = r3 = 1`, `ne12 = 1`) — never zero (div-by-zero in
        // upstream's index math; see Plan §Risks "Arg-struct misfill").
        let nb01 = UInt64((K / 256) * Self.bytesPerQ4KSuperBlock)
        let nb10 = UInt64(MemoryLayout<Float>.size)
        let nb11 = UInt64(K) * nb10
        var args = MulMMArgs(
            ne00: Int32(K),
            ne02: 1,
            nb01: nb01,
            nb02: nb01 * UInt64(N),
            nb03: nb01 * UInt64(N),
            ne12: 1,
            nb10: nb10,
            nb11: nb11,
            nb12: nb11 * UInt64(M),
            nb13: nb11 * UInt64(M),
            ne0: Int32(N),
            ne1: Int32(M),
            r2: 1,
            r3: 1
        )

        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            // Encoder creation only fails on impossible-state command
            // buffers (already-committed, out-of-memory). The caller's
            // contract is to pass a fresh command buffer; treat as a
            // programming error.
            preconditionFailure("Q4_K matmul: failed to create compute command encoder")
        }
        encoder.setComputePipelineState(pipeline)
        encoder.setBytes(&args, length: MemoryLayout<MulMMArgs>.size, index: 0)
        encoder.setBuffer(weight, offset: 0, index: 1)
        encoder.setBuffer(input,  offset: 0, index: 2)
        encoder.setBuffer(output, offset: 0, index: 3)
        encoder.setThreadgroupMemoryLength(Self.threadgroupMemoryBytes, index: 0)

        // Threadgroup grid: tgpig.x along M (NR1=32 each), tgpig.y
        // along N (NR0=64 each), tgpig.z = batch (= 1 here).
        let groupsX = (M + Self.tileM - 1) / Self.tileM
        let groupsY = (N + Self.tileN - 1) / Self.tileN
        encoder.dispatchThreadgroups(
            MTLSize(width: groupsX, height: groupsY, depth: 1),
            threadsPerThreadgroup: MTLSize(
                width: Self.threadsPerThreadgroup,
                height: 1,
                depth: 1
            )
        )
        encoder.endEncoding()
    }

    /// Mirrors the MSL `ggml_metal_kargs_mul_mm` byte-for-byte.
    /// `internal` so tests can exercise the layout assertion. Field
    /// order is load-bearing — changes must be mirrored in
    /// `Q4KMatMul.metal`.
    @frozen
    @usableFromInline
    struct MulMMArgs {
        var ne00: Int32
        var ne02: Int32
        var nb01: UInt64
        var nb02: UInt64
        var nb03: UInt64
        var ne12: Int32
        var nb10: UInt64
        var nb11: UInt64
        var nb12: UInt64
        var nb13: UInt64
        var ne0: Int32
        var ne1: Int32
        var r2: Int16
        var r3: Int16
    }
}

#endif // canImport(Metal)
