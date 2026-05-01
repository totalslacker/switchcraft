// SPDX-License-Identifier: Apache-2.0
//
// Swift host wrapper for the residual-add Metal kernel
// (`kernel_add_f32` — see
// `Sources/SwitchcraftMetal/Shaders/ResidualAdd.metal`). Issue #63.
//
// Public surface intentionally minimal:
//   - `init(context:)` resolves the pipeline state once and caches it.
//   - `encode(commandBuffer:a:b:output:count:)` records a single
//     dispatch into the caller-supplied command buffer; no commit, no
//     wait. Orchestration / batching is the T5MetalEmbedder's
//     responsibility (issue #64).
//
// `@_spi(SwitchcraftMetal) public` — visible to the SwitchcraftMetal
// target's tests and to the future T5MetalEmbedder (also in
// `SwitchcraftMetal`) but not part of the stable Switchcraft public
// API surface (ADR 009 / ADR 016).

#if canImport(Metal)

import Foundation
import Metal
@_spi(SwitchcraftMetal) import SwitchcraftCore

/// Single-dispatch encoder for `kernel_add_f32`.
///
/// Computes `output[i] = a[i] + b[i]` element-wise over `count`
/// FP32 elements. No broadcast — both inputs and the output must have
/// identical shape. The T5 encoder's residual connections (12 post-
/// attention + 12 post-FFN) all hit this contract.
///
/// Precision contract: FP32 throughout. The residual stream stays at
/// FP32 across the full encoder forward pass per ADR 003 + ADR 014(b).
@_spi(SwitchcraftMetal)
public struct ResidualAddKernel {
    /// Function name in MSL — matches the upstream symbol family.
    public static let kernelFunctionName = "kernel_add_f32"

    private let context: MetalContext
    private let pipeline: MTLComputePipelineState

    public init(context: MetalContext) throws {
        precondition(
            MemoryLayout<ResidualAddArgs>.size == 4,
            "ResidualAddArgs layout drifted from MSL ResidualAddArgs "
            + "(expected 4 bytes, got \(MemoryLayout<ResidualAddArgs>.size))"
        )
        try registerSwitchcraftMetalShaders(with: context)
        self.context = context
        self.pipeline = try context.pipeline(for: Self.kernelFunctionName)
    }

    /// Encode a single residual-add dispatch into `commandBuffer`. Does
    /// not commit or wait; the caller (typically `T5MetalEmbedder`,
    /// issue #64) batches dispatches into a per-encoder command buffer
    /// and commits once.
    ///
    /// - Parameters:
    ///   - commandBuffer: the command buffer to encode into.
    ///   - a: FP32 input buffer of length `count` (the residual stream).
    ///   - b: FP32 input buffer of length `count` (the sublayer output).
    ///   - output: FP32 output buffer of length `count`. May alias `a`
    ///     or `b` for in-place add — the kernel reads each element
    ///     once before writing, and writes are non-overlapping.
    ///   - count: number of FP32 elements (typically `M * D` for an
    ///     `M`-row × `D`-col matrix). Must be positive.
    public func encode(
        commandBuffer: MTLCommandBuffer,
        a: MTLBuffer,
        b: MTLBuffer,
        output: MTLBuffer,
        count: Int
    ) {
        precondition(count > 0, "ResidualAdd: count must be positive (got \(count))")
        precondition(count <= Int(UInt32.max),
                     "ResidualAdd: count \(count) exceeds UInt32.max")

        let totalBytes = count * MemoryLayout<Float>.size
        precondition(a.length >= totalBytes,
                     "ResidualAdd: a buffer length \(a.length) < required \(totalBytes)")
        precondition(b.length >= totalBytes,
                     "ResidualAdd: b buffer length \(b.length) < required \(totalBytes)")
        precondition(output.length >= totalBytes,
                     "ResidualAdd: output buffer length \(output.length) < required \(totalBytes)")

        var args = ResidualAddArgs(count: UInt32(count))

        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            preconditionFailure("ResidualAdd: failed to create compute command encoder")
        }
        encoder.setComputePipelineState(pipeline)
        encoder.setBytes(&args, length: MemoryLayout<ResidualAddArgs>.size, index: 0)
        encoder.setBuffer(a, offset: 0, index: 1)
        encoder.setBuffer(b, offset: 0, index: 2)
        encoder.setBuffer(output, offset: 0, index: 3)

        // `dispatchThreads` with non-uniform threadgroups (Apple Silicon
        // GPU Family 7+) handles the `count % tptg != 0` tail directly.
        // No grid-rounding logic needed; the kernel still has a defensive
        // bounds check.
        let tptg = min(pipeline.maxTotalThreadsPerThreadgroup, 256)
        encoder.dispatchThreads(
            MTLSize(width: count, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: tptg, height: 1, depth: 1)
        )
        encoder.endEncoding()
    }

    /// Mirrors the MSL `ResidualAddArgs` byte-for-byte. A single
    /// `UInt32` field — no padding, no alignment surprises.
    /// `MemoryLayout<ResidualAddArgs>.size == 4` is asserted in
    /// `init(context:)` so a layout drift traps immediately with a clear
    /// message rather than surfacing as silent garbage.
    struct ResidualAddArgs {
        var count: UInt32
    }
}

#endif // canImport(Metal)
