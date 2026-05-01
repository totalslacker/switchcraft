// SPDX-License-Identifier: Apache-2.0
//
// Swift host wrapper for the gated-GELU activation Metal kernel
// (`kernel_gated_gelu_f32` — see
// `Sources/SwitchcraftMetal/Shaders/GatedGELU.metal`). Issue #63.
//
// Public surface intentionally minimal:
//   - `init(context:)` resolves the pipeline state once and caches it.
//   - `encode(commandBuffer:gate:up:output:count:)` records a single
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

/// Single-dispatch encoder for `kernel_gated_gelu_f32`.
///
/// Computes `output[i] = gelu_new(gate[i]) * up[i]` element-wise over
/// `count` FP32 elements, where `gelu_new(x) = 0.5·x·(1 + tanh(√(2/π)·
/// (x + 0.044715·x³)))` (HuggingFace's `gelu_new`, matching ggml's
/// `kernel_gelu_f32`).
///
/// The T5 v1.1 / xtr-base-en FFN computes `wi_0(x)` and `wi_1(x)` as
/// two parallel matmuls (gate / up), feeds both to this kernel, and
/// projects the result through `wo`. `feed_forward_proj: "gated-gelu"`
/// in `xtr-base-en/config.json` locks this activation; the dimensions
/// are `768 → 2048 → 768` (T5 v1.1, **not** classic T5-base's 3072).
///
/// Precision contract: FP32 throughout. `gelu_new` uses `tanh` (not
/// `erf`) per upstream ggml; FP32 has comfortable headroom for typical
/// activation magnitudes.
@_spi(SwitchcraftMetal)
public struct GatedGELUKernel {
    /// Function name in MSL — fused `gelu_new(gate) * up`.
    public static let kernelFunctionName = "kernel_gated_gelu_f32"

    private let context: MetalContext
    private let pipeline: MTLComputePipelineState

    public init(context: MetalContext) throws {
        precondition(
            MemoryLayout<GatedGELUArgs>.size == 4,
            "GatedGELUArgs layout drifted from MSL GatedGELUArgs "
            + "(expected 4 bytes, got \(MemoryLayout<GatedGELUArgs>.size))"
        )
        try registerSwitchcraftMetalShaders(with: context)
        self.context = context
        self.pipeline = try context.pipeline(for: Self.kernelFunctionName)
    }

    /// Encode a single gated-GELU dispatch into `commandBuffer`. Does
    /// not commit or wait; the caller (typically `T5MetalEmbedder`,
    /// issue #64) batches dispatches into a per-encoder command buffer
    /// and commits once.
    ///
    /// - Parameters:
    ///   - commandBuffer: the command buffer to encode into.
    ///   - gate: FP32 input buffer of length `count` (the `wi_0(x)`
    ///     output that `gelu_new` is applied to).
    ///   - up: FP32 input buffer of length `count` (the `wi_1(x)`
    ///     output that the GELU result is multiplied with).
    ///   - output: FP32 output buffer of length `count`. May alias
    ///     `gate` or `up` for in-place activation.
    ///   - count: number of FP32 elements (typically `M * d_ff`, e.g.
    ///     `512 * 2048` for the xtr-base-en FFN). Must be positive.
    public func encode(
        commandBuffer: MTLCommandBuffer,
        gate: MTLBuffer,
        up: MTLBuffer,
        output: MTLBuffer,
        count: Int
    ) {
        precondition(count > 0, "GatedGELU: count must be positive (got \(count))")
        precondition(count <= Int(UInt32.max),
                     "GatedGELU: count \(count) exceeds UInt32.max")

        let totalBytes = count * MemoryLayout<Float>.size
        precondition(gate.length >= totalBytes,
                     "GatedGELU: gate buffer length \(gate.length) < required \(totalBytes)")
        precondition(up.length >= totalBytes,
                     "GatedGELU: up buffer length \(up.length) < required \(totalBytes)")
        precondition(output.length >= totalBytes,
                     "GatedGELU: output buffer length \(output.length) < required \(totalBytes)")

        var args = GatedGELUArgs(count: UInt32(count))

        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            preconditionFailure("GatedGELU: failed to create compute command encoder")
        }
        encoder.setComputePipelineState(pipeline)
        encoder.setBytes(&args, length: MemoryLayout<GatedGELUArgs>.size, index: 0)
        encoder.setBuffer(gate, offset: 0, index: 1)
        encoder.setBuffer(up, offset: 0, index: 2)
        encoder.setBuffer(output, offset: 0, index: 3)

        let tptg = min(pipeline.maxTotalThreadsPerThreadgroup, 256)
        encoder.dispatchThreads(
            MTLSize(width: count, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: tptg, height: 1, depth: 1)
        )
        encoder.endEncoding()
    }

    /// Mirrors the MSL `GatedGELUArgs` byte-for-byte. A single `UInt32`
    /// field — no padding. `MemoryLayout<GatedGELUArgs>.size == 4` is
    /// asserted in `init(context:)` so a layout drift traps immediately
    /// with a clear message rather than surfacing as silent garbage.
    struct GatedGELUArgs {
        var count: UInt32
    }
}

#endif // canImport(Metal)
