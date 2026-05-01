// SPDX-License-Identifier: Apache-2.0
//
// Swift host wrapper for the FP32 softmax Metal kernel
// (`kernel_soft_max_f32` — see
// `Sources/SwitchcraftMetal/Shaders/Softmax.metal`). Issue #62.
//
// Public surface intentionally minimal:
//   - `init(context:)` resolves the pipeline state once and caches it.
//   - `encode(commandBuffer:input:bias:output:B:N:scale:)` records a
//     single dispatch into the caller-supplied command buffer; no
//     commit, no wait. Orchestration / batching is the
//     T5MetalEmbedder's responsibility (issue #64).
//
// `@_spi(SwitchcraftMetal) public` — visible to the SwitchcraftMetal
// target's tests and to the future T5MetalEmbedder (also in
// `SwitchcraftMetal`) but not part of the stable Switchcraft public
// API surface (ADR 009 / ADR 016).
//
// Bias buffer convention
// ----------------------
//
// The MSL kernel detects "bias not bound" by comparing the bias and
// input pointers (`src1 != src0`). When the caller passes `bias = nil`
// the wrapper binds the input buffer to the bias slot — pointer
// equality becomes the sentinel that takes the `pmask = nullptr`
// branch. *Do not* swap this for a default zero-buffer: that would
// silently flip the kernel into "bias present" mode (an extra 2 KB of
// useless reads per row at T5 shape, plus a possible out-of-bounds
// read if the caller's dummy buffer is shorter than `B*N*4` bytes).
//
// The same trick disables the unused `src2` (attention sinks) slot
// — input is bound there unconditionally. T5 has no sinks; if a
// future caller needs them, expose a second `bias` parameter rather
// than re-purposing this slot.

#if canImport(Metal)

import Foundation
import Metal
@_spi(SwitchcraftMetal) import SwitchcraftCore

/// Single-dispatch encoder for `kernel_soft_max_f32`.
///
/// Computes a row-wise FP32 softmax over a flat `B × N` row-major
/// FP32 input. Each of the `B` rows is independently stabilised
/// (subtract-max), exponentiated, and normalised by the row sum. An
/// optional additive bias buffer (same `B × N` layout) is applied
/// element-wise *before* the max-reduction; T5 attention rides this
/// slot for the relative-position bias.
///
/// `scale` multiplies every input element prior to the bias add. T5
/// attention typically passes `scale = 1.0` because the `1/√d_k`
/// factor is folded into the `Q · Kᵀ` matmul accumulation; the slot
/// is preserved verbatim from upstream so future callers (e.g. a
/// non-T5 attention path) can hand in a different scale without a
/// kernel-wrapper change.
///
/// Precision contract: FP32 throughout (per ADR 014(i) — softmax is
/// the canonical sensitive op the FP32 carve-out cites; subtract-max,
/// `exp`, and reciprocal each amplify precision loss in ways that
/// would silently degrade attention scores under FP16).
@_spi(SwitchcraftMetal)
public struct SoftmaxKernel {
    /// Function name in MSL — matches upstream's `[[host_name(...)]]`.
    public static let kernelFunctionName = "kernel_soft_max_f32"

    /// Maximum threads per threadgroup. Apple Silicon's max is 1024;
    /// the kernel's two-stage simdgroup reduction also requires the
    /// number of simdgroups (`tptg / simdwidth`) to fit in the first
    /// simdgroup so `simd_max(buf[tiisg])` covers them all. With
    /// simdwidth=32 that bounds `tptg ≤ 32 * 32 = 1024` regardless.
    public static let maxThreadsPerThreadgroup: Int = 1024

    /// SIMD-group width on Apple Silicon GPUs.
    public static let simdWidth: Int = 32

    /// Threadgroup memory size in bytes — one float per simdgroup
    /// slot. The kernel writes `buf[sgitg]` once per simdgroup in the
    /// inter-simdgroup max/sum reductions and reads `buf[tiisg]` in
    /// the first simdgroup, so we need at least
    /// `simdWidth * sizeof(float)` bytes (= 128) regardless of the
    /// chosen `tptg`.
    public static let threadgroupMemoryBytes: Int = simdWidth * MemoryLayout<Float>.size

    private let context: MetalContext
    private let pipeline: MTLComputePipelineState

    public init(context: MetalContext) throws {
        // Trap immediately if the Swift mirror struct's layout drifts
        // from the MSL `ggml_metal_kargs_soft_max` definition. The MSL
        // side has a `static_assert(sizeof(...) == 128)` for the same
        // reason; the Swift side needs an explicit check because layout
        // drift would otherwise surface as silent garbage outputs /
        // failed parity tests rather than a clear init-time error.
        precondition(
            MemoryLayout<SoftmaxArgs>.size == 128,
            "SoftmaxArgs layout drifted from MSL ggml_metal_kargs_soft_max "
            + "(expected 128 bytes, got \(MemoryLayout<SoftmaxArgs>.size))"
        )
        try registerSwitchcraftMetalShaders(with: context)
        self.context = context
        self.pipeline = try context.pipeline(for: Self.kernelFunctionName)
    }

    /// Encode a single softmax dispatch into `commandBuffer`. Does not
    /// commit or wait; the caller (typically `T5MetalEmbedder`, issue
    /// #64) batches dispatches into a per-encoder command buffer and
    /// commits once.
    ///
    /// - Parameters:
    ///   - commandBuffer: the command buffer to encode into.
    ///   - input: FP32 row-major `B × N` activation matrix.
    ///   - bias: optional FP32 row-major `B × N` additive bias /
    ///     mask, applied element-wise before the max-reduction. Pass
    ///     `nil` for plain (no-bias) softmax. T5 attention flattens
    ///     its `(nheads, M, N)` relative-position bias into this
    ///     `B × N` shape before calling.
    ///   - output: FP32 row-major `B × N` output buffer.
    ///   - B: number of independent softmaxes (rows).
    ///   - N: width of each softmax (the inner dim).
    ///   - scale: per-element multiplier on `input` prior to the bias
    ///     add. T5 attention passes `1.0`; preserved in the signature
    ///     to match upstream's contract for future callers.
    public func encode(
        commandBuffer: MTLCommandBuffer,
        input: MTLBuffer,
        bias: MTLBuffer?,
        output: MTLBuffer,
        B: Int,
        N: Int,
        scale: Float
    ) {
        precondition(B > 0, "Softmax: B must be positive (got \(B))")
        precondition(N > 0, "Softmax: N must be positive (got \(N))")

        let floatBytes = UInt64(MemoryLayout<Float>.size)
        let rowBytes = UInt64(N) * floatBytes
        let planeBytes = rowBytes * UInt64(B)

        precondition(input.length >= Int(planeBytes),
                     "Softmax: input buffer length \(input.length) < required \(planeBytes)")
        precondition(output.length >= Int(planeBytes),
                     "Softmax: output buffer length \(output.length) < required \(planeBytes)")
        if let bias = bias {
            precondition(bias.length >= Int(planeBytes),
                         "Softmax: bias buffer length \(bias.length) < required \(planeBytes)")
        }

        // Build the args struct. Flat `B × N` framing maps to:
        //   ne01 = B (the outer dispatch axis; one threadgroup per row)
        //   ne02 = ne03 = 1 (collapse mid/outer; tgpig.y/z = 0)
        // Bias broadcast: ne12 = ne13 = 1 collapses `i02 % ne12` and
        // `i03 % ne13` to 0, with `nb12 = nb13 = 0`, so each row's
        // bias offset is `i01 * nb11`. Caller flattens any multi-axis
        // bias into the B × N order before binding.
        // ALiBi disabled (max_bias = m0 = m1 = 0, n_head_log2 = 0);
        // T5 uses additive relpos bias on the mask slot, not ALiBi.
        var args = SoftmaxArgs(
            ne00: Int32(N),
            ne01: Int32(B),
            ne02: 1,
            _pad0: 0,
            nb01: rowBytes,
            nb02: planeBytes,
            nb03: planeBytes,
            ne11: Int32(N),
            ne12: 1,
            ne13: 1,
            _pad1: 0,
            nb11: rowBytes,
            nb12: 0,
            nb13: 0,
            nb1: rowBytes,
            nb2: planeBytes,
            nb3: planeBytes,
            scale: scale,
            max_bias: 0,
            m0: 0,
            m1: 0,
            n_head_log2: 0,
            _pad2: 0
        )

        // Threads per threadgroup: round N up to a multiple of 32
        // (simdwidth) and clamp at 1024. The kernel's parallel-max
        // and parallel-sum loops (`for i00 = tpitg.x; i00 < ne00;
        // i00 += tptg.x`) handle any tptg correctly — extra threads
        // (N not a multiple of 32) skip the loop body and contribute
        // -INFINITY / 0 to the simdgroup reductions, which is the
        // identity for max / sum respectively.
        let roundedUpN = ((N + Self.simdWidth - 1) / Self.simdWidth) * Self.simdWidth
        let tptg = min(roundedUpN, Self.maxThreadsPerThreadgroup)

        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            preconditionFailure("Softmax: failed to create compute command encoder")
        }
        encoder.setComputePipelineState(pipeline)
        encoder.setBytes(&args, length: MemoryLayout<SoftmaxArgs>.size, index: 0)
        encoder.setBuffer(input, offset: 0, index: 1)
        // src1 (bias / mask). Bind input as the sentinel when no bias
        // was supplied — the kernel checks `src1 != src0` to decide
        // whether to read the buffer; pointer equality disables the
        // path. See file header §"Bias buffer convention".
        encoder.setBuffer(bias ?? input, offset: 0, index: 2)
        // src2 (attention sinks) — always disabled by self-binding.
        encoder.setBuffer(input, offset: 0, index: 3)
        encoder.setBuffer(output, offset: 0, index: 4)
        encoder.setThreadgroupMemoryLength(Self.threadgroupMemoryBytes, index: 0)

        // One threadgroup per softmax row (i01 = tgpig.x). i02 / i03
        // are 0 in this 1-D layout; broadcast modulus and zero strides
        // keep the bias indexing well-defined.
        encoder.dispatchThreadgroups(
            MTLSize(width: B, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: tptg, height: 1, depth: 1)
        )
        encoder.endEncoding()
    }

    /// Mirrors the MSL `ggml_metal_kargs_soft_max` byte-for-byte. Field
    /// order is load-bearing — changes must be mirrored in
    /// `Softmax.metal`. Two interior 4-byte pads are required by the
    /// `int32 → uint64` alignment transitions (after `ne02` at offset
    /// 12, after `ne13` at offset 52) and a third by the trailing
    /// 8-byte struct alignment (after `n_head_log2` at offset 124).
    /// `MemoryLayout<SoftmaxArgs>.size == 128` is asserted in
    /// `init(context:)` so a layout drift traps immediately with a
    /// clear message rather than surfacing as silent garbage.
    struct SoftmaxArgs {
        var ne00: Int32        // offset 0
        var ne01: Int32        // offset 4
        var ne02: Int32        // offset 8
        var _pad0: Int32       // offset 12 — align nb01 to 8
        var nb01: UInt64       // offset 16
        var nb02: UInt64       // offset 24
        var nb03: UInt64       // offset 32
        var ne11: Int32        // offset 40
        var ne12: Int32        // offset 44
        var ne13: Int32        // offset 48
        var _pad1: Int32       // offset 52 — align nb11 to 8
        var nb11: UInt64       // offset 56
        var nb12: UInt64       // offset 64
        var nb13: UInt64       // offset 72
        var nb1: UInt64        // offset 80
        var nb2: UInt64        // offset 88
        var nb3: UInt64        // offset 96
        var scale: Float       // offset 104
        var max_bias: Float    // offset 108
        var m0: Float          // offset 112
        var m1: Float          // offset 116
        var n_head_log2: Int32 // offset 120
        var _pad2: Int32       // offset 124 — round struct to 8-byte alignment
        // total: 128
    }
}

#endif // canImport(Metal)
