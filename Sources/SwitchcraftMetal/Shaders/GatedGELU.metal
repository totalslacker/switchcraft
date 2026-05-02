// SPDX-License-Identifier: Apache-2.0
//
// Gated-GELU activation Metal kernel for the T5 v1.1 / xtr-base-en FFN
// (issue #63). Computes `y[i] = gelu_new(gate[i]) * up[i]` element-wise
// in FP32. Pinned to upstream commit
// `b70770970e84c30a007b3859a453768b3ece2d3d`
// (`docs/porting/ggml-t5.md` §"ggml/llama.cpp pinned commit").
//
// Why this is a fused single-dispatch kernel
// ------------------------------------------
//
// Upstream ggml composes gated-GELU from `kernel_gelu_f32` followed by
// an element-wise `kernel_mul_f32`. The catalogue allows either form;
// the issue spec (#63) prefers fused for "one fewer barrier per layer
// × 12 layers" and the catalogue rows 100 / 282 lock the FP32 carve-
// out at all stages. With `D = d_ff = 2048` and 12 FFN dispatches per
// encoder forward pass, the saved command-encoder overhead is non-
// trivial — and the arithmetic is identical because both inputs and
// the output stay FP32 throughout.
//
// `gelu_new` formula
// ------------------
//
// HuggingFace's `gelu_new` (matches ggml's `kernel_gelu_f32`):
//
//   gelu_new(x) = 0.5 · x · (1 + tanh(√(2/π) · (x + 0.044715 · x³)))
//
// We use the `tanh`-form, **not** the `erf`-form, to match upstream
// ggml exactly (catalogue §"FFN activation determination" locks this
// for cosine ≥ 0.99999 parity against the pure-Swift reference).
//
// Constants:
//   √(2/π) ≈ 0.7978845608028654 — stored as the FP32-rounded
//   `0.79788456f`. Apple's MSL `tanh(float)` is FP32 precision-correct
//   over the range `gelu_new` evaluates at typical activation
//   magnitudes (drift inside cosine ≥ 0.99999 by RMSNorm/Softmax
//   precedent — FP64 reference vs FP32 kernel).
//
// Deviations from upstream
// ------------------------
//
// 1. Fused `gelu_new(gate) * up` rather than upstream's two-dispatch
//    composition. Output of the fused form equals the composed form
//    bit-identically modulo FP32 rounding order; we avoid an
//    intermediate buffer and one threadgroup barrier per FFN block.
// 2. Flat 1-D dispatch over `count = M * D` elements via
//    `dispatchThreads`. Same thread shape as the catalogue's
//    "element-wise multiply" leg of the composed form.
// 3. The kernel takes a single `count` arg rather than upstream's
//    `(ne00, ne01, ne02, ne03)` tile shape. The orchestration layer
//    (#64) only ever calls gated-GELU on identically-shaped gate / up
//    buffers (output of two parallel `wi_0` / `wi_1` matmuls), so the
//    broadcasting code path would be dead code here.

#include <metal_stdlib>

using namespace metal;

// ---------------------------------------------------------------------
// `GatedGELUArgs` — flat element-count framing for the no-broadcast
// case. Field order must match the host-side Swift `GatedGELUArgs`
// byte-for-byte.
// ---------------------------------------------------------------------

typedef struct {
    uint32_t count;
} GatedGELUArgs;

static_assert(
    sizeof(GatedGELUArgs) == 4,
    "GatedGELUArgs layout drifted (expected 4 bytes)"
);

// ---------------------------------------------------------------------
// `gelu_new` helper. Closed-form; inlined into the kernel below. Kept
// as a standalone function so a future code path that wants plain
// `gelu_new` (without the gate multiply) can call it directly.
// ---------------------------------------------------------------------

inline float gelu_new_f32(float x) {
    // √(2/π) — FP32-rounded constant (decimal: 0.7978845608028654).
    constexpr float SQRT_2_OVER_PI = 0.79788456f;
    // HuggingFace `gelu_new` cubic coefficient.
    constexpr float CUBIC_COEFF    = 0.044715f;
    const float x3    = x * x * x;
    const float inner = SQRT_2_OVER_PI * (x + CUBIC_COEFF * x3);
    // For |inner| >= 8, tanh saturates to ±1.0 in FP32
    // (tanh(8) ≈ 1 - 2.25e-7, which rounds to 1.0f).
    // Metal's built-in tanh() returns NaN or 0 for large |inner|
    // because exp(2·inner) overflows FP32 max (~3.4e38 = exp(88.7)).
    // Bypassing tanh for |inner| >= 8 avoids this Metal bug. (Issue #75.)
    const float t = (inner >= 8.0f)  ?  1.0f
                  : (inner <= -8.0f) ? -1.0f
                  : tanh(inner);
    return 0.5f * x * (1.0f + t);
}

// ---------------------------------------------------------------------
// `kernel_gated_gelu_f32` — fused gated-GELU. One thread per element.
//
//   y[i] = gelu_new(gate[i]) * up[i]
//
// FP32 throughout — no half-precision staging anywhere on the compute
// path. Both inputs and the output are flat row-major FP32 buffers of
// `count` elements (typically `M * d_ff` for an FFN block).
// ---------------------------------------------------------------------

kernel void kernel_gated_gelu_f32(
        constant GatedGELUArgs & args     [[buffer(0)]],
        device   const float   * gate     [[buffer(1)]],
        device   const float   * up       [[buffer(2)]],
        device         float   * y        [[buffer(3)]],
        uint                     gid      [[thread_position_in_grid]]) {
    if (gid >= args.count) {
        return;
    }
    y[gid] = gelu_new_f32(gate[gid]) * up[gid];
}
