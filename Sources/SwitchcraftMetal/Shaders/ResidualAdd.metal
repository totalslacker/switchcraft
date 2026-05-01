// SPDX-License-Identifier: Apache-2.0
//
// Residual add Metal kernel for the T5 encoder forward pass (issue #63).
// Based on ggml's `kernel_add_f32` — element-wise FP32 `c = a + b` with
// no broadcast. Pinned to upstream commit
// `b70770970e84c30a007b3859a453768b3ece2d3d`
// (`docs/porting/ggml-t5.md` §"ggml/llama.cpp pinned commit").
//
// What this file ports
// --------------------
//
// The simplest variant of upstream's binary element-wise `kernel_add_f32`
// (`src/ggml-metal/ggml-metal.metal`): one thread per output element,
// straight FP32 add, no broadcast / no shape arithmetic. T5 residual
// connections (12 post-attention + 12 post-FFN per encoder) are all
// same-shape `[seq_len, d_model]` adds, so the broadcasting code path
// in upstream's full `kernel_add_f32` template would be dead code here.
//
// Deviations from upstream
// ------------------------
//
// 1. Flat 1-D dispatch over `count = M * D` elements rather than
//    upstream's broadcast-aware (ne00, ne01, ne02, ne03) tile-shaped
//    grid. Same arithmetic; trivially smaller arg struct (`uint32 count`
//    only). The orchestration layer (#64) only ever calls residual add
//    on identically-shaped buffers, so the broadcast variant is out of
//    scope per the issue spec.
// 2. The residual stream is FP32 throughout per ADR 003 + ADR 014(b);
//    no FP16 staging anywhere on this code path.
// 3. No threadgroup memory, no reduction — one thread, one output
//    element. The wrapper dispatches with non-uniform threadgroups via
//    `dispatchThreads` so the bounds check inside the kernel is
//    sufficient (no need for a grid-rounding sentinel).

#include <metal_stdlib>

using namespace metal;

// ---------------------------------------------------------------------
// `ResidualAddArgs` — flat element-count framing for the no-broadcast
// case. Field order must match the host-side Swift `ResidualAddArgs`
// byte-for-byte. With a single `uint32` field there is no padding to
// worry about.
// ---------------------------------------------------------------------

typedef struct {
    uint32_t count;
} ResidualAddArgs;

static_assert(
    sizeof(ResidualAddArgs) == 4,
    "ResidualAddArgs layout drifted (expected 4 bytes)"
);

// ---------------------------------------------------------------------
// `kernel_add_f32` — flat FP32 element-wise add. One thread per element.
//
//   c[i] = a[i] + b[i]
//
// The wrapper dispatches `dispatchThreads(width: count)`, so threads
// past `count` are never launched on Apple Silicon GPU Family 7+. The
// bounds check is defensive against a future host-side bug; the GPU
// hardware itself does not require it for correctness with
// `dispatchThreads`.
// ---------------------------------------------------------------------

kernel void kernel_add_f32(
        constant ResidualAddArgs & args     [[buffer(0)]],
        device   const float     * a         [[buffer(1)]],
        device   const float     * b         [[buffer(2)]],
        device         float     * c         [[buffer(3)]],
        uint                       gid       [[thread_position_in_grid]]) {
    if (gid >= args.count) {
        return;
    }
    c[gid] = a[gid] + b[gid];
}
