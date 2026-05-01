// SPDX-License-Identifier: Apache-2.0
//
// Per-row L2 normalisation Metal kernel for the T5 encoder forward pass
// (issue #63). Computes `y_i = x_i / max(ε, sqrt(Σ x_i²))` row-by-row in
// FP32. Pinned to upstream commit
// `b70770970e84c30a007b3859a453768b3ece2d3d`
// (`docs/porting/ggml-t5.md` §"ggml/llama.cpp pinned commit").
//
// Why this is a fused single-dispatch kernel
// ------------------------------------------
//
// Upstream ggml has no dedicated single kernel for L2-norm; it composes
// the operation from `kernel_sqr_f32` + `kernel_sum_rows_f32` +
// element-wise scale. The catalogue (#58, row 105) allows either form;
// the issue spec (#63) leaves the choice to the implementer based on
// "threadgroup reduction efficiency at D=128". Three composed dispatches
// would dominate latency at small D and require an intermediate sum
// buffer per call. The fused form mirrors `RMSNorm.metal`'s reduction
// shape (one threadgroup per row, two-stage simdgroup reduction) — a
// proven pattern at this codebase.
//
// Fused also makes the all-zeros NaN-safety contract trivial to honour:
// `inv = 1 / max(ε, sqrt(Σ x²))` is finite at `Σ = 0`, and `0 × finite
// = 0`. Composed paths must ensure no intermediate buffer transiently
// exposes NaN to downstream consumers (#63's spec calls this out
// explicitly).
//
// Reduction structure
// -------------------
//
// Mirrors `RMSNorm.metal`'s two-stage simdgroup reduction:
//
//   1. Each thread sums `x²` over a stride of D.
//   2. `simd_sum` collapses each simdgroup's partial sum.
//   3. The first thread of each simdgroup writes its partial to
//      `shmem[sgitg]`.
//   4. The first simdgroup re-reads `shmem[tiisg]` and `simd_sum`s
//      across simdgroups to recover the row sum.
//   5. Every thread broadcasts the scale `1/max(ε, √sum)` and writes
//      `y[i] = x[i] * scale`.
//
// FP32 throughout. The `Σ x²` accumulation over D=768 has overflowed
// FP16 historically (ADR 014(b) point 2 — the canonical falsifying case
// behind the FP32 carve-out for L2-norm); this kernel must preserve
// that precision.
//
// Math difference vs RMSNorm
// --------------------------
//
//   RMSNorm: y[k] = x[k] · gain[k] / sqrt(mean(x²) + ε)
//   L2-norm: y[k] = x[k] /              max(ε, sqrt(Σ(x²)))
//
// Three substantive deltas:
//   - No division by `D` — L2 uses raw sum, RMSNorm uses mean.
//   - No gain multiply — L2 has no per-dim weight.
//   - The `ε` floor is on the *denominator* (`max(ε, √Σ)`), not added
//     under the square root. At Σ=0 this gives `inv = 1/ε`, finite, and
//     the per-element multiply produces exact zero. RMSNorm's `mean +
//     ε` form would produce non-zero output at all-zeros input.
//
// Deviations from upstream
// ------------------------
//
// 1. Single fused dispatch rather than three composed kernels (`sqr` +
//    `sum_rows` + element-wise scale). See "Why fused" above.
// 2. The `ε` floor uses `max(ε, √Σ)` form per the issue spec wording.
//    The `1/√(Σ + ε²)` form would also be NaN-safe at Σ=0; the chosen
//    form is the literal spec text and gives an exact zero output for
//    the all-zeros row regardless of `ε`'s magnitude. (Any finite ε
//    works; the test asserts exact zero.)
// 3. Threadgroup memory sized for the inter-simdgroup reduction (one
//    float per simdgroup slot = 32 × sizeof(float) = 128 bytes), same
//    as RMSNorm.

#include <metal_stdlib>

using namespace metal;

// ---------------------------------------------------------------------
// `L2NormArgs` — per-row reduction framing. Field order must match
// the host-side Swift `L2NormArgs` byte-for-byte.
//
//   cols     — D (row length; the reduction inner dim)
//   row_pitch_bytes
//            — byte stride between rows in both input and output
//              (equal to `cols * sizeof(float)` for tightly-packed
//              row-major). Kept as an explicit field so a future
//              caller with non-contiguous strides can supply a custom
//              pitch without a kernel change.
//   eps      — L2-norm epsilon floor on the denominator
//
// Single 4-byte pad after `cols` aligns `row_pitch_bytes` to its 8-byte
// natural alignment.
// ---------------------------------------------------------------------

typedef struct {
    int32_t  cols;
    int32_t  _pad0;
    uint64_t row_pitch_bytes;
    float    eps;
    int32_t  _pad1;
} L2NormArgs;

static_assert(
    sizeof(L2NormArgs) == 24,
    "L2NormArgs layout drifted (expected 24 bytes)"
);

// ---------------------------------------------------------------------
// `kernel_l2_norm_f32` — per-row L2 normalisation. One threadgroup per
// row; two-stage simdgroup reduction over the D-dim cols.
//
//   y[i, k] = x[i, k] / max(eps, sqrt(Σ_k x[i, k]²))
//
// All-zeros row contract: `eps > 0` ⇒ denominator is `eps`, so each
// output element is `0 / eps = 0`. No NaN.
// ---------------------------------------------------------------------

kernel void kernel_l2_norm_f32(
        constant L2NormArgs & args     [[buffer(0)]],
        device const char  * src       [[buffer(1)]],
        device       char  * dst       [[buffer(2)]],
        threadgroup float  * shmem_f32 [[threadgroup(0)]],
        uint3   tgpig [[threadgroup_position_in_grid]],
        ushort3 tpitg [[thread_position_in_threadgroup]],
        ushort  sgitg [[simdgroup_index_in_threadgroup]],
        ushort  tiisg [[thread_index_in_simdgroup]],
        ushort3   ntg [[threads_per_threadgroup]]) {
    if (sgitg == 0) {
        shmem_f32[tiisg] = 0.0f;
    }

    const int row = tgpig.x;
    const int cols = args.cols;

    device const float * x = (device const float *) (src + row * args.row_pitch_bytes);
    device       float * y = (device       float *) (dst + row * args.row_pitch_bytes);

    // Stage 1: parallel-sum of x² across the row.
    float sumf = 0.0f;
    for (int i = tpitg.x; i < cols; i += ntg.x) {
        const float v = x[i];
        sumf += v * v;
    }
    sumf = simd_sum(sumf);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tiisg == 0) {
        shmem_f32[sgitg] = sumf;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Stage 2: reduce per-simdgroup partials in the first simdgroup.
    sumf = shmem_f32[tiisg];
    sumf = simd_sum(sumf);

    // Scale: `1 / max(eps, √Σ)`. At Σ=0 the denominator is `eps`, so
    // `scale = 1/eps` (finite); the element-wise multiply below then
    // yields `x[i] * 1/eps = 0` for an all-zeros row. No NaN.
    const float norm  = sqrt(sumf);
    const float scale = 1.0f / max(args.eps, norm);

    for (int i = tpitg.x; i < cols; i += ntg.x) {
        y[i] = x[i] * scale;
    }
}
