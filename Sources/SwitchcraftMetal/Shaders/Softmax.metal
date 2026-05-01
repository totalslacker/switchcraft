// SPDX-License-Identifier: Apache-2.0
//
// Faithful port of ggml's Metal softmax kernel (`kernel_soft_max<float>`,
// host-named `kernel_soft_max_f32`). Pinned to upstream commit
// `b70770970e84c30a007b3859a453768b3ece2d3d` (`docs/porting/ggml-t5.md`
// §"ggml/llama.cpp pinned commit").
//
// What this file ports
// --------------------
//
// - `kernel_soft_max<T>` template body (upstream
//   `src/ggml-metal/ggml-metal.metal` lines 1854–1958). Subtract-max
//   stabilised softmax with two-stage simdgroup reductions for both the
//   max and the sum. The optional `src1` buffer is an additive bias /
//   mask applied to the input prior to the max-reduction (T5's
//   relative-position bias rides this slot). The optional `src2` buffer
//   is the recent ggml "attention sinks" feature; T5 has no sinks, but
//   the buffer slot is preserved so future commit-pin bumps remain
//   diff-able.
// - `ggml_metal_kargs_soft_max` arg struct (upstream
//   `src/ggml-metal/ggml-metal-impl.h` lines 778–799), kept verbatim.
//   The Swift host mirror in `SoftmaxKernel.swift` matches it byte-for-
//   byte; a `static_assert` here and a `MemoryLayout<...>.size`
//   precondition there guard against drift.
//
// Deviations from upstream
// ------------------------
//
// 1. Only the `T = float` instantiation (`kernel_soft_max_f32`) is
//    shipped. The FP16-mask sibling (`kernel_soft_max_f16`) and the
//    vec4 specialisations (`kernel_soft_max_f32_4`, `kernel_soft_max_f16_4`)
//    are omitted — T5's softmax precision contract is FP32 throughout
//    (ADR 014(i); the sensitive-op carve-out cites softmax explicitly),
//    and the vec4 form would be shadow code with no production caller
//    until / unless a perf gap demands it. Same rationale as RMSNorm
//    deviation #1.
// 2. ALiBi (`max_bias > 0.0f`) is preserved verbatim in the kernel body
//    but the Swift wrapper hard-codes `max_bias = 0` so the slope
//    branch is inert. T5 attention uses additive relpos bias (mask
//    slot), not ALiBi; mixing the two would produce wrong results.
// 3. The `src2` (attention sinks) buffer binding is preserved at
//    `[[buffer(3)]]`. The Swift wrapper binds the input buffer (src0)
//    to that slot; the kernel branches on `src2 != src0` to detect
//    "buffer not bound", so this self-binding makes `psrc2` `nullptr`
//    and disables the sinks contribution to the sum. Same sentinel
//    convention upstream uses for the no-mask case.
// 4. `simdgroup_*` requires Apple Silicon (Metal 2.3+, GPU Family 7+);
//    no `#if` guard here because `MetalAvailability.isAvailable`
//    test gating already skips when no Metal device is present.
// 5. The `N_SIMDWIDTH` macro (32) used by the two-stage reduction is
//    redefined locally to keep this `.metal` file self-contained when
//    compiled in isolation by the runtime-source-compile fallback path
//    (each `.metal` resource produces its own `MTLLibrary`; macros
//    from the upstream prelude do not flow across).
// 6. Upstream's `MAX(a, b)` macro (a `#define` in the ggml prelude that
//    expands to `((a) > (b) ? (a) : (b))`) is rewritten as the MSL
//    standard-library `max(a, b)` — same arithmetic, same FP32 NaN
//    handling per IEEE 754 / MSL spec, but no need to drag the
//    upstream macro into a self-contained source file.

#include <metal_stdlib>

using namespace metal;

#define N_SIMDWIDTH 32 // assuming SIMD group size is 32 (Apple Silicon)

// ---------------------------------------------------------------------
// `ggml_metal_kargs_soft_max` — verbatim port of upstream
// `src/ggml-metal/ggml-metal-impl.h:778-799`. Field order, types, and
// inline padding must match the host-side Swift `SoftmaxArgs` struct
// byte-for-byte. The two interior 4-byte pads (after `ne02` and after
// `n_head_log2`) are layout-load-bearing — they arise from the
// `int32_t → uint64_t` alignment transition at offset 12 and from
// rounding the struct to 8-byte alignment at the end. Drop either pad
// and every later field misaligns silently.
//
//   ne00 / ne01 / ne02   — src0 inner / mid / outer extents
//                            (N / M / nheads in T5 attention shape)
//   nb01 / nb02 / nb03   — src0 byte strides along axes 1/2/3
//   ne11 / ne12 / ne13   — src1 (mask/bias) extents along axes 1/2/3
//   nb11 / nb12 / nb13   — src1 byte strides along axes 1/2/3
//                            (broadcast: indexed via `i % ne1k`)
//   nb1  / nb2  / nb3    — dst byte strides along axes 1/2/3
//   scale                — per-element multiplier on src0
//                            (T5 passes 1.0; 1/√d_k is folded into the
//                            QKᵀ matmul accumulation)
//   max_bias             — ALiBi slope cap (0 disables; T5 = 0)
//   m0 / m1              — ALiBi base parameters (unused at max_bias=0)
//   n_head_log2          — ALiBi head-class threshold (unused at
//                            max_bias=0)
// ---------------------------------------------------------------------

typedef struct {
    int32_t  ne00;
    int32_t  ne01;
    int32_t  ne02;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne11;
    int32_t  ne12;
    int32_t  ne13;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
    float    scale;
    float    max_bias;
    float    m0;
    float    m1;
    int32_t  n_head_log2;
} ggml_metal_kargs_soft_max;

static_assert(
    sizeof(ggml_metal_kargs_soft_max) == 128,
    "ggml_metal_kargs_soft_max layout drifted from upstream (expected 128 bytes)"
);

// ---------------------------------------------------------------------
// `kernel_soft_max<T>` body — verbatim port of upstream lines 1854–1958.
// One threadgroup per `(i01, i02, i03)` softmax row. Algorithm:
//
//   1. Compute `psrc0 = src0 + i01*nb01 + i02*nb02 + i03*nb03`.
//   2. If src1 is bound (`src1 != src0`), `pmask` points into the bias
//      buffer at `i11=i01, i12=i02%ne12, i13=i03%ne13` (broadcast via
//      modulus).
//   3. If src2 is bound (`src2 != src0`), `psrc2` adds a per-head
//      scalar contribution to the denominator (attention sinks).
//   4. Two-stage simd_max reduction across `i00 = tpitg.x .. ne00`
//      computes the row max of `(x*scale + slope*mask)`.
//   5. Two-stage simd_sum reduction computes
//      `sum(exp((x*scale + slope*mask) - max))`, with each thread also
//      writing its `exp_psrc0` into `pdst[i00]`.
//   6. Optional sinks contribution `exp(psrc2[i02] - max)` is added to
//      sum.
//   7. Each thread multiplies its `pdst[i00]` slot by `1/sum`.
//
// FP32 throughout — every accumulator, every intermediate, the mask
// sum, and the reciprocal. ADR 014(i) names this kernel as the
// canonical sensitive op behind the FP32 carve-out: subtract-max +
// `exp` + `1/sum` each amplify precision loss; FP16 would silently
// degrade T5 attention scores at low loss before any test catches it.
// ---------------------------------------------------------------------

template<typename T>
kernel void kernel_soft_max(
        constant ggml_metal_kargs_soft_max & args,
        device const  char * src0,
        device const  char * src1,
        device const  char * src2,
        device        char * dst,
        threadgroup  float * buf [[threadgroup(0)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint3  tptg[[threads_per_threadgroup]]) {
    const int32_t i03 = tgpig.z;
    const int32_t i02 = tgpig.y;
    const int32_t i01 = tgpig.x;

    const int32_t i13 = i03%args.ne13;
    const int32_t i12 = i02%args.ne12;
    const int32_t i11 = i01;

    device const float * psrc0 =                (device const float *) (src0 + i01*args.nb01 + i02*args.nb02 + i03*args.nb03);
    device const     T * pmask = src1 != src0 ? (device const T *    ) (src1 + i11*args.nb11 + i12*args.nb12 + i13*args.nb13) : nullptr;
    device const float * psrc2 = src2 != src0 ? (device const float *) (src2)                                                 : nullptr;
    device       float * pdst  =                (device       float *) (dst  + i01*args.nb1  + i02*args.nb2  + i03*args.nb3);

    float slope = 1.0f;

    // ALiBi
    if (args.max_bias > 0.0f) {
        const int32_t h = i02;

        const float base = h < args.n_head_log2 ? args.m0 : args.m1;
        const int   exp  = h < args.n_head_log2 ? h + 1 : 2*(h - args.n_head_log2) + 1;

        slope = pow(base, exp);
    }

    // parallel max
    float lmax = psrc2 ? psrc2[i02] : -INFINITY;

    for (int i00 = tpitg.x; i00 < args.ne00; i00 += tptg.x) {
        lmax = max(lmax, psrc0[i00]*args.scale + (pmask ? slope*pmask[i00] : 0.0f));
    }

    // find the max value in the block
    float max_val = simd_max(lmax);
    if (tptg.x > N_SIMDWIDTH) {
        if (sgitg == 0) {
            buf[tiisg] = -INFINITY;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiisg == 0) {
            buf[sgitg] = max_val;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        max_val = buf[tiisg];
        max_val = simd_max(max_val);
    }

    // parallel sum
    float lsum = 0.0f;
    for (int i00 = tpitg.x; i00 < args.ne00; i00 += tptg.x) {
        const float exp_psrc0 = exp((psrc0[i00]*args.scale + (pmask ? slope*pmask[i00] : 0.0f)) - max_val);
        lsum += exp_psrc0;
        pdst[i00] = exp_psrc0;
    }

    // This barrier fixes a failing test
    // ref: https://github.com/ggml-org/ggml/pull/621#discussion_r1425156335
    threadgroup_barrier(mem_flags::mem_none);

    float sum = simd_sum(lsum);

    if (tptg.x > N_SIMDWIDTH) {
        if (sgitg == 0) {
            buf[tiisg] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiisg == 0) {
            buf[sgitg] = sum;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        sum = buf[tiisg];
        sum = simd_sum(sum);
    }

    if (psrc2) {
        sum += exp(psrc2[i02] - max_val);
    }

    const float inv_sum = 1.0f/sum;

    for (int i00 = tpitg.x; i00 < args.ne00; i00 += tptg.x) {
        pdst[i00] *= inv_sum;
    }
}

typedef decltype(kernel_soft_max<float>) kernel_soft_max_t;

template [[host_name("kernel_soft_max_f32")]]
kernel kernel_soft_max_t kernel_soft_max<float>;
