// SPDX-License-Identifier: Apache-2.0
//
// Faithful port of ggml's gain-fused RMSNorm Metal kernel
// (`kernel_rms_norm_mul_f32` — the F==2 specialisation of
// `kernel_rms_norm_fuse_impl<float, 2>`). Pinned to upstream commit
// `b70770970e84c30a007b3859a453768b3ece2d3d` (`docs/porting/ggml-t5.md`
// §"ggml/llama.cpp pinned commit").
//
// What this file ports
// --------------------
//
// - `kernel_rms_norm_fuse_impl<T, F>` template body (upstream
//   `src/ggml-metal/ggml-metal.metal` lines 2986–3049). Two-stage
//   simdgroup reduction: each thread accumulates a partial squared
//   sum over a stride of D, simd-summed into one float per simdgroup,
//   stashed in threadgroup memory, then a second simd-sum across the
//   first simdgroup recovers the full row sum. Mean / rsqrt / scale /
//   gain-multiply happen at FP32 throughout (T5 precision contract,
//   ADR 014(b)+(i) — FP16 RMSNorm saturates on T5 encoder
//   activations).
// - `ggml_metal_kargs_norm` arg struct (upstream
//   `src/ggml-metal/ggml-metal-impl.h` lines 549–564), kept verbatim
//   even though our wrapper only exercises the no-batch case. The
//   per-axis `nef*` / `nbf*` arrays cover src0 / f0 (gain) / f1
//   (residual; unread in F==2). Catalogue intent is a verbatim port
//   so future commit-pin bumps remain diff-able.
//
// Deviations from upstream
// ------------------------
//
// 1. Only the F==2 specialisation (`kernel_rms_norm_mul_f32`) is
//    instantiated. T5's `RMSNorm(x) * weight` is the gain-fused path;
//    the no-gain F==1 (`kernel_rms_norm_f32`) and the residual-fused
//    F==3 (`kernel_rms_norm_mul_add_f32`) variants are not needed by
//    the encoder forward pass and would be shadow code with no
//    production caller. The vec4 specialisations (`*_f32_4`) are also
//    omitted; the issue spec mandates the scalar form, and at T5-base
//    shapes (D=768) the compiler-inserted vector loads already
//    saturate the bandwidth path.
// 2. The `src1_1` (residual) buffer binding is preserved at
//    `[[buffer(3)]]` — upstream's host code always binds three source
//    buffers regardless of fuse selector. Our wrapper rebinds the
//    gain buffer as a stand-in for `src1_1`; the F==2 branch never
//    dereferences `f1`, so the buffer's contents are immaterial. The
//    `nef*[2] = 1`, `nbf*[2] = 0` arg-struct convention then keeps the
//    pointer arithmetic at offset 0, which is in-bounds for any non-
//    null binding.
// 3. The arg-struct's `ne00_t` field is set equal to `ne00` (= D);
//    this matches upstream's scalar instantiation (`T = float`). The
//    vec4 instantiation would set `ne00_t = ne00 / 4`, which we do
//    not use.
// 4. `simdgroup_*` requires Apple Silicon (Metal 2.3+, GPU Family 7+);
//    no `#if` guard here because `MetalAvailability.isAvailable` test
//    gating already skips when no Metal device is present.
// 5. The function-name suffix `_mul_f32` (vs. the issue body's
//    `kernel_rms_norm_f32`) names the gain-fused upstream symbol
//    accurately. The issue body's `kernel_rms_norm_f32` was upstream-
//    name shorthand; the catalogue (`docs/porting/ggml-t5.md`) is
//    amended in the same commit to point at `kernel_rms_norm_mul_f32`.
// 6. Upstream's parallel-sum line `sumf += dot(x[i00], x[i00])` is
//    rewritten as `sumf += dot_self(x[i00])`, a one-line helper that
//    delegates to the vector `dot()` for vector T and to `v*v` for
//    scalar `float`. Apple's MSL `dot` is only overloaded for vector
//    forms; the upstream form fails to compile for the T=float
//    instantiation. Arithmetic is unchanged.

#include <metal_stdlib>

using namespace metal;

// ---------------------------------------------------------------------
// `ggml_metal_kargs_norm` — verbatim port of upstream
// `src/ggml-metal/ggml-metal-impl.h:549-564`. Field order, types, and
// inline arrays must match the host-side Swift `RMSNormArgs` struct
// byte-for-byte.
//
//   ne00   — D (logical inner-dim length; the mean denominator)
//   ne00_t — per-thread loop bound (= ne00 for T=float, ne00/4 for
//            T=float4); we only ship the scalar form so ne00_t == ne00
//   nb1    — dst row byte stride (D * sizeof(float) for our 2-D case)
//   nb2    — dst dim2 byte stride. Indexed by tgpig.y, which is always
//            0 in our 2-D dispatch (height=1), so the value is unread
//            on this code path; the wrapper still populates it with the
//            contiguous-layout stride (M * D * sizeof(float)) so a
//            future batched extension picks up correct strides.
//   nb3    — dst dim3 byte stride. Same as nb2 but indexed by tgpig.z
//            (always 0 in 2-D); contiguous-layout stride matches nb2
//            since we have no dim2 extent yet.
//   eps    — RMSNorm epsilon (added to mean before rsqrt)
//   nef[123][i] — dim-{1,2,3} sizes for source i:
//                   i=0 src0 (input x)        — never broadcast, sizes
//                                                irrelevant
//                   i=1 src1_0 (gain)         — broadcast modulus per
//                                                dim
//                   i=2 src1_1 (residual)     — broadcast modulus per
//                                                dim (unread at F==2)
//                 Modulus values must be ≥ 1 — `i01 % nef1[i]` div-by-
//                 zero traps if any `nef*[i] == 0`.
//   nbf[123][i] — byte strides for source i, same indexing.
// ---------------------------------------------------------------------

typedef struct {
    int32_t  ne00;
    int32_t  ne00_t;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
    float    eps;
    int32_t  nef1[3];
    int32_t  nef2[3];
    int32_t  nef3[3];
    uint64_t nbf1[3];
    uint64_t nbf2[3];
    uint64_t nbf3[3];
} ggml_metal_kargs_norm;

static_assert(
    sizeof(ggml_metal_kargs_norm) == 144,
    "ggml_metal_kargs_norm layout drifted from upstream (expected 144 bytes)"
);

// ---------------------------------------------------------------------
// `dot_self` — `dot(x, x)` shorthand that compiles for both scalar
// `float` and the vector forms (`float4`). Apple's MSL `dot` is only
// overloaded for `float{2,3,4}` (and the half variants); upstream's
// `kernel_rms_norm_fuse_impl<float, ...>` instantiation relies on a
// scalar `dot(float, float)` that this Metal toolchain does not
// provide. Adding a one-line helper preserves the upstream loop body
// shape (`sumf += dot_self(x[i00])`) while keeping the scalar
// instantiation buildable. This is the only deviation in the
// arithmetic path — accumulation, reduction, and gain-multiply order
// are unchanged.
// ---------------------------------------------------------------------

template <typename T> inline float dot_self(T v)        { return dot(v, v); }
template <>           inline float dot_self<float>(float v) { return v * v; }

// ---------------------------------------------------------------------
// `kernel_rms_norm_fuse_impl<T, F>` body — verbatim port of upstream
// lines 2986–3049. Specialised here for the q-or-fp32 → FP32 gain-fused
// case (T = float, F = 2):
//
//   y[i] = (x[i] * scale) * gain[i]
//   scale = 1 / sqrt(mean(x²) + eps)
//   mean  = sum(x²) / ne00
//
// FP32 throughout — no half-precision staging anywhere on the compute
// path (the FP16 saturation point in `mean(x²)` for T5 encoder
// activations is the canonical falsifying case behind ADR 014(i)).
// ---------------------------------------------------------------------

template <typename T, short F>
kernel void kernel_rms_norm_fuse_impl(
        constant ggml_metal_kargs_norm & args,
        device const char * src0,
        device const char * src1_0,
        device const char * src1_1,
        device       char * dst,
        threadgroup float * shmem_f32 [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    if (sgitg == 0) {
        shmem_f32[tiisg] = 0.0f;
    }

    const int i01 = tgpig.x;
    const int i02 = tgpig.y;
    const int i03 = tgpig.z;

    device const T * x = (device const T *) (src0 + i03*args.nbf3[0] + i02*args.nbf2[0] + i01*args.nbf1[0]);

    device const T * f0 = (device const T *) (src1_0 + (i03%args.nef3[1])*args.nbf3[1] + (i02%args.nef2[1])*args.nbf2[1] + (i01%args.nef1[1])*args.nbf1[1]);
    device const T * f1 = (device const T *) (src1_1 + (i03%args.nef3[2])*args.nbf3[2] + (i02%args.nef2[2])*args.nbf2[2] + (i01%args.nef1[2])*args.nbf1[2]);

    float sumf = 0.0f;

    // parallel sum
    for (int i00 = tpitg.x; i00 < args.ne00_t; i00 += ntg.x) {
        sumf += dot_self(x[i00]);
    }
    sumf = simd_sum(sumf);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tiisg == 0) {
        shmem_f32[sgitg] = sumf;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    sumf = shmem_f32[tiisg];
    sumf = simd_sum(sumf);

    const float mean  = sumf/args.ne00;
    const float scale = 1.0f/sqrt(mean + args.eps);

    device T * y = (device T *) (dst + i03*args.nb3 + i02*args.nb2 + i01*args.nb1);
    for (int i00 = tpitg.x; i00 < args.ne00_t; i00 += ntg.x) {
        if (F == 1) {
            y[i00] = (x[i00]*scale);
        }
        if (F == 2) {
            y[i00] = (x[i00]*scale)*f0[i00];
        }
        if (F == 3) {
            y[i00] = (x[i00]*scale)*f0[i00] + f1[i00];
        }
    }
}

typedef decltype(kernel_rms_norm_fuse_impl<float, 2>) kernel_rms_norm_fuse_t;

template [[host_name("kernel_rms_norm_mul_f32")]]
kernel kernel_rms_norm_fuse_t kernel_rms_norm_fuse_impl<float, 2>;
