// SPDX-License-Identifier: Apache-2.0
//
// Faithful port of ggml's `kernel_mul_mm_q4_K_f32` (the Q4_K-weight ×
// FP32-activation matrix-matrix multiply that dominates the T5 encoder
// forward pass — issue #60). Pinned to upstream commit
// `b70770970e84c30a007b3859a453768b3ece2d3d`
// (`docs/porting/ggml-t5.md` §"ggml/llama.cpp pinned commit").
//
// What this file ports
// --------------------
//
// - The legacy non-`metal_tensor` `kernel_mul_mm` template body
//   (upstream `src/ggml-metal/ggml-metal.metal` lines 9434–9650, the
//   `#else` branch of `GGML_METAL_HAS_TENSOR`). The newer `metal_tensor`
//   path requires Metal 3.2 + `MetalPerformancePrimitives.h`, which we
//   deliberately do not depend on (ADR 015 §"Build / packaging" — Phase
//   2 ships custom kernels only).
// - `dequantize_q4_K` (upstream lines 681–697) — the per-16-element
//   register-tile dequantiser, NOT the whole-row `dequantize_row_q4_K`
//   form. The CPU reference in `Sources/SwitchcraftMetal/GGUF/Q4KDecode.swift`
//   (whole-row form) and this register-tile form are different
//   traversal orders of the same Q4_K spec; both must produce the same
//   FP32 outputs given the same bytes. The dequant-form-agreement
//   regression test asserts this on a canonical super-block.
// - `block_q4_K` struct (upstream `src/ggml-common.h` lines 317–328).
// - `ggml_metal_kargs_mul_mm` arg struct (upstream
//   `src/ggml-metal/ggml-metal-impl.h` lines 437–451), kept verbatim
//   even though our wrapper only exercises the no-batch case
//   (`r2=r3=ne12=ne13=1`). Catalogue intent is a verbatim port; future
//   commit-pin bumps remain diff-able.
// - Threadgroup tile constants (`NR0=64`, `NR1=32`, `NK=32`) and
//   simdgroup-reduction structure unchanged from upstream.
//
// Deviations from upstream
// ------------------------
//
// 1. `FC_mul_mm_bc_out = true` (compiled-in). Upstream selects this
//    function constant per-shape at pipeline-create time; we use a
//    single specialisation per ADR 015 "no runtime tuning". With
//    `bc_out = true` the kernel branches at run-time on tile alignment:
//    aligned tiles (`r0 + NR0 <= ne0 && r1 + NR1 <= ne1`) take the
//    fast direct-write path, non-aligned tiles take the bounds-checked
//    `temp_str` path. So setting `bc_out = true` is a strict superset
//    of `bc_out = false` — the fast path is preserved for T5-base
//    shapes (all aligned), and the boundary path is available for the
//    edge-case test (N=33). `FC_mul_mm_bc_inp = false` is sufficient
//    because (a) `K % 256 == 0` is a precondition on the wrapper, so K
//    is always a multiple of `NK=32`, and (b) Q4_K weights never go
//    through the FP-element-load path (`is_same<T0_4x4, block_q>` is
//    `false` for T0=float).
// 2. `mpp::tensor_ops::matmul2d` path omitted entirely (see "What this
//    file ports" above). The legacy simdgroup-tile path is the only
//    one shipped here.
// 3. `simdgroup_*` requires Apple Silicon (Metal Shading Language 2.3+,
//    GPU Family 7+). The package targets macOS 13+ / iOS 16+ which is
//    Apple-Silicon-only for the kernels of interest, so no `#if` guard
//    is added; `MetalAvailability.isAvailable` test gating already
//    skips when no Metal device is present.

#include <metal_stdlib>

using namespace metal;

// Match upstream's pragma-unroll macro so the loop body matches
// `ggml-metal.metal` line-for-line. `_Pragma("clang loop unroll(full)")`
// is what ggml uses; if the Metal compiler ever rejects it, downgrade
// to a plain `for`.
#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)

// SIMD width on Apple Silicon GPUs.
#define N_SIMDWIDTH 32

// ---------------------------------------------------------------------
// Q4_K block layout — verbatim port of upstream `block_q4_K` from
// `ggml/src/ggml-common.h:317-328` at the pinned commit. The MSL
// representation flattens the `union { struct { half d, dmin; }; half2
// dm; }` into the `dm` `half2` form because MSL doesn't support
// anonymous unions; access via `dm[0]` / `dm[1]` matches the C
// definition (little-endian on Apple Silicon).
// 144 bytes total: 4 (dm) + 12 (scales) + 128 (qs).
// ---------------------------------------------------------------------

#define QK_K          256
#define K_SCALE_SIZE  12

typedef struct {
    half2   dm;                  // [d, dmin]
    uint8_t scales[K_SCALE_SIZE]; // sub-block 6-bit scales/mins
    uint8_t qs[QK_K/2];          // 4-bit weights
} block_q4_K;

static_assert(
    sizeof(block_q4_K) == 2 * sizeof(half) + K_SCALE_SIZE + QK_K/2,
    "wrong q4_K block size/padding (expected 144 bytes)"
);

// ---------------------------------------------------------------------
// Per-16-element register-tile dequantiser — verbatim port of
// `dequantize_q4_K` (upstream lines 681–697). `il ∈ [0, 16)` selects
// which 16-element slice of the 256-element super-block to dequantise.
// `get_scale_min_k4_just2` (upstream line 675) unpacks two adjacent
// 6-bit scale/min pairs from the 12-byte `scales` packing.
// ---------------------------------------------------------------------

static inline uchar2 get_scale_min_k4_just2(int j, int k, device const uchar * q) {
    return j < 4
        ? uchar2{ uchar(q[j+0+k] & 63), uchar(q[j+4+k] & 63) }
        : uchar2{ uchar((q[j+4+k] & 0xF) | ((q[j-4+k] & 0xc0) >> 2)),
                  uchar((q[j+4+k] >> 4) | ((q[j-0+k] & 0xc0) >> 2)) };
}

template <typename type4x4>
void dequantize_q4_K(device const block_q4_K * xb, short il, thread type4x4 & reg) {
    device const uchar * q = xb->qs;

    short is = (il / 4) * 2;
    q = q + (il / 4) * 32 + 16 * (il & 1);
    il = il & 3;
    const uchar2 sc = get_scale_min_k4_just2(is, il / 2, xb->scales);
    const float d   = il < 2 ? float(xb->dm[0]) : float(xb->dm[0]) / 16.0f;
    const float min = float(xb->dm[1]);
    const float dl  = d * sc[0];
    const float ml  = min * sc[1];

    const ushort mask = il < 2 ? 0x0F : 0xF0;
    for (int i = 0; i < 16; ++i) {
        reg[i/4][i%4] = dl * (q[i] & mask) - ml;
    }
}

// Number of 16-element register-tile slices per Q4_K super-block. Equal
// to upstream's `QK_NL` (= QK_K / 16 = 16). Used as the `nl` template
// parameter in `kernel_mul_mm_q4_K_f32` instantiation.
#define QK_NL 16

// ---------------------------------------------------------------------
// `ggml_metal_kargs_mul_mm` — verbatim port of upstream
// `src/ggml-metal/ggml-metal-impl.h:437-451`. Field order and types
// must match the host-side Swift `MulMMArgs` literal byte-for-byte.
//
//   ne00 — K (inner / contiguous dim of src0)
//   ne02 — batch dim 2 of src0 (=1 here; populated to avoid div-by-zero
//          in upstream's index math; see Plan §Risks "Arg-struct misfill")
//   nb01 — bytes per row of src0 (Q4_K: (K/256)*144)
//   nb02 — bytes per src0 batch slice (=1*nb01*ne01 placeholder)
//   nb03 — bytes per src0 outer batch slice (=1)
//   ne12 — batch dim 1 of src1 (=1)
//   nb10 — bytes per src1 element (=sizeof(float)=4)
//   nb11 — bytes per src1 row (=K*sizeof(float))
//   nb12 — bytes per src1 batch slice (=ne1*nb11)
//   nb13 — bytes per src1 outer batch slice (=nb12)
//   ne0  — N (rows of A in ggml-storage = output cols)
//   ne1  — M (rows of B = output rows)
//   r2,r3 — broadcast ratios (=1 for no broadcast; never 0 — div-by-zero)
// ---------------------------------------------------------------------

typedef struct {
    int32_t  ne00;
    int32_t  ne02;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne12;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    int32_t  ne0;
    int32_t  ne1;
    int16_t  r2;
    int16_t  r3;
} ggml_metal_kargs_mul_mm;

// ---------------------------------------------------------------------
// `kernel_mul_mm` body — verbatim port of upstream lines 9434–9650
// (the `#else` branch of `GGML_METAL_HAS_TENSOR`).
//
// Specialised for the q4_K_f32 instantiation (upstream line 10112):
//   S0 = half, S0_4x4 = half4x4, S0_8x8 = simdgroup_half8x8     (threadgroup A)
//   S1 = half, S1_2x4 = half2x4, S1_8x8 = simdgroup_half8x8     (threadgroup B)
//   block_q = block_q4_K, nl = QK_NL = 16, dequantize = dequantize_q4_K
//   T0 = float, T0_4x4 = float4x4    (src0 element type — unused for q4_K
//                                      because is_same<T0_4x4, block_q4_K>
//                                      is false; the dequantise branch
//                                      always runs)
//   T1 = float, T1_2x4 = float2x4    (src1 element type — FP32 activations)
//
// `FC_mul_mm_bc_inp` is hard-coded `false` (see header §Deviations).
// `FC_mul_mm_bc_out` is hard-coded `true`. With `bc_out = true` aligned
// tiles still hit the fast simdgroup_store path; non-aligned tiles fall
// back to the threadgroup-staged write — strict superset of `false`.
// ---------------------------------------------------------------------

constant bool FC_mul_mm_bc_inp = false;
constant bool FC_mul_mm_bc_out = true;

kernel void kernel_mul_mm_q4_K_f32(
        constant ggml_metal_kargs_mul_mm & args [[buffer(0)]],
        device const char *  src0   [[buffer(1)]],
        device const char *  src1   [[buffer(2)]],
        device       char *  dst    [[buffer(3)]],
        threadgroup  char *  shmem  [[threadgroup(0)]],
        uint3                tgpig  [[threadgroup_position_in_grid]],
        ushort               tiitg  [[thread_index_in_threadgroup]],
        ushort               sgitg  [[simdgroup_index_in_threadgroup]]) {

    // Threadgroup A tile (dequantised half) and B tile (half src1).
    threadgroup half * sa = (threadgroup half *)(shmem);
    threadgroup half * sb = (threadgroup half *)(shmem + 4096);

    constexpr int NR0 = 64;     // tile rows along ne0 (N)
    constexpr int NR1 = 32;     // tile rows along ne1 (M)

    constexpr int NK  = 32;     // K iteration block
    constexpr int NL0 = NK/16;  // = 2  (per-thread A-load granularity)
    constexpr int NL1 = NK/8;   // = 4  (per-thread B-load granularity)

    const int im = tgpig.z;
    const int r0 = tgpig.y * NR0;
    const int r1 = tgpig.x * NR1;

    // Effective tile dimensions (handle right/bottom edges).
    const short nr0 = (args.ne0 - r0 < NR0) ? (args.ne0 - r0) : NR0;
    const short nr1 = (args.ne1 - r1 < NR1) ? (args.ne1 - r1) : NR1;

    // Per-thread row indices into the tile.
    const short lr0 = ((short)tiitg/NL0) < nr0 ? ((short)tiitg/NL0) : nr0 - 1;
    const short lr1 = ((short)tiitg/NL1) < nr1 ? ((short)tiitg/NL1) : nr1 - 1;

    const short il0 = (tiitg % NL0);

    short il = il0;

    const int i12 = im % args.ne12;
    const int i13 = im / args.ne12;

    const uint64_t offset0 = (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
    const short    offset1 = il0 / QK_NL;

    device const block_q4_K * x = (device const block_q4_K *)
        (src0 + args.nb01*(r0 + lr0) + offset0) + offset1;

    const short iy = 8 * (tiitg % NL1);

    device const float * y = (device const float *)(src1
        + args.nb13*i13
        + args.nb12*i12
        + args.nb11*(r1 + lr1)
        + args.nb10*iy);

    simdgroup_half8x8  ma[4];
    simdgroup_half8x8  mb[2];
    simdgroup_float8x8 mc[8];

    for (short i = 0; i < 8; i++) {
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }

    for (int loop_k = 0; loop_k < args.ne00; loop_k += NK) {
        // === PHASE 1A: load + dequantise A into threadgroup memory ===
        // For q4_K (T0=float, block_q=block_q4_K) the
        // is_same<T0_4x4, block_q>::value branch is false at compile
        // time (upstream's first `if` at line 9504), so we always take
        // the dequantise-and-store path. Hand-specialised here to keep
        // the MSL diff-able against upstream.
        {
            half4x4 temp_a;
            dequantize_q4_K(x, il, temp_a);

            threadgroup_barrier(mem_flags::mem_threadgroup);

            FOR_UNROLL (short i = 0; i < 16; i++) {
                const short sx = 2*il0 + i/8;
                const short sy = (tiitg/NL0)/8;

                const short lx = (tiitg/NL0)%8;
                const short ly = i%8;

                const short ib = 8*sx + sy;

                // Upstream comment preserved: "this is massively slower.. WTF?"
                // *(sa + 64*ib + 8*ly + lx) = temp_a[i/4][i%4];   // explicit deref form
                *(sa + 64*ib + 8*ly + lx) = temp_a[i/4][i%4];
            }
        }

        // === PHASE 1B: load B (FP32 activations) into threadgroup ===
        // FC_mul_mm_bc_inp is hard-coded false (K is a multiple of 32 by
        // construction), so we always take the direct-vector-load path.
        {
            const short sx = (tiitg % NL1);
            const short sy = (tiitg / NL1) / 8;
            const short ly = (tiitg / NL1) % 8;

            const short ib = 4*sx + sy;

            // Cast FP32 activation 8-vector to half2x4 (= 8 halves) and
            // store into threadgroup memory. This is where the precision
            // catalogue's "src1 cast to FP16 in threadgroup memory" step
            // lives (per `docs/porting/ggml-t5.md` §"Per-op precision matrix").
            *(threadgroup half2x4 *)(sb + 64*ib + 8*ly) =
                (half2x4)(*((device float2x4 *) y));
        }

        // Advance the per-iteration A pointer (`x`) and inner-block
        // index (`il`) per upstream lines 9577–9579.
        il = (il + 2 < QK_NL) ? il + 2 : il % 2;
        x  = (il < 2) ? x + (2 + QK_NL - 1)/QK_NL : x;

        y += NK;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // === PHASE 2: simdgroup matmul on the loaded NK-slice ===
        threadgroup const half * lsma = (sa + 4*64*(sgitg%2));
        threadgroup const half * lsmb = (sb + 2*64*(sgitg/2));

        FOR_UNROLL (short ik = 0; ik < NK/8; ik++) {
            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 4; i++) {
                simdgroup_load(ma[i], lsma + 64*i, 8, 0, false);
            }

            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 2; i++) {
                simdgroup_load(mb[i], lsmb + 64*i, 8, 0, false);
            }

            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 8; i++) {
                simdgroup_multiply_accumulate(mc[i], mb[i/4], ma[i%4], mc[i]);
            }

            lsma += 8*64;
            lsmb += 4*64;
        }
    }

    // === PHASE 3: write output tile to dst ===
    if (!FC_mul_mm_bc_out || (r0 + NR0 <= args.ne0 && r1 + NR1 <= args.ne1)) {
        // Tile fits inside dst — direct write via simdgroup_store.
        device float * C = (device float *) dst +
            (r0 + 32*(sgitg &  1)) +
            (r1 + 16*(sgitg >> 1)) * args.ne0 + im*args.ne1*args.ne0;

        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], C + 8*(i%4) + 8*args.ne0*(i/4), args.ne0, 0, false);
        }
    } else {
        // Tile partially outside dst — stage to threadgroup memory and
        // copy element-by-element only the in-bounds (nr0 x nr1) slice.
        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup float * temp_str = ((threadgroup float *) shmem) +
            32*(sgitg & 1) + (16*(sgitg >> 1))*NR0;

        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], temp_str + 8*(i%4) + 8*NR0*(i/4), NR0, 0, false);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sgitg == 0) {
            for (int j = tiitg; j < nr1; j += NR1) {
                device float  * D  = (device float  *) dst + r0 + (r1 + j)*args.ne0 + im*args.ne1*args.ne0;
                device float4 * D4 = (device float4 *) D;

                threadgroup float  * C  = temp_str + (j*NR0);
                threadgroup float4 * C4 = (threadgroup float4 *) C;

                int i = 0;
                for (; i < nr0/4; i++) {
                    *(D4 + i) = *(C4 + i);
                }

                i *= 4;
                for (; i < nr0; i++) {
                    *(D + i) = *(C + i);
                }
            }
        }
    }
}
