// SPDX-License-Identifier: Apache-2.0
//
// Batched FP32 × FP32 matmul Metal kernel for the T5 encoder forward
// pass (issue #64).
//
// What this file ports
// --------------------
//
// A simplified FP32-only counterpart of upstream's
// `kernel_mul_mm_f32_f32` (`src/ggml-metal/ggml-metal.metal`) covering
// the three activation matmuls in the encoder that cannot reuse the
// Q4_K matmul path (#60):
//
//   1. `Q · Kᵀ`  — per-head, K=64 (violates Q4_K's K%256==0 contract).
//   2. `softmax · V` — per-head, K=512.
//   3. `2_Dense` projection (`768 → 128`) — FP16 weights widened to
//      FP32 once at init in `T5MetalEmbedder` (Plan §"Key Decisions").
//
// One shader, three callers — keeps the kernel surface narrow and
// avoids landing a second activation-only matmul plus a separate FP16
// projection kernel. The trade-off is a naive per-element compute
// schedule (no threadgroup tiling) — acceptable because the parity
// gate (≥0.99999 cosine vs PyTorch FP32) is the correctness contract,
// not throughput. Performance optimisation is sub-issue #65 / future
// work.
//
// Shape model
// -----------
//
// All three operands are batched 3-D tensors:
//
//     A : [batch, M,                  K]        — row-major
//     B : [batch, transposeB ? N : K, transposeB ? K : N]
//                                                — row-major
//     C : [batch, M, N]                          — row-major
//
// `transposeB == 1`  →  C = A · Bᵀ  (Q·Kᵀ, 2_Dense)
// `transposeB == 0`  →  C = A · B   (softmax·V)
//
// Strides are computed implicitly from M / N / K — no broadcast,
// no padding, no batch-strided weight reuse. Callers that want to
// share a B tensor across batches must set `b_stride_batch = 0` via
// future args-struct extension; not needed for the current callers.
//
// Deviations from upstream
// ------------------------
//
// 1. **Naive per-element schedule.** Upstream's `mul_mm` family uses a
//    `simdgroup_matrix` tiled load/multiply-accumulate; we ship a
//    one-thread-per-output-element scalar kernel instead. The encoder
//    workload in #64 is dominated by the 84 Q4_K matmuls; these three
//    activation-only ops contribute O(few %) of cycles per encode and
//    aren't on the critical path for the parity gate.
// 2. **No FP16 path.** ggml's `_f16_f32` and `_f16_f16` instantiations
//    are not emitted; this kernel handles only the activations the
//    T5 encoder hands it. The 2_Dense FP16 weight is widened to FP32
//    once at `T5MetalEmbedder.init` (~384 KiB resident).
// 3. **Single transpose flag, not two.** Upstream supports A-transpose
//    too (e.g. for backward pass); we never need it on the encoder
//    forward path. Bake `A` as row-major-canonical and expose only
//    `transposeB`.

#include <metal_stdlib>

using namespace metal;

// ---------------------------------------------------------------------
// `FP32MatMulArgs` — flat-3D framing matching the host-side Swift
// `FP32MatMulArgs` byte-for-byte. All fields are `uint32_t` so the
// struct size is fixed at 32 bytes regardless of platform alignment.
// ---------------------------------------------------------------------

typedef struct {
    uint32_t M;                 // rows of A and C
    uint32_t N;                 // cols of B (or rows of B if transposed) and cols of C
    uint32_t K;                 // contraction dim
    uint32_t batch;             // number of independent matmuls
    uint32_t transpose_b;       // 0 → C = A·B  ;  1 → C = A·Bᵀ
    uint32_t a_stride_batch;    // floats between batches in A (typically M*K)
    uint32_t b_stride_batch;    // floats between batches in B
    uint32_t c_stride_batch;    // floats between batches in C (typically M*N)
} FP32MatMulArgs;

static_assert(
    sizeof(FP32MatMulArgs) == 32,
    "FP32MatMulArgs layout drifted (expected 32 bytes)"
);

// ---------------------------------------------------------------------
// `kernel_fp32_matmul` — batched FP32 matmul, naive per-thread schedule.
//
// One thread per output element `C[batch, m, n]`. The dispatch grid
// uses non-uniform threadgroups (Apple Silicon GPU Family 7+) so a
// bounds check inside the kernel is defensive only.
// ---------------------------------------------------------------------

kernel void kernel_fp32_matmul(
        constant FP32MatMulArgs & args  [[buffer(0)]],
        device   const float    * A     [[buffer(1)]],
        device   const float    * B     [[buffer(2)]],
        device         float    * C     [[buffer(3)]],
        uint3                     gid   [[thread_position_in_grid]]) {
    uint m = gid.x;
    uint n = gid.y;
    uint b = gid.z;

    if (m >= args.M || n >= args.N || b >= args.batch) {
        return;
    }

    device const float * a_row = A + b * args.a_stride_batch + m * args.K;
    device       float * c_row = C + b * args.c_stride_batch + m * args.N;

    float acc = 0.0f;

    if (args.transpose_b != 0u) {
        // B laid out as [batch, N, K] row-major — output (m,n) pairs
        // with row n of B (sized K).
        device const float * b_row = B + b * args.b_stride_batch + n * args.K;
        for (uint k = 0; k < args.K; ++k) {
            acc += a_row[k] * b_row[k];
        }
    } else {
        // B laid out as [batch, K, N] row-major — output (m,n) walks
        // the n-th column of B with stride N.
        device const float * b_col = B + b * args.b_stride_batch + n;
        for (uint k = 0; k < args.K; ++k) {
            acc += a_row[k] * b_col[k * args.N];
        }
    }

    c_row[n] = acc;
}
