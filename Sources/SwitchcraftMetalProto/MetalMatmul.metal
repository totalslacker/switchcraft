// SPDX-License-Identifier: Apache-2.0
//
// FP32 tiled GEMM kernel for the Phase 2 custom-Metal-kernels feasibility
// investigation (issue #49). Computes `C = A · B` where:
//
//   - A is row-major M × K
//   - B is row-major K × N
//   - C is row-major M × N
//
// The threadgroup tile shape (TG_M × TG_N) is set at pipeline-creation time
// via Metal function constants so the prototype's threadgroup-sweep suite can
// instantiate multiple pipelines from the same `.metal` source. The inner-K
// tile (BK) is fixed at 16, which divides all four T5-base shapes' K values
// (768 and 3072) cleanly.
//
// Threadgroup memory is statically sized to the largest tile combination the
// sweep covers (32 × 32 with BK = 16, totalling 4 KiB). Smaller tiles simply
// under-use the allocation; this wastes a little SRAM but keeps the kernel
// source tile-size-agnostic.
//
// This kernel deliberately avoids advanced optimisations (SIMD-group ops,
// `simdgroup_matrix`, register blocking, double-buffered loads). Per the
// issue, the goal is to land a *simple* tiled kernel and observe whether it
// can credibly approach `MPSMatrixMultiplication`. A simple kernel that
// can't is itself a valid no-go signal.

#include <metal_stdlib>
using namespace metal;

// Tile dimensions in threadgroup-thread space; one thread per output element.
constant uint TG_M [[function_constant(0)]];
constant uint TG_N [[function_constant(1)]];

constant uint BK = 16;

// Statically sized to the maximum tile (32×32) the sweep will use. Smaller
// tile shapes leave the tail of these arrays unused.
constant uint MAX_TILE_M = 32;
constant uint MAX_TILE_N = 32;

kernel void gemm_fp32(
    device const float* A           [[ buffer(0) ]],
    device const float* B           [[ buffer(1) ]],
    device float*       C           [[ buffer(2) ]],
    constant uint&      M           [[ buffer(3) ]],
    constant uint&      K           [[ buffer(4) ]],
    constant uint&      N           [[ buffer(5) ]],
    uint2 lid                       [[ thread_position_in_threadgroup ]],
    uint2 tgid                      [[ threadgroup_position_in_grid ]]
) {
    threadgroup float aTile[MAX_TILE_M * BK];
    threadgroup float bTile[BK * MAX_TILE_N];

    const uint row = tgid.y * TG_M + lid.y;
    const uint col = tgid.x * TG_N + lid.x;

    const uint numThreads = TG_M * TG_N;
    const uint linearTid = lid.y * TG_N + lid.x;

    float acc = 0.0;

    for (uint k0 = 0; k0 < K; k0 += BK) {
        // Cooperative load of A's tile (TG_M × BK) into threadgroup memory.
        const uint aSize = TG_M * BK;
        for (uint i = linearTid; i < aSize; i += numThreads) {
            const uint r = i / BK;
            const uint c = i % BK;
            const uint gRow = tgid.y * TG_M + r;
            const uint gCol = k0 + c;
            aTile[r * BK + c] = (gRow < M && gCol < K)
                ? A[gRow * K + gCol]
                : 0.0f;
        }

        // Cooperative load of B's tile (BK × TG_N) into threadgroup memory.
        const uint bSize = BK * TG_N;
        for (uint i = linearTid; i < bSize; i += numThreads) {
            const uint r = i / TG_N;
            const uint c = i % TG_N;
            const uint gRow = k0 + r;
            const uint gCol = tgid.x * TG_N + c;
            bTile[r * TG_N + c] = (gRow < K && gCol < N)
                ? B[gRow * N + gCol]
                : 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Accumulate this thread's dot-product slice. `lid.y` indexes A's
        // local row; `lid.x` indexes B's local column.
        if (row < M && col < N) {
            for (uint kk = 0; kk < BK; ++kk) {
                acc = fma(aTile[lid.y * BK + kk],
                          bTile[kk * TG_N + lid.x],
                          acc);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}
