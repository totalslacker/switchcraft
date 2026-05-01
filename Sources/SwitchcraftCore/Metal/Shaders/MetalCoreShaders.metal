// SPDX-License-Identifier: Apache-2.0
//
// Placeholder Metal shaders for the search-path scaffolding (issue #51).
//
// This file exists to validate the SwiftPM Metal build pipeline end-to-end —
// shader compile, library load, function lookup, pipeline-state creation,
// command-buffer dispatch, commit, wait, and readback — *before* any
// production kernel lands. Subsequent sub-issues drop their kernels into
// this directory and reuse the same `MetalContext` infrastructure.
//
// `identity_fp32` copies `in[i]` to `out[i]` for `i < count`. It does no
// useful work; it exists solely so the scaffolding's wiring smoke test has
// something to dispatch.
//
// Threadgroup size is fixed at 64 (1D) — see ADR 015 §"Threadgroup sizing".

#include <metal_stdlib>
using namespace metal;

kernel void identity_fp32(
    device const float* in   [[ buffer(0) ]],
    device float*       out  [[ buffer(1) ]],
    constant uint&      count [[ buffer(2) ]],
    uint                gid  [[ thread_position_in_grid ]]
) {
    if (gid < count) {
        out[gid] = in[gid];
    }
}
