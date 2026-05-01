**No-go** — see Recommendation for full reasoning.

# Custom Metal matmul feasibility for the T5 embedder hot path

## Recommendation

**No-go.** A simple FP32 tiled Metal compute shader cannot meet the bar on
the largest T5-base shape. Against `MPSMatrixMultiplication` it lands at
~69% of p50 (need ≥ 85%); against Accelerate `cblas_sgemm` — the path the
production `SearchEngine` already uses for centroid scoring — it is **~3.6×
slower**, where the bar required ≥ 1.5× faster. More importantly, the
investigation surfaced an unexpected primary finding: on M1 Ultra
**Accelerate's `cblas_sgemm` outperforms `MPSMatrixMultiplication` by
roughly 2.5×** for these shapes, almost certainly because Accelerate
dispatches to the Apple Matrix coprocessor (AMX) while MPS is bounded by
GPU memory bandwidth on a single 512-row left-hand matrix. Replacing CoreML
matmuls with hand-written Metal therefore has no plausible path to
ggml-equivalent performance on this hardware tier; the relevant
follow-on path is the one already documented in `docs/Plan.md`'s "If the
matmul investigation fails" subsection (accept the ~1.5–2× embedder gap
with Witchcraft, lean on Phase 2's other optimisations). No ADR amendment
is requested in this PR; ADR 014(g)'s "custom Metal kernels" ratchet
condition is now empirically falsified for FP32 GEMM, which a separate
direct-to-main commit can record.

## Hardware

| Field | Value |
|------|------|
| Chip | Apple M1 Ultra |
| Memory | 128 GB |
| OS | macOS 26.4.1 (Build 25E253) |
| Swift | 6.0 / `swift-tools-version: 6.0` |
| Metal compile | Runtime via `MTLDevice.makeLibrary(source:)` (SwiftPM 6 ships `.metal` resources as source, not metallib) |

The full `BenchmarkRun.HardwareInfo` is committed alongside the latency
data at `metal-matmul-feasibility-figures/results.json` and
`metal-matmul-feasibility-figures/threadgroup-sweep.json`. Numbers are
from a single M1 Ultra; cross-chip data is explicitly out of scope per
the issue (the gap to the bar is so large here that finer chips cannot
plausibly close it without a fundamentally different kernel design).

## Methodology

The prototype lives in the standalone, products-excluded SPM library
`SwitchcraftMetalProto`. Three matmul backends are exercised through a
shared timing harness:

- **Accelerate (`cblas_sgemm`)** — invoked through the production
  `SearchEngine.matmulQueryTimesRowMajorTranspose(...)` path so the
  numbers reflect what `SearchEngine.gemm` and `KMeans.batchedArgmax`
  actually do at runtime, not a clean-room re-wrap. `B` is pre-transposed
  outside the timed loop so the GEMM call alone is measured.
- **MPS (`MPSMatrixMultiplication`)** — `MPSMatrixDescriptor` /
  `MPSMatrix` set up against `.storageModeShared` `MTLBuffer`s; encode +
  commit + `waitUntilCompleted()` per iteration. All four T5-base shapes
  satisfy the 16-byte `rowBytes` alignment requirement by construction
  (4 B × 128 / 768 / 3072 are all multiples of 16); no padding workaround
  was needed.
- **Custom FP32 Metal kernel** — `Sources/SwitchcraftMetalProto/MetalMatmul.metal`
  is a tiled GEMM with threadgroup-shared `aTile`/`bTile` (BK = 16),
  one thread per output element, and `TG_M × TG_N` set per-pipeline via
  function constants. Pipeline-state objects are cached across iterations.

Inputs are generated from `SwitchcraftCore.SplitMix64` with fixed
per-shape seeds (see `Tests/SwitchcraftMetalProtoTests/Support/BenchmarkInputs.swift`)
so all numbers are reproducible from committed code.

The full T5-base CoreML matmul path was **not** isolated as a separate
backend (issue Open Question, option (c)). The production `MLModel`
consumes integer token IDs rather than raw FP32 matrices, so isolating a
single matmul layer would have required a one-layer synthetic
`.mlpackage` via `coremltools`. Given the gap between Accelerate and
MPS observed below, the additional data point would not change the
go/no-go signal; full-encode T5 numbers from `T5CoreMLEmbedderTests`
remain available as the production-CoreML ballpark if needed.

Timing uses `mach_absolute_time` with `mach_timebase_info` conversion.
Each shape × backend cell records 100 iterations after 10 warmup runs
(50 / 5 for the threadgroup sweep). p50 / p95 / mean / min are reported.
The MPS backend caches one `MPSMatrixMultiplication` operator per `(M,K,N)`
across iterations, so the timed loop measures encode + dispatch + wait
rather than per-iteration MPS object construction. Note that buffer
allocation and the host→device copy of `A`/`B` *are* still inside the
timed window for both Metal backends — this matches the prototype's
"prove a hand-written kernel can win on a fresh call" question; resident-
buffer steady-state is out of scope here and noted as a follow-on.

Correctness is gated at cosine ≥ 0.99999 against the `cblas_sgemm`
reference for both Metal backends, on all four shapes plus the
threadgroup sweep tiles. All pass:
`SWITCHCRAFT_METAL_PROTO=1 swift test --filter SwitchcraftMetalProtoTests`
runs ten parameterised cases with no failures (verified prior to writing
this report).

### Reproducing the numbers

```sh
# Correctness (fast):
SWITCHCRAFT_METAL_PROTO=1 swift test --filter SwitchcraftMetalProtoTests

# Benchmark + threadgroup sweep (release-only):
# Wall time depends on chip; on M1 Ultra the suite runs in ~5–10s after
# release build amortisation, dominated by the threadgroup sweep on `ffn_up`.
SWITCHCRAFT_METAL_PROTO_BENCH=1 swift test -c release \
  --filter SwitchcraftMetalProtoTests
```

Default `swift test` (no env vars) skips the entire prototype suite — the
benchmark gate requires both `SWITCHCRAFT_METAL_PROTO_BENCH=1` and a
release build; `SWITCHCRAFT_METAL_PROTO_RESULTS_DIR` redirects JSON
output if the worktree's figures dir is read-only.

## Latency table

p50 / p95 / mean in milliseconds; GFLOPS computed from the min-latency
sample for each cell (matches how MPS-published numbers are quoted).

| Shape (M × K × N) | Backend | p50 (ms) | p95 (ms) | Mean (ms) | GFLOPS |
|---|---|---:|---:|---:|---:|
| `qkv_proj` (512 × 768 × 768)   | accelerate `cblas_sgemm` | **0.265** | 0.558 | 0.314 | 2613.7 |
| `qkv_proj`                     | `MPSMatrixMultiplication` | 1.071 | 1.461 | 1.104 | 642.3 |
| `qkv_proj`                     | custom Metal (tile 16×16) | 2.581 | 4.556 | 2.555 | 430.4 |
| `ffn_up` (512 × 768 × 3072)    | accelerate `cblas_sgemm` | **1.142** | 1.891 | 1.332 | 2221.5 |
| `ffn_up`                       | `MPSMatrixMultiplication` | 2.823 | 3.261 | 2.862 | 990.6 |
| `ffn_up`                       | custom Metal (tile 16×16) | 4.109 | 4.855 | 4.198 | 683.0 |
| `ffn_down` (512 × 3072 × 768)  | accelerate `cblas_sgemm` | **1.317** | 2.389 | 1.605 | 2098.8 |
| `ffn_down`                     | `MPSMatrixMultiplication` | 2.604 | 3.544 | 2.754 | 1111.7 |
| `ffn_down`                     | custom Metal (tile 16×16) | 3.914 | 10.078 | 4.281 | 800.0 |
| `token_proj` (512 × 768 × 128) | accelerate `cblas_sgemm` | **0.107** | 0.113 | 0.108 | 952.3 |
| `token_proj`                   | `MPSMatrixMultiplication` | 0.552 | 0.599 | 0.557 | 192.0 |
| `token_proj`                   | custom Metal (tile 16×16) | 0.543 | 0.613 | 0.549 | 193.0 |

Source data: `metal-matmul-feasibility-figures/results.json`.

## Throughput on the largest shape

`ffn_up` is the most demanding projection (`2 · 512 · 768 · 3072 ≈ 2.42
GFLOP per call`):

- Accelerate `cblas_sgemm`: **~2222 GFLOPS** (min ~1.09 ms → ~2.2 TFLOPS)
- `MPSMatrixMultiplication`: **~991 GFLOPS**
- Custom Metal (default tile, 16×16): **~683 GFLOPS**; sweep best (tile 8×32): **~622 GFLOPS p50** / 800 GFLOPS on `ffn_down`

For context, ggml's published M1-class FP32 GEMM numbers (`ggml-bench`,
`llama.cpp` profiling) sit in roughly the **600–900 GFLOPS** band on
similar shapes. Our hand-written kernel at ~700 GFLOPS is in the same
ballpark as ggml's vanilla Metal FP32 path — i.e. a *simple* tiled kernel
on an M1 Ultra is already roughly where ggml lives. **The gap to ggml's
*actually competitive* path is in its quantised kernels (`mul_mat_q4_K_f32`
et al.) — a different campaign entirely**, see the conclusion.

## Threadgroup-size sweep

Sweep on `ffn_up` (50 iterations, 5 warmup):

| Tile (TG_M × TG_N) | p50 (ms) | p95 (ms) | GFLOPS | vs best |
|---|---:|---:|---:|---:|
| 8 × 32  | **4.069** | 4.800 | **622.2** | 100% |
| 32 × 8  | 4.290 | 5.518 | 682.2 | 95% |
| 16 × 16 | 4.414 | 4.731 | 605.1 | 92% |
| 8 × 8   | 6.080 | 6.429 | 410.8 | 67% |
| 32 × 32 | 6.389 | 6.901 | 522.3 | 64% |
| 8 × 4   | 11.298 | 15.128 | 252.2 | 36% |

Source data: `metal-matmul-feasibility-figures/threadgroup-sweep.json`.

The plateau spans 8 × 32, 32 × 8, and 16 × 16 within ~10% of each other.
Larger tiles (32 × 32) under-occupy the GPU because 4 KiB of threadgroup
memory is enough that fewer threadgroups schedule per SM; smaller tiles
(8 × 8, 8 × 4) leave the SIMD groups mostly idle. None of these tiles
changes the headline verdict — the best simple tiled kernel is still
~3× behind cblas and ~31% behind MPS on the largest shape.

## Performance verdict relative to the bar

The Plan-stage go/no-go bar (largest shape only, both clauses required):

| Clause | Threshold | Custom Metal best | Verdict |
|---|---|---|---|
| ≥ 85% of MPS p50 | p50 ≤ 2.823 / 0.85 = **3.321 ms** | **4.069 ms** (~69% of MPS p50) | **fail** |
| ≥ 1.5× `cblas_sgemm` | p50 ≤ 1.142 / 1.5 = **0.761 ms** | **4.069 ms** (~0.28× cblas, i.e. 3.6× slower) | **fail (by large margin)** |

Even with an aggressive recalibration of the bar (e.g. requiring only
parity with MPS, or relaxing to ≥ 1.0× cblas), the custom kernel does
not close the gap. The threadgroup sweep above already exhausted the
simple-kernel knobs in scope; closing the remaining 30% to MPS would
require SIMD-group operations / `simdgroup_matrix` / register blocking,
which are explicitly out of scope per the issue and which, on past
experience with similar GEMM kernels, do not in practice push past
MPS's own implementation.

## Closeability of the gap to ggml-equivalent performance

Witchcraft's published embedder-side speed advantage is **not** primarily
attributable to ggml's FP32 matmul. ggml's FP32 Metal kernel is in the
same ~600–800 GFLOPS band our prototype reaches. The gap comes from two
places that have nothing to do with FP32 GEMM:

1. **Quantised matmul kernels** (`mul_mat_q4_K_f32` and its siblings)
   that operate directly on Q4-quantised weights without dequantising
   to FP32 first. These are not GEMM; they are fused
   dequantise-and-multiply-and-accumulate kernels on a bespoke memory
   layout. Implementing them is a separate campaign that this
   investigation neither benchmarks nor de-risks.
2. **Per-op precision routing** (ADR 014(i)): RMSNorm / softmax / residual
   add at FP32, weights at FP16/Q4, outputs in whatever the consumer
   wants. This is a graph-orchestration property, not a kernel-perf
   property. It would belong in a hand-rolled embedder runtime, not in
   the matmul kernel.

In short, even a *competitive* hand-written FP32 matmul (i.e. one that
matches MPS) would not move the embedder needle the way the
custom-Metal-kernels strategic narrative implies. The lever is
quantised compute and per-op precision; FP32 matmul is a wash.

## Surprise finding: cblas beats MPS on M1 Ultra

The most actionable observation in this investigation has nothing to do
with the custom kernel: **Accelerate's `cblas_sgemm` is ~2–5× faster
than `MPSMatrixMultiplication` on M1 Ultra for these shapes**
(`ffn_up`: 1.14 ms vs 2.82 ms; `ffn_down`: 1.32 ms vs 2.60 ms;
`token_proj`: 0.11 ms vs 0.55 ms — the small-N case is particularly
lopsided). The most likely explanation is that Accelerate dispatches to
the Apple Matrix coprocessor (AMX), which is undocumented but known to
deliver ~2 TFLOPS of FP32 throughput per p-core cluster on M1-class
silicon, while MPS is bounded by GPU memory bandwidth on the
left-hand-matrix-stays-resident pattern these shapes exhibit. This
implies that:

- For Switchcraft's existing `SearchEngine.matmulQueryTimesRowMajorTranspose`
  call sites (centroid scoring, `KMeans.assign`), there is no Metal path
  worth pursuing — Accelerate already wins.
- A future quantised-matmul campaign should compare against
  AMX-on-CPU as well as MPS; on M1 Ultra in particular, "GPU" is not
  obviously the right target for FP32 work at these batch sizes.

Whether this generalises to A-series / smaller M-series chips is unknown
from this investigation alone (single-machine result per the issue scope).

## Conclusion

A simple FP32 tiled Metal kernel does not approach the go/no-go bar on
M1 Ultra; the threadgroup sweep already exhausted the in-scope
optimisation knobs. The hand-written matmul campaign is therefore
**not worth starting** as scoped, and the broader "custom Metal kernels
for the T5 embedder" path documented in `docs/Plan.md` is, on this
hardware tier, falsified for FP32 — though it remains technically
unfalsified for the *quantised* kernel path that is closer to what
ggml-Metal actually does in production. That path is a separate ~6–8
week campaign with a fundamentally different kernel design (Q4 layout,
fused dequant-multiply-accumulate, dedicated SIMD-group ops); it is
neither de-risked nor falsified by this report.

The applicable next step is the fallback path in `docs/Plan.md` "If the
matmul investigation fails": stay on the FP32-compute / FP16-output
CoreML embedder (ADR 010(c), per ADR 014(i)'s reasoning), accept the
~1.5–2× embedder-latency gap with Witchcraft, and lean on Phase 2's
other optimisations (parallel bucket search, indexer parallelism, INT8w
asset). The unexpected cblas-beats-MPS finding is an incidental win
worth recording so that any future GPU-matmul proposal for an existing
Switchcraft hot path starts from "we already have a faster CPU path."
