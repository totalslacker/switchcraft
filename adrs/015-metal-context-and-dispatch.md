# ADR 015 — Metal context, dispatch, and fallback policy for search-path kernels

**Status**: Accepted
**Date**: 2026-04-30
**Issue**: #51 (sub-issue of umbrella #50, Phase 2 Metal compute shaders for search hot paths)

This ADR records the shape of the shared Metal infrastructure that
search-path kernels (#50: Q4 dequant + matmul, centroid similarity,
residual MaxSim) build on. It locks in the `MetalContext` API,
synchronisation strategy, library-load fallback, transparent
Accelerate fallback, threadgroup-sizing policy, and test-gating policy
*before* any production kernel lands so the three kernel sub-issues can
each pull from a stable foundation.

ADR 014 §(g) names search-side Metal kernels as a Phase 2 ratchet
condition. ADR 012 owns the performance-regression methodology (50
iterations, p50/p95 floors, release-only gating, 300 MB peak-RSS
ceiling) that this scaffolding's perf suite reuses. ADR 009 fixes the
public API surface, which the scaffolding does not extend.

---

## (a) Context

`SwitchcraftCore` had no Metal dependency before issue #51. The three
search hot paths (`SearchEngine.matmulQueryTimesRowMajorTranspose`,
`KMeans` centroid assignment, residual / MaxSim scoring) all run on
`cblas_sgemm` and hand-rolled Swift loops today. The umbrella (#50)
plans three Metal kernels that replace these paths where Metal beats
Accelerate on the same shape.

A standalone investigation target (`SwitchcraftMetalProto`, issue
#49) shipped a complete `MTLDevice` + `MTLCommandQueue` + lazy
`MTLLibrary` scaffold and produced two key empirical findings:

1. SwiftPM 6's `.process(...)` resource declaration ships `.metal`
   files as **raw source**, not as a precompiled `.metallib`.
   `device.makeDefaultLibrary(bundle: Bundle.module)` returns `nil` and
   the consumer must fall back to `device.makeLibrary(source:)` reading
   the `.metal` file as a bundle resource string. (See
   `Sources/SwitchcraftMetalProto/MetalMatmul.swift:86-100` and
   `docs/investigations/metal-matmul-feasibility.md`.)
2. A simple FP32 GEMM Metal kernel cannot beat `cblas_sgemm` on the T5
   shapes (Accelerate dispatches to AMX). The proto's report concluded
   no-go on FP32 GEMM as the embedder path. The Phase 2 search-path
   kernels are a different shape (fused Q4 dequant + matmul; small-`M`
   batched centroid sim; reduction-shaped MaxSim) and their viability
   is independent of that result.

Either way, the scaffolding's correctness depends on the two findings
above being baked in: SwiftPM's resource-handling quirk and a
transparent fallback policy for kernels that don't pan out.

## (b) Decision

**`MetalContext` shape.** Single class living at
`Sources/SwitchcraftCore/Metal/MetalContext.swift`. Holds:

- `MTLDevice` (Apple-documented thread-safe).
- `MTLCommandQueue` (Apple-documented thread-safe).
- `MTLLibrary` (loaded once at `init`; **not** documented thread-safe
  but never mutated post-construction).
- A `[String: MTLComputePipelineState]` cache keyed by Metal function
  name, guarded by `NSLock`.

Exposed as a process-wide lazy singleton via `static let shared:
MetalContext?`. The `static let` initialiser runs exactly once on first
access (Swift Concurrency guarantee), satisfying `MTLLibrary`'s
"no concurrent initialisation" constraint without explicit
double-checked locking. The optional return resolves to `nil` when:

1. `MTLCreateSystemDefaultDevice()` returns `nil` (no GPU).
2. The `SWITCHCRAFT_FORCE_ACCELERATE` env var is set to a truthy value
   (any non-empty string other than `"0"`). Truthy parsing matches the
   existing `MetalProtoGate` pattern.
3. Library load fails (e.g. the placeholder shader had a compile
   error). Logged at debug level via `os.Logger`, never thrown.

`MetalContext` is **internal** to `SwitchcraftCore`. The public API
surface (`SwitchcraftStore`, `Embedder`, `StoreConfig`) is unchanged
per ADR 009. Issue #49's eventual production embedder Metal target may
adopt the same context (the design is kernel-agnostic) or fork it; the
coordination point is acknowledged here, not enforced.

**Sendable policy.** `MetalContext` is `@unchecked Sendable`.
Justification has three legs:

1. `MTLDevice` and `MTLCommandQueue` are documented thread-safe by
   Apple, so storing them in shared state is safe.
2. `MTLLibrary` is **not** documented thread-safe, but we resolve it
   exactly once during `init` (single-threaded by `static let`) and
   never mutate it afterwards. Reads from a fully-initialised,
   never-mutated library are safe in practice across all of Apple's
   shipping toolchains.
3. The pipeline cache is mutable. `pipeline(for:)` holds `cacheLock`
   for the entire critical section (read-or-build-and-store). The lock
   spans pipeline construction so two concurrent callers cannot each
   build their own state and race the writeback (which would leave one
   caller holding an orphan). `MTLLibrary.makeFunction` is undocumented
   for thread safety, which is the second reason the lock spans the
   whole section, not just the dictionary mutation.

`actor` isolation was rejected. Forcing `await` at every kernel
dispatch site would conflict with the synchronous-core tenet
(CLAUDE.md, ADR 009). `SearchEngine`'s hot paths are synchronous and
threading async-only Metal dispatch through them would ripple far
beyond the scaffolding's scope.

**Library load strategy.** Two-step, in order:

1. `device.makeDefaultLibrary(bundle: Bundle.module)`. Works when a
   future toolchain (or a future build configuration) compiles `.metal`
   resources to `default.metallib` at SPM build time.
2. On failure, locate `MetalCoreShaders.metal` in `Bundle.module`, read
   it as a UTF-8 string, and call `device.makeLibrary(source:)`. This
   is the path SwiftPM 6 actually exercises today.

Alternatives rejected:

- **Pre-built `.metallib` shipped as a resource.** SwiftPM has no
  documented build step for compiling `.metal` → `.metallib`. Adding
  a build script just to ship a pre-built artifact reverses the
  build-from-source default and complicates downstream packagers. The
  runtime-compile path is the proven baseline (#49 feasibility report).
- **Pure runtime compilation only (no `makeDefaultLibrary` attempt).**
  `makeDefaultLibrary` is faster on toolchains where the compile path
  *does* work; the fallback chain costs nothing on hosts where it
  doesn't. Try-then-fall-back is robust across toolchain versions.

**Transparent Accelerate fallback.** `runMetalOrFallback(metal:,
fallback:)` (at `Sources/SwitchcraftCore/Metal/MetalDispatch.swift`)
runs the `metal` closure with `MetalContext.shared`; on `nil`, throw,
or command-buffer error it silently runs `fallback`. Metal is an
optimisation, not a correctness requirement — making call sites handle
Metal errors leaks an implementation detail. Failures are logged at
debug level via `os.Logger(subsystem: "com.switchcraft.core",
category: "MetalDispatch")` so test/diagnostic runs can confirm the
fallback fired.

The `metal` and `fallback` closures must produce results equivalent
within the kernel's documented tolerance. Tolerance is per-kernel and
documented when each kernel sub-issue lands.

**Threadgroup sizing.** Fixed constants per kernel, declared at the
kernel-definition site. No runtime tuning. The placeholder
`identity_fp32` kernel uses 64-thread 1D groups; production kernels
specify their own (e.g. the Q4-dequant-and-matmul kernel will likely
land on 16×16 or 32×8 based on its register pressure and threadgroup
memory requirements). Runtime tuning is rejected as out of scope: it
adds per-call cost and complexity for marginal gains, and shipped
Metal code (e.g. ggml's `kernel_mul_mat_q4_K_f32`) standardises on
fixed sizes.

**Test gating.** `MetalAvailability.isAvailable` (in
`Tests/SwitchcraftTests/Metal/MetalKernelTestSupport.swift`) returns
`true` iff `MTLCreateSystemDefaultDevice() != nil` *and*
`SWITCHCRAFT_FORCE_ACCELERATE` is unset. Suites use Swift Testing's
`.enabled(if:)` trait so hosts without Metal (or test runs with the
env var set) skip cleanly without `XCTSkip` noise. Mirrors the
`CoreMLAsset.isAvailable` shape so the patterns stay parallel.

The `MetalContext.shared` singleton is read once per process; tests
that want to exercise the force-Accelerate path do so by running the
test process with the env var set (e.g.
`SWITCHCRAFT_FORCE_ACCELERATE=1 swift test --filter Metal`), not by
mutating the env var mid-suite. Manual verification of that path is
documented in PRs that touch the scaffolding.

**Performance regression suite.** A new `MetalPerformanceTests`
suite (at `Tests/SwitchcraftTests/Performance/MetalPerformanceTests.swift`)
runs alongside ADR 012's existing dims=32 `PerformanceTests`. It uses
`MockEmbedder(dims: 128)` (Witchcraft's projection dimension), 5,000
documents, the same `static let buildOnce` shared-fixture pattern, and
the same percentile-reporting helpers. Memory math:

```
5_000 docs × ~10 tokens × 128 dims × 4 B ≈ 25 MB
```

at the per-doc float ledger, comfortably under ADR 012's 300 MB
peak-RSS ceiling. Both suites run; the dims=128 suite does not replace
the dims=32 suite. In the scaffolding sub-issue the new suite contains
no kernel pass/fail floors — those land with each kernel sub-issue —
only the harness, fixture, and two wiring smoke cases that confirm the
end-to-end pipeline (test → corpus build → search; test → MetalContext
→ command buffer → readback) is intact.

## (c) Build / packaging

`Package.swift` extends the `SwitchcraftCore` target with:

```swift
.target(
    name: "SwitchcraftCore",
    path: "Sources/SwitchcraftCore",
    resources: [.process("Metal/Shaders")],
    linkerSettings: [
        .linkedFramework("Metal",
                         .when(platforms: [.macOS, .iOS, .visionOS])),
    ]
),
```

Mirrors the `SwitchcraftCoreML` linker-settings shape from
`Package.swift:54-61`. `MetalPerformanceShaders` is **not** linked
here — deferred to the first sub-issue that actually uses an MPS
primitive (per the issue spec).

## (d) Coordination with #49 (embedder Metal work)

ADR 014 §(g) anticipated that the embedder Metal path and the search
Metal path may share a context. `MetalContext` is **kernel-agnostic**
(no per-kernel cache keys, no shape-specific function constants in the
core API) so the eventual production embedder Metal target can adopt
it. The `SwitchcraftMetalProto` library is products-excluded
investigation code; whether to delete or keep-as-evidence is out of
scope here.

## (e) Consequences

- Search-path call sites that adopt Metal (sub-issues #2–#4 of the
  umbrella) need to thread `runMetalOrFallback(metal:, fallback:)`
  around their existing Accelerate path. The fallback shape stays
  testable on no-Metal hosts via the env var.
- Kernel correctness is per-kernel. The scaffolding gives sub-issues
  a place to drop `cosineSimilarity ≥ 0.99999` + `maxAbsError <
  tolerance` assertions but does not fix the tolerances — those are
  per-kernel decisions documented in each sub-issue's own ADR or
  commit body.
- The dims=128 perf suite must stay under the 300 MB ceiling. If a
  future change pushes it over, the response is to reduce the doc
  count (2,500 docs → ~12.5 MB), not to relax ADR 012 thresholds.
- The runtime-source library load makes `swift build` order matter:
  the `.metal` file must compile cleanly under `device.makeLibrary
  (source:)` *and* under any future `default.metallib` build step.
  The scaffolding's placeholder kernel is intentionally trivial so
  this hasn't been a constraint yet; production kernels need to be
  vetted for both.

## (f) Open questions deferred

- **No-Metal CI job.** The macos-15 GitHub Actions runner has Metal,
  so the skip path is not exercised by default CI today. Whether to
  add a third CI job that runs `SWITCHCRAFT_FORCE_ACCELERATE=1 swift
  test` is deferred to a follow-up. The scaffolding makes the path
  *exercisable* (the env var works as documented); the question is
  only how often we run it automatically.
- **`MTLBuffer` pooling.** Deferred to the first kernel sub-issue
  whose profile actually shows allocator pressure. Per-call
  `device.makeBuffer(...)` with `.storageModeShared` is the default.
- **Shipping a pre-built `.metallib`.** Reconsidered if a future
  SwiftPM toolchain compiles `.metal` resources at build time
  cleanly; the runtime-compile fallback then becomes a no-op safety
  net.

## References

- Issue #50 (umbrella) and #51 (this scaffolding sub-issue).
- `docs/investigations/metal-matmul-feasibility.md` — issue #49's
  empirical SwiftPM resource and FP32-GEMM findings.
- ADR 009 — public API shape (frozen here).
- ADR 012 — performance regression methodology (reused here).
- ADR 014 §(g) — search-side Metal kernels as a Phase 2 ratchet.
- `Sources/SwitchcraftMetalProto/MetalMatmul.swift:59-101` — direct
  prior art for the `MetalContext` shape.
- `Tests/SwitchcraftTests/Support/CoreMLAsset.swift` — direct prior
  art for the `MetalAvailability.isAvailable` shape.
- `Tests/SwitchcraftTests/Performance/PerformanceTests.swift` — direct
  prior art for the perf-suite shape.
