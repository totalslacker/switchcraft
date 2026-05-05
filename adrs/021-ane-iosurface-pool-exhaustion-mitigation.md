# ADR 021 — ANE IOSurface Pool Exhaustion Mitigation

## Status

Accepted

## Context

`T5CoreMLEmbedder` failed irrecoverably after ~1,000 successful inferences in
long-running bulk-index workloads. Every subsequent inference threw:

```
NSGenericException: Failed to allocate E5 buffer object.
E5RT: Failed to allocate memory IOSurface object. (3)
```

Once the failure started, 100% of subsequent inferences failed regardless of
input size. The host process had ~12 GB of free RAM — the exhaustion was
specific to the ANE IOSurface buffer pool, not main memory.

Key characteristics of the failure mode (from SafariUnfucker bulk-index run,
2026-05-05):
- **First failure at minute ~48** (~1,000 successful encodes).
- **Failure input-size distribution is broad** — not size-correlated. All input
  lengths (1–50k+ characters) failed equally after the pool exhausted.
- **`ANECompilerService` RSS ~700 MB** — consistent with CoreML compile/recompile
  churn under sustained load.
- **No self-recovery** — only restarting the host process cleared the state.

## Decision

Apply three defence layers in increasing cost order. All three are required; no
single layer is sufficient under sustained load:

### Layer 1 — `autoreleasepool` per window (unconditional)

Wrap every call to `predictor.predict(input:)` in an `autoreleasepool { }` block.

CoreML-internal Objective-C objects (ANE IOSurface buffers, Metal command
buffers) are managed by the ObjC autorelease mechanism. Under ARC, the implicit
autorelease pool only drains at a thread's run-loop iteration boundary — not
after every call in a tight Swift loop running in an actor context. Explicit
`autoreleasepool` boundaries ensure these objects are freed immediately after
each window prediction, not batched until the run-loop drains.

**Cost**: negligible. No allocation, no I/O, no stall.

### Layer 2 — Proactive model reload every `reloadInterval` encodes

Recreate the `MLModel` instance every `reloadInterval` encode calls by calling
the stored `predictorFactory` closure. This flushes any accumulated ANE
resources before the pool exhausts.

The `reloadInterval: Int = 500` parameter is configurable so callers can trade
stall frequency against pool pressure for their specific workload. The default
of 500 sits comfortably below the observed ~1,000-call failure point. Each
reload with ANE compilation takes 1–3 s on Apple Silicon; subsequent reloads
may be faster due to OS-level model compilation caching.

If the factory throws during a scheduled reload, the existing predictor is kept
and the error is logged at `error` severity. A transient compilation failure
should not abort a healthy bulk-index run.

**Cost**: 1–3 s stall every `reloadInterval` encodes.

### Layer 3 — Reactive CPU fallback on IOSurface exception

When an IOSurface allocation exception is caught despite Layers 1 and 2, retry
the failing window using a transient `.cpuOnly` model created by the
`cpuPredictorFactory` closure (wired to the same compiled model URL with
`.cpuOnly` compute units). If the CPU retry succeeds:
- A `"recovered_iosurface_exhaustion"` JSONL row with `"category": "warning"`
  is appended to `failureLogURL` (when set).
- The inference result is returned normally; the caller receives no error.

If the CPU retry also fails, the original IOSurface error is re-thrown.

This layer is a last resort for any exhaustion events that slip through
Layers 1 and 2. It does not permanently degrade throughput — the transient
CPU model is created for the failing window only and discarded afterward.

**Cost**: ~100–300 ms CPU model creation latency for the one failing window.

## Key Design Decisions

### Transient CPU model, not persistent predictor swap

Swapping `self.predictor` to `.cpuOnly` after the first IOSurface failure would
degrade every subsequent window and encode until the next scheduled reload.
A transient model scopes the performance hit to the one failing window and
simplifies the state machine (no "CPU-mode" flag, no special-case resume logic).

### Encode-level reload counter, not per-window

`callCount` increments once per `encode` call, not once per sliding-window
iteration. Real-world evidence counts ~1,000 "inferences" (encode calls) before
failure, not 1,000 window iterations. Per-window counting would trigger reloads
far too frequently on long inputs with many windows.

### `reloadInterval = 500` default

500 is approximately half of the observed ~1,000-call failure point, providing
comfortable margin. It is intentionally below the failure point: setting it at
or above the failure point defeats the proactive reload's purpose. The doc-comment
on the stored property explains this rationale so future tuners have context.

### `predictorFactory` closure, not a factory method on `MLPredictor`

Adding a factory method to the `MLPredictor` protocol would require updating all
existing stub predictors (`SlowStubPredictor`, `ThrowingStubPredictor`, the new
`CountingStubPredictor`). A closure stored on the actor is self-contained,
introduces no protocol surface changes, and is trivially injectable from both
the real public inits and the test-only internal inits.

### `cpuPredictorFactory` is `nil` in test inits

Test inits that inject a static stub predictor have no `compiledURL` to pass to
`MLModel(contentsOf:)`. Rather than making CPU fallback mandatory, the
`cpuPredictorFactory` is optional (`nil` in test stub inits). The test for CPU
fallback exercises the path by injecting a separate `cpuPredictorFactory` that
returns a succeeding `CountingStubPredictor`.

## Alternatives Considered

- **`autoreleasepool` alone**: Apple DTS guidance and community reports confirm
  this defers but does not prevent pool exhaustion under sustained load. Excluded
  as the sole fix.
- **Persistent `.cpuOnly` fallback after first failure**: Permanently degrades
  throughput. Excluded in favour of transient per-window retry.
- **Only periodic reload without `autoreleasepool`**: Leaves a gap between reload
  intervals where pool pressure can still accumulate. Excluded as insufficient.

## Consequences

- `T5CoreMLEmbedder` now supports ≥10,000 consecutive encode calls without
  IOSurface failure under the three-layer defence.
- Public API gains `reloadInterval: Int = 500` parameter on `init(modelURL:...)`
  and `init(bundle:...)`. Default is backward-compatible; existing callers are
  unaffected.
- Bulk-index operators may observe 1–3 s stalls every 500 encodes (proactive
  reload). `reloadInterval` can be tuned upward if the stall is unacceptable.
- `failureLogURL` JSONL schema gains an additive `category` field (`"error"` for
  terminal failures, `"warning"` for recovered IOSurface events).

## References

- ADR 010 §(g) — sanctions `.cpuOnly` compute-unit override for constrained
  environments; CPU fallback is consistent with this precedent.
- ADR 018 — ObjC `@try/@catch` bridge pattern (`catchingNSException`); the same
  bridge wraps the CPU-fallback retry.
- Issue #87 — original failure report with `coreml-failures.jsonl` evidence.
