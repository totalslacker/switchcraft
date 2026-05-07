# ADR 021 — ANE IOSurface Pool Exhaustion Mitigation

## Status

Amended (2026-05-06 — see addendum below)

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

## 2026-05-06 Addendum (Issue #90)

### Production evidence invalidated key assumptions

A SafariUnfucker bulk-index run against the post-#88 binary (issue #90 report) showed:

- **388 successful inferences, then 6,157 consecutive failures** — 100% identical IOSurface
  error, zero `category: "warning"` / `recovered_iosurface_exhaustion` entries.
- The first failure had `inputLength=593285` tokens (~600k). After this single oversized
  page the pool was poisoned for all subsequent inputs regardless of size.
- **The default `reloadInterval: 500` fired after the real-world failure point**, rendering
  proactive reload ineffective for this corpus.
- **CPU fallback silently failed** — the code logged `category: "error"` whether recovery
  was attempted or not, providing no diagnostic signal.
- **Reactive reload + ANE retry was missing entirely** — Layer 3 jumped directly to CPU
  fallback without first trying a fresh ANE model.

### Changes shipped in issue #90

**`reloadInterval` default lowered to 150.** 150 is ~2.6× below the observed 388-call
failure point, providing meaningful margin. The "≥10,000 consecutive encode calls" claim
in the Consequences section above applies to uniform small-input corpora. High-variance
corpora (e.g., web pages with a few very long inputs) exhaust the pool faster. The
default must sit below the observed real-world failure point, not be calibrated to a
single prior corpus.

**Three-state `category` scheme for JSONL rows.** The original design had two states
(`"error"`, `"warning"`). A third state is now needed:
- `"warning"` (`"recovered_iosurface_exhaustion"`): CPU fallback succeeded — unchanged.
- `"cpu_fallback_failed"`: CPU fallback was attempted but also failed. The row includes
  `cpuErrorName`, `cpuErrorReason`, `cpuCallStack` fields capturing the CPU-side error.
- `"error"`: IOSurface was not the cause, or `cpuPredictorFactory` is nil — unchanged.

**Reactive reload + ANE retry added to Layer 3.** On IOSurface failure, `predictWindow`
now force-reloads the predictor via `predictorFactory()` and retries on ANE before falling
back to CPU. This prevents a single poisoned inference from making the pool irrecoverable
for subsequent calls. Note: if the ANE IOSurface pool is process-global, the reload may
not recover it — but the CPU fallback still fires, and the new logging will confirm this.

**Adaptive byte-pressure reload threshold is a follow-up item.** A more principled
long-term solution tracks `inputLength × dims × 4 bytes` accumulated per call and reloads
when it exceeds a threshold (size-dominated exhaustion, not call-count-dominated). This
is out of scope for issue #90 and should be filed as a separate issue.

### References

- Issue #90 — production failure report and bug fixes.
- Issue #89 — parallel input-size guard (structural prevention of size-driven trigger).

## 2026-05-07 Addendum (Issue #93)

### Production evidence: Layer 3b CPU fallback has 0% recovery rate

A SafariUnfucker bulk-index run (2026-05-07) processed 5,570 successful inferences
and 1,038 failures:

```
1038  cpu_fallback_failed  (every single failure)
1038  "Failed to allocate E5 buffer object. E5RT: Failed to allocate memory IOSurface object. (3)"
0     recovered_iosurface_exhaustion  (none)
```

Every IOSurface exception that reached Layer 3b resulted in `cpu_fallback_failed`
— zero recoveries. The ANE pool DID self-recover three times during the run, with
recovery periods of 5–25 minutes between failure bursts. Recovery timing is
consistent with the proactive reload (Layer 2) firing at 500-call boundaries, not
with the CPU fallback.

### Decision: Remove Layer 3b CPU fallback

Layer 3b is removed. The production evidence is determinative: 0/1,038 recovery
rate across three distinct failure bursts, with per-failure overhead of one extra
`MLModel` load and `predict` call that contributed nothing.

After this change, the three-layer model becomes a two-layer model:

1. **Layer 1** — `autoreleasepool` per window (unchanged).
2. **Layer 2** — Proactive model reload every `reloadInterval` encodes (unchanged).
3. **Layer 3a** — Reactive reload + ANE retry on IOSurface failure. If the ANE
   retry also fails, the original error is logged with `category: "error"` and
   rethrown. No CPU fallback is attempted.

### Simplified JSONL category scheme

The three-state scheme from the 2026-05-06 addendum is reduced to two active
categories:

| Category | Meaning |
|----------|---------|
| `"error"` | Inference failed. Includes both non-IOSurface failures and IOSurface failures where the Layer 3a ANE retry also failed. |
| `"warning"` | (Reserved; not currently produced.) |

The `"cpu_fallback_failed"` category is retired. It will no longer appear in
`failureLogURL` JSONL output. The `cpuErrorName`, `cpuErrorReason`, and
`cpuCallStack` fields remain in `CoreMLFailureLogEntry` (out of scope to remove)
but will always be absent (`nil`) from JSON output going forward.

### ADR 010 §(g) no longer applies to the Layer 3 path

ADR 010 §(g) sanctions `.cpuOnly` compute-unit override for constrained
environments. This sanction was cited in the original ADR 021 to justify the CPU
fallback's use of `.cpuOnly`. With Layer 3b removed, ADR 010 §(g) no longer
applies to the Layer 3 path. The citation is noted here so future readers know
it was intentionally vacated, not overlooked.

### References

- Issue #93 — production evidence (0/1,038 CPU recovery rate) and Layer 3b removal.
