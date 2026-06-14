# ADR 031 — `Embedder.resetState()`: Explicit Consumer-Controlled ANE IOSurface Pool Flush

## Status

Accepted

## Context

### The IOSurface pool exhaustion failure mode

Under sustained bulk-inference load, the Apple Neural Engine (ANE) accumulates
`IOSurface` buffers that are not promptly released by CoreML. After approximately
388–1,100 sequential `encode()` calls inside a single process (the range depends on
input size diversity), every subsequent inference fails with:

```
nativeException(
  name: "NSGenericException",
  reason: "Failed to allocate E5 buffer object. E5RT: Failed to allocate memory
           IOSurface object. (3)",
  callStack: [
    "0   CoreFoundation   __exceptionPreprocess + 176",
    "1   libobjc.A.dylib  objc_exception_throw + 88",
    ...
    "3   CoreML           MLE5BindEmptyMemoryObjectToPort + 1672",
    "4   CoreML           -[MLE5OutputPortBinder bindAndReturnError:] + 200"
  ]
)
```

This is a hard failure — no self-recovery without releasing the `MLModel` instance.
The throughput degradation before the crash is the canonical "pool filling up" signature:

| Inference count | Cumulative wall time | Avg time/call |
|---|---|---|
| 100 | 200s | 2.0s |
| 500 | 1,498s | 3.0s |
| 1,000 | 3,513s | 3.5s |
| ~1,100 (crash) | 3,933s | ~5s |

### Existing defences (ADR 021)

`T5CoreMLEmbedder` already implements two internal mitigation layers:

- **Layer 2**: Proactive model reload every `reloadInterval` encodes (default 150;
  lowered from 500 after the 388-call production failure point). Flushes ANE resources
  before pool exhaustion.
- **Layer 3**: Reactive reload + ANE retry on IOSurface allocation failure. If the ANE
  retry also fails, the original error is rethrown.

These layers mitigate the problem for typical workloads but do not give consumers
explicit control over when the pool flush occurs. A consumer doing bulk indexing of
tens of thousands of documents across multi-hour runs cannot predict when the internal
`reloadInterval` fires relative to their batch boundaries.

### Why `SwitchcraftStore` teardown is the wrong fix

Without an explicit reset API, consumers must destroy the entire `SwitchcraftStore` to
release the IOSurface pool, which forces:

- SQLite connection close and reopen
- LSM ledger rehydration from storage
- Storage state cache invalidation
- Loss of any in-flight buffered writes

These costs are disproportionate to releasing a resource pool that lives exclusively in
CoreML's ownership.

## Decision

Add `resetState() async throws` to the `Embedder` protocol with a default no-op
implementation. `T5CoreMLEmbedder` overrides it to tear down and reload its `MLModel`
via the existing `predictorFactory` closure, resetting the IOSurface pool and
`callCount` to 0.

This is consumer-controlled Layer 0 — an explicit, on-demand flush that callers invoke
between batches. It does not replace the internal Layer 2/3 defences; those remain
active as safeguards for workloads that don't call `resetState()`.

## API design

### Protocol declaration

```swift
public protocol Embedder: Sendable {
    // ... existing requirements ...
    func resetState() async throws
}

public extension Embedder {
    func resetState() async throws { /* no-op */ }
}
```

The default no-op in `public extension Embedder` gives zero-cost conformance to
`MockEmbedder`, `T5MetalEmbedder`, and any future conformer. Metal GPU memory does not
accumulate ANE IOSurface state; no explicit reset is required for `T5MetalEmbedder`.

### `T5CoreMLEmbedder` implementation

`resetState()` hooks into the same `inFlight`/`waiters` re-entrancy guard used by
`_encodeImpl` (see ADR 028 for the guard's origin):

```swift
public func resetState() async throws {
    // Same re-entrancy guard pattern as _encodeImpl.
    if inFlight {
        await withCheckedContinuation { (cont: CheckedContinuation<Void, Never>) in
            waiters.append(cont)
        }
    }
    inFlight = true
    defer {
        if let next = waiters.first {
            waiters = Array(waiters.dropFirst())
            inFlight = false
            next.resume()
        } else {
            inFlight = false
        }
    }

    let newPredictor = try predictorFactory()
    self.predictor = newPredictor
    self.callCount = 0
}
```

The guard ensures:
1. Concurrent `encode()` calls in-flight when `resetState()` is called complete before
   the reset executes.
2. `encode()` calls that arrive after `resetState()` starts queue and resume after the
   reset finishes.
3. No deadlock: `resume()` runs the continuation synchronously within the actor until
   the next `await`, so there is no window where `inFlight` is unset between a drain
   and the next holder setting it.

## Determinism contract

`resetState()` is functionally transparent:

> For any input `s`, `encode(s)` returns byte-identical `[Float]` before and after a
> `resetState()` call.

The pool-flush is a runtime resource concern. Model weights, tokenization, and the
projection layer are unaffected. `CountingStubPredictor` produces deterministic output
(`1/√dims` per element), making the determinism property testable without the real
CoreML asset.

## Error propagation semantics

Unlike the Layer 2 proactive reload (which catches factory errors, logs them, and keeps
the old predictor), `resetState()` propagates factory errors to the caller. If
`predictorFactory()` throws:

- The error propagates immediately.
- `self.predictor` retains the old (stale) reference — the new predictor was never
  assigned.
- The `defer` block still fires, clearing `inFlight` and draining the waiter queue.
  Subsequent `encode()` calls proceed against the stale predictor and may also fail.
- The caller explicitly requested a reset; silently retaining the stale predictor would
  violate caller intent. The caller is responsible for handling the error (retry, log,
  fall back to store teardown).

## `callCount` reset cadence interaction with Layer 2

After `resetState()`, `callCount` is reset to 0. The next Layer 2 proactive reload
fires `reloadInterval` encodes after the explicit reset. This is intentional and
desirable: the pool was just flushed explicitly, so no reload is needed for the next
`reloadInterval` calls. Callers who interleave explicit resets with large batches should
be aware that the proactive-reload timer resets alongside the explicit flush.

## Alternatives considered

### Self-healing: `autoResetEvery: Int?` parameter

Add an `autoResetEvery: Int?` parameter to `T5CoreMLEmbedder` that automatically calls
the pool-flush logic every N encodes, invisibly to the caller. This is exactly what
Layer 2 (`reloadInterval`) already does — adding a second counter with the same
mechanism does not give callers new capability. Callers who want to control the cadence
still can't trigger a reset at a specific batch boundary. Rejected: duplicate of
existing Layer 2 mechanism; does not solve the consumer control requirement.

### Lifecycle hint: `lifecycleHint(_ hint: LifecycleHint)` API

A lifecycle-hint API (`.willStartBulkBatch`, `.didFinishBulkBatch`, etc.) gives the
embedder information about the caller's intent but delegates the decision of whether to
flush to the embedder. This is indirect and opaque. For the ANE IOSurface case, the
correct action on "willStartBulkBatch" is always "flush now" — there is no embedder
judgement involved. An explicit `resetState()` is clearer and testable. Rejected:
indirection without benefit.

### Store-level passthrough: `SwitchcraftStore.resetEmbedder()`

A passthrough method on `SwitchcraftStore` would let callers reset the embedder without
holding a separate reference. Rejected: the store holds `private let embedder: any
Embedder`. Callers already hold the `Embedder` reference they supplied at `init`.
Adding a store passthrough increases the store's public surface area for something
callers can do directly. A doc-comment on `SwitchcraftStore.add()` pointing to
`embedder.resetState()` is sufficient.

## Consequences

- `Embedder` gains a new `resetState() async throws` requirement, backward-compatible
  via the protocol default no-op.
- `T5CoreMLEmbedder` users gain explicit control over when the ANE IOSurface pool is
  flushed, without touching `reloadInterval` internals or destroying the `SwitchcraftStore`.
- `MockEmbedder`, `T5MetalEmbedder`, and other future conformers inherit the no-op at
  zero cost.
- Layer 2 (proactive reload every `reloadInterval` encodes) and Layer 3 (reactive reload
  on IOSurface failure) remain active as complementary defences.
- The `callCount` proactive-reload counter resets to 0 after each explicit `resetState()`,
  restarting the Layer 2 cadence from the flush point.

## Cross-references

- [ADR 021](021-ane-iosurface-pool-exhaustion-mitigation.md) — Layer 2/3 IOSurface
  pool exhaustion mitigations that `resetState()` complements.
- [ADR 024](024-rehydration-conflict-autorecovery.md) — Explicit consumer-controlled
  recovery philosophy (`.autoRecover` vs. explicit API); same design principle applied
  here (explicit beats implicit for self-healing).
- [ADR 028](028-query-token-surface-form-filter.md) — Origin of the `inFlight`/`waiters`
  re-entrancy guard that `resetState()` reuses. Explains why both `encode()` and
  `encodeQuery()` are thin wrappers rather than calling each other, and the correct
  `resume()` ordering that prevents re-entrancy windows.
