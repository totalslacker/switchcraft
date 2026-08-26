# ADR 032 — CompactionEvent Callback API

**Status**: Accepted  
**Issue**: #113  
**Date**: 2026-06-14

## Context

The `Indexer` already emits a `[Compact] started` log line that includes
`inputBytes`, `trigger`, and `inputSegments`. The downstream consumer's embedder needs
those same fields to enrich a `[VMPRESS] threshold-hit` log entry without having
to parse the `com.switchcraft.core` `os_log` subsystem. This ADR records the
design decisions made to expose a `CompactionEvent` callback on `SwitchcraftStore`.

## Decisions

### (a) Wiring approach: Option B (return-based) over Options A and C

Three options were evaluated:

| Option | Description |
|--------|-------------|
| A | Store an actor-isolated closure inside `Indexer` and call it at the start of `performFlush()` |
| B | `performFlush()` returns `CompactionEvent`; `flush()` propagates as `CompactionEvent?`; `SwitchcraftStore.flushAndClearPending()` dispatches |
| C | Introduce an `IndexerDelegate` protocol implemented by `SwitchcraftStore` |

**Option B was chosen** because:
- `Indexer` never receives a reference to the `SwitchcraftStore` actor or any actor-isolated closure; concurrency is handled entirely at the store layer.
- The return-based API composes cleanly with the existing leader/waiter pattern: waiters resume with `nil`; only the leader caller at `SwitchcraftStore` receives and dispatches the event.
- Option A requires `Indexer` to store a `@Sendable` async closure and call it from within its own actor isolation, complicating concurrency reasoning.
- Option C requires a new protocol, more indirection, and the same cross-actor isolation problem as Option A.

**Caller-visible change**: `Indexer.flush()` is annotated `@discardableResult` and returns `CompactionEvent?`. All existing call sites that discard the result continue to compile without modification.

### (b) Module placement: SwitchcraftCore, not Switchcraft

`CompactionEvent` is defined in `SwitchcraftCore` alongside `Indexer`, which is the layer that computes all event fields. `Switchcraft` (the public package target) re-exports `SwitchcraftCore` via `@_exported import SwitchcraftCore`, so consumers doing `import Switchcraft` see `CompactionEvent` and `CompactionEvent.Trigger` as first-class public types of the `Switchcraft` module without a second declaration.

### (c) Event constructed after ADR 030 self-recovery block

The preliminary `inputSegments` and `inputBytes` values at the top of `performFlush()` may be underestimates. If the mid-operation divergence recovery (ADR 030, lines 619-668 of `Indexer.swift`) absorbs surprise generations, `mergedGens` grows and `total` increases. The `CompactionEvent` struct is therefore constructed **after** the divergence guard resolves, using the final post-recovery values of `mergedGens.count`, `total`, and `targetLevel`. `startedAt` is still captured before k-means to reflect when the compaction work began.

The preliminary values are retained for the `[Compact] started` log line (which fires before recovery), for diagnostic purposes. The `[Compact] finished` log uses the post-recovery final values (consistent with the event).

### (d) Concurrency contract

- The callback fires from within `SwitchcraftStore.flushAndClearPending()`, which runs on the `SwitchcraftStore` actor. The callback is `await`ed while the store actor is active.
- Callers that arrive at `Indexer.flush()` as **waiters** (i.e., concurrent `flush()` calls that queue while a leader is in progress) resume with `nil`. Only the leader's `SwitchcraftStore` turn dispatches the callback. This prevents duplicate events.
- The callback does not fire for no-op flushes (where `flush()` returns `nil` via a fast-path guard before entering `performFlush()`).
- Registering the callback from outside the actor requires calling `await store.setOnCompactionEvent { ... }`, which is a method on the actor. A `public var` property setter would be inaccessible from non-actor contexts in Swift 6 strict concurrency.
- If the callback body calls a store method that triggers a flush (e.g. `index()`), that call is queued behind the current actor turn — it is safe but can cause unexpected recursive compaction. This is documented on `setOnCompactionEvent`.

### (e) Trigger string rename: "forced" → "manual"

The internal `performFlush()` log string previously used `"forced"` for non-cascade flushes. The public `CompactionEvent.Trigger` enum uses the more descriptive name `.manual`. The internal log string was updated to `"manual"` to keep internal and external representations consistent. See ADR 004 §h for the updated trigger table.

## Consequences

- `Indexer.flush()` is now `@discardableResult -> CompactionEvent?`. This is a public API change; existing callers silently discard the result and require no modifications.
- `SwitchcraftStore.setOnCompactionEvent` is a new public surface; it must be documented and maintained going forward.
- The `CompactionEvent` struct and `Trigger` enum are part of the public API and subject to source-stability expectations.
- A `compactionFinished` / `CompactionResult` callback (end-of-compaction notification with output stats) is explicitly deferred as a potential follow-up.
- An `AsyncStream<CompactionEvent>` alternative API is explicitly deferred.
