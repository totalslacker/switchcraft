# ADR 038 — Shutdown-Time Snapshot Encode: Instrumentation, Bulk-Buffer Rewrite, and Progress Hook

## Status

Accepted

## Context

A downstream consumer reported a 5–20+ second (occasionally unbounded,
force-quit-required) delay on every application quit, entirely inside
`SwitchcraftStore.shutdown()`. Because `Indexer.persistSnapshot()` runs
unconditionally on every shutdown — even a read-only/idle session — by
design (ADR 034 §4: the idle-session gap it closes), every quit paid this
cost regardless of session activity.

`SwitchcraftStore.shutdown()` had no timing instrumentation, so the consumer's
Research stage could narrow the problem to "somewhere inside `shutdown()`"
but not to a specific sub-cost. Static analysis of
`LedgerSnapshotCodec.encode(_:dims:)` found two independent, stackable
suspects:

1. **No `reserveCapacity`** on the output `Data` buffer, unlike `decode()`
   which pre-sizes its output collections — repeated reallocation on a
   large ledger.
2. **Per-byte `Data.append` calls.** `appendFloat32LE`/`appendInt64LE` each
   fan out into 4–8 individual single-byte `.append` calls. At ADR 034's
   observed production scale (~11.1M embeddings), that is on the order of
   hundreds of millions of individual append calls, each paying its own
   COW/bounds-check overhead — independent of and additional to (1).

A third, distinct problem: `encode()`'s per-chunk loop had no `await`
points. Running synchronously on the `Indexer` actor for a large ledger's
entire encode duration risks starving the cooperative thread pool the rest
of the host app's concurrency depends on — a known failure class in this
codebase (the consumer's own shutdown-coordination ADR), and a plausible explanation for
the "occasional hard hang requiring force-quit" distinct from a classic
deadlock.

Finally, the consumer's library layer (a sibling issue filed alongside
this one) needs to surface real shutdown progress to a user-facing UI rather
than an opaque work-item name, which `shutdown()` had no hook for.

## Decision

### 1. Instrumentation (REQ-1)

`SwitchcraftStore.shutdown()` times `flushAndClearPending`, the
`persistSnapshot` call (total wall-clock), the WAL checkpoint, and overall
elapsed time, logging three `.info` lines via the existing `logger`
(category `"Store"`) using the issue-specified literal prefix:

```
switchcraft-adapter-shutdown: wal_checkpoint took Xs
switchcraft-adapter-shutdown: persistSnapshot took Xs (encode Xs, write Xs, bytes=N, chunks=N)
switchcraft-adapter-shutdown: total Xs
```

The encode/write sub-times and bytes-serialized breakdown are captured
inside `Indexer.writeLedgerSnapshot()` and returned up as a `Sendable`
`PersistSnapshotStats` struct (`Sources/SwitchcraftCore/Indexer/SnapshotProgress.swift`)
rather than logged from `SwitchcraftCore` in that module's own `[Compact]`-
bracket convention (issue #110 precedent). This keeps the issue-mandated,
host-specific log string owned by the `Switchcraft` module, which is the
only place that sees the full cross-module breakdown (WAL-checkpoint is a
`Store`-level call; encode/write timing is internal to `Indexer`).

### 2. `reserveCapacity` fix (REQ-2), landed separately

`LedgerSnapshotCodec.encode(_:dims:)` gained an exact-size pre-pass
(`encodedByteCount`) and `out.reserveCapacity(...)` before the per-chunk
loop, as its own commit — ahead of and independent from the REQ-3 rewrite
below — so the issue's requested before/after instrumentation comparison
has a clean data point.

### 3. Bulk-buffer rewrite (REQ-3)

Per-byte-append call-volume (not raw bytes-per-row, and not fixed by
`reserveCapacity` alone) was diagnosed as the dominant remaining cost:
`reserveCapacity` alone still leaves the same hundreds-of-millions of
individual `Data.append` method-call overheads. `encode()` was rewritten
around a single preallocated `[UInt8]` buffer, sized exactly via the same
pre-pass, with indexed writes at precomputed offsets instead of
`Data.append`. `Data(buffer)` wraps the finished buffer once at the end.

**Rejected alternative: incremental/delta snapshots.** The issue's own
leading hypothesis was CPU-bound full-ledger serialization, with
incremental (only-what-changed) snapshot writes floated as a candidate
fix. This was not pursued: the diagnosed root cause is per-call overhead,
not the fact that the whole ledger is re-serialized every time, so a
bulk-buffer rewrite eliminates the actual cost directly, in-format, with
no new persistent state. An incremental design would need a crash-safe
"since-last-snapshot" cursor that itself must respect the write-time
consistency guard (ADR 034 §5) without weakening it — real complexity the
issue's own risk section flagged as non-trivial, unnecessary once the
per-call-overhead root cause was fixed directly. If the REQ-6 benchmark
had failed to meet the `<2s` target after this rewrite, that would have
been the signal to revisit this decision (via a follow-up issue, not
scope-creep here) — it did not (§5 below).

### 4. Yield points (REQ-4)

`encode()` is now `async`, with a periodic `await Task.yield()` every 512
chunks inside the per-chunk loop — frequent enough that a large encode
cannot occupy a cooperative-pool thread for more than a few hundred
chunks' worth of work at a stretch, infrequent enough that the yield
overhead stays negligible next to the encode work itself.

This is safe specifically because every existing caller
(`Indexer.writeLedgerSnapshot()`, via `persistSnapshot()` and the
post-flush write inside `performFlush()`) holds the `flushInProgress`
leader gate for the call's entire duration, so no concurrent
`add()`/`removeFromLedger()` can mutate the ledger between yields.
`encode()` now documents this as an explicit precondition — any future
caller that does not hold that gate would silently reintroduce a
mid-encode mutation race.

### 5. Progress hook (REQ-5)

`encode()` takes an optional `onProgress: (@Sendable (SnapshotProgress) async -> Void)?`,
invoked at the same yield checkpoints with chunks/bytes encoded so far,
threaded up through `writeLedgerSnapshot(onProgress:)` →
`persistSnapshot(onProgress:)` → `SwitchcraftStore.shutdown(onSnapshotProgress:)`.

**Shape: per-call optional closure parameter, not a stored property or
`AsyncStream`.** ADR 032 rejected storing an actor-isolated closure on
`Indexer` for compaction events ("Option A") in favor of return-based
propagation ("Option B") — but Option B only delivers a single value once
an operation completes, which cannot express *incremental* progress
during one long call. Since there is exactly one call site that needs
incremental progress (`SwitchcraftStore.shutdown()`), per-call threading
sidesteps ADR 032's actor-isolated-closure-storage concern entirely
(no stored state, no lifecycle question) without paying `AsyncStream`
lifecycle-management complexity for a single consumer. This is also a
deliberate departure from issue #110's precedent (log lines only for
compaction, progress-callback API explicitly scoped out) — justified
here because the actual REQ-5 consumer (SemanticHistory) needs to drive
a user-facing progress indicator, not parse logs.

### 6. Corpus-scale benchmark (REQ-6)

`Tests/SwitchcraftTests/ShutdownLatencyBenchmark/ShutdownLatencyBenchmarkTests.swift`,
env-gated behind `SWITCHCRAFT_SHUTDOWN_LATENCY_BENCHMARK` (following the
`LazyLedgerBenchmarkTests` precedent), asserts `SwitchcraftStore.shutdown()`
completes within 2s at ADR 034's observed production scale (~7,300 chunks,
~11.1M embeddings, dims=768).

Synthesizing real Q4-quantized bucket blobs at this scale would require
~34GB of transient `Float32` residual data before quantization — infeasible
for a test fixture. Since `persistSnapshot()`'s cost is driven by the
ledger's row count and token kind, not by whether real bucket blobs exist
on disk, the fixture instead builds the *target* ledger snapshot directly:
a pure `.bucketRef` ledger (matching the common post-flush/idle-session
shape per ADR 036) is encoded once and written straight to storage's
snapshot slot, with matching (otherwise-empty) chunk/generation rows so
the snapshot's fingerprint matches on load. `Indexer.init`'s existing ADR
034 fast path then populates the ledger in `O(chunks)` — no bucket decode,
no k-means. Fixture construction cost is therefore driven by chunk count
(7,300), not embedding count (11.1M).

The absolute-time assertion is Release-gated, matching
`IndexerSnapshotSpeedupTests`/`LazyLedgerBenchmarkTests`: Debug's
unoptimized `inout`/exclusivity-checking overhead on the encode loop's
hundreds of millions of small buffer writes measured ~223s at this scale
versus ~0.4s in Release (~560×) — not representative of what users
experience, and not a regression signal Debug can meaningfully gate on.

## Consequences

- **Shutdown latency** at production scale drops from the reported 5–20+s
  (occasionally unbounded) to a measured ~0.4s (Release), comfortably under
  the 2s target with ~5× headroom.
- **`LedgerSnapshotCodec.encode` is now `async`** — a source-breaking
  change for any external caller (in-package, the only caller is
  `Indexer.writeLedgerSnapshot()`, already `async`). Callers must add
  `await`.
- **`Indexer.persistSnapshot()` and `SwitchcraftStore.shutdown()` gain new
  optional parameters** (`onProgress` / `onSnapshotProgress`), both
  defaulted, so existing call sites remain source-compatible.
- **The "clean shutdown ⇒ fast next startup" guarantee (ADR 034/036) is
  unchanged**: the wire format, the write-time consistency guard, the
  fingerprint mechanics, and the shutdown-write-before-`walCheckpoint()`
  ordering are all untouched by this ADR — only the encode implementation
  and its calling convention changed.
- **Out of scope / deferred**: incremental/delta snapshots (rejected, §3
  above, but a legitimate follow-up if a future corpus scale re-exceeds the
  target); WAL-checkpoint cadence tuning (the issue's own evidence — no
  `busy_timeout`, 313KB WAL against a 1.8GB DB — says the checkpoint step
  itself isn't the bottleneck, and REQ-1's instrumentation is what would
  surface it if that ever changes).
