# ADR 033 — WAL Checkpoint Contract: `CheckpointResult`, Flush-First `walCheckpoint()`, and `shutdown()` Partial Logging

## Status

Accepted

## Context

### The unbounded WAL growth failure mode

`SwitchcraftStore.shutdown()` closes the SQLite connection, but prior to this fix it
did not run `PRAGMA wal_checkpoint(TRUNCATE)`. Under a sustained bulk-write workload
(bulk indexing, orphan recovery, or batch re-embedding), a consumer that repeatedly
called `store.add()` across multiple `shutdown()` / reopen cycles accumulated WAL
frames indefinitely. The WAL file grew without bound because no checkpoint ever ran.

Observed impact on a bulk-recovery workload (~6,000 sequential `add()` calls, ~14,000
chunks, 3 LSM generations, processed in batches):

| File | Size | Note |
|---|---|---|
| `<db>.sqlite` | 1.14 GB | 4.5× pre-workload |
| `<db>.sqlite-wal` | **23.3 GB** | started near zero |
| `<db>.sqlite-shm` | ≈45 MB | normal |

Throughput degraded from ~2 s/add at run start to ~30 s/add at WAL ≈10 GB, because
SQLite must scan WAL frames on every read.

A one-line `PRAGMA wal_checkpoint(TRUNCATE)` issued out-of-band restored full
throughput immediately.

### Why TRUNCATE mode

SQLite offers four checkpoint modes: PASSIVE, FULL, RESTART, and TRUNCATE.

`TRUNCATE` copies all WAL frames to the main database file and, if no reader holds a
WAL snapshot, resets the WAL file to zero bytes. It is the only mode that actually
reclaims disk space. PASSIVE and FULL leave the WAL file intact on disk; RESTART
resets the WAL write pointer but does not truncate.

`TRUNCATE` returns `SQLITE_BUSY` (non-blocking) rather than hanging when a reader
connection holds a WAL snapshot. This makes it safe to call from `shutdown()` without
risking a stall.

## Decision

### 1. `shutdown()` calls `wal_checkpoint(TRUNCATE)` before closing

After flushing the indexer (`flushAndClearPending()`), `shutdown()` calls
`storage.walCheckpoint()` directly (bypassing `self.walCheckpoint()` — see rationale
below). On partial checkpoint (`SQLITE_BUSY` from a reader), it logs via `os_log` at
`.error` level and continues to `storage.close()`. `shutdown()` remains `Void` and
does not throw on partial checkpoint: its contract is "leave the store in a safe
stopped state," which succeeded.

### 2. New public type `CheckpointResult`

```swift
public enum CheckpointResult: Sendable, Hashable {
    case complete                         // WAL truncated to zero
    case partial(framesRemaining: Int)    // reader blocked truncation; data is durable
}
```

`framesRemaining` is column 1 (`log`) from the PRAGMA result row `(busy, log,
checkpointed)`. In TRUNCATE mode with a blocked reader, `checkpointed` typically
equals `log` (all frames were copied to the main db; only the file truncation was
blocked), so `log - checkpointed` would often yield 0 — misleadingly suggesting the
WAL is empty. `log` gives the consumer the correct upper bound on WAL file size.

### 3. `walCheckpoint()` flushes the indexer first

```swift
public func walCheckpoint() async throws -> CheckpointResult {
    try ensureRunning()
    try await flushAndClearPending()
    return try await storage.walCheckpoint()
}
```

Flushing the indexer before issuing the checkpoint ensures that all `add()` calls
issued before this method returns are durable on disk. Without the flush, buffered
indexer writes would survive a `walCheckpoint()` call but be lost on a crash that
followed. The flush-first contract matches `shutdown()` and the natural consumer
expectation: "I just checkpointed — my data is safe."

**Why `shutdown()` calls `storage.walCheckpoint()` directly, not `self.walCheckpoint()`:**
`isShutDown = true` is set before the first `await` in `shutdown()`, so
`ensureRunning()` inside `self.walCheckpoint()` would throw `alreadyShutDown`.
Additionally, the indexer was already flushed by `flushAndClearPending()` earlier in
`shutdown()` — calling `self.walCheckpoint()` would flush again unnecessarily.

### 4. `SwitchcraftStorage` protocol updated; default no-op returns `.complete`

```swift
// Before
func walCheckpoint() async throws

// After
func walCheckpoint() async throws -> CheckpointResult
```

The default protocol extension returns `.complete` unconditionally, so non-SQLite
backends (in-memory, custom) opt out automatically without modification. The
`InMemoryStorage` conformance inherits the default no-op and does not need updating.

Third-party conformers that explicitly implemented `walCheckpoint() -> Void` will
receive a compile error after this change. They must either remove the override (to
inherit the new default) or update the return type to `-> CheckpointResult`.

### 5. `SQLiteWriterActor.walCheckpoint()` reads the PRAGMA result row

Previously `conn.execute()` was used, which calls `sqlite3_exec()` and discards all
result rows. It is replaced with `conn.prepare()` + `stmt.step()` + column reads:

```swift
func walCheckpoint() throws -> CheckpointResult {
    guard let conn = connection else { return .complete }
    let stmt = try conn.prepare("PRAGMA wal_checkpoint(TRUNCATE)")
    guard try stmt.step() else { return .complete }
    let busy = stmt.columnInt64(0)
    let log  = stmt.columnInt64(1)
    return busy == 0 ? .complete : .partial(framesRemaining: Int(log))
}
```

## Consequences

### For consumers

- `SwitchcraftStore.walCheckpoint()` now returns `CheckpointResult` instead of
  `Void`. Callers that discarded the return value (`try await store.walCheckpoint()`)
  continue to compile; callers that captured it gain structured feedback.
- Consumers running sustained bulk-write workloads (batch indexing, orphan recovery,
  batch re-embed) can call `walCheckpoint()` between batches when the store is not
  bounced. If `.partial(framesRemaining:)` is returned, the WAL file is still
  bounded by a single-batch's writes, not the full workload's; the consumer may retry
  after closing any open reader connections.
- `shutdown()` now always checkpoints. Workloads that repeatedly open and close the
  store without explicit `walCheckpoint()` between batches will no longer accumulate
  an unbounded WAL.

### For storage backend implementors

- The `walCheckpoint()` protocol requirement now returns `CheckpointResult`.
- Non-SQLite backends that don't override `walCheckpoint()` are unaffected (the
  default extension returns `.complete`).
- SQLite backends should read the PRAGMA result row to distinguish `.complete` from
  `.partial`; returning `.complete` unconditionally is safe but loses the feedback.

### When to call `walCheckpoint()` explicitly

Call `walCheckpoint()` between batches in a sustained-write workload where:
- The store is **not** bounced between batches (no `shutdown()` / reopen), **and**
- The WAL is growing due to accumulated writes.

When the store is bounced between batches (separate `shutdown()` per batch),
`shutdown()`'s built-in checkpoint is sufficient. No explicit `walCheckpoint()` call
is needed.

## Related

- **ADR 019** (`019-sqlite-writer-reader-split.md`) — WAL checkpoint runs on the
  writer connection, consistent with the writer/reader actor split.
- **ADR 029** (`029-orphan-chunk-detection-recovery.md`) — introduces
  `pendingChunkIDs` and `flushAndClearPending()`; `walCheckpoint()` calls
  `flushAndClearPending()` to maintain the pending invariant.
- **ADR 031** (`031-embedder-reset-state-iosurface-pool-flush.md`) — the ANE
  IOSurface pool flush for sustained inference workloads. `walCheckpoint()` is the
  WAL-compaction analogue: both are explicit consumer-controlled resource management
  APIs for the same sustained-workload failure class.
- **Issue #127** — the bug report and spec for this change.
- **Issue #120** — orphan chunk detection (same workload class).
- **Issue #125** — `Embedder.resetState()` (same workload class).
