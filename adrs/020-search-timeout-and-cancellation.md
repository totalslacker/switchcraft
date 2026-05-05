# ADR 020: Search timeout and cancellation design

**Status**: Accepted  
**Issue**: #83  
**Date**: 2026-05-05

## Context

`SwitchcraftStore.search()` had no upper bound on execution time. A single slow SQLite query could hold the storage actor indefinitely, blocking all concurrent operations and providing no mechanism for cancellation. The need for a configurable wall-time deadline and cooperative task cancellation was identified in issue #83.

The implementation spans three modules with strict dependency ordering:
- `SwitchcraftCore` — cannot import `Switchcraft` (would create a cycle)
- `Switchcraft` — imports `SwitchcraftCore`, defines public error types
- `SwitchcraftSQLite` — imports both `SwitchcraftCore` and `Switchcraft`

## Decision

### 1. Two-level error translation path

Pre-phase deadline checks inside `SearchEngine.searchHybrid` (which lives in `SwitchcraftCore`) cannot throw `SwitchcraftStoreError` (defined in `Switchcraft`) due to the module dependency constraint. Instead:

- **Pre-SQLite deadline expiry** (checked at the top of `searchHybrid`) throws `SearchEngine.Error.deadlineExceeded(elapsed:)`.
- `SwitchcraftStore.search()` catches this case and retranslates to `SwitchcraftStoreError.searchTimedOut(elapsed:)`.

- **`SQLITE_INTERRUPT` (code 9)** from the progress handler is caught by `SQLiteStorage` (which lives in `SwitchcraftSQLite` and imports `Switchcraft`) and translated directly to `SwitchcraftStoreError.searchTimedOut(elapsed:)`. The stored `currentDeadlineContext` provides the `searchStart` needed for total elapsed computation.

- **Post-flush / post-encode pre-phase checks** in `SwitchcraftStore.search()` throw `SwitchcraftStoreError.searchTimedOut(elapsed:)` directly.

All three paths report total elapsed time since `SwitchcraftStore.search()` was entered, satisfying the invariant `elapsed ≤ deadline + one_handler_tick`.

### 2. Protocol extension for deadline configuration

`SwitchcraftStorage` gains a `configureSearchDeadline(_ ctx: SearchDeadlineContext?) async` requirement with a default no-op implementation via a protocol extension. This approach:

- Keeps `InMemoryStorage` unchanged (inherits the no-op).
- Allows `SearchEngine` (in `SwitchcraftCore`) to call `configureSearchDeadline` through the `any SwitchcraftStorage` existential without knowing about `SQLiteStorage`.
- Future backends that support interruption can override; those that do not get the safe no-op default.
- Adding this method to the protocol is additive (the default no-op satisfies any existing conformers that do not override it).

### 3. `ProgressHandlerState` retained-pointer lifetime management

The C progress callback cannot capture Swift closures. A heap-allocated `ProgressHandlerState` object is passed to `sqlite3_progress_handler` as a retained opaque pointer:

```swift
// In SQLiteConnection.init:
progressStatePtr = Unmanaged.passRetained(progressState).toOpaque()
sqlite3_progress_handler(handle, 10_000, sqliteProgressCallback, progressStatePtr)

// In SQLiteConnection.deinit:
Unmanaged<ProgressHandlerState>.fromOpaque(progressStatePtr).release()
```

The `SQLiteStorage` actor serialises all access to `ProgressHandlerState`, so there is no concurrent mutation even though the C callback fires on the same thread as `sqlite3_step` (which is always the actor's executor thread).

The `isActive: Bool` flag on `ProgressHandlerState` is set to `false` by default and by `configureSearchDeadline(nil)`. This ensures that write operations (`add`, `flush`, `index`) issued after a timed-out search are never interrupted by a stale zero budget. `SwitchcraftStore.search()` calls `await storage.configureSearchDeadline(nil)` in both success and error return paths to disarm the handler.

### 4. `SearchDeadlineContext` placement in `SwitchcraftCore`

The `SearchDeadlineContext` struct (`searchStart: ContinuousClock.Instant`, `deadline: Duration`) is defined in `SwitchcraftCore` so it is visible to all three modules without introducing new dependencies. It is passed explicitly through `searchHybrid` as a defaulted parameter, making the data flow testable and avoiding Task-local storage.

### 5. Progress handler granularity caveat

`sqlite3_progress_handler` with `N = 10_000` fires every 10,000 SQLite VM instructions. For pure SQL queries this reliably fires every 5–20 ms at typical SQLite throughputs. For FTS5 virtual-table queries, the heavy processing (postings list traversal, BM25 scoring) runs in C code outside the VM instruction counter; each `sqlite3_step` call may execute fewer VM instructions than `N` even for large result sets. In practice this means:

- The progress handler is an effective last-resort timeout for long-running pure SQL queries.
- FTS5 queries on small datasets may complete before the handler fires. The pre-phase deadline checks (applied before SQLite executes) and cooperative `Task.checkCancellation()` calls remain effective regardless of query cost.

`N = 10_000` is treated as an internal constant pending benchmarks on production-scale datasets.

## Consequences

- `SearchConfig.searchDeadline: Duration` (default 5 s) is a new public field. Existing `SearchConfig()` constructions gain the default value without changes.
- `SwitchcraftStore.search(query:topK:filter:deadline:)` adds a defaulted `deadline: Duration? = nil` parameter. Existing call sites are source-compatible.
- `SwitchcraftStoreError.searchTimedOut(elapsed:)` is a new public error case. Exhaustive `switch` statements on `SwitchcraftStoreError` in downstream code will require a new case.
- Every `SwitchcraftStorage` conformer that does not override `configureSearchDeadline` gets a no-op for free. Conformers that want interruption must implement the override.
