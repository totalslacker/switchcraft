# ADR 019 — SQLiteStorage writer + reader actor split

## Status

Accepted

## Context

`SQLiteStorage` is a Swift actor that serialises all access to a single SQLite
connection. Every public method issues a synchronous `sqlite3_step` call on the
actor's serial executor. While `PRAGMA journal_mode = WAL` was already set,
WAL's ability to run concurrent reads alongside one writer was unused because
all operations shared a single connection on a single actor queue.

This caused a user-visible stall in SafariUnfucker: a search query (`MATCH`)
held the actor executor for its entire duration; the indexer's next `add` call
queued behind it and the counter froze until the search completed.

## Decision

Split `SQLiteStorage` into a thin façade over two internal actors:

- **`SQLiteWriterActor`** — owns a `READWRITE | FULLMUTEX` connection; handles
  all mutating operations (`upsertChunk`, `upsertDocument`, `insertGeneration`,
  `insertBucket`, `deleteGeneration`, `deleteDocument`, `clear`) plus lifecycle
  (`open` with WAL pragma + schema DDL, `close`).

- **`SQLiteReaderActor`** — owns a `READONLY | FULLMUTEX` connection; handles
  all query operations (`chunk(id:)`, `chunk(hash:)`, `chunkCount`,
  `document(uuid:)`, `documents(matching:)`, `documents(forChunkHash:)`,
  `documentCount`, `generations`, `buckets(forGeneration:)`, `searchFullText`).

The public `SwitchcraftStorage` protocol and all method signatures are
unchanged. `SQLiteStorage` remains the public type; the sub-actors are internal.

### Mode enum

`SQLiteStorage` holds a private `Mode` enum:

```swift
private enum Mode {
    case closed
    case inMemory(SQLiteConnection)
    case fileBacked(SQLiteWriterActor, SQLiteReaderActor)
}
```

### In-memory fallback

WAL concurrency is a file-backed-only property. For paths recognised as
in-memory (`":memory:"`, any path containing `":memory:"`, or any URI with
`"mode=memory"`), the façade retains a single `SQLiteConnection` directly
(`.inMemory` mode). This preserves the existing single-actor semantics and
avoids the fact that two separate `:memory:` connections see different,
independent databases.

### Per-connection pragmas

| Connection | Pragmas set at open |
|---|---|
| Writer | `foreign_keys = ON`, `journal_mode = WAL`, schema DDL |
| Reader | `foreign_keys = ON` only (WAL is a database-file property; reader sets nothing) |

WAL mode persists in the database file after the writer first sets it. The
reader connection does not re-issue the pragma.

### Reader opens READONLY

The reader connection is opened with `SQLITE_OPEN_READONLY | SQLITE_OPEN_FULLMUTEX`.
`READONLY` prevents accidental writes and clearly communicates intent. SQLite
READONLY connections in WAL mode still write to the `-shm` shared-memory file
(requiring directory write permission, not file permission) — this is the
standard case with any writable temp directory.

### open() partial-failure handling

Writer opens first (WAL + schema DDL). If the reader subsequently fails to
open, the writer is closed synchronously before the error is rethrown. The
façade surfaces the error with no half-open state stored.

### upsertChunk dedup stays on the writer

`upsertChunk` performs a check-then-insert to deduplicate by content hash. Both
the pre-check and the INSERT run inside the writer actor, which serialises them
— no TOCTOU risk because WAL guarantees only one concurrent writer. The
store-level pre-check (`storage.chunk(hash:)` → reader) is a best-effort
fast-path optimisation; the writer's dedup gate is authoritative.

## Consequences

**Positive**
- A slow FTS scan no longer blocks indexing writes. Reader and writer actors
  hold separate connections and run concurrently at the SQLite file level.
- The `SwitchcraftStorage` protocol and `SwitchcraftStore` public API are
  unchanged; callers see no difference.
- Forward-compatible with a V2 reader pool: turning one `SQLiteReaderActor`
  into N is a localised change inside the façade.

**Neutral**
- Slight actor-hopping overhead per call (~microseconds) as the façade awaits
  the sub-actor. Sequential-perf regression measured at ≤5% for 1,000 writes.
- WAL checkpoint growth is unchanged from before: the reader snapshot window
  is bounded by the longest in-flight read, same as today. Checkpoint
  management is deferred to V2.
- `clear()` followed by a concurrent read will briefly see stale pre-clear data
  (expected WAL snapshot semantics, documented in the spec).

**Negative**
- Slightly more code surface. Two new internal files (`SQLiteWriterActor.swift`,
  `SQLiteReaderActor.swift`) and a larger `SQLiteStorage.swift`.
