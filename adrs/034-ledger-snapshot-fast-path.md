# ADR 034 — Ledger Snapshot Fast Path: Skipping the Full Rehydration Walk on Clean-Shutdown Startups

## Status

Accepted

## Context

### The full-rehydration-walk cost

`Indexer.init` rehydrates the in-memory ledger (`[Int64: [[Float]]]`, chunkID →
per-token embeddings) from storage on **every** startup. It decodes every bucket in
every generation and reconstructs each per-token embedding as `center + dequantized
Q4 residuals`, so that a subsequent `add()` + `flush()` against non-empty storage
succeeds without `ledgerOutOfSync`. Two rehydration variants exist (ADR 024):
`rehydrateThrowError` and `rehydrateAutoRecover`, selected by
`IndexerConfig.rehydrationConflictBehavior`.

This walk is `O(all embeddings)`. For small/medium corpora it is cheap. At production
scale it is prohibitive:

- **Observed scale**: ~7,300 active chunks, 53,344 buckets, ~11.1 M embeddings.
- **Observed cost**: 10–30+ minutes of startup (100% CPU on one worker thread),
  ~15 GB RSS during the walk, dominated by the accumulation phase that appends every
  dequantized `[Float]` into an `accum` dictionary (~8.5 B inner-loop float ops).

The walk exists to detect and self-heal `rehydrationConflict` (ADR 024) — the same
chunkID present in two active generations because a process was killed mid-compaction
between "insert new gen" and "delete old gen." That safety net is real and necessary
**after an unclean shutdown**. But on a **clean shutdown** — `shutdown()` completed,
or the process exited normally after the last successful `performFlush()` — the
in-memory ledger at that moment is already known-consistent with storage, and
re-deriving that same state via a full walk is wasted work.

## Decision

Add a persisted, single-slot **ledger snapshot** that captures the in-memory ledger at
points where it is known-consistent with storage. On the next `Indexer.init`, if a
snapshot is present and a cheap fingerprint confirms storage hasn't changed, load it
directly (`O(chunks)` reads) instead of walking every bucket. On any miss (absent,
stale, or corrupt), fall back to the existing full `rehydrateThrowError` /
`rehydrateAutoRecover` walk — unchanged, and still the crash-recovery safety net.

This is a startup-latency fix for the common (clean-shutdown) case. It does **not**
change crash behavior, and it does **not** reduce the ledger's steady-state memory
footprint (the snapshot's on-disk size is the same order of magnitude as the
reconstructed ledger; reducing that is a separate lazy-materialization follow-up).

### 1. Storage primitive

Three methods on `SwitchcraftStorage`, with **safe no-op default extensions** (matching
the `walCheckpoint` / `freeListByteCount` / `chunkBucketRefCount` opt-out precedent,
*not* the required-with-no-default `replaceGeneration` precedent):

```swift
func saveLedgerSnapshot(_ snapshot: LedgerSnapshotRecord) async throws
func loadLedgerSnapshot() async throws -> LedgerSnapshotRecord?
func clearLedgerSnapshot() async throws
```

A backend that does not implement them never gets the fast path and always falls back
to full rehydration — never incorrect, just slower. Implemented in `InMemoryStorage`
(a stored `LedgerSnapshotRecord?`) and the SQLite backend (a single-row
`ledger_snapshot` table, `id INTEGER PRIMARY KEY CHECK (id = 1)`; save is an UPSERT on
`id = 1`). The table is created unconditionally via `CREATE TABLE IF NOT EXISTS` in
`Schema.statements`, so existing on-disk databases pick it up lazily with **no**
`PRAGMA user_version` bump (only the V0→V1 title migration, which altered an existing
table, needed conditional migration logic). Both `storage.clear()` and
`Indexer.clearIndex()` also wipe the snapshot for hygiene.

### 2. Snapshot payload

`LedgerSnapshotRecord` carries a `dims` value, the five fingerprint fields (below), and
a flat `payload: Data`. `LedgerSnapshotCodec` encodes the ledger as a self-describing
LE byte stream (chunkCount, then per chunk: chunkID, tokenCount, `tokenCount × dims`
Float32 bit patterns), chunks in ascending chunkID order. Decoding reconstructs the
**exact** in-memory structure the full walk would have produced — same float values,
same per-chunk token ordering. This bit-for-bit parity is required, or NFCorpus
retrieval results would silently drift.
>
> **Amended by ADR 036 (issue #137):** the paragraph above described the
> original (v1) payload, which stored every row as a full-precision float —
> at the time, this made the snapshot *more* faithful than full-walk
> rehydration (which always went through Q4-lossy `center + residual`
> reconstruction). Once ADR 036 restructured the ledger itself to hold
> lazy bucket-refs for any already-flushed row, that claim no longer holds:
> the payload format was bumped to v2 (a per-token tag distinguishing
> materialized floats from bucket-refs), so a `.bucketRef` row in a
> snapshot resolves through the same Q4-lossy path a full walk would use,
> whenever it's eventually materialized. Both paths are now consistently
> Q4-lossy for any previously-flushed row; only not-yet-flushed
> (`.materialized`) rows are ever full-precision, identically on both
> paths. See ADR 036 §6 for the full rationale and the v1→v2 compatibility
> handling (a v1 snapshot from before that upgrade is rejected via a
> magic/version header and falls back to full rehydration, rather than
> being misparsed).

### 3. Fingerprint

`LedgerSnapshotFingerprint` is computed from metadata already in hand at
`Indexer.init` / `performFlush()`, with **no new storage query**:

| Field | Source |
|---|---|
| `chunkCount` | `storage.chunkCount()` (O(1) `COUNT(*)`) |
| `maxChunkID` | `max(gen.maxChunkID)` over `storage.generations()` |
| `totalEmbeddings` | `sum(gen.numEmbeddings)` over generations |
| `maxGenerationID` | `max(gen.id)` over generations |
| `generationCount` | `generations.count` |

"Max chunk id" is derived as `max(gen.maxChunkID)` rather than an independent query —
no cheap such query exists, and it is correct at both write points, where every
committed chunk belongs to some generation. This is a fast staleness/corruption check,
**not** a cryptographic integrity guarantee; correctness is backstopped by the
full-walk fallback on any mismatch. It correctly detects out-of-band mutation such as
`vacuum()` (issue #134), which changes exactly `chunkCount` and per-generation
`numEmbeddings`.

### 4. Write points

A snapshot is written at exactly two points, both known-consistent:

1. **After a successful `performFlush()` commit** — a best-effort call at the end of
   `flush()`'s leader success path, still holding the `flushInProgress` gate so no
   concurrent `add()`/`removeFromLedger()` can mutate the ledger mid-encode. A snapshot
   write failure here is logged and swallowed: the compaction already committed, and
   the snapshot is a pure optimization backstopped by full rehydration.
2. **At the end of a clean `SwitchcraftStore.shutdown()`** — a new public
   `Indexer.persistSnapshot()` (which waits out any in-flight flush, then becomes the
   leader itself), called **unconditionally** after `flushAndClearPending()` and
   **before** `storage.walCheckpoint()`.

The shutdown write must be unconditional and separate from the post-flush write:
`flush()` has a `pendingCount == 0` fast-path no-op, so an idle/read-only session
(searches only, no `add()`) never reaches `performFlush()`. Combined with
invalidate-on-load (below), relying solely on the post-flush write would mean a
snapshot loaded-and-cleared at startup, followed by an idle session, followed by
shutdown, leaves *no* snapshot — silently regressing the next startup to a full walk
despite nothing having changed. Ordering the shutdown write before `walCheckpoint()`
ensures it is covered by the same `wal_checkpoint(TRUNCATE)` (ADR 033).

A session that `add()`s then `shutdown()`s writes the snapshot twice (once post-flush,
once at shutdown). This is accepted for simplicity — tracking "already fresh" state
would reintroduce exactly the bookkeeping fragility that created the idle-session gap.

### 5. Write-time consistency guard

A snapshot is only valid when the in-memory ledger is consistent with committed
storage. By the LSM invariant every ledger row belongs to exactly one active
generation, so **total ledger row count must equal `sum(gen.numEmbeddings)`**.
`writeLedgerSnapshot()` checks this; on divergence it refuses to write and clears any
prior snapshot. This closes a hole where the ledger could be mutated out-of-band
relative to storage (e.g. a crash leaving a partial-orphan generation that is later
rewritten directly behind the Indexer's back, as exercised by the issue #132 tests):
without the guard, the shutdown write would bake in a *stale* ledger with a fingerprint
recomputed from the *mutated* storage — a fingerprint that would falsely match on
reopen, firing the fast path with wrong data. The guard forces the full-walk fallback
in exactly that case.

### 6. `Indexer.init` fast path and invalidate-on-load

`init` calls `loadFromSnapshotIfValid(gens:)` before the rehydration switch:

1. `loadLedgerSnapshot()` → if `nil`, fall back.
2. Compare the record's fingerprint against a freshly-computed one → on mismatch,
   clear the stale snapshot and fall back.
3. `LedgerSnapshotCodec.decode` → on a corrupt/truncated payload (throws), clear and
   fall back (the throw never escapes `init`).
4. On success: adopt `ledger` and `dims`, set `recoveredConflictCount = 0` (the
   snapshot represents already-conflict-free state — nothing to recover), and
   **clear the on-disk snapshot immediately** (invalidate-on-load).

Invalidate-on-load guarantees that if the process crashes after loading but before the
next successful flush/shutdown, the *next* startup falls back to the full walk rather
than trusting a snapshot now stale relative to that crash. A fresh snapshot is written
again at the next successful flush or clean shutdown.

## Relationship to ADR 024

ADR 024 (`rehydrationConflict` auto-recovery) is **unchanged**. `rehydrateAutoRecover`
/ `rehydrateThrowError`, `IndexerConfig.rehydrationConflictBehavior`, the winner/loser
`(level DESC, created DESC, id DESC)` semantics, the atomicity contracts, and the
`recoveredConflictCount` field all remain exactly as before. The snapshot is a **fast
path layered on top of** that safety net, never a replacement: every miss falls through
to the ADR 024 walk, which still finds and heals any conflict a crash left behind
(`recoveredConflictCount` populated exactly as today). The two paths coexist; this ADR
does not remove or weaken the safety net.

## Consequences

- **Startup latency** on clean-shutdown restarts drops from `O(all embeddings)` bucket
  decode + Q4 dequant to `O(chunks)` snapshot reads + one flat decode.
- **Disk footprint** roughly doubles at-rest data for large corpora (the payload is the
  same order of magnitude as the bucket/residual data already on disk). No opt-out is
  required for v1 (default-on, matching ADR 024's `rehydrationConflictBehavior`
  precedent); compression or an opt-out are natural follow-ups if disk cost bites (e.g.
  storage-constrained iOS deployments).
- **Flush latency** gains a snapshot write proportional to ledger size after each
  compaction. It is synchronous within the flush leader; if this becomes a measured
  problem, a background write is a follow-up.
- **Third-party storage backends**: the three new protocol methods have default no-op
  implementations, so existing conformers compile unchanged and simply never get the
  fast path (same impact category as ADR 024's `replaceGeneration` and ADR 033's
  `walCheckpoint` — flagged here rather than in user-facing docs, since the public
  `SwitchcraftStore` API is unchanged in signature).
- **Out of scope** (deferred follow-ups): ledger memory footprint (lazy
  materialization), snapshotting after `remove()`, continuous/online snapshot
  maintenance, and payload compression.
