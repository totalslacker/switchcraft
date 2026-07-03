# ADR 029 — Orphan Chunk Detection and Recovery

**Status:** Accepted
**Date:** 2026-06-13
**Refs:** [ADR 004](004-lsm-cascade-policy.md), [ADR 005](005-bucket-indices-encoding.md), [ADR 009](009-public-api-shape.md)

---

## Context

`SwitchcraftStore.add()` deduplicates chunks by `SHA-256(embeddingText)`. When a
chunk row already exists for a given hash, the store reuses that chunk's `id` and
skips `indexer.add()`. This is correct behaviour for fully-indexed chunks, but
creates a silent permanent-invisibility failure mode — an **orphan** — when
`storage.upsertChunk()` succeeded but `indexer.add()` did not.

### (a) The orphan failure mode

A chunk becomes orphaned when:

1. `storage.upsertChunk()` persists the chunk row (with its `counts` field).
2. `indexer.add()` is not called, or is called but interrupted before
   `performFlush()` writes any bucket blobs — for example by a process kill,
   OOM, or disk-full mid-write.

After a restart, the Indexer rehydrates its ledger from existing bucket data
(ADR 004 §g). A chunk that never reached bucket blobs is absent from the
rehydrated ledger and invisible to all subsequent vector-search queries.

On the next `store.add()` call with the same content, the pre-fix dedup branch
finds the existing chunk row and returns early — the orphan persists forever.
The failure is completely silent: no error, no log entry, no FTS gap
(the FTS index is populated via `upsertDocument`, not the indexer path).

Reproduced in production: a chunk with 4,056 tokens and a valid model identifier
existed in the `chunk` table for months with zero bucket assignments, causing
the document to be invisible in vector search even for exact-content queries.

---

## Decision

### (b) `add()` recovery contract (R1–R3)

After finding an existing chunk row in the dedup branch, `store.add()` applies
a three-way check before deciding whether to call `indexer.add()`:

| Case | Condition | Action |
|------|-----------|--------|
| R2 — Pending | `pendingChunkIDs.contains(chunkID)` | Skip `indexer.add()` — already buffered in this store's lifetime, not yet flushed. Double-buffering the same chunkID violates the Indexer's single-reference invariant. |
| R3 — Indexed | `storage.chunkBucketRefCount(chunkID) >= existing.counts.reduce(0, +)` | Skip `indexer.add()` — chunk's committed bucket reference count equals its expected token count; fully indexed. |
| R1 — Orphan or Partial | `actualRefs < expectedTokenCount` and not pending | Call `indexer.removeFromLedger(chunkID)` to clear any stale rehydrated rows, then call `indexer.add()` with the freshly embedded vectors to re-buffer. Add to `pendingChunkIDs`. Covers both `actualRefs == 0` (full orphan) and `0 < actualRefs < expectedTokenCount` (partial-indexing anomaly). |

Recovery is **always-on**: no opt-in flag is needed. The only meaningful
exception — the pending case — is already handled by R2.

New chunks follow the same `pendingChunkIDs` tracking: after `indexer.add()` is
called for a newly inserted chunk, the chunk's id is added to `pendingChunkIDs`.

`pendingChunkIDs` is cleared by `flushAndClearPending()`, a private helper that
wraps `indexer.flush()`. All code paths that were calling `indexer.flush()`
directly — `index()`, `search()`, `score()`, `shutdown()` — now call
`flushAndClearPending()` instead.

### (c) `findOrphanedChunks()` API contract

```swift
public func findOrphanedChunks() async throws -> [OrphanedChunkInfo]

public struct OrphanedChunkInfo: Sendable, Equatable {
    public let chunkID: Int64
    public let hash: String
    public let model: String
    public let expectedTokenCount: Int
    public let bucketReferenceCount: Int
    public let owningDocuments: [String]
}
```

`findOrphanedChunks()` scans committed storage state only (no flush required).
It enumerates all chunks via `storage.allChunks()`, decodes all bucket blobs
once to build a `[Int64: Int]` chunkID→pair-count map, and emits an
`OrphanedChunkInfo` for any chunk where `bucketReferenceCount < expectedTokenCount`.

The `owningDocuments` field contains the UUIDs of all document rows whose hash
maps to the orphaned chunk, allowing a caller to surface "URL X is orphaned"
without a second storage query.

**Filtering conventions:**
- `bucketReferenceCount == 0` → true orphan (never indexed).
- `0 < bucketReferenceCount < expectedTokenCount` → partial-indexing anomaly.

**Recovery:** re-add any document in `owningDocuments` through `store.add()`.
`add()` detects incomplete indexing (R1: `actualRefs < expectedRefs`) for both
full orphans (`actualRefs == 0`) and partial-indexing anomalies (`0 < actualRefs
< expectedRefs`), clears any stale rehydrated ledger rows via
`indexer.removeFromLedger()`, and re-buffers the embeddings; the next
`store.search()` or `store.index()` call flushes them into the bucket index.

### (d) Always-on bucket-presence check — rationale

Alternatives considered:

1. **Opt-in flag** (`repairOrphans: Bool = false`): breaks the contract that
   `add()` is idempotent. A caller who forgets the flag silently re-creates
   orphans. Rejected.

2. **Transactional atomicity** (wrap `upsertChunk + indexer.add` in a single
   transaction): the Indexer's in-memory ledger is not a storage transaction
   participant. Making it one would require significant Indexer API changes and
   cross-actor transaction semantics. Deferred.

3. **Always-on bucket-completeness check** (chosen): one storage call per dedup
   hit on a chain of dedup hits. The check compares `chunkBucketRefCount()`
   against `existing.counts.reduce(0, +)` to distinguish fully-indexed chunks
   from partial-indexing anomalies and full orphans. For the SQLite backend the
   fast-path range check (`generation.min_chunk_id / max_chunk_id`) eliminates
   most BLOB decoding; the slow path decodes all blobs in matching generations
   to accumulate the count. For the in-memory backend it is
   O(total_indexed_tokens) but InMemoryStorage is not a production path. The
   check is conservative: the default implementation in `SwitchcraftStorage`
   returns `Int.max` (assume fully indexed), so external conformers opt out of
   orphan recovery rather than being forced to re-index on every `add()`.

### (e) R2 guard — Store-side `pendingChunkIDs`

Within a single store lifetime (between process restarts), a chunk added to the
indexer ledger but not yet flushed has no bucket assignments in committed
storage. A naive orphan check would misidentify it as an orphan and call
`indexer.add()` again — violating Indexer invariant 4 ("every token referenced
by exactly one bucket in the most recent generation").

Guard mechanism: `private var pendingChunkIDs: Set<Int64>` on `SwitchcraftStore`
(actor-isolated). Populated by every call to `indexer.add()` inside `add()`.
Cleared by `flushAndClearPending()` after a successful `indexer.flush()`.

Alternative considered: expose `Indexer.isChunkPending(_:)` (Option A). Rejected
because it requires adding to the Indexer's public API and carefully integrating
with the flush-waiter machinery; the responsibility split is natural since
`SwitchcraftStore` is already the sole caller of `indexer.add()`.

---

## New Storage Protocol Methods

```swift
// Returns all chunk records. Default: [] (opt-out for external conformers).
func allChunks() async throws -> [ChunkRecord]

// Returns the total (chunkID, tokenOffset) pair count for the chunk across all
// committed bucket blobs. Default: Int.max (conservative — assume fully indexed;
// opt-out for external conformers).
func chunkBucketRefCount(_ chunkID: Int64) async throws -> Int
```

Adding a required method to `SwitchcraftStorage` is source-breaking for
external conformers. Both methods are supplied with safe defaults in a protocol
extension: `allChunks()` returns `[]` (empty orphan scan) and
`chunkBucketRefCount()` returns `Int.max` (skip `indexer.add()` — orphan
recovery is opt-out, not opt-in). This matches the precedent from
`configureSearchDeadline` and `walCheckpoint`.

`IndicesCodec.decode()` errors (corrupt BLOB) are propagated, not swallowed.
A corrupt bucket blob is a real storage-integrity problem; silently treating
it as "no assignment" would mask corruption and cause perpetual re-indexing.

---

## Consequences

- Orphaned chunks created by past failures (process kills, OOM, disk errors)
  are automatically recovered the next time any owning document is re-added.
- `findOrphanedChunks()` gives downstream consumers a stable API to enumerate
  existing orphans without reverse-engineering the `BucketRecord.indices` blob
  format.
- The `chunkBucketRefCount()` call adds one storage round-trip per dedup hit in
  `add()`. For the SQLite backend it cannot short-circuit on first hit; it must
  decode all blobs in matching generations to accumulate the count — O(k) blobs
  where k ≈ 16√n centroids, compared to the previous O(1..k) short-circuit.
  For InMemoryStorage it is O(total_indexed_tokens). A benchmark (R14) confirms
  the overhead is below 100 µs/call on the InMemoryStorage path; the change is
  on the dedup path only (one call per `add()` for existing chunks), not on the
  search path.
- External `SwitchcraftStorage` conformers continue to compile unmodified; they
  receive no orphan-recovery benefit until they implement the new methods.
- `findOrphanedChunks()` is O(total_indexed_tokens) for any backend. It is a
  diagnostic API, not a hot path; a streaming alternative can be added in a
  future issue if corpus size warrants it.

---

## Amendment — issue #132: R1 path is now `add()`-safe end-to-end

**Date:** 2026-06-15

After PR #131 landed the `chunkBucketRefCount`-based R1 routing, calling
`store.add()` for a partial orphan (0 < actualRefs < expectedTokenCount) after a
process restart threw `Indexer.Error.ledgerOutOfSync`. This amendment documents
the root cause and the fix.

### Root cause

After a process restart, `Indexer.init` rehydrates the in-memory ledger from
existing bucket data. A partial orphan's generation has P bucket pairs
(P < M = expectedTokenCount), so the ledger starts with P rows for that chunkID.

When R1 fires:
1. `indexer.removeFromLedger(chunkID)` removes the P rehydrated rows.
2. `indexer.add(chunkID, M_embeddings)` adds M fresh rows; `pendingCount += M`.
3. `performFlush()` cascade walk initialises `total = pendingCount = M`, then adds
   `levelSums[0] = gen.numEmbeddings = P` → `total = M + P`.
4. Ledger sweep yields `m = M` (only the M fresh rows remain).
5. `m < total` with no surprise gens → `ledgerOutOfSync(ledgerRows: M, expected: M+P)`.

The drift magnitude equals P — the rehydrated rows that `removeFromLedger` removed
without informing the cascade walk.

### Fix

A new `private var removedFromLedgerCount: Int = 0` property accumulates the count
of rows removed by `removeFromLedger()` since the last successful flush.
`performFlush()` captures this value before the cascade walk and subtracts it from
`pendingCount`:

```swift
var total = pending - capturedRemovedCount
```

This gives `total = M - P + levelSums[0] = M - P + P = M = m`. ✓

`removedFromLedgerCount` is reset to 0 in step 12 (success path only) and in
`clearIndex()`. On error it persists so that a retry flush uses the same correction.

### Interaction with ADR 024 (step 3.5)

[ADR 024](024-rehydration-conflict-autorecovery.md) step 3.5 (`rehydrateAutoRecover`)
corrects stale `numEmbeddings` in storage to match the actual bucket pair count P
before the ledger is populated. This ensures `gen.numEmbeddings = P` exactly, so
`removedFromLedgerCount` equals the overcounting precisely. The fix is exact in
`.autoRecover` mode (the production default). In `.throwError` mode, step 3.5 does
not run; if a gen has stale `numEmbeddings ≠ P` (crash between gen insert and
bucket inserts), residual drift may remain — a known, pre-existing limitation.

### Relation to ADR 030

[ADR 030](030-mid-operation-compaction-ledger-divergence.md) covers `m > total`
(surprise gens at compaction boundaries). This amendment covers the orthogonal
`m < total` case caused by R1 ledger removals. The two self-recovery mechanisms
are additive and do not interfere.

### Contract guarantee after this fix

Calling `store.add(id:body:)` for all documents enumerated by
`findOrphanedChunks()` — whether full orphans (`actualRefs == 0`) or partial
orphans (`0 < actualRefs < expectedTokenCount`) — no longer throws
`Indexer.Error.ledgerOutOfSync`, and does not increase the partial-chunk count.
The `findOrphanedChunks()` → `store.add()` recovery pattern is now safe to run in
a batch loop without per-document error handling for ledger drift.

---

## Amendment — issue #134: Vacuum / disk reclaim

**Date:** 2026-07-03

`findOrphanedChunks()` and its `add()`-based recovery path (§b–c above) handle
chunks that have an owning document but are incompletely indexed. They are
silent on the orthogonal case this amendment adds a fix for: chunks whose
`hash` matches **no** `document.hash` at all — "abandoned" chunks, produced by
natural re-indexing (content changes, old hash orphaned), by `store.remove(id:)`
(which drops the document row but leaves the chunk and its bucket entries in
place), or by bulk delete-then-readd workflows. These chunks can never appear
in search results (no document points to them) but their tokens' Q4-encoded
residuals remain in `bucket.residuals` BLOBs indefinitely, tying up disk. A
production corpus reached ~8,200 abandoned chunks out of ~15,600 total chunk
rows before this fix.

### Contract

```swift
public func vacuum(maxBatch: Int? = nil) async throws -> VacuumResult

public struct VacuumResult: Sendable, Equatable {
    public let chunksRemoved: Int
    public let bucketPairsRemoved: Int
    public let approximateDiskReclaimed: Int64
    public let generationsAffected: [Int64]
    public let remainingCandidates: Int
    public let checkpoint: CheckpointResult
}
```

`vacuum()` flushes pending writes first (same contract as `walCheckpoint()`),
computes `abandoned = allChunks().filter { !documentHashes().contains($0.hash) }`,
processes at most `maxBatch` of them (all of them when `maxBatch` is `nil`),
and returns. It is the **sole** public entry point for this operation — a
narrower `removeAbandonedChunks(_ infos: [OrphanedChunkInfo])` form was
considered and explicitly deferred (see "API shape" below).

### Relationship to `findOrphanedChunks()`

The two APIs share a detection-and-decode shape but apply opposite
dispositions to disjoint sets of chunks:

| | `findOrphanedChunks()` | `vacuum()` |
|---|---|---|
| Predicate | `bucketReferenceCount < expectedTokenCount` | `chunk.hash` not in any `document.hash` |
| Targets | Incompletely-indexed chunks **with** an owning document | Chunks **without** any owning document |
| Disposition | Recovery candidate — re-add via `store.add()` | Removal candidate — deleted outright |

A chunk with an owning document is never removed by `vacuum()`, even if it is
only partially indexed (requirement 13) — that is `findOrphanedChunks()`'s
territory. Conversely, a fully-abandoned chunk is not a `findOrphanedChunks()`
match unless it also happens to be under-indexed; `vacuum()` is the only path
that removes it. Both APIs continue to operate independently and unchanged by
each other's presence — `findOrphanedChunks()` needed no modification for this
amendment.

### Batch / looping semantics

`vacuum(maxBatch: N)` processes at most `N` abandoned chunks in this call and
returns — it does not internally loop through the full backlog.
`VacuumResult.remainingCandidates` reports how many abandoned chunks this call
left unprocessed (always `0` when `maxBatch` is `nil`). A consumer draining a
large backlog calls `vacuum()` repeatedly until `remainingCandidates == 0`.
Batches are taken by ascending chunk id for determinism, so repeated calls
(including ones racing new abandonment) are idempotent: a chunk already deleted
by a prior call is simply absent from the next call's candidate scan, and the
end state of N bounded calls converges to the same result as one unbounded
call (verified by `VacuumTests.vacuumBatchingAndLoopParity`).

Rejected alternative: internal looping through the whole backlog in one call.
This would make `vacuum()` an unbounded-duration operation the caller cannot
interrupt or budget for — a bad fit for interactive workloads that need to
yield between batches. Consumer-driven looping matches the batched
orphan-recovery pattern this codebase already uses elsewhere.

### Cascade-delete behavior

If removing a chunk's `(chunkID, tokenOffset)` pairs leaves a bucket's
`indices` blob with zero surviving pairs, that bucket row is deleted outright.
If that leaves a generation with zero surviving buckets, the generation row is
deleted too (cascade). Surviving generations (at least one bucket left) have
`numEmbeddings` corrected to the post-removal pair count via a targeted
`UPDATE`, not a full `replaceGeneration()` — this preserves the generation's
id, matching the reasoning `updateGenerationEmbeddingCount` already documents
for the ADR 024 rehydration-conflict path. `minChunkID`/`maxChunkID` are
deliberately **not** narrowed on survivors, mirroring `Indexer.removeFromLedger()`'s
existing rationale: a wider-than-necessary range is harmless, and narrowing it
is unneeded complexity. `VacuumResult.generationsAffected` includes both
partially-rewritten and fully-deleted generation ids.

Half-cleanup (deleting empty buckets but leaving empty generations, or vice
versa) was rejected as inconsistent: an empty container has zero query
utility and only costs the search engine a lookup it can never satisfy.

### New storage primitives

Three new `SwitchcraftStorage` methods, all shipped with safe-default
extensions per the ADR 029/033 precedent:

```swift
// Bulk set of document.hash values, for one-shot detection instead of one
// storage round trip per chunk. Default: derived from documents(matching: .all).
func documentHashes() async throws -> Set<String>

// Atomically execute a pre-computed VacuumPlan (bucket updates/deletes,
// generation deletes/count-updates, guarded chunk deletes). Returns the
// number of chunk rows actually deleted. Default: no-op, returns 0.
func applyVacuumPlan(_ plan: VacuumPlan) async throws -> Int

// PRAGMA freelist_count × PRAGMA page_size for SQLite backends. Default: 0.
func freeListByteCount() async throws -> Int64
```

`VacuumPlan` is a plain data type (chunk ids to delete, re-encoded bucket
records to update, bucket/generation ids to delete outright, generation
`numEmbeddings` corrections) built by a new pure `VacuumPlanBuilder` in the
engine layer, mirroring `Indexer.rehydrateAutoRecover()`'s existing
filter-and-re-encode shape for surviving bucket pairs. `applyVacuumPlan`
executes the plan without any codec knowledge, keeping bucket-blob logic out
of the storage layer per this package's design tenets. `VacuumPlanBuilder`
range-prunes generations whose `[minChunkID, maxChunkID]` doesn't overlap the
batch's abandoned-id range before ever fetching their buckets, turning the
common case (abandoned ids clustered from a bulk delete) into O(chunks
actually affected) rather than O(total indexed tokens) — the same fast-path
principle `chunkBucketRefCount`'s SQLite range check already uses.

The guarded chunk-row delete (`DELETE ... WHERE NOT EXISTS (SELECT 1 FROM
document WHERE document.hash = chunk.hash)` in SQLite; an equivalent
in-memory hash-membership check in `InMemoryStorage`) protects against a
document racing back onto an abandoned chunk's hash between vacuum's
detection scan and its write phase — the chunk row survives as a (now
self-healing, via the existing `findOrphanedChunks()` → `add()` path)
true-orphan rather than leaving a document pointing at a deleted row.
`applyVacuumPlan` returns the actual deleted count so `VacuumResult.chunksRemoved`
stays truthful even when the guard fires.

### Ledger consistency — targeted invalidation, not full re-init

Vacuum owns ledger consistency itself rather than depending on this ADR's
own mid-add self-recovery (the R1 path, §b–c above) or ADR 030's
mid-compaction self-recovery, both of which target different drift classes. A
new `Indexer` method:

```swift
public func removeAbandonedFromLedger(_ chunkIDs: Set<Int64>) async throws
```

removes the ledger rows for exactly the chunk ids vacuum just deleted from
storage, gated by the same `flushInProgress` waiter loop `add()` and
`removeFromLedger()` already use. This was chosen over the
originally-researched approach of reassigning `self.indexer` via a fresh
`Indexer.init` to force full rehydration, because that approach reintroduces
an ADR-030-class race: a concurrent `add()` landing between vacuum's storage
commit and the indexer reassignment would buffer into the old, about-to-be-discarded
`Indexer` actor and silently lose data, and a rehydration failure after the
storage transaction had already committed would leave the store in a new,
previously-impossible partial state. `removeAbandonedFromLedger()` never swaps
any actor reference, eliminating both failure classes by construction, and
costs O(batch size) rather than O(all generations).

`removeAbandonedFromLedger()` deliberately does **not** touch
`removedFromLedgerCount` (the counter `removeFromLedger()` increments for the
R1 clear-and-refill path, per the issue #132 amendment above).
`removeFromLedger()`'s counter compensates the cascade walk for rows that were
cleared and are about to be refilled by an immediately-following `add()` —
storage still expects the full pre-clear count until the refill lands.
Vacuum's case is a **true delete**: it already decremented the corresponding
`generation.numEmbeddings` in storage for these exact chunk ids before calling
this method, so ledger and storage stay in lockstep with no compensation
needed. Incrementing `removedFromLedgerCount` here would double-subtract on
the next flush and desync it. This distinction is spelled out here so a
future reader doesn't try to "fix" this into using the same counter.

### Idempotency

Calling `vacuum()` on a clean store, or after a prior call's
`remainingCandidates` reached `0`, returns an all-zero `VacuumResult`
(`chunksRemoved`, `bucketPairsRemoved`, `approximateDiskReclaimed`,
`generationsAffected` all zero/empty) with no write I/O and no checkpoint —
the empty-batch case short-circuits before `applyVacuumPlan`/`walCheckpoint`
are ever called.

### `approximateDiskReclaimed` measurement contract

Measured as `(freelist_count_after − freelist_count_before) × page_size`,
sampled via `PRAGMA freelist_count`/`PRAGMA page_size` immediately before and
after the vacuum call's write phase, clamped to a minimum of `0`. This
reflects space returned to SQLite's internal free-list for reuse by future
inserts — it does **not** reflect a reduction in on-disk file size, since
`DELETE` + `wal_checkpoint(TRUNCATE)` does not return pages to the OS; the
file only shrinks after a full `PRAGMA vacuum`, which is out of scope for this
operation (it can require up to 2× the database size in temporary disk and is
incompatible with `maxBatch` batching). A consumer wanting an actual file-size
reduction should run `PRAGMA vacuum` themselves after looping `vacuum()` to
`remainingCandidates == 0`. Always `0` for non-SQLite backends. Verified by
`VacuumTests.vacuumFreelistAndFileSizeContract`, which also asserts
`PRAGMA page_count × page_size` (file size) is unchanged immediately after
`vacuum()`.

### API shape — `vacuum()` only

A narrower `removeAbandonedChunks(_ infos: [OrphanedChunkInfo])` form was
considered and rejected for this issue: `vacuum()` is functionally equivalent
to `removeAbandonedChunks(findOrphanedChunks().filter { $0.owningDocuments.isEmpty })`,
and no consumer had asked for chunk-level filtering (by age, by hash, etc.).
Shipping only the higher-level wrapper keeps the surface minimal; a
`keepPredicate` closure on `vacuum()` or a standalone `removeAbandonedChunks(_:)`
remain backward-compatible additions for a future issue if that need
materializes.

### Concurrency contract

`vacuum()` requires no concurrent `add()`/`index()` calls against the same
store from the same process — same class of caveat `clear()` already carries
(non-atomic across `indexer.clearIndex()` + `storage.clear()`), extended here
to same-process concurrent `Task`s rather than just other processes. The
guarded chunk delete bounds the worst case of a violation to a self-healing
true-orphan rather than data loss or corruption; building full cross-actor
synchronization for a maintenance operation the issue itself frames as
batch/consumer-driven was judged out of proportion to the risk.

### Out of scope

Auto-vacuum during `add()` (unpredictable latency), re-clustering/compaction
of remaining sparse buckets, cross-process coordination, and full SQLite
`PRAGMA vacuum` (file-size compaction) are all explicitly out of scope for
this operation — see the issue's Scope section for the full list.
