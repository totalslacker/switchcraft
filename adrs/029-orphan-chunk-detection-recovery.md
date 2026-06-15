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
