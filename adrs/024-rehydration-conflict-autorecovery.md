# ADR 024 — `rehydrationConflict` Auto-Recovery

**Status**: Accepted (amended 2026-05-27 for issue #103)  
**Date**: 2026-05-27  
**Issues**: #101, #103

---

## Problem

`Indexer.init` throws `Indexer.Error.rehydrationConflict(chunkID:)` when it finds the
same chunkID in two different active generations during ledger rehydration. This
indicates storage corruption from an interrupted write (e.g., a process killed mid-flush
before the old generation was deleted). The throw is conservative and correct as a
safety gate — but it forces **every consumer** of Switchcraft to ship its own "wipe and
rebuild" recovery UI, even though Switchcraft already holds all the LSM context needed
to repair the inconsistency.

**Reproducing incident**: A 5,710-document index (108 MB DB, 5.5 MB WAL) was left in a
corrupt state after an MPSGraph SIGSEGV during XCTest exit. Three active generations
existed, two of them sharing the chunkID range `[1203, 1223]`. Every subsequent app
launch failed with `rehydrationConflict`. `PRAGMA integrity_check` returned `ok`
— the SQLite file was structurally intact; only Switchcraft's LSM invariants were
violated. This is a Switchcraft-level inconsistency that Switchcraft is best positioned
to repair.

---

## Decision

`Indexer.init` gains a configurable `rehydrationConflictBehavior` via `IndexerConfig`:

```swift
public enum RehydrationConflictBehavior: Sendable, Hashable {
    case autoRecover   // new default
    case throwError    // original behaviour, now an explicit opt-out
}
```

**`.autoRecover` (default)**: When a `rehydrationConflict` is detected, apply LSM
winner/loser semantics to select the authoritative generation for each conflicting
chunkID, prune the loser's data from storage atomically, log a structured warning, and
continue. The ledger is populated from the winning generation's data only. The consumer
sees a successfully initialised indexer, not an error.

**`.throwError`**: Unchanged from the original behaviour — throws
`Indexer.Error.rehydrationConflict(chunkID:)` on the first detected conflict. This is
the backwards-compatible opt-out for tests and safety-critical callers.

`recoveredConflictCount: Int` (public, actor-isolated, read-only) on `Indexer` reflects
the number of distinct chunkID conflicts auto-recovered during the last `init`. Zero when
no conflicts were found or when `rehydrationConflictBehavior == .throwError`.

---

## Winner Selection Rule

Applied per conflicting chunkID, pairwise, iterating over all genIDs in the conflict set
(handles N-way conflicts):

1. **Primary**: Higher `level` wins. (More-merged tier is presumed canonical.)
2. **Tiebreaker**: Higher `created` timestamp wins. (Last-write-wins at same level,
   matching LSM compaction semantics.)
3. **Final tiebreaker**: Higher `id` wins. (Monotonically assigned backend ID; higher id
   ≡ last inserted. Guards against same-microsecond writes at same level.)

The comparator is total-order and deterministic regardless of iteration order over the
conflict set.

---

## `replaceGeneration` Atomicity Contract

A new required protocol method was added to `SwitchcraftStorage`:

```swift
func replaceGeneration(
    losingGenerationID: Int64,
    survivingRecord: GenerationRecord,
    survivingBuckets: [BucketRecord]
) async throws -> GenerationRecord?
```

Contract:
- The entire operation (delete loser + optionally insert survivor with new id) executes
  in a single atomic transaction.
- On any error: transaction is rolled back, storage state is unchanged (still detectable
  as a conflict on the next init; never half-pruned), and the error is re-thrown to the
  caller.
- If `survivingBuckets` is empty (generation fully pruned), returns `nil` and no new
  generation row is created.
- Backends assign a fresh monotonically-increasing id to the new generation; `survivingBuckets[*].generationID` is overwritten by the backend.

**InMemoryStorage**: No explicit rollback needed — no `await` points in the method body,
so the actor serialises it atomically. An injected error (via `FailingReplaceStorage`
test stub) fires before any mutation.

**SQLiteWriterActor / SQLiteStorage inMemory path**: Uses `conn.transaction { BEGIN / COMMIT / ROLLBACK }`, the established pattern also used by `clear()`.

---

## `updateGenerationEmbeddingCount` Atomicity Contract (issue #103)

A new required protocol method added for step 3.5:

```swift
func updateGenerationEmbeddingCount(id: Int64, count: Int) async throws
```

Contract:
- Updates the `numEmbeddings` field for the generation with the given id. No-op if absent.
- **InMemoryStorage**: Direct mutation of `generations[id]?.numEmbeddings`; actor-isolated, no transaction needed.
- **SQLiteWriterActor / SQLiteStorage inMemory path**: `UPDATE generation SET num_embeddings = ? WHERE id = ?` — a single-row UPDATE, atomic at the SQLite level. No explicit transaction needed.
- Does **not** change the generation's `id`, `level`, `minChunkID`, `maxChunkID`, or `created`. The `id` is preserved, which is required because the cascade walk references generation IDs and existing test assertions verify winner ID continuity.
- A targeted UPDATE is preferred over `replaceGeneration` here: `replaceGeneration` assigns a new id (required for its transactional delete+insert semantics), which would break the id-preservation invariant.

---

## Logging

One `os.warning` line is emitted per **(chunkID, loser-gen) pair** via:

```swift
private let indexerLogger = Logger(subsystem: "com.switchcraft.core", category: "Indexer")
```

Log format (per pair):

```
Recovered rehydrationConflict: chunkID=<C> winner=gen<W.id>(level:<W.level>, created:<W.created>) loser=gen<L.id>(level:<L.level>, created:<L.created>)
```

A 2-way conflict for chunkID 5 emits 1 line. A 3-way conflict (chunkID 5 in gens 1, 2,
3) emits 2 lines (one per loser). Visible via `Console.app` or `log stream --predicate
'subsystem == "com.switchcraft.core"'`.

---

## `numEmbeddings` Staleness (issue #103)

`performFlush()` writes the generation row in **step 9** (before bucket inserts) and
writes bucket rows in **step 10**. A process killed between steps 9 and 10 leaves the
generation row with a `numEmbeddings` value that exceeds the number of pairs actually
present in the (incomplete) bucket set.

The original auto-recovery path correctly recomputes `numEmbeddings` for **loser**
generations (via `survivingRecord` passed to `replaceGeneration`). But it left **winner**
generations untouched. If the winner's `performFlush()` was the interrupted write, its
`numEmbeddings` is stale. The cascade walk in `performFlush()` accumulates `total` from
`gen.numEmbeddings`; the ledger's `m = allRows.count` is derived from decoded bucket
pairs. When `total != m`, `ledgerOutOfSync` fires — the index appears to initialise
successfully but fails on the first flush.

**Fix**: Step 3.5 (below) corrects stale `numEmbeddings` on all surviving (non-loser)
generations using already-decoded bucket data from step 1 (zero extra I/O).

---

## Rehydration Algorithm (`.autoRecover` mode)

Five-pass, separate from the `.throwError` single-pass path (the existing path is
byte-for-byte unchanged):

1. **Accumulation**: Decode all buckets; accumulate genID-tagged `(genID, tokenOffset,
   embedding)` entries per chunkID; build `chunkGenSets: [Int64: Set<Int64>]`; stash all
   decoded bucket data in-memory for the prune pass.

2. **Winner resolution**: For each chunkID with `|genIDs| > 1`, apply the pairwise
   winner-selection rule iteratively, producing `winnerGenID: [Int64: Int64]`.

3. **Storage prune**: For each loser generation with conflicted chunkIDs:
   - Re-encode surviving bucket blobs (pairs and residuals for non-losing chunkIDs) using
     `IndicesCodec.encode` and `Q4Codec.encodeResiduals` (same codec path as flush).
   - Recompute `numEmbeddings`, `minChunkID`, `maxChunkID` from surviving pairs.
   - Call `storage.replaceGeneration(...)` atomically. Throws propagate immediately.

3.5. **`numEmbeddings` correction** (issue #103): For each surviving (non-loser)
   generation whose stored `numEmbeddings` does not equal its actual decoded pair count,
   call `storage.updateGenerationEmbeddingCount(id:count:)`. Uses already-decoded
   `decodedBuckets` from step 1 — no additional I/O. A targeted UPDATE (not delete +
   re-insert) preserves the generation's id, which the cascade walk depends on and which
   existing test assertions verify. If correction fails mid-loop, the next
   `rehydrateAutoRecover` call will retry idempotently (the count mismatch guard fires
   again).

4. **Record `recoveredConflictCount`**: Set to `winnerGenID.count`.

5. **Ledger population**: For conflicted chunkIDs, filter accum entries to the winning
   generation; for uncontested chunkIDs, use all entries. Sort by `tokenOffset`.

---

## Alternatives Considered

**Path B — General-purpose `withTransaction` protocol method**  
A `func withTransaction<T>(_ body: () async throws -> T) async throws -> T` on
`SwitchcraftStorage` would let the Indexer sequence arbitrary storage calls inside an
explicit transaction. Rejected: `InMemoryStorage` would need a full dict-snapshot before
every transaction body and restore on error — disproportionate complexity for a single
targeted operation. The narrower `replaceGeneration` keeps the atomicity contract inside
each backend's native mechanism.

**Path C — Non-atomic multi-call sequence (`deleteGeneration` + `insertGeneration` + `insertBucket`)**  
No transaction boundary between calls. A crash between any two calls leaves the database
in a half-pruned state (loser gone, survivor not yet inserted, or vice versa). Rejected
as violating requirement #6 (storage state must be no worse than input on any error).

---

## Consequences

- **Consumers** no longer need to ship "wipe and rebuild" recovery UI for this class of
  LSM inconsistency caused by interrupted writes. The downstream consumer's
  stopgap (`isHealingStorage` wipe-rebuild UI) can be removed once this issue ships and
  the dependency is bumped.

- **Protocol conformers** must implement `replaceGeneration`. Any custom `SwitchcraftStorage`
  implementation that does not will fail to compile. The conformance suite
  (`runReplaceGeneration` in `StorageConformance`) verifies the atomicity and correctness
  requirements.

- **Protocol conformers** (issue #103 amendment) must also implement
  `updateGenerationEmbeddingCount(id:count:)`. No default implementation is provided
  (same policy as `replaceGeneration`). A silent no-op default would leave custom backends
  with stale `numEmbeddings` and mask the bug rather than surfacing it. The conformance
  suite (`runUpdateGenerationEmbeddingCount` in `StorageConformance`) verifies the
  semantics.

- **`rehydrationBucketCorrupt` recovery** is out of scope for this ADR — the
  `rehydrationBucketCorrupt` error still propagates as-is. A separate issue will address
  it if needed.

- **Root cause prevention** (shutdown coordination to avoid partial generation writes) is
  a separate concern not addressed here.

---

## References

- ADR 004 §(g) — Original `rehydrationConflict` throw policy (unchanged for `.throwError` callers).
- ADR 005 — Bucket indices encoding (`IndicesCodec`, `Q4Codec`); re-used in the prune re-encoding step.
- ADR 019 — SQLite writer/reader split; `replaceGeneration` and `updateGenerationEmbeddingCount` implemented in `SQLiteWriterActor`.
- Issue #101 — Specification, real-world reproducing incident, and acceptance criteria for the original auto-recovery.
- Issue #103 — Follow-up: `numEmbeddings` staleness causing `ledgerOutOfSync` after a successful `rehydrationConflict` recovery; step 3.5 and `updateGenerationEmbeddingCount`.
