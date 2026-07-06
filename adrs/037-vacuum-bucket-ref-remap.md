# ADR 037 — Vacuum Preserves Live Chunks' `.bucketRef`s via Compaction-Time Remap

## Status

Accepted

## Context

`SwitchcraftStore.vacuum()` (#134, ADR 029 amended) removes abandoned
chunks' `(chunkID, tokenOffset)` pairs from bucket blobs, compacting the
surviving pairs into a shorter array — shifting their positional offsets
(`VacuumPlanBuilder.buildPlan`, the `survivingPairs`/`survivingResiduals`
accumulation). Under ADR 036's lazy embedding materialization (#137), a
still-live chunk that shares a bucket with an abandoned chunk holds a
`.bucketRef(genID:bucketID:pairOffset:)` token in the `Indexer` ledger,
pointing at that row's *pre-compaction* position. Vacuum's own ledger
cleanup, `Indexer.removeAbandonedFromLedger(_:)`, only deleted the
abandoned chunks' own ledger entries — it never walked or updated the
surviving live chunks' `.bucketRef` offsets in the bucket(s) it just
rewrote.

ADR 036 §5 reasoned that `removeFromLedger`/`removeAbandonedFromLedger`
needed no logic change because both are element-type-agnostic (`.count`/
`removeValue` don't care whether tokens are `.materialized` or
`.bucketRef`). That reasoning covered only the *abandoned* chunks' own
entries — it did not account for a `.bucketRef`'s positional validity
depending on a bucket layout that something *outside* the `Indexer`
(vacuum) can rewrite. ADR 036's own stated assumption ("nothing outside
the Indexer's own flush/compaction path would rewrite bucket contents out
from under a live ref") is exactly what vacuum violated — a data-integrity
bug at the intersection of two individually-correct features (#134, #137).

Two failure modes result from one stale ref:

1. **Out-of-range shift** — the live chunk's stale `pairOffset` exceeds
   the post-compaction bucket's pair count. `Indexer.resolveBucketRefRow`
   throws `Error.bucketRefUnresolvable`. Recoverable (a clean error), but
   spurious — the chunk was never actually corrupted, only vacuum's own
   bookkeeping was incomplete. This is the mechanism behind the reported
   symptom: repeated auto-vacuum cycles against a corpus with shared
   buckets across live/abandoned chunks left a growing fraction of
   abandoned chunks permanently stuck (18 of 20, in the reported case),
   because each cycle's flush-triggered resolution of an untouched stale
   ref failed loudly enough to abort that cycle's progress.
2. **In-range shift** — the stale offset happens to still be valid, but
   now addresses a *different* pair's row. `resolveBucketRefRow` returns a
   syntactically valid `[Float]` silently. No error surfaces. Downstream
   consumers (k-means training in the next cascade, MaxSim scoring if that
   embedding is ever re-read through the ledger) proceed with corrupted
   data, and later LSM generations built from it inherit the corruption
   permanently.

Both failure modes only ever manifest when something later *resolves* the
stale ref via `Indexer.resolveBucketRefRow` — i.e. the next cascade whose
merge range includes that chunk's id. Storage-level vacuum counters
(`chunksRemoved`, `remainingCandidates`) and `SearchEngine`'s search path
(which decodes bucket blobs directly, never going through the ledger)
cannot observe either failure mode — this is why the existing
`VacuumTests.swift` suite, which asserts on those counters plus `search()`
results, didn't catch it, and why it needed a sustained multi-cycle
workload with deliberately shared buckets to surface at all.

## Decision

Implement **Option A: remap outstanding bucket-refs during vacuum's own
compaction**, computed in the exact same loop that produces the compacted
bytes, and applied to the ledger in the same call that deletes the
abandoned chunks' entries.

### Mechanism

`VacuumPlanBuilder.buildPlan`'s per-bucket loop already iterates every
pair in a touched bucket, partitioning it into `survivingPairs` (appended
in order) and abandoned ones (dropped). Each surviving pair's new index
*is* `survivingPairs.count` at the moment of append — data already fully
computed, previously discarded. The fix accumulates one
`VacuumPlanBuilder.BucketRefRemap` entry per surviving pair alongside that
append:

```swift
public struct BucketRefRemap: Sendable, Equatable {
    public let chunkID: Int64
    public let tokenOffset: UInt32
    public let genID: Int64
    public let bucketID: Int64
    public let newPairOffset: Int
}
```

`VacuumPlanBuilder.Result` gains a `bucketRefRemaps: [BucketRefRemap]`
field alongside the existing `plan`/`bucketPairsRemoved`/
`generationsAffected`. `SwitchcraftStore.vacuum()` threads it from
`planResult.bucketRefRemaps` into a renamed `Indexer` method:

```swift
public func applyVacuumLedgerUpdates(
    abandonedChunkIDs chunkIDs: Set<Int64>, remaps: [VacuumPlanBuilder.BucketRefRemap]
) async throws
```

(renamed from `removeAbandonedFromLedger(_:)`, since it now owns both
responsibilities). After deleting the abandoned chunks' ledger entries, it
applies every remap:

```swift
for remap in remaps {
    ledger[remap.chunkID]?[Int(remap.tokenOffset)] = .bucketRef(
        genID: remap.genID, bucketID: remap.bucketID, pairOffset: remap.newPairOffset
    )
}
```

This mirrors `performFlush()`'s own established pattern for first writing
a `.bucketRef` (`ledger[chunkID]?[tokenOffset] = .bucketRef(...)`,
Indexer.swift, ADR 036 §4) — vacuum's remap is structurally the same
operation (write a fresh bucket-ref for a token whose backing bucket
position just changed), just triggered by compaction-via-deletion instead
of compaction-via-merge.

### Why every surviving pair gets a remap entry, not only shifted ones

Cheaper to always emit than to track "did this offset actually change" —
already `O(bucketPairs)` since it happens inside the existing per-pair
loop, so there is no asymptotic cost to unconditional emission, and it
removes a class of "did I compute the shifted-or-not check correctly"
bugs. An unshifted pair's remap is simply a no-op overwrite with the same
value.

### Why no remap for deleted buckets/generations

`bucketIDsToDelete` and `generationIDsToDelete` are only populated when a
bucket/generation has **zero** surviving pairs — by construction, no live
ledger ref can point there, so there is nothing to remap. The remap
accumulator only ever appends inside the `survivingPairs.append` branch,
so this requires no special-casing.

### Why a blind overwrite, no defensive pre-check

`vacuum()`'s documented concurrency contract already excludes concurrent
same-process `add()`/`index()` racing vacuum's own write phase
(`SwitchcraftStore.swift`, "Concurrency contract" section) — "bucket-blob
rewrites are not re-validated after the scan, so avoid concurrent
same-process writers as a rule." A defensive check (verify
`ledger[chunkID]?[tokenOffset]` still equals the expected pre-remap
`.bucketRef` before overwriting) would only guard against a scenario the
existing contract already excludes, and `performFlush()`'s own precedent
(Indexer.swift) overwrites unconditionally under the same
`flushInProgress`-gated assumption. Adding one would be speculative error
handling for a scenario that cannot happen given the documented contract.

### No `SwitchcraftStorage` protocol change

Both `applyVacuumPlan` implementations (`InMemoryStorage`,
`SQLiteWriterActor`) are blind row-level rewrites — `UPDATE bucket SET
indices=?, residuals=? WHERE id=?` or a dictionary replace — neither
interprets bucket contents. The entire fix is engine-layer
(`VacuumPlanBuilder` + `Indexer`), exactly as the issue's scope required.

## Invariant preserved

After `vacuum()` returns, every live chunk's `.bucketRef` resolves to the
exact same embedding it referenced before vacuum ran, for any bucket
vacuum rewrote — regardless of whether that chunk's chunkID is anywhere
near the abandoned chunkID range (buckets are k-means clusters, not
chunkID-range partitions, so this is not merely a "numerically adjacent
IDs" special case). `Indexer.Error.bucketRefUnresolvable` can no longer
fire as a side effect of vacuum's own compaction for any chunk vacuum
didn't abandon. Vacuum's existing contract for genuinely-abandoned chunks
(ledger entries removed, emptied buckets/generations cascade-deleted,
`numEmbeddings` corrected) and `findOrphanedChunks()`/R1 recovery are
unaffected — neither code path was touched.

## Alternatives considered

- **Option B — stable per-pair IDs instead of positional offsets.** Store
  a monotonically increasing per-pair ID alongside each pair; refs resolve
  by ID lookup rather than array index, so compaction can never invalidate
  them. This eliminates the whole bug *class* (any future bucket-rewriting
  code path would inherit the same immunity, not just vacuum), but is a
  broader wire-format change: every bucket writer (`performFlush()`,
  `rehydrateAutoRecover`'s `replaceGeneration` path) and reader
  (`SearchEngine`, `resolveBucketRefRow`) would need updating, plus
  per-pair storage overhead. Rejected as disproportionate to closing this
  specific gap — nothing in the current codebase other than vacuum
  rewrites bucket contents out from under a live ref, so the broader
  immunity has no current beneficiary.
- **Option C — tombstone instead of compact.** Mark abandoned pairs as
  tombstoned in place rather than removing them; bucket layout (and thus
  every live offset) stays stable, with space reclamation deferred to the
  next full bucket rewrite (a higher-generation compaction, which already
  updates ledger refs via `performFlush()`'s existing path). Simplest
  conceptually, but `SearchEngine`'s MaxSim scoring and the k-means
  training-sample loop both currently treat every `bucket.indices` entry
  as live/scorable/trainable with no tombstone concept — adopting this
  would require teaching both to skip tombstones, a wider surface than
  Option A touches. It also defers vacuum's actual purpose (bounding disk
  space and k-means training noise) indefinitely for any bucket that
  keeps getting touched, trading bounded reclaim latency for
  implementation simplicity elsewhere. Rejected: broader blast radius than
  Option A for no correctness benefit Option A doesn't already provide.

Option A was chosen because it is the minimal, fully localized fix: no
`SwitchcraftStorage` protocol change, no wire-format change, no change to
`SearchEngine` or any other bucket-content consumer, and no change to
`bucketRefUnresolvable`'s error type, message, or handling elsewhere —
consistent with the issue's explicit scope boundary.

## Cost / tradeoffs

- **Compute**: `O(bucketPairs)` per bucket vacuum touches — identical
  order to the existing per-bucket compaction loop, since the remap is
  computed from data that loop already produces. No new storage read, no
  new decode pass.
- **Memory**: one `BucketRefRemap` (2×`Int64` chunk/gen id fields plus a
  `UInt32`, an `Int64` bucket id, and an `Int` offset) per surviving pair
  in a touched bucket, held transiently for the duration of one
  `vacuum()` call. Not persisted, not part of any on-disk format.
- **No disk cost**: unlike Option C, steady-state disk usage is unchanged
  from before this fix — vacuum still actually removes abandoned pairs'
  bytes immediately, it just also fixes up the ledger for what remains.
- **API surface**: `Indexer.removeAbandonedFromLedger(_:)` renamed to
  `applyVacuumLedgerUpdates(abandonedChunkIDs:remaps:)`. One in-repo call
  site (`SwitchcraftStore.vacuum()`); an external `SwitchcraftCore`
  consumer calling this `Indexer`-internal method directly would need to
  adapt, acceptable pre-1.0 for internal LSM-compaction machinery, not
  documented public API.

## Consequences

- `VacuumTests.swift` gains three regression tests: a shared-bucket
  out-of-range reproducer (a live chunk's ref resolves correctly instead
  of throwing `bucketRefUnresolvable`), an adversarial in-range
  silent-wrong-value test (asserts the live chunk's resolved embedding
  equals its own golden value and explicitly *not* the value it would
  have silently resolved to pre-fix), and a sustained multi-cycle
  shared-bucket workload test that forces a cascade after every vacuum
  call — necessary because neither vacuum's own counters nor `search()`
  can observe a stale ref; only a subsequent cascade's
  `resolveBucketRefRow` call can.
- No change to retrieval/ranking behavior for live data — NFCorpus and
  33-fact corpus benchmarks are unaffected (asset-gated in this
  environment, unchanged skip state; this fix touches no scoring path).
- `findOrphanedChunks()`/R1 recovery unaffected — orthogonal code paths,
  neither read nor written by this change.

## Relationship to prior ADRs

- **ADR 029** (orphan chunk detection/recovery, amended for vacuum #134):
  vacuum's contract for abandoned-chunk removal, cascade deletion, and
  `numEmbeddings` correction is unchanged; this ADR only adds the missing
  live-ref-preservation half of vacuum's own "Ledger consistency"
  guarantee.
- **ADR 036** (lazy ledger bucket-ref materialization, #137): this ADR
  closes the gap between ADR 036's stated assumption (nothing outside the
  Indexer's own flush/compaction path rewrites bucket contents under a
  live ref) and vacuum's actual behavior. `.bucketRef`'s shape,
  `resolveBucketRefRow`, and `bucketRefUnresolvable`'s semantics are all
  unchanged — this is a vacuum-side fix, not a lazy-materialization
  change, per the issue's explicit scope.

## Out of scope

- Any change to `.bucketRef`'s shape or to `resolveBucketRefRow`/
  `bucketRefUnresolvable`'s error semantics.
- Any change to `SwitchcraftStorage` protocol surface or on-disk bucket
  format.
- Stable per-pair IDs (Option B) or tombstoning (Option C) — evaluated
  and rejected above in favor of the narrower, fully localized fix.
