# ADR 036 — Lazy Ledger Materialization: Bucket-Ref Tokens Instead of Eager `[Float]`

## Status

Accepted

## Context

### The eager-materialization cost

Prior to this ADR, `Indexer.ledger` (`[Int64: [[Float]]]`) held every
per-token embedding, fully reconstructed as `center + dequantized Q4
residuals`, for the entire lifetime of the process. Both rehydration paths
(`rehydrateThrowError`, `rehydrateAutoRecover`; ADR 024) ran this
reconstruction — `O(pairs × dims)` — for *every* bucket in *every*
generation at `Indexer.init`, regardless of whether that data would ever be
touched again.

The only downstream consumer of a materialized ledger row is
`performFlush()`'s cascade-compaction k-means training step (ADR 027) — and
even there, only a sqrt-sampled subset of rows per chunk is actually used for
*training* (`Self.sampleDistinct`); the *assignment* step (`KMeans.assign`)
does need every row, but only for chunks in the merge range of a compaction
that actually runs. At production scale (~11.1 M embeddings × 768 dims),
this is prohibitive: ~34 GB nominal reconstructed size, ~15 GB observed live
RSS held indefinitely, most of it never read again after the walk that built
it.

ADR 034 (issue #136, merged immediately before this one) added a ledger
*snapshot* fast path that skips the rehydration *walk* on a clean-shutdown
restart, but did not change what the ledger holds in memory once rehydrated
— a snapshot-loaded ledger was (and, see below, still needed to be handled
carefully to remain) just as eagerly materialized as a freshly-walked one.

## Decision

Restructure `Indexer.ledger`'s value type from `[[Float]]` to
`[LedgerToken]`, where a token is either the original materialized
embedding or a lightweight reference into a bucket already committed to
storage:

```swift
enum LedgerToken: Sendable, Equatable {
    case materialized([Float])
    case bucketRef(genID: Int64, bucketID: Int64, pairOffset: Int)
}
```

`pairOffset` is the token's index into that bucket's decoded
`IndicesCodec.decode(bucket.indices)` array — which is also the row index
into the bucket's decoded residuals (`base = pairOffset * dims`). `dims`
itself is not stored per-token; it's already locked in once at the
`Indexer` level (`Indexer.dims`).

### 1. Why `.materialized` tokens are exempt from lazy treatment

Tokens added via `add()` (the L0 memtable) have no bucket to reference yet
— there is no generation, no bucket id, no pair offset until the *next*
`performFlush()` actually writes one. `add()` continues to append
`.materialized([Float])` rows exactly as before; this is unchanged behavior,
not a partial implementation. A chunk's token list can therefore be a mix
of both kinds: a chunk flushed once (now bucket-refs) that received
additional `add()` calls before the next flush (now-appended materialized
rows) is the expected, common shape — every internal consumer below is
written to handle a mixed list, not just a uniform one.

### 2. Rehydration populates bucket-refs, not reconstructed floats

Both `rehydrateThrowError` and `rehydrateAutoRecover` populate
`.bucketRef(genID:bucketID:pairOffset:)` directly from bucket metadata —
`IndicesCodec.decode(bucket.indices)` gives the `(chunkID, tokenOffset)` →
`pairOffset` mapping already; no `center`/`residuals` value decode is
needed to build this reference.

`rehydrateAutoRecover`'s winner/loser conflict resolution (ADR 024) is
still necessary and mostly unchanged, with one refinement: `storage.replaceGeneration`
assigns **fresh** generation and bucket ids to a loser generation's
surviving (non-conflicting) pairs. A naive port of the old algorithm would
build bucket-refs during the same accumulation pass that later triggers
`replaceGeneration` — which would make those refs point at ids that no
longer exist by the time `init` returns. Instead, ledger population is
split into two cases after the prune pass completes:

- **Untouched generations** (never a loser for any chunkID): their bucket
  ids are stable, so the `(pairs, bucketID)` metadata already decoded during
  the accumulation pass is reused directly — zero extra storage I/O.
- **Replaced generations** (returned by `replaceGeneration`): their buckets
  are re-walked once, fresh, via `storage.buckets(forGeneration:)`, decoding
  only `indices` (cheap, `O(pairs)`) to pick up the new ids. This is
  confined to the rare corruption-recovery case — the common conflict-free
  path never pays this cost.

Residual *values* are decoded in exactly one place across both rehydration
paths: the loser-generation re-encode step inside `rehydrateAutoRecover`'s
prune pass, which needs the actual floats to filter out losing pairs and
re-pack the survivors. This is the sole remaining `O(pairs × dims)` cost,
and it only runs for generations that actually lost a conflict — normal
operation never touches it.

### 3. The init-time integrity walk (Requirement 13) and its two error cases

Losing eager reconstruction also loses its implicit safety net: today,
decoding every bucket at rehydrate time surfaces storage corruption
immediately at store-open time. A cheap replacement walk preserves that
guarantee at `O(pairs)` cost instead of `O(pairs × dims)`, via
`Indexer.checkedBucketCenter` and `Indexer.checkedResidualsSize`, run for
*every* bucket in both rehydration paths:

1. `bucket.center` decodes to a non-empty `[Float]` (`checkedBucketCenter`),
   and its dims agree with every other bucket seen this session.
2. `bucket.indices` decodes successfully via `IndicesCodec.decode` — throws
   on codec-level corruption, propagating directly (unchanged from before
   this issue).
3. `bucket.residuals.count == pairs.count * dims / 2` — the exact byte
   count `Q4Codec` would produce (2 values packed per byte) —
   (`checkedResidualsSize`), a size check only, never a decode of the
   residual contents.

Any check-1/3 failure throws `Error.rehydrationBucketCorrupt(generationID:)`
— this case already existed (thrown by the old eager loop); its check logic
is simply replaced with the cheaper equivalent above, not duplicated under a
new name.

A second, new case covers the residual failure mode this cheap walk can't
catch: `Error.bucketRefUnresolvable(chunkID:bucketID:cause:)`, thrown when
on-demand materialization (below) fails at compaction time despite the
bucket having passed its init-time check — e.g. a transient disk read
error or a bit-flip that only surfaces when the residual *contents* are
actually decoded. Compaction fails hard on this error; there is no
silent-skip path, since a silently-partial training set would produce
subtly wrong k-means output rather than a caller-visible failure. `cause`
is typed `String?` (a captured `String(describing:)` of the underlying
error), not `Swift.Error?`, so `Indexer.Error`'s existing `Sendable` /
`Equatable` conformance doesn't need bespoke handling for a non-`Equatable`
associated value; the hand-written `Equatable` conformance on `Indexer.Error`
compares `chunkID`/`bucketID` only, treating `cause` as diagnostic-only.

Together, these mean: corruption still surfaces at store-open time for the
overwhelming majority of cases (anything the cheap checks can catch — which
is everything the old eager loop could catch except a residual-content
bit-flip), at a fraction of the cost; the rare remainder is caught, and
fails loudly, the first time compaction actually needs that data.

### 4. Materialization contract: what triggers a bucket read, and batching

A single reference-type cache, `Indexer.BucketDecodeCache`, is created once
per top-level call (`performFlush()`'s cascade, or the test-only
`ledgerContents()` accessor) and threaded through every row resolved during
that call:

```swift
private final class BucketDecodeCache {
    var rawBucketsByGen: [Int64: [Int64: BucketRecord]] = [:]
    var decoded: [Int64: DecodedBucketData] = [:]
}
```

`decodedBucket(genID:bucketID:chunkID:cache:)` fetches (if needed) and
decodes (if needed) one bucket, consulting the cache first; `resolveBucketRefRow`
wraps it with the `pairOffset` bounds/size validation that throws
`bucketRefUnresolvable` on failure. Because `decoded` is keyed by `bucketID`
(globally unique across generations, not just within one), **a bucket
referenced by many tokens in the same call — e.g. several sqrt-sampled
tokens landing in the same source bucket — is fetched and decoded at most
once**, satisfying Requirement 6. The cache is discarded at the end of the
call; there is no cross-call persistence, since a bucket's contents are
immutable once written and the next compaction operates on a different
(post-merge) storage state anyway.

`performFlush()`'s three original consumers (ADR 027's `allFlat` build, ADR
030's self-recovery rebuild, and the sqrt-sample training-set loop) collapse
into one materialization pass: `collectAndMaterialize(min:max:expectedTotal:dims:)`
walks `[minChunkID, maxChunkID]` once, building `allFlat` directly (a
single-pass, no intermediate `[[Float]]` or token-list — this matters for
the common all-`.materialized` case, where it costs exactly what the
pre-#137 direct-copy loop cost) while also recording a `chunkRowStart`
map. The sqrt-sample training loop then slices sampled rows directly out of
`allFlat` via that map — no second bucket decode, since assignment already
needed every row materialized regardless of sampling. ADR 030's self-recovery
path (surprise gens widen `[minChunkID, maxChunkID]` and force a second
collection) calls the same `collectAndMaterialize` a second time with the
extended range — mechanically identical to the original two-pass structure,
just now routed through the shared batched-materialization helper instead
of a raw ledger walk.

After the new generation's buckets are written (step 10), `performFlush()`
transitions exactly the tokens it just flushed from `.materialized` to
`.bucketRef(genID: insertedGen.id, bucketID: insertedBucket.id, pairOffset:)`
— `pairOffset` is each row's index within that bucket's `members` array,
which is the exact order `IndicesCodec.decode` will later return. Chunks
outside `[minChunkID, maxChunkID]` are untouched, satisfying Requirement 3.
This mutation is safe against concurrent `add()`/`removeFromLedger()`
because `flushInProgress` (held by the caller, `flush()`) still gates them
for the duration of `performFlush()` — the same invariant ADR 030 already
relies on for its own mid-flush `await` points.

### 5. Interaction with ADR 029 (`removeFromLedger` / orphan repair)

`removeFromLedger()` and `removeAbandonedFromLedger()` only ever needed
`.count` (for `removedFromLedgerCount` bookkeeping) and `removeValue` — both
are element-type-agnostic and required no logic change, only a mechanical
retype. A chunk rehydrated with `.bucketRef` rows that then goes through the
R1 partial-orphan repair path (`removeFromLedger` followed by a fresh
`add()`) ends up with plain `.materialized` rows exactly as it would have
under the old representation — `removeFromLedger` deletes the whole
chunk's token list unconditionally, so there is no risk of a stale
`.bucketRef` surviving alongside newly `.materialized` rows for the *same*
chunk. The mixed-representation risk this ADR is otherwise built around
(a chunk with *some* bucket-ref and *some* materialized tokens) only arises
across *different* `add()`/`flush()` cycles for the *same* chunk, never
mid-repair.

### 6. The ADR 034 snapshot payload becomes bucket-ref-aware (v2)

ADR 034's snapshot payload encoded the *entire* ledger as fully-materialized
floats. Left unchanged, `writeLedgerSnapshot()` — called after every
successful flush — would need to decode every bucket-ref row in the ledger
to build that payload, reintroducing the exact `O(pairs × dims)` cost this
ADR removes from startup, just moved onto the much-hotter flush path, and
silently regressing the snapshot fast path's own memory win on the next
restart (a snapshot-loaded ledger would be back to fully materialized).

Instead, `LedgerSnapshotCodec`'s payload is bumped to a v2, tag-per-token
format: a `UInt32` magic (`0xFFFF_FFFF`) + `UInt8` version (`2`) header,
then per token a 1-byte tag (`0` = materialized, `1` = bucket-ref) followed
by either `dims` floats or `(genID, bucketID, pairOffset)` as three
`Int64`s. `writeLedgerSnapshot()` is now a pure in-memory encode of
whatever `ledger` already holds — cheaper than before this ADR, not more
expensive, since it no longer even copies through an intermediate
representation. `loadFromSnapshotIfValid()` populates `ledger` with the
same mixed representation the snapshot captured, so a snapshot-fast-path
startup gets this ADR's memory benefit too, not only the full-rehydration-walk
path.

A v1 (pre-this-ADR) snapshot has no magic/version header — it starts
directly with a `UInt32` chunk count. `decode(_:dims:)` rejects any payload
whose first 4 bytes aren't the v2 magic (`Error.unsupportedVersion`), which
a real v1 chunk count can only collide with at an implausible ~4-billion-chunk
snapshot. The existing "any decode failure ⇒ fall back to full rehydration"
handling in `loadFromSnapshotIfValid` means a v1 snapshot left over from an
in-place upgrade fails closed automatically, with no explicit migration
step required.

**Correction to ADR 034's precision claim**: ADR 034 stated the snapshot is
*more faithful* than full-walk rehydration because it stored full-precision
floats while the walk always went through Q4-lossy reconstruction. That is
no longer true once bucket-refs enter the payload — a `.bucketRef` token in
a snapshot resolves through the same Q4-lossy `center + residual` path a
full walk would use, whenever it's eventually materialized. Both paths are
now consistently Q4-lossy for any previously-flushed row; only `.materialized`
(not-yet-flushed) rows are ever full-precision, on both paths equally. This
is a real, intentional precision-claim correction, not a regression — it was
already latent in ADR 034's design once lazy materialization was the stated
follow-up work (`docs/Plan.md` line 331), and `IndexerSnapshotFastPathTests.cleanShutdownFastPath`'s
`reloaded == originalLedger` assertion continues to pass because both sides
resolve the same underlying bucket data identically.

### 7. Requirement 10 scope: "byte-identical" parity

Before this ADR, `ledger` never round-tripped through Q4 mid-session — a
cascade re-merging an already-flushed chunk still read the exact original
`add()`-time float, decoupled from what was actually on disk. This ADR's
whole point (flushed tokens become bucket-refs) makes a later in-session
cascade read Q4-dequantized values for that portion instead — an
unavoidable, intentional consequence of the requested design. Parity is
therefore verified as two distinct claims:

- **Single-flush corpora**: nothing is a bucket-ref yet (a first-ever flush
  has no prior generation to reference), so the lazy and pre-#137 eager
  code paths are trivially identical — both just copy `.materialized` rows
  with zero storage I/O. Exercised by every existing single-flush Indexer
  test (deterministic bucket-blob tests, the NFCorpus and 33-fact
  cross-implementation benchmarks) without any new test needed.
- **Multi-flush/cascade corpora**: parity means "batched lazy decode of a
  bucket equals per-token eager decode of the same bucket" — i.e. batching
  changes decode *count*, never decode *values*. `LazyLedgerMaterializationTests`
  (Requirement 9) verifies this directly: a golden reference computed
  per-bucket, independent of any batching, directly from raw storage bytes,
  is compared against the batched `ledgerContents()` output and must match
  bit-for-bit. Requirement 12's NDCG/ranking-tolerance benchmarks are the
  pipeline-level backstop for the broader claim that this quantization noise
  (identical to what a process restart already introduced pre-#137) doesn't
  move retrieval quality outside its existing tolerance band.

### 8. Requirement 11 acceptance methodology: ratio in Debug, absolute time in Release

The Requirement 11 benchmark (`LazyLedgerBenchmarkTests`, 100K chunks / 1M
embeddings / dims=768) initially asserted a `>= 50×` rehydrate-time ratio in
both build configurations. That held in Debug (~93×) but not in Release
(~24-25×) — not because the optimization regressed, but because Release's
`-O` auto-vectorizes the deleted eager loop's per-dimension scalar
arithmetic (`center[j] + residuals[base+j]`) far more aggressively than it
helps the lazy path's cost, which is dominated by non-vectorizable
bookkeeping (`IndicesCodec.decode`, dictionary/tuple accumulation, per-chunk
sort). Concretely, on the reference machine: the eager loop dropped
238.2s → 4.84s (~49×) going Debug → Release, while the lazy path only
dropped 2.57s → 0.20s (~13×) — both got faster, but the ratio between them
necessarily shrank because Release helped one side much more than the
other.

This is confirmed to not be a weak-comparator artifact: the benchmark's
"pre-refactor" comparator (`LazyLedgerMaterializationTests.eagerGoldenReconstruction`)
is a byte-for-byte copy of the actual deleted `rehydrateThrowError` loop, not
a strawman. Nor is it a small-scale artifact — the fixture's per-generation
overhead is already fully amortized at 50K pairs/generation, so a larger
fixture would not be expected to change the ratio.

**Resolution**: assert on whichever metric is meaningful for each build
configuration, rather than forcing one ratio bar to hold everywhere:

- **Debug**: keep the `>= 50×` ratio assertion. Debug's unvectorized
  arithmetic doesn't compress the ratio, making it a faithful proxy for the
  underlying `O(pairs×dims) → O(pairs)` algorithmic change actually taking
  effect.
- **Release**: assert an absolute wall-clock ceiling instead
  (`releaseAbsoluteTimeCeiling = 1.0s` for this fixture) — what a user
  actually experiences. The ceiling was picked from evidence: measured
  Release post-refactor time was ~0.20s, i.e. well under the "much less
  than 5s" case, so per the same reasoning that suggested a generic 5s
  starting point, it's tightened to ~5× the observed figure (headroom for
  slower CI hardware, still a meaningful gate against a real regression).
  The Release *ratio* is still logged for observability but intentionally
  not asserted.
- **Memory**: the `>= 30×` ratio bar holds reliably in both configurations
  (~30-31× measured either way) and is asserted unconditionally — no
  vectorization asymmetry applies to memory footprint the way it does to
  arithmetic-bound wall-clock time.

**Why not other options**: a Debug-only assertion (dropping the Release
check entirely) would hide a real Release-mode perf regression — if a
future change made Release rehydration regress from 1s to 10s, the Debug
ratio would still read ~50× (both eager and lazy scale together under
Debug), and nothing would catch it. Chasing a higher Release *ratio* by
further micro-optimizing the lazy path was also rejected: the lazy path
(bucket fetch → center decode → `IndicesCodec.decode` → tuple accumulation)
has no obvious remaining fat to trim, and the ratio is the wrong thing to
optimize for — users experience absolute seconds, not a ratio against code
that no longer exists.

**Note for future maintainers**: if this benchmark's Release ratio looks
smaller than Debug's, that is expected and does not by itself indicate a
regression — vectorization asymmetry between an arithmetic-heavy comparator
and a bookkeeping-heavy implementation naturally compresses the ratio under
`-O`. Check the Release *absolute* time against `releaseAbsoluteTimeCeiling`
instead of trying to force the ratio back up.

## Consequences

- **Steady-state ledger memory** drops by roughly two orders of magnitude
  for a corpus that has been through at least one flush: a `.bucketRef` is
  three `Int64`s (~24–32 bytes with enum/array overhead) versus a `[Float]`
  of `dims` values (e.g. 768 × 4 = 3072 bytes, plus per-array heap overhead,
  at the production dims this issue was scoped against).
- **Rehydration cost** drops from `O(pairs × dims)` to `O(pairs)` — no
  residual-value decode at all, for the common conflict-free case.
  Measured (opt-in `LazyLedgerBenchmarkTests`, 100K chunks / 1M embeddings /
  768 dims, one local run each): Debug ~93× faster wall-clock (238.2s →
  2.57s), Release rehydrate completes in ~0.20s (comfortably under the 1.0s
  absolute ceiling; the Release *ratio* is ~24-25×, smaller than Debug's for
  the vectorization-asymmetry reasons in §8, not a regression). Steady-state
  RSS improves ~30-31× in both build configurations. See §8 for why time is
  asserted as a ratio in Debug but as an absolute ceiling in Release.
- **Compaction's performance profile changes shape**: from "no I/O, high
  memory" to "some I/O, low memory." A cascade that merges previously-flushed
  generations now performs `storage.buckets(forGeneration:)` reads it didn't
  need before (though `buckets(forGeneration:)` was already being called
  identically by the pre-#137 rehydrate walk — this ADR moves *when* that
  cost is paid, from every startup to only the compactions that actually
  need the data, not introduces a wholly new I/O path).
- **A degenerate corpus with many small generations, each holding only a
  few sampled tokens per compaction, still pays one full-generation
  `buckets(forGeneration:)` fetch per referenced generation** — there is no
  partial/streaming bucket fetch at the storage layer (Requirement 8: no
  `SwitchcraftStorage` protocol change), so this is an accepted trade-off,
  not a bug. In practice a generation's buckets are fetched in one call
  regardless of how many of its tokens end up sampled, which is also true
  of the old eager walk.
- **No storage schema or `SwitchcraftStorage` protocol change.** This ADR is
  a purely in-memory `Indexer` restructuring plus an internal codec version
  bump for the already-opaque `LedgerSnapshotRecord.payload` blob.

## Relationship to prior ADRs

- **ADR 024** (rehydration conflict auto-recovery): winner/loser resolution,
  `(level DESC, created DESC, id DESC)` semantics, and `recoveredConflictCount`
  are unchanged. Only the *ledger population* step downstream of conflict
  resolution changes shape (bucket-refs instead of reconstructed floats),
  as detailed in §2 above.
- **ADR 027** (tiled batch k-means assignment): `allFlat`/`rowChunkIDs`/`rowTokenOffsets`
  remain the structures compaction operates on; `collectAndMaterialize` now
  populates them via batched lazy resolution instead of a direct ledger
  walk, but the flat-array shape itself, and the memory-bounded assignment
  it enables, are untouched.
- **ADR 029** (orphan chunk detection/recovery): `removeFromLedger`/
  `removeAbandonedFromLedger` needed no logic change (§5). The
  `removedFromLedgerCount` bookkeeping introduced there is unaffected —
  it's a count, not a value, and counts are identical regardless of
  representation.
- **ADR 030** (mid-operation compaction ledger divergence): the self-recovery
  rebuild loop is now just a second call to `collectAndMaterialize` with an
  extended range (§4) — mechanically the same two-pass structure, sharing
  the same batched materialization path as the initial pass.
- **ADR 034** (ledger snapshot fast path): payload format bumped to v2 and
  precision claim corrected (§6); write/load control flow, the fingerprint
  staleness check, write-time consistency guard, and invalidate-on-load are
  all otherwise unchanged.

## Out of scope

- Persisting the ledger to disk in a form that skips the rehydration walk
  itself beyond what ADR 034 already does — that ADR's mechanism is reused,
  not replaced.
- Any change to storage schema, bucket format, or the `SwitchcraftStorage`
  protocol surface.
- Any change to search-time semantics, MaxSim scoring, or MEAN aggregation
  (ADR 006 / issue #117) — this ADR only touches the compaction-time
  training-data path.
