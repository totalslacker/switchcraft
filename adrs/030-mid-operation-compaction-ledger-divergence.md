# ADR 030 — Mid-Operation Compaction Ledger Divergence

**Status:** Accepted
**Date:** 2026-06-13
**Refs:** [ADR 004](004-lsm-cascade-policy.md), [ADR 024](024-rehydration-conflict-resolution.md), [ADR 029](029-orphan-chunk-detection-recovery.md)

---

## Context

`Indexer.performFlush()` throws `ledgerOutOfSync` when `m != total` at the
row-count check immediately after the cascade-walk collects `mergedGens`.

In the upstream Witchcraft Rust port this invariant is always satisfied because
the Rust store serialises all mutations on a single thread. In Swift, the
`SwitchcraftStore` actor allows the orphan-recovery path (ADR 029 §b) to call
`indexer.add()` between the moment a caller enters the flush queue and the
moment `performFlush()` executes. This interleaving can cause a ledger count that
is higher than what the cascade walk expected, triggering the error mid-operation.

### (a) The trigger workload

The failure reproduces reliably with the following sequence:

1. A caller invokes `store.add("doc-a")` which queues one or more orphan chunks
   for re-indexing via `indexer.add()` (ADR 029 R1). The chunk's embeddings land
   in the Indexer's in-memory ledger and increment `pendingCount`.
2. While the flush is queued (or between compaction cascade steps), `store.add()`
   is called again, bumping the ledger for chunks whose `chunkID` falls inside
   an existing generation's `[minChunkID..maxChunkID]` range.
3. `performFlush()` runs the cascade walk from `pending` embeddings at `l0`.
   The walk accumulates `mergedGens` at level ≤ `targetLevel`. The resulting
   `total = pending + sum(mergedGens.numEmbeddings)`.
4. When the ledger rows are collected for `chunkIDs ∈ [minChunkID..maxChunkID]`,
   the sweep includes chunk rows that belong to a **higher-level generation** not
   included in `mergedGens`. The row count `m` therefore exceeds `total`.

### (b) Why the range sweeps in extra rows

The LSM cascade walk terminates when adding the next level would exceed
`capacity(targetLevel)`. This means there are often surviving generations at
levels higher than `targetLevel` whose chunk-ID ranges may overlap the
current flush range `[minChunkID..maxChunkID]`. Each such "surprise gen"
contributes rows to the ledger sweep but zero to `total`, causing `m > total`.

This scenario became possible once ADR 029's orphan recovery was merged: without
it, `indexer.add()` was only called from `store.add()` for brand-new content,
and the cascade walk boundaries never overlapped with existing generations in
practice.

---

## Decision

### (c) Self-recovery in `performFlush()`

When `m != total` is detected after the initial ledger sweep, instead of
throwing immediately, `performFlush()` attempts a **single extension pass**:

1. Build `mergedGenIDs = Set(mergedGens.map { $0.id })`.
2. Find **surprise gens**: all gens in `allGens` not in `mergedGenIDs` whose
   range overlaps `[minChunkID..maxChunkID]`:
   ```
   g.minChunkID <= maxChunkID && g.maxChunkID >= minChunkID
   ```
3. If no surprise gens exist, the divergence has no structural explanation —
   throw `ledgerOutOfSync(ledgerRows: m, expected: total)` as before.
4. Otherwise: log a warning, append surprise gens to `mergedGens`, extend
   `minChunkID`, `maxChunkID`, `total`, and `targetLevel` to subsume them.
5. Rebuild `sortedChunkIDs`, `allFlat`, `rowChunkIDs`, `rowTokenOffsets` for
   the extended range.
6. Recompute `m`. If `m != total`, throw `ledgerOutOfSync` with updated counts.
   If `m == total`, continue to step 4 (k-means training) with the extended
   `mergedGens` and row arrays.

After recovery, the k-means training set and centroid assignment use data from
all gens in the extended `mergedGens`. The extended `targetLevel` governs the
resulting generation's `level` field. All merged gens (original + surprise) are
deleted in step 11 as before.

### (d) Why a single extension pass suffices

The surprise-gen scan explicitly tests range overlap: a gen at any level is
included if and only if its chunk-ID range intersects `[minChunkID..maxChunkID]`.
After extending the range to cover the surprise gens, any gens that overlapped
the *original* range also overlap the *new* (wider) range. Any additional gens
whose ranges only become relevant because of the extension would have needed to
overlap the original range already (because surprise gens can only widen the
range, never introduce entirely disjoint intervals into the ledger sweep). A
single pass is therefore sufficient; a fixpoint loop is not needed.

### (e) Why `updateGenerationEmbeddingCount` is not called

The fix absorbs surprise gens into `mergedGens` and deletes all of them in
step 11. There are no surviving gens with stale `numEmbeddings` to correct.
The ADR 024 construction-time analog calls `updateGenerationEmbeddingCount` for
the *winning* gen that survives rehydration conflict — that pattern does not
apply here because all merged gens are deleted rather than preserved.

### (f) Log output

The recovery path emits an `os_log` warning at level `.warning`:

```
[Compact] mid-operation ledger divergence detected (ledger=<m> expected=<total>);
absorbing <n> surprise gen(s) — see ADR 030
```

On successful re-check, it emits an `os_log` info entry:

```
[Compact] self-recovered: extended to <mergedGens.count> merged gens,
targetLevel=<targetLevel>, total=<total>
```

---

## Consequences

**Positive:**
- Eliminates `ledgerOutOfSync` thrown mid-operation during sequential `add()`
  calls that cross a compaction boundary — previously a hard crash path.
- The fix is localised to the divergence check in `performFlush()` and does not
  change the storage protocol, public API, or LSM cascade policy.
- Surprise-gen detection is O(|allGens|) — negligible compared to the O(K)
  k-means training cost.

**Negative / risks:**
- `mergedGens`, `m`, and `sortedChunkIDs` must be `var` instead of `let` inside
  `performFlush()`. The mutable state is local to the function; there is no
  shared mutable state concern.
- A second ledger scan is required after extension. For large datasets the second
  scan is O(K) and dominated by the k-means cost that follows it.
- The safety guard (throw after the second `m != total` check) means that true
  invariant violations — data corruption, codec bugs — are still surfaced as
  errors rather than silently masked.

**Relationship to ADR 024 (rehydration conflict):** ADR 024 resolves conflicts
at *construction time* when two surviving gens claim overlapping chunk-ID ranges
from a previous session. ADR 030 resolves a *runtime* conflict inside a single
flush operation when in-flight `add()` calls widen the ledger beyond what the
cascade walk anticipated. The two paths are disjoint and complementary.
