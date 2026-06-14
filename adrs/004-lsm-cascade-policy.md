# ADR 004 — LSM cascade policy

**Status**: Accepted
**Date**: 2026-04-28
**Issue**: #10 (Phase 1: LSM-tree index)

This ADR records the cascade thresholds, k-formula, training-set
sampling, k-means iteration count, and deterministic-seed convention
used by `Indexer` in `Sources/SwitchcraftCore/Indexer/Indexer.swift`.
The values are taken directly from upstream Witchcraft (`src/lib.rs`)
so that an indexer built with `IndexerConfig.production` produces
buckets that approximate Witchcraft's behaviour as closely as the
spec's invariants allow.

Cite this ADR in any future PR that changes a cascade or k-means
parameter on the index path. Changing these values invalidates every
existing on-disk index.

---

## (a) Per-level capacity

`level_capacity(L) = l0Capacity * lsmFanout^(L + 1)`

This matches `level_capacity` in `lib.rs` (line 1208) exactly. With
`IndexerConfig.production` (`l0Capacity = 1024`, `lsmFanout = 16`):

| Level | Capacity (embeddings) |
|------:|----------------------:|
|     0 |                16 384 |
|     1 |               262 144 |
|     2 |             4 194 304 |
|     3 |            67 108 864 |

`IndexerConfig.testing(...)` defaults to (`4`, `2`), giving caps of
8, 16, 32, 64 — sized to trigger cascades in unit tests without
adding thousands of embeddings.

## (b) Cascade trigger

On `flush`, the indexer runs Witchcraft's `index_chunks` walk:

```
total = pending
target = 0
loop {
    cap = levelCapacity(target)
    total += sum(numEmbeddings of generations at target)
    if total <= cap { break }
    target += 1
}
```

The new generation is built at `target_level`; generations at every
level `<= target_level` are deleted (FK cascade removes their
buckets). The new generation's `numEmbeddings` equals `total`.

**Divergence from upstream (intentional, per spec)**: Witchcraft also
short-circuits with `if pending < L0_CAPACITY: return` — i.e., it
*buffers* below the L0 threshold and only writes a generation once
the buffer fills. Switchcraft's spec requires that any non-empty
`flush` produce a generation, so this guard is omitted. In steady
state (regular flushes) you still see ≤1 generation per level
because every flush merges the current level's contents into a
single new generation at `target_level`.

## (c) k-formula

`k = max(round(kCoefficient * sqrt(n)), 1)`

with `kCoefficient = 16.0` by default. `round` is round-half-away-
from-zero (Swift's `.rounded()`), matching Rust's `f64::round`.
`n` is the total embedding count to be clustered (i.e., the cascade
walk's `total`).

Two clamps follow:

1. If the training-set size `m < k`, then `k = max(m / 4, 1)`. This
   matches `run_kmeans_for_index` in `lib.rs` (lines 1196–1202) and
   handles tiny inputs (single-chunk single-token cases).
2. Final clamp `k = min(k, m)` so the `k <= m` precondition of
   `KMeans.cluster` always holds.

## (d) Training-set sampling

K-means is trained on a per-chunk square-root sample, not on the full
embedding matrix. For each chunk in the cascade range:

```
sampleK = ceil(sqrt(tokensInChunk))
pick sampleK distinct rows from that chunk via Floyd's algorithm
```

The sample feeds `KMeans.cluster`; the resulting centroids are then
used to *assign* the full embedding matrix (`KMeans.assign`).
Mirrors `sample_embeddings_for_kmeans` in `lib.rs` (lines 1168–1192).

This is what makes the 5,000-embedding perf gate achievable: the
training matrix scales with `Σ ceil(√tokensPerChunk)`, not the full
token count.

## (e) K-means iteration count

`kmeansIterations = 5` by default. Overrides
`KMeans.cluster`'s library default of 25 to match
`run_kmeans_for_index` line 1203 (`kmeans(matrix, k, 5)`).

## (f) Deterministic seed convention

The indexer holds a single `SplitMix64` seed (`IndexerConfig.seed`,
default `42`) which is re-instantiated at the start of every
`flush` and threaded through:

1. Per-chunk training-set sampling (Floyd's),
2. K-means centroid initialisation (`KMeans.cluster`'s internal RNG),
3. K-means empty-cluster reseeding.

Same seed + same input order + same config ⇒ byte-identical
`bucket.center`, `bucket.indices`, `bucket.residuals` blobs, as
required by the spec's determinism acceptance criterion. (Bucket and
generation IDs are backend-assigned and not part of the parity
contract.)

## (g) Indexer cascade source: in-memory ledger with rehydration

The indexer keeps every embedding in a `[Int64: [[Float]]]` ledger
keyed by chunkID. Cascades re-cluster by reading from the ledger
rather than from `storage.chunk(id:)`.

**Rehydration on construction (issue #80)**: `Indexer.init` is now
`async throws`. On construction it reads all active generations and
their buckets from storage, reconstructs each token embedding as
`center + dequantize(residuals)` (Option B — bucket reconstruction),
and populates the ledger so that `add` + `flush` sequences succeed
after any actor restart against non-empty persistent storage.

Reconstruction reverses the write path in `performFlush`: each stored
bucket contains a Float32-LE centroid (`center`) and a Q4-encoded
residual blob (`residuals`). Decoding gives `embedding[j] ≈ center[j] +
residuals[base + j]` for each token in the bucket. Tokens for the same
chunkID may be spread across multiple buckets; rehydration accumulates
all (tokenOffset, embedding) triples per chunkID and sorts by
tokenOffset before inserting into the ledger.

**Option B tradeoff**: reconstruction is lossy by Q4 quantization error
(~MSE 0.05/dim for normalised vectors). This shifts cascade k-means
centres slightly from a fresh-index baseline, but the NFCorpus NDCG@10
gate (0.31–0.33) is unaffected because the perturbation is small
relative to the variance in the embedding space. Zero additional storage
overhead, zero storage-protocol additions required.

**Storage-corruption detection**: if the same chunkID appears in two
different active generations during rehydration,
`Indexer.Error.rehydrationConflict(chunkID:)` is thrown. Normal
operation under LSM invariants guarantees each chunkID is in at most
one active generation.

## Memory footprint at scale

In-memory ledger size for 1M tokens at 128-dim Float32 ≈ 520 MB
(plus per-token Array overhead). Tractable for the spec's perf
gate and the 33-fact corpus; revisit before NFCorpus (~3.6M docs).

## (h) Compaction observability — `[Compact]` os.Logger events

Every `performFlush()` invocation emits structured info-level log lines
to the existing `indexerLogger` (`Logger(subsystem: "com.switchcraft.core",
category: "Indexer")`). No new subsystem or category is introduced.

### Log line formats

```
[Compact] started:    input_segments=<N> input_bytes=<X> trigger=<T>
[Compact] finished:   input_segments=<N> output_segments=<M> input_bytes=<X> output_bytes=<Y> elapsed_ms=<ms>
[Compact] cancelled:  processed_segments=<P>/<N> elapsed_ms=<ms>
[Compact] failed:     input_segments=<N> elapsed_ms=<ms> error=<description>
```

### Field semantics

| Field | Value |
|-------|-------|
| `input_segments` | `mergedGens.count + 1` — stored generations at level ≤ targetLevel plus the pending L0 buffer |
| `input_bytes` | `total × dims × 4` — cascade walk's total embedding count times Float32 byte size |
| `trigger` | `cascade` when `targetLevel > 0` (embeddings promoted into higher levels); `manual` when `targetLevel == 0` (confined L0 write). Renamed from `forced` in issue #113 to match the public `CompactionEvent.Trigger.manual` enum case. |
| `output_segments` | Always `1` — the current implementation produces exactly one output generation per flush |
| `output_bytes` | `m × dims × 4` — ledger row count (verified equal to `total`) times Float32 byte size |
| `elapsed_ms` | Wall-clock milliseconds from `[Compact] started` to the terminal event, using `Date()` |
| `processed_segments` | Segments written before cancellation; always `0/N` because the single cancellation checkpoint is at the pre-write boundary (step 9, before `storage.insertGeneration`) |
| `error` | `error.localizedDescription` from the catch arm |

### Privacy levels

All numeric values and the `trigger` enum string use `privacy: .public`
(safe to emit in production logs without redaction). The `error` field
uses `privacy: .auto` (redacted in release builds by default, visible in
debug and with explicit entitlement).

### Cancellation checkpoint

A single `Task.checkCancellation()` is inserted immediately before
`storage.insertGeneration` (step 9). This is the last safe cancellation
boundary: the cascade walk and k-means are pure computation, no storage
state has been mutated, and cancellation here leaves the database
consistent. No checkpoints are added inside the step-10 bucket-write
loop to avoid partial-write/ledger-corruption risk.

### `[Compact] progress` — omitted

The optional progress line is not implemented. `performFlush()` completes
in seconds on representative corpora; per-bucket progress tracking would
require non-trivial loop instrumentation for marginal benefit. File a
follow-up issue if long-running compactions become observable in
production workloads.

### Observer command

```
log show --info --debug --predicate 'eventMessage CONTAINS "[Compact]"'
```

This command shows all four event types. Combine with `--start` /
`--end` to scope to a workload window.
