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

## (g) Indexer cascade source: in-memory ledger

For the MVP the indexer keeps every embedding ever passed to `add`
in a `[Int64: [[Float]]]` ledger keyed by chunkID. Cascades re-cluster
by reading from the ledger rather than from `storage.chunk(id:)`.

This works around the absence of a defined `chunk.embeddings`
binary layout in Switchcraft today. The Search-pipeline issue will
introduce a `ChunkEmbeddingCodec`, after which the indexer will
switch to reading source embeddings from storage. Documented as a
known MVP limit on `Indexer`'s doc-comment; the ledger does not
survive an actor restart.

## Memory footprint at scale

In-memory ledger size for 1M tokens at 128-dim Float32 ≈ 520 MB
(plus per-token Array overhead). Tractable for the spec's perf
gate and the 33-fact corpus; revisit before NFCorpus (~3.6M docs).
