# ADR 006 — Search constants and aggregation choices

**Status**: Accepted
**Date**: 2026-04-29
**Issue**: #12 (Phase 1: Search pipeline)

This ADR locks the tuneable defaults and aggregation functions used
by `SearchEngine` in `Sources/SwitchcraftCore/Search/SearchEngine.swift`
and exposed through `SearchConfig`. The values are taken directly from
upstream Witchcraft's `match_centroids` (`src/lib.rs`) so that the
NFCorpus NDCG@10 cross-implementation parity gate (target: 0.31–0.33)
is achievable once `Tests/Fixtures/nfcorpus/` lands.

Cite this ADR in any future PR that retunes a search constant or
swaps an aggregation function. Changing these affects retrieval
ranking on every existing index.

---

## (a) Tuneable defaults

| Constant   | Value   | Source                         |
|------------|---------|--------------------------------|
| `k`        | 32      | Witchcraft `match_centroids`   |
| `tPrime`   | 40 000  | Witchcraft `match_centroids`   |
| `threshold`| 0.0     | Witchcraft default             |

`k` is the maximum number of centroids selected per query token, per
LSM generation. `tPrime` is the per-query-token soft budget on the
cumulative number of candidate token embeddings selected from buckets.

## (b) MaxSim aggregation = MEAN, not SUM

For each query token `q`, the engine takes the maximum dot product
across all candidate token embeddings of a chunk/document. Per-q-token
maxima are aggregated into a final score by **arithmetic mean** across
query tokens:

```
score = (1/n) · Σ_q max_t (q · t)
```

ColBERT-style late interaction uses **sum**; XTR/WARP follows ColBERT
in the paper. Witchcraft's implementation, however, applies a
`scaler = 1.0 / n_query_tokens` to the per-token maxima (`src/lib.rs`,
`match_centroids` final scoring loop) — i.e. mean. Switchcraft mirrors
that choice for byte-level parity at the score level, accepting that
ranking is unaffected when comparing scores within a single search
(the divisor is constant) but absolute scores differ from a literal
ColBERT implementation by exactly `1/n`.

## (c) `tPrime` stopping rule = per-query-token soft cap, append-then-check

Per query token, the selection loop appends a centroid to the candidate
set **before** checking the cumulative-token budget:

```
for j in 0..min(k, n_centroids):
    select centroid j
    cumsum += bucket_token_count[j]
    if cumsum >= tPrime: break
# cumsum resets for the next query token
```

The centroid that pushes `cumsum` over `tPrime` IS included. This
matches Witchcraft (`src/lib.rs`, ~line 1110). Each query token can
therefore select at most `min(k, n_centroids)` centroids per generation,
and at most one of them straddles the budget.

## (d) Document aggregation = pool all candidate tokens, per-q-token max, then mean

A document may be split into multiple chunks. Witchcraft uses a
running per-q-token max via `vmax_inplace + copy_from_slice` that
carries the maximum forward across chunks of the same document.
Equivalently: pool every candidate token across every chunk of the
document, take per-q-token max across the pool, then mean across `q`.

`SearchHit` is one entry per document uuid, scored as above. RRF
fusion (next issue) consumes `[SearchHit]` directly.

## (e) `missing[q]` baseline

When a query token's top-`min(k, n_centroids)` centroids in a
generation do not exhaust `tPrime`, the engine records:

```
missing[q] ← max(missing[q], score of the k-th centroid in this generation)
```

Documents that appear in the candidate set but do not contribute a
candidate token for query `q` use `missing[q]` as the per-token score.
This is Witchcraft's `missing_similarities` mechanism and is essential
for NFCorpus NDCG@10 parity. If a query token has no `missing[q]`
update from any generation (e.g. the index is empty), documents
contributing no evidence for that token are dropped from the result.

## (f) Determinism

`cblas_sgemm` is deterministic for a fixed input layout and serial
evaluation order. Final ordering uses the total order `(-score, uuid)`
ascending, so ties are broken deterministically without relying on
sort stability (Swift's `Array.sort` is not guaranteed stable).

**Amended by ADR 035** (issue #140): the default search path
(`SearchEngine.searchOptimized`, since ADR 035) *does* introduce
`TaskGroup`-based parallelism for the bucket-decode step — the
original "no `TaskGroup`" language above no longer holds for that
path. The bit-identical output guarantee is preserved, not loosened,
via a specific construction:

- Each `TaskGroup` child task decodes a contiguous, disjoint slice of
  the (already query-order-independent) deduplicated bucket set
  `selected` and touches no actor-isolated mutable state — only local
  `Data`/`[Float]` buffers and the immutable `BucketRecord` values in
  its slice.
- Child results are written into an array indexed by the child's
  chunk position (not by completion order) and concatenated back in
  that fixed order once every child has returned.
- `cblas_sgemm` itself is still invoked exactly once per phase (once
  per LSM generation for centroid scoring, once for final candidate
  scoring) with the same fixed argument order as before — parallelism
  is confined to the decode step between those two matmuls, which
  contains no floating-point reduction (`center + residual` is an
  elementwise add, not a sum-reduction, so operation grouping doesn't
  matter).

Because decode-chunk assignment depends only on `selected`'s length
(itself a deterministic function of the query and storage state, not
of wall-clock scheduling), output is bit-identical across repeated
runs of the same query against the same storage state, matching the
pre-#140 guarantee. `SearchDeterminismTests` asserts this directly:
repeated identical `search()` calls produce byte-identical
`[SearchHit]`.

Query-token dedup (ADR 035, applying ADR 028's proof that repeated
query tokens are a no-op under MEAN aggregation, in the collapse
direction) is a **separate** transformation with a separate
guarantee: `searchOptimized` and `searchLegacy` produce the same
ranked document order and scores within a tight tolerance, but not
necessarily bit-identical scores, when the query has duplicate
tokens. `searchOptimized` computes a duplicated token's contribution
as `v * Float(duplicateCount)`; `searchLegacy` computes it as
`duplicateCount` separate floating-point additions interleaved with
other tokens' contributions in the original per-token loop. Both
compute the same real number, but IEEE-754 addition is not
associative, so the two summation orders can round to adjacent
floats (observed: ~1e-7 relative, i.e. one ULP at this magnitude) —
the same class of arithmetic-reordering tolerance already accepted
elsewhere in this ADR for BLAS reduction order and in the NFCorpus /
facts-corpus parity suites (±0.01–0.025). `SearchDeterminismTests`
verifies both properties: bit-identical output for duplicate-free
queries, and same-ranking / tolerance-bounded scores for queries with
duplicated tokens.

The legacy path (`SearchEngine.searchLegacy`, reachable via
`SearchConfig.legacySequentialSearch = true`) is unchanged: still
fully sequential, no `TaskGroup`, no query-token dedup — retained as
a benchmark-only "before" baseline, not for production use.

## (g) Bucket-byte vs ranking-level parity

ADR 005 stores bucket pairs as `(chunkID, tokenOffset)` rather than
upstream's `(documentRowID, subIdx)`. Switchcraft bridges back to
documents via `storage.documents(forChunkHash:)`. Bucket bytes
captured from upstream Witchcraft will **not** byte-match Switchcraft
buckets, but the algorithm produces equivalent retrieval rankings.
NFCorpus NDCG@10 parity is verified at the ranking level
(±0.01 score tolerance, target 0.31–0.33), not the bucket-byte level.

## (h) Multi-document chunk attribution

Chunks are content-deduplicated. When two documents share a chunk
hash, both receive the same chunk-level evidence. Strong query
matches will therefore lift every sharing document together. This is
a deliberate consequence of ADR 005's choice to dedup chunks; it is
not a bug.
