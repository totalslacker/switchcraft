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
evaluation order. The engine never introduces parallelism (no
`TaskGroup`, no `cblas_set_num_threads` overrides) so identical inputs
and storage state produce bit-identical `[SearchHit]` output. Tie
breaks on the final list use `(-score, uuid)` ascending via Swift's
stable sort.

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
