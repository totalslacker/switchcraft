# ADR 027 — Tiled-batch KMeans assignment to bound peak memory

**Status**: Accepted
**Date**: 2026-06-04
**Issue**: #112 (bug: LSM cascade compaction allocates >100 GB anonymous heap)

This ADR documents why `KMeans.assignAll()` now processes data in tiles
of B rows rather than computing the full m × k scores matrix at once,
the choice of tile size B = 4 096, and the constraint that `assignBatchSize`
must not be raised without re-evaluating peak memory at expected scale.

---

## (a) The O(m^1.5) pathology

`KMeans.assignAll()` computes the scores matrix `S = data × centroidsᵀ`
(m rows × k columns) via a single `cblas_sgemm` call, then takes the
per-row argmax. With Witchcraft's k-formula (`k = round(16 × √m)`),
the matrix is O(m × k) = O(m × 16√m) = **O(m^1.5)** bytes:

| m (embeddings) | k (centroids) | S size (bytes) |
|---------------:|-------------:|---------------:|
|       33 facts |          ~70 |          ~18 KB |
|    5 000 (perf gate) |       ~1 131 |          ~23 MB |
|    600 000 (issue #112 reproducer) |     ~12 395 |       ~29 GB |
|  1 585 218 (integrator incident) |     ~20 142 |     ~128 GB |
| ~720 000 000 (NFCorpus est.) |    ~429 447 |      ~1.2 PB |

At reproducer scale (m = 1 585 218, k = 20 142), a single `cblas_sgemm`
call allocates **~128 GB** of anonymous heap, causing an OOM before
`[Compact] finished` can fire.

The growth signature matched the incident data precisely: sustained
linear ~15 GB / 5 s, fresh anonymous pages (not file-mapped), zero
`swapouts_delta`, and ~2 099 MALLOC_LARGE regions averaging ~128 MB each
(the successive 128 GB allocations freed then reallocated across k-means
training iterations).

## (b) The fix: tiled SGEMM

The fix is to partition the m rows into tiles of B rows, compute the
B × k scores for each tile, take the per-row argmax immediately, and
reuse the B × k buffer for the next tile. Peak scores memory drops from
O(m^1.5) to O(B × k) = O(B × √m) — constant for fixed B.

Per-row argmax is **row-independent**: row i's assignment depends only
on row i's scores against all k centroids, not on any other row. Tiling
the SGEMM call produces bit-identical per-row argmax results compared to
the single-call version (ignoring float tie-breaking, which is equally
indeterminate in both cases). The NFCorpus NDCG@10 gate (0.31–0.33)
confirms correctness post-fix.

The fix is applied inside `assignAll()`, so it benefits both callers
automatically:
- `KMeans.cluster()` training iterations (on `trainFlat`, trainM rows)
- `KMeans.assign()` final assignment (on `allFlat`, all m rows)

## (c) Tile size choice: B = 4 096

The dominant cost of the fix is the number of `cblas_sgemm` calls:
`⌈m / B⌉` calls replace 1. Each call carries a fixed setup cost
(negligible for Accelerate's pre-linked BLAS), and the centroids matrix
(k × dims = ~10 MB at reproducer scale) must be re-read for each tile.
At reproducer scale this matrix fits in L3 cache on Apple silicon, so
call overhead is dominated by the BLAS startup, not cache misses.

| B | scores buffer | SGEMM calls (m=1.585M) | SGEMM calls (trainM=112K) |
|----:|-------------:|------------------------:|--------------------------:|
|  512 |        41 MB |                    3 096 |                         218 |
| 1 024 |        82 MB |                    1 548 |                         110 |
| **4 096** |  **330 MB** |              **387** |                      **28** |
| 16 384 |     1 320 MB |                       97 |                           7 |

B = 4 096 caps the scores buffer at **~330 MB** at reproducer scale while
keeping SGEMM call counts (387 for the full-m assign, 28 per training
iteration × 6 = 168 total) well below any overhead threshold. On the
search path (m = query token count ≈ 64–256), B = 4 096 always covers
the entire input in a single call — no performance regression.

The tile size is defined as:

```swift
private static let assignBatchSize = 4_096
```

It is intentionally `private` (not a public configuration knob) so that
callers cannot inadvertently raise it to a value that reintroduces the
pathology. See §(e).

## (d) Secondary improvements in the same fix

Two additional allocation reductions were made alongside the tiling fix:

1. **Eliminated `allRows: [[Float]]` in `performFlush()`.**
   The old code built an intermediate `[[Float]]` (one Swift Array object
   per token) and then copied it into a flat `allFlat: [Float]` for BLAS.
   At 1.585M tokens this created 1.585M heap-allocated Array objects
   (~876 MB of overhead). The new code builds `allFlat`, `rowChunkIDs`,
   and `rowTokenOffsets` in a single pass through the ledger, eliminating
   both the intermediate `[[Float]]` and the second copy of float data.

2. **Added `autoreleasepool` around `BucketRecord` construction.**
   `IndicesCodec.encode` and `Q4Codec.encodeResiduals` produce Foundation
   `Data` objects. In an `async` context these accumulate until the Swift
   concurrency runtime drains the autorelease pool at an unspecified time.
   With k = 20 142 iterations and ~5 KB per bucket, the accumulation
   reached ~100 MB. Wrapping the synchronous codec calls in
   `autoreleasepool { BucketRecord(...) }` (outside the `await`) ensures
   transient ObjC temporaries are released after each bucket.

## (e) Constraint: do not raise `assignBatchSize` without scale analysis

The per-tile memory is `assignBatchSize × k × 4` bytes. Because k grows
as `16 × √m`, raising B by a factor of N multiplies peak tile memory by N:

| assignBatchSize | Peak tile (m=1.585M, k=20 142) |
|----------------:|-------------------------------:|
|           4 096 |                          330 MB |
|          16 384 |                        1 320 MB |
|          65 536 |                        5 280 MB |
|         262 144 |                       21 120 MB |

At NFCorpus scale (m ≈ 720M, k ≈ 429K):

| assignBatchSize | Peak tile (NFCorpus) |
|----------------:|---------------------:|
|           4 096 |                  703 GB |
|           1 024 |                  176 GB |
|             256 |                   44 GB |

NFCorpus at dims=128 would require B ≤ ~180 to stay within a 32 GB tile.
The current B = 4 096 is already too large for NFCorpus scale — a
follow-up issue must either reduce `assignBatchSize` for that workload or
switch to a streaming argmax that never materialises a scores buffer at all
(e.g., computing dot products one centroid at a time or in centroid-blocks).

**Do not raise `assignBatchSize`** for a BLAS-efficiency benefit without
first computing the tile size at the largest expected m and k, and
verifying it stays below the available RAM ceiling on the target hardware.

## (f) Outcome

Peak `phys_footprint` growth during the 600K-embedding cascade compaction
test (3 input segments, dims=128, k ≈ 12 395):

- Before fix: >100 GB (extrapolated from the integrator incident)
- After fix:  **342 MB** (~1.2× ledger, well inside the 4× soft target)

`[Compact] finished` now fires for every `[Compact] started` under the
reproducer conditions. The fix preserves the log shape mandated by
ADR 004 §h, and produces no NDCG@10 regression against the conformance
corpus.
