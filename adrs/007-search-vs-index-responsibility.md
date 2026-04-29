# ADR 007 — Search-vs-index responsibility split

**Status**: Accepted
**Date**: 2026-04-29
**Issue**: #12 (Phase 1: Search pipeline)

This ADR records who owns the in-memory representation of bucket
centroids at query time and how the search engine reads them. The
choice affects every search call's I/O profile and the storage
backends' caching responsibilities.

Cite this ADR in any future PR that introduces a centroid cache, an
in-memory generation manifest, or any sidecar structure used by the
search engine to avoid loading bucket records.

---

## (a) Storage owns persisted centroids

`SwitchcraftStorage` is the authoritative source for every `BucketRecord`,
including its `center` blob. The protocol exposes
`storage.buckets(forGeneration:)` which returns full `BucketRecord`
values; nothing in `SwitchcraftStorage` is allowed to omit the `center`
or substitute a cached representation that is stale relative to the
record on disk.

## (b) Search engine loads centroids per query, no cache

`SearchEngine` calls `storage.generations()` and then
`storage.buckets(forGeneration:)` for every generation on every
`search` call. It decodes each `bucket.center` (Float32 LE → `[Float]`)
on the fly and constructs the per-generation centroid matrix as a
local stack value.

There is no engine-side or storage-side in-memory cache of the
centroid matrix. The naive iterate-all-buckets approach matches
upstream Witchcraft's `get_all_generation_centers`.

## (c) Why no cache for MVP

- **Simplicity**: no consistency story is needed. Every search reads
  the current persisted state.
- **Tractable scale**: for the spec's 50-doc / 16K-embedding test
  corpus the per-query I/O is sub-millisecond. NFCorpus (~3.6 M docs)
  is still tractable on Apple silicon with `cblas_sgemm`-vectorised
  centroid scoring.
- **Spec budget**: per-test ≤ 5 s and per-suite ≤ 2 min are met with
  comfortable margin (`Search Engine` suite runs in ~25 ms in debug).
- **Phase 2 escape hatch**: this ADR is explicitly revisitable. If
  benchmarks on production-scale corpora demand an in-memory centroid
  matrix, a follow-up ADR will define ownership (engine vs. storage)
  and an invalidation contract.

## (d) Implications for storage backends

- Backends MUST keep `bucket.center` available for read on every
  `buckets(forGeneration:)` call.
- Backends MAY internally cache decoded centroids, but the protocol
  contract is that the engine sees the persisted bytes — any cache
  must be transparent.
- Backends MUST NOT skip writing centroids to disk in anticipation
  of an in-memory representation.

## (e) Concurrency

The search engine's centroid scan runs sequentially across generations
to keep `cblas_sgemm` summation order deterministic (see ADR 006 (f)).
Backends are async actors, so each `buckets(forGeneration:)` call is
non-blocking, but the engine awaits each call before issuing the
next. No `TaskGroup` is used in this issue.

## (f) Centroid format references back to ADR 005

`bucket.center` is `dims * sizeof(Float32)` bytes in explicit
little-endian, decoded by `Indexer.decodeFloat32LE`. Search must
decode through that helper (or its byte-equivalent) to remain robust
to any future big-endian backend.
