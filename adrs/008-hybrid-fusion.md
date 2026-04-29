# ADR 008 — Hybrid fusion (RRF + FTS5) constants and policy

**Status**: Accepted
**Date**: 2026-04-29
**Issue**: #14 (Phase 1: RRF + FTS5 hybrid)

This ADR locks the parameters and policy for `SearchEngine.searchHybrid`
in `Sources/SwitchcraftCore/Search/SearchEngine.swift`, exposed through
`HybridConfig`. Hybrid search fuses the vector pipeline (per-token
MaxSim over LSM buckets, ADR 006) with the storage layer's full-text
search (`FullTextHit`, BM25 in SQLite, naive overlap in
`InMemoryStorage`) into a single ranked list via Reciprocal Rank
Fusion. Cite this ADR in any future PR that retunes a hybrid constant
or alters the fusion math.

---

## (a) Fusion formula = standard equal-weight Reciprocal Rank Fusion

For every uuid present in either source's top-`perSourceBudget` results:

```
rrf(uuid) = Σ_{s ∈ {vector, fts}} 1 / (rrfK + rank_s(uuid))
```

`rank_s` is **1-indexed** within source `s`. A document absent from a
source contributes 0 to the sum (no penalty term). Both sources
contribute with equal weight.

This is the formulation in Cormack & Clarke, "Reciprocal Rank Fusion
Outperforms Condorcet and Individual Rank Learning Methods" (SIGIR
2009), with no per-source weighting.

## (b) `rrfK` default = 60

`60` is the canonical Cormack & Clarke value. It is widely cited as
the default in modern hybrid retrieval systems and is the value
upstream Witchcraft uses for its (asymmetric) RRF as well — see (e).

## (c) `perSourceBudget` default = 50

Upstream Witchcraft does not hard-code a per-source budget; its CLI
forwards `top_k` (interactive: 10, NFCorpus benchmark CSV: 100) to
both sub-searches. Switchcraft separates the two: `topK` is the size
of the result set, `perSourceBudget` is the size of the candidate pool
each source contributes. `50` covers typical `topK ≤ 10` use cases
without inflating the union/filter pass; the NFCorpus benchmark issue
may override it to match upstream's `top_k = 100` if NDCG@10 misses
target.

## (d) Rank base = 1-indexed

The rank-1 document in a source contributes `1 / (rrfK + 1)`. This
matches Cormack & Clarke's original formulation and reads naturally:
no rank-0 magic constant. With `rrfK = 60`, rank 1 → `1/61`.

Upstream Witchcraft enumerates with Rust's `enumerate()` (0-based) so
its rank-1 contribution is `1 / (60 + 0) = 1/60`. The numerical
difference is small (≈1.6%) and uniform across all results, so it
does not change the relative ordering of fused hits — only the
absolute scores. Switchcraft prioritises a clean, conventional
formula over byte parity with upstream's hybrid scores.

## (e) Equal-weight (no `+3` FTS rank offset)

Upstream Witchcraft's `reciprocal_rank_fusion` (`src/lib.rs`) applies
an asymmetric rank offset of `+3` to its FTS source, i.e. the FTS
contribution is `1 / (3 + k + rank)` rather than `1 / (k + rank)`.
This down-weights BM25 relative to semantic similarity by a fixed
amount. Switchcraft does **not** replicate the offset; it ships
equal-weight RRF as the standard formulation.

Rationale:

- **Cross-implementation parity is at the ranking level, not the score
  level** (ADR 006(g)). NFCorpus NDCG@10 must land within 0.31–0.33;
  byte-identical hybrid scores are not a parity goal.
- **Cleaner primitive.** `HybridConfig` is a small public surface; an
  asymmetric per-source offset would force a configurable knob and a
  policy decision that we do not yet have data to justify.
- **Reversibility.** If NFCorpus benchmarks (separate issue) miss
  target with equal-weight RRF, a follow-up PR can add a configurable
  `ftsRankOffset` defaulting to 0; existing callers stay on the
  current behaviour.

## (f) Single-source fallback formula

When one source contributes no candidates (empty embeddings, empty
text, or that source returned `[]`), the fused score is just the
present source's contribution:

```
rrf(uuid) = 1 / (rrfK + rank_s(uuid))
```

No re-normalisation, no penalty term. Documents from one source that
do not appear in the other are first-class fused hits, not second-rate.
This is the same as the general formula with the absent source's term
fixed at 0.

When both sources are empty (`queryEmbeddings.count == 0` AND
`queryText` is empty/whitespace-only), `searchHybrid` returns `[]`.

## (g) Filter dual-push + union pass

`StorageFilter` is forwarded to both sub-calls (vector search and
`searchFullText`) so each backend can lower it natively for
performance. The fused result is then re-evaluated against the filter
by re-fetching each candidate `DocumentRecord` and calling
`StorageFilter.matches(_:)`. This is the correctness gate against any
future divergence between a backend's native filter lowering and the
canonical Swift evaluator.

`filter == .all` skips the per-uuid document fetch (the union pass
becomes a no-op).

## (h) Determinism

`searchHybrid` awaits the vector and FTS sub-calls **sequentially**
(no `TaskGroup`). Combined with:

- ADR 006(f) determinism for vector search,
- The `(score DESC, uuid ASC)` total order applied by both
  `InMemoryStorage.searchFullText` and `SQLiteStorage.searchFullText`
  (added in this issue),
- The `(score DESC, uuid ASC)` total order applied to the fused list,

identical inputs and storage state produce **byte-identical**
`[HybridHit]` output across runs. Tied per-source FTS scores no longer
introduce nondeterminism.
