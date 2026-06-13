# ADR 028 — Query-Token Surface-Form Length Filter

**Status:** Accepted  
**Issue:** #117  
**Date:** 2026-06-13

---

## Context

When a query is a single proper noun that SentencePiece tokenises into a small number of short, common subword fragments, XTR MaxSim scoring produces noisy top-K results at corpus scale.

**Concrete reproducer:** Query `"bartleby"` → tokens `["bar", "t", "le", "by"]`.

At ~5,000-document scale, the target document ranks first — but positions 2–10 are semantically unrelated pages. Their RRF fusion scores are within 0.01 of the target. Precision@5 ≈ 0.2.

**Root cause:** Tokens `"t"`, `"le"`, and `"by"` are extremely common subwords. At small scale they find genuine matches in the target. At large scale, nearly every page contains words with those fragments. The MaxSim operation correctly finds the *best* match per token, but "best" now means "least-bad in a large spurious-match set." The noise floor rises, compressing the precision gap.

The problem is asymmetric: multi-word queries (e.g. `"bartleby scrivener"`) produce 8+ query tokens — common fragments are outvoted by more-specific tokens. Single-word queries with short decompositions leave common fragments with outsized influence.

---

## Decision

Add an optional **query-token surface-form length filter** controlled by `SearchConfig.queryMinSurfaceFormLength: Int` (default `0`, meaning no filter).

When `queryMinSurfaceFormLength > 0`, `T5CoreMLEmbedder.encodeQuery()` suppresses token embedding rows whose decoded SentencePiece surface form has `count <= queryMinSurfaceFormLength`. At threshold `2`:

- `"t"` (length 1) → **filtered**
- `"le"` (length 2) → **filtered**
- `"by"` (length 2) → **filtered**
- `"bar"` (length 3) → **retained**

The filter is applied inside `T5CoreMLEmbedder._encodeImpl` after `SlidingWindow.merge()`, using `tokenizer.decode([tokenID])` to retrieve the surface form for each kept position.

### Interface

```swift
// Embedder protocol — default no-op for non-tokenizer-backed embedders
func encodeQuery(_ text: String, minSurfaceFormLength: Int) async throws -> [Float]

// SearchConfig
var queryMinSurfaceFormLength: Int = 0
```

`SwitchcraftStore.search()` calls `encodeQuery(query, minSurfaceFormLength: config.search.queryMinSurfaceFormLength)` instead of `encode(query)`, so the filter is applied on the query path only.

### Re-entrancy safety

`T5CoreMLEmbedder` is actor-isolated and guards against concurrent invocations with an `inFlight` flag. `encodeQuery()` would deadlock if it delegated to `self.encode()` while holding the guard. Instead, both `encode()` and `encodeQuery()` are thin wrappers over a private `_encodeImpl(_ text: String, minSurfaceFormLength: Int)` that owns the re-entrancy guard and `callCount` increment.

---

## Interaction with ADR 006 (MEAN aggregation)

ADR 006 §(b) specifies MEAN aggregation: `score = (1/n) Σ maxSim(qi)` where `n` is the number of query-token embeddings.

When `queryMinSurfaceFormLength > 0`, the filter removes rows from the query embedding buffer, reducing `n`. This changes absolute scores but **not the relative ranking** within a single query — the same scale factor applies to all candidates.

**Witchcraft parity is preserved at the default (`queryMinSurfaceFormLength = 0`).** The 33-fact corpus parity tests and NFCorpus NDCG@10 gate (0.31–0.33) use the default and remain unaffected.

When the filter is enabled, NFCorpus NDCG@10 may shift. Callers enabling the filter should re-run the NFCorpus benchmark to confirm scores remain acceptable for their use case. The filter is intended for short proper-noun queries; multi-word medical/scientific queries (typical NFCorpus workload) are unlikely to be affected.

---

## Rejected Alternatives

### Approach 1 — Query expansion (`"bartleby bartleby"`)

With MEAN aggregation, repeating the query doubles `n` while doubling each per-token contribution equally: `mean([a,b,c,d,a,b,c,d]) = mean([a,b,c,d])`. **This is a no-op.** Would only work with SUM aggregation, which Witchcraft does not use (ADR 006).

### Approach 2 — IDF-weighted MaxSim

Downweight query-token contributions whose corpus IDF falls below a threshold. Semantically correct but requires:
- A new `SwitchcraftStorage` protocol method (e.g. `tokenIDF(tokenID:)`).
- IDF tracking during indexing (per-token document frequency counts).
- A new SQLite table + schema migration.
- Implementation in both `SQLiteStorage` and `InMemoryStorage`.

This is estimated at 3–4× the cost of the surface-form filter for comparable precision improvement. Deferred to a future issue.

### Approach 4 — Symmetric query-side `minNorm` filter

Apply a higher `minNorm` to query-token embeddings (analogous to the existing document-side filter). The proposal was that short common subword tokens might have lower pre-normalization L2 norms than specific tokens.

Empirical verification is required to confirm the L2-norm correlation for subword fragments like `"t"` and `"le"`. This data is not available in the codebase and cannot be confirmed within this issue. If future investigation demonstrates a strong correlation, this approach could complement the surface-form filter.

---

## Witchcraft Parity Guarantee

At `queryMinSurfaceFormLength = 0` (the default), `encodeQuery` delegates to `encode` and no rows are suppressed. The query embedding and downstream MaxSim scoring are byte-identical to the pre-ADR 028 code path.
