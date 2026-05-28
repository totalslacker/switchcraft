# ADR 025 — Title prepend for proper-noun query alignment

**Status**: Accepted
**Date**: 2026-05-28
**Issue**: #105 (XTR ranks bartleby.com #2 vs alfredapp.com #37 for query 'alfred')

---

## Context

For single-token proper-noun queries the XTR MaxSim pipeline can produce
counter-intuitive rankings. The concrete reproducer: query `alfred` against a
~5,600-page web history index returns `bartleby.com` (the Melville-character
homework-help site) at vector rank #2 (score 0.7251) while `alfredapp.com`
lands at vector rank #37 (score 0.5591). BM25/FTS ranks them correctly
(alfredapp.com #4, bartleby.com #11), so this is not a tokenizer or indexing
pipeline issue — it is specific to the XTR vector stage.

Three root-cause hypotheses were evaluated:

### H1 — T5 embedding-space proximity between proper-noun token clusters

In T5's 128-dim projection space, tokens whose surface forms are similar (or
that co-occur in similar literary contexts) cluster close together regardless
of semantic meaning. The "alfred" query token and some bartleby.com body token
may simply be nearby neighbours in that space, causing a high MaxSim score
that no code-level fix can fully erase. **H1 requires model retraining and is
explicitly out of scope for this issue.**

### H2 — alfredapp.com's canonical "Alfred" tokens miss the top-k centroid scan

With k=32 centroids scanned per query token and a corpus of ~5,600 documents
(~280k total tokens ≈ 8,471 centroids), alfredapp.com's "Alfred" body tokens
may land in centroids ranked #33+ — outside the scan window. Those tokens
never appear in the candidate set; the document gets a score of at most
`missing[q]` (the 32nd-centroid similarity), which may be lower than
bartleby.com's score from a token that does fall in the top-32. **H2 fixes
(`k`, `tPrime`, or the `missing[q]` update rule) carry NFCorpus NDCG@10
regression risk and are deferred.** The relevant constants are locked by ADR
006 / Witchcraft parity.

### H3 — Title tokens absent or weak at embedding time

`SwitchcraftStore.add` before this ADR passed `body` verbatim to the embedder.
For a landing-page document, the title (e.g. "Alfred — Productivity App for
macOS") may not appear in the body at all, or may be short enough relative to
the body that its tokens receive high-context, richly-attended embeddings that
diverge from the same token embedded in a short standalone query. Prepending
the title to the body at embedding time places title tokens at position 0 of
the T5 encoder's context — the position closest to the minimal-context shape
of a standalone query token. **H3 is the code-level fix adopted by this ADR.**

---

## Decision

### (a) Title-prepend policy

When `SwitchcraftStore.add` receives a non-nil `title`, the text passed to
the embedder is:

```
embeddingText = "\(title)\n\(body)"
```

The `"\n"` separator produces a natural sentence boundary in T5's SentencePiece
tokenizer (treated as whitespace), keeping the title as a distinct leading
sequence without injecting a special token. A space separator was rejected
because it can fuse the last title word with the first body word at a
wordpiece boundary.

When `title == nil`, `embeddingText == body` and the entire code path is
byte-identical to the pre-025 behaviour. Existing callers, the 33-fact corpus
tests, and the NFCorpus benchmark are unaffected.

### (b) Hash computation

The chunk dedup key changes from `SHA-256(body)` to `SHA-256(embeddingText)`.

This is required for correctness: two documents with the same body but
different titles produce different token-embedding matrices. Sharing a chunk
record (and therefore the same centroid assignments) for documents with
different titles would silently mis-score the document whose title was not
used during chunk creation.

When `title == nil`, `embeddingText == body` and the hash is identical to the
pre-025 formula — backward-compatible for all existing chunks.

### (c) Store-version compatibility and re-indexing

Stores built before this change contain chunks keyed by `SHA-256(body)`. After
this change, re-adding an existing document with a `title` computes
`SHA-256(title + "\n" + body)` — a different hash — and inserts a new chunk.
The old chunk becomes an orphan (no document maps to it; the search engine
drops orphans). This is not a correctness bug but does increase storage bloat
for callers who add titles during a re-indexing run.

**Recommendation**: callers who want to add titles to an existing store should
call `store.clear()` before re-adding documents with titles. Incremental
re-add without clear leaves stale orphan chunks.

### (d) DocumentRecord.body stores caller-supplied body only

`DocumentRecord.body` continues to store the caller-supplied `body` (not the
title-prefixed embedding text). This preserves round-trip semantics: a caller
that fetches a document back via `storage.document(uuid:)` receives the
original `body`, not a prefixed variant. The FTS/BM25 pipeline continues to
see the original body.

Callers who want the title indexed for BM25 matching should include it in
`body` or `metadata`. This is documented on `SwitchcraftStore.add`.

Option C (adding a `title` field to `DocumentRecord` and the SQLite schema)
is cleaner long-term but requires a schema migration and multi-chunk work. It
is deferred to a future cycle.

### (e) Relationship to ADR 006 — H2 constants unchanged

`k=32`, `tPrime=40_000`, and the `missing[q]` update rule (ADR 006 sections
(b), (c), (e)) are unchanged. H2 is a separate follow-up issue. The NFCorpus
NDCG@10 gate (0.31–0.33) continues to apply.

### (f) Parity tests unaffected

- **33-fact corpus** (`FactsCorpusParityTests`): facts are indexed without a
  `title`; `embeddingText == body`; scores are byte-identical to pre-025.
- **NFCorpus benchmark** (`NFCorpusBenchmarkTests`): passages are indexed
  without a `title`; same invariant applies.

### (g) Rationale for `"\n"` separator

T5 SentencePiece treats `\n` as generic whitespace (mapped to `▁`) and does
not assign it a special-token ID. This keeps the title as a grammatically
distinct leading span without adding a dedicated separator like `[SEP]`, which
XTR-base-en was not trained with. A plain space was rejected because the
SentencePiece tokenizer at word boundaries can produce a wordpiece that
conflates the last title token with the first body token if the body begins
mid-sentence.

---

## Consequences

- Callers supplying a `title` get meaningfully improved vector ranking for
  short proper-noun queries targeting the document's title.
- No change for callers that omit `title` — the parameter defaults to `nil`.
- Two documents with the same body but different titles now produce distinct
  chunk records. Chunk dedup across titles is intentionally disabled.
- Re-indexing a store with titles requires `clear()` to avoid orphan bloat.
- H1 (T5 embedding-space proximity between unrelated proper nouns) remains
  unaddressed. The gap between the improved ranking and NomicBert-style
  single-vector cosine similarity may persist for cases where H1 dominates.
- H2 (`k`/`tPrime`/`missing[q]` adjustment) remains deferred.

---

## References

- ADR 006 (`006-search-constants.md`) — k=32, tPrime=40_000, missing[q] policy (unchanged)
- ADR 008 (`008-hybrid-fusion.md`) — RRF fusion (unchanged)
- ADR 009 (`009-public-api-shape.md`) — public API surface (amended below)
- ADR 011 (`011-sliding-window-long-input-strategy.md`) — title tokens land in window 0
- Issue #105 — reproducer, H1/H2/H3 diagnosis, regression test
