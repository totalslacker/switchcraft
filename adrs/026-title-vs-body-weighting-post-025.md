# ADR 026 — Title-vs-body weighting for proper-noun queries post-#426

**Status**: Accepted
**Date**: 2026-06-02
**Issue**: SafariUnfucker#445 (characterize title-vs-body weighting tradeoff after v2 re-embed pass)
**Prior art**: ADR 025 (`025-title-prepend-for-query-alignment.md`)

---

## Context

ADR 025 introduced the title-prepend fix (H3) to address the alfred/bartleby ranking inversion
first reported in switchcraft#105. After the SafariUnfucker v2 re-embed pass
(commit `555b838b`, 6,932 → 19,383 pages, June 2026), alfredapp.com ranks #1 for the query
`alfred` with RRF score 0.032787 — confirming the fix works when the document body reinforces
the title token.

The same pass exposed the **inverse failure mode**: for the query `bartleby`, `bartleby.com` is
absent from the top results. NomicBert returns `bartleby.com` at rank #1 with score 0.999, but
Switchcraft's hybrid pipeline does not surface it. This ADR documents the diagnosis, the
per-source data, and the decision.

---

## Diagnostic data

Data collected from a debug build of SafariUnfucker against the production index (19,383 pages)
using a temporary `[BARTLEBY_DIAG]` log added to `SwitchcraftSearchAdapter.search()` with
`switchcraftMinCandidatePool` raised to 500. Collection method: HTTP search endpoint
`GET /search?q=bartleby&backend=switchcraft&limit=20`.

### alfred query (working case)

| Document | vectorRank | vectorScore | ftsRank | ftsScore | RRF fused score |
|---|---|---|---|---|---|
| alfredapp.com/powerpack/ | (present) | — | (present) | — | **0.032787** |
| alfredapp.com/ | (present) | — | (present) | — | 0.032258 |

Both vector and FTS pipelines contribute — the body of alfredapp.com discusses "Alfred"
throughout, so title-prepend + body content together produce strong MaxSim and BM25 signals.

### bartleby query (failing case)

| Document | vectorRank | vectorScore | ftsRank | ftsScore | RRF fused score |
|---|---|---|---|---|---|
| bartleby.com | **nil** | nil | **5** | 10.903894 | **~0.01538** |
| GitHub issues about bartleby | (present) | — | (present) | — | ~0.016 |

`nil` vectorRank means bartleby.com did not appear in the vector pipeline's top candidates,
even with topK=500 out of 19,383 total pages. The BM25/FTS pipeline correctly finds
bartleby.com at rank 5 (the URL itself contains "bartleby"), but the absence of a vector
component means the fused RRF score is only `1/(60+5) ≈ 0.01538` — below the scores of
documents that appear in both pipelines.

---

## Root-cause analysis

### Why title-prepend is sufficient for alfred but not for bartleby

The title-prepend fix (ADR 025, section a) places the title at position 0 of the T5 encoder
context, which aligns it with the minimal-context shape of a standalone query token. This works
**when the body also reinforces the title token**:

- `alfredapp.com` body: discusses "Alfred" throughout ("Alfred is a productivity application",
  "Alfred Workflows", etc.) → hundreds of "alfred"-adjacent tokens in the body → T5 produces
  embeddings where the query token "alfred" finds strong MaxSim matches in the chunk set.

- `bartleby.com` body: homepage is homework-help chrome ("Homework Help is Here", "Literature
  guides", "Sign up free", etc.) → **zero "bartleby"-adjacent tokens in the body** → the
  title "Bartleby.com" contributes ONE short token to the T5 embedding context, while the
  long homework-help body dominates.

In T5's sliding-window strategy (ADR 011), the title token appears only in window 0. Windows
1..N contain pure body content with no title repetition. For a long body, the overall chunk
embedding matrix is dominated by body semantics. The query token "bartleby" can't find strong
MaxSim matches against chunks whose dominant semantic is homework-help content.

### R2 determination: which pipeline is the actionable lever

This is the asymmetric case: **only one pipeline fails**.

- **Vector (XTR/T5 MaxSim): completely absent** — vectorRank nil even at topK=500. The
  homework-help body semantics render bartleby.com invisible to XTR regardless of topK.
- **FTS/BM25: working reasonably** — ftsRank 5, ftsScore 10.9. BM25 finds the URL.

The FTS pipeline is the actionable lever. The vector pipeline's failure is a fundamental
consequence of body-semantics dominance; fixing it requires either changing the embedding
strategy (out of scope per ADR 025, section (e) — H2 constants deferred) or separating
title indexing from body indexing at the store level.

### Why FTS-only is insufficient for top-10 surfacing

With RRF (ADR 008, K=60):
- A document in both pipelines at ranks 1 and 1 scores `1/61 + 1/61 ≈ 0.0328`.
- A document in FTS-only at rank 5 scores `1/65 ≈ 0.0154`.

In a 19,383-page corpus where many documents discussing "bartleby" appear in **both** pipelines
(GitHub issues, PRs, search-engine results pages), those dual-pipeline documents consistently
outrank bartleby.com in the fused result, pushing it out of the default top-10 and often
beyond top-50.

---

## Decision

### Current behavior is **not** the correct semantic tradeoff for this case

The current behavior can be defended as "correct semantic search" only if the desired outcome
is to prioritize documents that *discuss* a topic richly over documents that *are* the
authoritative source for that topic. But for single-token proper-noun queries issued from a
browser history tool, the user's intent is clearly the canonical domain — not secondary sources.

The root gap is documented in ADR 025, section (d):

> `DocumentRecord.body` continues to store the caller-supplied `body` (not the
> title-prefixed embedding text) … The FTS/BM25 pipeline continues to see the
> original body. Callers who want the title indexed for BM25 matching should
> include it in `body` or `metadata`. Option C (adding a `title` field to
> `DocumentRecord` and the SQLite schema) is deferred to a future cycle.

The bartleby.com case is the concrete production failure that demonstrates Option C is
necessary. Without a title-aware BM25 field, FTS can only boost a title-only document (one
whose body is semantically disjoint from its title) via accidental URL indexing — an
unreliable, non-generalizable mechanism.

### Follow-up actions

This ADR files two targeted follow-up issues rather than implementing a fix inline:

1. **Upstream (switchcraft)**: Implement Option C from ADR 025 — add a `title` field to
   `DocumentRecord` and the SQLite FTS schema so BM25 can match on the title with a
   configurable weight boost. The fix should be additive: callers that omit `title` get
   identical behaviour to today.

2. **Adapter layer (SafariUnfucker)**: Until the upstream fix lands, evaluate prepending
   the title into `body` at the `SwitchcraftSearchAdapter.add()` call site as a workaround
   so BM25 sees the title token. Risk: this changes the stored `body` content, which affects
   snippet generation and round-trip semantics. Defer if Option C lands quickly.

### No ranking hacks

No code shall force `bartleby.com` or any specific URL to a higher rank. The fix must be
general (title-field BM25 weight applies to all documents, not a URL allowlist).

---

## Consequences

- The alfred ranking regression (switchcraft#105) is confirmed fixed by ADR 025 + v2 re-embed.
- The bartleby.com ranking failure (SafariUnfucker#445) is confirmed a separate failure mode:
  body-disjoint pages. ADR 025 is necessary but not sufficient.
- The correct fix requires Option C upstream (title field in BM25 schema). Until then, the
  failure persists for any page whose title contains the search term but whose body does not.
- The interim adapter-layer workaround (title in body) is available but has side effects on
  snippet quality; it should be evaluated only if Option C is delayed.

---

## References

- ADR 006 (`006-search-constants.md`) — k=32, tPrime=40_000 (unchanged)
- ADR 008 (`008-hybrid-fusion.md`) — RRF fusion, K=60 (unchanged)
- ADR 011 (`011-sliding-window-long-input-strategy.md`) — title lands in window 0 only
- ADR 025 (`025-title-prepend-for-query-alignment.md`) — title-prepend fix; Option C deferred
- switchcraft#105 — original alfred/bartleby ranking issue (closed; alfred case fixed)
- SafariUnfucker#445 — this issue; bartleby.com diagnostic + tradeoff characterization
